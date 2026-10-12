from pathlib import Path

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions, load_project_rules
from wren.config import ConfigError, PermissionRules, load_config
from wren.tools.files import EditFile, ReadFile, WriteFile
from wren.tools.shell import Bash
from wren.tools.web import WebFetch

READ, EDIT, WRITE, BASH = ReadFile(), EditFile(), WriteFile(), Bash()
LS = {"command": "ls"}


@pytest.mark.parametrize("mode,read,edit,bash", [
    ("ask", False, True, True),
    ("accept_edits", False, False, True),
    ("plan", False, None, True),          # None: edit is blocked outright
    ("auto", False, False, False),
])
def test_mode_matrix(mode, read, edit, bash):
    p = Permissions(mode=mode)
    assert p.needs_approval(READ, {}) is read
    assert p.needs_approval(BASH, LS) is bash
    if edit is None:
        assert "Plan mode" in p.blocked(EDIT, {})
    else:
        assert p.blocked(EDIT, {}) is None and p.needs_approval(EDIT, {}) is edit


def test_plan_mode_ignores_the_allowlist():
    p = Permissions(mode="ask")
    p.remember(BASH, LS)
    assert not p.needs_approval(BASH, LS)
    p.mode = "plan"
    assert p.needs_approval(BASH, LS)


def test_cycle():
    p = Permissions()
    assert [p.cycle() for _ in range(3)] == ["accept_edits", "plan", "ask"]
    yolo = Permissions(mode="auto", allow_auto=True)
    assert [yolo.cycle() for _ in range(4)] == ["ask", "accept_edits", "plan", "auto"]


def test_agent_refuses_edits_in_plan_mode(ctx, model):
    (ctx.cwd / "a.py").write_text("x = 1\n")
    provider = ScriptedProvider([
        call("read_file", "t1", path="a.py"),
        call("edit_file", "t2", path="a.py", old_string="x = 1", new_string="x = 2"),
        reply("here is my plan"),
    ])
    ui = RecordingUI()
    agent = Agent(provider, model, ctx, ui, Permissions(mode="plan"))
    agent.run("plan a change")
    assert (ctx.cwd / "a.py").read_text() == "x = 1\n"
    result = provider.requests[2][-1].content[0]
    assert result.is_error and "Plan mode" in result.content
    assert not any(e[0] == "confirm" for e in ui.events)   # refused, not asked


# --- rules --------------------------------------------------------------------


def ruled(tmp_path=None, mode="ask", **rules) -> Permissions:
    return Permissions(mode=mode, rules=PermissionRules(**rules), cwd=tmp_path or Path("/proj"))


def sh(command: str) -> dict:
    return {"command": command}


def test_a_bash_allow_rule_covers_the_command_with_any_arguments():
    p = ruled(allow=["bash:git diff", "bash:pytest*"])
    assert not p.needs_approval(BASH, sh("git diff"))
    assert not p.needs_approval(BASH, sh("git  diff --stat HEAD"))
    assert not p.needs_approval(BASH, sh("pytest -x tests/"))
    assert p.needs_approval(BASH, sh("git diffx"))
    assert p.needs_approval(BASH, sh("git push"))


def test_an_allow_rule_never_covers_a_compound_command_it_doesnt_spell_out():
    p = ruled(allow=["bash:git diff", "bash:make build && make test"])
    assert p.needs_approval(BASH, sh("git diff; rm -rf ~"))
    assert p.needs_approval(BASH, sh("git diff > /etc/passwd"))
    assert not p.needs_approval(BASH, sh("make build && make test"))


def test_deny_and_ask_rules_look_inside_compound_commands():
    p = ruled(mode="auto", deny=["bash:rm"], ask=["bash:git push"])
    for command in ("rm -rf build", "ls && rm x", "(cd a; rm x)", "echo $(rm x)"):
        assert p.blocked(BASH, sh(command)), command
    assert p.blocked(BASH, sh("rmdir x")) is None
    assert p.must_ask(BASH, sh("git status && git push --force origin main"))
    assert not p.must_ask(BASH, sh("git pushd"))


def test_deny_beats_everything_and_ask_beats_allow():
    p = ruled(mode="auto", deny=["edit:.env"], ask=["edit"], allow=["edit"])
    assert "permission rule `edit:.env`" in p.blocked(WRITE, {"path": ".env"})
    assert p.must_ask(WRITE, {"path": "a.py"})
    assert not ruled(allow=["edit"]).must_ask(WRITE, {"path": "a.py"})


def test_path_rules_are_relative_to_the_working_directory(tmp_path):
    p = ruled(tmp_path, deny=["edit:*.lock", "edit:vendor/*", "read_file:*secret*"])
    assert p.blocked(WRITE, {"path": "sub/uv.lock"})
    assert p.blocked(EDIT, {"path": str(tmp_path / "vendor" / "x.py")})
    assert p.blocked(EDIT, {"path": "src/vendor.py"}) is None
    assert p.blocked(READ, {"path": "conf/secrets.toml"})
    assert p.blocked(WRITE, {"path": "src/a.py"}) is None


def test_rules_name_tools_with_wildcards_and_hosts():
    class McpTool(ReadFile):
        name = "mcp__github__create_issue"
    p = ruled(allow=["web_fetch:*.python.org"], deny=["mcp__github__*"])
    fetch = WebFetch.__new__(WebFetch)
    assert not p.needs_approval(fetch, {"url": "https://docs.python.org/3/"})
    assert p.needs_approval(fetch, {"url": "https://python.org.evil.com/"})
    assert p.blocked(McpTool(), {})


def test_always_in_this_project_is_saved_and_read_back(tmp_path):
    saved = tmp_path / "home" / "permissions.toml"
    p = Permissions(project_file=saved)
    p.remember(BASH, sh("pytest -q"), persist=True)
    p.remember(BASH, sh('grep "é" a | wc -l'), persist=True)
    p.remember(EDIT, {}, persist=False)
    assert load_project_rules(saved).allow == ["bash:pytest", 'bash:grep "é" a | wc -l']
    fresh = Permissions(rules=load_project_rules(saved))
    assert not fresh.needs_approval(BASH, sh("pytest tests/x.py"))
    assert not fresh.needs_approval(BASH, sh('grep "é" a | wc -l'))
    assert fresh.needs_approval(EDIT, {})


def test_config_rules_are_checked(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[permissions]\nallow = ["bash:pytest"]\ndeny = ["bash:git push"]\n')
    assert load_config(path).permissions == PermissionRules(allow=["bash:pytest"], deny=["bash:git push"])
    for bad in ('allow = "bash"', 'deny = ["git push"]', 'alow = []'):
        path.write_text(f"[permissions]\n{bad}\n")
        with pytest.raises(ConfigError):
            load_config(path)


def test_the_agent_asks_on_an_ask_rule_even_in_auto_mode_and_never_remembers(ctx, model):
    provider = ScriptedProvider([
        call("bash", "t1", command="git push"),
        call("bash", "t2", command="git push"),
        call("bash", "t3", command="rm -rf build"),
        reply("done"),
    ])
    ui = RecordingUI()
    ui.decisions = [Decision(allow=True, remember=True)] * 2
    p = Permissions(mode="auto", rules=PermissionRules(ask=["bash:git push"], deny=["bash:rm"]), cwd=ctx.cwd)
    Agent(provider, model, ctx, ui, p).run("go")
    assert [e[1] for e in ui.events if e[0] == "confirm"] == ["bash", "bash"]
    refused = provider.requests[3][-1].content[0]
    assert refused.is_error and "bash:rm" in refused.content
