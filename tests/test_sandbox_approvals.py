"""With a sandbox, commands inside it run without asking; what reaches beyond it asks."""

import io
import uuid
from pathlib import Path

from conftest import RecordingUI, ScriptedProvider, call, reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from test_sandbox import make, needs_sandbox

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.agent.subagents import builtin_agent_types
from wren.cli import repl
from wren.cli.repl import Repl
from wren.cli.ui import RichUI
from wren.config import ModelConfig, PermissionRules, load_config
from wren.llm.types import ToolResultBlock
from wren.settings import Settings
from wren.tools import ToolContext
from wren.tools.files import EditFile, WriteFile
from wren.tools.shell import Bash, SandboxedBash

BASH, WRITE, EDIT = SandboxedBash(), WriteFile(), EditFile()


def results(provider) -> dict[str, ToolResultBlock]:
    return {b.tool_use_id: b for msgs in provider.requests for m in msgs for b in m.content
            if isinstance(b, ToolResultBlock)}


def test_commands_in_the_sandbox_need_no_approval(tmp_path):
    sb = make(tmp_path)
    for mode in ("ask", "accept_edits", "plan", "auto"):
        p = Permissions(mode=mode, sandbox=sb, cwd=sb.cwd)
        assert not p.needs_approval(BASH, {"command": "rm -rf build"}), mode
    # Without one, as before.
    assert Permissions(mode="ask").needs_approval(Bash(), {"command": "rm -rf build"})
    assert Permissions(mode="plan").needs_approval(Bash(), {"command": "ls"})


def test_rules_still_apply_inside(tmp_path):
    p = Permissions(mode="ask", sandbox=make(tmp_path),
                    rules=PermissionRules(ask=["bash:git push"], deny=["bash:rm"]))
    assert p.must_ask(BASH, {"command": "git push"})
    assert p.blocked(BASH, {"command": "rm x"})


def test_file_edits_beyond_the_sandbox_always_ask(tmp_path):
    sb = make(tmp_path)
    for mode in ("ask", "accept_edits", "auto"):
        p = Permissions(mode=mode, sandbox=sb, cwd=sb.cwd)
        assert not p.must_ask(WRITE, {"path": "src/a.py"}), mode
        assert p.must_ask(WRITE, {"path": str(Path.home() / "elsewhere.py")}), mode
        assert p.must_ask(EDIT, {"path": ".git/hooks/pre-commit"}), mode
        assert p.must_ask(EDIT, {"path": "../proj/.git/config"}), mode
        assert p.must_ask(WRITE, {"path": "~/.bashrc"}), mode
    # Temp directories are inside, as for commands.
    assert not Permissions(sandbox=sb, cwd=sb.cwd).must_ask(WRITE, {"path": "/tmp/scratch.txt"})
    assert not Permissions(cwd=sb.cwd).must_ask(WRITE, {"path": "~/.bashrc"})   # no sandbox


def test_an_edit_beyond_it_is_asked_about_and_labelled_even_in_auto(tmp_path, model, request):
    sb = make(tmp_path)
    target = Path.home() / f".wren-test-{uuid.uuid4().hex}"   # outside the project and temp
    request.addfinalizer(lambda: target.unlink(missing_ok=True))
    provider = ScriptedProvider([call("write_file", "t1", path=str(target), content="x\n"),
                                 call("write_file", "t2", path=str(target), content="y\n"),
                                 reply("done")])
    ui = RecordingUI()
    ui.decisions = [Decision(allow=True, remember=True), Decision(allow=False)]
    agent = Agent(provider, model, ToolContext(cwd=sb.cwd, sandbox=sb), ui,
                  Permissions(mode="auto", sandbox=sb, cwd=sb.cwd))
    agent.run("go")
    assert [e[1] for e in ui.events if e[0] == "confirm"] == ["write_file", "write_file"]  # not remembered
    assert ("tool", "write_file", f"{target} (outside the sandbox)") in ui.events
    assert target.read_text() == "x\n"


def test_the_bash_tool_and_system_prompt_follow_the_sandbox(tmp_path, model):
    sb = make(tmp_path)
    agent = Agent(ScriptedProvider([]), model, ToolContext(cwd=sb.cwd), RecordingUI(),
                  Permissions(cwd=sb.cwd))
    assert type(agent.tools["bash"]) is Bash and "# Sandbox" not in agent.system
    agent.set_sandbox(sb)
    assert type(agent.tools["bash"]) is SandboxedBash and "# Sandbox" in agent.system
    assert agent.permissions.sandbox is sb and agent.ctx.sandbox is sb
    agent.set_sandbox(None)
    assert type(agent.tools["bash"]) is Bash and "# Sandbox" not in agent.system


@needs_sandbox
def test_plan_mode_commands_run_read_only_without_asking(tmp_path, model):
    sb = make(tmp_path)
    probe = f"wren-{uuid.uuid4().hex}"
    provider = ScriptedProvider([
        call("bash", "t1", command=f"echo y > /tmp/{probe} && echo tmp-ok; rm -f /tmp/{probe}; echo x > made.txt"),
        reply("plan: ..."),
    ])
    ui = RecordingUI()
    agent = Agent(provider, model, ToolContext(cwd=sb.cwd, sandbox=sb), ui,
                  Permissions(mode="plan", sandbox=sb, cwd=sb.cwd))
    agent.run("investigate")
    out = results(provider)["t1"].content
    assert not (sb.cwd / "made.txt").exists() and "tmp-ok" in out
    assert "sandbox: false" in out and "may write only in" in out
    assert not [e for e in ui.events if e[0] == "confirm"]


@needs_sandbox
def test_explore_runs_any_command_read_only_and_never_outside(tmp_path, model):
    sb = make(tmp_path)
    (sb.cwd / "a.txt").write_text("hi\n")
    task = call("task", "t1", description="look", prompt="look around", agent="explore")
    provider = ScriptedProvider([
        task,
        call("bash", "c1", command="python3 -c 'print(6 * 7)'"),            # not on the allowlist
        call("bash", "c2", command="rm a.txt"),
        call("bash", "c3", command="rm a.txt", sandbox=False),
        reply("report"), reply("done"),
    ])
    ui = RecordingUI()
    agent = Agent(provider, model, ToolContext(cwd=sb.cwd, sandbox=sb), ui,
                  Permissions(mode="ask", sandbox=sb, cwd=sb.cwd), agent_types=builtin_agent_types())
    agent.run("go")
    res = results(provider)
    assert "42" in res["c1"].content
    assert res["c2"].is_error and (sb.cwd / "a.txt").exists()
    assert "sandbox: false" not in res["c2"].content            # it can't leave, so isn't told to
    assert res["c3"].is_error and "can't run commands outside the sandbox" in res["c3"].content
    assert not [e for e in ui.events if e[0] == "confirm"]


def test_sandbox_command(ctx, tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(repl, "CONFIG_DIR", home)
    out = io.StringIO()
    ui = RichUI(Console(file=out, width=200, color_system=None), interactive=True)
    agent = Agent(ScriptedProvider([]), ModelConfig(name="fake", model="f"), ctx, ui, Permissions(cwd=ctx.cwd))
    r = Repl(agent, ui, load_config(home / "none.toml"), Settings(suggestions=False))
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        r.command("/sandbox")
        r.command("/sandbox on")
        on = agent.ctx.sandbox
        r.command("/sandbox off")
        r.command("/sandbox sideways")
    text = out.getvalue()
    assert "sandbox off" in text and "usage: /sandbox [on|off]" in text
    if needs_sandbox.args[0]:   # no sandbox here: turning it on says why
        assert "no sandbox:" in text and on is None
    else:
        assert on is not None and "writable" in text and str(Path(ctx.cwd).resolve()) in text
    assert agent.ctx.sandbox is None and type(agent.tools["bash"]) is Bash
