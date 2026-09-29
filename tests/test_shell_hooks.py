"""User shell hooks, run for real through bash."""

import json

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent import shell_hooks
from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.agent.session import SessionLog
from wren.config import ConfigError, HookSpec, parse_hooks


def make(ctx, model, turns, hooks, mode="ask", decisions=()):
    agent = Agent(ScriptedProvider(turns), model, ctx, RecordingUI(list(decisions)),
                  Permissions(mode=mode), log=SessionLog(directory=None))
    shell_hooks.install(agent, [HookSpec(**h) for h in hooks])
    return agent


def result_of(agent, request=1):
    return agent.provider.requests[request][-1].content[0]


def test_config_parsing():
    specs = parse_hooks({"PostToolUse": [{"matcher": "edit_file|write_file", "command": "fmt"}],
                         "Stop": {"command": "pytest", "timeout": 5}}, source="user")
    assert [(s.event, s.matcher, s.timeout) for s in specs] == [
        ("PostToolUse", "edit_file|write_file", 60), ("Stop", None, 5)]
    with pytest.raises(ConfigError, match="unknown event"):
        parse_hooks({"BeforeTool": [{"command": "x"}]}, "user")
    with pytest.raises(ConfigError, match="needs a 'command'"):
        parse_hooks({"Stop": [{"cmd": "x"}]}, "user")
    with pytest.raises(ConfigError, match="bad matcher"):
        parse_hooks({"PreToolUse": [{"command": "x", "matcher": "("}]}, "user")


def test_pre_tool_exit_2_blocks_with_stderr(ctx, model, tmp_path):
    payload = tmp_path / "payload.json"
    agent = make(ctx, model, [call("bash", command="rm -rf build"), reply("ok")], mode="auto", hooks=[
        {"event": "PreToolUse", "matcher": "bash",
         "command": f"cat > {payload}; echo 'no rm here' >&2; exit 2"}])
    agent.run("clean")
    r = result_of(agent)
    assert r.is_error and r.content == "no rm here"
    data = json.loads(payload.read_text())
    assert data["tool_name"] == "bash" and data["tool_input"] == {"command": "rm -rf build"}
    assert data["event"] == "PreToolUse" and data["permission_mode"] == "auto"
    assert any(e[0] == "hook" and e[2].startswith("deny") for e in agent.ui.events)


def test_matcher_limits_which_tools_run_the_hook(ctx, model):
    agent = make(ctx, model, [call("glob", pattern="*"), reply("ok")], hooks=[
        {"event": "PreToolUse", "matcher": "bash", "command": "exit 2"}])
    agent.run("list")
    assert not result_of(agent).is_error


def test_json_allow_skips_the_prompt(ctx, model):
    agent = make(ctx, model, [call("bash", command="echo hi"), reply("ok")], hooks=[
        {"event": "PreToolUse", "command": 'echo \'{"decision": "allow"}\''}])
    agent.run("hi")
    assert not any(e[0] == "confirm" for e in agent.ui.events)


def test_post_tool_env_and_feedback(ctx, model, tmp_path):
    seen = tmp_path / "seen"
    agent = make(ctx, model, [call("write_file", path="a.py", content="x=1\n"), reply("ok")],
                 mode="auto", hooks=[{"event": "PostToolUse", "matcher": "write_file|edit_file",
                                      "command": f'echo "$WREN_FILE $WREN_TOOL_NAME" > {seen}; '
                                                 'echo "E225 missing whitespace" >&2; exit 2'}])
    agent.run("write")
    assert seen.read_text().split() == [str(ctx.cwd / "a.py"), "write_file"]
    assert "E225 missing whitespace" in result_of(agent).content
    assert (ctx.cwd / "a.py").read_text() == "x=1\n"          # the tool still ran


def test_prompt_submit_context_and_block(ctx, model):
    agent = make(ctx, model, [reply("ok")], hooks=[
        {"event": "UserPromptSubmit", "command": (
            "python3 -c \"import json, sys\n"
            "if 'password' in json.load(sys.stdin)['prompt']:\n"
            "    print('contains a password', file=sys.stderr)\n"
            "    sys.exit(2)\n"
            "print('branch: main')\"")}])
    agent.run("hello")
    assert "branch: main" in agent.messages[0].content[1].text
    agent.run("my password is x")
    assert agent.status == "blocked"


def test_stop_hook_keeps_the_model_going(ctx, model, tmp_path):
    marker = tmp_path / "active"
    hook = (f"python3 -c \"import json,sys; d=json.load(sys.stdin); "
            f"open('{marker}','a').write(str(d['stop_hook_active'])+' ')\"; "
            f"[ -f {ctx.cwd}/done ] || {{ echo 'tests fail: create done' >&2; exit 2; }}")
    agent = make(ctx, model, [reply("finished?"), call("bash", command="touch done"), reply("now done")],
                 mode="auto", hooks=[{"event": "Stop", "command": hook}])
    assert agent.run("work") == "now done"
    assert "tests fail: create done" in agent.provider.requests[1][-1].content[-1].text
    assert marker.read_text().split() == ["False", "True"]


def test_session_start_context_goes_into_the_first_prompt(ctx, model):
    agent = make(ctx, model, [reply("ok")], hooks=[{"event": "SessionStart", "command": "echo 'on branch main'"}])
    agent.start_session("startup")
    agent.run("hi")
    assert "on branch main" in agent.messages[0].content[1].text


def test_notification_before_asking(ctx, model, tmp_path):
    note = tmp_path / "note"
    agent = make(ctx, model, [call("bash", command="ls"), reply("ok")], decisions=[Decision(True)],
                 hooks=[{"event": "Notification",
                         "command": f"python3 -c \"import json,sys; open('{note}','w').write(json.load(sys.stdin)['message'])\""}])
    agent.run("ls")
    assert note.read_text() == "wren needs your approval: bash ls"


def test_errors_and_timeouts_are_not_blocking(ctx, model):
    agent = make(ctx, model, [call("glob", pattern="*"), reply("ok")], hooks=[
        {"event": "PreToolUse", "command": "echo broken >&2; exit 1"},
        {"event": "PostToolUse", "command": "sleep 5", "timeout": 1}])
    agent.run("list")
    assert not result_of(agent).is_error
    statuses = [e[2] for e in agent.ui.events if e[0] == "hook"]
    assert statuses == ["exit 1: broken", "timed out after 1s"]
