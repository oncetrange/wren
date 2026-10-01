"""Noticing when the model goes in circles: repeated identical failures and failure streaks."""

import json

from conftest import RecordingUI, ScriptedProvider, reply
from fake_anthropic import FakeAnthropic, text_turn

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SessionLog
from wren.config import ModelConfig
from wren.llm.types import Message, Response, ToolResultBlock, ToolUseBlock, Usage


def use(call_id, name, **input):
    return Response(Message("assistant", [ToolUseBlock(call_id, name, input)]), "tool_use", Usage(10, 5))


def make(ctx, turns, **kw):
    return Agent(ScriptedProvider(turns), ModelConfig(name="f", model="f"), ctx, RecordingUI(),
                 Permissions(mode="auto"), **kw)


def results(agent):
    return [b for m in agent.messages for b in m.content if isinstance(b, ToolResultBlock)]


def test_same_failure_three_times(ctx, tmp_path):
    turns = [use(f"c{i}", "bash", command="exit 3") for i in range(3)] + [reply("ok")]
    agent = make(ctx, turns, log=SessionLog(directory=tmp_path / "s"))
    agent.run("go")
    r = results(agent)
    assert "has failed" not in r[1].content
    assert "This exact call has failed 3 times" in r[2].content
    entries = [json.loads(l) for l in agent.log.path.read_text().splitlines()]
    assert {"reason": "repeated_failure", "tool": "bash", "count": 3}.items() <= next(
        e for e in entries if e["kind"] == "reminder").items()


def test_rerunning_a_changing_test_is_not_a_loop(ctx):
    # The same command fails with different output each time: progress, not a loop.
    (ctx.cwd / "n").write_text("0")
    cmd = "n=$(cat n); echo $((n+1)) > n; echo failing $n; exit 1"
    turns = [use(f"c{i}", "bash", command=cmd) for i in range(3)] + [reply("ok")]
    agent = make(ctx, turns)
    agent.run("go")
    assert not any("has failed" in r.content for r in results(agent))


def test_failure_streak_and_reset(ctx):
    failing = [use(f"f{i}", "bash", command=f"exit {i + 1}") for i in range(5)]
    turns = [*failing, use("ok", "bash", command="true"),
             *[use(f"g{i}", "bash", command=f"exit {i + 10}") for i in range(4)], reply("done")]
    agent = make(ctx, turns)
    agent.run("go")
    r = {b.tool_use_id: b.content for b in results(agent)}
    assert "Your last 5 tool calls all failed" in r["f4"]
    assert not any("all failed" in r[f"g{i}"] for i in range(4))  # the success reset the streak


def test_state_resets_per_request(ctx):
    turns = [use("a1", "bash", command="exit 3"), use("a2", "bash", command="exit 3"), reply("x"),
             use("b1", "bash", command="exit 3"), reply("y")]
    agent = make(ctx, turns)
    agent.run("first")
    agent.run("second")
    assert not any("has failed" in r.content for r in results(agent))


def test_experiment_flags(tmp_path):
    import os
    import subprocess
    import sys
    home, project = tmp_path / "home", tmp_path / "p"
    home.mkdir()
    project.mkdir()
    with FakeAnthropic([text_turn("ok")]) as server:
        (home / "config.toml").write_text(f'[models.fake]\nmodel = "m"\nbase_url = "{server.url}"\n'
                                          'api_key_env = "K"\nprompt_cache = false\n')
        proc = subprocess.run([sys.executable, "-m", "wren.cli.main", "-p", "hi", "-m", "fake", "--no-subagents",
                               "--mask-at", "12000", "--compact-at", "50000", "--no-final-check"],
                              cwd=project, env={**os.environ, "WREN_HOME": str(home), "K": "k"},
                              capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    tools = {t["name"] for t in server.requests[0]["body"]["tools"]}
    assert "task" not in tools and "read_file" in tools
    start = json.loads(next((home / "sessions").glob("*.jsonl")).read_text().splitlines()[0])
    assert (start["mask_at"], start["compact_at"]) == (12000, 50000)
