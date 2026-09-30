"""Background commands: bash run_in_background, bash_output, kill_job, notices and cleanup."""

import os
import time

import pytest
from conftest import RecordingUI, ScriptedProvider, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.subagents import builtin_agent_types
from wren.config import ModelConfig
from wren.llm.types import Message, Response, ToolResultBlock, ToolUseBlock, Usage
from wren.tools.base import ToolError
from wren.tools.jobs import BashOutput, KillJob
from wren.tools.shell import Bash


def bg(ctx, command):
    return Bash().run({"command": command, "run_in_background": True}, ctx)


def test_start_read_and_kill(ctx):
    out = bg(ctx, "echo starting; sleep 30")
    assert out.summary == "started bg1" and "Output so far:\nstarting" in out.content
    job = ctx.jobs.get("bg1")
    assert job.running
    assert "(no new output)" in BashOutput().run({"id": "bg1"}, ctx).content
    stopped = KillJob().run({"id": "bg1"}, ctx)
    assert stopped.summary == "stopped bg1" and not job.running
    assert "exited with code" in BashOutput().run({"id": "bg1"}, ctx).content
    with pytest.raises(ToolError, match="no background job 'bg9'"):
        BashOutput().run({"id": "bg9"}, ctx)


def test_immediate_failure_is_reported_at_once(ctx):
    out = bg(ctx, "echo port in use; exit 3")
    assert out.is_error and "already exited with code 3" in out.content and "port in use" in out.content


def test_wait_until_output_or_exit(ctx):
    bg(ctx, "sleep 2; echo 'listening on :8000'; sleep 30")
    t = time.monotonic()
    out = BashOutput().run({"id": "bg1", "wait": 20, "until": "listening on"}, ctx)
    assert "listening on :8000" in out.content and "running" in out.content
    assert time.monotonic() - t < 10
    bg(ctx, "sleep 2; echo finished")
    out = BashOutput().run({"id": "bg2", "wait": 20}, ctx)
    assert "[bg2 exited with code 0" in out.content and "finished" in out.content
    ctx.jobs.kill_all()


def test_long_output_keeps_the_end(ctx):
    bg(ctx, "sleep 1.6; seq 1 20000")
    out = BashOutput().run({"id": "bg1", "wait": 20}, ctx)
    assert "earlier characters skipped" in out.content and out.content.rstrip().endswith("20000")


def test_kill_stops_the_whole_process_group(ctx, tmp_path):
    pid_file = tmp_path / "child.pid"
    bg(ctx, f"sleep 60 & echo $! > {pid_file}; wait")
    time.sleep(0.3)
    child = int(pid_file.read_text())
    KillJob().run({"id": "bg1"}, ctx)
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def use(call_id, name, **input):
    return Response(Message("assistant", [ToolUseBlock(call_id, name, input)]), "tool_use", Usage(10, 5))


def test_model_hears_when_a_job_ends_and_session_end_stops_jobs(ctx):
    turns = [use("a", "bash", command="sleep 2.5; echo done", run_in_background=True),
             use("b", "bash", command="sleep 2"),
             use("c", "bash", command="sleep 60", run_in_background=True), reply("ok")]
    agent = Agent(ScriptedProvider(turns), ModelConfig(name="f", model="f"), ctx, RecordingUI(),
                  Permissions(mode="auto"))
    agent.run("start it")
    results = {b.tool_use_id: b.content for m in agent.messages for b in m.content
               if isinstance(b, ToolResultBlock)}
    assert "Background jobs finished:\n- bg1 (`sleep 2.5; echo done`) exited with code 0" in results["b"]
    assert "Background jobs finished" not in results["c"]            # told once
    long_running = ctx.jobs.get("bg2")
    assert long_running.running
    agent.end_session()
    assert not long_running.running


def test_subagent_jobs_end_with_it(ctx):
    turns = [use("t", "task", description="d", prompt="serve", agent="general"),
             use("s", "bash", command="sleep 60", run_in_background=True), reply("started"), reply("done")]
    agent = Agent(ScriptedProvider(turns), ModelConfig(name="f", model="f"), ctx, RecordingUI(),
                  Permissions(mode="auto"), agent_types=builtin_agent_types())
    started = []
    real_start = type(ctx.jobs).start

    def spy(self, *args, **kw):
        job = real_start(self, *args, **kw)
        started.append(job)
        return job

    type(ctx.jobs).start = spy
    try:
        agent.run("go")
    finally:
        type(ctx.jobs).start = real_start
    assert len(started) == 1 and not started[0].running
    assert ctx.jobs.jobs == {}                                  # it was the subagent's, not ours
