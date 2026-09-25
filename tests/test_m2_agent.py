"""Compaction, checkpoints/rewind and session resume, end to end through Agent."""

from pathlib import Path

import pytest

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SessionLog, list_sessions, load_session
from wren.checkpoint import Checkpoints
from wren.config import ModelConfig
from wren.llm.types import Message, Response, TextBlock, ToolResultBlock, Usage

from conftest import RecordingUI, ScriptedProvider, call, reply


@pytest.fixture
def ctx(tmp_path):
    # The project lives in its own directory so that the session logs and
    # shadow repo under tmp_path are outside the snapshotted work tree.
    from wren.tools import ToolContext
    project = tmp_path / "project"
    project.mkdir()
    return ToolContext(cwd=project.resolve())


def make_agent(ctx, provider, tmp_path, model=None, log=True):
    return Agent(
        provider, model or ModelConfig(name="fake", model="fake-1"), ctx, RecordingUI(),
        Permissions(mode="auto"),
        log=SessionLog(directory=tmp_path / "sessions") if log else None,
        checkpoints=Checkpoints(ctx.cwd, root=tmp_path / "shadow"),
    )


def test_auto_compaction_replaces_history_and_continues(ctx, tmp_path):
    big = Response(Message("assistant", [TextBlock("x")]), "end_turn", Usage(input_tokens=900))
    provider = ScriptedProvider([big, reply("SUMMARY: user wants y"), reply("done")])
    model = ModelConfig(name="fake", model="fake-1", context_window=1000)
    agent = make_agent(ctx, provider, tmp_path, model)

    agent.run("first")          # leaves the context at ~900 tokens
    assert agent.run("second") == "done"

    summary_request = provider.requests[1]
    assert "about to be compacted" in summary_request[-1].content[-1].text
    # The model then continued from the summary plus the new prompt only.
    final_request = provider.requests[2]
    assert len(final_request) == 1
    assert "SUMMARY: user wants y" in final_request[0].content[0].text
    assert final_request[0].content[-1].text == "second"
    assert any("compacted" in e[1] for e in agent.ui.events if e[0] == "notice")


def test_undo_restores_files_and_conversation(ctx, tmp_path):
    f = ctx.cwd / "a.py"
    f.write_text("x = 1\n")
    provider = ScriptedProvider([
        reply("hello"),
        call("read_file", "t1", path="a.py"),
        call("edit_file", "t2", path="a.py", old_string="x = 1", new_string="x = 2"),
        call("bash", "t3", command="echo new > made.txt"),
        reply("changed"),
    ])
    agent = make_agent(ctx, provider, tmp_path)
    agent.run("say hi")
    agent.run("change things")
    assert f.read_text() == "x = 2\n" and (ctx.cwd / "made.txt").exists()

    agent.rewind(agent.checkpoints.history[-1])
    assert f.read_text() == "x = 1\n"
    assert not (ctx.cwd / "made.txt").exists()   # bash side effects are undone too
    assert [m.text() for m in agent.messages] == ["say hi", "hello"]
    assert len(agent.checkpoints.history) == 1
    assert agent.ctx.read_files == {}


def test_resume_rebuilds_the_same_conversation(ctx, tmp_path):
    provider = ScriptedProvider([
        reply("one"), call("glob", "t1", pattern="*"), reply("two"), reply("three"),
    ])
    agent = make_agent(ctx, provider, tmp_path)
    agent.run("a")
    agent.run("b")
    agent.run("c")
    agent.rewind(agent.checkpoints.history[-1])   # drops "c"

    state = load_session(agent.log.path)
    assert [m.to_dict() for m in state.messages] == [m.to_dict() for m in agent.messages]
    assert [c.commit for c in state.checkpoints] == [c.commit for c in agent.checkpoints.history]
    assert state.usage == agent.usage and state.first_prompt == "a"
    assert list_sessions(ctx.cwd, tmp_path / "sessions")[0].path == agent.log.path

    resumed = make_agent(ctx, ScriptedProvider([reply("back")]), tmp_path, log=False)
    resumed.restore(state)
    assert resumed.run("d") == "back"
    assert [m.role for m in resumed.messages] == ["user", "assistant"] * 4  # a, b+tool, d


def test_resume_answers_dangling_tool_calls(ctx, tmp_path):
    log = SessionLog(directory=tmp_path / "sessions")
    log.record("session_start", model="fake", cwd=str(ctx.cwd))
    log.record("message", **Message("user", [TextBlock("go")]).to_dict())
    log.record("message", **call("bash", "t9", command="sleep 100").message.to_dict())

    agent = make_agent(ctx, ScriptedProvider([]), tmp_path, log=False)
    agent.restore(load_session(log.path))
    last = agent.messages[-1]
    assert last.role == "user" and isinstance(last.content[0], ToolResultBlock)
    assert last.content[0].tool_use_id == "t9"
