"""Compaction, rewind and session resume, end to end through Agent."""

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SessionLog, list_sessions, load_session
from wren.checkpoint import Checkpoints
from wren.config import ModelConfig
from wren.llm.types import Message, Response, TextBlock, ToolResultBlock, Usage
from wren.tools import ToolContext


@pytest.fixture
def ctx(tmp_path):
    # The project lives in its own directory so that the session logs and
    # shadow repo under tmp_path are outside the snapshotted work tree.
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


def big_reply(text="x"):
    return Response(Message("assistant", [TextBlock(text)]), "end_turn", Usage(input_tokens=9000))


# Compacts past 8k tokens: above the system prompt and tool definitions (~2k).
SMALL = dict(name="fake", model="fake-1", context_window=10_000, mask_at=0)


def texts(agent):
    return [m.text() for m in agent.messages]


def test_auto_compaction_keeps_new_prompt_verbatim(ctx, tmp_path):
    provider = ScriptedProvider([big_reply(), reply("SUMMARY: user wants y"), reply("done")])
    model = ModelConfig(**SMALL)
    agent = make_agent(ctx, provider, tmp_path, model)

    agent.run("first")          # leaves the context at ~9k tokens
    assert agent.run("second") == "done"

    assert "about to be compacted" in provider.requests[1][-1].content[-1].text
    final_request = provider.requests[2]
    assert len(final_request) == 1
    assert "SUMMARY: user wants y" in final_request[0].content[0].text
    assert final_request[0].content[-1].text == "second"


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
    assert agent.changed_files(agent.conv.timeline[-1]) == ["M a.py", "A made.txt"]

    point = agent.rewind(len(agent.conv.timeline) - 1)
    assert point.label == "change things"
    assert f.read_text() == "x = 1\n"
    assert not (ctx.cwd / "made.txt").exists()   # bash side effects are undone too
    assert texts(agent) == ["say hi", "hello"]
    assert len(agent.conv.timeline) == 1
    assert agent.ctx.read_files == {}


def test_rewind_past_a_compaction_restores_full_history(ctx, tmp_path):
    provider = ScriptedProvider([big_reply("one"), reply("SUMMARY"), reply("two"), reply("three")])
    model = ModelConfig(**SMALL)
    agent = make_agent(ctx, provider, tmp_path, model)
    agent.run("a")
    agent.run("b")               # compacts, then answers "two"
    agent.run("c")
    kinds = [p.kind for p in agent.conv.timeline]
    assert kinds == ["turn", "compaction", "turn", "turn"]

    # "b" started right after the compaction, so going back to it gives the
    # compacted conversation...
    agent.rewind(2)
    assert len(agent.messages) == 1 and "SUMMARY" in agent.messages[0].text()
    # ...while undoing the compaction brings back the full history.
    agent.rewind(1)
    assert texts(agent) == ["a", "one"]


def test_undo_compaction_keeps_what_came_after(ctx, tmp_path):
    provider = ScriptedProvider([big_reply("one"), reply("SUMMARY"), reply("two")])
    model = ModelConfig(**SMALL)
    agent = make_agent(ctx, provider, tmp_path, model)
    agent.run("a")
    agent.run("b")
    index = next(i for i, p in enumerate(agent.conv.timeline) if p.kind == "compaction")

    agent.rewind(index)
    assert texts(agent) == ["a", "one", "b", "two"]
    assert [p.kind for p in agent.conv.timeline] == ["turn", "turn"]


def test_resume_rebuilds_the_same_state(ctx, tmp_path):
    provider = ScriptedProvider([
        big_reply("one"), call("glob", "t1", pattern="*"), reply("two"), reply("three"),
    ])
    agent = make_agent(ctx, provider, tmp_path)
    agent.run("a")
    agent.run("b")
    agent.run("c")
    agent.rewind(len(agent.conv.timeline) - 1)   # drops "c"

    state = load_session(agent.log.path)
    assert [m.to_dict() for m in state.messages] == [m.to_dict() for m in agent.messages]
    assert [(p.kind, p.label, p.commit) for p in state.conversation.timeline] == \
        [(p.kind, p.label, p.commit) for p in agent.conv.timeline]
    assert state.usage == agent.usage and state.first_prompt == "a"
    assert list_sessions(ctx.cwd, tmp_path / "sessions")[0].path == agent.log.path

    resumed = make_agent(ctx, ScriptedProvider([reply("back")]), tmp_path, log=False)
    resumed.restore(state)
    assert resumed.run("d") == "back"
    assert [m.role for m in resumed.messages] == ["user", "assistant"] * 4  # a, b+tool, d
    resumed.rewind(0)            # rewind points survive the resume
    assert resumed.messages == []


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


def test_new_session_starts_a_separate_log(ctx, tmp_path):
    agent = make_agent(ctx, ScriptedProvider([reply("one"), reply("two")]), tmp_path)
    agent.run("a")
    first_log = agent.log.path
    agent.new_session(SessionLog(directory=tmp_path / "sessions"))
    agent.run("b")
    assert texts(agent) == ["b", "two"]
    assert [s.first_prompt for s in list_sessions(ctx.cwd, tmp_path / "sessions")] == ["b", "a"]
    assert load_session(first_log).messages[0].text() == "a"
