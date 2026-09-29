import io

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply
from rich.console import Console

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SessionLog, load_session
from wren.agent.todos import TodoItem, format_todos, parse_todos
from wren.checkpoint import Checkpoints
from wren.cli.ui import RichUI
from wren.config import ModelConfig
from wren.tools import ToolError
from wren.tools.todo import TodoWrite

PLAN = [{"content": "Read the code", "status": "completed"},
        {"content": "Fix the bug", "status": "in_progress"},
        {"content": "Run the tests", "status": "pending"}]


def test_parse_and_format():
    items = parse_todos(PLAN)
    assert items[1] == TodoItem("Fix the bug", "in_progress")
    assert format_todos(items) == "✓ Read the code\n▸ Fix the bug\n○ Run the tests"


@pytest.mark.parametrize("bad,msg", [
    ([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}], "only one"),
    ([{"content": " ", "status": "pending"}], "non-empty"),
    ([{"content": "a", "status": "doing"}], "use one of"),
    ("a list", "must be a list"),
])
def test_invalid_lists_are_explained(bad, msg, ctx):
    with pytest.raises(ToolError, match=msg):
        TodoWrite().run({"todos": bad}, ctx)


def test_tool_output(ctx):
    out = TodoWrite().run({"todos": PLAN}, ctx)
    assert out.content == "Task list updated (1/3 done). In progress: Fix the bug"
    assert out.summary == "1/3 done" and out.todos[0].status == "completed"
    assert "▸ Fix the bug" in out.display


def make_agent(ctx, tmp_path, turns):
    return Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f"), ctx,
                 RecordingUI(), Permissions(mode="auto"),
                 log=SessionLog(directory=tmp_path / "sessions"),
                 checkpoints=Checkpoints(ctx.cwd, root=tmp_path / "shadow"))


@pytest.fixture
def ctx(tmp_path):
    from wren.tools import ToolContext
    (tmp_path / "p").mkdir()
    return ToolContext(cwd=(tmp_path / "p").resolve())


def test_agent_keeps_list_across_replay_and_rewind(ctx, tmp_path):
    first = [{"content": "a", "status": "in_progress"}]
    done = [{"content": "a", "status": "completed"}]
    agent = make_agent(ctx, tmp_path, [
        call("todo_write", "t1", todos=first), reply("started"),
        reply("pausing: waiting for the user"),               # answers the unfinished-items reminder
        call("todo_write", "t2", todos=done), reply("finished"),
    ])
    agent.run("one")
    assert agent.conv.todos == [TodoItem("a", "in_progress")]
    agent.run("two")
    assert agent.conv.todos == [TodoItem("a", "completed")]

    assert load_session(agent.log.path).conversation.todos == agent.conv.todos
    agent.rewind(1)                                   # back to before "two"
    assert agent.conv.todos == [TodoItem("a", "in_progress")]
    assert load_session(agent.log.path).conversation.todos == agent.conv.todos


def test_compaction_note_carries_the_list(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [call("todo_write", "t1", todos=PLAN), reply("ok"),
                                       reply("paused"), reply("SUMMARY")])
    agent.run("plan it")
    agent.compact()
    note = agent.messages[0].text()
    assert "▸ Fix the bug" in note and "todo_write" in note


def test_todos_are_shown_under_the_tool_line():
    out = io.StringIO()
    ui = RichUI(Console(file=out, width=80, color_system=None), interactive=False)
    ui.tool_started("todo_write", "3 items")
    ui.tool_finished("todo_write", TodoWrite().run({"todos": PLAN}, None))
    assert "✓ Read the code" in out.getvalue() and "○ Run the tests" in out.getvalue()


def test_one_reminder_about_unfinished_items(ctx, tmp_path):
    todos = [{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]
    agent = make_agent(ctx, tmp_path, [
        call("todo_write", "t1", todos=todos), reply("all done!"),       # finishes too early
        reply("b is waiting for your answer"), reply("unused"),
    ])
    assert agent.run("do a and b") == "b is waiting for your answer"
    reminder = agent.provider.requests[2][-1].content[-1].text
    assert "unfinished" in reminder and "○ b" in reminder and "✓ a" not in reminder
    assert len(agent.provider.requests) == 3                             # reminded once only


def test_no_reminder_when_everything_is_done(ctx, tmp_path):
    todos = [{"content": "a", "status": "completed"}]
    agent = make_agent(ctx, tmp_path, [call("todo_write", "t1", todos=todos), reply("done")])
    agent.run("do a")
    assert len(agent.provider.requests) == 2
