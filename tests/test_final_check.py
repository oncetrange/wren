"""End-of-request safeguards: the final instruction check and the turn-budget warning."""

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.plans import is_reminder
from wren.config import ModelConfig
from wren.llm.types import TextBlock

from conftest import RecordingUI, ScriptedProvider, call, reply


def make(ctx, turns, final_check=True, max_turns=100):
    return Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f"), ctx, RecordingUI(),
                 Permissions(mode="auto"), final_check=final_check, max_turns=max_turns)


def last_text(request):
    block = request[-1].content[-1]
    return block.text if isinstance(block, TextBlock) else ""


def test_final_check_after_changes(ctx):
    agent = make(ctx, [call("write_file", path="a.txt", content="x"), reply("done"),
                       reply("checked: committed as asked")])
    assert agent.run("write a.txt and commit it") == "checked: committed as asked"
    text = last_text(agent.provider.requests[2])
    assert is_reminder(text) and "re-read the user's request" in text
    assert len(agent.provider.requests) == 3


def test_no_final_check_without_changes_or_when_off(ctx):
    agent = make(ctx, [call("read_file", path="missing.txt"), reply("nothing to do")])
    agent.run("look")
    assert len(agent.provider.requests) == 2
    agent = make(ctx, [call("write_file", path="a.txt", content="x"), reply("done")], final_check=False)
    agent.run("write")
    assert len(agent.provider.requests) == 2


def test_todo_and_final_check_share_one_message(ctx):
    todos = [{"content": "write a.txt", "status": "completed"}, {"content": "commit", "status": "pending"}]
    agent = make(ctx, [call("todo_write", "t1", todos=todos),
                       call("write_file", "t2", path="a.txt", content="x"),
                       reply("done"), reply("committed now"), reply("unused")])
    assert agent.run("write and commit") == "committed now"
    text = last_text(agent.provider.requests[3])
    assert "○ commit" in text and "re-read the user's request" in text
    assert len(agent.provider.requests) == 4


def test_turn_budget_warning(ctx):
    # max_turns=10: warn once when 3 calls are left
    turns = [call("glob", f"t{i}", pattern="*") for i in range(10)]
    agent = make(ctx, turns, max_turns=10)
    agent.run("keep going")
    warnings = [i for i, r in enumerate(agent.provider.requests) if "model calls left" in last_text(r)]
    assert warnings == [7] and "3 model calls left" in last_text(agent.provider.requests[7])
    assert agent.status == "max_turns"
