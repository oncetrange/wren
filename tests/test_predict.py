import threading

import pytest

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.predict import Predictor, clean, predict_next_prompt
from wren.config import ModelConfig, Price
from wren.llm.types import Message, Response, TextBlock, ToolUseBlock, Usage

from conftest import RecordingUI, ScriptedProvider, call, reply


@pytest.mark.parametrize("raw,expected", [
    ('"run the tests"', "run the tests"),
    ("「运行测试」", "运行测试"),
    ("commit it\nand push", "commit it"),
    ("NONE", None), ("none.", None), ("", None), ("x" * 300, None),
])
def test_clean(raw, expected):
    assert clean(raw) == expected


def make(ctx, turns):
    return Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f", price=Price(1.0, 2.0)),
                 ctx, RecordingUI(), Permissions(mode="auto"))


def test_predicts_without_touching_history(ctx):
    agent = make(ctx, [reply("Fixed the bug."), reply("now run the tests")])
    agent.run("fix the bug")
    before = [m.to_dict() for m in agent.messages]
    result = predict_next_prompt(agent)
    assert result.text == "now run the tests"
    assert [m.to_dict() for m in agent.messages] == before
    request = agent.provider.requests[-1]
    assert "Predict the user's next message" in request[-1].content[0].text


def test_no_prediction_mid_turn_or_on_tool_calls(ctx):
    agent = make(ctx, [])
    assert predict_next_prompt(agent) is None                       # nothing said yet
    agent.conv.append(Message("user", [TextBlock("hi")]))
    agent.conv.append(Message("assistant", [ToolUseBlock("t", "glob", {"pattern": "*"})]))
    assert predict_next_prompt(agent) is None                       # unfinished turn
    agent = make(ctx, [reply("done"), call("glob", pattern="*")])
    agent.run("x")
    assert predict_next_prompt(agent).text is None                  # model tried a tool


def test_predictor_hands_over_on_the_main_thread(ctx):
    agent = make(ctx, [reply("Done."), reply("push it")])
    agent.run("commit")
    ready = threading.Event()
    predictor = Predictor(agent, on_ready=ready.set)
    output_before = agent.usage.output_tokens
    predictor.start()
    assert ready.wait(5)
    assert agent.usage.output_tokens == output_before   # not accounted by the worker thread
    assert predictor.take() == "push it"
    assert agent.usage.output_tokens == output_before + 5 and agent.cost > 0


def test_stale_predictions_are_dropped_but_paid_for(ctx):
    agent = make(ctx, [reply("Done."), reply("old guess")])
    agent.run("x")
    predictor = Predictor(agent)
    predictor.start()
    predictor.clear()                                   # the user already moved on
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(5)
    assert predictor.take() is None
    assert agent.usage.output_tokens == 10              # both requests counted
