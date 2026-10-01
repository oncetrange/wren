"""Responses cut off at max_tokens before acting: the run goes on instead of ending."""

from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.loop import MAX_CUT_OFFS, Agent
from wren.agent.permissions import Permissions
from wren.agent.plans import FINAL_CHECK
from wren.config import ModelConfig
from wren.llm.types import Message, Response, TextBlock, ThinkingBlock, Usage


def cut_off(*blocks):
    return Response(Message("assistant", list(blocks)), "max_tokens", Usage(10, 32000))


def make(ctx, turns):
    return Agent(ScriptedProvider(turns), ModelConfig(name="f", model="f"), ctx, RecordingUI(),
                 Permissions(mode="auto"))


def test_thinking_cut_off_goes_on(ctx):
    agent = make(ctx, [cut_off(ThinkingBlock("let me design this " * 1000)),
                       call("glob", pattern="*"), reply("done")])
    assert agent.run("build the feature") == "done"
    assert agent.status == "done"
    second = agent.provider.requests[1]
    # The cut-off thinking (maybe unsigned) isn't sent back; a placeholder keeps the turn.
    assert second[1].content == [TextBlock("[response cut off at the output limit]")]
    assert "cut off at the output token limit" in second[2].content[0].text
    assert ("notice", "response was cut off at the max_tokens limit; asking the model to go on") \
        in agent.ui.events


def test_partial_text_is_kept(ctx):
    agent = make(ctx, [cut_off(ThinkingBlock("x"), TextBlock("Summary: I changed")), reply("Summary: done.")])
    assert agent.run("go") == "Summary: done."
    assert agent.provider.requests[1][1].content == [TextBlock("Summary: I changed")]


def test_gives_up_after_repeated_cut_offs(ctx):
    turns = [cut_off(ThinkingBlock("x")) for _ in range(MAX_CUT_OFFS + 1)] + [reply("never")]
    agent = make(ctx, turns)
    agent.run("go")
    assert agent.status == "max_tokens"                  # not "done"
    assert len(agent.provider.requests) == MAX_CUT_OFFS + 1


def test_counter_resets_after_progress(ctx):
    turns = []
    for i in range(MAX_CUT_OFFS + 1):                    # more cut-offs in all, never in a row
        turns += [cut_off(ThinkingBlock("x")), call("glob", f"g{i}", pattern="*")]
    agent = make(ctx, turns + [reply("done")])
    assert agent.run("go") == "done" and agent.status == "done"


def test_empty_response_is_not_done(ctx):
    agent = make(ctx, [Response(Message("assistant", []), "end_turn", Usage(1, 0))])
    agent.run("go")
    assert agent.status == "empty_response"


def test_final_check_asks_for_the_projects_own_tests():
    text = " ".join(FINAL_CHECK.split())
    assert "run them again now, the existing ones too" in text and "or a copy elsewhere" in text
