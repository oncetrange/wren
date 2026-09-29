"""Read-only subagents requested in the same turn run at the same time."""

import threading

from conftest import RecordingUI, reply

from wren.agent import loop
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.subagents import builtin_agent_types
from wren.config import ModelConfig
from wren.llm.base import Provider
from wren.llm.types import Completed, Message, Response, TextDelta, ToolResultBlock, ToolUseBlock, Usage


class Router(Provider):
    """Parent turns in order; each subagent's turns keyed by its prompt."""

    def __init__(self, parent, children, on_child_start=None):
        self.parent, self.children = list(parent), {k: list(v) for k, v in children.items()}
        self.on_child_start = on_child_start
        self.lock = threading.Lock()

    def stream(self, *, system, messages, tools):
        first = messages[0].content[0].text
        if first in self.children:
            if len(messages) == 1 and self.on_child_start:
                self.on_child_start(first)
            with self.lock:
                response = self.children[first].pop(0)
        else:
            response = self.parent.pop(0)
        if text := response.message.text():
            yield TextDelta(text)
        yield Completed(response)


def tasks(*specs):
    blocks = [ToolUseBlock(f"t{i}", "task", {"description": f"job {i}", "prompt": prompt, "agent": kind})
              for i, (kind, prompt) in enumerate(specs, 1)]
    return Response(Message("assistant", blocks), "tool_use", Usage(10, 5))


def make(ctx, provider):
    return Agent(provider, ModelConfig(name="fake", model="f"), ctx, RecordingUI(),
                 Permissions(mode="ask"), agent_types=builtin_agent_types())


def results(agent):
    return [b for m in agent.messages for b in m.content if isinstance(b, ToolResultBlock)]


def test_explore_tasks_run_concurrently_and_keep_their_order(ctx):
    barrier = threading.Barrier(2, timeout=5)  # breaks unless both are in flight together
    provider = Router([tasks(("explore", "a?"), ("explore", "b?")), reply("both done")],
                      {"a?": [reply("A")], "b?": [reply("B")]},
                      on_child_start=lambda _: barrier.wait())
    agent = make(ctx, provider)
    assert agent.run("look at a and b") == "both done"
    assert [(r.tool_use_id, r.content, r.is_error) for r in results(agent)] == [
        ("t1", "A", False), ("t2", "B", False)]
    # Each call is shown with its result once all are done, in order.
    shown = [e[:2] for e in agent.ui.events if e[0] in ("tool", "result")]
    assert shown == [("tool", "task"), ("result", "task"), ("tool", "task"), ("result", "task")]
    assert len(agent.subagent_runs) == 2 and agent.usage.input_tokens == 40


def test_writing_subagents_run_one_at_a_time(ctx):
    order = []
    provider = Router([tasks(("explore", "a?"), ("general", "b!")), reply("done")],
                      {"a?": [reply("A")], "b!": [reply("B")]}, on_child_start=order.append)
    agent = make(ctx, provider)
    agent.run("go")
    assert order == ["a?", "b!"]
    assert [r.content for r in results(agent)] == ["A", "B"]


def test_interrupt_stops_every_running_subagent(ctx, monkeypatch):
    started = threading.Barrier(3, timeout=5)  # two workers + the main thread

    class Blocking(Router):
        def stream(self, *, system, messages, tools):
            first = messages[0].content[0].text
            if first in self.children:
                started.wait()
                agent.cancel_seen.wait(5)
                yield TextDelta("partial")  # the worker notices the cancel here
                yield Completed(reply("never"))
                return
            yield from super().stream(system=system, messages=messages, tools=tools)

    real_wait = loop.wait

    def interrupted_wait(futures):
        started.wait()
        agent._progress.cancel.set()
        agent.cancel_seen.set()
        real_wait(futures)
        raise KeyboardInterrupt

    monkeypatch.setattr(loop, "wait", interrupted_wait)
    agent = make(ctx, Blocking([tasks(("explore", "a?"), ("explore", "b?"))], {"a?": [], "b?": []}))
    agent.cancel_seen = threading.Event()
    agent.run("go")
    assert agent.status == "interrupted"
    assert [r.content for r in results(agent)] == ["Interrupted by the user."] * 2
    assert agent.messages[-1].role == "user"  # the history stays valid


def test_cancelled_agent_stops(ctx):
    agent = Agent(Router([reply("hi")], {}), ModelConfig(name="fake", model="f"), ctx, RecordingUI())
    agent.cancel = threading.Event()
    agent.cancel.set()
    agent.run("go")
    assert agent.status == "interrupted"
