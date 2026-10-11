"""A model request that fails in a way that may pass is retried instead of ending the run."""

import pytest
from conftest import RecordingUI, ScriptedProvider, reply
from fake_anthropic import FakeAnthropic, error_event_turn, stalled_turn, text_turn

from wren.agent import loop
from wren.agent.loop import Agent
from wren.config import ModelConfig
from wren.llm.anthropic_provider import AnthropicProvider
from wren.llm.types import LLMError, Message, TextBlock, retryable_status


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(loop, "MODEL_RETRY_DELAYS", (0, 0))


def provider(server, read_timeout: float = 0.5) -> AnthropicProvider:
    p = AnthropicProvider(ModelConfig(name="fake", model="m", base_url=server.url,
                                      api_key_env="K", prompt_cache=False))
    p.client = p.client.with_options(timeout=read_timeout, max_retries=0)
    return p


def drain(p: AnthropicProvider) -> None:
    list(p.stream(system="s", messages=[Message("user", [TextBlock("go")])], tools=[]))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("K", "k")


def test_a_stream_that_stalls_is_a_retryable_error():
    # What crashed a DeepSWE run: the read timed out mid-stream, past the SDK's retries.
    with FakeAnthropic([stalled_turn(2)]) as server, pytest.raises(LLMError, match="connection lost") as e:
        drain(provider(server))
    assert e.value.retryable


def test_an_error_event_mid_stream_is_retryable():
    with FakeAnthropic([error_event_turn("overloaded_error")]) as server, pytest.raises(LLMError) as e:
        drain(provider(server))
    assert e.value.retryable


def test_only_transient_statuses_are_retryable():
    assert all(retryable_status(s) for s in (200, 408, 429, 500, 529))
    assert not any(retryable_status(s) for s in (400, 401, 403, 404, 413))


def test_the_agent_carries_on_after_a_dropped_stream(ctx):
    with FakeAnthropic([stalled_turn(2), error_event_turn(), text_turn("done")]) as server:
        ui = RecordingUI()
        agent = Agent(provider(server), ModelConfig(name="fake", model="m"), ctx, ui)
        assert agent.run("go") == "done"
    assert len(server.requests) == 3
    assert agent.status != "error"
    assert sum(1 for e in ui.events if e[0] == "notice" and "retrying" in e[1]) == 2


class Failing(ScriptedProvider):
    def __init__(self, error: LLMError, failures: int):
        super().__init__([reply("done")])
        self.error, self.failures, self.calls = error, failures, 0

    def stream(self, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        yield from super().stream(**kw)


def test_a_permanent_error_is_not_retried(ctx, model):
    p = Failing(LLMError("HTTP 400: bad request"), failures=1)
    agent = Agent(p, model, ctx, RecordingUI())
    agent.run("go")
    assert p.calls == 1 and agent.status == "error"


def test_retries_give_up_eventually(ctx, model):
    p = Failing(LLMError("connection lost", retryable=True), failures=99)
    agent = Agent(p, model, ctx, RecordingUI())
    agent.run("go")
    assert p.calls == len(loop.MODEL_RETRY_DELAYS) + 1 and agent.status == "error"
