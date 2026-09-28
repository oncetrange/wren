"""Predict the user's next prompt after the model finishes answering.

The prediction is one extra model request: the conversation so far (the same
prefix as the last request, so it is mostly a prompt-cache read) plus an
instruction to guess the next message. It is never added to the history.

`Predictor` runs it on a background thread while the user reads the answer.
The thread only computes: showing the result and accounting for its usage
happen on the main thread (`take`), so nothing races with the agent's own run.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wren.agent.plans import reminder
from wren.llm.types import Completed, Message, TextBlock, ToolUseBlock, Usage

if TYPE_CHECKING:
    from wren.agent.loop import Agent

PREDICT_REQUEST = """Predict the user's next message in this conversation: what they \
would most likely type next, given what just happened. Reply with only that message, \
written as the user would write it (same language, same tone), at most 15 words. If \
there is no obvious next step, reply with exactly NONE. Do not call any tools."""
MAX_CHARS = 200


@dataclass
class Prediction:
    text: str | None
    usage: Usage


def predict_next_prompt(agent: Agent) -> Prediction | None:
    """One prediction request. None if the conversation isn't at a finished answer."""
    messages = agent.messages
    if not messages or messages[-1].role != "assistant" or messages[-1].tool_uses():
        return None
    request = [Message(m.role, list(m.content)) for m in messages]
    request.append(Message("user", [TextBlock(reminder(PREDICT_REQUEST))]))
    response = None
    for event in agent.provider.stream(system=agent.system, messages=request,
                                       tools=[t.spec() for t in agent.tools.values()]):
        if isinstance(event, Completed):
            response = event.response
    if response is None:
        return None
    if any(isinstance(b, ToolUseBlock) for b in response.message.content):
        return Prediction(None, response.usage)
    return Prediction(clean(response.message.text()), response.usage)


def clean(text: str) -> str | None:
    """The first line, unquoted; None for NONE or nothing."""
    line = next((l.strip() for l in text.strip().splitlines() if l.strip()), "")
    line = line.strip("\"'`“”‘’「」").strip()
    if not line or line.upper().rstrip(".") == "NONE" or len(line) > MAX_CHARS:
        return None
    return line


class Predictor:
    """Background predictions, one at a time, handed over on the main thread."""

    def __init__(self, agent: Agent, on_ready: Callable[[], None] = lambda: None):
        self.agent = agent
        self.on_ready = on_ready  # called from the worker thread when a result arrives
        self.text: str | None = None  # the current prediction, once taken
        self._lock = threading.Lock()
        self._done: list[Prediction] = []  # finished, usage not yet accounted
        self._fresh: str | None = None     # newest current-generation prediction
        self._generation = 0

    def start(self) -> None:
        """Predict in the background; a newer start supersedes an older one."""
        self.clear()
        generation = self._generation
        threading.Thread(target=self._work, args=(generation,), daemon=True).start()

    def _work(self, generation: int) -> None:
        try:
            result = predict_next_prompt(self.agent)
        except Exception:  # a failed guess must never disturb the session
            return
        if result is None:
            return
        with self._lock:
            self._done.append(result)
            fresh = generation == self._generation
            if fresh:
                self._fresh = result.text
        if fresh:
            self.on_ready()

    def take(self) -> str | None:
        """On the main thread: account for finished requests; return the latest prediction."""
        with self._lock:
            done, self._done = self._done, []
            fresh, self._fresh = self._fresh, None
        for result in done:
            self.agent.record_side_usage(result.usage, purpose="prediction")
        if fresh is not None:
            self.text = fresh
        return self.text

    def clear(self) -> None:
        """Forget the current prediction: delivered, or still in flight."""
        self.text = None
        with self._lock:
            self._fresh = None
            self._generation += 1
