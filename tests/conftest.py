from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wren.agent.permissions import Decision
from wren.agent.ui import AgentUI
from wren.config import ModelConfig
from wren.llm.base import Provider
from wren.llm.types import (
    Completed,
    Message,
    Response,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from wren.tools import ToolContext, ToolOutput


class ScriptedProvider(Provider):
    """Replays canned assistant turns and records what it was sent."""

    def __init__(self, turns: list[Response]):
        self.turns = list(turns)
        self.requests: list[list[Message]] = []

    def stream(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> Iterator[StreamEvent]:
        self.requests.append([Message(m.role, list(m.content)) for m in messages])
        response = self.turns.pop(0)
        if text := response.message.text():
            yield TextDelta(text)
        yield Completed(response)


def reply(text: str) -> Response:
    return Response(Message("assistant", [TextBlock(text)]), "end_turn", Usage(10, 5))


def call(name: str, id: str = "t1", **input: Any) -> Response:
    return Response(Message("assistant", [ToolUseBlock(id, name, input)]), "tool_use", Usage(10, 5))


class RecordingUI(AgentUI):
    def __init__(self, decisions: list[Decision] | None = None, plan_decisions=None):
        self.decisions = list(decisions or [])
        self.plan_decisions = list(plan_decisions or [])
        self.events: list[tuple] = []

    def model_started(self): pass
    def text_delta(self, text): self.events.append(("text", text))
    def thinking_delta(self, text): pass
    def tool_call_started(self, name): pass
    def model_finished(self): pass
    def tool_started(self, name, label): self.events.append(("tool", name, label))

    def confirm(self, tool, args, label, preview) -> Decision:
        self.events.append(("confirm", tool.name, preview))
        return self.decisions.pop(0) if self.decisions else Decision(allow=True)

    def tool_finished(self, name, output: ToolOutput): self.events.append(("result", name, output))

    def review_plan(self, plan):
        self.events.append(("plan", plan))
        return self.plan_decisions.pop(0) if self.plan_decisions else None
    def hook_ran(self, name, status): self.events.append(("hook", name, status))
    def notice(self, text): self.events.append(("notice", text))
    def error(self, text): self.events.append(("error", text))


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path.resolve())


@pytest.fixture
def model() -> ModelConfig:
    return ModelConfig(name="fake", model="fake-1")
