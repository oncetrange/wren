"""Lifecycle events and the handlers that react to them.

Built-in policies (plan mode, reminders) and user hooks are all handlers, run
in registration order, so the rules for combining them live in one place:

  pre_tool     before a tool runs. Any "deny" stops the call (later handlers
               are skipped, so built-in guards registered first can't be
               overridden); an "allow" with no deny skips the permission prompt;
               otherwise the permission mode decides.
  post_tool    after a tool ran. Each `context` is appended to the tool result.
  prompt       a user prompt was submitted. "deny" rejects it (reason shown to
               the user); each `context` is added for the model.
  stop         the model is about to finish. Each "block" keeps it going, with
               the reasons sent as one message. Capped per request.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from wren.llm.types import ToolUseBlock
from wren.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    from wren.agent.loop import Agent

EventName = Literal["pre_tool", "post_tool", "prompt", "stop"]


@dataclass
class PreToolUse:
    agent: Agent
    tool: Tool
    call: ToolUseBlock


@dataclass
class PostToolUse:
    agent: Agent
    tool: Tool
    call: ToolUseBlock
    output: ToolOutput


@dataclass
class PromptSubmit:
    agent: Agent
    prompt: str


@dataclass
class Stop:
    agent: Agent
    final_text: str
    # Names of handlers that already blocked stopping during this request.
    blocked_by: set[str] = field(default_factory=set)


Event = PreToolUse | PostToolUse | PromptSubmit | Stop


@dataclass
class Verdict:
    """A handler's answer. All fields optional: None/"" means no opinion."""

    decision: Literal["allow", "deny", "block"] | None = None
    # Why (deny / block): sent to the model, or shown to the user for prompts.
    reason: str = ""
    # Extra text for the model (post_tool, prompt).
    context: str = ""
    # Filled in by the registry.
    source: str = ""


Handler = Callable[[Any], Verdict | None]


@dataclass
class Registration:
    name: str
    fn: Handler


class Hooks:
    def __init__(self) -> None:
        self._handlers: dict[EventName, list[Registration]] = {
            "pre_tool": [], "post_tool": [], "prompt": [], "stop": []}

    def on(self, event: EventName, name: str, fn: Handler) -> None:
        self._handlers[event].append(Registration(name, fn))

    def names(self, event: EventName) -> list[str]:
        return [r.name for r in self._handlers[event]]

    def run(self, event_name: EventName, event: Event) -> list[Verdict]:
        """Call the handlers in order. For pre_tool and prompt, a deny ends the chain."""
        verdicts = []
        for reg in self._handlers[event_name]:
            verdict = reg.fn(event)
            if verdict is None:
                continue
            verdict.source = reg.name
            verdicts.append(verdict)
            if verdict.decision == "deny" and event_name in ("pre_tool", "prompt"):
                break
        return verdicts


def denial(verdicts: list[Verdict]) -> Verdict | None:
    return next((v for v in verdicts if v.decision == "deny"), None)


def allowed(verdicts: list[Verdict]) -> bool:
    return any(v.decision == "allow" for v in verdicts) and denial(verdicts) is None
