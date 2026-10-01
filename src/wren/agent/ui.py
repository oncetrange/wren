"""What the agent tells the user and asks them.

Every method has a quiet default, so a UI implements only what it shows: the
terminal UI (cli/ui.py), a subagent's view of its parent's (agent/subagents.py),
a test's recorder. Defaults that answer a question answer as if nobody were
there: a tool call is refused, a plan isn't approved.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from wren.agent.permissions import Decision

if TYPE_CHECKING:
    from wren.agent.plans import PlanDecision
    from wren.tools.base import Tool, ToolOutput


class AgentUI:
    # --- the model's output, as it streams
    def model_started(self) -> None: ...
    def text_delta(self, text: str) -> None: ...
    def thinking_delta(self, text: str) -> None: ...
    def tool_call_started(self, name: str) -> None: ...
    def model_finished(self) -> None: ...

    # --- tool calls
    def tool_started(self, name: str, label: str) -> None: ...
    def tool_finished(self, name: str, output: ToolOutput) -> None: ...

    def confirm(self, tool: Tool, args: dict[str, Any], label: str, preview: str | None) -> Decision:
        """May this call run? (Only asked when the permission mode says so.)"""
        return Decision(allow=False)

    def review_plan(self, plan: str) -> PlanDecision | None:
        """Approve a plan from plan mode; None when there's nobody to ask."""
        return None

    # --- everything else
    def hook_ran(self, name: str, status: str) -> None: ...
    def notice(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...

    def attached(self, label: str, summary: str) -> None:
        """A file @-mentioned in the prompt, and what was attached."""

    def nested(self) -> AbstractContextManager[None]:
        """Output inside this context belongs to a subagent (shown indented)."""
        return contextlib.nullcontext()

    def progress(self, lines: list[str] | None) -> None:
        """Live status lines of subagents running at the same time; None ends them."""
