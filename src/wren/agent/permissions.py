from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from wren.tools.base import Tool

Mode = Literal["ask", "accept_edits", "plan", "auto"]

# Shift+Tab cycles through these; "auto" (--yolo) is only offered if started with it.
CYCLE: tuple[Mode, ...] = ("ask", "accept_edits", "plan")
LABELS: dict[Mode, str] = {
    "ask": "ask before edits",
    "accept_edits": "accept edits",
    "plan": "plan mode (read-only)",
    "auto": "auto: no approvals",
}


@dataclass
class Decision:
    allow: bool
    remember: bool = False
    # When denying, an optional note passed back to the model ("use pytest -x instead").
    feedback: str = ""


@dataclass
class Permissions:
    """What may run without asking.

    ask           reads run freely; edits and commands ask (unless always-allowed)
    accept_edits  edits run freely too; commands ask
    plan          read-only: edits are refused, every command asks (the allowlist
                  doesn't apply, since a command could modify files)
    auto          everything runs
    """

    mode: Mode = "ask"
    allowed: set[str] = field(default_factory=set)
    # Whether Shift+Tab may cycle into "auto" (only when started with --yolo).
    allow_auto: bool = False

    def blocked(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """Why the call may not run at all in this mode, or None."""
        if self.mode == "plan" and tool.edits_files:
            return ("Plan mode is on, so files can't be changed yet. Keep investigating, then "
                    "present your plan for the user to approve.")
        return None

    def needs_approval(self, tool: Tool, args: dict[str, Any]) -> bool:
        if tool.read_only or self.mode == "auto":
            return False
        if self.mode == "plan":
            return True
        if self.mode == "accept_edits" and tool.edits_files:
            return False
        return tool.permission_key(args) not in self.allowed

    def remember(self, tool: Tool, args: dict[str, Any]) -> None:
        self.allowed.add(tool.permission_key(args))

    def cycle(self) -> Mode:
        """Switch to the next mode (Shift+Tab) and return it."""
        modes = CYCLE + (("auto",) if self.allow_auto else ())
        i = modes.index(self.mode) if self.mode in modes else -1
        self.mode = modes[(i + 1) % len(modes)]
        return self.mode
