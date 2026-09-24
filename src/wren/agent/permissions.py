from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from wren.tools.base import Tool


@dataclass
class Decision:
    allow: bool
    remember: bool = False
    # When denying, an optional note passed back to the model ("use pytest -x instead").
    feedback: str = ""


@dataclass
class Permissions:
    mode: Literal["ask", "auto"] = "ask"
    allowed: set[str] = field(default_factory=set)

    def needs_approval(self, tool: Tool, args: dict[str, Any]) -> bool:
        if tool.read_only or self.mode == "auto":
            return False
        return tool.permission_key(args) not in self.allowed

    def remember(self, tool: Tool, args: dict[str, Any]) -> None:
        self.allowed.add(tool.permission_key(args))
