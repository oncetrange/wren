"""Small text helpers shared by the CLI modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from wren.agent.session import SessionState


def one_line(text: str, width: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1] + "…"


def session_options(sessions: list[SessionState], current: Path | None = None):
    """Picker rows for sessions: when, the first prompt, and size."""
    options = []
    for s in sessions:
        when = datetime.fromtimestamp(s.updated).strftime("%m-%d %H:%M")
        mark = " (current)" if s.path == current else ""
        options.append((s, f"{when}  {one_line(s.first_prompt, 50)}  · {len(s.messages)} messages{mark}"))
    return options
