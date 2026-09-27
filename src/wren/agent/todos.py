"""The task list the model keeps while working on multi-step requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal["pending", "in_progress", "completed"]
STATUSES: tuple[Status, ...] = ("pending", "in_progress", "completed")
_MARKS = {"completed": "✓", "in_progress": "▸", "pending": "○"}


@dataclass(frozen=True)
class TodoItem:
    content: str
    status: Status = "pending"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_todos(raw: Any) -> list[TodoItem]:
    """Validate the model's list. Raises ValueError with a message for the model."""
    if not isinstance(raw, list):
        raise ValueError("todos must be a list")
    items = []
    for i, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"item {i} must be an object with content and status")
        content, status = entry.get("content"), entry.get("status", "pending")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"item {i} needs a non-empty content")
        if status not in STATUSES:
            raise ValueError(f"item {i} has status {status!r}; use one of {', '.join(STATUSES)}")
        items.append(TodoItem(content.strip(), status))
    if sum(t.status == "in_progress" for t in items) > 1:
        raise ValueError("only one item can be in_progress at a time; finish or pause the others")
    return items


def format_todos(items: list[TodoItem]) -> str:
    return "\n".join(f"{_MARKS[t.status]} {t.content}" for t in items)


def progress(items: list[TodoItem]) -> str:
    done = sum(t.status == "completed" for t in items)
    return f"{done}/{len(items)} done"
