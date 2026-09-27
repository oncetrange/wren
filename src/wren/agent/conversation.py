"""Conversation state and its restore points.

All changes to the conversation go through this class, both while the agent
runs and when a session log is replayed, so a resumed session ends up in
exactly the state the live one was in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wren.agent.compact import mask_old_tool_traffic
from wren.agent.todos import TodoItem
from wren.llm.types import ContentBlock, Message, TextBlock, ThinkingBlock


@dataclass
class RestorePoint:
    kind: Literal["turn", "compaction"]
    # The prompt that started the turn, or a description of the compaction.
    label: str
    # The conversation right before the turn / compaction.
    messages: list[Message]
    # Workspace snapshot taken before the turn (None if checkpoints are off,
    # and always None for compactions, which don't touch files).
    commit: str | None = None
    # For compactions: the summary text that replaced the history, and the
    # index in `messages` from which the tail was kept verbatim.
    summary: str | None = None
    kept_from: int = 0
    # The task list at that point.
    todos: list[TodoItem] | None = None


class Conversation:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.timeline: list[RestorePoint] = []
        self.todos: list[TodoItem] = []

    # --- messages ------------------------------------------------------------

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def add_user(self, blocks: list[ContentBlock]) -> None:
        # Consecutive user content (e.g. tool results followed by a new prompt
        # after an interrupt) is merged into one message.
        if self.messages and self.messages[-1].role == "user":
            self.messages[-1].content.extend(blocks)
        else:
            self.messages.append(Message("user", list(blocks)))

    def strip_thinking(self) -> None:
        """Drop thinking blocks, which only the model that wrote them may accept."""
        for m in self.messages:
            m.content = [b for b in m.content if not isinstance(b, ThinkingBlock)]
        self.messages = [m for m in self.messages if m.content]

    # --- restore points ------------------------------------------------------

    def start_turn(self, prompt: str, commit: str | None) -> None:
        self.timeline.append(RestorePoint("turn", prompt, _copy(self.messages), commit,
                                          todos=list(self.todos)))

    def mask(self, keep_turns: int, min_chars: int) -> int:
        """L1: mask old tool traffic in place. Returns characters freed."""
        self.messages, freed = mask_old_tool_traffic(self.messages, keep_turns, min_chars)
        return freed

    def compacted(self, summary: str, note: Message, kept_from: int | None = None) -> None:
        """Replace messages[:kept_from] with `note` (all of them if None)."""
        kept_from = len(self.messages) if kept_from is None else kept_from
        self.timeline.append(RestorePoint("compaction", "compaction", _copy(self.messages),
                                          summary=summary, kept_from=kept_from))
        self.messages = [note] + self.messages[kept_from:]

    def rewind(self, index: int) -> RestorePoint:
        """Go back to a restore point.

        A turn point restores the conversation as it was before that turn and
        drops every later point. A compaction point undoes that compaction: the
        full history comes back, followed by everything said since.
        """
        point = self.timeline[index]
        if point.kind == "turn":
            self.messages = _copy(point.messages)
            self.todos = list(point.todos or [])
            del self.timeline[index:]
        else:
            # Everything but the summary note is still current: put the
            # summarized head back in front of it.
            since = _copy(self.messages)
            if since and since[0].role == "user" and _is_summary(since[0], point.summary):
                since[0].content = since[0].content[1:]
            self.messages = _copy(point.messages[: point.kept_from])
            for m in since:
                if not m.content:
                    continue
                if m.role == "user":
                    self.add_user(m.content)
                else:
                    self.append(m)
            del self.timeline[index]
        self.strip_thinking()
        return point

    def clear(self) -> None:
        self.messages = []
        self.todos = []


def _copy(messages: list[Message]) -> list[Message]:
    # Blocks are never mutated in place, so copying the lists is enough.
    return [Message(m.role, list(m.content)) for m in messages]


def _is_summary(message: Message, summary: str | None) -> bool:
    first = message.content[0] if message.content else None
    return isinstance(first, TextBlock) and summary is not None and summary in first.text
