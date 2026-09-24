from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from wren.llm.types import Message, StreamEvent, ToolSpec


class Provider(ABC):
    """A chat model that supports tool calling.

    `stream` yields incremental events and must finish with exactly one
    `Completed` event carrying the full assistant message.
    """

    @abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]: ...
