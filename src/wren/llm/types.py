"""Provider-neutral message types.

The agent only ever speaks in these types; each provider adapter converts
them to and from its own wire format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    # Opaque provider data that must be replayed verbatim (e.g. Anthropic signatures).
    signature: str = ""
    redacted_data: str | None = None


# In a ToolUseBlock's input: the model's arguments weren't a JSON object; the
# raw text is kept under this key so the call can be refused and replayed.
INVALID_JSON_KEY = "__wren_invalid_json__"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock

_BLOCK_TYPES: dict[str, type] = {
    "text": TextBlock,
    "thinking": ThinkingBlock,
    "tool_use": ToolUseBlock,
    "tool_result": ToolResultBlock,
}
_TYPE_NAMES = {cls: name for name, cls in _BLOCK_TYPES.items()}


def block_to_dict(block: ContentBlock) -> dict[str, Any]:
    return {"type": _TYPE_NAMES[type(block)], **asdict(block)}


def block_from_dict(data: dict[str, Any]) -> ContentBlock:
    data = dict(data)
    return _BLOCK_TYPES[data.pop("type")](**data)


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": [block_to_dict(b) for b in self.content]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(data["role"], [block_from_dict(b) for b in data["content"]])


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def context_tokens(self) -> int:
        """Total prompt size of the request this usage came from."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class Response:
    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)


# --- stream events -----------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCallStarted:
    """The model began emitting a tool call (its input is still streaming)."""

    name: str


@dataclass
class Completed:
    response: Response


StreamEvent = TextDelta | ThinkingDelta | ToolCallStarted | Completed


class LLMError(Exception):
    """A provider request failed after the SDK's own retries."""
