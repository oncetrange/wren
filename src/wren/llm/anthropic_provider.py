"""Adapter for the Anthropic Messages API and Anthropic-compatible endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic

from wren.config import ModelConfig
from wren.llm.base import Provider
from wren.llm.types import (
    Completed,
    ContentBlock,
    LLMError,
    Message,
    Response,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallStarted,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

_CACHE = {"type": "ephemeral"}
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicProvider(Provider):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        key = cfg.api_key()
        # Pass credentials explicitly so the SDK never falls back to
        # ANTHROPIC_API_KEY and ships it to a third-party base_url.
        auth: dict[str, Any] = {"api_key": key}
        if cfg.auth == "bearer":
            auth["auth_token"] = key
        self.client = anthropic.Anthropic(base_url=cfg.base_url, max_retries=4, **auth)

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        params: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "system": _system_param(system, self.cfg.prompt_cache),
            "messages": _messages_param(messages, self.cfg.prompt_cache),
        }
        if tools:  # an empty list is left out rather than sent
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        if self.cfg.thinking:
            params["thinking"] = _thinking_param(self.cfg.thinking)
        try:
            with self.client.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                    elif event.type == "thinking":
                        yield ThinkingDelta(event.thinking)
                    elif (
                        event.type == "content_block_start"
                        and event.content_block.type == "tool_use"
                    ):
                        yield ToolCallStarted(event.content_block.name)
                final = stream.get_final_message()
        except anthropic.APIStatusError as e:
            raise LLMError(f"{self.cfg.name}: HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{self.cfg.name}: connection failed: {e}") from e
        yield Completed(_to_response(final))


def _thinking_param(thinking: str | dict[str, Any]) -> dict[str, Any]:
    return {"type": thinking} if isinstance(thinking, str) else dict(thinking)


def _system_param(system: str, cache: bool) -> Any:
    if not cache:
        return system
    return [{"type": "text", "text": system, "cache_control": _CACHE}]


def _messages_param(messages: list[Message], cache: bool) -> list[dict[str, Any]]:
    out = [
        {"role": m.role, "content": [_block_param(b) for b in m.content]} for m in messages
    ]
    # One breakpoint at the tail caches the whole conversation prefix; the
    # next request reads it back and only pays for the new turn.
    if cache and out and out[-1]["content"]:
        out[-1]["content"][-1]["cache_control"] = _CACHE
    return out


def _block_param(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock(text):
            return {"type": "text", "text": text}
        case ThinkingBlock(_, _, redacted) if redacted is not None:
            return {"type": "redacted_thinking", "data": redacted}
        case ThinkingBlock(thinking, signature):
            return {"type": "thinking", "thinking": thinking, "signature": signature}
        case ToolUseBlock(id, name, input):
            return {"type": "tool_use", "id": id, "name": name, "input": input}
        case ToolResultBlock(tool_use_id, content, is_error):
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
            }
    raise TypeError(f"unknown block: {block!r}")


def _to_response(msg: Any) -> Response:
    blocks: list[ContentBlock] = []
    for b in msg.content:
        if b.type == "text":
            blocks.append(TextBlock(b.text))
        elif b.type == "thinking":
            blocks.append(ThinkingBlock(b.thinking, b.signature or ""))
        elif b.type == "redacted_thinking":
            blocks.append(ThinkingBlock("", redacted_data=b.data))
        elif b.type == "tool_use":
            blocks.append(ToolUseBlock(b.id, b.name, b.input if isinstance(b.input, dict) else {}))
    u = msg.usage
    usage = Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
    )
    return Response(
        Message("assistant", blocks),
        _STOP_REASONS.get(msg.stop_reason or "", "other"),
        usage,
    )
