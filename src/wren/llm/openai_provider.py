"""Adapter for OpenAI-compatible Chat Completions endpoints.

Covers OpenAI itself and the many services and local servers that speak the
same protocol (DeepSeek, DashScope and Moonshot's compatible modes,
OpenRouter, vLLM, Ollama, LM Studio...). The differences from wren's
Anthropic-shaped types are handled here:

- the system prompt is the first message;
- tool results are separate `tool` messages that must directly follow the
  assistant message that called them, so a user message holding results and
  text becomes the tool messages first, then one user message with the text;
- there is no error flag on tool results, so errors are marked in the text;
- tool call arguments are a JSON string, streamed in pieces per call index;
- reasoning arrives as `reasoning_content` (or `reasoning`) and is only sent
  back when the model's API requires it (`replay_reasoning`);
- prompt caching is automatic; cache hits are reported inside prompt_tokens.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import openai

from wren.config import ModelConfig
from wren.llm.base import Provider
from wren.llm.types import (
    INVALID_JSON_KEY,
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

OPENAI_URL = "https://api.openai.com/v1"
_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIProvider(Provider):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # The key and base URL are always explicit, so the SDK never falls back
        # to OPENAI_API_KEY / OPENAI_BASE_URL and sends one to the other's host.
        self.client = openai.OpenAI(api_key=cfg.api_key() or "none",
                                    base_url=cfg.base_url or OPENAI_URL, max_retries=4)

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        params: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages_param(system, messages, self.cfg.replay_reasoning),
            "stream": True,
            "stream_options": {"include_usage": True},
            _max_tokens_param(self.cfg): self.cfg.max_tokens,
        }
        if tools:  # some servers reject an empty list
            params["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]
        if self.cfg.reasoning_effort:
            params["reasoning_effort"] = self.cfg.reasoning_effort
        if self.cfg.extra_body:
            params["extra_body"] = self.cfg.extra_body
        acc = _Accumulator()
        try:
            for chunk in self.client.chat.completions.create(**params):
                yield from acc.add(chunk)
        except openai.APIStatusError as e:
            raise LLMError(f"{self.cfg.name}: HTTP {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"{self.cfg.name}: connection failed: {e}") from e
        yield Completed(acc.response())


def _max_tokens_param(cfg: ModelConfig) -> str:
    if cfg.max_tokens_param:
        return cfg.max_tokens_param
    return "max_completion_tokens" if cfg.base_url in (None, OPENAI_URL) else "max_tokens"


# --- requests ------------------------------------------------------------------


def messages_param(system: str, messages: list[Message], replay_reasoning: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "assistant":
            out.append(_assistant_param(m, replay_reasoning))
            continue
        texts = []
        for b in m.content:
            if isinstance(b, ToolResultBlock):
                content = f"Error: {b.content}" if b.is_error else b.content
                out.append({"role": "tool", "tool_call_id": b.tool_use_id, "content": content})
            elif isinstance(b, TextBlock):
                texts.append(b.text)
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
    return out


def _assistant_param(m: Message, replay_reasoning: bool) -> dict[str, Any]:
    text = m.text()
    calls = [{"id": b.id, "type": "function",
              "function": {"name": b.name, "arguments": _arguments(b.input)}}
             for b in m.tool_uses()]
    msg: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "")}
    if calls:
        msg["tool_calls"] = calls
        reasoning = "".join(b.thinking for b in m.content if isinstance(b, ThinkingBlock))
        if replay_reasoning and reasoning:
            msg["reasoning_content"] = reasoning
    return msg


def _arguments(input: dict[str, Any]) -> str:
    # A call whose arguments weren't valid JSON goes back exactly as the model wrote it.
    if INVALID_JSON_KEY in input:
        return str(input[INVALID_JSON_KEY])
    return json.dumps(input, ensure_ascii=False)


# --- responses -----------------------------------------------------------------


class _Accumulator:
    """Builds the final response from stream chunks, yielding UI events on the way."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish: str | None = None
        self.usage: Any = None

    def add(self, chunk: Any) -> Iterator[StreamEvent]:
        if chunk.usage is not None:
            self.usage = chunk.usage
        for choice in chunk.choices[:1]:
            delta = choice.delta
            if reasoning := (getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)):
                if isinstance(reasoning, str):
                    self.reasoning.append(reasoning)
                    yield ThinkingDelta(reasoning)
            if delta.content:
                self.text.append(delta.content)
                yield TextDelta(delta.content)
            for tc in delta.tool_calls or []:
                slot = self.calls.setdefault(tc.index, {"id": "", "name": "", "args": []})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function is not None:
                    if tc.function.name and not slot["name"]:
                        slot["name"] = tc.function.name
                        yield ToolCallStarted(tc.function.name)
                    if tc.function.arguments:
                        slot["args"].append(tc.function.arguments)
            if choice.finish_reason:
                self.finish = choice.finish_reason

    def response(self) -> Response:
        blocks: list[ContentBlock] = []
        if self.reasoning:
            blocks.append(ThinkingBlock("".join(self.reasoning)))
        if text := "".join(self.text):
            blocks.append(TextBlock(text))
        for _, slot in sorted(self.calls.items()):
            blocks.append(ToolUseBlock(slot["id"] or f"call_{uuid.uuid4().hex[:24]}", slot["name"],
                                       _parse_arguments("".join(slot["args"]))))
        stop = _STOP_REASONS.get(self.finish or "", "other")
        if self.calls and stop == "end_turn":
            stop = "tool_use"  # some servers finish tool calls with "stop"
        return Response(Message("assistant", blocks), stop, _usage(self.usage))


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {INVALID_JSON_KEY: raw}
    return value if isinstance(value, dict) else {INVALID_JSON_KEY: raw}


def _usage(u: Any) -> Usage:
    if u is None:
        return Usage()
    prompt = u.prompt_tokens or 0
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) if details is not None else None) \
        or getattr(u, "prompt_cache_hit_tokens", None) or 0  # DeepSeek's field
    return Usage(input_tokens=max(prompt - cached, 0), output_tokens=u.completion_tokens or 0,
                 cache_read_tokens=cached)
