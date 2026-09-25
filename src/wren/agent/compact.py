"""Conversation compaction: replace history with a model-written summary."""

from __future__ import annotations

import json

from wren.llm.base import Provider
from wren.llm.types import (
    Completed,
    LLMError,
    Message,
    TextBlock,
    ToolSpec,
    block_to_dict,
)

# Compact once the prompt reaches this share of the context window, leaving
# room for the summary request itself and the next response.
COMPACT_AT = 0.8

SUMMARY_REQUEST = """\
The conversation is about to be compacted to free up context. Write a summary \
that lets you continue the work seamlessly with no other history. Do not call \
any tools; reply with the summary only, using these sections:

1. User requests: every request and instruction from the user, quoting the most \
recent one verbatim, including preferences and constraints they stated.
2. Progress: what has been done, key decisions and why.
3. Files: each file read or changed, with its path and the details that matter \
(function names, signatures, important snippets).
4. Errors and fixes: problems hit and how they were resolved.
5. Current state: exactly what was being worked on right before this summary.
6. Next steps: what remains, in order. Only include steps the user asked for.

Be specific: exact paths, names, commands and values beat prose."""

RESUME_NOTE = """\
This session was compacted to save context. Summary of the conversation so far:

<summary>
{summary}
</summary>

Continue from where the conversation left off. File contents are not in context \
anymore: re-read files before editing them."""


def is_summary_note(text: str) -> bool:
    return text.startswith(RESUME_NOTE.split("\n", 1)[0])


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token count (~4 characters per token) for not-yet-billed content."""
    chars = sum(len(json.dumps(block_to_dict(b), ensure_ascii=False)) for m in messages for b in m.content)
    return chars // 4


def summarize(provider: Provider, system: str, messages: list[Message], tools: list[ToolSpec]) -> str:
    request = [Message(m.role, list(m.content)) for m in messages]
    if request and request[-1].role == "user":
        request[-1].content.append(TextBlock(SUMMARY_REQUEST))
    else:
        request.append(Message("user", [TextBlock(SUMMARY_REQUEST)]))

    # Tools stay declared: the API rejects tool_use blocks in the history otherwise.
    response = None
    for event in provider.stream(system=system, messages=request, tools=tools):
        if isinstance(event, Completed):
            response = event.response
    summary = response.message.text().strip() if response else ""
    if not summary:
        raise LLMError("compaction failed: the model returned no summary")
    return summary


def compacted_history(summary: str) -> list[Message]:
    return [Message("user", [TextBlock(RESUME_NOTE.format(summary=summary))])]
