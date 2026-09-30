"""Context management, in layers from cheapest and least lossy to most:

L0  Tool outputs are truncated when produced (tools/base.py).
L1  Masking: once the prompt passes `mask_at` tokens, large tool outputs
    older than the last KEEP_TURNS model turns are replaced by a short
    placeholder. The call itself stays, so the model can re-run it: nothing
    is lost for good. The model's own messages, tool call arguments included,
    are never masked: models imitate their earlier turns, and on a DeepSWE run
    masked edit arguments were copied into new edits, writing the placeholder
    into source files. Their thinking in those old turns is dropped, though:
    it is often the largest part of a thinking model's history (47% on one
    DeepSWE run), nothing imitates it, and APIs only need the thinking of the
    latest turns, which stay.
L2  Anchored summary: once the prompt passes `compact_at`, everything but the
    last KEEP_TURNS turns is summarized. A previous summary is updated rather
    than rewritten, so details don't drift away over repeated compactions.
L3  Archive: the text being summarized is appended to a transcript file whose
    path goes into the summary, so exact details stay reachable with grep or
    read_file.

Masking is batched with hysteresis (triggered at `mask_at`, it goes down to
MASK_TARGET of it) because every change to earlier messages invalidates the
provider's prompt cache from there on.
"""

from __future__ import annotations

import json
from pathlib import Path

from wren.llm.base import Provider
from wren.llm.types import (
    Completed,
    LLMError,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
    block_to_dict,
)

# Model turns whose tool traffic is kept verbatim, normally and at minimum.
KEEP_TURNS = 8
MIN_KEEP_TURNS = 3
# Tool outputs / arguments shorter than this are left alone: masking them saves little.
MASK_MIN_CHARS = 400
# Masking and compaction bring the prompt down to this share of their
# threshold (hysteresis), so they happen in occasional batches rather than on
# every turn once the recent turns alone are large.
MASK_TARGET = 0.6
COMPACT_TARGET = 0.5
# Masking only happens if it frees at least this share of the threshold:
# each batch costs a prompt-cache miss.
MASK_MIN_GAIN = 0.1

SUMMARY_REQUEST = """\
The conversation is about to be compacted to free up context: every message \
above will be replaced by your summary, and the most recent turns will follow \
it verbatim. {anchor}Do not call any tools; reply with the summary only, using \
these sections:

1. User requests: every request and instruction from the user, quoting the most \
recent one verbatim, including preferences and constraints they stated.
2. Progress: what has been done, key decisions and why.
3. Files: each file read or changed, with its path and the details that matter \
(function names, signatures, important snippets).
4. Errors and fixes: problems hit and how they were resolved.
5. Current state: exactly what was being worked on right before this summary.
6. Next steps: what remains, in order. Only include steps the user asked for.

Be specific: exact paths, names, commands and values beat prose."""

ANCHOR = """The first message holds the summary from an earlier compaction: produce \
an updated version of it. Keep everything in it that is still relevant, add what \
happened since, and drop what is obsolete. """

RESUME_NOTE = """\
Earlier parts of this session were compacted to save context. Summary:

<summary>
{summary}
</summary>
{archive}{todos}
The most recent messages follow verbatim. File contents seen earlier may be \
out of date: re-read files before editing them."""

ARCHIVE_NOTE = """
The full transcript of everything summarized is saved at {path}; search it with \
grep or read_file when you need exact details (earlier outputs, error messages, code).
"""


def is_summary_note(text: str) -> bool:
    return text.startswith(RESUME_NOTE.split("\n", 1)[0])


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token count (~4 characters per token)."""
    chars = sum(len(json.dumps(block_to_dict(b), ensure_ascii=False)) for m in messages for b in m.content)
    return chars // 4


# --- L1: masking -------------------------------------------------------------


def recent_start(messages: list[Message], keep_turns: int = KEEP_TURNS) -> int:
    """Index of the first message of the last `keep_turns` model turns
    (an assistant message), or 0 if there aren't that many turns."""
    turns = [i for i, m in enumerate(messages) if m.role == "assistant"]
    return turns[-keep_turns] if len(turns) >= keep_turns else 0


def mask_old_tool_traffic(messages: list[Message], keep_turns: int = KEEP_TURNS,
                          min_chars: int = MASK_MIN_CHARS) -> tuple[list[Message], int]:
    """Return (new messages, characters freed). Input messages are not mutated."""
    cutoff = recent_start(messages, keep_turns)
    calls = {b.id: b for m in messages[:cutoff] for b in m.content if isinstance(b, ToolUseBlock)}
    freed = 0
    out: list[Message] = []
    for i, m in enumerate(messages):
        if i >= cutoff:
            out.append(m)
            continue
        thinking = [b for b in m.content if isinstance(b, ThinkingBlock)]
        if thinking and len(thinking) < len(m.content):  # never leave a message empty
            freed += sum(len(b.thinking) + len(b.redacted_data or "") for b in thinking)
        else:
            thinking = []
        blocks = []
        for b in m.content:
            if isinstance(b, ThinkingBlock) and thinking:
                continue
            if isinstance(b, ToolResultBlock) and len(b.content) > min_chars and not _is_masked(b.content):
                call = calls.get(b.tool_use_id)
                placeholder = _result_placeholder(call, len(b.content))
                freed += len(b.content) - len(placeholder)
                b = ToolResultBlock(b.tool_use_id, placeholder, b.is_error)
            blocks.append(b)
        out.append(Message(m.role, blocks))
    return out, freed


def _result_placeholder(call: ToolUseBlock | None, size: int) -> str:
    what = f"{call.name} call" if call else "tool call"
    return f"[output omitted to save context: {size} characters from this {what}. Re-run the call if you need it again.]"


def _is_masked(text: str) -> bool:
    return text.startswith("[output omitted to save context")


# --- L2 + L3: anchored summary with an archive ------------------------------


def summarize(provider: Provider, system: str, span: list[Message],
              tools: list[ToolSpec]) -> tuple[str, Usage]:
    """Summarize `span` (which ends with a user message). If it starts with a
    previous summary note, that summary is updated instead of rewritten.
    Returns the summary and the request's usage."""
    request = [Message(m.role, list(m.content)) for m in span]
    anchored = bool(span) and any(isinstance(b, TextBlock) and is_summary_note(b.text)
                                  for b in span[0].content)
    text = SUMMARY_REQUEST.format(anchor=ANCHOR if anchored else "")
    if request and request[-1].role == "user":
        request[-1].content.append(TextBlock(text))
    else:
        request.append(Message("user", [TextBlock(text)]))

    # Tools stay declared: the API rejects tool_use blocks in the history otherwise.
    response = None
    for event in provider.stream(system=system, messages=request, tools=tools):
        if isinstance(event, Completed):
            response = event.response
    summary = response.message.text().strip() if response else ""
    if not summary or response is None:
        raise LLMError("compaction failed: the model returned no summary")
    return summary, response.usage


def summary_note(summary: str, archive: Path | None, todos: str = "") -> Message:
    archive_text = ARCHIVE_NOTE.format(path=archive) if archive else ""
    todos_text = f"\nYour task list (keep maintaining it with todo_write):\n{todos}\n" if todos else ""
    return Message("user", [TextBlock(RESUME_NOTE.format(summary=summary, archive=archive_text,
                                                         todos=todos_text))])


def append_archive(path: Path, span: list[Message], heading: str) -> None:
    """Append a readable transcript of `span` to the archive file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"\n\n# {heading}\n"]
    for m in span:
        for b in m.content:
            if isinstance(b, TextBlock):
                if not is_summary_note(b.text):  # already archived by the previous compaction
                    parts.append(f"\n## {m.role}\n\n{b.text}\n")
            elif isinstance(b, ToolUseBlock):
                parts.append(f"\n## tool call: {b.name} ({b.id})\n\n```json\n"
                             f"{json.dumps(b.input, ensure_ascii=False, indent=2)}\n```\n")
            elif isinstance(b, ToolResultBlock):
                label = "error" if b.is_error else "result"
                parts.append(f"\n## tool {label} ({b.tool_use_id})\n\n```\n{b.content}\n```\n")
    with path.open("a", encoding="utf-8") as f:
        f.write("".join(parts))


def plan_mask(messages: list[Message], current_tokens: int, mask_at: int) -> tuple[int, int]:
    """Choose how many recent turns to keep so masking gets the prompt under
    MASK_TARGET * mask_at. Returns (keep_turns, chars freed); keeps as many
    turns as possible, down to MIN_KEEP_TURNS. Freed is 0 when the gain would
    be too small to be worth a cache miss."""
    target = mask_at * MASK_TARGET
    best = (KEEP_TURNS, 0)
    for keep in range(KEEP_TURNS, MIN_KEEP_TURNS - 1, -1):
        _, freed = mask_old_tool_traffic(messages, keep)
        best = (keep, freed)
        if current_tokens - freed // 4 <= target:
            break
    if best[1] // 4 < mask_at * MASK_MIN_GAIN:
        return best[0], 0
    return best


def plan_compaction(messages: list[Message], fixed_tokens: int, compact_at: int) -> int:
    """Index from which to keep messages verbatim: as many recent turns as fit
    in COMPACT_TARGET * compact_at alongside the note, or len(messages) (keep
    nothing, summarize everything) if not even one turn fits."""
    target = compact_at * COMPACT_TARGET
    for keep in range(KEEP_TURNS, 0, -1):
        start = recent_start(messages, keep)
        if start >= 2 and fixed_tokens + estimate_tokens(messages[start:]) <= target:
            return start
    return len(messages)
