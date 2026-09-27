"""read_file / write_file / edit_file."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput, read_text, write_text
from wren.tools.matching import closest_region, find_loose, split_block

MAX_LINES = 2000
MAX_LINE_CHARS = 2000


def number_lines(lines: list[str], first: int) -> str:
    """read_file's format. The separator is an arrow, not a tab: with a tab,
    models can't tell it apart from indentation in tab-indented files."""
    return "\n".join(
        f"{n:>6}→{line[:MAX_LINE_CHARS]}{'…' if len(line) > MAX_LINE_CHARS else ''}"
        for n, line in enumerate(lines, start=first)
    )


def make_diff(old: str, new: str, label: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        )
    )


def diff_stat(diff: str) -> str:
    lines = diff.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return f"+{added} -{removed}"


def _reject_placeholders(args: dict[str, Any], *keys: str) -> None:
    """Refuse text that is a context-management placeholder, not file content.

    Models sometimes copy the "[... omitted to save context]" markers they see
    in their history into new edits; written to disk they corrupt the file.
    """
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and "omitted to save context" in value:
            raise ToolError(
                f"{key} contains a context placeholder (\"... omitted to save context\"), which "
                "marks text removed from the conversation, not real file content. Re-read the "
                "file if needed and send the actual text."
            )


def _check_fresh(path: Path, ctx: ToolContext) -> None:
    """Refuse to modify a file the model hasn't seen in its current state."""
    seen = ctx.read_files.get(path)
    if seen is None:
        raise ToolError(f"read {ctx.display_path(path)} with read_file before modifying it")
    if path.stat().st_mtime_ns != seen:
        raise ToolError(
            f"{ctx.display_path(path)} changed on disk since it was last read; read it again"
        )


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file. Each output line is prefixed with its line number and an arrow, "
        "e.g. '    12→    return x' (the prefix is not part of the file). Reads up to 2000 lines by default; "
        "use offset/limit for large files. A file must be read before it can be edited."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, absolute or relative to the working directory"},
            "offset": {"type": "integer", "description": "1-based line number to start from"},
            "limit": {"type": "integer", "description": "Maximum number of lines to return"},
        },
        "required": ["path"],
    }
    read_only = True

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("path", "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args["path"])
        if not path.exists():
            raise ToolError(f"file not found: {args['path']}")
        if path.is_dir():
            raise ToolError(f"{args['path']} is a directory; use glob to list files")
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            raise ToolError(f"{args['path']} looks like a binary file")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{args['path']} is not valid UTF-8 text") from None

        lines = text.splitlines()
        offset = max(args.get("offset", 1), 1)
        limit = max(args.get("limit", MAX_LINES), 1)
        chunk = lines[offset - 1 : offset - 1 + limit]
        ctx.mark_read(path)

        if not lines:
            return ToolOutput("(empty file)", summary="empty file")
        if not chunk:
            raise ToolError(f"offset {offset} is past the end of the file ({len(lines)} lines)")
        body = number_lines(chunk, offset)
        end = offset + len(chunk) - 1
        if end < len(lines):
            body += f"\n\n(showing lines {offset}-{end} of {len(lines)}; use offset to read more)"
        return ToolOutput(body, summary=f"read {len(chunk)} lines")


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. Prefer edit_file for "
        "changing existing files. Overwriting requires reading the file first. "
        "Parent directories are created as needed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    }

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("path", "")

    edits_files = True

    def permission_key(self, args: dict[str, Any]) -> str:
        return "edit"

    def _plan(self, args: dict[str, Any], ctx: ToolContext) -> tuple[Path, str, str]:
        _reject_placeholders(args, "content")
        path = ctx.resolve(args["path"])
        if path.is_dir():
            raise ToolError(f"{args['path']} is a directory")
        old = ""
        if path.exists():
            _check_fresh(path, ctx)
            old = read_text(path)
        return path, old, make_diff(old, args["content"], ctx.display_path(path))

    def preview(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        try:
            return self._plan(args, ctx)[2]
        except ToolError:
            return None

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path, old, diff = self._plan(args, ctx)
        existed = path.exists()
        write_text(path, args["content"])
        ctx.mark_read(path)
        verb = "updated" if existed else "created"
        return ToolOutput(
            f"{verb} {ctx.display_path(path)}",
            summary=f"{verb} ({diff_stat(diff)})",
            diff=diff,
        )


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match the file content exactly, "
        "including whitespace and indentation (do not include read_file's 'N→' line prefix), "
        "and must be unique in the file unless replace_all is true. Include enough surrounding "
        "lines to make it unique. The file must have been read first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("path", "")

    edits_files = True

    def permission_key(self, args: dict[str, Any]) -> str:
        return "edit"

    def _plan(self, args: dict[str, Any], ctx: ToolContext) -> tuple[Path, str, str, int]:
        _reject_placeholders(args, "old_string", "new_string")
        path = ctx.resolve(args["path"])
        old_s, new_s = args["old_string"], args["new_string"]
        if not path.is_file():
            raise ToolError(f"file not found: {args['path']} (use write_file to create files)")
        if old_s == new_s:
            raise ToolError("old_string and new_string are identical")
        if not old_s:
            raise ToolError("old_string must not be empty")
        _check_fresh(path, ctx)
        text = read_text(path)

        # Pure CRLF files are edited as LF and converted back, so the model's
        # LF strings match.
        crlf = "\r\n" in text and text.count("\r\n") == text.count("\n")
        work = text.replace("\r\n", "\n") if crlf else text
        if crlf:
            old_s, new_s = old_s.replace("\r\n", "\n"), new_s.replace("\r\n", "\n")

        count = work.count(old_s)
        self.last_note = ""
        if count == 0:
            new_work, count = self._loose_replace(work, old_s, new_s), 1
        elif count > 1 and not args.get("replace_all"):
            raise ToolError(
                f"old_string occurs {count} times; add surrounding context to make it "
                "unique, or set replace_all to true"
            )
        else:
            new_work = work.replace(old_s, new_s) if args.get("replace_all") else work.replace(old_s, new_s, 1)
        new_text = new_work.replace("\n", "\r\n") if crlf else new_work
        return path, new_text, make_diff(text, new_text, ctx.display_path(path)), count

    def _loose_replace(self, work: str, old_s: str, new_s: str) -> str:
        """No exact match: match ignoring indentation, then re-indent new_string."""
        lines = work.split("\n")
        matches = find_loose(lines, old_s)
        if len(matches) == 1:
            m = matches[0]
            new_lines = split_block(m.reindent(new_s)) if new_s else []
            self.last_note = (
                f"old_string matched lines {m.start + 1}-{m.end} only after adjusting indentation; "
                "new_string was re-indented to match the file. Check the diff."
            )
            return "\n".join(lines[: m.start] + new_lines + lines[m.end:])
        if len(matches) > 1:
            where = ", ".join(str(m.start + 1) for m in matches[:10])
            raise ToolError(
                f"old_string not found exactly; ignoring indentation it matches {len(matches)} "
                f"places (lines {where}). Add surrounding lines to make it unique."
            )
        message = "old_string not found in file."
        region = closest_region(lines, old_s)
        if region:
            best, first, snippet = region
            message += (f" The most similar part of the file is around line {best} (current "
                        f"content, with line numbers):\n{number_lines(snippet, first)}")
        else:
            message += " Re-read the file to see its current content."
        raise ToolError(message)

    def preview(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        try:
            return self._plan(args, ctx)[2]
        except ToolError:
            return None

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path, new_text, diff, count = self._plan(args, ctx)
        write_text(path, new_text)
        ctx.mark_read(path)
        n = count if args.get("replace_all") else 1
        note = f"\n{self.last_note}" if self.last_note else ""
        return ToolOutput(
            f"edited {ctx.display_path(path)} ({n} replacement{'s' if n > 1 else ''}){note}",
            summary=diff_stat(diff),
            diff=diff,
        )
