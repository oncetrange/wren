"""read_file / write_file / edit_file."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput, read_text, write_text

MAX_LINES = 2000
MAX_LINE_CHARS = 2000


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
        "Read a text file. Output lines are prefixed with their line number and a tab "
        "(the prefix is not part of the file). Reads up to 2000 lines by default; "
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
        body = "\n".join(
            f"{n:>6}\t{line[:MAX_LINE_CHARS]}{'…' if len(line) > MAX_LINE_CHARS else ''}"
            for n, line in enumerate(chunk, start=offset)
        )
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

    def permission_key(self, args: dict[str, Any]) -> str:
        return "edit"

    def _plan(self, args: dict[str, Any], ctx: ToolContext) -> tuple[Path, str, str]:
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
        "including whitespace and indentation (do not include read_file's line-number prefix), "
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

    def permission_key(self, args: dict[str, Any]) -> str:
        return "edit"

    def _plan(self, args: dict[str, Any], ctx: ToolContext) -> tuple[Path, str, str, int]:
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

        # Files with CRLF endings: let the model's LF-only strings still match.
        if "\r\n" in text and "\r\n" not in old_s:
            old_s, new_s = old_s.replace("\n", "\r\n"), new_s.replace("\n", "\r\n")

        count = text.count(old_s)
        if count == 0:
            raise ToolError(
                "old_string not found in file. Check whitespace and indentation, "
                "and re-read the file if unsure of its current content."
            )
        if count > 1 and not args.get("replace_all"):
            raise ToolError(
                f"old_string occurs {count} times; add surrounding context to make it "
                "unique, or set replace_all to true"
            )
        new_text = text.replace(old_s, new_s) if args.get("replace_all") else text.replace(old_s, new_s, 1)
        return path, new_text, make_diff(text, new_text, ctx.display_path(path)), count

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
        return ToolOutput(
            f"edited {ctx.display_path(path)} ({n} replacement{'s' if n > 1 else ''})",
            summary=diff_stat(diff),
            diff=diff,
        )
