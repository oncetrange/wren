"""@-mentions: `@path` in a prompt attaches that file (or directory listing).

    explain @src/wren/cron.py            the whole file (up to read_file's limit)
    fix the loop in @src/app.py#L40-80   lines 40-80
    what's in @docs/                     a directory listing
    @"notes with spaces.md"              quoted, for paths with spaces

A mention counts only at the start of the prompt or after whitespace, and only
if the path exists, so e-mail addresses and handles pass through untouched.
The file goes in as read_file would return it (numbered lines), and counts as
read, so the model can edit it without reading it again.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from wren.tools.base import ToolContext, ToolError
from wren.tools.files import ReadFile

MENTION = re.compile(r'(?:(?<=\s)|^)@(?:"([^"]+)"|([^\s"]+))')
LINES = re.compile(r"^(.*?)#L(\d+)(?:-L?(\d+))?$")
TRAILING = ",;:!?)]}'"
MAX_LISTED = 200
MAX_FILES = 20_000  # listed for completion
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             "dist", "build", "target", ".next", ".tox"}
TAG = "<wren-file"


@dataclass
class Mention:
    text: str  # as written, e.g. "@src/app.py#L40-80"
    path: Path
    start: int | None = None
    end: int | None = None


def find(prompt: str, cwd: Path) -> list[Mention]:
    """The mentions in `prompt` that name existing paths, without duplicates."""
    found: list[Mention] = []
    for m in MENTION.finditer(prompt):
        raw = m.group(1) or m.group(2).rstrip(TRAILING)
        if m.group(2) and raw.endswith(".") and not (cwd / raw).exists():
            raw = raw.rstrip(".")  # "look at @README.md." ends a sentence
        start = end = None
        if lines := LINES.match(raw):
            raw, start = lines.group(1), int(lines.group(2))
            end = int(lines.group(3)) if lines.group(3) else start
        path = (cwd / Path(raw).expanduser()).resolve()
        if path.exists() and not any(f.path == path and (f.start, f.end) == (start, end) for f in found):
            found.append(Mention(m.group(0), path, start, end))
    return found


def attach(mentions: list[Mention], ctx: ToolContext) -> tuple[list[str], list[tuple[str, str]]]:
    """Attachments for the prompt, and (label, summary) lines to show the user."""
    attachments, shown = [], []
    for m in mentions:
        label = ctx.display_path(m.path) + ("/" if m.path.is_dir() else "")
        if m.start:
            label += f"#L{m.start}-{m.end}" if m.end != m.start else f"#L{m.start}"
        try:
            if m.path.is_dir():
                body, summary = _listing(m.path)
            else:
                args: dict = {"path": ctx.display_path(m.path)}
                if m.start:
                    args.update(offset=m.start, limit=max((m.end or m.start) - m.start + 1, 1))
                out = ReadFile().run(args, ctx)
                body, summary = out.content, out.summary
        except (ToolError, OSError) as e:
            shown.append((label, f"not attached: {e}"))
            continue
        attachments.append(f'{TAG} path="{label}">\n{body}\n</wren-file>')
        shown.append((label, summary))
    return attachments, shown


def _listing(path: Path) -> tuple[str, str]:
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    names = [p.name + ("/" if p.is_dir() else "") for p in entries if p.name not in SKIP_DIRS]
    body = "\n".join(names[:MAX_LISTED])
    if len(names) > MAX_LISTED:
        body += f"\n… {len(names) - MAX_LISTED} more"
    return body or "(empty directory)", f"{len(names)} entries"


# --- completion --------------------------------------------------------------------


class FileIndex:
    """The project's files and directories, for completing @-mentions.
    Uses git's view of the tree (tracked and untracked but not ignored) when it
    can, and is refreshed at most every few seconds."""

    def __init__(self, cwd: Path, ttl: float = 5.0):
        self.cwd, self.ttl = cwd, ttl
        self._paths: list[str] = []
        self._at = float("-inf")

    def paths(self) -> list[str]:
        if time.monotonic() - self._at > self.ttl:
            self._paths, self._at = self._scan(), time.monotonic()
        return self._paths

    def _scan(self) -> list[str]:
        files = _git_files(self.cwd)
        if files is None:
            files = _walk(self.cwd)
        dirs = {str(Path(f).parent) + "/" for f in files if "/" in f}
        for d in list(dirs):  # and every ancestor
            parts = d.rstrip("/").split("/")
            dirs.update("/".join(parts[:i]) + "/" for i in range(1, len(parts)))
        return sorted(dirs) + sorted(files)

    def complete(self, prefix: str, limit: int = 50) -> list[str]:
        """Paths for `@prefix`: prefix matches first, then name matches, then substring,
        then fuzzy (the characters in order); shorter paths first within each."""
        paths = self.paths()
        if not prefix:
            return [p for p in paths if p.count("/") <= (1 if p.endswith("/") else 0)][:limit]
        low = prefix.lower()

        def rank(p: str) -> int | None:
            lp = p.lower()
            name = lp.rstrip("/").rsplit("/", 1)[-1]
            if lp.startswith(low):
                return 0
            if name.startswith(low):
                return 1
            if low in lp:
                return 2
            return 3 if _subsequence(low, lp) else None

        ranked = [(r, len(p), p) for p in paths if p != prefix and (r := rank(p)) is not None]
        return [p for _, _, p in sorted(ranked)[:limit]]


def _subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(c in it for c in needle)


def _git_files(cwd: Path) -> list[str] | None:
    try:
        proc = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                              cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.split("\0") if p][:MAX_FILES]


def _walk(cwd: Path) -> list[str]:
    out: list[str] = []
    for root, dirs, files in os.walk(cwd):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        rel = Path(root).relative_to(cwd)
        out += [str(rel / f) if str(rel) != "." else f for f in sorted(files)]
        if len(out) >= MAX_FILES:
            break
    return out[:MAX_FILES]
