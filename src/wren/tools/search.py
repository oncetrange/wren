"""glob / grep."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput, truncate

MAX_RESULTS = 200
MAX_MATCH_CHARS = 300
IGNORED_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build"}


def _search_root(args: dict[str, Any], ctx: ToolContext) -> Path:
    root = ctx.resolve(args.get("path") or ".")
    if not root.exists():
        raise ToolError(f"path not found: {args.get('path')}")
    return root


class Glob(Tool):
    name = "glob"
    description = (
        "Find files by glob pattern (e.g. '**/*.py', 'src/**/test_*.ts'), newest first. "
        "Skips VCS, dependency and cache directories such as .git and node_modules."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search (default: working directory)"},
        },
        "required": ["pattern"],
    }
    read_only = True

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("pattern", "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        root = _search_root(args, ctx)
        matches = [
            p for p in root.glob(args["pattern"])
            if p.is_file() and not IGNORED_DIRS.intersection(p.relative_to(root).parts)
        ]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return ToolOutput("no files found", summary="0 files")
        shown = [ctx.display_path(p) for p in matches[:MAX_RESULTS]]
        body = "\n".join(shown)
        if len(matches) > MAX_RESULTS:
            body += f"\n\n({len(matches) - MAX_RESULTS} more not shown; narrow the pattern)"
        return ToolOutput(body, summary=f"{len(matches)} file{'s' if len(matches) != 1 else ''}")


class Grep(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Returns matching lines as "
        "path:line:text. Uses ripgrep when available (respects .gitignore)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "File or directory to search (default: working directory)"},
            "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
        },
        "required": ["pattern"],
    }
    read_only = True

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("pattern", "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        root = _search_root(args, ctx)
        try:
            flags = re.IGNORECASE if args.get("ignore_case") else 0
            regex = re.compile(args["pattern"], flags)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from None

        rg = shutil.which("rg")
        lines = self._ripgrep(rg, args, root, ctx) if rg else self._python(regex, args, root, ctx)
        if not lines:
            return ToolOutput("no matches", summary="0 matches")
        body = "\n".join(lines[:MAX_RESULTS])
        if len(lines) > MAX_RESULTS:
            body += f"\n\n({len(lines) - MAX_RESULTS} more matches not shown; narrow the search)"
        return ToolOutput(truncate(body), summary=f"{len(lines)} match{'es' if len(lines) != 1 else ''}")

    def _ripgrep(self, rg: str, args: dict[str, Any], root: Path, ctx: ToolContext) -> list[str]:
        cmd = [rg, "--line-number", "--no-heading", "--color=never", "--max-columns",
               str(MAX_MATCH_CHARS), "--max-columns-preview"]
        if args.get("ignore_case"):
            cmd.append("--ignore-case")
        if args.get("glob"):
            cmd += ["--glob", args["glob"]]
        cmd += ["-e", args["pattern"], str(root)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode > 1:
            raise ToolError(proc.stderr.strip() or "ripgrep failed")
        prefix = f"{ctx.cwd}{os.sep}"
        return [l.removeprefix(prefix) for l in proc.stdout.splitlines()]

    def _python(self, regex: re.Pattern, args: dict[str, Any], root: Path, ctx: ToolContext) -> list[str]:
        pattern = args.get("glob")
        files = [root] if root.is_file() else self._walk(root)
        out: list[str] = []
        for path in files:
            if pattern and not (fnmatch.fnmatch(path.name, pattern)
                                or fnmatch.fnmatch(ctx.display_path(path), pattern)):
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    for n, line in enumerate(f, 1):
                        if regex.search(line):
                            out.append(f"{ctx.display_path(path)}:{n}:{line.rstrip()[:MAX_MATCH_CHARS]}")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable
            if len(out) > MAX_RESULTS * 5:
                break
        return out

    @staticmethod
    def _walk(root: Path):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                yield Path(dirpath, name)
