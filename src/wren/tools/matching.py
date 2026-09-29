"""Whitespace-tolerant matching for edit_file.

Models often get indentation depth wrong when they copy code into old_string,
especially in tab-indented files: one tab too many or too few. When the exact
string isn't found, `find_loose` looks for a block whose lines match ignoring
leading and trailing whitespace, and works out the indentation shift so
new_string can be re-indented to fit the file.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class LooseMatch:
    start: int  # first matched line (0-based)
    end: int    # one past the last matched line
    old_prefix: str  # indentation the model wrote, beyond what the file has...
    new_prefix: str  # ...and what the file has in its place

    def reindent(self, text: str) -> str:
        """Apply the same indentation shift to `text` (the model's new_string)."""
        out = []
        for line in text.split("\n"):
            if line.strip() and line.startswith(self.old_prefix):
                line = self.new_prefix + line[len(self.old_prefix):]
            out.append(line)
        return "\n".join(out)


def split_block(text: str) -> list[str]:
    """Lines of a block, without the empty item a trailing newline produces."""
    lines = text.split("\n")
    return lines[:-1] if len(lines) > 1 and lines[-1] == "" else lines


def find_loose(file_lines: list[str], old: str) -> list[LooseMatch]:
    """All places where old's lines match file lines ignoring surrounding
    whitespace, with a consistent indentation shift on every line."""
    old_lines = split_block(old)
    if not any(l.strip() for l in old_lines):
        return []
    keys = [l.strip() for l in old_lines]
    first = next(i for i, k in enumerate(keys) if k)
    matches = []
    for start in range(len(file_lines) - len(old_lines) + 1):
        if file_lines[start + first].strip() != keys[first]:
            continue
        window = file_lines[start:start + len(old_lines)]
        if all(f.strip() == k for f, k in zip(window, keys, strict=False)):
            shift = _shift(window, old_lines)
            if shift is not None:
                matches.append(LooseMatch(start, start + len(old_lines), *shift))
    return matches


def _shift(file_block: list[str], old_block: list[str]) -> tuple[str, str] | None:
    """The (old_prefix, new_prefix) swap that turns every old line's
    indentation into the file line's, or None if there is no single one."""
    pairs = [(_indent(o), _indent(f)) for f, o in zip(file_block, old_block, strict=False) if f.strip()]
    old_ws, file_ws = pairs[0]
    common = _common_suffix(old_ws, file_ws)
    old_prefix, new_prefix = old_ws[: len(old_ws) - common], file_ws[: len(file_ws) - common]
    for o, f in pairs:
        if not o.startswith(old_prefix) or new_prefix + o[len(old_prefix):] != f:
            return None
    return old_prefix, new_prefix


def closest_region(file_lines: list[str], old: str,
                   context: int = 2) -> tuple[int, int, list[str]] | None:
    """The file region most similar to `old`, for error messages:
    (best matching line number, first shown line number, lines)."""
    old_lines = split_block(old)
    anchor = next((l.strip() for l in old_lines if l.strip()), "")
    if not anchor or not file_lines:
        return None
    best, best_ratio = 0, 0.0
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(anchor)
    for i, line in enumerate(file_lines):
        stripped = line.strip()
        if not stripped:
            continue
        matcher.set_seq1(stripped)
        if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best, best_ratio = i, ratio
    if best_ratio < 0.6:
        return None
    start = max(0, best - context)
    end = min(len(file_lines), best + len(old_lines) + context)
    return best + 1, start + 1, file_lines[start:end]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _common_suffix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n
