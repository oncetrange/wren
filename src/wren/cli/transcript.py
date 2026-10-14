"""The full-screen UI's conversation: blocks of rich renderables, laid out as
lines at the window's width (again whenever it changes), and the control that
shows them, scrolled.

Blocks are added and changed from the session's thread while the UI thread
draws them, so both go through the transcript's lock.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text, split_lines
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from rich.console import Console, RenderableType

Line = StyleAndTextTuples


def to_lines(renderables: list[RenderableType], width: int, color_system: str | None = "truecolor") -> list[Line]:
    """Render with rich at `width` and convert to prompt_toolkit lines."""
    console = Console(file=io.StringIO(), width=max(width, 10), force_terminal=True,
                      color_system=color_system, highlight=False)  # type: ignore[arg-type]
    for renderable in renderables:
        console.print(renderable)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    lines = list(split_lines(to_formatted_text(ANSI(text.removesuffix("\n")))))
    return lines if text else []


def highlight(line: Line, term: str, style: str) -> Line:
    """`line` with each occurrence of `term` (any case) in `style` as well."""
    low = fragment_list_to_text(line).lower()
    spans, i = [], low.find(term)
    while term and i >= 0:
        spans.append((i, i + len(term)))
        i = low.find(term, i + len(term))
    if not spans:
        return line
    chars = list(explode_text_fragments(line))
    for start, end in spans:
        for k in range(start, end):
            chars[k] = (f"{chars[k][0]} {style}", chars[k][1])
    return chars


class Block:
    """Renderables shown together, and optionally a longer form of them, shown
    instead while the transcript is expanded."""

    def __init__(self, renderables: list[RenderableType], expanded: list[RenderableType] | None = None):
        self.renderables = renderables
        self.expanded = expanded
        self._cache: tuple[int, bool, list[Line]] | None = None  # (width, expanded, lines)

    def lines(self, width: int, color_system: str | None, expanded: bool = False) -> list[Line]:
        expanded = expanded and self.expanded is not None
        if self._cache is None or self._cache[:2] != (width, expanded):
            shown = self.expanded if expanded and self.expanded is not None else self.renderables
            self._cache = (width, expanded, to_lines(shown, width, color_system))
        return self._cache[2]


class Transcript:
    def __init__(self, color_system: str | None = "truecolor"):
        self.color_system = color_system
        self.lock = threading.RLock()
        self.blocks: list[Block] = []
        self.expanded = False  # blocks show their longer form (tool output, whole diffs)
        # Called after every change (from any thread): the UI redraws.
        self.changed: Callable[[], None] = lambda: None

    def add(self, *renderables: RenderableType, expanded: list[RenderableType] | None = None) -> Block:
        block = Block(list(renderables), expanded)
        with self.lock:
            self.blocks.append(block)
        self.changed()
        return block

    def update(self, block: Block, *renderables: RenderableType) -> None:
        """Replace what a block shows (e.g. text still streaming in)."""
        with self.lock:
            block.renderables, block._cache = list(renderables), None
        self.changed()

    def clear(self) -> None:
        with self.lock:
            self.blocks = []
        self.changed()

    def lines(self, width: int) -> list[Line]:
        with self.lock:
            out: list[Line] = []
            for block in self.blocks:
                out += block.lines(width, self.color_system, self.expanded)
            return out

    def toggle(self) -> None:
        """Between the short and the long form of the blocks that have both."""
        with self.lock:
            self.expanded = not self.expanded
        self.changed()

    def renderables(self) -> list[RenderableType]:
        with self.lock:
            return [r for block in self.blocks for r in block.renderables]


class TranscriptControl(UIControl):
    """The transcript's last lines, or earlier ones once scrolled back (the
    mouse wheel, PgUp/PgDn). It follows new output while at the bottom."""

    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self.offset = 0  # lines scrolled back from the bottom
        self._height = 1
        self._total = 0
        # While searching: what is looked for, and the line of the current match.
        self.term = ""
        self.current: int | None = None

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self.transcript.lines(width)
        if self.offset:  # scrolled back: stay on the same lines as more arrive below
            self.offset += max(0, len(lines) - self._total)
        self._height, self._total = height, len(lines)
        self.offset = min(self.offset, self.max_offset)
        bottom = len(lines) - self.offset
        top = max(0, bottom - height)
        visible = lines[top:bottom]
        if self.term:
            visible = [highlight(line, self.term, "class:search.current" if top + i == self.current
                                 else "class:search") for i, line in enumerate(visible)]
        return UIContent(get_line=lambda i: visible[i], line_count=len(visible), show_cursor=False)

    def show_line(self, index: int) -> None:
        """Scroll so that line `index` is in the middle of the window."""
        self.offset = max(0, min(self.max_offset, self._total - index - 1 - self._height // 2))

    @property
    def max_offset(self) -> int:
        return max(0, self._total - self._height)

    @property
    def below(self) -> int:
        """Lines below the window, when scrolled back."""
        return self.offset

    @property
    def position(self) -> tuple[int, int, int]:
        """The first and last line shown (from 1) and how many there are."""
        bottom = self._total - self.offset
        return max(0, bottom - self._height) + 1, bottom, self._total

    def to_top(self) -> None:
        self.offset = 1 << 30  # clamped once laid out

    def scroll(self, lines: int) -> None:
        """Back (positive) or forward (negative)."""
        self.offset = max(0, min(self.max_offset, self.offset + lines))

    def page(self, direction: int) -> None:
        self.scroll(direction * max(1, self._height - 2))

    def to_bottom(self) -> None:
        self.offset = 0

    def is_focusable(self) -> bool:
        return False

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(-3)
            return None
        return NotImplemented
