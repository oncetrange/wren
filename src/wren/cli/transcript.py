"""The full-screen UI's conversation: blocks of rich renderables, laid out as
lines at the window's width (again whenever it changes), and the control that
shows them, scrolled, and lets the mouse select text in them.

Selected text is copied as it reads, not as it is laid out: where rich wrapped
a line to fit the width, the pieces are joined again. Which breaks are wraps is
found by laying the same block out at a width where nothing wraps and matching
the two (see soft_breaks); where they don't match (a panel's borders, say), the
text is copied as shown.

Blocks are added and changed from the session's thread while the UI thread
draws them, so both go through the transcript's lock.
"""

from __future__ import annotations

import io
import re
import threading
import time
from collections.abc import Callable

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text, split_lines
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from rich.console import Console, RenderableType

Line = StyleAndTextTuples
Position = tuple[int, int]  # (line, character) in the transcript as laid out
UNWRAPPED_WIDTH = 2000  # wide enough that nothing wraps (see soft_breaks)
DOUBLE_CLICK = 0.4  # seconds
WORD = re.compile(r"[\w./~:@%+=-]+")


def to_lines(renderables: list[RenderableType], width: int, color_system: str | None = "truecolor") -> list[Line]:
    """Render with rich at `width` and convert to prompt_toolkit lines."""
    console = Console(file=io.StringIO(), width=max(width, 10), force_terminal=True,
                      color_system=color_system, highlight=False)  # type: ignore[arg-type]
    for renderable in renderables:
        console.print(renderable)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    lines = list(split_lines(to_formatted_text(ANSI(text.removesuffix("\n")))))
    return lines if text else []


def plain(line: Line) -> str:
    return fragment_list_to_text(line).rstrip()


def plain_lines(renderables: list[RenderableType], width: int) -> list[str]:
    console = Console(file=io.StringIO(), width=width, color_system=None, highlight=False)
    for renderable in renderables:
        console.print(renderable)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    return [line.rstrip() for line in text.removesuffix("\n").split("\n")] if text else []


def soft_breaks(wrapped: list[str], unwrapped: list[str]) -> dict[int, str] | None:
    """Where line i of `wrapped` only continues on line i + 1 because it was
    wrapped, and what joins the two (a space, or nothing within a word): each
    line of `unwrapped` must be some run of `wrapped` lines. None if the two
    don't line up that way."""
    def norm(text: str) -> str:
        return " ".join(text.split())

    joins: dict[int, str] = {}
    i = 0
    for line in unwrapped:
        if i >= len(wrapped):
            return None
        target, done = norm(line), norm(wrapped[i])
        i += 1
        while done != target:
            if i >= len(wrapped) or not target.startswith(done):
                return None
            piece = norm(wrapped[i])
            joins[i - 1] = " " if target.startswith(f"{done} {piece}") else ""
            done = f"{done}{joins[i - 1]}{piece}"
            i += 1
    return joins if i == len(wrapped) else None


def restyle(line: Line, start: int, end: int, style: str) -> Line:
    """`line` with characters start..end (exclusive) in `style` as well."""
    if start >= end:
        return line
    chars = list(explode_text_fragments(line))
    for k in range(max(0, start), min(end, len(chars))):
        chars[k] = (f"{chars[k][0]} {style}", chars[k][1])
    return chars


def char_at(text: str, x: int) -> int:
    """The character in `text` under screen column `x` (wide characters take two)."""
    cells = 0
    for i, ch in enumerate(text):
        cells += get_cwidth(ch)
        if cells > x:
            return i
    return len(text)


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
        self._joins: tuple[int, bool, dict[int, str]] | None = None  # the same, for soft_breaks

    def lines(self, width: int, color_system: str | None, expanded: bool = False) -> list[Line]:
        expanded = expanded and self.expanded is not None
        if self._cache is None or self._cache[:2] != (width, expanded):
            shown = self.expanded if expanded and self.expanded is not None else self.renderables
            self._cache = (width, expanded, to_lines(shown, width, color_system))
        return self._cache[2]

    def joins(self, width: int, color_system: str | None, expanded: bool = False) -> dict[int, str]:
        """Which of its lines (at `width`) were wrapped, and what joins each to the next."""
        expanded = expanded and self.expanded is not None
        if self._joins is None or self._joins[:2] != (width, expanded):
            shown = self.expanded if expanded and self.expanded is not None else self.renderables
            wrapped = [plain(line) for line in self.lines(width, color_system, expanded)]
            found = soft_breaks(wrapped, plain_lines(shown, UNWRAPPED_WIDTH)) or {}
            self._joins = (width, expanded, found)
        return self._joins[2]


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
            block.renderables, block._cache, block._joins = list(renderables), None, None
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

    def text(self, width: int, start: Position, end: Position) -> str:
        """The text from `start` to `end` (exclusive) as laid out at `width`,
        with wrapped lines joined again."""
        (first, a), (last, b) = sorted((start, end))
        pieces: list[str] = []
        with self.lock:
            base = 0
            for block in self.blocks:
                lines = block.lines(width, self.color_system, self.expanded)
                if base + len(lines) <= first or base > last:
                    base += len(lines)
                    continue
                joins = block.joins(width, self.color_system, self.expanded)
                for k in range(max(0, first - base), min(len(lines), last - base + 1)):
                    n = base + k
                    text = plain(lines[k])
                    piece = text[a if n == first else 0:b if n == last else len(text)]
                    if pieces and pieces[-1] != "\n":
                        piece = piece.lstrip()  # a wrapped line's indentation
                    pieces.append(piece)
                    if n != last:
                        pieces.append(joins.get(k, "\n"))
                base += len(lines)
        return "".join(pieces)


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
        # Text selected with the mouse: the characters where the drag began and
        # where it is now, both selected.
        self.anchor: Position | None = None
        self.cursor: Position | None = None
        self.on_copy: Callable[[str], None] = lambda text: None
        self._dragging = False
        self._clicked = False  # pressed, and not dragged since: nothing is selected
        self._layout: tuple[int, bool] = (0, False)  # (width, expanded) the positions refer to
        self._top = 0
        self._visible: list[Line] = []
        self._last_click: tuple[float, Position, int] = (0.0, (-1, -1), 0)  # (when, where, clicks)

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self.transcript.lines(width)
        layout = (width, self.transcript.expanded)
        if layout != self._layout or len(lines) < self._total:  # positions no longer hold
            self.anchor = self.cursor = None
            self._dragging = False
        self._layout = layout
        if self.offset:  # scrolled back: stay on the same lines as more arrive below
            self.offset += max(0, len(lines) - self._total)
        self._height, self._total = height, len(lines)
        self.offset = min(self.offset, self.max_offset)
        bottom = len(lines) - self.offset
        top = max(0, bottom - height)
        visible = lines[top:bottom]
        self._top, self._visible = top, visible
        if self.term:
            visible = [highlight(line, self.term, "class:search.current" if top + i == self.current
                                 else "class:search") for i, line in enumerate(visible)]
        if (selected := self.selection) is not None:
            (first, a), (last, b) = selected
            visible = [restyle(line, a if top + i == first else 0, b if top + i == last else len(plain(line)),
                               "class:selection") if first <= top + i <= last else line
                       for i, line in enumerate(visible)]
        return UIContent(get_line=lambda i: visible[i], line_count=len(visible), show_cursor=False)

    @property
    def selection(self) -> tuple[Position, Position] | None:
        """The selected range, start to end (exclusive), if any: a press
        without a drag selects nothing."""
        if self.anchor is None or self.cursor is None or self._clicked:
            return None
        start, (line, index) = sorted((self.anchor, self.cursor))
        return start, (line, index + 1)

    def selected_text(self) -> str:
        selected = self.selection
        return self.transcript.text(self._layout[0], *selected) if selected else ""

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
        kind, (x, y) = mouse_event.event_type, mouse_event.position
        if kind == MouseEventType.SCROLL_UP:
            self.scroll(3)
        elif kind == MouseEventType.SCROLL_DOWN:
            self.scroll(-3)
        elif kind == MouseEventType.MOUSE_DOWN and mouse_event.button == MouseButton.LEFT:
            self._press(self._at(x, y))
        elif kind == MouseEventType.MOUSE_MOVE and self._dragging:
            if y >= self._height - 1:  # at the edges, the conversation scrolls along
                self.scroll(-1)
            elif y <= 0:
                self.scroll(1)
            self.cursor = self._at(x, y)
            self._clicked = self._clicked and self.cursor == self.anchor
        elif kind == MouseEventType.MOUSE_UP and self._dragging:
            self._dragging = False
            self._copy()
        else:
            return NotImplemented
        return None

    def _at(self, x: int, y: int) -> Position:
        """The character under screen cell (x, y) of the window (past the end
        of a line: its last one)."""
        top = max(0, self._total - self.offset - self._height)  # after any scrolling
        line = max(0, min(self._total - 1, top + y))
        lines = self.transcript.lines(self._layout[0]) if not top <= line < top + len(self._visible) else None
        text = plain(lines[line] if lines is not None else self._visible[line - top]) if self._total else ""
        return line, max(0, min(len(text) - 1, char_at(text, x)))

    def _press(self, at: Position) -> None:
        when, where, clicks = self._last_click
        now = time.monotonic()
        clicks = clicks + 1 if now - when < DOUBLE_CLICK and where[0] == at[0] and abs(where[1] - at[1]) <= 1 else 1
        self._last_click = (now, at, clicks)
        line, index = at
        self._clicked = clicks == 1
        text = plain(self._visible[line - self._top]) if 0 <= line - self._top < len(self._visible) else ""
        if clicks == 2:  # a word
            word = next((m for m in WORD.finditer(text) if m.start() <= index < m.end()), None)
            self.anchor, self.cursor = ((line, word.start()), (line, word.end() - 1)) if word else (at, at)
            self._copy()
        elif clicks >= 3:  # the whole line
            self.anchor, self.cursor = (line, 0), (line, max(0, len(text) - 1))
            self._copy()
        else:
            self.anchor = self.cursor = at
            self._dragging = True

    def _copy(self) -> None:
        if text := self.selected_text():
            self.on_copy(text)
