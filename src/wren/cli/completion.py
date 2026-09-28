"""Slash-command completion without changing the layout.

Terminals can't draw over earlier output and give it back, and growing the
prompt area near the bottom of the screen scrolls everything up for good. So
completion uses the space that is always there:

- the bottom toolbar line lists the matching commands and skills while the
  input is a bare "/prefix" (the selected one with its description), and
- the selected command is shown as grey ghost text after the cursor.

↑/↓ change the selection, Tab completes it (→ also accepts the ghost text),
Enter runs it. Matching and selection live in `SlashMenu`; only `toolbar` and
the ghost text are about how it is shown, so a full-screen UI could reuse the rest.
"""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings

STYLES = {"menu.selected": "reverse", "menu.meta": "ansigray", "menu.hint": "ansigray"}
HINT = "  ↑↓ Tab ↵"


def matches(text: str, entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Entries completing `text`, if it is a bare `/prefix` (no spaces yet)."""
    if not text.startswith("/") or any(c.isspace() for c in text):
        return []
    return [(name, desc) for name, desc in entries if name.startswith(text)]


class SlashMenu:
    def __init__(self, entries: Callable[[], list[tuple[str, str]]]):
        self.entries = entries
        self.index = 0
        self._last_text = ""

    def items(self, text: str | None = None) -> list[tuple[str, str]]:
        if text is None:
            text = get_app().current_buffer.text
        if text != self._last_text:  # typing resets the selection
            self._last_text, self.index = text, 0
        return matches(text, self.entries())

    def selected(self, text: str | None = None) -> str | None:
        items = self.items(text)
        return items[self.index % len(items)][0] if items else None

    def attach(self, session: PromptSession) -> None:
        """Show the ghost text as soon as the input changes (prompt_toolkit's
        own auto-suggest runs asynchronously, a beat later)."""
        session.default_buffer.on_text_changed += self._update_ghost

    def _update_ghost(self, buf: Buffer) -> None:
        name = self.selected(buf.text)
        buf.suggestion = Suggestion(name[len(buf.text):]) if name and len(name) > len(buf.text) else None

    @property
    def is_open(self) -> Condition:
        return Condition(lambda: bool(self.items()))

    # --- display -----------------------------------------------------------

    def toolbar(self, width: int | None = None, text: str | None = None) -> StyleAndTextTuples | None:
        """One line: the matches, scrolled so the selected one (with its
        description) is visible. None when the menu is closed."""
        items = self.items(text)
        if not items:
            return None
        width = width or get_app().output.get_size().columns
        sel = self.index % len(items)
        room = width - 2 - len(HINT)

        def cell(i: int) -> StyleAndTextTuples:
            name, desc = items[i]
            if i == sel:
                return [("class:menu.selected", f" {name} "), ("class:menu.meta", f" {desc}  ")]
            return [("", f" {name}  ")]

        def size(i: int) -> int:
            return sum(len(t) for _, t in cell(i))

        start = 0  # scroll until the selected item fits
        while start < sel and sum(size(i) for i in range(start, sel + 1)) > room:
            start += 1
        line: StyleAndTextTuples = [("", "  ")]
        used = 0
        for i in range(start, len(items)):
            if used + size(i) > room:
                if i == sel:  # the selected item alone is too long: cut its description
                    name_part, desc_part = cell(i)
                    left = max(0, room - used - len(name_part[1]) - 1)
                    line += [name_part, ("class:menu.meta", desc_part[1][:left] + "…")]
                else:
                    line.append(("class:menu.hint", "…"))
                break
            line += cell(i)
            used += size(i)
        line.append(("class:menu.hint", HINT))
        return line

    # --- keys ----------------------------------------------------------------

    def bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_open = self.is_open

        @kb.add("down", filter=is_open)
        @kb.add("c-n", filter=is_open)
        def _(event) -> None:
            self.index += 1
            self._update_ghost(event.current_buffer)

        @kb.add("up", filter=is_open)
        @kb.add("c-p", filter=is_open)
        def _(event) -> None:
            self.index -= 1
            self._update_ghost(event.current_buffer)

        @kb.add("tab", filter=is_open)
        def _(event) -> None:
            buf = event.current_buffer
            buf.text = self.selected(buf.text) + " "
            buf.cursor_position = len(buf.text)

        @kb.add("enter", filter=is_open)
        def _(event) -> None:
            buf = event.current_buffer
            buf.text = self.selected(buf.text)
            buf.validate_and_handle()

        return kb
