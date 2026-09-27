"""Key bindings shared by every text input."""

from __future__ import annotations

from prompt_toolkit.key_binding import KeyBindings


def newline_bindings() -> KeyBindings:
    """Esc Enter and Ctrl-J (and Shift+Enter where the terminal reports it,
    see terminal.register_shift_enter) insert a newline instead of submitting."""
    kb = KeyBindings()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    return kb
