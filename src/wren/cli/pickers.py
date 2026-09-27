"""Interactive prompts: arrow-key selection and line-edited text input."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import choice

from wren.cli.keys import newline_bindings

T = TypeVar("T")


def pick(message: str, options: Sequence[tuple[T, str]], default: T | None = None) -> T | None:
    """Let the user choose with ↑/↓ and Enter. None if cancelled (Ctrl-C / Esc)."""
    if not options:
        return None
    try:
        return choice(message=message, options=list(options), default=default,
                      key_bindings=_escape_cancels())
    except (KeyboardInterrupt, EOFError):
        return None


def ask_text(message: str = "  › ") -> str:
    """Read text with the same editing as the main input: cursor keys, Esc
    Enter / Ctrl-J / Shift+Enter for newlines. Ctrl-C propagates."""
    return prompt(HTML(f"<b>{message}</b>"), key_bindings=newline_bindings())


def confirm(message: str, default: bool = False) -> bool:
    return bool(pick(message, [(True, "Yes"), (False, "No")], default=default))


def _escape_cancels():
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event) -> None:
        event.app.exit(exception=KeyboardInterrupt())

    return kb
