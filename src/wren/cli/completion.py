"""Completion for slash commands and skills in the input box."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class SlashCompleter(Completer):
    """Completes `/name` at the start of the input from (name, description) pairs."""

    def __init__(self, entries: Callable[[], list[tuple[str, str]]]):
        self.entries = entries

    def get_completions(self, document: Document, event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or any(c.isspace() for c in text):
            return
        for name, description in self.entries():
            if name.startswith(text):
                yield Completion(name, start_position=-len(text), display_meta=description)
