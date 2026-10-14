"""Where slash commands and the session's own messages go, and how they ask
the user something: printing, pickers, a line of text, a status line, a pager,
a program that takes over the terminal (an editor).

`Screen` is the inline form: output goes to the terminal's own scrollback and
questions are asked under it. Commands only use these methods, so another
form of the session (a full-screen one) can show the same things its own way.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any, Protocol

from rich.console import Console, RenderableType

from wren.cli import pickers
from wren.cli.terminal import distinguish_shift_enter
from wren.cli.typeahead import TypeAhead


class Printer(Protocol):
    """Anything with rich's `print`: a Console, or a Screen."""

    def print(self, *objects: Any, **kwargs: Any) -> None: ...


class Screen:
    def __init__(self, console: Console, interactive: bool):
        self.console = console
        self.interactive = interactive
        # While a prompt runs, keys go to the agent (see RichUI.typing_ahead);
        # anything that asks the user takes the keyboard back meanwhile.
        self.typeahead: TypeAhead | None = None

    @property
    def width(self) -> int:
        return self.console.width

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self.console.print(*objects, **kwargs)

    def clear(self) -> None:
        self.console.clear()

    # --- asking ----------------------------------------------------------------

    def pick[T](self, message: str, options: Sequence[tuple[T, str]], default: T | None = None) -> T | None:
        """One of the options, chosen with ↑/↓ and Enter; None if cancelled."""
        with self._keyboard():
            return pickers.pick(message, options, default=default)

    def confirm(self, message: str, default: bool = False) -> bool:
        return bool(self.pick(message, [(True, "Yes"), (False, "No")], default=default))

    def ask_text(self, question: str | None = None) -> str:
        """A line (or several) of text, under `question` (rich markup) if given.
        Ctrl-C propagates; EOFError when there's no input."""
        if question:
            self.console.print(question)
        with self._keyboard(), distinguish_shift_enter():
            return pickers.ask_text()

    def _keyboard(self) -> contextlib.AbstractContextManager[None]:
        return self.typeahead.paused() if self.typeahead else contextlib.nullcontext()

    # --- longer things -----------------------------------------------------------

    def status(self, message: str) -> contextlib.AbstractContextManager[Any]:
        """A spinner with a message while something slow runs; `.update(message)` changes it."""
        return self.console.status(message)

    def page(self, *renderables: RenderableType) -> None:
        """Something long, through a pager the way git does it (`$PAGER` or less,
        which prints it as is when it fits the screen)."""
        pager = os.environ.get("PAGER") or ("less" if shutil.which("less") else "")
        if not (self.interactive and self.console.is_terminal and pager):
            self.console.print(*renderables)
            return
        with self.console.capture() as captured:
            self.console.print(*renderables)
        env = {**os.environ, "LESS": os.environ.get("LESS", "FRX")}
        with contextlib.suppress(OSError):
            subprocess.run(pager, shell=True, input=captured.get(), text=True, env=env, check=False)

    def write_terminal(self, sequence: str) -> None:
        """Send the terminal a control sequence (e.g. OSC 52, to set the clipboard)."""
        if self.console.is_terminal:
            self.console.file.write(sequence)
            self.console.file.flush()

    def run_program(self, argv: list[str]) -> int:
        """Hand the terminal to a program (an editor) until it exits."""
        with self._keyboard():
            return subprocess.run(argv, check=False).returncode
