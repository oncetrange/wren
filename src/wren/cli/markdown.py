"""Render streamed markdown block by block.

Complete blocks (paragraphs, lists, fenced code) are printed as rendered
markdown; the block still being written is shown raw in a transient live
region below them, so output appears immediately and settles into its final
form once the block ends.
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

LIVE_TAIL_LINES = 12
_FENCES = ("```", "~~~")


class MarkdownStream:
    def __init__(self, console: Console, code_theme: str = "ansi_dark"):
        self.console = console
        self.code_theme = code_theme
        self.rendered = console.is_terminal
        self._partial = ""  # text after the last newline
        self._block: list[str] = []
        self._fence: str | None = None  # the open fence marker, if inside a code block
        self._printed_any = False
        self._live: Live | None = None

    def feed(self, text: str) -> None:
        if not self.rendered:
            self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
            return
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        for line in lines:
            self._line(line)
        self._show_partial()

    def close(self) -> None:
        if not self.rendered:
            self.console.print()
            return
        if self._partial:
            self._line(self._partial)
            self._partial = ""
        self._flush()
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _line(self, line: str) -> None:
        marker = line.strip()[:3]
        if self._fence is not None:
            self._block.append(line)
            if line.strip().startswith(self._fence) and line.strip().strip(self._fence[0]) == "":
                self._fence = None
                self._flush()
        elif marker in _FENCES:
            self._flush()
            self._fence = marker
            self._block.append(line)
        elif not line.strip():
            self._flush()
        else:
            self._block.append(line)

    def _flush(self) -> None:
        if not any(l.strip() for l in self._block):
            self._block = []
            return
        if self._live is not None:
            self._live.update(Text(""), refresh=True)
        if self._printed_any:
            self.console.print()
        self.console.print(Markdown("\n".join(self._block), code_theme=self.code_theme))
        self._printed_any = True
        self._block = []

    def _show_partial(self) -> None:
        lines = self._block + ([self._partial] if self._partial else [])
        if not lines:
            return
        tail = "\n".join(lines[-LIVE_TAIL_LINES:])
        if self._live is None:
            self._live = Live(Text(tail), console=self.console, transient=True,
                              auto_refresh=False, vertical_overflow="crop")
            self._live.start()
        self._live.update(Text(tail), refresh=True)
