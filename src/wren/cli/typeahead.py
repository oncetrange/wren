"""Typing while the agent works: keys are read in the background, the line being
typed shows under the spinner, and Enter hands it to the agent (Agent.steer),
which passes it to the model after the current tool calls.

The terminal is in cbreak mode meanwhile (no echo, no line editing, but Ctrl-C
still interrupts). Anything else that reads the keyboard, like an approval
prompt, runs inside `paused()`.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import select
import sys
import termios
import threading
from collections.abc import Callable, Iterator


class TypeAhead:
    def __init__(self, submit: Callable[[str], None], fd: int | None = None):
        self.submit = submit
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.draft = ""
        self._saved: list | None = None
        # Held while reading, so pausing never loses a key; only for the read itself,
        # since a lock taken again at once can keep paused() waiting indefinitely.
        self._lock = threading.Lock()
        self._paused = False
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        # Inside an escape sequence (arrow keys and the like), which is dropped:
        # "esc" right after ESC, "seq" in its parameters until the final byte.
        self._escape: str | None = None

    def __enter__(self) -> TypeAhead:
        self._saved = termios.tcgetattr(self.fd)
        self._cbreak()
        self._thread = threading.Thread(target=self._read_keys, name="typeahead", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stopped.set()
        with self._lock:
            self._restore()
        if self._thread is not None:
            self._thread.join(timeout=1)

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Give the keyboard back, in its normal mode, to whatever reads it now."""
        with self._lock:
            self._paused = True
            self._restore()
        try:
            yield
        finally:
            with self._lock:
                self._paused = False
                if not self._stopped.is_set():
                    self._cbreak()

    def feed(self, text: str) -> None:
        """Edit the draft with what was typed (or pasted)."""
        for i, ch in enumerate(text):
            if self._escape == "esc":
                self._escape = "seq" if ch in "[O" else None
                continue
            if self._escape == "seq":
                if "@" <= ch <= "~":
                    self._escape = None
                continue
            if ch == "\x1b":
                self._escape = "esc"
            elif ch in "\r\n":
                # A newline with more text right behind it is part of a paste.
                if text[i + 1:].strip():
                    self.draft += "\n"
                else:
                    self._send()
            elif ch in "\x7f\x08":
                self.draft = self.draft[:-1]
            elif ch == "\x15":  # Ctrl-U
                self.draft = ""
            elif ch == "\x17":  # Ctrl-W
                self.draft = self.draft.rstrip()
                self.draft = self.draft[:max(self.draft.rfind(" "), self.draft.rfind("\n")) + 1]
            elif ch == "\t" or ch >= " ":
                self.draft += ch

    def _send(self) -> None:
        text, self.draft = self.draft.strip(), ""
        if text:
            self.submit(text)

    def _read_keys(self) -> None:
        while not self._stopped.is_set():
            if self._paused:
                self._stopped.wait(0.05)
                continue
            if not select.select([self.fd], [], [], 0.05)[0]:
                continue
            with self._lock:  # paused meanwhile, or its reader took the keys: leave them
                ready = not self._paused and select.select([self.fd], [], [], 0)[0]
                data = os.read(self.fd, 1024) if ready else b""
            if data:
                self.feed(self._decoder.decode(data))

    def _cbreak(self) -> None:
        if self._saved is None:
            return
        mode = termios.tcgetattr(self.fd)
        mode[3] &= ~(termios.ECHO | termios.ICANON)
        mode[6][termios.VMIN], mode[6][termios.VTIME] = 1, 0
        termios.tcsetattr(self.fd, termios.TCSANOW, mode)

    def _restore(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._saved)
