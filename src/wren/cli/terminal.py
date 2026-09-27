"""Terminal capabilities: background color detection and Shift+Enter."""

from __future__ import annotations

import atexit
import os
import re
import select
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

Background = Literal["dark", "light"]

# Shift+Enter (and Ctrl/Ctrl+Shift+Enter) as sent by terminals with CSI-u or
# modifyOtherKeys reporting on. We read them as Esc Enter, which inserts a newline.
_NEWLINE_SEQUENCES = ("\x1b[13;2u", "\x1b[13;5u", "\x1b[13;6u",
                      "\x1b[27;2;13~", "\x1b[27;5;13~", "\x1b[27;6;13~")
# Shift+Tab under modifyOtherKeys (WezTerm sends it instead of the usual CSI Z).
_BACKTAB_SEQUENCES = ("\x1b[27;2;9~",)
# Ctrl+Shift+letter and Ctrl(+Shift)+Tab under modifyOtherKeys: swallow them
# instead of typing the bytes into the input.
_IGNORED_SEQUENCES = (tuple(f"\x1b[27;6;{ord(c)}~" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                      + ("\x1b[27;5;9~", "\x1b[27;6;9~"))


# xterm modifyOtherKeys mode 1: modified keys without a well-known encoding
# (such as Shift+Enter) are reported as CSI 27;<mods>;<key>~, while Ctrl+C,
# Ctrl+D, Esc and Backspace keep their usual bytes. WezTerm and xterm honor it
# without any user configuration; other terminals ignore the request.
_MOK_ON, _MOK_OFF = "\x1b[>4;1m", "\x1b[>4;0m"


def register_shift_enter() -> None:
    for seq in _NEWLINE_SEQUENCES:
        ANSI_SEQUENCES[seq] = (Keys.Escape, Keys.ControlM)
    for seq in _BACKTAB_SEQUENCES:
        ANSI_SEQUENCES[seq] = Keys.BackTab
    for seq in _IGNORED_SEQUENCES:
        ANSI_SEQUENCES[seq] = Keys.Ignore


@contextmanager
def distinguish_shift_enter(stream=None) -> Iterator[None]:
    """Ask the terminal to report Shift+Enter distinctly while reading input.

    Only on while we wait for input, so programs the agent runs and the shell
    after we exit get the terminal in its normal state.
    """
    stream = stream or sys.stdout
    if not stream.isatty():
        yield
        return
    stream.write(_MOK_ON)
    stream.flush()
    atexit.register(_reset, stream)
    try:
        yield
    finally:
        _reset(stream)
        atexit.unregister(_reset)


def _reset(stream) -> None:
    try:
        stream.write(_MOK_OFF)
        stream.flush()
    except (OSError, ValueError):
        pass


def detect_background(timeout: float = 0.15) -> Background | None:
    """Ask the terminal for its background color (OSC 11); None if unknown."""
    if fgbg := os.environ.get("COLORFGBG"):
        # "15;0" means light-on-dark; the last field is the background palette index.
        bg = fgbg.split(";")[-1]
        if bg.isdigit():
            return "dark" if int(bg) in (0, 1, 2, 3, 4, 5, 6, 8) else "light"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import termios
        import tty
    except ImportError:
        return None

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    reply = b""
    try:
        tty.setcbreak(fd)
        os.write(sys.stdout.fileno(), b"\x1b]11;?\x1b\\")
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                break
            reply += os.read(fd, 64)
            if reply.endswith((b"\x07", b"\x1b\\")):
                break
    except OSError:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return parse_osc11(reply.decode(errors="replace"))


def parse_osc11(reply: str) -> Background | None:
    m = re.search(r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", reply)
    if not m:
        return None
    r, g, b = (int(c, 16) / (16 ** len(c) - 1) for c in m.groups())
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "light" if luminance > 0.5 else "dark"


def shift_enter_help() -> str:
    """How to make Shift+Enter insert a newline in the user's terminal."""
    term = os.environ.get("TERM_PROGRAM", "")
    snippets = {
        "WezTerm": "Shift+Enter works out of the box in WezTerm; no configuration needed.",
        "iTerm.app": (
            "iTerm2 → Settings → Profiles → Keys → Key Mappings → +\n"
            "  Shortcut: Shift+Enter · Action: Send Escape Sequence · Esc+: (a single carriage return, ^M)\n"
            "  or enable \"Report keys using CSI u\" under Profiles → Keys."
        ),
        "vscode": (
            "Add to VS Code keybindings.json:\n\n"
            '  { "key": "shift+enter", "command": "workbench.action.terminal.sendSequence",\n'
            '    "args": { "text": "\\u001b\\r" }, "when": "terminalFocus" }'
        ),
        "Apple_Terminal": (
            "Terminal.app can't remap Shift+Enter. Enable Settings → Profiles → Keyboard →\n"
            "\"Use Option as Meta key\" and use Option+Enter instead."
        ),
    }
    generic = ("Configure your terminal to send Esc followed by Enter (\\x1b\\r) for Shift+Enter,\n"
               "or turn on CSI-u key reporting. Esc Enter and Ctrl-J always work.")
    return snippets.get(term, generic)
