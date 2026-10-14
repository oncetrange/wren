"""Copying text to the user's clipboard: the system's own when wren runs on
the user's machine, otherwise through the terminal (OSC 52), which also works
over SSH in the terminals that allow it."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from collections.abc import Callable


def copy(text: str, write_terminal: Callable[[str], None]) -> str:
    """Copy `text`; returns where it went."""
    if not os.environ.get("SSH_CONNECTION") and (command := _system_command()):
        try:
            subprocess.run(command, input=text.encode(), check=True, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "the clipboard"
        except (OSError, subprocess.SubprocessError):
            pass
    write_terminal(osc52(text))
    return "the terminal's clipboard (OSC 52)"


def osc52(text: str) -> str:
    sequence = f"\x1b]52;c;{base64.b64encode(text.encode()).decode()}\x07"
    if os.environ.get("TMUX"):  # passed through to the terminal tmux runs in
        sequence = "\x1bPtmux;" + sequence.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    return sequence


def _system_command() -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("pbcopy"):
        return ["pbcopy"]
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return ["wl-copy"]
    if os.environ.get("DISPLAY"):
        if shutil.which("xclip"):
            return ["xclip", "-selection", "clipboard"]
        if shutil.which("xsel"):
            return ["xsel", "--clipboard", "--input"]
    return None
