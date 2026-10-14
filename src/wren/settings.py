"""UI preferences that wren writes itself (config.toml stays hand-edited)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from wren.config import CONFIG_DIR

SETTINGS_FILE = CONFIG_DIR / "settings.json"

Theme = Literal["auto", "dark", "light"]
UiMode = Literal["auto", "fullscreen", "inline"]


@dataclass
class Settings:
    theme: Theme = "auto"
    # Draw the whole terminal, or print into its scrollback; auto: full screen where it works.
    ui: UiMode = "auto"
    # Predict the next prompt after each answer (one extra, mostly cached, request).
    suggestions: bool = True
    # Review the conversation for memories at compaction and session end (it
    # takes a model request, so a moment at exit).
    memory_auto: bool = False

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> Settings:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path = SETTINGS_FILE) -> None:
        """Only what differs from the defaults, so a changed default reaches
        everyone who didn't choose otherwise."""
        default = asdict(type(self)())
        chosen = {k: v for k, v in asdict(self).items() if v != default[k]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(chosen, indent=2) + "\n")
