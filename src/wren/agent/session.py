"""Append-only JSONL log of everything that happens in a session.

One JSON object per line, e.g.
  {"ts": ..., "kind": "message", "role": "assistant", "content": [...]}
The log is the source for debugging, replay and (later) evaluation.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from wren.config import CONFIG_DIR

SESSIONS_DIR = CONFIG_DIR / "sessions"


class SessionLog:
    def __init__(self, directory: Path | None = SESSIONS_DIR):
        self.id = uuid.uuid4().hex[:8]
        self.path: Path | None = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = directory / f"{stamp}-{self.id}.jsonl"

    def record(self, kind: str, **data: Any) -> None:
        if self.path is None:
            return
        entry = {"ts": round(time.time(), 3), "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
