"""Append-only JSONL log of everything that happens in a session.

One JSON object per line, e.g.
  {"ts": ..., "kind": "message", "role": "assistant", "content": [...]}
The log is the source for debugging, resuming sessions and (later) evaluation;
`load_session` replays it to rebuild the conversation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from wren.checkpoint import Checkpoint
from wren.config import CONFIG_DIR
from wren.llm.types import Message, ThinkingBlock, Usage

SESSIONS_DIR = CONFIG_DIR / "sessions"


class SessionLog:
    def __init__(self, directory: Path | None = SESSIONS_DIR, path: Path | None = None):
        """Start a new log in `directory`, or append to an existing `path`."""
        self.path = path
        if path is not None:
            self.id = path.stem.rsplit("-", 1)[-1]
        else:
            self.id = uuid.uuid4().hex[:8]
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


@dataclass
class SessionState:
    path: Path
    cwd: str = ""
    model: str | None = None
    messages: list[Message] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float | None = None
    first_prompt: str = ""
    updated: float = 0.0


def load_session(path: Path) -> SessionState:
    state = SessionState(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a crash
        state.updated = e.get("ts", state.updated)
        match e["kind"]:
            case "session_start":
                state.cwd, state.model = e["cwd"], e["model"]
            case "message":
                msg = Message.from_dict(e)
                if state.messages and state.messages[-1].role == msg.role == "user":
                    state.messages[-1].content.extend(msg.content)
                else:
                    state.messages.append(msg)
                if msg.role == "user" and not state.first_prompt:
                    state.first_prompt = msg.text()
            case "usage":
                state.usage += Usage(**e["usage"])
                if e.get("cost") is not None:
                    state.cost = (state.cost or 0.0) + e["cost"]
            case "checkpoint":
                state.checkpoints.append(Checkpoint(e["commit"], e["message_index"], e["prompt"]))
            case "rewind":
                if e["message_index"] is not None:
                    state.messages = state.messages[: e["message_index"]]
                state.checkpoints = state.checkpoints[: e["checkpoint_index"]]
            case "compact":
                state.messages = [Message.from_dict(e["message"])]
                for c in state.checkpoints:
                    c.message_index = None
            case "clear":
                state.messages = []
                for c in state.checkpoints:
                    c.message_index = None
            case "model_switch":
                state.model = e["model"]
                for m in state.messages:
                    m.content = [b for b in m.content if not isinstance(b, ThinkingBlock)]
                state.messages = [m for m in state.messages if m.content]
    return state


def list_sessions(cwd: Path, directory: Path = SESSIONS_DIR, limit: int = 10) -> list[SessionState]:
    """Sessions started in `cwd` that have a conversation, newest first."""
    if not directory.is_dir():
        return []
    found: list[SessionState] = []
    for path in sorted(directory.glob("*.jsonl"), reverse=True):
        if not _started_in(path, cwd):
            continue
        state = load_session(path)
        if state.messages:
            found.append(state)
        if len(found) >= limit:
            break
    return found


def _started_in(path: Path, cwd: Path) -> bool:
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    try:
        return json.loads(first).get("cwd") == str(cwd)
    except json.JSONDecodeError:
        return False
