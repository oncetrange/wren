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

from wren.agent.conversation import Conversation
from wren.config import CONFIG_DIR
from wren.llm.types import Message, Usage

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
    conversation: Conversation = field(default_factory=Conversation)
    usage: Usage = field(default_factory=Usage)
    cost: float | None = None
    first_prompt: str = ""
    updated: float = 0.0

    @property
    def id(self) -> str:
        return self.path.stem.rsplit("-", 1)[-1]

    @property
    def messages(self) -> list[Message]:
        return self.conversation.messages


def load_session(path: Path) -> SessionState:
    """Rebuild a session by replaying its log through `Conversation`."""
    state = SessionState(path)
    conv = state.conversation
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
                if msg.role == "user":
                    conv.add_user(msg.content)
                    if not state.first_prompt:
                        state.first_prompt = msg.text()
                else:
                    conv.append(msg)
            case "usage":
                state.usage += Usage(**e["usage"])
                if e.get("cost") is not None:
                    state.cost = (state.cost or 0.0) + e["cost"]
            case "checkpoint":
                conv.start_turn(e["prompt"], e.get("commit"))
            case "mask":
                conv.mask(e["keep_turns"], e["min_chars"])
            case "compact":
                conv.compacted(e["summary"], Message.from_dict(e["message"]), e.get("kept_from"))
            case "rewind":
                conv.rewind(e["index"])
            case "clear":
                conv.clear()
            case "model_switch":
                state.model = e["model"]
                conv.strip_thinking()
    return state


def list_sessions(cwd: Path, directory: Path = SESSIONS_DIR, limit: int = 10) -> list[SessionState]:
    """Sessions started in `cwd` that have a conversation, most recently used first."""
    if not directory.is_dir():
        return []
    found: list[SessionState] = []
    paths = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in paths:
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
