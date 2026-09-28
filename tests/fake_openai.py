"""A tiny local server speaking OpenAI's streaming Chat Completions API.

Like fake_anthropic.py: it replays scripted turns so the real SDK, the adapter
and the agent loop run end to end without network access. A turn is a dict:

  {"text": "...", "reasoning": "...", "calls": [(name, input_dict_or_raw_str)],
   "finish": "stop" | "tool_calls" | "length" | ..., "usage": {...}}
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def text_turn(text: str, **kw) -> dict:
    return {"text": text, "finish": "stop", **kw}


def tool_turn(*calls: tuple[str, Any], text: str = "", **kw) -> dict:
    return {"text": text, "calls": list(calls), "finish": "tool_calls", **kw}


class FakeOpenAI:
    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                with server._lock:
                    server.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    turn = server.turns.pop(0) if server.turns else text_turn("(script exhausted)")
                    n = len(server.requests)
                if "status" in turn:
                    payload = json.dumps({"error": {"message": turn["error"], "type": "invalid_request_error"}})
                    self.send_response(turn["status"])
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload.encode())
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for chunk in _chunks(turn, n):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self) -> FakeOpenAI:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _chunk(n: int, delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> dict:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    chunk = {"id": f"chatcmpl-{n}", "object": "chat.completion.chunk", "created": 0,
             "model": "fake", "choices": choices}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _pieces(text: str, size: int = 7):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _chunks(turn: dict, n: int):
    yield _chunk(n, {"role": "assistant", "content": ""})
    for piece in _pieces(turn.get("reasoning", "")):
        yield _chunk(n, {"reasoning_content": piece})
    for piece in _pieces(turn.get("text", "")):
        yield _chunk(n, {"content": piece})
    for i, (name, args) in enumerate(turn.get("calls", [])):
        raw = args if isinstance(args, str) else json.dumps(args)
        yield _chunk(n, {"tool_calls": [{"index": i, "id": f"call_{n}_{i}", "type": "function",
                                         "function": {"name": name, "arguments": ""}}]})
        for piece in _pieces(raw, 10):  # arguments stream in pieces, keyed by index only
            yield _chunk(n, {"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
    yield _chunk(n, {}, finish=turn["finish"])
    usage = turn.get("usage", {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050})
    yield _chunk(n, None, usage=usage)
