"""A tiny local server speaking the Anthropic streaming Messages API.

It replays scripted assistant turns, so the real SDK, the provider adapter
and the agent loop can be exercised end to end without network access.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def text_turn(text: str) -> dict:
    return {"blocks": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def tool_turn(name: str, input: dict, text: str = "") -> dict:
    blocks = [{"type": "text", "text": text}] if text else []
    blocks.append({"type": "tool_use", "name": name, "input": input})
    return {"blocks": blocks, "stop_reason": "tool_use"}


class FakeAnthropic:
    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                with server._lock:
                    server.requests.append({"headers": dict(self.headers), "body": body})
                    turn = server.turns.pop(0) if server.turns else text_turn("(script exhausted)")
                    n = len(server.requests)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for event in _events(turn, n):
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> FakeAnthropic:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _events(turn: dict, n: int):
    yield {"type": "message_start", "message": {
        "id": f"msg_{n}", "type": "message", "role": "assistant", "model": "fake", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 1}}}
    for i, block in enumerate(turn["blocks"]):
        if block["type"] == "text":
            yield {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": i,
                   "delta": {"type": "text_delta", "text": block["text"]}}
        else:
            yield {"type": "content_block_start", "index": i, "content_block": {
                "type": "tool_use", "id": f"toolu_{n}_{i}", "name": block["name"], "input": {}}}
            raw = json.dumps(block["input"])
            for j in range(0, len(raw), 20):  # stream the arguments in pieces
                yield {"type": "content_block_delta", "index": i,
                       "delta": {"type": "input_json_delta", "partial_json": raw[j:j + 20]}}
        yield {"type": "content_block_stop", "index": i}
    yield {"type": "message_delta", "delta": {"stop_reason": turn["stop_reason"], "stop_sequence": None},
           "usage": {"output_tokens": 50}}
    yield {"type": "message_stop"}
