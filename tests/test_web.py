"""web_search and web_fetch: backends, HTML to text, paging, permissions, config."""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import RecordingUI, ScriptedProvider, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.config import ConfigError, ModelConfig, load_config
from wren.integrations import pier_support
from wren.llm.types import Message, Response, ToolResultBlock, ToolUseBlock, Usage
from wren.tools.base import ToolError
from wren.tools.web import (
    MAX_CHARS,
    SearchConfig,
    WebFetch,
    WebSearch,
    html_to_text,
)
from wren.tools.web import (
    Response as HttpResponse,
)

PAGE = """<!doctype html><html><head><title> Widgets  API </title><style>.x{}</style></head>
<body><nav role="navigation"><a href="/">Home</a> | <a href="/docs">Docs</a></nav>
<header role="banner">Site header</header>
<main><h1>Widgets</h1><p>Make   a <a href="https://example.com/w">widget</a> with <code>make()</code>.</p>
<ul><li>fast</li><li>small</li></ul>
<table><tr><th>Name</th><th>Type</th></tr><tr><td><p>size</p></td><td>int</td></tr></table>
<pre>def make():
    return 1</pre><script>alert(1)</script>
<div aria-hidden="true">hidden</div></main><footer>© footer</footer></body></html>"""


def test_html_to_text():
    title, text = html_to_text(PAGE)
    assert title == "Widgets API"
    assert "# Widgets" in text and "Make a widget (https://example.com/w) with `make()`." in text
    assert "- fast\n- small" in text
    assert "| Name | Type |\n| size | int |" in text
    assert "```\ndef make():\n    return 1\n```" in text          # code keeps its indentation
    for chrome in ("Home", "Site header", "alert", "footer", "hidden", ".x{}"):
        assert chrome not in text


class Server:
    """Serves a few pages on localhost."""

    def __init__(self):
        pages = {"/page": ("text/html; charset=utf-8", PAGE.encode()),
                 "/long": ("text/plain", ("0123456789" * 3000).encode()),
                 "/gbk": ("text/plain; charset=gbk", "中文内容".encode("gbk")),
                 "/image": ("image/png", b"\x89PNG..."),
                 "/redirect": None}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/page")
                    self.end_headers()
                    return
                page = pages.get(self.path)
                if page is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-type", page[0])
                self.end_headers()
                self.wfile.write(page[1])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_fetch_pages(ctx):
    fetch = WebFetch()
    with Server() as s:
        page = fetch.run({"url": s.url + "/redirect"}, ctx)
        assert page.content.startswith(f"Widgets API\n{s.url}/page\n") and "# Widgets" in page.content
        first = fetch.run({"url": s.url + "/long"}, ctx).content
        assert f"[characters 0-{MAX_CHARS} of 30000; continue with start={MAX_CHARS}]" in first
        rest = fetch.run({"url": s.url + "/long", "start": MAX_CHARS}, ctx).content
        assert f"[characters {MAX_CHARS}-30000 of 30000]" in rest
        assert "中文内容" in fetch.run({"url": s.url + "/gbk"}, ctx).content
        with pytest.raises(ToolError, match="image/png, not text"):
            fetch.run({"url": s.url + "/image"}, ctx)
        with pytest.raises(ToolError, match="HTTP 404"):
            fetch.run({"url": s.url + "/missing"}, ctx)
    with pytest.raises(ToolError, match="only http"):
        fetch.run({"url": "file:///etc/passwd"}, ctx)
    with pytest.raises(ToolError, match="couldn't reach"):
        fetch.run({"url": "http://127.0.0.1:1/"}, ctx)


# --- search backends ---------------------------------------------------------------

def fake_fetch(status, body, seen):
    def fetch(request):
        seen.append(request)
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        return HttpResponse(request.full_url, status, "application/json", payload)
    return fetch


def test_brave(ctx, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "bk")
    seen = []
    body = {"web": {"results": [{"title": "<strong>Widgets</strong>", "url": "https://w.dev",
                                 "description": "All about &amp; widgets"}]}}
    out = WebSearch(SearchConfig(), fake_fetch(200, body, seen)).run({"query": "widgets", "count": 3}, ctx)
    assert out.summary == "1 results (brave)"
    assert out.content == "1. Widgets\n   https://w.dev\n   All about & widgets"
    assert seen[0].get_header("X-subscription-token") == "bk"
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(seen[0].full_url).query) == {"q": ["widgets"], "count": ["3"]}


def test_tavily_and_errors(ctx, monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tk")
    seen = []
    body = {"results": [{"title": "T", "url": "https://t.dev", "content": "snippet"}]}
    out = WebSearch(SearchConfig(), fake_fetch(200, body, seen)).run({"query": "q"}, ctx)
    assert out.summary == "1 results (tavily)" and json.loads(seen[0].data)["query"] == "q"
    with pytest.raises(ToolError, match="Tavily search failed: HTTP 401"):
        WebSearch(SearchConfig(), fake_fetch(401, {"error": "bad key"}, [])).run({"query": "q"}, ctx)
    monkeypatch.delenv("TAVILY_API_KEY")
    with pytest.raises(ToolError, match=r"needs an API key: set \$TAVILY_API_KEY"):
        WebSearch(SearchConfig("tavily"), fake_fetch(200, body, [])).run({"query": "q"}, ctx)


def test_duckduckgo_without_keys(ctx, monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    page = b"""<div><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.dev%2Fx&amp;rut=1">
    First <b>result</b></a><a class="result__snippet" href="#">About <b>a</b></a></div>
    <div><a class="result__a" href="https://b.dev/">Second</a></div>"""
    out = WebSearch(SearchConfig(), fake_fetch(200, page, [])).run({"query": "q"}, ctx)
    assert out.content == "1. First result\n   https://a.dev/x\n   About a\n\n2. Second\n   https://b.dev/"
    with pytest.raises(ToolError, match="DuckDuckGo answered HTTP 403"):
        WebSearch(SearchConfig(), fake_fetch(403, b"", [])).run({"query": "q"}, ctx)


# --- agent, config, benchmarks -----------------------------------------------------

def test_fetch_asks_per_domain(ctx):
    def use(call_id, url):
        return Response(Message("assistant", [ToolUseBlock(call_id, "web_fetch", {"url": url})]), "tool_use",
                        Usage(10, 5))

    with Server() as s:
        turns = [use("a", s.url + "/page"), use("b", s.url + "/long"), reply("done")]
        ui = RecordingUI([Decision(allow=True, remember=True)])
        agent = Agent(ScriptedProvider(turns), ModelConfig(name="f", model="f"), ctx, ui,
                      Permissions(mode="accept_edits"))
        agent.tools["web_fetch"] = WebFetch()
        agent.run("read the docs")
    assert [e[1] for e in ui.events if e[0] == "confirm"] == ["web_fetch"]   # the second: same host
    results = [b for m in agent.messages for b in m.content if isinstance(b, ToolResultBlock)]
    assert not results[0].is_error and not results[1].is_error


def test_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[web]\nsearch = "brave"\napi_key_env = "MY_BRAVE"\n')
    web = load_config(path).web
    assert (web.enabled, web.search, web.api_key_env) == (True, "brave", "MY_BRAVE")
    path.write_text('[web]\nsearch = "google"\n')
    with pytest.raises(ConfigError, match="search must be"):
        load_config(path)


def test_benchmark_runs_have_no_web():
    model = ModelConfig(name="m", model="m")
    toml = pier_support.model_config_toml(model)
    assert "[web]\nenabled = false" in toml
    import tomllib
    assert tomllib.loads(toml)["web"] == {"enabled": False}


def test_only_writing_subagents_get_web_tools(ctx):
    from wren.agent.subagents import builtin_agent_types

    class ToolRecording(ScriptedProvider):
        def stream(self, *, system, messages, tools):
            self.seen = [*getattr(self, "seen", []), {t.name for t in tools}]
            yield from super().stream(system=system, messages=messages, tools=tools)

    def task(kind, call_id):
        return Response(Message("assistant", [ToolUseBlock(call_id, "task", {
            "description": "d", "prompt": f"p {kind}", "agent": kind})]), "tool_use", Usage(10, 5))

    agent = Agent(ToolRecording([task("general", "t1"), reply("r"), task("explore", "t2"), reply("r"),
                                 reply("done")]), ModelConfig(name="f", model="f"), ctx, RecordingUI(),
                  Permissions(mode="auto"), agent_types=builtin_agent_types())
    agent.tools.update({"web_search": WebSearch(), "web_fetch": WebFetch()})
    agent.run("go")
    general, explore = agent.provider.seen[1], agent.provider.seen[3]
    assert {"web_search", "web_fetch"} <= general and not {"web_search", "web_fetch"} & explore
