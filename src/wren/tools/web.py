"""Web access: search the web and read pages.

web_search sends the query to a search backend, set in config.toml:

    [web]
    search = "brave"               # "brave", "tavily" or "duckduckgo"
    api_key_env = "BRAVE_API_KEY"  # default: BRAVE_API_KEY / TAVILY_API_KEY

Without a setting, the first backend whose key is in the environment is used,
else DuckDuckGo's HTML page (no key, but best-effort: it may change or refuse).

web_fetch downloads a page and returns it as text (HTML is reduced to readable
markdown-ish text), in pages of MAX_CHARS the model can continue with `start`.

Both reach the network, which is how data leaves the machine (a URL can carry
anything the model read), so neither is read-only: they ask for approval like
commands do, web_fetch per domain.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from wren import __version__
from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput

WEB_TOOLS = frozenset({"web_search", "web_fetch"})
MAX_CHARS = 20_000
MAX_BYTES = 5_000_000
TIMEOUT = 20
USER_AGENT = f"Mozilla/5.0 (compatible; wren/{__version__}; +https://github.com/oncetrange/wren)"
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "application/javascript")


@dataclass
class Response:
    url: str  # after redirects
    status: int
    content_type: str
    body: bytes


Fetcher = Callable[[urllib.request.Request], Response]


def http(request: urllib.request.Request) -> Response:
    """Do a request (following redirects). Raises ToolError."""
    request.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
            body = r.read(MAX_BYTES + 1)
            return Response(r.geturl(), r.status, r.headers.get("content-type", ""), body[:MAX_BYTES])
    except urllib.error.HTTPError as e:
        return Response(e.geturl() or request.full_url, e.code, e.headers.get("content-type", "") if e.headers else "",
                        e.read(MAX_BYTES) if e.fp else b"")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise ToolError(f"couldn't reach {urllib.parse.urlsplit(request.full_url).netloc}: {reason}") from None


# --- search ------------------------------------------------------------------------


@dataclass
class Result:
    title: str
    url: str
    snippet: str


@dataclass
class SearchConfig:
    backend: str | None = None  # brave | tavily | duckduckgo; None: pick by available key
    api_key_env: str | None = None

    def resolve(self) -> tuple[str, str | None]:
        """(backend, key) to use."""
        if self.backend:
            env = self.api_key_env or {"brave": "BRAVE_API_KEY", "tavily": "TAVILY_API_KEY"}.get(self.backend)
            key = os.environ.get(env) if env else None
            if self.backend != "duckduckgo" and not key:
                raise ToolError(f"web search via {self.backend} needs an API key: set ${env}")
            return self.backend, key
        for backend, env in (("brave", "BRAVE_API_KEY"), ("tavily", "TAVILY_API_KEY")):
            if key := os.environ.get(env):
                return backend, key
        return "duckduckgo", None


def search(query: str, count: int, config: SearchConfig, fetch: Fetcher = http) -> tuple[str, list[Result]]:
    backend, key = config.resolve()
    return backend, BACKENDS[backend](query, count, key, fetch)


def _brave(query: str, count: int, key: str | None, fetch: Fetcher) -> list[Result]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": count})
    r = fetch(urllib.request.Request(url, headers={"X-Subscription-Token": key or "", "Accept": "application/json"}))
    data = _json(r, "Brave")
    return [Result(_strip_tags(x.get("title", "")), x.get("url", ""), _strip_tags(x.get("description", "")))
            for x in data.get("web", {}).get("results", [])[:count]]


def _tavily(query: str, count: int, key: str | None, fetch: Fetcher) -> list[Result]:
    body = json.dumps({"query": query, "max_results": count, "api_key": key}).encode()
    r = fetch(urllib.request.Request("https://api.tavily.com/search", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}))
    data = _json(r, "Tavily")
    return [Result(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in data.get("results", [])[:count]]


def _duckduckgo(query: str, count: int, key: str | None, fetch: Fetcher) -> list[Result]:
    body = urllib.parse.urlencode({"q": query}).encode()
    r = fetch(urllib.request.Request("https://html.duckduckgo.com/html/", data=body, method="POST"))
    if r.status != 200:
        raise ToolError(f"DuckDuckGo answered HTTP {r.status}; configure a search API in config.toml "
                        "([web] search = \"brave\" or \"tavily\")")
    page = r.body.decode("utf-8", errors="replace")
    results = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
                         page, re.S):
        href, title, rest = m.groups()
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
        results.append(Result(_strip_tags(title), _ddg_target(html.unescape(href)),
                              _strip_tags(snippet.group(1)) if snippet else ""))
        if len(results) >= count:
            break
    return results


def _ddg_target(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect: //duckduckgo.com/l/?uddg=<url>."""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
    return query["uddg"][0] if "uddg" in query else href


BACKENDS: dict[str, Callable[[str, int, str | None, Fetcher], list[Result]]] = {
    "brave": _brave, "tavily": _tavily, "duckduckgo": _duckduckgo}


def _json(r: Response, name: str) -> dict[str, Any]:
    if r.status != 200:
        detail = r.body[:300].decode("utf-8", errors="replace")
        raise ToolError(f"{name} search failed: HTTP {r.status}: {detail}")
    try:
        return json.loads(r.body)
    except ValueError:
        raise ToolError(f"{name} search returned something that isn't JSON") from None


def _strip_tags(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", text)).split())


class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the web. Returns titles, URLs and snippets; read a page with web_fetch. Use it for "
        "information that isn't in the repository or that may have changed since your training: "
        "library documentation, error messages, release notes, APIs. Write queries as you would "
        "in a search engine (the library name and version help)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "description": "How many results (default 8, at most 20)"},
        },
        "required": ["query"],
    }

    def __init__(self, config: SearchConfig | None = None, fetch: Fetcher = http):
        self.config, self.fetch = config or SearchConfig(), fetch

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("query", "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        count = min(max(args.get("count", 8), 1), 20)
        backend, results = search(args["query"], count, self.config, self.fetch)
        if not results:
            return ToolOutput(f"No results for {args['query']!r}.", summary=f"no results ({backend})")
        text = "\n\n".join(f"{i}. {r.title}\n   {r.url}" + (f"\n   {r.snippet}" if r.snippet else "")
                           for i, r in enumerate(results, 1))
        return ToolOutput(text, summary=f"{len(results)} results ({backend})")


# --- fetch -------------------------------------------------------------------------


class WebFetch(Tool):
    name = "web_fetch"
    description = (
        f"Fetch a web page (http or https) and return its text; HTML is reduced to readable text "
        f"with links. Long pages come in parts of {MAX_CHARS} characters: pass `start` to read "
        "on. Use it to read documentation or pages found with web_search, or URLs the user gave."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "start": {"type": "integer", "description": "Character offset to read from (default 0)"},
        },
        "required": ["url"],
    }

    def __init__(self, fetch: Fetcher = http):
        self.fetch = fetch

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("url", "")

    def permission_key(self, args: dict[str, Any]) -> str:
        return f"web_fetch:{urllib.parse.urlsplit(args.get('url', '')).hostname or ''}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        url = args["url"].strip()
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ToolError("only http(s) URLs can be fetched")
        r = self.fetch(urllib.request.Request(url, headers={"Accept": "text/html,text/plain,*/*;q=0.5"}))
        kind = r.content_type.split(";")[0].strip().lower()
        if r.status >= 400:
            raise ToolError(f"HTTP {r.status} for {r.url}")
        if kind and not kind.startswith(TEXT_TYPES):
            raise ToolError(f"{r.url} is {kind}, not text; only text and HTML pages can be read")
        text = r.body.decode(_charset(r.content_type), errors="replace")
        title = ""
        if "html" in kind or (not kind and "<html" in text[:1000].lower()):
            title, text = html_to_text(text)
        start = max(args.get("start", 0), 0)
        page = text[start:start + MAX_CHARS]
        header = f"{title}\n{r.url}\n" if title else f"{r.url}\n"
        if start or start + MAX_CHARS < len(text):
            end = start + len(page)
            header += f"[characters {start}-{end} of {len(text)}" + (
                f"; continue with start={end}]" if end < len(text) else "]")
        return ToolOutput(f"{header}\n\n{page}" if page else f"{header}\n(no text at this offset)",
                          summary=f"{len(text):,} characters" + (f" · {title[:60]}" if title else ""))


def _charset(content_type: str) -> str:
    m = re.search(r"charset=([\w-]+)", content_type, re.I)
    return m.group(1) if m else "utf-8"


class _TextExtractor(HTMLParser):
    """HTML to plain text that keeps structure: headings, lists, code, tables, links.
    Page chrome (scripts, navigation, headers and footers) is left out."""

    SKIP = {"script", "style", "noscript", "svg", "template", "head", "nav", "footer", "iframe", "form",
            "button", "select"}
    SKIP_ROLES = {"navigation", "banner", "contentinfo", "search", "complementary"}
    BLOCK = {"p", "div", "section", "article", "main", "header", "table", "blockquote", "dl", "dt", "dd",
             "ul", "ol", "br", "hr", "figure", "figcaption", "details", "summary"}
    VOID = {"br", "hr", "img", "input", "meta", "link", "wbr", "source", "col", "area", "base", "embed"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title = ""
        self._skip: list[str] = []  # the tag that started skipping, repeated for nested ones
        self._pre = 0
        self._cell = 0  # inside a table cell: keep its content on the row's line
        self._in_title = False
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if self._skip:
            if tag == self._skip[0] and tag not in self.VOID:
                self._skip.append(tag)
            return
        if tag in self.SKIP or a.get("role") in self.SKIP_ROLES or a.get("aria-hidden") == "true":
            if tag not in self.VOID:
                self._skip.append(tag)
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self._pre += 1
            self.out.append("\n```\n")
        elif tag == "code" and not self._pre:
            self.out.append("`")
        elif tag == "tr":
            self.out.append("\n|")
        elif tag in ("td", "th"):
            self._cell += 1
            self.out.append(" ")
        elif tag == "a":
            self._href = a.get("href")
        elif tag in self.BLOCK and not self._cell:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self._skip:
            if tag == self._skip[0]:
                self._skip.pop()
            return
        if tag == "pre":
            self._pre = max(self._pre - 1, 0)
            self.out.append("\n```\n")
        elif tag == "code" and not self._pre:
            self.out.append("`")
        elif tag in ("td", "th"):
            self._cell = max(self._cell - 1, 0)
            self.out.append(" |")
        elif tag == "a":
            if self._href and self._href.startswith(("http://", "https://")):
                self.out.append(f" ({self._href})")
            self._href = None
        elif (tag in self.BLOCK or re.fullmatch(r"h[1-6]", tag)) and not self._cell:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip:
            return
        self.out.append(data if self._pre else re.sub(r"\s+", " ", data))


def html_to_text(page: str) -> tuple[str, str]:
    """(title, text) of an HTML page."""
    parser = _TextExtractor()
    parser.feed(page)
    parser.close()
    lines, in_code = [], False
    for line in "".join(parser.out).splitlines():
        if line.strip() == "```":
            in_code = not in_code
        lines.append(line.rstrip() if in_code else line.strip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return " ".join(parser.title.split()), text
