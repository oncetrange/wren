"""MCP servers: connecting to them and calling their tools.

Servers come from the user config ([mcp.<name>] in ~/.wren/config.toml) and
from a project's .mcp.json (Claude Code's format), which, since it launches
programs, is used only once its exact content is trusted, like project hooks.

The MCP SDK is asynchronous and wren is not, so `McpServers` runs one asyncio
loop on a background thread. Each server lives in its own task there, inside
its client's context, until `close()`; tool calls from the agent's threads are
submitted to that loop and waited for. The SDK is imported only when a server
is configured.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wren.config import CONFIG_DIR, ConfigError, McpServerConfig, parse_mcp_servers

PROJECT_MCP = ".mcp.json"
LOG_DIR = CONFIG_DIR / "logs"
CONNECT_TIMEOUT = 30.0
MAX_INSTRUCTIONS = 2000


@dataclass
class ProjectMcp:
    path: Path
    servers: list[McpServerConfig]
    digest: str


def load_project_mcp(cwd: Path) -> ProjectMcp | None:
    """Parse <cwd>/.mcp.json. Raises ConfigError if it is malformed."""
    path = cwd / PROJECT_MCP
    if not path.is_file():
        return None
    data = path.read_bytes()
    try:
        raw = json.loads(data)
    except (ValueError, UnicodeDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("mcpServers", {}), dict):
        raise ConfigError(f"{path}: expected an object with \"mcpServers\"")
    servers = parse_mcp_servers(raw.get("mcpServers", {}), source=PROJECT_MCP)
    return ProjectMcp(path, servers, hashlib.sha256(data).hexdigest())


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


@dataclass
class ServerState:
    config: McpServerConfig
    status: str = "pending"  # pending | connected | failed | closed
    error: str = ""
    tools: list[ToolInfo] = field(default_factory=list)
    instructions: str = ""
    client: Any = None
    stop: asyncio.Event | None = None
    task: asyncio.Task | None = None


class McpServers:
    def __init__(self, configs: list[McpServerConfig], cwd: Path, log_dir: Path = LOG_DIR):
        self.cwd, self.log_dir = cwd, log_dir
        self.servers = {c.name: ServerState(c) for c in configs}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp", daemon=True)
        self._thread.start()

    # --- lifecycle -----------------------------------------------------------------

    def connect(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Connect to every server at once; failures are recorded, not raised."""
        async def all_of_them() -> None:
            await asyncio.gather(*(self._start(s, timeout) for s in self.servers.values()))
        asyncio.run_coroutine_threadsafe(all_of_them(), self._loop).result()

    async def _start(self, state: ServerState, timeout: float) -> None:
        ready = asyncio.Event()
        state.stop = asyncio.Event()
        state.task = asyncio.create_task(self._serve(state, ready))
        try:
            await asyncio.wait_for(ready.wait(), timeout)
        except TimeoutError:
            state.task.cancel()
            state.status, state.error = "failed", f"no response within {timeout:.0f}s"

    async def _serve(self, state: ServerState, ready: asyncio.Event) -> None:
        from mcp import Client
        from mcp.types import Implementation

        from wren import __version__

        try:
            async with self._transport(state.config) as transport, Client(
                transport, client_info=Implementation(name="wren", version=__version__)
            ) as client:
                state.tools = await _list_tools(client)
                state.instructions = (client.instructions or "").strip()[:MAX_INSTRUCTIONS]
                state.client, state.status = client, "connected"
                ready.set()
                assert state.stop is not None
                await state.stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # noqa: BLE001 - anyio wraps errors in exception groups
            state.status, state.error = "failed", _describe(e)
        finally:
            state.client = None
            if state.status == "connected":
                state.status = "closed"
            ready.set()

    @asynccontextmanager
    async def _transport(self, cfg: McpServerConfig):
        """The transport as an async context manager yielding itself (a Transport)."""
        if cfg.url:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client  # the SDK's timeouts

            async with create_mcp_http_client(headers=cfg.headers or None) as http:
                yield streamable_http_client(cfg.url, http_client=http)
            return
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert cfg.command is not None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # The server's stderr goes to a log file, not over the terminal.
        with (self.log_dir / f"mcp-{cfg.name}.log").open("a", encoding="utf-8") as errlog:
            params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None,
                                           cwd=self.cwd)
            yield stdio_client(params, errlog=errlog)

    def close(self, timeout: float = 5.0) -> None:
        async def stop_all() -> None:
            tasks = [s.task for s in self.servers.values() if s.task is not None]
            for s in self.servers.values():
                if s.stop is not None:
                    s.stop.set()
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=timeout)
                for t in pending:
                    t.cancel()
        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(stop_all(), self._loop).result(timeout + 1)
            except Exception:  # noqa: BLE001 - shutting down regardless
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)

    # --- use -----------------------------------------------------------------------

    def call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """Call a tool; returns the SDK's CallToolResult. Raises McpCallError."""
        state = self.servers[server]
        client = state.client
        if client is None:
            raise McpCallError(f"MCP server {server!r} is not connected"
                               + (f": {state.error}" if state.error else ""))
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(client.call_tool(tool, arguments), state.config.timeout), self._loop)
        try:
            return future.result()
        except KeyboardInterrupt:
            future.cancel()
            raise
        except TimeoutError:
            raise McpCallError(f"{tool} timed out after {state.config.timeout:.0f}s "
                               f"(timeout is set per server)") from None
        except Exception as e:  # noqa: BLE001 - whatever the server or transport did
            raise McpCallError(f"MCP server {server!r} failed: {_describe(e)}") from e

    def instructions(self) -> str:
        """The connected servers' own usage notes, for the system prompt."""
        parts = [f"## {s.config.name}\n{s.instructions}" for s in self.servers.values()
                 if s.status == "connected" and s.instructions]
        if not parts:
            return ""
        return ("# MCP servers\nTools named mcp__<server>__<tool> come from these servers, "
                "which describe themselves as follows.\n\n" + "\n\n".join(parts) + "\n")


class McpCallError(Exception):
    pass


async def _list_tools(client: Any) -> list[ToolInfo]:
    tools, cursor = [], None
    while True:
        page = await client.list_tools(cursor=cursor)
        for t in page.tools:
            hints = t.annotations
            tools.append(ToolInfo(t.name, t.description or t.title or "", dict(t.input_schema),
                                  read_only=bool(hints and hints.read_only_hint)))
        cursor = page.next_cursor
        if not cursor:
            return tools


def _describe(e: BaseException) -> str:
    """A readable message, unwrapping exception groups to their first leaf."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    text = str(e).strip()
    return f"{type(e).__name__}: {text}" if text else type(e).__name__
