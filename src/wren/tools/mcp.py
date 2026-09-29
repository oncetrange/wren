from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput, truncate

if TYPE_CHECKING:
    from wren.mcp_servers import McpServers, ToolInfo

PREFIX = "mcp__"
MAX_NAME = 64  # the model APIs' limit on tool names
MAX_DESCRIPTION = 2000


def tool_name(server: str, tool: str) -> str:
    return (PREFIX + re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool}"))[:MAX_NAME]


class McpTool(Tool):
    """One tool of a connected MCP server, as `mcp__<server>__<tool>`.

    It goes through the same permission prompts and hooks as built-in tools;
    tools the server marks read-only run without asking. The server validates
    the arguments against its own schema, which may use JSON Schema features
    wren's minimal check doesn't know."""

    strict_args = False
    name = ""  # set per instance
    description = ""
    input_schema: dict[str, Any] = {}

    def __init__(self, servers: McpServers, server: str, info: ToolInfo):
        self.servers, self.server, self.tool = servers, server, info.name
        self.name = tool_name(server, info.name)  # type: ignore[misc]
        self.description = (info.description or f"{info.name} (MCP server {server})")[:MAX_DESCRIPTION]  # type: ignore[misc]
        schema = dict(info.input_schema)
        schema.setdefault("type", "object")
        self.input_schema = schema  # type: ignore[misc]
        self.read_only = info.read_only  # type: ignore[misc]

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        text = json.dumps(args, ensure_ascii=False)
        return text if len(text) <= 120 else text[:117] + "…"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        from wren.mcp_servers import McpCallError

        try:
            result = self.servers.call(self.server, self.tool, args)
        except McpCallError as e:
            raise ToolError(str(e)) from None
        text = result_text(result)
        lines = text.count("\n") + 1 if text else 0
        summary = (text.splitlines()[0][:100] if result.is_error and text
                   else f"{lines} line{'s' if lines != 1 else ''}")
        return ToolOutput(truncate(text) or "(no output)", is_error=bool(result.is_error), summary=summary)


def result_text(result: Any) -> str:
    """A CallToolResult as text: text parts as they are, other media described."""
    parts = []
    for block in result.content or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} ({block.mime_type}), not shown]")
        elif kind == "resource":
            resource = block.resource
            text = getattr(resource, "text", None)
            parts.append(text if text is not None else f"[binary resource {resource.uri}, not shown]")
        elif kind == "resource_link":
            parts.append(f"[resource: {getattr(block, 'uri', '')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content, not shown]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def mcp_tools(servers: McpServers) -> list[McpTool]:
    """Tools of the connected servers; the first of two with the same name wins."""
    tools: dict[str, McpTool] = {}
    for name, state in servers.servers.items():
        if state.status == "connected":
            for info in state.tools:
                tool = McpTool(servers, name, info)
                tools.setdefault(tool.name, tool)
    return list(tools.values())
