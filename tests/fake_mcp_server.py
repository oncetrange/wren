"""A small MCP server for tests, run over stdio: python fake_mcp_server.py"""

import sys
import time

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("fake", instructions="Use echo to repeat text back.")


@server.tool(description="Repeat the text back")
def echo(text: str) -> str:
    return f"echo: {text}"


@server.tool(description="Add two numbers", annotations=ToolAnnotations(read_only_hint=True))
def add(a: int, b: int) -> str:
    return str(a + b)


@server.tool(description="Always fails")
def fail() -> str:
    raise ValueError("the thing broke")


@server.tool(description="Sleep for a while")
def slow(seconds: float) -> str:
    time.sleep(seconds)
    return "woke up"


if __name__ == "__main__":
    print("fake server starting", file=sys.stderr)
    if len(sys.argv) > 1:  # a port: serve Streamable HTTP at http://127.0.0.1:<port>/mcp
        server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
    else:
        server.run("stdio")
