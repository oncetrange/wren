# MCP servers

wren connects to [MCP](https://modelcontextprotocol.io) servers at startup and gives the model
their tools as `mcp__<server>__<tool>`. Configure them in `~/.wren/config.toml`:

```toml
[mcp.github]                           # a local server, over stdio
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "${GITHUB_TOKEN}" }   # ${VAR} reads your environment

[mcp.docs]                             # a remote server, over Streamable HTTP
url = "https://example.com/mcp"
headers = { Authorization = "Bearer ${DOCS_TOKEN}" }
timeout = 60                           # seconds per tool call (default 120)
```

A project can list servers in `.mcp.json`, in Claude Code's format (`{"mcpServers": {...}}`).
Since that starts programs, wren asks first and remembers the answer until the file changes;
headless runs need `--trust-project-mcp`. MCP tools go through the same permission prompts and
hooks as built-in ones; tools the server marks read-only run without asking. The `general`
subagent can use them, `explore` can't. A server that fails to start is reported and skipped;
`/mcp` shows each server's status, tools and log file (a stdio server's stderr goes to
`~/.wren/logs/mcp-<name>.log`). Only tools are supported, not MCP resources or prompts.
