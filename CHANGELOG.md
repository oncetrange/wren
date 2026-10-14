# Changelog

## Unreleased

- `wren --fullscreen` (experimental): a full-screen UI. The conversation
  scrolls above a fixed input line, and pickers and approvals open in its place.
  Long output opens in a pager, Ctrl-G edits the input in `$EDITOR`, Ctrl-Z
  suspends, and the conversation is left in the scrollback on exit.

## 1.0.0 — 2026-10-12

The first release on PyPI, as `wren-agent`.

### The agent

- Works with any Anthropic- or OpenAI-compatible model.
- Builtin models: Kimi (the default), Qwen, Claude and DeepSeek. Ollama and other local
  endpoints are configured in `~/.wren/config.toml`.
- `wren setup` walks you through the first model and its key, with the key kept in `~/.wren/env`.
- Retries model requests that fail mid-stream (dropped or timed-out connections, overloaded
  servers) instead of ending the run.
- Context management in layers:
  - large outputs of older tool calls are masked;
  - older turns are summarized, and a later summary updates the earlier one;
  - the summarized text is archived.

  Both steps keep the prompt cache warm, and a compaction can be undone.
- A task list (`todo_write`) that survives compaction and `/undo`.
- Notices repeated failures, and asks the model to go on when a response is cut off before it
  acts.

### Working in the terminal

- Checkpoints in a shadow git repository before every prompt, with `/undo`, `/rewind` and
  `/diff`. Changes made by shell commands are included.
- Permission modes: ask, accept edits, plan (read-only, with plan review) and auto (`--yolo`).
- Allow, deny and ask rules in `[permissions]`. At a prompt, `p` keeps an answer for the
  project.
- Type while the agent works: the message reaches the model after its current step.
- `@file` mentions with completion, and next-prompt suggestions.
- Sessions are logged and can be resumed (`-c`, `-r`, `/resume`).

### Sandbox

- `--sandbox` runs shell commands in Seatbelt (macOS) or bubblewrap (Linux). Commands there:
  - may write only in the project and the temp directories;
  - can't touch `.git/hooks` or `.git/config`;
  - can't read credentials or see API keys.
- Commands in the sandbox need no approval. Leaving it, or writing files beyond it, always asks.
- Plan mode and read-only subagents use a read-only form of the sandbox.
- Network is off by default, or limited to an allowlist (`"@packages"` for the package
  registries) through a proxy inside wren.

### Extending

- Skills, in the Agent Skills format.
- Long-term memory that the model keeps itself.
- Subagents: built-in `explore` and `general`, plus custom ones in Claude Code's format.
  Read-only subagents run in parallel.
- MCP servers, over stdio and Streamable HTTP.
- Shell hooks on lifecycle events.
- Background commands, `web_search` and `web_fetch`.

### Headless and benchmarks

- `wren -p` with JSON output, `--max-turns`, `--time-limit`, and a final check of the request's
  instructions.
- Scheduled runs on a cron schedule (`wren schedule`).
- A Pier/Harbor adapter for DeepSWE and other Harbor-format benchmarks.
- `bench/evolve.py`, which judges a harness change against measured noise.
