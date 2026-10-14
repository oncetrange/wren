# Changelog

## Unreleased

- A full-screen UI, now the default. The conversation scrolls above an input
  line that is always there, and pickers and approvals open in its place.
  Completions open as a list above the input line. The bottom bar fits its
  width, shortening or leaving out the least useful parts first.
  - Ctrl-O shows what tool calls returned, and whole diffs.
  - Ctrl-F finds text in the conversation.
  - Long output opens in a pager.
  - Dragging selects text and copies it on release, with wrapped lines
    joined again; double-click copies a word, triple-click a line.
  - Ctrl-S hands the mouse to the terminal, for its own selection.
  - Ctrl-G edits the input in `$EDITOR`, and Ctrl-Z suspends wren.
- MCP servers connect in the background, so the session opens at once; the
  first prompt waits for servers still connecting.
- Reviewing the session for memories at exit (and at compaction) is off by
  default; `/memory auto` turns it on. Settings now store only what differs
  from the defaults, so a changed default reaches everyone who didn't choose.
- `/copy` copies a reply to the clipboard, or through the terminal (OSC 52)
  over SSH.
- `/ui`, `wren --ui` and the `ui` setting choose between the full-screen and
  the inline UI. `auto` picks full screen, except in Zellij.

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
