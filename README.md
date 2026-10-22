# wren

A coding agent for the terminal that works with any Anthropic- or OpenAI-compatible model. It is
built from scratch on the raw model SDKs, with no agent framework.

## Install

```bash
uv tool install wren-agent      # or: pipx install wren-agent
```

This installs the `wren` command. It needs Python 3.12+ and runs on macOS or Linux. To install
from a checkout, for development, run `uv tool install -e .`.

The first time you run `wren`, if no model is usable yet, it asks which one you want:

- a builtin model (see [Models](#models)),
- any Anthropic- or OpenAI-compatible endpoint, or
- a local Ollama model.

It then takes the API key with hidden input, checks that the model answers, and makes it the
default. Run `wren setup` to do that again or to add another model.

## Quick start

```bash
export MOONSHOT_API_KEY=sk-...   # for the default `kimi` model
wren                              # interactive session
wren -m claude                    # pick a model
wren -p "fix the failing test"    # one-shot, non-interactive
wren -c                           # continue the last session in this directory
wren -r                           # pick a session to resume (or: wren -r <id>)
wren --plan                       # start in plan mode: investigate, propose, then act
wren --no-sandbox                 # ask before commands instead of sandboxing them
wren --yolo                       # never ask before editing files or running commands
wren --ui inline                  # print into the terminal's scrollback instead of full screen
```

In a session:

- `/help` lists the commands.
- Shift+Tab switches the permission mode.
- `@path` attaches a file.
- Ctrl-C interrupts the agent. You can also type while it works: Enter sends the message, and the
  model reads it after its current step.
- In the full-screen UI, Ctrl-O shows what tool calls returned and Ctrl-F searches the
  conversation. `/help` lists the keys.

wren reads project instructions from `WREN.md` or `AGENTS.md` in the working directory, and
global instructions from `~/.wren/WREN.md`.

## What it does

- **Undo and review**: the workspace is snapshotted before every prompt. `/undo` and `/rewind`
  restore files and the conversation, changes made by shell commands included, and `/diff`
  shows what changed. See [usage](https://github.com/oncetrange/wren/blob/main/docs/usage.md).
- **Permission modes and rules**: wren can ask before edits, accept edits, work in read-only plan
  mode, or never ask. Rules that allow, deny or always ask about certain calls last across
  sessions. See [permissions](https://github.com/oncetrange/wren/blob/main/docs/permissions.md).
- **Sandbox**: shell commands run in Seatbelt on macOS or bubblewrap on Linux, by default. They
  can write only in the project, can't read credentials, and reach only the hosts you allow, so
  they need no approval. See [sandbox](https://github.com/oncetrange/wren/blob/main/docs/sandbox.md).
- **Long sessions**: old tool output is masked and old turns are summarized, while the prompt
  cache stays warm. The full text is kept for `/rewind`. See [usage](https://github.com/oncetrange/wren/blob/main/docs/usage.md).
- **Subagents**: the model can hand self-contained jobs to fresh agents. Read-only ones run in
  parallel, and you can define your own. See [subagents](https://github.com/oncetrange/wren/blob/main/docs/subagents.md).
- **Skills, memory, MCP, hooks**: these use Claude Code's formats where one exists. See
  [skills](https://github.com/oncetrange/wren/blob/main/docs/skills.md), [memory](https://github.com/oncetrange/wren/blob/main/docs/memory.md), [MCP servers](https://github.com/oncetrange/wren/blob/main/docs/mcp.md) and
  [hooks](https://github.com/oncetrange/wren/blob/main/docs/hooks.md).
- **Background commands and the web**: dev servers and watchers keep running in the background,
  and the model can search the web and read pages. See [tools](https://github.com/oncetrange/wren/blob/main/docs/tools.md).
- **Headless and scheduled runs**: `wren -p` can print JSON output and stop at a time limit, and
  prompts can run on a cron schedule. See [headless](https://github.com/oncetrange/wren/blob/main/docs/headless.md).
- **Benchmarks**: wren runs on DeepSWE and other Harbor-format benchmarks, and a harness change
  can be judged against noise, or proposed, screened and judged by wren itself. See [benchmarks](https://github.com/oncetrange/wren/blob/main/docs/benchmarks.md).

## Models

| name | model | endpoint | key |
|---|---|---|---|
| `qwen` | qwen3-coder-plus | DashScope, Anthropic-compatible | `$DASHSCOPE_API_KEY` |
| `kimi` (default) | kimi-k2.7-code (thinking always on) | Moonshot, Anthropic-compatible | `$MOONSHOT_API_KEY` |
| `claude` | claude-opus-5 | Anthropic | `$ANTHROPIC_API_KEY` |
| `qwen-openai` | qwen3-coder-plus | DashScope, OpenAI-compatible | `$DASHSCOPE_API_KEY` |
| `deepseek` | deepseek-chat (no price set) | DeepSeek, OpenAI-compatible | `$DEEPSEEK_API_KEY` |

API keys come from the environment, or from `~/.wren/env`, which holds `KEY=value` lines,
is readable only by you, and is written by `wren setup`. A variable set in your shell wins over
the file. If the default model has no key but another model does, wren uses that one and says
so.

You can override the builtins, or add any Anthropic- or OpenAI-compatible model (Ollama, vLLM, a
gateway), in `~/.wren/config.toml`. See [models](https://github.com/oncetrange/wren/blob/main/docs/models.md).

## Documentation

| | |
|---|---|
| [Using wren](https://github.com/oncetrange/wren/blob/main/docs/usage.md) | commands, mentions, checkpoints, context management, sessions |
| [Models](https://github.com/oncetrange/wren/blob/main/docs/models.md) | configuring models and endpoints |
| [Permission rules](https://github.com/oncetrange/wren/blob/main/docs/permissions.md) | allow, deny and ask rules |
| [Sandbox](https://github.com/oncetrange/wren/blob/main/docs/sandbox.md) | what sandboxed commands may do, and network allowlists |
| [Skills](https://github.com/oncetrange/wren/blob/main/docs/skills.md) · [Memory](https://github.com/oncetrange/wren/blob/main/docs/memory.md) · [Subagents](https://github.com/oncetrange/wren/blob/main/docs/subagents.md) | extending what the model knows and can do |
| [MCP servers](https://github.com/oncetrange/wren/blob/main/docs/mcp.md) · [Hooks](https://github.com/oncetrange/wren/blob/main/docs/hooks.md) | connecting tools, and rules that run as shell commands |
| [Background commands and the web](https://github.com/oncetrange/wren/blob/main/docs/tools.md) | long-running commands, `web_search`, `web_fetch` |
| [Headless and scheduled runs](https://github.com/oncetrange/wren/blob/main/docs/headless.md) | `wren -p`, JSON output, cron schedules |
| [Benchmarks](https://github.com/oncetrange/wren/blob/main/docs/benchmarks.md) | the Pier/Harbor adapter, and measuring a harness change |
| [Architecture](https://github.com/oncetrange/wren/blob/main/docs/architecture.md) | how the code is laid out |
| [Changelog](https://github.com/oncetrange/wren/blob/main/CHANGELOG.md) | what changed in each release |

## Development

```bash
uv run pytest        # tests (a fake API server, no keys or network needed)
uv run ruff check    # lint
uv run pyright       # type check
```

CI runs all three on every pull request, on Python 3.12 and 3.13.

To release:

1. Bump `__version__` in `src/wren/__init__.py`, add the release to `CHANGELOG.md`, and merge.
2. Push a matching tag: `git tag v1.0.1 && git push origin v1.0.1`.

The release workflow then checks, builds and publishes to PyPI with trusted publishing. Running
the workflow by hand publishes to TestPyPI instead.

## License

MIT
