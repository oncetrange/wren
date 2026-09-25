# wren

A multi-model coding agent for the terminal, built from scratch on the raw model SDKs — no agent framework.

## Install

```bash
uv tool install -e .     # installs the `wren` command
```

## Usage

```bash
export MOONSHOT_API_KEY=sk-...   # for the default `kimi` model
wren                              # interactive session
wren -m claude                    # pick a model
wren -p "fix the failing test"    # one-shot, non-interactive
wren -c                           # continue the last session in this directory
wren -r                           # pick a session to resume (or: wren -r <id>)
wren --yolo                       # never ask before editing files or running commands
```

In a session: `/undo`, `/rewind`, `/compact`, `/model [name]`, `/clear`, `/cost`, `/help`. Esc-Enter inserts a newline; Ctrl-C interrupts the agent.

- **Checkpoints**: the workspace is snapshotted before every prompt into a shadow git repository under `~/.wren/checkpoints` (your project's own git is never touched). `/undo` reverts the last turn's file changes, including those made by shell commands, and rewinds the conversation; `/rewind` goes back further.
- **Compaction**: when the conversation nears the model's context window it is replaced by a structured summary, automatically or with `/compact`.

Project-specific instructions are read from `WREN.md` or `AGENTS.md` in the working directory, and global ones from `~/.wren/WREN.md`.

## Models

| name | model | endpoint | key |
|---|---|---|---|
| `qwen` | qwen3-coder-plus | DashScope, Anthropic-compatible | `$DASHSCOPE_API_KEY` |
| `kimi` (default) | kimi-k2.7-code (thinking always on) | Moonshot, Anthropic-compatible | `$MOONSHOT_API_KEY` |
| `claude` | claude-opus-5 | Anthropic | `$ANTHROPIC_API_KEY` |

Override builtins or add any Anthropic-compatible model in `~/.wren/config.toml`:

```toml
default_model = "kimi"

[models.kimi]                      # fields merge over the builtin
base_url = "https://api.moonshot.ai/anthropic"   # international platform key

[models.local]                     # any Anthropic-compatible endpoint
model = "my-model"
base_url = "http://localhost:8000"
api_key_env = "LOCAL_API_KEY"
auth = "bearer"                    # send the key as Authorization: Bearer
prompt_cache = false
thinking = { type = "enabled", budget_tokens = 8000 }
price = { input = 0.6, output = 2.5 }   # USD per million tokens, enables cost display
```

Models are only read from the user config, never from the project directory, so a repository can't redirect your API key to its own endpoint.

## Architecture

```
src/wren/
├── llm/      provider-neutral message types + one adapter per API (Anthropic for now)
├── tools/    read_file, write_file, edit_file, bash, grep, glob
├── agent/    the loop, compaction, permissions, system prompt, JSONL session log/replay
├── checkpoint.py  workspace snapshots in a shadow git repo
└── cli/      REPL and rich rendering
```

- The agent only sees `wren.llm.types`; adding a provider means writing one adapter.
- `edit_file` does exact string replacement, requires the file to have been read, and refuses if it changed on disk since.
- Tool failures are returned to the model as error results so it can correct itself.
- Every session is logged to `~/.wren/sessions/*.jsonl`; resuming replays the log (messages, compactions, rewinds) to rebuild the exact conversation.

## Development

```bash
uv run pytest
```
