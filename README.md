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
wren --plan                       # start in plan mode: investigate, propose, then act
wren --yolo                       # never ask before editing files or running commands
```

In a session: `/undo`, `/rewind`, `/compact`, `/todos`, `/skills`, `/hooks`, `/resume`, `/clear`, `/model`, `/theme`, `/keys`, `/suggest`, `/cost`, `/help`. Pickers use ↑/↓ and Enter; Esc cancels. Ctrl-C interrupts the agent.

- **Permission modes** (Shift+Tab to switch, shown in the bottom bar): *ask before edits* (default), *accept edits* (edits run freely, commands still ask), *plan* (read-only), and *auto* (`--yolo`).
- **Plan mode**: the model investigates without changing anything, then presents a plan. Approve it (auto-accepting edits or asking for each), or send it back with what to change. Approved plans are saved to `.wren/plans/`. Headless, `wren -p "..." --plan` returns the plan without touching files.
- **Task list**: for multi-step work the model keeps a checklist (`todo_write`), shown as it changes and with `/todos`. It survives compaction and follows `/undo`; if the model tries to finish with open items it gets one reminder.
- **Next-prompt suggestions**: after each answer, wren guesses what you'll type next and shows it as grey text in the empty input; Tab or → accepts it, typing something else dismisses it. It costs one extra request per answer (the same prefix as the last one, so mostly a prompt-cache read); `/suggest` turns it off.
- **Newlines**: Esc Enter or Ctrl-J. For Shift+Enter, run `/keys` for a snippet for your terminal (WezTerm, iTerm2, VS Code).
- **Checkpoints**: the workspace is snapshotted before every prompt into a shadow git repository under `~/.wren/checkpoints` (your project's own git is never touched). `/undo` reverts the last turn's file changes, including those made by shell commands, and rewinds the conversation; `/rewind` goes back further.
- **Context management**, in layers: (1) past `mask_at` tokens (default 40k), large outputs of older tool calls are replaced by a placeholder the model can undo by re-running the call; (2) past `compact_at` (default 80% of the window, at most 200k), older turns are summarized while recent ones stay verbatim, and a later compaction updates the previous summary instead of rewriting it; (3) the summarized text is archived under `~/.wren/transcripts/` and the summary points the model at it. Both steps use hysteresis so they run in occasional batches and keep the prompt cache warm. `/compact` forces a summary; compactions are restore points, so `/undo` or `/rewind` bring the full history back. On a real 88-turn benchmark session, replaying the log with masking cut cumulative prompt tokens by 43%.
- **Sessions**: `/clear` starts a new session; `/resume` (or `wren -c` / `wren -r`) switches back and replays the transcript.
- **Theme**: syntax colors follow the terminal background (detected via OSC 11); override with `/theme`.

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
├── tools/    read_file, write_file, edit_file, bash, grep, glob, todo_write, exit_plan_mode, skill
├── skills.py  skill discovery (Agent Skills format)
├── agent/    the loop, lifecycle events + hooks, conversation state + restore points,
│             compaction, permissions, plan mode, system prompt, JSONL session log/replay
├── checkpoint.py  workspace snapshots in a shadow git repo
└── cli/      REPL and rich rendering
```

- The agent only sees `wren.llm.types`; adding a provider means writing one adapter.
- `edit_file` does exact string replacement, requires the file to have been read, and refuses if it changed on disk since. When the exact text isn't found it tolerates indentation mistakes (a unique match ignoring leading whitespace, with one consistent shift, gets `new_string` re-indented to fit), otherwise it shows the closest region of the file. `read_file` prefixes lines with `N→` rather than a tab, which models confuse with indentation.
- Tool failures are returned to the model as error results so it can correct itself.
- Every session is logged to `~/.wren/sessions/*.jsonl`; resuming replays the log (messages, compactions, rewinds) to rebuild the exact conversation.

## Skills

A skill packages instructions (and any scripts or references they need) for a particular task,
such as your team's commit format or a release checklist. Only each skill's name and description
sit in the model's context; it loads the full instructions with the `skill` tool when a task
matches, so you can keep many skills around at no cost. Run one yourself with `/<name> [arguments]`
(Tab completes it), in a session or headless: `wren -p "/release 1.2.0"`.

```
~/.wren/skills/<name>/SKILL.md      # yours, in every project
.wren/skills/<name>/SKILL.md        # the project's, shareable in git
```

```markdown
---
name: release
description: Cut a release. Use when asked to release or publish a new version.
argument-hint: "[version]"
disable-model-invocation: true    # optional: only run when the user types /release
---
Bump the version to $ARGUMENTS, update CHANGELOG.md, run scripts/check.sh, then tag.
```

Skills use the Agent Skills format, so `~/.claude/skills` and `.claude/skills` are read too (wren's
own directories win on name clashes). `/skills` lists them. See `examples/skills/commit-message`
for a complete one with a helper script.

## Hooks

Hooks run your shell commands at points in the agent's lifecycle: deterministic rules instead of
hoping the model remembers. Define them in `~/.wren/config.toml`, or per project in
`.wren/hooks.toml` (shareable; wren asks before running a project's hooks and again whenever the file
changes; headless runs need `--trust-project-hooks`). `/hooks` lists what is active.

```toml
# format Python files after every edit; lint errors go back to the model
[[hooks.PostToolUse]]
matcher = "edit_file|write_file"
command = 'case "$WREN_FILE" in *.py) ruff format -q "$WREN_FILE" && ruff check -q "$WREN_FILE" >&2 || exit 2;; esac'

# refuse dangerous commands
[[hooks.PreToolUse]]
matcher = "bash"
command = 'jq -r .tool_input.command | grep -qE "rm -rf /|git push --force" && { echo "not allowed" >&2; exit 2; } || true'

# don't let the model finish while the tests fail
[[hooks.Stop]]
command = 'pytest -q -x >/dev/null 2>&1 || { echo "the test suite fails; fix it before finishing" >&2; exit 2; }'
timeout = 300

# desktop notification when wren waits for you (macOS)
[[hooks.Notification]]
command = "osascript -e 'display notification \"wren is waiting for you\"'"
```

Events: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, `Notification`,
`SessionEnd`. Each hook gets the event as JSON on stdin and `WREN_EVENT`, `WREN_PROJECT_DIR`,
`WREN_SESSION_ID`, `WREN_TOOL_NAME`, `WREN_FILE` in its environment. Exit 0 is success (stdout of
`SessionStart` / `UserPromptSubmit` becomes context for the model); exit 2 blocks: the tool call or
prompt is refused, or on `Stop` the model keeps going, with stderr as the reason; other exit codes
and timeouts only warn. On exit 0, a JSON object on stdout can say
`{"decision": "allow" | "deny" | "block", "reason": ..., "additional_context": ...}`; `allow` on
`PreToolUse` skips the permission prompt. `Stop` hooks can keep the model going at most 3 times per
request (`stop_hook_active` in the input tells a hook it already did). Built-in rules such as plan
mode's read-only guard run first and can't be overridden.

## Headless use

```bash
wren -p "fix the failing test" --yolo --output-format json   # one JSON object on stdout
echo "task text" | wren -p - --yolo --max-turns 50            # prompt from stdin
```

Exit code is 0 when the agent finished normally, 1 otherwise (`status` in the JSON says why:
`max_turns`, `error`, ...). Headless runs also have the model re-check the request's explicit
instructions (commit, run tests, ...) once before finishing a request that changed files, and warn
it when `--max-turns` is nearly used up (`--no-final-check` disables both; `--final-check` enables
them interactively). `-m` also accepts a model id such as `moonshot/kimi-k2.7-code`, and
`WREN_MODEL` sets the default.

## Benchmarks (Pier / Harbor)

wren ships a Pier/Harbor installed-agent adapter, so it runs on
[DeepSWE](https://github.com/datacurve-ai/deep-swe) and other Harbor-format benchmarks:

```bash
uv tool install datacurve-pier --with git+https://github.com/oncetrange/wren
git clone https://github.com/datacurve-ai/deep-swe
pier run -p deep-swe/tasks --env modal --n-tasks 10 --sample-seed 0 \
    --agent-import-path wren.integrations.pier_agent:WrenAgent \
    -m moonshot/kimi-k2.7-code --ae MOONSHOT_API_KEY=$MOONSHOT_API_KEY \
    --ak version=main --ak max_turns=150
```

Inside each task container the adapter installs wren from this repository at the given git ref,
runs it headless with network access limited to the model's API host, and reports tokens, cost,
steps and compactions back to Pier. Session logs are kept under the trial's `agent/wren/` logs.

## Development

```bash
uv run pytest
```
