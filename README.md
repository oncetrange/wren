# wren

A multi-model coding agent for the terminal, built from scratch on the raw model SDKs — no agent framework.

## Install

```bash
uv tool install wren-agent      # or: pipx install wren-agent
```

This installs the `wren` command (Python 3.12+, macOS or Linux). From a checkout, for development:
`uv tool install -e .`.

The first time you run `wren` without a model it can use, it asks which one you want (a builtin
below, any Anthropic- or OpenAI-compatible endpoint, or a local Ollama model), takes the API key
with hidden input, checks that the model answers, and makes it the default. Run `wren setup` to do
that again or add another model.

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

In a session: `/undo`, `/rewind`, `/diff`, `/compact`, `/todos`, `/skills`, `/agents`, `/memory`, `/remember`, `/mcp`, `/jobs`, `/schedule`, `/loop`, `/hooks`, `/permissions`, `/resume`, `/clear`, `/model`, `/theme`, `/keys`, `/suggest`, `/cost`, `/help`. Pickers use ↑/↓ and Enter; Esc cancels. Ctrl-C interrupts the agent.

Mention files with `@`: `explain @src/app.py`, `@src/app.py#L40-80` for some lines, `@docs/` for
a directory listing, `@"name with spaces.md"`. Typing `@` completes paths in the bottom line (↑/↓,
Tab or Enter). A mentioned file is attached to the prompt as `read_file` would return it, so the
model can edit it right away; mentions of paths that don't exist (or e-mail addresses) are left
alone.

- **Permission modes** (Shift+Tab to switch, shown in the bottom bar): *ask before edits* (default), *accept edits* (edits run freely, commands still ask), *plan* (read-only), and *auto* (`--yolo`).
- **Permission rules**: at an approval prompt, `a` allows that command (by its program, e.g. every `pytest …`) for the session and `p` for this project from now on. For rules of your own, see [Permission rules](#permission-rules).
- **Plan mode**: the model investigates without changing anything, then presents a plan. Approve it (auto-accepting edits or asking for each), or send it back with what to change. Approved plans are saved to `.wren/plans/`. Headless, `wren -p "..." --plan` returns the plan without touching files.
- **Task list**: for multi-step work the model keeps a checklist (`todo_write`), shown as it changes and with `/todos`. It survives compaction and follows `/undo`; if the model tries to finish with open items it gets one reminder.
- **Next-prompt suggestions**: after each answer, wren guesses what you'll type next and shows it as grey text in the empty input; Tab or → accepts it, typing something else dismisses it. It costs one extra request per answer (the same prefix as the last one, so mostly a prompt-cache read); `/suggest` turns it off.
- **Typing while wren works**: the input stays open while the agent runs. What you type shows under the spinner, and Enter sends it: the model gets it right after the tool calls it is running, without being interrupted. If the model had already finished, it answers the message as a new prompt. A message that never reached the model, for example after Ctrl-C, comes back in the input line.
- **Newlines**: Esc Enter or Ctrl-J. For Shift+Enter, run `/keys` for a snippet for your terminal (WezTerm, iTerm2, VS Code).
- **Checkpoints**: the workspace is snapshotted before every prompt into a shadow git repository under `~/.wren/checkpoints` (your project's own git is never touched). `/undo` reverts the last turn's file changes, including those made by shell commands, and rewinds the conversation; `/rewind` goes back further. `/diff` shows what changed since the last prompt (`/diff all`: since the session began; add paths to narrow it), shell commands' changes included, through `$PAGER` or `less` like `git diff`.
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
| `qwen-openai` | qwen3-coder-plus | DashScope, OpenAI-compatible | `$DASHSCOPE_API_KEY` |
| `deepseek` | deepseek-chat (no price set) | DeepSeek, OpenAI-compatible | `$DEEPSEEK_API_KEY` |

API keys come from the environment, or from `~/.wren/env` (`KEY=value` lines, readable only by
you; `wren setup` writes it). A variable set in your shell wins over the file. When the default
model has no key but another model does, wren uses that one and says so.

Override builtins or add any Anthropic- or OpenAI-compatible model in `~/.wren/config.toml`:

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

[models.ollama]                    # any OpenAI-compatible chat completions endpoint
provider = "openai"
model = "qwen3-coder:30b"
base_url = "http://localhost:11434/v1"
api_key_env = ""                   # no key needed (default: $OPENAI_API_KEY)
context_window = 65536
reasoning_effort = "medium"        # optional, sent as is
extra_body = { enable_thinking = true }   # optional vendor-specific request fields
replay_reasoning = true            # send reasoning back with tool calls (DeepSeek/Kimi thinking)
```

With `provider = "openai"`, wren speaks Chat Completions: tool results become `tool` messages,
tool errors are marked with `Error:`, reasoning streams as thinking, and cached prompt tokens are
read from the usage report (caching itself is automatic). For OpenAI's own endpoint the output
limit goes in `max_completion_tokens`; set `max_tokens_param` to override.

Models are only read from the user config, never from the project directory, so a repository can't redirect your API key to its own endpoint.

## Permission rules

Rules say what runs without asking, what is refused, and what always asks, in every project
(`~/.wren/config.toml`):

```toml
[permissions]
allow = ["bash:pytest", "bash:git diff", "bash:git log", "web_fetch:*.python.org"]
deny = ["bash:git push --force", "bash:rm -rf", "edit:.env", "read_file:*secret*"]
ask = ["bash:git push", "mcp__github__*"]     # asks even in auto mode (--yolo)
```

A rule is a tool name, or `edit` for both `edit_file` and `write_file`, optionally followed by
`:pattern`, where `*` matches anything. What the pattern is matched against depends on the tool:

- For `bash`, it is matched against the command. `bash:git diff` covers `git diff` with any
  arguments, but not `git diffx`.
- For the file tools, it is matched against the path, relative to the working directory.
  A pattern without `/` also matches a file name, so `edit:*.lock` matches anywhere in the tree.
- For `web_fetch`, it is matched against the host.

A bare name covers every call of that tool. Tool names can use `*` too, as in
`mcp__github__*`.

Deny beats ask, and ask beats allow. Deny and ask rules apply in every mode, auto included.
A call matched by an ask rule is never remembered.

An allow rule only covers a simple command. A command with `&&`, `|`, `;`, a redirection or
`$(…)` is allowed only when the rule spells out that whole command. Deny and ask rules look at
every command inside a compound one, so `deny = ["bash:rm"]` also refuses `make && rm -rf x`.
That check is a guard against mistakes, not a sandbox. For a hard rule, use a `PreToolUse`
hook.

Answering `p` at a prompt saves the allow rule in
`~/.wren/projects/<path>/permissions.toml`, which uses the same format and can be edited by
hand. The file lives outside the repository on purpose: a cloned repository can't grant itself
any permissions. `/permissions` shows the current mode, every rule in force, and what this
session allows.

## Architecture

```
src/wren/
├── llm/      provider-neutral message types + one adapter per API (Anthropic, OpenAI chat completions)
├── tools/    read_file, write_file, edit_file, bash, bash_output, kill_job, grep, glob, todo_write, exit_plan_mode, skill, task, memory, web_search, web_fetch, MCP tools
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

## Memory

wren keeps a long-term memory across sessions, and the model maintains it itself with the
`memory` tool: what you corrected or confirmed about how it works, who you are and what you prefer,
project context that isn't in the code (decisions, deadlines), and where to find external
resources. It doesn't store what the repository already records, such as code structure or history.

```
~/.wren/memory/                       # about you, used in every project
~/.wren/projects/<path>/memory/       # about this project (outside the repo: never committed)
```

Each memory is one markdown file with frontmatter; `MEMORY.md` in each directory is an index that
wren regenerates. The indexes are in the system prompt, and the model reads a memory's text when
it looks relevant. Memories are background knowledge, not instructions: when one conflicts with
what you ask now, your request wins. Saving needs no approval, but every change shows as a line:

```
● memory write user/prefers-pytest
  ⎿ saved user/prefers-pytest: User prefers pytest over unittest
```

Models seldom stop mid-task to take notes, so before a compaction and when a session ends
(`/exit`, `/clear`, `/resume`, Ctrl-D) the model also reviews the conversation for anything worth
keeping, in one side request that mostly reads from the prompt cache (Ctrl-C skips it).

- `/memory` lists memories to view, edit in `$EDITOR` or delete
- `/remember <text>` has the model save something now
- `/memory off` / `/memory on` turns memory off or on for this project; `/memory auto` toggles the
  end-of-session review
- Headless runs (`wren -p`) don't use memory unless given `--memory`, so benchmark runs stay
  independent of each other

## Subagents

With the `task` tool the model hands a self-contained job to a subagent: a fresh agent with its
own context that returns only its final report. The files it reads and the searches it runs
never enter the main conversation, which keeps long sessions small.

| Type | Tools | For |
|---|---|---|
| `explore` | read_file, grep, glob, bash (read-only commands only) | finding where and how things are |
| `general` | everything except task, todo_write, exit_plan_mode | a well-defined piece of work |

`explore` never asks for approval and never changes anything: bash runs only commands that read
(`ls`, `grep`, `find`, `git log/diff/show` and the like, piped together or not); anything else,
including redirections into files, is refused. `general` asks for approval exactly like the main
agent, under the same permission mode. In plan mode only `explore` runs. A subagent's tool calls
show indented under its task; its changes are part of the current turn, so `/undo` reverts them.
Its usage counts toward `/cost`, and its full log is in `~/.wren/sessions/subagents/`.
Read-only subagents requested together run at the same time (up to 4), with one status line
each; Ctrl-C stops them all. Subagents that can change files run one after another.

Define your own subagents as markdown files; `/agents` lists what the model can use.

```
~/.wren/agents/<name>.md      # yours, in every project
.wren/agents/<name>.md        # the project's, shareable in git
```

```markdown
---
name: reviewer
description: Reviews a finished change for bugs and missed cases. Use after making a change.
tools: read_file, grep, glob, bash   # optional, default all; Claude Code names (Read, Bash…) work too
read-only: true                      # optional: no edits, read-only bash (default: true only if
                                     # the tools can't write files or run commands)
model: kimi                          # optional: a model from config.toml, e.g. a cheaper one
max-turns: 20                        # optional, default 30
---
Review the change described in the task. Run `git diff` to see it, read the surrounding code,
and report concrete problems with path:line, most serious first.
```

The file format is Claude Code's, so `~/.claude/agents` and `.claude/agents` are read too (wren's
own directories win on name clashes, and a definition named `explore` or `general` replaces the
built-in one). See `examples/agents/reviewer.md`.

## Background commands

For commands that keep running (a dev server, a watcher, a long build) the model runs `bash` with
`run_in_background`: it gets a job id right away, reads new output with `bash_output` (optionally
waiting for the job to finish or for a line like "listening on" to appear) and stops the job with
`kill_job`. It is told when a job ends. Jobs run in their own process group and are stopped when
wren exits (a subagent's when the subagent finishes); `/jobs` lists them, shows their output or
stops one, and the bottom line counts the running ones.

## Web access

`web_search` searches the web and `web_fetch` reads a page as text (HTML reduced to headings,
lists, tables, code and links; long pages in parts). Both ask for approval, `web_fetch` per
domain, since a URL can carry out anything the model has read. Searching uses Brave or Tavily
when `$BRAVE_API_KEY` or `$TAVILY_API_KEY` is set, otherwise DuckDuckGo's HTML page (no key, but
best-effort). To choose, or to turn web access off:

```toml
[web]
search = "tavily"                  # brave, tavily or duckduckgo
api_key_env = "MY_TAVILY_KEY"      # default: BRAVE_API_KEY / TAVILY_API_KEY
# enabled = false                  # no web tools (or run with --no-web)
```

The `general` subagent gets the web tools too; `explore` doesn't. Benchmark runs through the Pier
adapter have no web access: the tasks come from public repositories whose real fixes are online.

## MCP servers

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

## Scheduled runs

wren can run prompts on a cron schedule, headless, even when no session is open: "every weekday
at 9, check CI on main and summarize failures". Ask for one in a session (the model proposes the
job and you approve it; creating or deleting a job always asks, whatever the permission mode), or:

```bash
wren schedule add "0 9 * * 1-5" "check CI on main and summarize failures"   # runs in this directory
wren schedule add @daily "run the tests; fix failures on a branch" --mode accept_edits --id nightly
wren schedule install     # once: runs `wren schedule tick` every minute (launchd on macOS, else crontab)
wren schedule list | logs <id> | run <id> | pause <id> | resume <id> | rm <id> | status | uninstall
```

- Cron expressions have five fields in local time (`*/15`, `1-5`, `mon-fri`, `jan,jul`) or a
  shortcut (`@hourly`, `@daily`, `@weekly`, `@monthly`).
- Jobs run in plan mode (read-only) unless given `--mode accept_edits` or `--mode auto`; anything
  that would need approval is refused, since nobody is there to give it.
- Each run is a normal `wren -p` session: `wren schedule logs <id>` shows its result and cost,
  and `wren -r <session>` continues it interactively. A desktop notification reports each run.
- A run missed while the machine was asleep runs once when it wakes (`--no-catch-up` skips it
  instead); a run due while the previous one is still going is skipped.
- Scheduled runs don't see your shell's environment: put API keys in `~/.wren/env` as `KEY=value`
  lines (`chmod 600` it); `wren schedule install` tells you which are missing. Jobs are stored in
  `~/.wren/schedules.json`, runs in `~/.wren/schedules/<id>/`.

In a session, `/schedule` lists jobs to run now, pause or delete, and `/loop 10m <prompt>` repeats
a prompt every ten minutes while the session is open (Ctrl-C stops it).

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

`--time-limit MINUTES` bounds a run's wall-clock time: with a quarter of it left the model is
reminded to save its progress (e.g. commit what works), near the end to wrap up, and at the limit
wren stops (`status: time_limit`). Set it a little under a harness's own timeout, so the work is
committed before the harness kills the run.

For experiments, `--mask-at TOKENS` and `--compact-at TOKENS` override the model's context
thresholds and `--no-subagents` removes the task tool; the thresholds used are recorded at the
start of the session log.

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

`-m` takes a builtin model or one from your `~/.wren/config.toml` (by name or model id): the
adapter reads the definition on your machine and writes just that model into the container's
config, without keys; pass the key with `--ae KEY_ENV=...` as above. Pier imports the adapter
from the wren installed alongside it (`--with`), so update that install to get adapter changes.

The same experiment options are agent kwargs: `--ak mask_at=24000`, `--ak compact_at=100000`,
`--ak no_subagents=true`, and `--ak time_limit=MINUTES` (a little under the agent timeout).

Inside each task container the adapter installs wren from this repository at the given git ref,
runs it headless with network access limited to the model's API host, and reports tokens, cost,
steps and compactions back to Pier. Session logs are kept under the trial's `agent/wren/` logs.

### Measuring a harness change

`bench/evolve.py` decides whether a change to wren (a commit, or agent kwargs such as `mask_at`)
actually helps, using the selection rules of [RRSI](https://github.com/google-research/rrsi). One
trial says little: a task that passes once can fail the next time. So changes are measured on a
fixed evolve split of DeepSWE (`bench/split.json`, 10 tasks) with k trials per task, against a noise
band measured by running the same harness twice:

```bash
python bench/evolve.py eval main                  # twice: two measurements of the same harness
python bench/evolve.py calibrate <name> <name>    # noise band delta
python bench/evolve.py baseline <name>            # the incumbent
python bench/evolve.py eval my-branch             # a pushed commit; or: eval main --ak mask_at=80000
python bench/evolve.py judge <name> --component context_mgmt --hypothesis "..."
python bench/evolve.py status                     # incumbent, delta, ledger
```

A trial without a grade counts as a failure. A candidate is accepted only if it doesn't fall
below the best score so far minus delta, and pays for any extra tokens with a gain larger than
delta (inside the band, only a token saving gets in). Every verdict, rejections included, goes into
a ledger in `bench/runs/`. The other 103 tasks are held out: `eval <ref> --set heldout` reports a
result but is never used to choose. Settings are in `bench/config.json`.

## Development

```bash
uv run pytest        # tests (a fake API server, no keys or network needed)
uv run ruff check    # lint
uv run pyright       # type check
```

CI runs all three on every pull request, on Python 3.12 and 3.13.

Releases: bump `__version__` in `src/wren/__init__.py`, merge, then push a matching tag
(`git tag v0.2.0 && git push origin v0.2.0`); the release workflow checks, builds and publishes
to PyPI with trusted publishing. Running the workflow by hand publishes to TestPyPI instead.
