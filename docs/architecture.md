# Architecture

```
src/wren/
├── llm/            provider-neutral message types, one adapter per API (Anthropic, OpenAI chat completions)
├── agent/
│   ├── loop.py         the agent: call the model, manage context, account for usage
│   ├── tool_runner.py  run a response's tool calls: checks, hooks, approval, concurrency
│   ├── conversation.py history, restore points, masking
│   ├── compact.py      summaries and the transcript archive
│   ├── permissions.py  modes, allow/deny/ask rules, what leaves the sandbox
│   ├── events.py       lifecycle events; builtin_hooks.py and shell_hooks.py handle them
│   ├── subagents.py    the task tool and subagent types
│   └── …               system prompt, plan mode, task list, memory review, session log/replay
├── tools/          read_file, write_file, edit_file, bash, bash_output, kill_job, grep, glob,
│                   todo_write, exit_plan_mode, skill, memory, web_search, web_fetch, MCP tools
├── sandbox.py      Seatbelt / bubblewrap sandbox for shell commands
├── netproxy.py     the allowlist proxy sandboxed commands reach the network through
├── netbridge.py    reaches that proxy from inside a Linux sandbox's network namespace
├── checkpoint.py   workspace snapshots in a shadow git repo
├── config.py       models and the rest of ~/.wren/config.toml
├── skills.py · memory.py · mcp_servers.py · mentions.py · schedules.py
├── integrations/   the Pier/Harbor benchmark adapter
└── cli/            command line, REPL, slash commands, typing while the agent works;
                    render.py draws things, screen.py is where commands print and ask
```

- The agent only sees `wren.llm.types`; adding a provider means writing one adapter.
- `edit_file` does exact string replacement, requires the file to have been read, and refuses if it changed on disk since. When the exact text isn't found it tolerates indentation mistakes (a unique match ignoring leading whitespace, with one consistent shift, gets `new_string` re-indented to fit), otherwise it shows the closest region of the file. `read_file` prefixes lines with `N→` rather than a tab, which models confuse with indentation.
- Tool failures are returned to the model as error results so it can correct itself.
- Every session is logged to `~/.wren/sessions/*.jsonl`; resuming replays the log (messages, compactions, rewinds) to rebuild the exact conversation.
