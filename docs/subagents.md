# Subagents

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
