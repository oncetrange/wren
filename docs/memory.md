# Memory

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

Models seldom stop mid-task to take notes. With `/memory auto` on, before a compaction and when a
session ends (`/exit`, `/clear`, `/resume`, Ctrl-D) the model also reviews the conversation for
anything worth keeping, in one side request that mostly reads from the prompt cache (Ctrl-C skips
it). It is off by default, since it makes leaving a session take a moment.

- `/memory` lists memories to view, edit in `$EDITOR` or delete
- `/remember <text>` has the model save something now
- `/memory off` / `/memory on` turns memory off or on for this project; `/memory auto` turns the
  review at compaction and session end on or off (default: off)
- Headless runs (`wren -p`) don't use memory unless given `--memory`, so benchmark runs stay
  independent of each other
