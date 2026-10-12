# Using wren

In a session: `/undo`, `/rewind`, `/diff`, `/compact`, `/todos`, `/skills`, `/agents`, `/memory`, `/remember`, `/mcp`, `/jobs`, `/schedule`, `/loop`, `/hooks`, `/permissions`, `/sandbox`, `/resume`, `/clear`, `/model`, `/theme`, `/keys`, `/suggest`, `/cost`, `/help`. Pickers use ↑/↓ and Enter; Esc cancels. Ctrl-C interrupts the agent.

Mention files with `@`: `explain @src/app.py`, `@src/app.py#L40-80` for some lines, `@docs/` for
a directory listing, `@"name with spaces.md"`. Typing `@` completes paths in the bottom line (↑/↓,
Tab or Enter). A mentioned file is attached to the prompt as `read_file` would return it, so the
model can edit it right away; mentions of paths that don't exist (or e-mail addresses) are left
alone.

- **Permission modes** (Shift+Tab to switch, shown in the bottom bar): *ask before edits* (default), *accept edits* (edits run freely, commands still ask), *plan* (read-only), and *auto* (`--yolo`).
- **Permission rules**: at an approval prompt, `a` allows that command (by its program, e.g. every `pytest …`) for the session and `p` for this project from now on. For rules of your own, see [Permission rules](permissions.md).
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
