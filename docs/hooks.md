# Hooks

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
