# Permission rules

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
- `net:<host>` isn't a tool: it is about the hosts sandboxed commands may reach through the
  sandbox's proxy (`net:github.com`, `net:*.example.com`). See [the sandbox](sandbox.md).

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
