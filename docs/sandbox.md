# Sandbox

wren runs the model's shell commands, background jobs included, in an OS sandbox. On macOS
this uses Seatbelt (`sandbox-exec`); on Linux it uses
[bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`). Inside the sandbox, a command:

- can write only in the project directory and the temp directories. `.git/hooks` and
  `.git/config` stay read-only, because either could make git run code later, outside the
  sandbox. Commits work as usual.
- can read everything except where credentials live: `~/.ssh`, `~/.aws`, `~/.gnupg`,
  `~/.config/gh`, `~/.netrc`, `~/.wren/env` and the like.
- reaches only the hosts you allow, and asks about others (see below). On macOS, servers on localhost
  are still reachable. On Linux, each command has its own localhost, so it can reach only
  servers it started itself.
- doesn't see API keys and tokens (`*_API_KEY`, `*_TOKEN`, each model's key variable).
- can't reach Docker, Podman or other container engines through their sockets (Docker
  Desktop, Colima, OrbStack, Rancher Desktop, Lima, `DOCKER_HOST`): a container started that
  way could mount any directory. A command that needs Docker has to leave the sandbox, which
  asks you first.
- uses package caches of its own, in `~/.wren/sandbox-cache`: uv, pip, npm, pnpm, yarn, bun,
  go, gradle and pre-commit are pointed there. Your own caches stay as they are, because a
  command that could change them could change what runs later outside the sandbox (uv even
  hard-links its cache into virtual environments). A project whose dependencies are already
  installed runs without the network; installing a package the sandbox's cache doesn't have
  yet needs it (see `network` below).

```toml
[sandbox]
enabled = "auto"                   # the default; or true, or false
allow_write = ["~/.cargo/registry", "~/.cargo/git"]   # e.g. so cargo can update its cache
deny_read = ["~/work/secrets"]
network = "ask"                    # the default; or "deny", "allow", or a list of hosts, below
```

`enabled = "auto"` sandboxes commands wherever the sandbox works, except with `--yolo`, which
asks for no guard rails at all. `--sandbox` (or `enabled = true`) sandboxes them with `--yolo`
too, and `--no-sandbox` (or `enabled = false`) never does.

cargo keeps its cache inside `~/.cargo`, next to your cargo configuration, so it isn't moved:
cargo needs `allow_write` as above, with the risk above. To share another tool's cache with
the sandbox, set its variable (such as `UV_CACHE_DIR`) to a directory in `allow_write`; wren
then leaves it alone.

To let commands reach certain hosts, make `network` a list of them:

```toml
network = ["@packages", "github.com", "*.githubusercontent.com"]
```

`@packages` stands for the package registries: PyPI, npm and yarn, crates.io, the Go module
proxy, RubyGems and Maven Central. `example.com` allows that host only, and `*.example.com`
allows its subdomains.

With `network = "ask"`, the default, a command can reach a host once you allow it. A host that isn't
allowed yet is refused at once, and when the command ends wren asks you about it: `y` allows it
for the session, `p` always in this project, `n` refuses it for the session. The model is told
what you said, so it can run the command again. `"@ask"` in a list asks about the hosts that
aren't on it, such as `["@packages", "@ask"]`. Headless runs have nobody to ask, so the
host stays refused. Read-only subagents don't ask either.

Permission rules can name hosts too, as `net:<host>`. `allow = ["net:github.com"]` lets
commands reach it, with or without asking, and `deny = ["net:*.example.com"]` refuses those
hosts even when `network` lists them. `p` saves the same rule for the project.

The traffic goes through a proxy inside wren, which commands get in `HTTP_PROXY` and
`HTTPS_PROXY`. pip, uv, npm, cargo, go and git over https all use those variables. The sandbox
blocks every other connection, so a host that isn't on the list is refused, whichever program
tries.

HTTPS stays end to end. The proxy sees only the host name and never decrypts anything.

On Linux, the command's own network namespace can't reach the proxy directly. So a small bridge
(`wren.netbridge`) carries the connection through a unix socket.

When the proxy refuses a host, the model is told which host it was. `/sandbox` lists the hosts
refused lately. `github.com` is left out of `@packages` on purpose: anyone who can reach it
can push code out of the sandbox. Add it yourself if you need it.

Commands in the sandbox run without asking, in every permission mode. Deny and ask rules still
apply to them.

- **Plan mode** and read-only subagents such as `explore` run commands in a read-only form of
  the sandbox: only the temp directories and `allow_write` stay writable, and the project does
  not, even if it lives in `/tmp`. So investigating needs no approvals. `explore` can run any
  command, not just its list of known read-only ones, but it can't leave the sandbox.
- **The file tools** run inside wren, outside the sandbox, so wren checks their writes against
  the same bounds. Writing a file the sandbox wouldn't let a command write, outside the project
  or into `.git/hooks` or `.git/config`, asks every time, in every mode, and is never
  remembered.
- **`/sandbox`** shows what the sandbox allows, and `/sandbox on` / `/sandbox off` switch it
  mid-session. The bottom bar shows `sandbox` while it's on.

When a command fails in a way that looks like the sandbox's doing, the model is told what the
sandbox allows. If the task really needs more, the model can run the command again with
`sandbox: false`. That always asks you first, in every mode including `--yolo`, and is refused
in headless runs. At the prompt, `a` or `p` remember it per program as `unsandboxed:<program>`,
so allowing `uv` inside the sandbox never lets it out. Permission rules can use the same name,
for example `allow = ["unsandboxed:git fetch"]`, and deny rules apply inside the sandbox and
out.

If the sandbox can't be set up, wren says why. Common causes are bwrap not being installed,
or Ubuntu 24.04's AppArmor blocking the user namespaces bwrap needs, and Windows has none.
Commands then ask for approval as the permission mode says. With `--sandbox` or
`enabled = true`, a headless or `--yolo` run refuses to start instead, since nobody would be
asked. Benchmark runs leave the sandbox off: they already run in a container.
