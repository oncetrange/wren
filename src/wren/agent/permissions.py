"""What may run without asking.

Besides the mode and the session's "always allow" answers there are rules, from
`[permissions]` in ~/.wren/config.toml and from the answers saved for a project
(~/.wren/projects/<path>/permissions.toml, outside the repo so a clone can't
grant itself anything). A rule is a tool name, or `edit` for both file-writing
tools, optionally with `:pattern` (`bash:git diff`, `edit:*.lock`,
`web_fetch:*.python.org`, `mcp__github__*`). Deny beats ask beats allow, and deny
and ask hold in every mode, auto included. `net:<host>` rules are about the hosts
sandboxed commands may reach (`net:github.com`, `net:*.example.com`).
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from wren.config import ConfigError, PermissionRules, parse_permission_rules, project_home
from wren.tools.base import Tool, wildcard

if TYPE_CHECKING:
    from wren.sandbox import Sandbox

Mode = Literal["ask", "accept_edits", "plan", "auto"]

# Shift+Tab cycles through these; "auto" (--yolo) is only offered if started with it.
CYCLE: tuple[Mode, ...] = ("ask", "accept_edits", "plan")
LABELS: dict[Mode, str] = {
    "ask": "ask before edits",
    "accept_edits": "accept edits",
    "plan": "plan mode (read-only)",
    "auto": "auto: no approvals",
}


@dataclass
class Decision:
    allow: bool
    remember: bool = False
    # Remember it for the project, across sessions (implies remember).
    persist: bool = False
    # When denying, an optional note passed back to the model ("use pytest -x instead").
    feedback: str = ""


@dataclass
class Permissions:
    """What may run without asking.

    ask           reads run freely; edits and commands ask (unless always-allowed)
    accept_edits  edits run freely too; commands ask
    plan          read-only: edits are refused, every command asks (the allowlist
                  doesn't apply, since a command could modify files)
    auto          everything runs
    """

    mode: Mode = "ask"
    allowed: set[str] = field(default_factory=set)
    # Whether Shift+Tab may cycle into "auto" (only when started with --yolo).
    allow_auto: bool = False
    rules: PermissionRules = field(default_factory=PermissionRules)
    # What path rules are relative to.
    cwd: Path = field(default_factory=Path.cwd)
    # Where "always allow in this project" answers are saved; None: not offered.
    project_file: Path | None = None
    # The sandbox shell commands run in, if any. Commands inside it need no
    # approval; leaving it, or writing files beyond it, always does.
    sandbox: Sandbox | None = None
    # This session's answers about hosts for the sandbox's proxy: allowed or not.
    hosts: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bind(self.sandbox)

    def bind(self, sandbox: Sandbox | None) -> None:
        """Use this sandbox, and have its proxy follow the rules and answers here."""
        self.sandbox = sandbox
        if sandbox is not None:
            sandbox.decide = self.host_decision

    def host_decision(self, host: str) -> bool | None:
        """Whether sandboxed commands may reach `host`: a `net:` deny rule or a "no"
        says False, an answer or allow rule True; None leaves it to [sandbox] network."""
        def covered(kind: str) -> bool:
            return any(name == "net" and (not sep or wildcard(pattern.lower(), host))
                       for name, sep, pattern in (r.partition(":") for r in getattr(self.rules, kind)))
        if covered("deny"):
            return False
        if host in self.hosts:
            return self.hosts[host]
        return True if covered("allow") else None

    def answer_host(self, host: str, allow: bool, persist: bool = False) -> None:
        """Remember the user's answer about a host, for the project too if `persist`."""
        self.hosts[host] = allow
        key = f"net:{host}"
        if allow and persist and self.project_file is not None and key not in self.rules.allow:
            self.rules.allow.append(key)
            saved = load_project_rules(self.project_file)
            saved.allow.append(key)
            save_project_rules(self.project_file, saved)

    def blocked(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """Why the call may not run at all, or None."""
        if rule := self.matching("deny", tool, args):
            return (f"The user's permission rule `{rule}` refuses this call. Don't try to get "
                    "around it; find another way or ask the user.")
        if self.mode == "plan" and tool.edits_files:
            return ("Plan mode is on, so files can't be changed yet. Keep investigating, then "
                    "present your plan for the user to approve.")
        return None

    def must_ask(self, tool: Tool, args: dict[str, Any]) -> bool:
        """Ask whatever the mode, and don't remember the answer: an ask rule covers
        the call, or it writes a file the sandbox wouldn't let a command write."""
        return self.matching("ask", tool, args) is not None or self.writes_outside(tool, args)

    def needs_approval(self, tool: Tool, args: dict[str, Any]) -> bool:
        if tool.read_only:
            return False
        if self.escapes(tool, args):  # in every mode, auto too
            return self.mode == "plan" or (self.key(tool, args) not in self.allowed
                                           and self.matching("allow", tool, args) is None)
        if self.runs_sandboxed(tool, args):  # in plan mode too, where the sandbox is read-only
            return False
        if self.mode == "plan":
            return True
        if self.mode == "auto" or (self.mode == "accept_edits" and tool.edits_files):
            return False
        return (tool.permission_key(args) not in self.allowed
                and self.matching("allow", tool, args) is None)

    def escapes(self, tool: Tool, args: dict[str, Any]) -> bool:
        """The call asks to run outside the sandbox there is."""
        return self.sandbox is not None and tool.escapes_sandbox(args)

    def runs_sandboxed(self, tool: Tool, args: dict[str, Any]) -> bool:
        return self.sandbox is not None and tool.sandboxed and not tool.escapes_sandbox(args)

    def writes_outside(self, tool: Tool, args: dict[str, Any]) -> bool:
        """A file tool writes where a sandboxed command couldn't: outside the
        project, or into .git/hooks or .git/config."""
        return self.sandbox is not None and any(
            not self.sandbox.allows_write(self.cwd / Path(p).expanduser()) for p in tool.write_paths(args))

    def leaves_sandbox(self, tool: Tool, args: dict[str, Any]) -> bool:
        return self.escapes(tool, args) or self.writes_outside(tool, args)

    def key(self, tool: Tool, args: dict[str, Any]) -> str:
        """What "always allow" remembers for this call: `unsandboxed:<program>`
        for leaving the sandbox, so allowing a command inside it never lets it out."""
        key = tool.permission_key(args)
        return "unsandboxed:" + key.partition(":")[2] if self.escapes(tool, args) else key

    def remember(self, tool: Tool, args: dict[str, Any], persist: bool = False) -> None:
        key = self.key(tool, args)
        self.allowed.add(key)
        if persist and self.project_file is not None and key not in self.rules.allow:
            self.rules.allow.append(key)
            saved = load_project_rules(self.project_file)
            saved.allow.append(key)
            save_project_rules(self.project_file, saved)

    def matching(self, kind: Literal["allow", "deny", "ask"], tool: Tool,
                 args: dict[str, Any]) -> str | None:
        """The first rule of this kind that covers the call."""
        names = tool.rule_names(args)
        if self.escapes(tool, args):
            # Only an `unsandboxed:` rule lets a call out; deny and ask rules see it as itself too.
            names = ("unsandboxed",) if kind == "allow" else (*names, "unsandboxed")
        for rule in getattr(self.rules, kind):
            name, sep, pattern = rule.partition(":")
            if not any(wildcard(name, n) for n in names):
                continue
            if not sep or tool.matches_rule(pattern, args, self.cwd, allow=kind == "allow"):
                return rule
        return None

    def cycle(self) -> Mode:
        """Switch to the next mode (Shift+Tab) and return it."""
        modes = CYCLE + (("auto",) if self.allow_auto else ())
        i = modes.index(self.mode) if self.mode in modes else -1
        self.mode = modes[(i + 1) % len(modes)]
        return self.mode


PROJECT_RULES = "permissions.toml"


def project_rules_file(cwd: Path, home: Path | None = None) -> Path:
    return project_home(cwd, home) / PROJECT_RULES


def load_project_rules(path: Path) -> PermissionRules:
    """The rules saved for a project. Raises ConfigError if the file is malformed."""
    if not path.exists():
        return PermissionRules()
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    return parse_permission_rules(raw, source=str(path))


def save_project_rules(path: Path, rules: PermissionRules) -> None:
    lines = ["# wren's permission rules for this project; \"always in this project\" adds to allow.",
             "# Same format as [permissions] in ~/.wren/config.toml."]
    for kind in ("allow", "deny", "ask"):
        if items := getattr(rules, kind):
            # A JSON string is a valid TOML basic string.
            lines += [f"{kind} = ["] + [f"    {json.dumps(r, ensure_ascii=False)}," for r in items] + ["]"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
