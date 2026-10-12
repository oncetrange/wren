"""Run the model's shell commands in an OS sandbox.

Inside, a command can read almost anything but write only to the working
directory and the temp directories, can't read where credentials live, and
has no network beyond localhost, or only to the hosts on an allowlist, through
a proxy (netproxy.py). Inside the working directory, `.git/hooks`
and `.git/config` stay read-only: either could make git run code later,
outside the sandbox. API keys are left out of the environment.

macOS uses Seatbelt (`sandbox-exec`), Linux bubblewrap (`bwrap`). When neither
works, there is no sandbox: wren says so, and commands ask for approval as
before; nothing runs unsandboxed without the user knowing.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from wren.netproxy import Proxy

# Where credentials usually live, relative to home: never readable inside.
SECRET_PATHS = (
    ".ssh", ".gnupg", ".aws", ".azure", ".config/gcloud", ".kube", ".docker/config.json",
    ".config/gh", ".netrc", ".git-credentials", ".npmrc", ".pypirc", ".cargo/credentials.toml",
    ".wren/env", "Library/Keychains",
)
# Environment variables left out inside, besides each model's api_key_env.
_SECRET_ENV = re.compile(r"(API_KEY|_TOKEN|_SECRET|PASSWORD)$|^(AWS_SECRET|GITHUB_TOKEN|GH_TOKEN)", re.I)
# Output that suggests the sandbox, not the command, is what failed.
_BLOCKED = re.compile(
    r"Operation not permitted|Read-only file system|Permission denied|"
    r"Could not resolve host|Couldn't connect|Network is unreachable|Temporary failure in name resolution|"
    r"Name or service not known|nodename nor servname|Failed to establish a new connection|"
    r"ConnectionRefused|getaddrinfo|ENOTFOUND|EAI_AGAIN|connection refused|"
    r"403 Forbidden|ProxyError|Tunnel connection failed|CONNECT tunnel failed",
    re.I)


@dataclass
class SandboxConfig:
    """[sandbox] in ~/.wren/config.toml."""

    enabled: bool = False
    # "deny": localhost only; "allow": any; or hosts to allow through a proxy
    # ("pypi.org", "*.example.com", "@packages" for the package registries).
    network: str | list[str] = "deny"
    # More places commands may write (e.g. a package cache), ~ allowed.
    allow_write: list[str] = field(default_factory=list)
    # More places commands may not read.
    deny_read: list[str] = field(default_factory=list)


class SandboxUnavailable(Exception):
    pass


class Sandbox:
    def __init__(self, cwd: Path, config: SandboxConfig, secret_env: tuple[str, ...] = (),
                 home: Path | None = None, platform: str = sys.platform):
        self.config, self.platform = config, platform
        self.home = _real(home or Path.home())
        self.cwd = _real(cwd)
        temp = {_real(Path(tempfile.gettempdir())), _real(Path("/tmp"))}
        # Writable even read-only (plan mode, read-only subagents): scratch space and caches.
        self.scratch = [*sorted(temp), *(_real(Path(p).expanduser()) for p in config.allow_write)]
        self.protected = [self.cwd / ".git" / "hooks", self.cwd / ".git" / "config"]
        self.hidden = [self.home / p for p in SECRET_PATHS] + [
            _real(Path(p).expanduser()) for p in config.deny_read]
        self.secret_env = set(secret_env)
        self._proxy: Proxy | None = None
        self._proxy_lock = threading.Lock()

    def writable(self, read_only: bool = False) -> list[Path]:
        return self.scratch if read_only else [self.cwd, *self.scratch]

    def read_only_paths(self, read_only: bool = False) -> list[Path]:
        """Never writable, even inside a writable directory (the project may be in /tmp)."""
        return [self.cwd] if read_only else self.protected

    def allows_write(self, path: Path) -> bool:
        """Whether a command in the sandbox could write `path` (for the file tools,
        which run outside it)."""
        real = _real(path)
        if any(real.is_relative_to(p) for p in self.protected):
            return False
        return any(real.is_relative_to(p) for p in self.writable())

    @property
    def network(self) -> bool:
        """Any network at all."""
        return self.config.network == "allow"

    @property
    def allowed_hosts(self) -> list[str]:
        """The allowlist, if the network goes through the proxy."""
        return self.config.network if isinstance(self.config.network, list) else []

    @property
    def running_proxy(self) -> Proxy | None:
        """The proxy if it has started (without starting it)."""
        return self._proxy

    def proxy(self) -> Proxy | None:
        """The proxy to the allowed hosts, started on first use."""
        if not self.allowed_hosts:
            return None
        with self._proxy_lock:
            if self._proxy is None:
                self._proxy = Proxy(self.allowed_hosts, unix_socket=self.platform != "darwin")
                atexit.register(self._proxy.close)
            return self._proxy

    def check(self) -> None:
        """Raise SandboxUnavailable if commands can't be sandboxed here."""
        if reason := _unavailable(self.platform):
            raise SandboxUnavailable(reason)

    def wrap(self, argv: list[str], cwd: Path, read_only: bool = False) -> list[str]:
        """The command line that runs `argv` inside the sandbox, from `cwd`.
        `read_only`: not even the project is writable, only scratch space."""
        if self.platform == "darwin":
            return ["sandbox-exec", "-p", self.seatbelt_profile(read_only), *argv]
        if (proxy := self.proxy()) is not None:
            # The command's network namespace can't reach the proxy: bridge to its socket.
            argv = [sys.executable, "-I", "-m", "wren.netbridge", str(proxy.unix_path),
                    str(proxy.port), "--", *argv]
        return self.bwrap_args(cwd, read_only) + ["--", *argv]

    def env(self, env: dict[str, str]) -> dict[str, str]:
        """The environment without API keys and tokens, and with the proxy, if any."""
        env = {k: v for k, v in env.items() if k not in self.secret_env and not _SECRET_ENV.search(k)}
        if (proxy := self.proxy()) is not None:
            env.update(proxy.env())
        return env

    def describe(self, read_only: bool = False) -> str:
        """The rules, for the model and for error hints."""
        if self.network:
            net = "any network"
        elif hosts := self.allowed_hosts:
            shown = ", ".join("the package registries (PyPI, npm, crates.io, Go, RubyGems, Maven)"
                              if h == "@packages" else h for h in hosts)
            net = (f"network only to {shown}, through a proxy set in HTTP(S)_PROXY (other hosts "
                   "are refused)")
        elif self.platform == "darwin":
            net = "no network except localhost"
        else:  # each command gets a network namespace of its own
            net = "no network (localhost reaches only servers started by the same command)"
        where = ", ".join(map(str, self.writable(read_only)))
        guarded = "" if read_only else " (not in .git/hooks or .git/config)"
        return (f"commands may write only in {where}{guarded}, can't read credentials such as "
                f"~/.ssh, and have {net}")

    def blocked_hint(self, output: str, read_only: bool = False, can_leave: bool = True,
                     since: float | None = None) -> str | None:
        """A note for the model when a failure looks like the sandbox's doing; `since`
        (time.monotonic()) is when the command started, to name the hosts refused meanwhile."""
        refused = self._proxy.refused_since(since) if self._proxy and since is not None else []
        if not refused and not _BLOCKED.search(output):
            return None
        way_out = (" If the task really needs this, run the command again with sandbox: false; the "
                   "user will be asked to approve it." if can_leave else "")
        which = f" The proxy refused {', '.join(refused)}." if refused else ""
        return f"[This may be the sandbox: {self.describe(read_only)}.{which}{way_out}]"

    # --- per platform -----------------------------------------------------

    def seatbelt_profile(self, read_only: bool = False) -> str:
        def paths(kind: str, items: list[Path]) -> str:
            return " ".join(f'({kind} "{_sb_quote(str(p))}")' for p in items)

        lines = [
            "(version 1)",
            "(allow default)",
            "(deny file-write*)",
            f'(allow file-write* {paths("subpath", self.writable(read_only))} (subpath "/dev"))',
            f"(deny file-write* {paths('subpath', self.read_only_paths(read_only))})",
            f"(deny file-read* file-write* {paths('subpath', self.hidden)})",
        ]
        if not self.network:
            lines += [
                "(deny network*)",
                '(allow network* (remote ip "localhost:*"))',
                '(allow network-bind (local ip "*:*"))',
                '(allow network-inbound (local ip "localhost:*"))',
                "(allow network* (remote unix-socket))",
            ]
        return "\n".join(lines) + "\n"

    def bwrap_args(self, cwd: Path, read_only: bool = False) -> list[str]:
        args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--unshare-pid", "--die-with-parent", "--new-session"]
        for p in self.writable(read_only):
            if p.exists():
                args += ["--bind", str(p), str(p)]
        for p in self.read_only_paths(read_only):
            if p.exists():
                args += ["--ro-bind", str(p), str(p)]
        for p in self.hidden:
            if p.is_dir():
                args += ["--tmpfs", str(p)]
            elif p.exists():
                args += ["--ro-bind", "/dev/null", str(p)]
        if not self.network:
            args.append("--unshare-net")
        return args + ["--chdir", str(cwd)]


def _real(p: Path) -> Path:
    return Path(os.path.realpath(p))


def _sb_quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


@cache
def _unavailable(platform: str) -> str | None:
    """Why there is no sandbox on this machine, or None. Tried once per process."""
    if platform == "darwin":
        tool, trial = "sandbox-exec", ["sandbox-exec", "-p", "(version 1)(allow default)", "true"]
    elif platform.startswith("linux"):
        tool = "bwrap"
        trial = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-pid", "--unshare-net", "true"]
    else:
        return f"no sandbox support on {platform}"
    if shutil.which(tool) is None:
        hint = " (install bubblewrap)" if tool == "bwrap" else ""
        return f"{tool} is not installed{hint}"
    try:
        proc = subprocess.run(trial, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{tool} failed: {e}"
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"
        hint = (" (unprivileged user namespaces may be disabled, e.g. by AppArmor on Ubuntu 24.04)"
                if tool == "bwrap" else "")
        return f"{tool} doesn't work here: {detail}{hint}"
    return None
