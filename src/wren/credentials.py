"""API keys kept in ~/.wren/env, for people who'd rather not export them.

The file holds KEY=value lines (comments and `export` allowed), readable only
by its owner. wren loads it at startup; a variable already set in the
environment wins, so a shell export still overrides the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from wren.config import CONFIG_DIR

ENV_FILE_NAME = "env"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def env_file(home: Path | None = None) -> Path:
    return (home or CONFIG_DIR) / ENV_FILE_NAME


def read_env_file(home: Path | None = None) -> dict[str, str]:
    try:
        text = env_file(home).read_text()
    except OSError:
        return {}
    env = {}
    for line in text.splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def load_env_file(home: Path | None = None) -> list[str]:
    """Put the file's variables into os.environ unless already set. Returns the names loaded."""
    loaded = []
    for key, value in read_env_file(home).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def save_key(name: str, value: str, home: Path | None = None) -> Path:
    """Set NAME=value in the file (replacing an earlier line for it), owner-only,
    and in this process's environment."""
    if not _NAME.match(name):
        raise ValueError(f"not an environment variable name: {name!r}")
    if "\n" in value:
        raise ValueError("the value can't contain a newline")
    path = env_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(name)}\s*=")
    lines = [line for line in lines if not pattern.match(line)] + [f"{name}={value}"]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)  # also when the file existed with looser permissions
    os.environ[name] = value
    return path
