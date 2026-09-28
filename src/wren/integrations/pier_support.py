"""Pier-independent logic behind the Pier/Harbor adapter (see pier_agent.py).

Kept free of Pier imports so it can be tested without installing Pier.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from urllib.parse import urlparse

from wren.config import ConfigError, ModelConfig, load_config

REPO_ARCHIVE = "https://github.com/oncetrange/wren/archive/{ref}.tar.gz"
WREN_BIN = "$HOME/.local/bin/wren"
UV_BIN = "$HOME/.local/bin/uv"
LOG_DIR = "/logs/agent"
RESULT_FILE = "wren-result.json"
STDERR_FILE = "wren.log"
WREN_HOME = f"{LOG_DIR}/wren"  # session logs land here and are synced back with the agent logs


def resolve_model(model_name: str | None) -> ModelConfig:
    """Map a Pier model name ("moonshot/kimi-k2.7-code") to a builtin wren model.

    Only builtins are considered: the container has no user config file.
    """
    if not model_name:
        raise ConfigError("wren needs a model: pass -m, e.g. -m moonshot/kimi-k2.7-code")
    return load_config(Path("/nonexistent/config.toml")).model(model_name)


def model_hosts(model: ModelConfig) -> list[str]:
    """Domains the agent must reach at run time (for air-gapped tasks)."""
    default = "api.openai.com" if model.provider == "openai" else "api.anthropic.com"
    host = urlparse(model.base_url).hostname if model.base_url else default
    return [host] if host else []


def install_commands(ref: str) -> list[tuple[str, str]]:
    """(user, shell command) install steps for a Debian/Ubuntu or Alpine image."""
    return [
        ("root",
         "command -v curl >/dev/null 2>&1 || "
         "{ (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates) "
         "|| apk add --no-cache curl ca-certificates; }"),
        ("agent",
         "set -euo pipefail; "
         "curl -LsSf https://astral.sh/uv/install.sh | sh; "
         f"{UV_BIN} tool install --force --python 3.12 {shlex.quote(REPO_ARCHIVE.format(ref=ref))}; "
         f"{WREN_BIN} --version"),
    ]


def run_command(instruction: str, model: ModelConfig, extra_flags: str = "") -> str:
    """Run wren once. Exits 0 whenever a result was produced, so the verifier
    still grades partial work after e.g. hitting max turns; the end state is in
    the result JSON. Fails only if wren couldn't start at all."""
    wren = (f"{WREN_BIN} -p {shlex.quote(instruction)} -m {shlex.quote(model.name)} "
            f"--yolo --no-checkpoints --output-format json {extra_flags}").strip()
    return (
        f"mkdir -p {WREN_HOME}; "
        f"{wren} > {LOG_DIR}/{RESULT_FILE} 2> {LOG_DIR}/{STDERR_FILE} < /dev/null; "
        "code=$?; "
        f"if [ -s {LOG_DIR}/{RESULT_FILE} ]; then cat {LOG_DIR}/{RESULT_FILE}; exit 0; fi; "
        f"tail -n 50 {LOG_DIR}/{STDERR_FILE} >&2; exit $code"
    )


def read_run(logs_dir: Path) -> dict | None:
    """The result JSON plus figures derived from the session log, or None."""
    path = logs_dir / RESULT_FILE
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    peak, compactions = 0, 0
    for session in (logs_dir / "wren" / "sessions").glob("*.jsonl"):
        for line in session.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "usage":
                u = e["usage"]
                peak = max(peak, u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"])
            elif e.get("kind") == "compact":
                compactions += 1
    result["peak_context_tokens"] = peak
    result["compactions"] = compactions
    return result
