"""User configuration: model definitions and defaults.

Models are only read from the user-level config file, never from the
project directory: a cloned repo must not be able to point a model at its
own base_url and collect the API key named by `api_key_env`.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from wren.llm.types import Usage

CONFIG_DIR = Path(os.environ.get("WREN_HOME", Path.home() / ".wren"))
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class Price:
    """USD per million tokens.

    Providers that price by prompt length list `tiers`; each request is billed
    at the first tier whose `up_to` covers its prompt (the last tier otherwise).
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None
    cache_write: float | None = None
    up_to: int | None = None
    tiers: list[Price] | None = None

    def cost(self, u: Usage) -> float:
        if self.tiers:
            tier = next((t for t in self.tiers if t.up_to is None or u.context_tokens <= t.up_to),
                        self.tiers[-1])
            return tier.cost(u)
        cache_read = self.input * 0.1 if self.cache_read is None else self.cache_read
        cache_write = self.input * 1.25 if self.cache_write is None else self.cache_write
        return (
            u.input_tokens * self.input
            + u.output_tokens * self.output
            + u.cache_read_tokens * cache_read
            + u.cache_write_tokens * cache_write
        ) / 1_000_000


@dataclass
class ModelConfig:
    name: str
    model: str
    # "anthropic": the Messages API; "openai": OpenAI-compatible chat completions.
    provider: Literal["anthropic", "openai"] = "anthropic"
    base_url: str | None = None
    # The environment variable holding the key; default ANTHROPIC_API_KEY or
    # OPENAI_API_KEY by provider. "" for endpoints without keys (local servers).
    api_key_env: str | None = None
    # "api_key" sends X-Api-Key; "bearer" sends it as Authorization: Bearer as
    # well, which most Anthropic-compatible gateways expect.
    auth: Literal["api_key", "bearer"] = "api_key"
    max_tokens: int = 32000
    # Total tokens the model accepts.
    context_window: int = 200_000
    # Context management thresholds, in prompt tokens: mask old tool outputs
    # past `mask_at` (0 disables), summarize past `compact_at` (default: 80% of
    # the window, at most 200k; long prompts cost more and degrade quality).
    mask_at: int = 40_000
    compact_at: int | None = None
    prompt_cache: bool = True
    # The request's `thinking` parameter: a type string such as "adaptive", or a
    # full table like {type = "enabled", budget_tokens = 16000}. None omits it.
    thinking: str | dict[str, Any] | None = None
    price: Price | None = None
    # OpenAI provider only. Sent as `reasoning_effort` ("low", "medium", "high").
    reasoning_effort: str | None = None
    # Extra request fields for vendor-specific options (e.g. {enable_thinking = true}).
    extra_body: dict[str, Any] | None = None
    # Send the model's reasoning back with its tool calls, as `reasoning_content`
    # (DeepSeek and Kimi require this in thinking mode; others reject or ignore it).
    replay_reasoning: bool = False
    # "max_tokens", or "max_completion_tokens" (OpenAI's own newer models);
    # default: the latter for api.openai.com, the former elsewhere.
    max_tokens_param: str | None = None

    def ignored_options(self) -> list[str]:
        """Options set for this model that its provider doesn't use."""
        if self.provider == "openai":
            unused = {"thinking": self.thinking is not None, "auth": self.auth != "api_key"}
        else:
            unused = {"reasoning_effort": self.reasoning_effort is not None,
                      "extra_body": self.extra_body is not None,
                      "replay_reasoning": self.replay_reasoning,
                      "max_tokens_param": self.max_tokens_param is not None}
        return [name for name, is_set in unused.items() if is_set]

    @property
    def key_env(self) -> str:
        if self.api_key_env is not None:
            return self.api_key_env
        return "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"

    def api_key(self) -> str:
        """The API key ("" for a model configured without one)."""
        if self.key_env == "":
            return ""
        key = os.environ.get(self.key_env)
        if not key:
            raise ConfigError(
                f"model {self.name!r} needs an API key: set ${self.key_env}"
            )
        return key

    def cost(self, usage: Usage) -> float | None:
        return self.price.cost(usage) if self.price else None

    @property
    def compact_threshold(self) -> int:
        return self.compact_at or min(int(self.context_window * 0.8), 200_000)


BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "qwen": {
        "model": "qwen3-coder-plus",
        "context_window": 1_000_000,
        "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
        "api_key_env": "DASHSCOPE_API_KEY",
        "auth": "bearer",
        "max_tokens": 32000,
        "prompt_cache": False,
        # International pricing; only the <=32k-prompt tier is published on
        # qwencloud.com, longer prompts are billed at it until tiers are added.
        "price": {"tiers": [{"up_to": 32_000, "input": 1.0, "output": 5.0, "cache_read": 0.2}]},
    },
    "kimi": {
        "model": "kimi-k2.7-code",
        "context_window": 262_144,
        "base_url": "https://api.moonshot.cn/anthropic",
        "api_key_env": "MOONSHOT_API_KEY",
        "auth": "bearer",
        "max_tokens": 32768,
        # Moonshot caches repeated prefixes automatically; no breakpoints needed.
        "prompt_cache": False,
        # K2.7 Code rejects requests without thinking enabled.
        "thinking": {"type": "enabled", "budget_tokens": 16000},
        "price": {"input": 0.95, "output": 4.0, "cache_read": 0.19, "cache_write": 0.95},
    },
    # The same model as "qwen" through DashScope's OpenAI-compatible endpoint,
    # for comparing the two protocols.
    "qwen-openai": {
        "provider": "openai",
        "model": "qwen3-coder-plus",
        "context_window": 1_000_000,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "max_tokens": 32000,
        "price": {"tiers": [{"up_to": 32_000, "input": 1.0, "output": 5.0, "cache_read": 0.2}]},
    },
    # No price: set one in config.toml ([models.deepseek.price]) to see costs.
    "deepseek": {
        "provider": "openai",
        "model": "deepseek-chat",
        "context_window": 128_000,
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_tokens": 8192,
    },
    "claude": {
        "model": "claude-opus-5",
        "context_window": 1_000_000,
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 64000,
        "thinking": "adaptive",
        "price": {"input": 5.0, "output": 25.0},
    },
}


HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop",
               "Notification", "SessionEnd")


@dataclass
class HookSpec:
    """A shell command run on a lifecycle event (see agent/shell_hooks.py)."""

    event: str
    command: str
    # Regex matched against the tool name (PreToolUse / PostToolUse); None matches all.
    matcher: str | None = None
    timeout: int = 60
    # Where it was configured: "user" or the project hooks file.
    source: str = "user"


@dataclass
class McpServerConfig:
    """An MCP server to connect to: a local command (stdio) or a URL (Streamable HTTP)."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # Seconds a tool call may take.
    timeout: float = 120
    # Where it was configured: "user" or the project's .mcp.json.
    source: str = "user"

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"


@dataclass
class Config:
    default_model: str = "kimi"
    models: dict[str, ModelConfig] = field(default_factory=dict)
    hooks: list[HookSpec] = field(default_factory=list)
    mcp: list[McpServerConfig] = field(default_factory=list)

    def model(self, name: str | None = None) -> ModelConfig:
        """Look a model up by its config name, or by its model id with an
        optional provider prefix ("kimi-k2.7-code", "moonshot/kimi-k2.7-code")."""
        name = name or os.environ.get("WREN_MODEL") or self.default_model
        if name in self.models:
            return self.models[name]
        model_id = name.split("/", 1)[-1]
        for m in self.models.values():
            if m.model == model_id:
                return m
        known = ", ".join(sorted(self.models))
        raise ConfigError(f"unknown model {name!r} (configured: {known})")


class ConfigError(Exception):
    pass


def load_config(path: Path = CONFIG_FILE) -> Config:
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            raw = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from e

    model_defs = {name: dict(spec) for name, spec in BUILTIN_MODELS.items()}
    for name, spec in raw.get("models", {}).items():
        # A user entry overrides individual fields of a builtin of the same name.
        model_defs[name] = {**model_defs.get(name, {}), **spec}

    return Config(
        default_model=raw.get("default_model", Config.default_model),
        models={name: _model_config(name, spec) for name, spec in model_defs.items()},
        hooks=parse_hooks(raw.get("hooks", {}), source="user"),
        mcp=parse_mcp_servers(raw.get("mcp", {}), source="user"),
    )


_MCP_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_MCP_KEYS = {"type", "command", "args", "env", "url", "headers", "timeout", "enabled"}
_ENV_REF = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def parse_mcp_servers(raw: dict[str, Any], source: str) -> list[McpServerConfig]:
    """[mcp.<name>] tables, or .mcp.json's "mcpServers", -> configs.

    Values may reference environment variables as ${VAR} or ${VAR:-default},
    so tokens can stay out of the file."""
    servers = []
    for name, spec in raw.items():
        where = f"mcp server {name!r} ({source})"
        if not _MCP_NAME.match(name):
            raise ConfigError(f"{where}: names may only use letters, digits, '-' and '_'")
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: must be a table")
        if unknown := set(spec) - _MCP_KEYS:
            raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
        if spec.get("enabled", True) is False:
            continue
        kind = spec.get("type", "http" if "url" in spec else "stdio")
        if kind == "sse":
            raise ConfigError(f"{where}: the legacy SSE transport isn't supported; use its "
                              "Streamable HTTP URL (type = \"http\")")
        if kind not in ("stdio", "http"):
            raise ConfigError(f"{where}: type must be 'stdio' or 'http', got {kind!r}")
        if kind == "stdio" and not isinstance(spec.get("command"), str):
            raise ConfigError(f"{where}: needs a 'command' (or a 'url')")
        if kind == "http" and not isinstance(spec.get("url"), str):
            raise ConfigError(f"{where}: needs a 'url'")
        timeout = spec.get("timeout", McpServerConfig.timeout)
        if not isinstance(timeout, int | float) or isinstance(timeout, bool) or timeout <= 0:
            raise ConfigError(f"{where}: 'timeout' must be a positive number of seconds")
        servers.append(McpServerConfig(
            name=name,
            command=_expand(spec["command"]) if kind == "stdio" else None,
            args=[_expand(str(a)) for a in spec.get("args", [])] if kind == "stdio" else [],
            env={k: _expand(str(v)) for k, v in spec.get("env", {}).items()},
            url=_expand(spec["url"]) if kind == "http" else None,
            headers={k: _expand(str(v)) for k, v in spec.get("headers", {}).items()},
            timeout=float(timeout),
            source=source,
        ))
    return servers


def _expand(value: str) -> str:
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)


def parse_hooks(raw: dict[str, Any], source: str) -> list[HookSpec]:
    """[[hooks.<Event>]] tables -> HookSpecs, validated."""
    specs = []
    for event, entries in raw.items():
        if event not in HOOK_EVENTS:
            raise ConfigError(f"hooks: unknown event {event!r} (one of {', '.join(HOOK_EVENTS)})")
        for entry in entries if isinstance(entries, list) else [entries]:
            unknown = set(entry) - {"command", "matcher", "timeout"}
            if unknown or not isinstance(entry.get("command"), str):
                raise ConfigError(f"hooks.{event}: each hook needs a 'command' "
                                  f"(and optionally 'matcher', 'timeout'); got {sorted(entry)}")
            if entry.get("matcher"):
                try:
                    re.compile(entry["matcher"])
                except re.error as e:
                    raise ConfigError(f"hooks.{event}: bad matcher {entry['matcher']!r}: {e}") from e
            specs.append(HookSpec(event, entry["command"], entry.get("matcher"),
                                  int(entry.get("timeout", 60)), source))
    return specs


def _model_config(name: str, spec: dict[str, Any]) -> ModelConfig:
    known = {f.name for f in fields(ModelConfig)} - {"name"}
    unknown = set(spec) - known
    if unknown:
        raise ConfigError(f"model {name!r}: unknown keys {sorted(unknown)}")
    if "model" not in spec:
        raise ConfigError(f"model {name!r}: missing 'model'")
    if spec.get("provider", "anthropic") not in ("anthropic", "openai"):
        raise ConfigError(f"model {name!r}: provider must be 'anthropic' or 'openai', "
                          f"got {spec['provider']!r}")
    spec = dict(spec)
    if isinstance(spec.get("price"), dict):
        spec["price"] = _price(spec["price"])
    return ModelConfig(name=name, **spec)


def _price(spec: dict[str, Any]) -> Price:
    spec = dict(spec)
    if "tiers" in spec:
        spec["tiers"] = [_price(t) for t in spec["tiers"]]
    return Price(**spec)
