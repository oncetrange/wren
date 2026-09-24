"""User configuration: model definitions and defaults.

Models are only read from the user-level config file, never from the
project directory: a cloned repo must not be able to point a model at its
own base_url and collect the API key named by `api_key_env`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from wren.llm.types import Usage

CONFIG_DIR = Path(os.environ.get("WREN_HOME", Path.home() / ".wren"))
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class Price:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None
    cache_write: float | None = None

    def cost(self, u: Usage) -> float:
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
    provider: str = "anthropic"
    base_url: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    # "api_key" sends X-Api-Key; "bearer" sends it as Authorization: Bearer as
    # well, which most Anthropic-compatible gateways expect.
    auth: Literal["api_key", "bearer"] = "api_key"
    max_tokens: int = 32000
    prompt_cache: bool = True
    # The request's `thinking` parameter: a type string such as "adaptive", or a
    # full table like {type = "enabled", budget_tokens = 16000}. None omits it.
    thinking: str | dict[str, Any] | None = None
    price: Price | None = None

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(
                f"model {self.name!r} needs an API key: set ${self.api_key_env}"
            )
        return key

    def cost(self, usage: Usage) -> float | None:
        return self.price.cost(usage) if self.price else None


BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "qwen": {
        "model": "qwen3-coder-plus",
        "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
        "api_key_env": "DASHSCOPE_API_KEY",
        "auth": "bearer",
        "max_tokens": 32000,
        "prompt_cache": False,
    },
    "kimi": {
        "model": "kimi-k2.7-code",
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
    "claude": {
        "model": "claude-opus-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 64000,
        "thinking": "adaptive",
        "price": {"input": 5.0, "output": 25.0},
    },
}


@dataclass
class Config:
    default_model: str = "qwen"
    models: dict[str, ModelConfig] = field(default_factory=dict)

    def model(self, name: str | None = None) -> ModelConfig:
        name = name or self.default_model
        if name not in self.models:
            known = ", ".join(sorted(self.models))
            raise ConfigError(f"unknown model {name!r} (configured: {known})")
        return self.models[name]


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
    )


def _model_config(name: str, spec: dict[str, Any]) -> ModelConfig:
    known = {f.name for f in fields(ModelConfig)} - {"name"}
    unknown = set(spec) - known
    if unknown:
        raise ConfigError(f"model {name!r}: unknown keys {sorted(unknown)}")
    if "model" not in spec:
        raise ConfigError(f"model {name!r}: missing 'model'")
    spec = dict(spec)
    if isinstance(spec.get("price"), dict):
        spec["price"] = Price(**spec["price"])
    return ModelConfig(name=name, **spec)
