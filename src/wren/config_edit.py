"""Small, careful edits to the user's config.toml (made by `wren setup`).

tomllib only reads, so these edit the text: they add a model table at the end
and set the top-level default_model, leaving everything else (comments
included) as it was. The result is parsed before it is written.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from wren.config import CONFIG_FILE, ConfigError


def toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are valid TOML basic strings
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{toml_key(k)} = {toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"can't write {value!r} as TOML")


def model_table(name: str, fields: dict[str, Any]) -> str:
    lines = [f"[models.{toml_key(name)}]"]
    lines += [f"{toml_key(k)} = {toml_value(v)}" for k, v in fields.items() if v is not None]
    return "\n".join(lines) + "\n"


def add_model(name: str, fields: dict[str, Any], path: Path = CONFIG_FILE) -> None:
    text = path.read_text() if path.exists() else ""
    if name in _parse(text, path).get("models", {}):
        raise ConfigError(f"{path} already defines a model named {name!r}")
    _write(path, text.rstrip("\n") + ("\n\n" if text.strip() else "") + model_table(name, fields))


def set_default_model(name: str, path: Path = CONFIG_FILE) -> None:
    text = path.read_text() if path.exists() else ""
    line = f"default_model = {toml_value(name)}"
    lines = text.splitlines()
    first_table = next((i for i, l in enumerate(lines) if l.lstrip().startswith("[")), len(lines))
    for i in range(first_table):
        if re.match(r"\s*default_model\s*=", lines[i]):
            lines[i] = line
            break
    else:
        lines.insert(0, line)
    _write(path, "\n".join(lines).rstrip("\n") + "\n")


def _parse(text: str, path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None


def _write(path: Path, text: str) -> None:
    _parse(text, path)  # never write something that doesn't load
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
