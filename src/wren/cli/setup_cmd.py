"""`wren setup`: choose a model, save its API key, check that it answers.

It also runs by itself the first time wren starts interactively with no model
it can use. Keys go to ~/.wren/env (owner-only, see credentials.py), new
models and the default to ~/.wren/config.toml (see config_edit.py).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import prompt
from rich.console import Console
from rich.markup import escape

from wren.cli.pickers import pick
from wren.config import BUILTIN_MODELS, CONFIG_FILE, Config, ConfigError, ModelConfig, load_config
from wren.config_edit import add_model, set_default_model
from wren.credentials import env_file, save_key
from wren.llm.types import Completed, LLMError, Message, TextBlock

Verify = Callable[[ModelConfig], tuple[bool, str]]
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
CUSTOM = {
    "anthropic": "Another Anthropic-compatible endpoint (base URL + key)",
    "openai": "Another OpenAI-compatible endpoint (base URL + key)",
    "ollama": "A local model with Ollama (no key)",
}


class NoUsableModel(Exception):
    pass


def default_model(config: Config) -> tuple[ModelConfig, str | None]:
    """The default model, or, if its key is missing, the first one whose key is
    there (your own models before the builtins), with a note saying so. Raises
    NoUsableModel when no model has a key."""
    model = config.model(None)
    if model.has_key():
        return model, None
    candidates = sorted(config.models.values(), key=lambda m: m.name in BUILTIN_MODELS)
    usable = next((m for m in candidates if m.has_key()), None)
    if usable is None:
        raise NoUsableModel(no_model_message(config))
    return usable, (f"default model {model.name!r} has no API key (${model.key_env}); using "
                    f"{usable.name!r}. `wren setup` changes the default.")


def no_model_message(config: Config) -> str:
    by_key: dict[str, list[str]] = {}
    for m in config.models.values():
        if m.key_env:
            by_key.setdefault(m.key_env, []).append(m.name)
    keys = ", ".join(f"${k} ({', '.join(names)})" for k, names in by_key.items())
    return f"no model has an API key yet. Run `wren setup`, or set one of: {keys}"


def verify_model(model: ModelConfig) -> tuple[bool, str]:
    """One tiny request: does the model answer with this key?"""
    from wren.llm.factory import create_provider

    start = time.monotonic()
    try:
        provider = create_provider(model)
        response = None
        for event in provider.stream(system="This is a connection check.",
                                     messages=[Message("user", [TextBlock("Reply with OK.")])], tools=[]):
            if isinstance(event, Completed):
                response = event.response
    except (LLMError, ConfigError) as e:
        return False, str(e)
    if response is None:
        return False, "no response"
    return True, f"answered in {time.monotonic() - start:.1f}s"


def setup_main(argv: list[str], console: Console | None = None) -> int:
    console = console or Console(highlight=False)
    if argv and argv[0] in ("-h", "--help"):
        console.print("usage: wren setup\n\nChoose a model, save its API key and check it works.")
        return 0
    try:
        return 0 if run_setup(console) else 1
    except ConfigError as e:
        console.print(f"[bold red]error:[/] {escape(str(e))}")
        return 1


def run_setup(console: Console, config_file: Path = CONFIG_FILE, home: Path | None = None,
              verify: Verify = verify_model) -> str | None:
    """The wizard. Returns the model name it set up, or None if cancelled."""
    config = load_config(config_file)
    console.print("[bold]Set up a model for wren[/]  [dim](Esc cancels)[/]")
    choice = pick("Which model?", setup_options(config))
    if choice is None:
        return None
    if choice in CUSTOM:
        name = _custom_model(console, config, choice, config_file, home)
        if name is None:
            return None
    else:
        name = choice
        model = config.models[name]
        if not model.has_key() and not _ask_key(console, model.key_env, home):
            return None
    while True:
        model = load_config(config_file).models[name]
        with console.status(f"checking {name} ({model.model})…"):
            ok, detail = verify(model)
        if ok:
            console.print(f"[green]✓[/] {escape(name)} works: {escape(detail)}")
            break
        console.print(f"[red]✗[/] {escape(name)} didn't answer: {escape(detail)}")
        options = [("retry", "Try again"), ("keep", "Keep it anyway"), ("cancel", "Cancel")]
        if model.key_env:
            options.insert(1, ("key", f"Enter ${model.key_env} again"))
        action = pick("What now?", options, default="retry")
        if action == "key":
            if not _ask_key(console, model.key_env, home):
                return None
        elif action == "keep":
            break
        elif action != "retry":
            return None
    if load_config(config_file).default_model != name:
        set_default_model(name, config_file)
    console.print(f"[dim]default model: {name} (set in {config_file})[/]")
    return name


def setup_options(config: Config) -> list[tuple[str, str]]:
    rows = []
    for m in config.models.values():
        where = m.base_url.split("//")[-1].split("/")[0] if m.base_url else (
            "api.openai.com" if m.provider == "openai" else "api.anthropic.com")
        key = "key found" if m.has_key() else (f"needs ${m.key_env}" if m.key_env else "no key needed")
        rows.append((m.name, f"{m.name} · {m.model} · {where} · {key}"))
    return rows + list(CUSTOM.items())


def _ask_key(console: Console, env_name: str, home: Path | None) -> bool:
    console.print(f"Paste the API key for [bold]${env_name}[/] (input is hidden; it is saved to "
                  f"{env_file(home)}, readable only by you)")
    try:
        key = prompt("  key: ", is_password=True).strip()
    except (KeyboardInterrupt, EOFError):
        return False
    if not key:
        return False
    path = save_key(env_name, key, home)
    console.print(f"[dim]saved ${env_name} to {path}[/]")
    return True


def _ask(label: str, default: str = "") -> str | None:
    try:
        text = prompt(f"  {label}{f' [{default}]' if default else ''}: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    return text or default


def _custom_model(console: Console, config: Config, kind: str, config_file: Path,
                  home: Path | None) -> str | None:
    console.print("[dim]A name for it in wren, the endpoint and the model id the endpoint expects.[/]")
    suggested = "local" if kind == "ollama" else "mymodel"
    while True:
        name = _ask("name", suggested)
        if name is None:
            return None
        if not NAME_RE.match(name):
            console.print("[yellow]letters, digits, - and _ only[/]")
        elif name in config.models:
            console.print(f"[yellow]{name!r} already exists; pick another name[/]")
        else:
            break
    default_url = {"ollama": "http://localhost:11434/v1", "openai": "https://api.openai.com/v1"}.get(kind, "")
    base_url = _ask("base URL", default_url)
    model_id = _ask("model id", "qwen3-coder:30b" if kind == "ollama" else "")
    if not base_url or not model_id:
        return None
    fields: dict = {"provider": "anthropic" if kind == "anthropic" else "openai", "model": model_id,
                    "base_url": base_url}
    if kind == "ollama":
        fields["api_key_env"] = ""
    else:
        env_name = _ask("environment variable for its key", re.sub(r"\W", "_", name).upper() + "_API_KEY")
        if not env_name:
            return None
        fields["api_key_env"] = env_name
        if not os.environ.get(env_name) and not _ask_key(console, env_name, home):
            return None
    add_model(name, fields, config_file)
    console.print(f"[dim]added [models.{name}] to {config_file}; edit it there for prices, context "
                  "window and other options[/]")
    return name
