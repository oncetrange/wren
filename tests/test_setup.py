"""First-run setup: saved keys, config edits, picking a usable model, and the wizard."""

import os
import stat
import subprocess
import sys

import pytest
from fake_anthropic import FakeAnthropic, text_turn
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from wren.cli.setup_cmd import CUSTOM, NoUsableModel, default_model, run_setup, setup_options, verify_model
from wren.config import ConfigError, load_config
from wren.config_edit import add_model, set_default_model
from wren.credentials import load_env_file, read_env_file, save_key

KEYS = ["DASHSCOPE_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "MINE_API_KEY", "WREN_MODEL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No real keys leak in, and keys the code saves don't leak out."""
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k not in KEYS})


# --- credentials and config edits ----------------------------------------------

def test_save_key(tmp_path):
    path = save_key("MINE_API_KEY", "first", tmp_path)
    save_key("OTHER", "x", tmp_path)
    save_key("MINE_API_KEY", "second", tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == "OTHER=x\nMINE_API_KEY=second\n"
    assert os.environ["MINE_API_KEY"] == "second"
    with pytest.raises(ValueError):
        save_key("bad name", "x", tmp_path)


def test_env_file_never_overrides_the_shell(tmp_path):
    (tmp_path / "env").write_text("# keys\nexport MINE_API_KEY='from-file'\nDEEPSEEK_API_KEY=d\n")
    os.environ["MINE_API_KEY"] = "from-shell"
    assert load_env_file(tmp_path) == ["DEEPSEEK_API_KEY"]
    assert os.environ["MINE_API_KEY"] == "from-shell" and os.environ["DEEPSEEK_API_KEY"] == "d"
    assert read_env_file(tmp_path)["MINE_API_KEY"] == "from-file"


def test_config_edits_keep_what_was_there(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("# my settings\n\n[models.kimi]  # override\nmax_tokens = 1000\n")
    set_default_model("mine", path)
    add_model("mine", {"provider": "openai", "model": "m", "base_url": "http://x/v1", "api_key_env": ""}, path)
    set_default_model("mine", path)            # replaces, doesn't add a second line
    text = path.read_text()
    assert text.startswith('default_model = "mine"\n# my settings') and text.count("default_model") == 1
    assert "[models.kimi]  # override" in text
    config = load_config(path)
    assert config.default_model == "mine" and config.models["mine"].api_key_env == ""
    assert config.models["kimi"].max_tokens == 1000
    with pytest.raises(ConfigError, match="already defines"):
        add_model("mine", {"model": "x"}, path)


# --- which model to start with -------------------------------------------------

def test_default_model_falls_back_to_one_with_a_key(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[models.mine]\nmodel = "m"\nbase_url = "http://x"\napi_key_env = "MINE_API_KEY"\n')
    with pytest.raises(NoUsableModel, match=r"\$DASHSCOPE_API_KEY \(qwen, qwen-openai\)"):
        default_model(load_config(path))
    os.environ["ANTHROPIC_API_KEY"] = "a"
    model, note = default_model(load_config(path))
    assert model.name == "claude" and "default model 'kimi' has no API key" in note
    os.environ["MINE_API_KEY"] = "m"           # your own models come first
    assert default_model(load_config(path))[0].name == "mine"
    os.environ["MOONSHOT_API_KEY"] = "k"
    assert default_model(load_config(path)) == (load_config(path).models["kimi"], None)


# --- the wizard ----------------------------------------------------------------

def wizard(tmp_path, keys, verify=lambda m: (True, "answered in 0.1s")):
    out = Console(file=open(os.devnull, "w"), force_terminal=False)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        return run_setup(out, tmp_path / "config.toml", tmp_path, verify=verify)


def option(tmp_path, value):
    return [v for v, _ in setup_options(load_config(tmp_path / "config.toml"))].index(value)


def test_wizard_builtin_model(tmp_path):
    down = "\x1b[B" * option(tmp_path, "deepseek")
    assert wizard(tmp_path, down + "\r" + "sk-test\r") == "deepseek"
    assert read_env_file(tmp_path) == {"DEEPSEEK_API_KEY": "sk-test"}
    assert load_config(tmp_path / "config.toml").default_model == "deepseek"


def test_wizard_custom_endpoint_is_checked(tmp_path):
    down = "\x1b[B" * option(tmp_path, "anthropic")
    with FakeAnthropic([text_turn("OK")]) as server:
        name = wizard(tmp_path, down + "\r" + "mine\r" + f"{server.url}\r" + "my-model-1\r" + "\r" + "k-123\r",
                      verify=verify_model)
    assert name == "mine"
    config = load_config(tmp_path / "config.toml")
    m = config.models["mine"]
    assert (m.provider, m.model, m.base_url, m.api_key_env) == ("anthropic", "my-model-1", server.url, "MINE_API_KEY")
    assert config.default_model == "mine" and read_env_file(tmp_path)["MINE_API_KEY"] == "k-123"
    assert server.requests[0]["body"]["model"] == "my-model-1"


def test_wizard_failed_check_offers_choices(tmp_path):
    down = "\x1b[B" * option(tmp_path, "ollama")
    attempts = []

    def failing(model):
        attempts.append(model.name)
        return False, "connection refused"

    # local model, defaults for URL; check fails; "Try again" fails; "Keep it anyway".
    keys = down + "\r" + "\r" + "\r" + "\r" + "\r" + "\x1b[B\r"
    assert wizard(tmp_path, keys, verify=failing) == "local"
    assert attempts == ["local", "local"]
    m = load_config(tmp_path / "config.toml").models["local"]
    assert (m.provider, m.base_url, m.api_key_env, m.model) == ("openai", "http://localhost:11434/v1", "",
                                                                "qwen3-coder:30b")


def test_wizard_cancel(tmp_path):
    assert wizard(tmp_path, "\x1b") is None
    assert not (tmp_path / "config.toml").exists()
    assert list(CUSTOM) == ["anthropic", "openai", "ollama"]


# --- the CLI -------------------------------------------------------------------

def run_cli(home, *args, env=None):
    project = home.parent / "project"
    project.mkdir(exist_ok=True)
    base = {k: v for k, v in os.environ.items() if k not in KEYS}
    return subprocess.run([sys.executable, "-m", "wren.cli.main", *args], cwd=project,
                          env={**base, "WREN_HOME": str(home), **(env or {})},
                          capture_output=True, text=True, timeout=60)


def test_headless_without_keys_explains(tmp_path):
    proc = run_cli(tmp_path / "home", "-p", "hi")
    assert proc.returncode == 1 and "Run `wren setup`" in proc.stdout + proc.stderr


def test_keys_from_the_env_file_and_fallback_note(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    with FakeAnthropic([text_turn("hello")]) as server:
        (home / "config.toml").write_text(f'[models.mine]\nmodel = "m"\nbase_url = "{server.url}"\n'
                                          'api_key_env = "MINE_API_KEY"\nprompt_cache = false\n')
        save_key("MINE_API_KEY", "from-file", home)
        proc = run_cli(home, "-p", "hi", "--no-final-check")
    assert proc.returncode == 0, proc.stderr
    assert "default model 'kimi' has no API key" in proc.stdout and "using 'mine'" in proc.stdout
    assert {k.lower(): v for k, v in server.requests[0]["headers"].items()}["x-api-key"] == "from-file"
