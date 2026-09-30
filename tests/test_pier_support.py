import json
import subprocess
from pathlib import Path

import pytest

from wren.config import ConfigError, load_config
from wren.integrations import pier_support as support

NO_CONFIG = Path("/nonexistent/config.toml")


def resolve(name, config_file=NO_CONFIG):
    return support.resolve_model(name, config_file)


def test_resolve_model_by_pier_name():
    assert resolve("moonshot/kimi-k2.7-code").name == "kimi"
    assert resolve("kimi").name == "kimi"
    with pytest.raises(ConfigError):
        resolve("openai/gpt-x")


def test_model_hosts():
    assert support.model_hosts(resolve("kimi")) == ["api.moonshot.cn"]
    assert support.model_hosts(resolve("claude")) == ["api.anthropic.com"]


def test_install_commands_pin_ref():
    (_, root), (user, agent) = support.install_commands("abc123")
    assert user == "agent" and "wren/archive/abc123.tar.gz" in agent
    assert "--python 3.12" in agent


def fake_wren(tmp_path, body: str) -> str:
    """A run_command whose wren binary is a stub script, run with bash."""
    stub = tmp_path / "wren"
    stub.write_text(f"#!/bin/bash\n{body}\n")
    stub.chmod(0o755)
    cmd = support.run_command("fix it; don't `rm` anything", resolve("kimi"), "--max-turns 5")
    return (cmd.replace(support.WREN_BIN, str(stub))
               .replace(support.LOG_DIR, str(tmp_path / "logs")))


def test_run_command_passes_prompt_and_reports_json(tmp_path):
    (tmp_path / "logs").mkdir()
    cmd = fake_wren(tmp_path, 'printf "%s\\n" "$@" > "$(dirname "$0")/args"; echo \'{"status": "max_turns"}\'; exit 1')
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0                         # a result exists: let the verifier grade
    assert json.loads(proc.stdout) == {"status": "max_turns"}
    args = (tmp_path / "args").read_text().splitlines()
    assert args[:4] == ["-p", "fix it; don't `rm` anything", "-m", "kimi"]
    assert "--yolo" in args and "--max-turns" in args


def test_run_command_fails_without_result(tmp_path):
    (tmp_path / "logs").mkdir()
    cmd = fake_wren(tmp_path, 'echo "error: model needs an API key" >&2; exit 1')
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 1 and "API key" in proc.stderr


def test_read_run(tmp_path):
    (tmp_path / "wren" / "sessions").mkdir(parents=True)
    (tmp_path / support.RESULT_FILE).write_text(json.dumps({"status": "done", "turns": 3}))
    events = [
        {"kind": "usage", "usage": {"input_tokens": 10, "cache_read_tokens": 90, "cache_write_tokens": 0}},
        {"kind": "compact"},
        {"kind": "usage", "usage": {"input_tokens": 5, "cache_read_tokens": 20, "cache_write_tokens": 0}},
    ]
    (tmp_path / "wren" / "sessions" / "s.jsonl").write_text("\n".join(map(json.dumps, events)))
    run = support.read_run(tmp_path)
    assert run["peak_context_tokens"] == 100 and run["compactions"] == 1
    assert support.read_run(tmp_path / "missing") is None


USER_CONFIG = """
[models.tokenplan]
model = "qwen3.8-flash"
base_url = "https://token-plan.example.com/apps/anthropic"
api_key_env = "PLAN_API_KEY"
thinking = { type = "enabled", budget_tokens = 8000 }
price = { tiers = [{ up_to = 32000, input = 0.1, output = 0.4 }, { input = 0.2, output = 0.8 }] }
extra_body = { "enable-x" = true, note = "a \\"quoted\\" value" }
"""


def test_user_models_resolve_and_travel_to_the_container(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(USER_CONFIG)
    model = resolve("qwen3.8-flash", config)          # by model id, as Pier passes it
    assert model.name == "tokenplan" and resolve("tokenplan", config) == model
    assert support.model_hosts(model) == ["token-plan.example.com"]
    shipped = tmp_path / "shipped.toml"
    shipped.write_text(support.model_config_toml(model))
    assert "PLAN_API_KEY" in shipped.read_text()
    assert load_config(shipped).models["tokenplan"] == model   # round-trips exactly


def test_run_command_writes_the_model_config(tmp_path):
    (tmp_path / "logs").mkdir()
    config = tmp_path / "config.toml"
    config.write_text(USER_CONFIG)
    stub = tmp_path / "wren"
    stub.write_text("#!/bin/bash\necho '{}'\n")
    stub.chmod(0o755)
    cmd = support.run_command("go", resolve("tokenplan", config))
    cmd = cmd.replace(support.WREN_BIN, str(stub)).replace(support.LOG_DIR, str(tmp_path / "logs"))
    assert subprocess.run(["bash", "-c", cmd], capture_output=True).returncode == 0
    written = tmp_path / "logs" / "wren" / "config.toml"
    assert load_config(written).models["tokenplan"].base_url == "https://token-plan.example.com/apps/anthropic"
