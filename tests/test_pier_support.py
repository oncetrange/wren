import json
import subprocess

import pytest

from wren.config import ConfigError
from wren.integrations import pier_support as support


def test_resolve_model_by_pier_name():
    assert support.resolve_model("moonshot/kimi-k2.7-code").name == "kimi"
    assert support.resolve_model("kimi").name == "kimi"
    with pytest.raises(ConfigError):
        support.resolve_model("openai/gpt-x")


def test_model_hosts():
    assert support.model_hosts(support.resolve_model("kimi")) == ["api.moonshot.cn"]
    assert support.model_hosts(support.resolve_model("claude")) == ["api.anthropic.com"]


def test_install_commands_pin_ref():
    (_, root), (user, agent) = support.install_commands("abc123")
    assert user == "agent" and "wren/archive/abc123.tar.gz" in agent
    assert "--python 3.12" in agent


def fake_wren(tmp_path, body: str) -> str:
    """A run_command whose wren binary is a stub script, run with bash."""
    stub = tmp_path / "wren"
    stub.write_text(f"#!/bin/bash\n{body}\n")
    stub.chmod(0o755)
    cmd = support.run_command("fix it; don't `rm` anything", support.resolve_model("kimi"), "--max-turns 5")
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
