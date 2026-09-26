"""`wren -p` end to end: real CLI process, real SDK, local fake Anthropic server."""

import json
import os
import subprocess
import sys

import pytest

from fake_anthropic import FakeAnthropic, text_turn, tool_turn


@pytest.fixture
def run_wren(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()

    def run(server, *args, stdin=None):
        (home / "config.toml").write_text(
            "[models.fake]\n"
            'model = "fake-model-1"\n'
            f'base_url = "{server.url}"\n'
            'api_key_env = "FAKE_KEY"\n'
            "prompt_cache = false\n"
            "price = { input = 1.0, output = 2.0 }\n"
        )
        env = {**os.environ, "WREN_HOME": str(home), "FAKE_KEY": "k"}
        env.pop("ANTHROPIC_API_KEY", None)
        return subprocess.run([sys.executable, "-m", "wren.cli.main", *args], cwd=project, env=env,
                              input=stdin, capture_output=True, text=True, timeout=60)
    run.project = project
    return run


def test_json_result(run_wren):
    turns = [tool_turn("write_file", {"path": "hello.txt", "content": "hi\n"}),
             text_turn("Created hello.txt.")]
    with FakeAnthropic(turns) as server:
        proc = run_wren(server, "-p", "make hello.txt", "--yolo", "-m", "provider/fake-model-1",
                        "--output-format", "json", "--no-checkpoints")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)            # stdout holds only the JSON object
    assert out["status"] == "done" and out["result"] == "Created hello.txt."
    assert (out["model"], out["turns"], out["tool_calls"], out["tool_errors"]) == ("fake", 2, 1, 0)
    assert out["usage"]["input_tokens"] == 2000 and out["cost_usd"] == pytest.approx(0.0022)
    assert (run_wren.project / "hello.txt").read_text() == "hi\n"
    assert "write_file" in proc.stderr       # progress still visible, on stderr


def test_prompt_from_stdin_and_max_turns_exit_code(run_wren):
    turns = [tool_turn("glob", {"pattern": "*"}), tool_turn("glob", {"pattern": "*"})]
    with FakeAnthropic(turns) as server:
        proc = run_wren(server, "-p", "-", "-m", "fake", "--max-turns", "1",
                        "--output-format", "json", stdin="look around")
        assert server.requests[0]["body"]["messages"][0]["content"][0]["text"] == "look around"
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["status"] == "max_turns"


def test_without_yolo_writes_are_denied_headless(run_wren):
    turns = [tool_turn("bash", {"command": "touch x"}), text_turn("ok")]
    with FakeAnthropic(turns) as server:
        proc = run_wren(server, "-p", "go", "-m", "fake")
    assert not (run_wren.project / "x").exists()
    assert "denied" in proc.stdout
