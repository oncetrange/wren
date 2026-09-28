"""The OpenAI-compatible adapter: message conversion, streaming, and the agent end to end."""

import json

import pytest

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.config import ConfigError, ModelConfig, load_config
from wren.llm.openai_provider import OpenAIProvider, messages_param
from wren.llm.types import (
    INVALID_JSON_KEY,
    Completed,
    LLMError,
    Message,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallStarted,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

from conftest import RecordingUI
from fake_openai import FakeOpenAI, text_turn, tool_turn


def model(server=None, **kw) -> ModelConfig:
    return ModelConfig(name="fake", model="fake-model", provider="openai",
                       base_url=server.url if server else "http://localhost:1/v1",
                       api_key_env="FAKE_OPENAI_KEY", **kw)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FAKE_OPENAI_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")


def stream(server, messages, tools=(), **kw):
    events = list(OpenAIProvider(model(server, **kw)).stream(
        system="be brief", messages=messages, tools=list(tools)))
    return events, events[-1].response


# --- request conversion ------------------------------------------------------------

def test_messages_param_splits_tool_results_from_text():
    messages = [
        Message("user", [TextBlock("fix it")]),
        Message("assistant", [ThinkingBlock("let me look"), TextBlock("Looking."),
                              ToolUseBlock("c1", "read_file", {"path": "a.py"}),
                              ToolUseBlock("c2", "bash", {"command": "ls"})]),
        Message("user", [ToolResultBlock("c1", "1→x = 1"), ToolResultBlock("c2", "boom", is_error=True),
                         TextBlock("<wren-reminder>keep going</wren-reminder>")]),
        Message("assistant", [TextBlock("Done.")]),
    ]
    out = messages_param("sys", messages, replay_reasoning=False)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool", "user", "assistant"]
    assert out[0] == {"role": "system", "content": "sys"}
    call = out[2]["tool_calls"][0]
    assert call == {"id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
    assert out[2]["content"] == "Looking." and "reasoning_content" not in out[2]
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "1→x = 1"}
    assert out[4]["content"] == "Error: boom"
    assert out[5] == {"role": "user", "content": "<wren-reminder>keep going</wren-reminder>"}


def test_reasoning_replay_and_edge_cases():
    thinking_call = Message("assistant", [ThinkingBlock("plan"), ToolUseBlock("c1", "glob", {"pattern": "*"})])
    out = messages_param("s", [Message("user", [TextBlock("go")]), thinking_call], replay_reasoning=True)
    assert out[2]["reasoning_content"] == "plan" and out[2]["content"] is None
    # Only turns with tool calls carry reasoning back; a text-only turn doesn't.
    out = messages_param("s", [Message("assistant", [ThinkingBlock("x"), TextBlock("hi")])], True)
    assert "reasoning_content" not in out[1]
    # A call whose arguments weren't JSON is replayed as written.
    bad = Message("assistant", [ToolUseBlock("c1", "bash", {INVALID_JSON_KEY: '{"command": "ls'})])
    assert messages_param("s", [bad], False)[1]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls'


# --- streaming ---------------------------------------------------------------------

def test_streams_text_reasoning_and_tool_calls():
    turn = tool_turn(("read_file", {"path": "a.py"}), ("grep", {"pattern": "def x", "path": "src"}),
                     text="Checking.", reasoning="need to read a.py",
                     usage={"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
                            "prompt_tokens_details": {"cached_tokens": 1000}})
    with FakeOpenAI([turn]) as server:
        events, response = stream(server, [Message("user", [TextBlock("go")])],
                                  tools=[ToolSpec("read_file", "Read", {"type": "object"})])
    body = server.requests[0]["body"]
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert {k.lower(): v for k, v in server.requests[0]["headers"].items()}["authorization"] == "Bearer sk-test"
    assert body["stream"] and body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 32000 and "max_completion_tokens" not in body
    assert body["tools"][0] == {"type": "function", "function": {
        "name": "read_file", "description": "Read", "parameters": {"type": "object"}}}
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Checking."
    assert "".join(e.text for e in events if isinstance(e, ThinkingDelta)) == "need to read a.py"
    assert [e.name for e in events if isinstance(e, ToolCallStarted)] == ["read_file", "grep"]
    blocks = response.message.content
    assert isinstance(blocks[0], ThinkingBlock) and blocks[0].thinking == "need to read a.py"
    assert blocks[1] == TextBlock("Checking.")
    assert blocks[2] == ToolUseBlock("call_1_0", "read_file", {"path": "a.py"})
    assert blocks[3] == ToolUseBlock("call_1_1", "grep", {"pattern": "def x", "path": "src"})
    assert response.stop_reason == "tool_use"
    u = response.usage
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (200, 1000, 80)


@pytest.mark.parametrize("finish, calls, stop", [
    ("stop", [], "end_turn"), ("length", [], "max_tokens"), ("content_filter", [], "refusal"),
    ("stop", [("glob", {"pattern": "*"})], "tool_use"),   # some servers end tool calls with "stop"
])
def test_stop_reasons(finish, calls, stop):
    with FakeOpenAI([{"text": "x", "calls": calls, "finish": finish}]) as server:
        _, response = stream(server, [Message("user", [TextBlock("go")])])
    assert response.stop_reason == stop


def test_deepseek_cache_usage_and_invalid_arguments():
    turn = tool_turn(("bash", '{"command": "ls'),
                     usage={"prompt_tokens": 500, "completion_tokens": 5, "total_tokens": 505,
                            "prompt_cache_hit_tokens": 300, "prompt_cache_miss_tokens": 200})
    with FakeOpenAI([turn]) as server:
        _, response = stream(server, [Message("user", [TextBlock("go")])])
    assert (response.usage.input_tokens, response.usage.cache_read_tokens) == (200, 300)
    assert response.message.tool_uses()[0].input == {INVALID_JSON_KEY: '{"command": "ls'}


def test_options_reach_the_request():
    with FakeOpenAI([text_turn("ok")]) as server:
        stream(server, [Message("user", [TextBlock("go")])], reasoning_effort="high",
               extra_body={"enable_thinking": True}, max_tokens_param="max_completion_tokens")
    body = server.requests[0]["body"]
    assert body["reasoning_effort"] == "high" and body["enable_thinking"] is True
    assert body["max_completion_tokens"] == 32000 and "max_tokens" not in body
    assert "tools" not in body


def test_http_errors_become_llm_errors():
    with FakeOpenAI([{"status": 400, "error": "bad tool schema"}]) as server:
        with pytest.raises(LLMError, match="fake: HTTP 400: .*bad tool schema"):
            stream(server, [Message("user", [TextBlock("go")])])


# --- config ------------------------------------------------------------------------

def test_key_defaults_and_keyless_models(monkeypatch):
    assert ModelConfig(name="a", model="m").key_env == "ANTHROPIC_API_KEY"
    assert ModelConfig(name="o", model="m", provider="openai").key_env == "OPENAI_API_KEY"
    local = ModelConfig(name="l", model="m", provider="openai", base_url="http://localhost:11434/v1",
                        api_key_env="")
    assert local.api_key() == ""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigError, match=r"\$DEEPSEEK_API_KEY"):
        load_config().model("deepseek").api_key()


def test_config_validation_and_ignored_options(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[models.x]\nmodel = "m"\nprovider = "gemini"\n')
    with pytest.raises(ConfigError, match="provider must be"):
        load_config(path)
    path.write_text('[models.x]\nmodel = "m"\nprovider = "openai"\nthinking = "adaptive"\n'
                    'extra_body = { enable_thinking = true }\n')
    cfg = load_config(path).model("x")
    assert cfg.ignored_options() == ["thinking"] and cfg.extra_body == {"enable_thinking": True}
    assert load_config(path).model("qwen-openai").provider == "openai"


# --- end to end --------------------------------------------------------------------

def test_agent_runs_tools_over_openai(ctx):
    (ctx.cwd / "a.txt").write_text("hello\n")
    turns = [tool_turn(("read_file", {"path": "a.txt"}), reasoning="read it first"),
             tool_turn(("bash", '{"command": "ls')),                       # broken arguments
             text_turn("a.txt says hello.")]
    with FakeOpenAI(turns) as server:
        agent = Agent(OpenAIProvider(model(server, replay_reasoning=True)), model(server), ctx,
                      RecordingUI(), Permissions(mode="auto"))
        assert agent.run("what does a.txt say?") == "a.txt says hello."
    second = server.requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["reasoning_content"] == "read it first"
    assert "1→hello" in second[3]["content"]
    third = server.requests[2]["body"]["messages"]
    assert third[-1]["role"] == "tool" and third[-1]["content"].startswith("Error: invalid arguments for bash")
    assert agent.usage.input_tokens == 3000 and agent.turns == 3


def test_cli_with_an_openai_model(tmp_path):
    import os
    import subprocess
    import sys

    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    turns = [tool_turn(("write_file", {"path": "hi.txt", "content": "hi\n"})), text_turn("Wrote hi.txt.")]
    with FakeOpenAI(turns) as server:
        (home / "config.toml").write_text(
            f'[models.local]\nprovider = "openai"\nmodel = "m"\nbase_url = "{server.url}"\n'
            'api_key_env = ""\nprice = { input = 1.0, output = 2.0 }\n')
        env = {**os.environ, "WREN_HOME": str(home)}
        proc = subprocess.run([sys.executable, "-m", "wren.cli.main", "-p", "write hi.txt", "-m", "local",
                               "--yolo", "--no-final-check", "--no-checkpoints", "--output-format", "json"],
                              cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["result"] == "Wrote hi.txt." and out["tool_calls"] == 1
    assert out["cost_usd"] == pytest.approx(2 * (1000 * 1 + 50 * 2) / 1e6)
    assert (project / "hi.txt").read_text() == "hi\n"
    assert {k.lower(): v for k, v in server.requests[0]["headers"].items()}["authorization"] == "Bearer none"
