from types import SimpleNamespace as NS

from wren.llm.anthropic_provider import _messages_param, _system_param, _to_response
from wren.llm.types import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock


def test_messages_param_round_trip_and_cache_breakpoint():
    history = [
        Message("user", [TextBlock("hi")]),
        Message("assistant", [ThinkingBlock("hmm", "sig"), ToolUseBlock("t1", "bash", {"command": "ls"})]),
        Message("user", [ToolResultBlock("t1", "a.py", False)]),
    ]
    out = _messages_param(history, cache=True)
    assert out[1]["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    assert out[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[0]["content"][0]
    assert "cache_control" not in _messages_param(history, cache=False)[2]["content"][0]
    assert _system_param("sys", cache=False) == "sys"


def test_to_response_maps_blocks_and_usage():
    msg = NS(
        content=[
            NS(type="thinking", thinking="t", signature="s"),
            NS(type="text", text="hello"),
            NS(type="tool_use", id="t1", name="read_file", input={"path": "a"}),
        ],
        usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=100,
                 cache_creation_input_tokens=None),
        stop_reason="tool_use",
    )
    r = _to_response(msg)
    assert r.stop_reason == "tool_use"
    assert r.message.text() == "hello"
    assert r.message.tool_uses()[0].input == {"path": "a"}
    assert (r.usage.cache_read_tokens, r.usage.cache_write_tokens) == (100, 0)
