from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.llm.types import Message, Response, TextBlock, ToolResultBlock, ToolUseBlock, Usage

from conftest import RecordingUI, ScriptedProvider, call, reply


def results(message: Message) -> list[ToolResultBlock]:
    return [b for b in message.content if isinstance(b, ToolResultBlock)]


def test_runs_tools_until_done(ctx, model):
    (ctx.cwd / "a.py").write_text("x = 1\n")
    provider = ScriptedProvider([
        call("read_file", "t1", path="a.py"),
        call("edit_file", "t2", path="a.py", old_string="x = 1", new_string="x = 2"),
        reply("done"),
    ])
    agent = Agent(provider, model, ctx, RecordingUI(), Permissions(mode="auto"))
    assert agent.run("bump x") == "done"
    assert (ctx.cwd / "a.py").read_text() == "x = 2\n"
    assert [m.role for m in agent.messages] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert agent.usage.output_tokens == 15


def test_tool_errors_go_back_to_model(ctx, model):
    provider = ScriptedProvider([call("read_file", path="missing.py"), reply("ok")])
    agent = Agent(provider, model, ctx, RecordingUI())
    agent.run("read it")
    [result] = results(provider.requests[1][-1])
    assert result.is_error and "file not found" in result.content


def test_invalid_arguments_and_unknown_tool(ctx, model):
    provider = ScriptedProvider([
        Response(Message("assistant", [
            ToolUseBlock("t1", "read_file", {"file": "a"}),
            ToolUseBlock("t2", "nope", {}),
        ]), "tool_use"),
        reply("ok"),
    ])
    Agent(provider, model, ctx, RecordingUI()).run("go")
    r1, r2 = results(provider.requests[1][-1])
    assert r1.is_error and "missing required argument(s): path" in r1.content
    assert r2.is_error and "unknown tool" in r2.content


def test_denial_without_feedback_stops_and_skips_rest(ctx, model):
    provider = ScriptedProvider([
        Response(Message("assistant", [
            ToolUseBlock("t1", "bash", {"command": "touch x"}),
            ToolUseBlock("t2", "bash", {"command": "touch y"}),
        ]), "tool_use"),
    ])
    agent = Agent(provider, model, ctx, RecordingUI([Decision(allow=False)]))
    agent.run("make files")
    assert not (ctx.cwd / "x").exists() and not (ctx.cwd / "y").exists()
    r1, r2 = results(agent.messages[-1])
    assert "rejected" in r1.content and "Not run" in r2.content
    assert len(provider.requests) == 1  # control returned to the user


def test_denial_feedback_is_a_user_message_and_skips_the_rest(ctx, model):
    provider = ScriptedProvider([
        Response(Message("assistant", [
            ToolUseBlock("t1", "bash", {"command": "pytest tests/"}),
            ToolUseBlock("t2", "bash", {"command": "touch ran"}),
        ]), "tool_use"),
        reply("understood"),
    ])
    agent = Agent(provider, model, ctx, RecordingUI([Decision(allow=False, feedback="不用进行测试了")]))
    assert agent.run("clean") == "understood"
    last = provider.requests[1][-1]
    r1, r2 = results(last)
    assert "rejected" in r1.content and "不用进行测试了" not in r1.content
    assert "Not run" in r2.content and not (ctx.cwd / "ran").exists()
    # the feedback is the user's own text, after the tool results
    assert isinstance(last.content[-1], TextBlock) and last.content[-1].text == "不用进行测试了"


def test_always_allow_is_remembered(ctx, model):
    provider = ScriptedProvider([
        call("bash", "t1", command="echo 1"), call("bash", "t2", command="echo 2"), reply("ok"),
    ])
    ui = RecordingUI([Decision(allow=True, remember=True)])
    Agent(provider, model, ctx, ui).run("echo twice")
    assert sum(1 for e in ui.events if e[0] == "confirm") == 1


def test_interrupt_during_tool_keeps_history_valid(ctx, model):
    class InterruptingUI(RecordingUI):
        def tool_started(self, name, label):
            raise KeyboardInterrupt

    provider = ScriptedProvider([call("read_file", path="a.py"), reply("resumed")])
    agent = Agent(provider, model, ctx, InterruptingUI())
    agent.run("go")
    [r] = results(agent.messages[-1])
    assert r.tool_use_id == "t1" and "Interrupted" in r.content

    # The next prompt merges into the same user message instead of adding a
    # second consecutive user turn.
    agent.ui = RecordingUI()
    assert agent.run("continue") == "resumed"
    assert [m.role for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert isinstance(agent.messages[2].content[-1], TextBlock)


def test_truncated_tool_call_is_not_executed(ctx, model):
    provider = ScriptedProvider([
        Response(Message("assistant", [ToolUseBlock("t1", "write_file", {"path": "big.py"})]),
                 "max_tokens", Usage()),
        reply("ok"),
    ])
    Agent(provider, model, ctx, RecordingUI(), Permissions(mode="auto")).run("write")
    [r] = results(provider.requests[1][-1])
    assert r.is_error and "output token limit" in r.content
    assert not (ctx.cwd / "big.py").exists()
