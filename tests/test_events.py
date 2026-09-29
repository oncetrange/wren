"""Combination rules of the event layer, exercised with ad-hoc handlers."""

from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.events import Verdict
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions


def make(ctx, model, turns, mode="ask"):
    return Agent(ScriptedProvider(turns), model, ctx, RecordingUI(), Permissions(mode=mode))


def test_builtins_are_registered_first(ctx, model):
    agent = make(ctx, model, [])
    assert agent.hooks.names("pre_tool") == ["plan_guard"]
    assert agent.hooks.names("stop") == ["unfinished_todos", "final_check"]


def test_allow_skips_the_permission_prompt(ctx, model):
    agent = make(ctx, model, [call("bash", command="echo hi"), reply("ok")])
    agent.hooks.on("pre_tool", "trust-echo",
                   lambda e: Verdict("allow") if e.call.input.get("command", "").startswith("echo") else None)
    agent.run("say hi")
    assert not any(ev[0] == "confirm" for ev in agent.ui.events)
    assert "hi" in agent.provider.requests[1][-1].content[0].content


def test_deny_stops_the_chain_and_reaches_the_model(ctx, model):
    seen = []
    agent = make(ctx, model, [call("bash", command="rm -rf /"), reply("ok")], mode="auto")
    agent.hooks.on("pre_tool", "no-rm", lambda e: Verdict("deny", "rm is not allowed here"))
    agent.hooks.on("pre_tool", "later", lambda e: seen.append(e) or None)
    agent.run("clean up")
    result = agent.provider.requests[1][-1].content[0]
    assert result.is_error and result.content == "rm is not allowed here" and seen == []


def test_plan_guard_cannot_be_overridden(ctx, model):
    agent = make(ctx, model, [call("write_file", path="a.txt", content="x"), reply("ok")], mode="plan")
    agent.hooks.on("pre_tool", "allow-all", lambda e: Verdict("allow"))
    agent.run("write")
    assert not (ctx.cwd / "a.txt").exists()


def test_post_tool_context_is_appended(ctx, model):
    agent = make(ctx, model, [call("write_file", path="a.py", content="x=1"), reply("ok")], mode="auto")
    agent.hooks.on("post_tool", "lint",
                   lambda e: Verdict(context="lint: missing spaces") if e.tool.edits_files else None)
    agent.run("write")
    assert agent.provider.requests[1][-1].content[0].content.endswith("lint: missing spaces")


def test_prompt_handlers_block_or_add_context(ctx, model):
    agent = make(ctx, model, [reply("ok")])
    agent.hooks.on("prompt", "branch", lambda e: Verdict(context="current branch: main"))
    agent.run("hello")
    assert "current branch: main" in agent.messages[0].content[1].text

    agent.hooks.on("prompt", "no-secrets",
                   lambda e: Verdict("deny", "looks like a secret") if "sk-" in e.prompt else None)
    before = len(agent.messages)
    agent.run("my key is sk-123")
    assert agent.status == "blocked" and len(agent.messages) == before
    assert any("looks like a secret" in ev[1] for ev in agent.ui.events if ev[0] == "notice")


def test_stop_blocks_are_merged_and_capped(ctx, model):
    turns = [reply(f"attempt {i}") for i in range(10)]
    agent = make(ctx, model, turns)
    agent.hooks.on("stop", "tests-fail", lambda e: Verdict("block", "tests still fail"))
    agent.hooks.on("stop", "lint-fail", lambda e: Verdict("block", "lint still fails"))
    assert agent.run("fix it") == "attempt 3"          # 1 + 3 blocked stops
    reminder = agent.provider.requests[1][-1].content[-1].text
    assert "tests still fail" in reminder and "lint still fails" in reminder
