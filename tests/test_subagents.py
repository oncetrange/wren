"""Subagents: the task tool, what a subagent can do, and what comes back."""

import io
import json
from pathlib import Path

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply
from rich.console import Console

from wren.agent import subagents
from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.agent.session import SessionLog
from wren.agent.subagents import (
    AgentTypeError,
    builtin_agent_types,
    discover_agent_types,
    parse_agent_type,
)
from wren.cli.ui import RichUI
from wren.config import ModelConfig, Price
from wren.llm.types import ToolResultBlock
from wren.tools import ToolOutput
from wren.tools.readonly import is_read_only_command


class SystemRecordingProvider(ScriptedProvider):
    def __init__(self, turns):
        super().__init__(turns)
        self.systems: list[str] = []
        self.tool_names: list[list[str]] = []

    def stream(self, *, system, messages, tools):
        self.systems.append(system)
        self.tool_names.append([t.name for t in tools])
        yield from super().stream(system=system, messages=messages, tools=tools)


def make(ctx, turns, mode="auto", decisions=(), **kw):
    model = ModelConfig(name="fake", model="f", price=Price(1.0, 2.0))
    return Agent(SystemRecordingProvider(turns), model, ctx, RecordingUI(list(decisions)),
                 Permissions(mode=mode), agent_types=builtin_agent_types(), **kw)


def task(agent="explore", prompt="find the config loader", id="t1"):
    return call("task", id, description="find config", prompt=prompt, agent=agent)


def result_of(agent: Agent, id="t1") -> ToolResultBlock:
    return next(b for m in agent.messages for b in m.content
                if isinstance(b, ToolResultBlock) and b.tool_use_id == id)


def test_report_comes_back_and_child_starts_fresh(ctx):
    (ctx.cwd / "config.py").write_text("def load(): ...\n")
    agent = make(ctx, [task(), call("grep", "c1", pattern="def load"),
                       reply("load() is in config.py:1"), reply("It's in config.py.")])
    assert agent.run("where is the config loaded?") == "It's in config.py."
    res = result_of(agent)
    assert not res.is_error and res.content == "load() is in config.py:1"
    p = agent.provider
    # The subagent sees only its prompt, its own system prompt and its own tools.
    child_first = p.requests[1]
    assert len(child_first) == 1 and child_first[0].content[0].text == "find the config loader"
    assert "subagent of Wren" in p.systems[1] and "role: explore" in p.systems[1]
    assert "subagent" not in p.systems[0]
    assert sorted(p.tool_names[1]) == ["bash", "glob", "grep", "read_file"]
    assert "task" in p.tool_names[0]
    # Its tool calls reached the UI; its text didn't.
    ui = agent.ui
    assert ("tool", "grep", "def load") in ui.events
    assert ("text", "load() is in config.py:1") not in ui.events


def test_general_has_all_but_excluded_tools(ctx):
    agent = make(ctx, [task("general"), reply("did it"), reply("ok")])
    agent.run("go")
    names = set(agent.provider.tool_names[1])
    assert {"read_file", "write_file", "edit_file", "bash"} <= names
    assert not names & {"task", "todo_write", "exit_plan_mode"}


def test_explore_is_read_only(ctx):
    (ctx.cwd / "a.txt").write_text("hi\n")
    agent = make(ctx, [task(), call("bash", "c1", command="ls && git status 2>/dev/null | head"),
                       call("bash", "c2", command="rm a.txt"),
                       call("write_file", "c3", path="b.txt", content="x"),
                       reply("report"), reply("done")], mode="ask")
    agent.run("look")
    child = agent.provider.requests[-2]
    results = {b.tool_use_id: b for m in child for b in m.content if isinstance(b, ToolResultBlock)}
    assert "[exit code" in results["c1"].content
    assert results["c2"].is_error and "read-only" in results["c2"].content
    assert results["c3"].is_error and "unknown tool 'write_file'" in results["c3"].content
    assert (ctx.cwd / "a.txt").exists() and not (ctx.cwd / "b.txt").exists()
    # The read-only command ran without asking, in ask mode.
    assert not [e for e in agent.ui.events if e[0] == "confirm"]


def test_general_asks_through_the_parent_and_marks_changes(ctx):
    agent = make(ctx, [task("general", "write b.txt"), call("write_file", "c1", path="b.txt", content="x"),
                       reply("wrote b.txt"), reply("done")], mode="ask",
                 decisions=[Decision(allow=True)])
    agent.run("write it via a subagent")
    assert [e[1] for e in agent.ui.events if e[0] == "confirm"] == ["write_file"]
    assert (ctx.cwd / "b.txt").read_text() == "x"
    assert agent.changed


def test_plan_mode_allows_only_read_only_subagents(ctx):
    agent = make(ctx, [task("general"), reply("ok")], mode="plan")
    agent.run("plan it")
    res = result_of(agent)
    assert res.is_error and "only read-only subagents" in res.content and "explore" in res.content


def test_usage_and_runs_roll_up(ctx):
    agent = make(ctx, [task(), reply("report"), reply("done")])
    agent.run("go")
    # Three model calls in all, Usage(10, 5) each; only the parent's two are its turns.
    assert agent.turns == 2
    assert agent.usage.input_tokens == 30 and agent.usage.output_tokens == 15
    assert agent.cost == pytest.approx(3 * (10 * 1 + 5 * 2) / 1e6)
    assert agent.subagent_runs == [{"agent": "explore", "model": "fake", "description": "find config", "status": "done",
                                    "turns": 1, "tool_calls": 0, "cost": pytest.approx(20 / 1e6)}]


def test_out_of_turns_returns_partial_report(ctx, monkeypatch):
    kinds = builtin_agent_types()
    kinds["explore"].max_turns = 2
    agent = Agent(SystemRecordingProvider([task(), call("glob", "c1", pattern="*"),
                                           call("glob", "c2", pattern="*"), reply("done")]),
                  ModelConfig(name="fake", model="f"), ctx, RecordingUI(), Permissions(mode="auto"),
                  agent_types=kinds)
    agent.run("go")
    res = result_of(agent)
    assert res.is_error and "ran out of turns (2)" in res.content


def test_interrupt_stops_the_parent_too(ctx):
    class Boom(SystemRecordingProvider):
        def stream(self, *, system, messages, tools):
            if len(self.requests) == 1:
                raise KeyboardInterrupt
            yield from super().stream(system=system, messages=messages, tools=tools)

    agent = Agent(Boom([task(), reply("never")]), ModelConfig(name="fake", model="f"), ctx,
                  RecordingUI(), Permissions(mode="auto"), agent_types=builtin_agent_types())
    agent.run("go")
    assert agent.status == "interrupted"
    assert result_of(agent).content == "Interrupted by the user."


def test_unknown_type(ctx):
    agent = make(ctx, [task("nope"), reply("ok")])
    agent.run("go")
    assert "no subagent type 'nope'" in result_of(agent).content


def test_child_log_is_linked_from_the_parent(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(subagents, "SUBAGENTS_DIR", tmp_path / "subs")
    agent = make(ctx, [task(), reply("report"), reply("done")],
                 log=SessionLog(directory=tmp_path / "sessions"))
    agent.run("go")
    entries = [json.loads(l) for l in agent.log.path.read_text().splitlines()]
    start = next(e for e in entries if e["kind"] == "subagent")
    child_log = tmp_path / "subs" / start["log"].rsplit("/", 1)[-1]
    kinds = [json.loads(l)["kind"] for l in child_log.read_text().splitlines()]
    assert kinds[0] == "session_start" and "message" in kinds
    assert any(e["kind"] == "usage" and e.get("purpose") == "subagent" for e in entries)


def test_nested_rendering():
    out = io.StringIO()
    ui = RichUI(Console(file=out, width=80, force_terminal=False), interactive=False)
    ui.tool_started("task", "find config (explore)")
    with ui.nested():
        ui.tool_started("grep", "def load")
        ui.tool_finished("grep", ToolOutput("x", summary="3 matches"))
        ui.tool_finished("bash", ToolOutput("boom", is_error=True, summary="exit 1"))
    ui.tool_finished("task", ToolOutput("r", summary="explore · 2 turns"))
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("● task")
    assert lines[1].startswith("    ● grep def load")
    assert "3 matches" not in out.getvalue()
    assert lines[2].startswith("      ⎿ exit 1")
    assert lines[-1].startswith("  ⎿ explore · 2 turns")


@pytest.mark.parametrize("command", [
    "ls -la", "git log --oneline -5", "git diff HEAD~1 -- src", "grep -rn 'a|b' src | head -20",
    "find . -name '*.py' | wc -l", "cat a.txt 2>/dev/null", "cd src && ls",
    "rg foo 2>&1 | sort | uniq -c", "git show HEAD:src/a.py > /dev/null",
])
def test_read_only_commands(command):
    assert is_read_only_command(command)


@pytest.mark.parametrize("command", [
    "rm a.txt", "echo x > a.txt", "cat a >> b", "ls; rm -rf x", "find . -delete",
    "find . -exec rm {} \\;", "sort -o out in", "sort -ro out in", "git commit -m x", "git push",
    "git branch -D x", "git diff --output=x", "ls $(rm x)", "ls `rm x`", "ls\nrm x",
    "FOO=1 ls", "python -c 'print(1)'", "sed -i s/a/b/ f", "ls &", "(rm x)", "cat <<EOF",
    "tee a.txt", "ls | xargs rm", "unterminated 'quote",
])
def test_writing_commands(command):
    assert not is_read_only_command(command)


# --- agent definitions ---------------------------------------------------------


def write_agent(directory, name, frontmatter, body="Review the change."):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(f"---\n{frontmatter}\n---\n{body}\n")
    return path


def test_parse_claude_code_definition(tmp_path):
    path = write_agent(tmp_path, "reviewer", "name: reviewer\ndescription: Reviews code.\n"
                                             "tools: Read, Grep, Glob, WebFetch\nmodel: inherit")
    kind, warnings = parse_agent_type(path, ".claude/agents")
    assert kind.tools == frozenset({"read_file", "grep", "glob"})
    assert kind.read_only and kind.model is None and kind.source == ".claude/agents"
    assert "# Your role: reviewer\nReview the change." in kind.prompt
    assert warnings == [f"agent 'reviewer' ({path}): wren has no tool 'WebFetch'; ignoring it"]


def test_parse_options_and_errors(tmp_path):
    path = write_agent(tmp_path, "fixer", "name: fixer\ndescription: Fixes.\ntools: [read_file, bash]\n"
                                          "read-only: true\nmodel: kimi\nmax-turns: 5")
    kind, _ = parse_agent_type(path, "x")
    assert kind.read_only and kind.tools == frozenset({"read_file", "bash"})
    assert kind.model == "kimi" and kind.max_turns == 5
    all_tools, _ = parse_agent_type(write_agent(tmp_path, "any", "name: any\ndescription: d"), "x")
    assert all_tools.tools is None and not all_tools.read_only
    with pytest.raises(AgentTypeError, match="'name'"):
        parse_agent_type(write_agent(tmp_path, "bad", "name: Bad Name\ndescription: d"), "x")
    with pytest.raises(AgentTypeError, match="empty"):
        parse_agent_type(write_agent(tmp_path, "empty", "name: empty\ndescription: d", body=""), "x")
    with pytest.raises(AgentTypeError, match="max-turns"):
        parse_agent_type(write_agent(tmp_path, "mt", "name: mt\ndescription: d\nmax-turns: 0"), "x")


def test_discovery_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    project, wren_home = tmp_path / "p", tmp_path / "wrenhome"
    write_agent(tmp_path / "home" / ".claude" / "agents", "r", "name: reviewer\ndescription: claude's")
    write_agent(project / ".wren" / "agents", "r", "name: reviewer\ndescription: project's")
    write_agent(project / ".wren" / "agents", "e", "name: explore\ndescription: my explore")
    (project / ".wren" / "agents" / "broken.md").write_text("no frontmatter")
    types, warnings = discover_agent_types(project, wren_home)
    assert types["reviewer"].description == "project's" and types["reviewer"].source == ".wren/agents"
    assert types["explore"].description == "my explore"   # built-ins can be replaced
    assert "general" in types
    assert any("overrides the one in ~/.claude/agents" in w for w in warnings)
    assert any(w.startswith("skipping agent:") for w in warnings)


def test_custom_read_only_agent_and_its_model(ctx):
    kinds = builtin_agent_types()
    kinds["reviewer"] = subagents.AgentType("reviewer", "d", "# Your role: reviewer\n",
                                            tools=frozenset({"read_file", "write_file"}),
                                            read_only=True, model="cheap")
    cheap = ModelConfig(name="cheap", model="c", price=Price(0.1, 0.2))
    child_provider = SystemRecordingProvider([call("write_file", "c1", path="x", content="y"),
                                              reply("looks fine")])
    agent = Agent(SystemRecordingProvider([task("reviewer"), reply("done")]),
                  ModelConfig(name="fake", model="f"), ctx, RecordingUI(), Permissions(mode="auto"),
                  agent_types=kinds)
    agent.resolve_model = lambda name: (child_provider, cheap)
    agent.run("review")
    denied = [b for m in child_provider.requests[-1] for b in m.content if isinstance(b, ToolResultBlock)]
    assert denied[0].is_error and "read-only and can't use write_file" in denied[0].content
    assert not (ctx.cwd / "x").exists()
    assert agent.subagent_runs[0]["model"] == "cheap"
    assert agent.subagent_runs[0]["cost"] == pytest.approx(2 * (10 * 0.1 + 5 * 0.2) / 1e6)


def test_unknown_model_falls_back(ctx):
    from wren.config import ConfigError

    kinds = builtin_agent_types()
    kinds["explore"].model = "missing"
    agent = Agent(SystemRecordingProvider([task(), reply("report"), reply("done")]),
                  ModelConfig(name="fake", model="f"), ctx, RecordingUI(), Permissions(mode="auto"),
                  agent_types=kinds)

    def resolve(name):
        raise ConfigError(f"unknown model {name!r}")
    agent.resolve_model = resolve
    agent.run("go")
    assert result_of(agent).content == "report"
    assert any(e[0] == "notice" and "can't use model 'missing'" in e[1] for e in agent.ui.events)


def test_subagent_tools_follow_each_tools_rule(ctx, tmp_path):
    from wren.schedules import Schedules
    from wren.tools.schedule import ScheduleTool
    from wren.tools.web import WebFetch, WebSearch

    parent = Agent(ScriptedProvider([]), ModelConfig(name="fake", model="f"), ctx, RecordingUI(),
                   agent_types=builtin_agent_types(),
                   extra_tools=[ScheduleTool(Schedules(tmp_path)), WebSearch(), WebFetch()])
    kinds = builtin_agent_types()
    general = {t.name for t in subagents.subagent_tools(parent, kinds["general"])}
    explore = {t.name for t in subagents.subagent_tools(parent, kinds["explore"])}
    assert {"read_file", "edit_file", "bash", "bash_output", "web_search", "web_fetch"} <= general
    assert not general & {"task", "todo_write", "exit_plan_mode", "schedule"}       # "never"
    assert explore == {"read_file", "grep", "glob", "bash"}                         # "writers" too
    # The parent's instances are shared, not copied.
    assert next(t for t in subagents.subagent_tools(parent, kinds["general"]) if t.name == "web_fetch") \
        is parent.tools["web_fetch"]
