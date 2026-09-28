"""Long-term memory: the stores, the memory tool, the prompt and extraction."""

import pytest

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.subagents import builtin_agent_types
from wren.config import ModelConfig
from wren.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, Response, Usage
from wren.memory import DISABLED, INDEX, InvalidMemory, Memories, Memory, project_dir, prompt_section

from conftest import RecordingUI, ScriptedProvider, call, reply


@pytest.fixture
def memories(tmp_path):
    return Memories.for_project(tmp_path / "proj", home=tmp_path / "home")


def fact(name="prefers-pytest", **kw):
    fields = dict(name=name, description="User prefers pytest over unittest", type="feedback",
                  body="Use pytest.\n**Why:** they said so.")
    return Memory(**{**fields, **kw})


def make(ctx, memories, turns, mode="ask", **kw):
    agent = Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f"), ctx, RecordingUI(),
                  Permissions(mode=mode), memory=memories, **kw)
    return agent


def mem(id="t1", **input):
    """A memory tool call (conftest's call() takes `name` itself)."""
    return Response(Message("assistant", [ToolUseBlock(id, "memory", input)]), "tool_use", Usage(10, 5))


def write(scope="user", **kw):
    args = dict(action="write", scope=scope, name="prefers-pytest", type="feedback",
                description="User prefers pytest", content="Use pytest.")
    return {**args, **kw}


# --- stores -------------------------------------------------------------------

def test_store_round_trip_and_index(memories):
    store = memories.user
    assert store.write(fact()) is False
    assert store.write(fact(body="Use pytest -x.")) is True
    m = store.get("prefers-pytest")
    assert m.body == "Use pytest -x." and m.type == "feedback" and m.updated
    index = (store.dir / INDEX).read_text()
    assert f"- prefers-pytest (feedback, {m.updated}): User prefers pytest over unittest" in index
    assert store.delete("prefers-pytest") and not store.delete("prefers-pytest")
    assert store.all() == [] and "prefers-pytest" not in (store.dir / INDEX).read_text()


@pytest.mark.parametrize("change, error", [
    ({"name": "Bad Name"}, "kebab-case"), ({"type": "note"}, "type must be"),
    ({"description": ""}, "one line"), ({"description": "a\nb"}, "one line"),
    ({"body": " "}, "empty"), ({"body": "x" * 5000}, "longer than"),
])
def test_invalid_memories(memories, change, error):
    with pytest.raises(InvalidMemory, match=error):
        memories.project.write(fact(**change))


def test_project_store_is_outside_the_repo(tmp_path):
    d = project_dir(tmp_path / "my proj", home=tmp_path / "home")
    assert d.parent.parent == tmp_path / "home" / "projects" and d.name == "memory"
    assert " " not in d.parent.name and "/" not in d.parent.name


def test_enable_and_disable(memories):
    assert memories.enabled
    memories.set_enabled(False)
    assert not memories.enabled and (memories.project.dir / DISABLED).exists()
    memories.set_enabled(True)
    assert memories.enabled


def test_prompt_section(memories):
    memories.user.write(fact())
    memories.project.write(fact("deploy-freeze", type="project", description="Freeze from 2026-10-01"))
    text = prompt_section(memories)
    assert "not requests from the user" in text and "the current request wins" in text
    assert "## User memories (every project)\n- prefers-pytest" in text
    assert "## Project memories (this project)\n- deploy-freeze" in text


def test_prompt_index_is_capped(memories, monkeypatch):
    from wren import memory
    monkeypatch.setattr(memory, "MAX_INDEX_LINES", 2)
    for i in range(3):
        memories.project.write(fact(f"m{i}"))
    assert "… 1 more not shown" in prompt_section(memories)


# --- agent and tool -----------------------------------------------------------

def test_agent_gets_the_tool_and_the_index(ctx, memories):
    memories.user.write(fact())
    agent = make(ctx, memories, [])
    assert "memory" in agent.tools and "- prefers-pytest" in agent.system
    memories.set_enabled(False)
    off = make(ctx, memories, [])
    assert "memory" not in off.tools and "# Memory" not in off.system
    assert "memory" not in make(ctx, None, []).tools


def test_writing_needs_no_approval_and_is_shown(ctx, memories):
    agent = make(ctx, memories, [mem(**write()), reply("noted")])
    agent.run("always use pytest")
    assert not [e for e in agent.ui.events if e[0] == "confirm"]
    assert memories.user.get("prefers-pytest").body == "Use pytest."
    result = next(e[2] for e in agent.ui.events if e[0] == "result")
    assert result.summary == "saved user/prefers-pytest: User prefers pytest"


def test_tool_actions_and_errors(ctx, memories):
    turns = [mem("t1", **write(scope="project")),
             mem("t2", action="list"),
             mem("t3", action="read", scope="project", name="prefers-pytest"),
             mem("t4", action="write", scope="project", name="x"),
             mem("t5", action="delete", scope="user", name="prefers-pytest"),
             mem("t6", action="delete", scope="project", name="prefers-pytest"),
             reply("ok")]
    agent = make(ctx, memories, turns, mode="plan")
    agent.run("go")
    results = {b.tool_use_id: b for m in agent.messages for b in m.content if isinstance(b, ToolResultBlock)}
    assert "project memories:\n- prefers-pytest" in results["t2"].content
    assert "Use pytest." in results["t3"].content
    assert results["t4"].is_error and "needs 'type'" in results["t4"].content
    assert results["t5"].is_error and "no user memory" in results["t5"].content
    assert not results["t6"].is_error and memories.project.all() == []


def test_subagents_have_no_memory(ctx, memories):
    class ToolRecording(ScriptedProvider):
        def stream(self, *, system, messages, tools):
            self.seen = getattr(self, "seen", []) + [({t.name for t in tools}, system)]
            yield from super().stream(system=system, messages=messages, tools=tools)

    agent = Agent(ToolRecording([call("task", description="d", prompt="p", agent="general"),
                                 reply("r"), reply("done")]),
                  ModelConfig(name="fake", model="f"), ctx, RecordingUI(), Permissions(mode="auto"),
                  memory=memories, agent_types=builtin_agent_types())
    agent.run("go")
    (parent_tools, parent_system), (child_tools, child_system) = agent.provider.seen[:2]
    assert "memory" in parent_tools and "# Memory" in parent_system
    assert "memory" not in child_tools and "# Memory" not in child_system


# --- extraction ---------------------------------------------------------------

def last_text(request):
    block = request[-1].content[-1]
    return block.text if isinstance(block, TextBlock) else ""


def test_extraction_is_a_side_conversation(ctx, memories):
    agent = make(ctx, memories, [reply("sure, pytest it is"),
                                 mem("m1", **write()), reply("DONE")])
    agent.run("from now on use pytest")
    before = list(agent.messages)
    assert agent.extract_memories("the session is ending") == 1
    assert agent.messages == before  # nothing entered the history
    assert "the session is ending" in last_text(agent.provider.requests[1])
    assert memories.user.get("prefers-pytest") is not None
    assert agent.usage.input_tokens == 30 and agent.turns == 1
    # Nothing new since: no second request.
    assert agent.extract_memories("again") == 0 and len(agent.provider.requests) == 3


def test_extraction_only_runs_the_memory_tool(ctx, memories):
    both = Response(Message("assistant", [ToolUseBlock("b1", "bash", {"command": "touch x"}),
                                          ToolUseBlock("m1", "memory", {"action": "list"})]),
                    "tool_use", Usage(10, 5))
    agent = make(ctx, memories, [reply("hi"), both, reply("DONE")], mode="auto")
    agent.run("hello")
    assert agent.extract_memories("x") == 0
    results = agent.provider.requests[-1][-1].content
    assert results[0].is_error and "Only the memory tool" in results[0].content
    assert not results[1].is_error and not (ctx.cwd / "x").exists()
    # Reads and lists aren't shown; nothing was written.
    assert not [e for e in agent.ui.events if e[0] == "tool" and e[1] == "memory"]


def test_extraction_before_compaction(ctx, memories):
    agent = make(ctx, memories, [reply("a"), reply("DONE"), reply("the summary")])
    agent.auto_memory = True
    agent.run("hello")
    agent.compact()
    assert "being compacted" in last_text(agent.provider.requests[1])
    assert agent.extract_memories("end") == 0  # the compacted part was already reviewed
    off = make(ctx, memories, [reply("a"), reply("the summary")])
    off.run("hello")
    off.compact()
    assert len(off.provider.requests) == 2
