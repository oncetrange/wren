import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.tools.files import EditFile, ReadFile
from wren.tools.shell import Bash

READ, EDIT, BASH = ReadFile(), EditFile(), Bash()
LS = {"command": "ls"}


@pytest.mark.parametrize("mode,read,edit,bash", [
    ("ask", False, True, True),
    ("accept_edits", False, False, True),
    ("plan", False, None, True),          # None: edit is blocked outright
    ("auto", False, False, False),
])
def test_mode_matrix(mode, read, edit, bash):
    p = Permissions(mode=mode)
    assert p.needs_approval(READ, {}) is read
    assert p.needs_approval(BASH, LS) is bash
    if edit is None:
        assert "Plan mode" in p.blocked(EDIT, {})
    else:
        assert p.blocked(EDIT, {}) is None and p.needs_approval(EDIT, {}) is edit


def test_plan_mode_ignores_the_allowlist():
    p = Permissions(mode="ask")
    p.remember(BASH, LS)
    assert not p.needs_approval(BASH, LS)
    p.mode = "plan"
    assert p.needs_approval(BASH, LS)


def test_cycle():
    p = Permissions()
    assert [p.cycle() for _ in range(3)] == ["accept_edits", "plan", "ask"]
    yolo = Permissions(mode="auto", allow_auto=True)
    assert [yolo.cycle() for _ in range(4)] == ["ask", "accept_edits", "plan", "auto"]


def test_agent_refuses_edits_in_plan_mode(ctx, model):
    (ctx.cwd / "a.py").write_text("x = 1\n")
    provider = ScriptedProvider([
        call("read_file", "t1", path="a.py"),
        call("edit_file", "t2", path="a.py", old_string="x = 1", new_string="x = 2"),
        reply("here is my plan"),
    ])
    ui = RecordingUI()
    agent = Agent(provider, model, ctx, ui, Permissions(mode="plan"))
    agent.run("plan a change")
    assert (ctx.cwd / "a.py").read_text() == "x = 1\n"
    result = provider.requests[2][-1].content[0]
    assert result.is_error and "Plan mode" in result.content
    assert not any(e[0] == "confirm" for e in ui.events)   # refused, not asked
