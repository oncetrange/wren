import io
import json

import pytest
from rich.console import Console

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.plans import PlanDecision, is_reminder
from wren.agent.session import SessionLog, load_session
from wren.checkpoint import Checkpoints
from wren.cli.ui import RichUI
from wren.config import ModelConfig
from wren.llm.types import TextBlock, ToolResultBlock
from wren.tools import ToolContext

from conftest import RecordingUI, ScriptedProvider, call, reply

PLAN = "# Bump x\n\n1. Change `x = 1` to `x = 2` in a.py\n2. Verify by reading it back"


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "p").mkdir()
    ctx = ToolContext(cwd=(tmp_path / "p").resolve())
    (ctx.cwd / "a.py").write_text("x = 1\n")
    return ctx


def make_agent(ctx, tmp_path, turns, plan_decisions=(), mode="plan"):
    return Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f"), ctx,
                 RecordingUI(plan_decisions=list(plan_decisions)), Permissions(mode=mode),
                 log=SessionLog(directory=tmp_path / "sessions"),
                 checkpoints=Checkpoints(ctx.cwd, root=tmp_path / "shadow"))


def plan_then_edit():
    return [
        call("read_file", "t1", path="a.py"),
        call("exit_plan_mode", "t2", plan=PLAN),
        call("edit_file", "t3", path="a.py", old_string="x = 1", new_string="x = 2"),
        reply("done"),
    ]


def last_user(provider):
    return provider.requests[-1][-1]


def test_approval_ends_plan_mode_and_saves_the_plan(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, plan_then_edit(), [PlanDecision(True, "accept_edits")])
    assert agent.run("bump x") == "done"
    assert agent.permissions.mode == "accept_edits"
    assert (ctx.cwd / "a.py").read_text() == "x = 2\n"          # edited in the same run
    assert agent.plan_file.parent == ctx.cwd / ".wren" / "plans"
    assert agent.plan_file.name.endswith("-bump-x.md") and agent.plan_file.read_text().startswith("# Bump x")
    assert (ctx.cwd / ".wren" / ".gitignore").read_text() == "*\n!hooks.toml\n!skills/\n!skills/**\n!agents/\n!agents/**\n"
    result = agent.provider.requests[2][-1].content[0]
    assert "approved" in result.content and "todo_write" in result.content


def test_feedback_keeps_planning(ctx, tmp_path):
    turns = [call("exit_plan_mode", "t1", plan=PLAN), call("exit_plan_mode", "t2", plan=PLAN + "\n3. Add a test"),
             reply("ok")]
    agent = make_agent(ctx, tmp_path, turns, [PlanDecision(False, feedback="also add a test"),
                                              PlanDecision(True, "ask")])
    agent.run("bump x")
    second_request = agent.provider.requests[1][-1]
    assert "did not approve" in second_request.content[0].content
    assert second_request.content[-1] == TextBlock("also add a test")   # the user's own words
    assert agent.permissions.mode == "ask"


def test_rejection_without_feedback_stops(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [call("exit_plan_mode", "t1", plan=PLAN), reply("never")],
                       [PlanDecision(False)])
    agent.run("bump x")
    assert len(agent.provider.requests) == 1 and agent.permissions.mode == "plan"
    assert agent.plan_file is None


def test_headless_saves_plan_and_stops(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, plan_then_edit())      # review_plan -> None
    agent.run("bump x")
    assert agent.plan_text == PLAN and agent.plan_file.exists()
    assert (ctx.cwd / "a.py").read_text() == "x = 1\n"
    assert agent.permissions.mode == "plan"


def test_exit_plan_mode_outside_plan_mode(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [call("exit_plan_mode", "t1", plan=PLAN), reply("ok")], mode="ask")
    agent.run("x")
    result = agent.provider.requests[1][-1].content[0]
    assert result.is_error and "not on" in result.content


def test_reminders_are_sent_and_hidden(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [reply("a"), reply("b"), reply("c")])
    agent.run("look around")
    assert is_reminder(agent.messages[0].content[1].text) and "Plan mode is on" in agent.messages[0].content[1].text
    agent.permissions.mode = "accept_edits"
    agent.run("now do it")
    assert "Plan mode is off" in agent.messages[2].content[1].text
    agent.run("more")
    assert len(agent.messages[4].content) == 1                   # told once, not every time

    assert load_session(agent.log.path).first_prompt == "look around"
    out = io.StringIO()
    RichUI(Console(file=out, width=80, color_system=None), interactive=False).render_history(
        agent.messages, agent.tools, ctx)
    assert "look around" in out.getvalue() and "Plan mode" not in out.getvalue()


def test_undo_keeps_saved_plans(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, plan_then_edit(), [PlanDecision(True, "accept_edits")])
    agent.run("bump x")
    agent.rewind(0)
    assert (ctx.cwd / "a.py").read_text() == "x = 1\n"          # edit undone
    assert agent.plan_file.exists()                              # plan kept (.wren is ignored)


def test_headless_cli_plan(tmp_path):
    import os
    import subprocess
    import sys

    from fake_anthropic import FakeAnthropic, text_turn, tool_turn

    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir(), project.mkdir()
    (project / "a.py").write_text("x = 1\n")
    turns = [tool_turn("read_file", {"path": "a.py"}), tool_turn("exit_plan_mode", {"plan": PLAN})]
    with FakeAnthropic(turns) as server:
        (home / "config.toml").write_text(
            f'[models.fake]\nmodel = "f"\nbase_url = "{server.url}"\napi_key_env = "K"\nprompt_cache = false\n')
        proc = subprocess.run(
            [sys.executable, "-m", "wren.cli.main", "-p", "bump x", "--plan", "-m", "fake",
             "--output-format", "json"],
            cwd=project, env={**os.environ, "WREN_HOME": str(home), "K": "k"},
            capture_output=True, text=True, timeout=60)
    out = json.loads(proc.stdout)
    assert proc.returncode == 0 and out["plan"] == PLAN
    assert out["plan_file"].startswith(str(project / ".wren" / "plans"))
    assert (project / "a.py").read_text() == "x = 1\n"
