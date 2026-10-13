"""Slash commands driven through a scripted Screen: what they show and what
they do with the answers, without a terminal."""

import io

import pytest
from conftest import ScriptedProvider, call, reply
from rich.console import Console

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.checkpoint import Checkpoints
from wren.cli.repl import Repl
from wren.cli.screen import Screen
from wren.cli.ui import RichUI
from wren.config import ModelConfig, load_config
from wren.memory import Memories, Memory
from wren.settings import Settings


class ScriptedScreen(Screen):
    """Answers pickers from a script (by option label) and records what was asked."""

    def __init__(self, console: Console, answers: list):
        super().__init__(console, interactive=True)
        self.answers = list(answers)
        self.asked: list[tuple[str, list[str]]] = []
        self.paged: list = []
        self.programs: list[list[str]] = []

    def pick(self, message, options, default=None):
        self.asked.append((message, [label for _, label in options]))
        answer = self.answers.pop(0)
        if answer is None:
            return None
        return next(value for value, label in options if answer in label)

    def page(self, *renderables):
        self.paged.extend(renderables)
        self.print(*renderables)

    def run_program(self, argv):
        self.programs.append(argv)
        return 0


@pytest.fixture
def make(ctx, tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")   # outside the project, whose files are snapshotted

    def make(*answers, turns=(), checkpoints=False):
        out = io.StringIO()
        console = Console(file=out, width=120, color_system=None)
        ui = RichUI(console, interactive=True)
        agent = Agent(ScriptedProvider(list(turns)), ModelConfig(name="fake", model="fake-1"), ctx, ui,
                      Permissions(mode="auto"),
                      checkpoints=Checkpoints(ctx.cwd, root=home / "shadow") if checkpoints else None)
        r = Repl(agent, ui, load_config(home / "none.toml"), Settings(suggestions=False),
                 Memories.for_project(ctx.cwd, home=home))
        r.screen = ScriptedScreen(console, list(answers))
        return r, out
    return make


def test_rewind_shows_the_files_to_restore_and_asks(make, ctx):
    r, out = make("before: make b", "Yes", checkpoints=True,
                  turns=[call("write_file", "t1", path="b.txt", content="new\n"), reply("made b")])
    r.run("make b")
    assert (ctx.cwd / "b.txt").exists()
    r.command("/rewind")
    assert [q for q, _ in r.screen.asked] == ["Go back to which point?", "Proceed?"]
    assert "delete" in out.getvalue() and "b.txt" in out.getvalue()
    assert not (ctx.cwd / "b.txt").exists()
    assert r.prefill == "make b"                       # the prompt comes back to the input line
    assert r.agent.messages == []


def test_cancelling_a_picker_changes_nothing(make):
    r, _ = make(None, turns=[reply("hi")])
    r.run("hello")
    r.command("/rewind")
    assert len(r.agent.messages) == 2 and r.prefill == ""


def test_diff_goes_to_the_pager(make, ctx):
    (ctx.cwd / "a.txt").write_text("one\n")
    r, out = make(checkpoints=True, turns=[call("bash", "t1", command="echo two > a.txt"), reply("ok")])
    r.run("change a")
    r.command("/diff")
    assert r.screen.paged and "+two" in out.getvalue()


def test_model_picker_switches_model(make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r, out = make("claude-opus")
    r.command("/model")
    assert r.agent.model.name == "claude"
    assert "switched to claude" in out.getvalue()


def test_memory_can_be_edited_or_deleted(make):
    r, out = make("project/style", "Edit", "project/style", "Delete", "Yes")
    store = r.memories.project
    store.write(Memory("style", "how code is written here", "feedback", "Use tabs."))
    r.command("/memory")
    assert r.screen.programs and r.screen.programs[0][-1] == str(store.path("style"))
    assert "Use tabs." in out.getvalue()
    r.command("/memory")
    assert store.get("style") is None and "deleted" in out.getvalue()


def test_todos_print_through_the_screen(make):
    r, out = make(turns=[call("todo_write", "t1", todos=[
        {"content": "write it", "status": "completed"}, {"content": "test it", "status": "completed"}]),
        reply("ok")])
    r.run("plan")
    r.command("/todos")
    text = out.getvalue()
    assert "Tasks" in text and "write it" in text and "test it" in text
