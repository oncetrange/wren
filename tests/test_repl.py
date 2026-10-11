"""The interactive loop end to end, driven by keystrokes: prompts, slash commands, exit."""

import io

import pytest
from conftest import ScriptedProvider, call, reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.checkpoint import Checkpoints
from wren.cli import commands, repl
from wren.cli.repl import Repl
from wren.cli.ui import RichUI
from wren.config import ModelConfig, load_config
from wren.settings import Settings


@pytest.fixture
def session(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(repl, "CONFIG_DIR", tmp_path / "home")      # history goes here, not ~/.wren

    def run(keys, turns=()):
        out = io.StringIO()
        ui = RichUI(Console(file=out, width=120, color_system=None), interactive=True)
        agent = Agent(ScriptedProvider(list(turns)), ModelConfig(name="fake", model="fake-1"), ctx, ui,
                      Permissions())
        r = Repl(agent, ui, load_config(tmp_path / "none.toml"), Settings(suggestions=False))
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            inp.send_text(keys)
            code = r.loop()
        return code, out.getvalue(), agent
    return run


def test_prompt_commands_and_exit(session):
    code, out, agent = session("hello\r/cost\r/nope\r/todos\r/exit\r", turns=[reply("hi there")])
    assert code == 0
    assert "hi there" in out                                   # the prompt ran
    assert "input 10 · output 5" in out                        # /cost
    assert "unknown command /nope; see /help" in out
    assert "no task list" in out
    assert agent.provider.requests[0][0].content[0].text == "hello"


def test_alias_and_eof(session):
    assert session("/quit\r")[0] == 0
    assert session("\x04")[0] == 0                            # Ctrl-D


def test_help_lists_every_command(session):
    _, out, _ = session("/help\r/exit\r")
    for c in commands.COMMANDS:
        assert c.name in out
    assert "/quit" not in [c.name for c in commands.COMMANDS] and "quit" in commands.BUILTIN_NAMES


def test_registry():
    names = [c.name for c in commands.COMMANDS]
    assert len(names) == len(set(names)) and names[0] == "/undo" and names[-1] == "/exit"
    assert commands.find("/quit") is commands.find("/exit")
    assert commands.find("/missing") is None
    assert commands.MENU[0] == ("/undo", commands.COMMANDS[0].help)


def test_commands_do_not_reach_the_model(session):
    _, out, agent = session("/cost\r/exit\r")
    assert "input 0" in out and not agent.provider.requests


def test_diff_shows_what_changed_since_the_last_prompt_or_the_session(ctx, tmp_path_factory, monkeypatch):
    tmp_path = tmp_path_factory.mktemp("home")   # outside the project, whose files are compared
    monkeypatch.setattr(repl, "CONFIG_DIR", tmp_path)
    (ctx.cwd / "a.txt").write_text("one\n")
    out = io.StringIO()
    ui = RichUI(Console(file=out, width=120, color_system=None), interactive=True)
    turns = [call("write_file", "t1", path="b.txt", content="new\n"), reply("made b"),
             call("bash", "t2", command="echo two > a.txt"), reply("changed a")]
    agent = Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="fake-1"), ctx, ui,
                  Permissions(mode="auto"), checkpoints=Checkpoints(ctx.cwd, root=tmp_path / "shadow"))
    r = Repl(agent, ui, load_config(tmp_path / "none.toml"), Settings(suggestions=False))
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("/diff\rmake b\rchange a\r/diff\r/diff all\r/diff all b.txt\r/exit\r")
        r.loop()
    text = out.getvalue()
    assert "no prompt sent yet" in text
    last, whole, narrowed = text.split("changes since")[1:4]
    assert "before: change a" in last and "+two" in last and "b.txt" not in last
    assert "start of this session" in whole and "2 files changed" in whole
    assert "1 file changed" in narrowed and "a.txt" not in narrowed
