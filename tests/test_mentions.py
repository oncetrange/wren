"""@-mentions: finding and attaching files, and completing paths in the input."""

import subprocess

import pytest
from conftest import RecordingUI, ScriptedProvider, reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from wren import mentions
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.cli.completion import SlashMenu, mention_prefix
from wren.cli.main import run_prompt
from wren.config import ModelConfig
from wren.llm.types import TextBlock
from wren.mentions import FileIndex
from wren.tools import ToolContext


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "main.py").write_text("".join(f"line {i}\n" for i in range(1, 101)))
    (tmp_path / "src" / "app" / "util.py").write_text("def f():\n    return 1\n")
    (tmp_path / "README.md").write_text("# Readme\n")
    (tmp_path / "notes with spaces.md").write_text("spaced\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\0\0\0")
    return tmp_path


def test_find(project):
    text = ('look at @src/app/main.py#L10-12, @README.md. and @"notes with spaces.md" '
            "but not me@example.com or @missing.py; @src/app/ too, @src/app/main.py#L10-12 again")
    found = mentions.find(text, project)
    assert [(m.path.relative_to(project).as_posix(), m.start, m.end) for m in found] == [
        ("src/app/main.py", 10, 12), ("README.md", None, None), ("notes with spaces.md", None, None),
        ("src/app", None, None)]


def test_attach(project):
    ctx = ToolContext(cwd=project)
    found = mentions.find("@src/app/main.py#L10-12 @src/app/ @logo.png @README.md", project)
    files, shown = mentions.attach(found, ctx)
    assert files[0] == ('<wren-file path="src/app/main.py#L10-12">\n'
                        "    10→line 10\n    11→line 11\n    12→line 12\n\n"
                        "(showing lines 10-12 of 100; use offset to read more)\n</wren-file>")
    assert files[1] == '<wren-file path="src/app/">\nmain.py\nutil.py\n</wren-file>'
    assert "# Readme" in files[2]
    assert shown[2] == ("logo.png", "not attached: logo.png looks like a binary file")
    assert (project / "README.md").resolve() in ctx.read_files       # editable without re-reading


def test_run_prompt_attaches_mentions(project):
    agent = Agent(ScriptedProvider([reply("ok")]), ModelConfig(name="f", model="f"), ToolContext(cwd=project),
                  RecordingUI(), Permissions())
    shown = []
    agent.ui.attached = lambda label, summary: shown.append((label, summary))
    run_prompt(agent, "what does @src/app/util.py do?")
    blocks = agent.provider.requests[0][0].content
    assert blocks[0] == TextBlock("what does @src/app/util.py do?")
    assert blocks[1].text.startswith('<wren-file path="src/app/util.py">') and "return 1" in blocks[1].text
    assert shown == [("src/app/util.py", "read 2 lines")]


def test_file_index(project):
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / ".gitignore").write_text("ignored/\n")
    (project / "ignored").mkdir()
    (project / "ignored" / "x.py").write_text("")
    index = FileIndex(project)
    paths = index.paths()
    assert "src/" in paths and "src/app/" in paths and "src/app/main.py" in paths
    assert not any(p.startswith("ignored") for p in paths)
    assert index.complete("src/app/m")[0] == "src/app/main.py"           # prefix
    assert index.complete("util")[0] == "src/app/util.py"                # file name
    assert index.complete("sapm")[0] == "src/app/main.py"                # fuzzy
    assert set(index.complete("")) >= {"README.md", "src/"}              # top level only
    assert "src/app/main.py" not in index.complete("")


def test_mention_prefix():
    assert mention_prefix("see @src/ap") == "src/ap"
    assert mention_prefix("@") == ""
    assert mention_prefix("mail me@exa") is None
    assert mention_prefix("see @src/app.py and more") is None


def run_keys(keys, fn):
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        return fn()


def menu_prompt(project, keys):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

    menu = SlashMenu(lambda: [("/help", "help")], FileIndex(project))
    ghosts = []
    probe = KeyBindings()

    @probe.add("c-t")
    def _(event):
        s = event.current_buffer.suggestion
        ghosts.append(s.text if s else None)

    def prompt():
        session = PromptSession(key_bindings=merge_key_bindings([menu.bindings(), probe]))
        menu.attach(session)
        return session.prompt("› ")

    return run_keys(keys, prompt), ghosts


def test_completing_a_path_with_tab_and_enter(project):
    # "src/app/" is a directory: Tab keeps no space, so the file can follow.
    text, ghosts = menu_prompt(project, "explain @src/a\x14\t" + "ma\x14\r" + " please\r")
    assert ghosts == ["pp/", "in.py"]
    assert text == "explain @src/app/main.py  please"


def test_arrows_pick_among_matches(project):
    text, _ = menu_prompt(project, "@src/app/\x1b[B\r\r")
    assert text == "@src/app/util.py "
