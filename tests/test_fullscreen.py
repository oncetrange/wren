"""The full-screen session end to end: the UI runs on its thread reading keys
from a pipe, the session on the test's (main) thread."""

import contextlib
import signal
import threading
import time
import traceback

import pytest
from conftest import ScriptedProvider, call, reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.checkpoint import Checkpoints
from wren.cli import fullscreen, repl
from wren.cli.fullscreen import FullscreenRepl, FullscreenUI
from wren.cli.transcript import Transcript, TranscriptControl, to_lines
from wren.config import ModelConfig, load_config
from wren.settings import Settings

UP, CTRL_C, CTRL_D = "\x1b[A", "\x03", "\x04"


def wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def plain(ui: FullscreenUI) -> str:
    return "\n".join("".join(text for _, text, *_ in line) for line in ui.view.transcript.lines(120))


class Driver:
    """Types into the UI, waiting for the session between lines."""

    def __init__(self, r: FullscreenRepl, inp):
        self.r, self.inp, self.handled = r, inp, 0
        handle = r.handle

        def counted(text):
            try:
                return handle(text)
            finally:
                self.handled += 1
        r.handle = counted

    def line(self, text: str) -> None:
        """Send a line and wait until the session is done with it."""
        before = self.handled
        self.inp.send_text(text + "\r")
        wait_for(lambda: self.handled > before and not self.r.busy)

    def send(self, keys: str) -> None:
        self.inp.send_text(keys)

    def question(self):
        wait_for(lambda: self.r.view.question is not None)
        return self.r.view.question

    def answer(self, keys: str):
        """Answer the question being asked (the next one may follow at once)."""
        question = self.question()
        self.inp.send_text(keys)
        wait_for(lambda: self.r.view.question is not question)


@pytest.fixture
def session(ctx, tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")   # outside the project, whose files are snapshotted
    monkeypatch.setattr(repl, "CONFIG_DIR", home)
    monkeypatch.setattr(fullscreen, "CONFIG_DIR", home)

    def run(script, turns=(), mode="auto", checkpoints=False):
        ui = FullscreenUI(Console(width=120, color_system=None))
        agent = Agent(ScriptedProvider(list(turns)), ModelConfig(name="fake", model="fake-1"), ctx, ui,
                      Permissions(mode=mode),
                      checkpoints=Checkpoints(ctx.cwd, root=home / "shadow") if checkpoints else None)
        r = FullscreenRepl(agent, ui, load_config(home / "none.toml"), Settings(suggestions=False))
        failure: list[BaseException] = []
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            driver = Driver(r, inp)
            main = threading.get_ident()

            def drive():
                try:
                    wait_for(lambda: r.view.running)
                    script(driver)
                except BaseException as e:
                    failure.append(e)
                    r.inbox.put(None)                      # end the session even if it is stuck
                    r.view.call(lambda: r.view.answer(None, error=KeyboardInterrupt()))
                    with contextlib.suppress(Exception):
                        signal.pthread_kill(main, signal.SIGINT)
                finally:
                    with contextlib.suppress(AssertionError):
                        wait_for(lambda: not r.busy, timeout=5)
                    r.view.call(r.input.reset)             # Ctrl-D quits from an empty line
                    inp.send_text(CTRL_D)
            thread = threading.Thread(target=drive, daemon=True)
            thread.start()
            code = r.loop()
            thread.join()
        if failure:
            traceback.print_exception(failure[0])
            raise failure[0]
        return code, plain(ui), r
    return run


def test_prompts_commands_and_exit(session):
    code, text, r = session(lambda d: (d.line("hello"), d.line("/cost"), d.line("/nope")),
                            turns=[reply("hi there")])
    assert code == 0
    assert "› hello" in text and "hi there" in text
    assert "input 10 · output 5" in text
    assert "unknown command /nope" in text
    assert "context" in r._usage                         # the usage line went to the toolbar
    assert r.agent.provider.requests[0][0].content[0].text == "hello"


def test_approving_and_refusing_tool_calls(session, ctx):
    def script(d):
        d.send("make files\r")
        d.answer("y\r")                                    # the first write
        d.answer("no, call it c.txt\r")                    # the second: refused with a message
        wait_for(lambda: not d.r.busy)
    _, text, r = session(script, mode="ask", turns=[
        call("write_file", "t1", path="a.txt", content="a\n"),
        call("write_file", "t2", path="b.txt", content="b\n"),
        reply("ok")])
    assert (ctx.cwd / "a.txt").exists() and not (ctx.cwd / "b.txt").exists()
    assert "Allow?" in text and "+a" in text               # the question and the diff it is about
    last = r.agent.provider.requests[-1][-1].content
    assert any(getattr(b, "text", "") == "no, call it c.txt" for b in last)


def test_pickers_choose_with_arrows_and_cancel_with_escape(session, ctx):
    def script(d):
        d.line("make b")
        d.send("/rewind\r")
        assert d.question().message == "Go back to which point?"
        d.answer("\x1b")                                   # Esc: nothing happens
        wait_for(lambda: not d.r.busy)
        d.send("/rewind\r")
        d.answer("\r")                                     # the only point
        assert d.question().message == "Proceed?"
        d.answer(UP + "\r")                                # No → Yes
        wait_for(lambda: not d.r.busy)
        wait_for(lambda: d.r.input.text == "make b")      # the prompt is back in the input line
        d.send(CTRL_C)                                     # Ctrl-C clears it
        wait_for(lambda: d.r.input.text == "")
    _, text, r = session(script, checkpoints=True, turns=[
        call("write_file", "t1", path="b.txt", content="new\n"), reply("made b")])
    assert not (ctx.cwd / "b.txt").exists() and r.agent.messages == []
    assert "restored" in text


def test_typing_while_it_works_steers_and_ctrl_c_interrupts(session):
    def script(d):
        d.send("run it\r")
        wait_for(lambda: "Running bash" in str(d.r.view.status))
        d.send("also check the logs\r")                    # while the command runs: steering
        wait_for(lambda: d.r.agent.queued == 1)
        d.send(CTRL_C)                                     # stops the command
        # Not sent to the model before it stopped: back in the input line.
        wait_for(lambda: d.r.input.text == "also check the logs")
    start = time.monotonic()
    _, text, r = session(script, turns=[call("bash", "t1", command="sleep 30"), reply("never")])
    assert time.monotonic() - start < 20
    assert r.agent.status == "interrupted" and "back in the input line" in text


def test_a_steered_message_reaches_the_model(session):
    def script(d):
        d.send("run it\r")
        wait_for(lambda: "Running bash" in str(d.r.view.status))
        d.send("and then say done\r")
        wait_for(lambda: not d.r.busy and d.handled)
    _, text, r = session(script, turns=[call("bash", "t1", command="sleep 1"), reply("done")])
    assert "(sent while wren worked)" in text
    assert any(getattr(b, "text", "") == "and then say done" for b in r.agent.provider.requests[1][-1].content)


# --- the transcript ---------------------------------------------------------------


def test_lines_follow_the_width():
    block = [Text("word " * 30)]
    assert len(to_lines(block, 40, None)) > len(to_lines(block, 120, None)) > 1


def test_scrolling_back_stays_put_while_output_arrives():
    t = Transcript(None)
    control = TranscriptControl(t)
    for i in range(50):
        t.add(Text(f"line {i}"))

    def shown():
        content = control.create_content(80, 10)
        return ["".join(s for _, s, *_ in content.get_line(i)) for i in range(content.line_count)]
    assert shown()[-1] == "line 49"                        # follows the end
    control.page(1)
    assert shown()[-1] == "line 41" and control.below == 8
    t.add(Text("line 50"))
    assert shown()[-1] == "line 41" and control.below == 9 # new output doesn't move it
    control.scroll(-100)
    assert shown()[-1] == "line 50" and control.below == 0
    control.scroll(1000)
    assert shown()[0] == "line 0"
