"""The full-screen session end to end: the UI runs on its thread reading keys
from a pipe, the session on the test's (main) thread."""

import contextlib
import io
import signal
import threading
import time
import traceback

import pytest
from conftest import ScriptedProvider, call, reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.checkpoint import Checkpoints
from wren.cli import fullscreen, repl
from wren.cli.fullscreen import FullscreenRepl, FullscreenUI, Item, fit_bar
from wren.cli.transcript import Transcript, TranscriptControl, soft_breaks, to_lines
from wren.cli.transcript import plain as plain_text
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
        out = io.StringIO()
        ui = FullscreenUI(Console(file=out, width=120, color_system=None))
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


def plain_lines(fragments) -> str:
    return "".join(text for _, text, *_ in fragments)


def test_prompts_commands_and_exit(session):
    code, text, r = session(lambda d: (d.line("hello"), d.line("/cost"), d.line("/nope"), d.line("/help")),
                            turns=[reply("hi there")])
    assert code == 0
    assert "› hello" in text and "hi there" in text
    assert "input 10 · output 5" in text
    assert "unknown command /nope" in text
    assert r._usage is not None and "context" in plain_lines(r._toolbar_line())  # usage is in the bar
    assert "PgUp/PgDn scroll the conversation" in text     # /help has the full-screen keys
    assert r.agent.provider.requests[0][0].content[0].text == "hello"


def test_approving_and_refusing_tool_calls(session, ctx):
    asked = []

    def script(d):
        d.send("make files\r")
        d.question()
        asked.append(plain_lines(d.r._question_header()))
        d.answer("y\r")                                    # the first write
        d.answer("no, call it c.txt\r")                    # the second: refused with a message
        wait_for(lambda: not d.r.busy)
    _, text, r = session(script, mode="ask", turns=[
        call("write_file", "t1", path="a.txt", content="a\n"),
        call("write_file", "t2", path="b.txt", content="b\n"),
        reply("ok")])
    assert (ctx.cwd / "a.txt").exists() and not (ctx.cwd / "b.txt").exists()
    assert "Allow?" in asked[0] and "always for edit" in asked[0]   # asked in the panel
    assert "Allow?" not in text and "+a" in text           # the conversation keeps the diff,
    assert "› no, call it c.txt" in text                   # and what was answered
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


def test_long_output_opens_a_pager(session, ctx):
    (ctx.cwd / "a.txt").write_text("".join(f"line {i}\n" for i in range(100)))

    def script(d):
        d.line("change it")
        d.send("/diff\r")
        wait_for(lambda: d.r.view.pager is not None and d.r.view.pager.control.position[2] > 0)
        first, last, total = d.r.view.pager.control.position
        assert first == 1 and total > 100                  # from the top
        d.send(" ")                                        # a page down
        wait_for(lambda: d.r.view.pager.control.position[0] > 1)
        d.send("G")
        wait_for(lambda: d.r.view.pager.control.position[1] == total)
        d.send("q")
        wait_for(lambda: d.r.view.pager is None and not d.r.busy)
    _, text, _ = session(script, checkpoints=True, turns=[
        call("bash", "t1", command="sed -i.bak s/line/LINE/ a.txt && rm a.txt.bak"), reply("ok")])
    assert "shown in the pager" in text and "+LINE 99" not in text


def test_ctrl_o_shows_what_calls_returned_and_whole_diffs(session):
    body = "".join(f"row {i}\n" for i in range(40))
    seen = {}

    def script(d):
        d.line("go")
        seen["short"] = plain(d.r.ui)
        d.send("\x0f")                                     # Ctrl-O
        wait_for(lambda: d.r.view.transcript.expanded)
        seen["full"] = plain(d.r.ui)
        d.send("\x0f")
        wait_for(lambda: not d.r.view.transcript.expanded)
    session(script, turns=[call("bash", "t1", command="printf 'the-%s' output"),
                           call("write_file", "t2", path="rows.txt", content=body), reply("done")])
    assert "the-output" not in seen["short"] and "the-output" in seen["full"]
    assert "Ctrl-O shows all" in seen["short"] and "+row 39" not in seen["short"]
    assert "+row 39" in seen["full"]


def test_searching_the_conversation(session):
    seen = {}

    def script(d):
        for word in ("apple one", "banana", "apple two"):
            d.line(word)
        d.send("\x06")                                     # Ctrl-F
        wait_for(lambda: d.r.searching)
        d.send("APPLE")
        wait_for(lambda: len(d.r._matches) == 2)           # any case
        lines = d.r.view.transcript.lines(d.r._width())
        newest = "".join(t for _, t, *_ in lines[d.r.view.control.current])
        d.send("\r")                                       # Enter: the older one
        wait_for(lambda: d.r._match == 0)
        older = "".join(t for _, t, *_ in lines[d.r.view.control.current])
        seen.update(newest=newest, older=older, where=plain_lines(d.r._search_line()))
        d.send("\x1b")
        wait_for(lambda: not d.r.searching)
        seen["term"] = d.r.view.control.term
        d.send("\x06/diff")                                # a slash doesn't open the command menu
        wait_for(lambda: d.r.find.text == "/diff")
        seen["menu"] = d.r.menu.items()
        d.send("\x03")                                     # Ctrl-C closes it too
        wait_for(lambda: not d.r.searching)
    session(script, turns=[reply("ok"), reply("ok"), reply("ok")])
    assert "apple two" in seen["newest"] and "apple one" in seen["older"]
    assert "1/2" in seen["where"] and seen["term"] == "" and seen["menu"] == []


def test_ctrl_s_gives_the_mouse_to_the_terminal(session):
    states = []

    def script(d):
        states.append(d.r.app.mouse_support())
        d.send("\x13")
        wait_for(lambda: d.r.selecting)
        states.append(d.r.app.mouse_support())
        assert "selecting" in plain_lines(d.r._toolbar_line())
    session(script)
    assert states == [True, False]


# --- selecting with the mouse -------------------------------------------------------


def mouse(kind, x, y, button=MouseButton.LEFT):
    return MouseEvent(Point(x, y), kind, button, frozenset())


def drag(control, start, end):
    control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, *start))
    control.mouse_handler(mouse(MouseEventType.MOUSE_MOVE, *end))
    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, *end))


def test_wrapped_lines_are_found_by_laying_out_unwrapped():
    wrapped = ["A fairly long paragraph that will", "certainly wrap.", "", " • a list item that is",
               "   long", " • short"]
    unwrapped = ["A fairly long paragraph that will certainly wrap.", "",
                 " • a list item that is long", " • short"]
    assert soft_breaks(wrapped, unwrapped) == {0: " ", 3: " "}
    assert soft_breaks(["averylongw", "ord"], ["averylongword"]) == {0: ""}   # cut inside a word
    assert soft_breaks(["│ a │", "│ b │"], ["│ a b │"]) is None                # a panel: copied as shown


def test_copied_text_joins_what_was_wrapped():
    t = Transcript(None)
    t.add(Markdown("A fairly long paragraph that will certainly wrap at forty columns wide.\n\n"
                   "- a list item that is also long enough to wrap around\n- short\n\n"
                   "```\nprint('a very long line of code that goes past forty columns')\n```"))
    lines = [plain_text(line) for line in t.lines(40)]
    last = len(lines) - 1
    text = t.text(40, (0, 0), (last, len(lines[last])))
    assert "A fairly long paragraph that will certainly wrap at forty columns wide." in text
    assert "• a list item that is also long enough to wrap around\n" in text
    assert "print('a very long line of code that goes past forty columns')" in text


def test_dragging_selects_and_copies():
    t = Transcript(None)
    for i in range(30):
        t.add(Text(f"line {i} alpha beta"))
    control = TranscriptControl(t)
    copied = []
    control.on_copy = copied.append
    content = control.create_content(40, 10)                # lines 20..29 shown

    drag(control, (5, 0), (9, 1))                           # both ends included
    assert copied == ["20 alpha beta\nline 21 al"]
    shown = control.create_content(40, 10).get_line(1)
    assert any("selection" in style for style, *_ in shown)  # highlighted

    control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 3, 2))   # a click: no selection
    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 3, 2))
    assert control.selection is None and len(copied) == 1

    control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 9, 4))   # double-click: the word
    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 9, 4))
    control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 9, 4))
    assert copied[-1] == "alpha"
    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 9, 4))
    control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 9, 4))   # triple: the line
    assert copied[-1] == "line 24 alpha beta"
    assert content.line_count == 10


def test_dragging_to_the_edge_scrolls_along():
    t = Transcript(None)
    for i in range(30):
        t.add(Text(f"line {i}"))
    control = TranscriptControl(t)
    copied = []
    control.on_copy = copied.append
    control.create_content(40, 10)
    control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 0, 5))   # "line 25"
    for _ in range(3):
        control.mouse_handler(mouse(MouseEventType.MOUSE_MOVE, 0, 0))  # at the top edge
        control.create_content(40, 10)
    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 0, 0))
    assert copied == ["\n".join([f"line {i}" for i in range(17, 25)] + ["l"])]


def test_the_bottom_bar_drops_what_matters_least_when_narrow():
    left = [Item("⏵ ask before edits", priority=0, short="⏵ ask", shorten_at=2),
            Item("copied 12 characters", priority=1),
            Item("shift+tab: mode", priority=8)]
    right = [Item("kimi", priority=5), Item("context 12.0k/200.0k (6%)", priority=6, short="6%"),
             Item("$0.0123", priority=7)]
    wide = plain_lines(fit_bar(list(left), list(right), 120))
    assert "shift+tab: mode" in wide and wide.rstrip().endswith("$0.0123") and len(wide) == 120
    narrow = plain_lines(fit_bar(list(left), list(right), 60))
    assert len(narrow) <= 60 and "copied 12 characters" in narrow and "shift+tab" not in narrow
    assert "kimi" in narrow
    tiny = plain_lines(fit_bar(list(left), list(right), 30))
    assert tiny.startswith("  ⏵ ask ·") and "copied" in tiny and "kimi" not in tiny  # shortened first


def test_completions_open_as_a_list_above_the_input(session):
    shown = {}

    def script(d):
        d.send("/m")
        wait_for(lambda: d.r.input.text == "/m")
        shown["list"] = plain_lines(d.r.menu.rows(80, text="/m"))
        shown["bar"] = plain_lines(d.r._toolbar_line())
        d.send("\x15")                                     # Ctrl-U clears the line
        wait_for(lambda: not d.r.input.text)
    session(script)
    rows = shown["list"].splitlines()
    assert rows[0].lstrip().startswith("/memory") and any(r.lstrip().startswith("/mcp") for r in rows)
    assert "↑↓ choose" in rows[-1] and "/memory" not in shown["bar"]  # the bar keeps its own line
