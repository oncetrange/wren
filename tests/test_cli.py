import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from wren.cli.pickers import confirm, pick
from wren.cli.terminal import parse_osc11, register_shift_enter


def run_keys(keys: str, fn):
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        return fn()


OPTIONS = [("a", "first"), ("b", "second"), ("c", "third")]


def test_arrow_keys_and_enter():
    assert run_keys("\x1b[B\x1b[B\r", lambda: pick("?", OPTIONS)) == "c"
    assert run_keys("\x1b[A\r", lambda: pick("?", OPTIONS, default="c")) == "b"


@pytest.mark.parametrize("key", ["\x03", "\x1b"])
def test_ctrl_c_and_escape_cancel(key):
    assert run_keys(key, lambda: pick("?", OPTIONS)) is None


def test_confirm_defaults():
    assert run_keys("\r", lambda: confirm("?", default=True)) is True
    assert run_keys("\r", lambda: confirm("?")) is False


def test_shift_enter_inserts_newline():
    from prompt_toolkit import PromptSession

    from wren.cli.main import _key_bindings

    register_shift_enter()
    text = run_keys("a\x1b[13;2ub\x1b\rc\r",
                    lambda: PromptSession(key_bindings=_key_bindings()).prompt())
    assert text == "a\nb\nc"


def test_parse_osc11():
    assert parse_osc11("\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\") == "dark"
    assert parse_osc11("\x1b]11;rgb:ffff/fdfd/f6f6\x07") == "light"
    assert parse_osc11("") is None


@pytest.mark.parametrize("seq", ["\x1b[27;2;13~", "\x1b[27;5;13~", "\x1b[13;2u"])
def test_modified_enter_sequences_insert_newline(seq):
    from prompt_toolkit import PromptSession

    from wren.cli.main import _key_bindings

    register_shift_enter()
    text = run_keys(f"a{seq}b\x1b[27;6;65~\r",
                    lambda: PromptSession(key_bindings=_key_bindings()).prompt())
    assert text == "a\nb"


def test_modify_other_keys_is_scoped_to_input():
    import io

    from wren.cli.terminal import distinguish_shift_enter

    class Tty(io.StringIO):
        def isatty(self):
            return True

    out = Tty()
    with pytest.raises(KeyboardInterrupt):
        with distinguish_shift_enter(out):
            raise KeyboardInterrupt
    assert out.getvalue() == "\x1b[>4;1m\x1b[>4;0m"   # always switched back off


def test_permission_prompt_supports_cursor_keys():
    from rich.console import Console

    from wren.cli.ui import RichUI
    from wren.tools.shell import Bash

    ui = RichUI(Console(file=__import__("io").StringIO()), interactive=True)
    # type "run tsts", move left 3, insert "e": feedback becomes "run tests"
    decision = run_keys("run tsts\x1b[D\x1b[D\x1b[De\r",
                        lambda: ui.confirm(Bash(), {"command": "ls"}, "ls", None))
    assert not decision.allow and decision.feedback == "run tests"
    assert run_keys("a\r", lambda: ui.confirm(Bash(), {"command": "ls"}, "ls", None)).remember


@pytest.mark.parametrize("seq", ["\x1b[Z", "\x1b[27;2;9~"])
def test_shift_tab_sequences(seq):
    from prompt_toolkit import PromptSession

    from wren.cli.keys import newline_bindings

    register_shift_enter()
    pressed = []
    kb = newline_bindings()
    kb.add("s-tab")(lambda event: pressed.append(True))
    text = run_keys(f"a{seq}b\x1b[27;5;9~\r", lambda: PromptSession(key_bindings=kb).prompt())
    assert pressed == [True] and text == "ab"


ENTRIES = (("/release", "cut a release"), ("/resume", "switch session"), ("/help", "help"))


def menu_prompt(keys, entries=ENTRIES):
    """Run a prompt with the slash menu. Ctrl-T in `keys` records the ghost text."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

    from wren.cli.completion import SlashMenu
    from wren.cli.keys import newline_bindings

    menu = SlashMenu(lambda: list(entries))
    ghosts = []
    probe = KeyBindings()

    @probe.add("c-t")
    def _(event):
        s = event.current_buffer.suggestion
        ghosts.append(s.text if s else None)

    def prompt():  # created inside run_keys, where the pipe input is active
        session = PromptSession(key_bindings=merge_key_bindings([newline_bindings(), menu.bindings(), probe]))
        menu.attach(session)
        return session.prompt("› ")

    return run_keys(keys, prompt), ghosts


def line_text(line):
    return "".join(t for _, t in line) if line else ""


def test_slash_menu_enter_runs_the_selection():
    text, _ = menu_prompt("/re\x1b[B\r")           # down to the second match, Enter
    assert text == "/resume"


def test_slash_menu_tab_completes_for_arguments():
    text, _ = menu_prompt("/rel\t1.0\r")
    assert text == "/release 1.0"


def test_toolbar_line_shows_matches_and_the_selected_description():
    from wren.cli.completion import SlashMenu
    menu = SlashMenu(lambda: list(ENTRIES))
    assert menu.toolbar(80, text="hi") is None           # closed: the usual mode line shows
    first = line_text(menu.toolbar(80, text="/re"))
    assert "/release  cut a release" in first and "/resume" in first and "switch session" not in first
    menu.index = 1
    second = line_text(menu.toolbar(80, text="/re"))
    assert "/resume  switch session" in second and "cut a release" not in second


def test_ghost_text_follows_the_selection():
    text, ghosts = menu_prompt("/re\x14\x1b[B\x14\x1b[B\x14\r")
    assert ghosts == ["lease", "sume", "lease"]            # wraps around
    assert text == "/release"


def test_long_lists_scroll_to_the_selection():
    from wren.cli.completion import SlashMenu
    entries = [(f"/command-{i:02d}", "a fairly long description " * 3) for i in range(20)]
    menu = SlashMenu(lambda: entries)
    menu.items("/c")
    menu.index = 15
    text = line_text(menu.toolbar(60, text="/c"))
    assert "/command-15" in text and "/command-00" not in text and len(text) <= 60
    assert text.endswith("↑↓ Tab ↵")


def test_up_down_still_browse_history_without_menu():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    from wren.cli.completion import SlashMenu

    history = InMemoryHistory()
    history.append_string("earlier prompt")
    menu = SlashMenu(lambda: [("/help", "")])

    def prompt():
        return PromptSession(history=history, key_bindings=menu.bindings()).prompt("› ")
    assert run_keys("\x1b[A\r", prompt) == "earlier prompt"


def prediction_prompt(keys, prediction):
    from prompt_toolkit import PromptSession

    from wren.cli.completion import SlashMenu

    menu = SlashMenu(lambda: list(ENTRIES))
    menu.prediction = prediction

    def prompt():
        session = PromptSession(key_bindings=menu.bindings())
        menu.attach(session)
        return session.prompt("› ", pre_run=lambda: menu.show_prediction(prediction, session.default_buffer))

    return run_keys(keys, prompt)


def test_tab_accepts_the_prediction():
    assert prediction_prompt("\t\r", "run the tests") == "run the tests"
    assert prediction_prompt("run t\t\r", "run the tests") == "run the tests"   # rest of it


def test_prediction_disappears_when_typing_diverges():
    assert prediction_prompt("rub\t\r", "run the tests") == "rub"
    assert prediction_prompt("/he\t\r", "run the tests") == "/help "           # commands win
