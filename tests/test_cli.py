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
