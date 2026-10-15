"""Messages the user types while the agent works reach the model at the next step."""

import os
import termios
import time

from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.cli.typeahead import TypeAhead
from wren.llm.types import TextBlock


def texts(message) -> list[str]:
    return [b.text for b in message.content if isinstance(b, TextBlock)]


def test_a_message_goes_in_after_the_tool_results(ctx, model):
    provider = ScriptedProvider([call("bash", "t1", command="true"), reply("ok")])
    agent = Agent(provider, model, ctx, RecordingUI(), Permissions(mode="auto"))
    agent.steer("use the v2 API instead")
    assert agent.run("go") == "ok"
    sent = provider.requests[1][-1]
    assert sent.content[0].tool_use_id == "t1"
    assert "while you were working" in texts(sent)[0]
    assert texts(sent)[1] == "use the v2 API instead"
    assert agent.queued == 0


def test_a_message_after_the_last_step_gets_an_answer_too(ctx, model):
    provider = ScriptedProvider([reply("first"), reply("second")])
    agent = Agent(provider, model, ctx, RecordingUI())
    agent.steer("and the docs")
    assert agent.run("go") == "second"
    assert texts(provider.requests[1][-1])[1] == "and the docs"


def test_a_message_keeps_the_agent_going_after_a_bare_no(ctx, model):
    provider = ScriptedProvider([call("bash", "t1", command="rm -rf build"), reply("ok")])
    ui = RecordingUI([Decision(allow=False)])
    agent = Agent(provider, model, ctx, ui)
    agent.steer("don't delete it, move it")
    assert agent.run("go") == "ok"
    assert len(provider.requests) == 2


def test_unsent_messages_are_taken_back(ctx, model):
    agent = Agent(ScriptedProvider([]), model, ctx, RecordingUI())
    agent.steer("a")
    agent.steer("b")
    assert agent.unsent() == ["a", "b"] and agent.queued == 0


# --- the keyboard ---------------------------------------------------------------


def typed(*chunks: str) -> tuple[TypeAhead, list[str]]:
    sent: list[str] = []
    ahead = TypeAhead(sent.append, fd=-1)
    for chunk in chunks:
        ahead.feed(chunk)
    return ahead, sent


def test_editing_the_line():
    ahead, sent = typed("fix tset", "\x7f\x7f\x7fest", "\n")
    assert sent == ["fix test"] and ahead.draft == ""
    ahead, _ = typed("one two three", "\x17")
    assert ahead.draft == "one two "
    ahead, _ = typed("abc", "\x15", "x")
    assert ahead.draft == "x"


def test_arrow_keys_are_ignored_and_pastes_keep_their_newlines():
    ahead, sent = typed("a\x1b[Ab\x1bOc\x1b[1;5Dd")
    assert ahead.draft == "abd" and not sent
    ahead, sent = typed("line one\nline two", "\r")
    assert sent == ["line one\nline two"]
    _, sent = typed("   \n")
    assert sent == []


def lflag(fd: int) -> int:
    # The kernel sets PENDIN itself when the mode changes with input pending.
    return termios.tcgetattr(fd)[3] & ~termios.PENDIN


def test_waiting_for_keys_leaves_the_lock_free():
    # Waiting with it held kept paused() (an approval prompt) waiting for minutes on CI.
    master, slave = os.openpty()
    try:
        with TypeAhead(lambda _: None, fd=slave) as ahead:
            time.sleep(0.05)
            samples = []
            for _ in range(200):
                samples.append(ahead._lock.locked())
                time.sleep(0.001)
            assert sum(samples) < 20
    finally:
        os.close(master)
        os.close(slave)


def test_it_reads_a_terminal_and_gives_it_back_when_paused():
    master, slave = os.openpty()
    try:
        sent: list[str] = []
        before = lflag(slave)
        with TypeAhead(sent.append, fd=slave) as ahead:
            assert not termios.tcgetattr(slave)[3] & termios.ECHO
            os.write(master, "改成 v2\n".encode())
            deadline = time.monotonic() + 2
            while not sent and time.monotonic() < deadline:
                time.sleep(0.01)
            assert sent == ["改成 v2"]
            with ahead.paused():
                assert lflag(slave) == before
                os.write(master, b"y\n")
                time.sleep(0.15)
                assert os.read(slave, 10) == b"y\n"   # left for whoever asked
            assert not termios.tcgetattr(slave)[3] & termios.ECHO
        assert lflag(slave) == before
    finally:
        os.close(master)
        os.close(slave)
