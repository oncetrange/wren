"""--time-limit: reminders as the limit nears, and stopping when it passes."""

import json
import subprocess
import sys

from conftest import RecordingUI, ScriptedProvider, reply

from wren.agent import loop
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.config import ModelConfig
from wren.llm.types import Message, Response, TextBlock, ToolUseBlock, Usage


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Slow(ScriptedProvider):
    """Each model call takes `minutes` on the fake clock."""

    def __init__(self, turns, clock, minutes):
        super().__init__(turns)
        self.clock, self.minutes = clock, minutes

    def stream(self, *, system, messages, tools):
        self.clock.now += self.minutes * 60
        yield from super().stream(system=system, messages=messages, tools=tools)


def glob(i):
    return Response(Message("assistant", [ToolUseBlock(f"g{i}", "glob", {"pattern": "*"})]), "tool_use",
                    Usage(10, 5))


def reminders(agent):
    return [(i, b.text) for i, r in enumerate(agent.provider.requests) for b in r[-1].content
            if isinstance(b, TextBlock) and "minutes" in b.text]


def test_reminds_then_stops(ctx, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop.time, "monotonic", clock)
    agent = Agent(Slow([glob(i) for i in range(20)], clock, minutes=10), ModelConfig(name="f", model="f"),
                  ctx, RecordingUI(), Permissions(mode="auto"))
    agent.set_time_limit(100)
    agent.run("work")
    # Calls end at 10, 20, ... minutes: 80 → 20 left (≤25%): check; 90 → 10 left (≤10%): wrap up.
    notes = reminders(agent)
    assert [i for i, _ in notes] == [8, 9]
    assert "about 20 minutes are left of this run's 100-minute limit" in notes[0][1]
    assert "About 10 minutes are left before you are stopped. Wrap up now" in notes[1][1]
    assert agent.status == "time_limit" and len(agent.provider.requests) == 10
    assert ("notice", "stopped at the 100-minute time limit") in agent.ui.events


def test_no_limit_no_reminders(ctx, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop.time, "monotonic", clock)
    agent = Agent(Slow([glob(0), glob(1), reply("done")], clock, minutes=60), ModelConfig(name="f", model="f"),
                  ctx, RecordingUI(), Permissions(mode="auto"))
    assert agent.run("work") == "done" and not reminders(agent)


def test_short_limits_warn_three_minutes_ahead(ctx, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(loop.time, "monotonic", clock)
    agent = Agent(Slow([glob(i) for i in range(10)], clock, minutes=1), ModelConfig(name="f", model="f"),
                  ctx, RecordingUI(), Permissions(mode="auto"))
    agent.set_time_limit(10)
    agent.run("work")
    notes = reminders(agent)
    # 25% of 10 min is 2.5 min, but the wrap-up comes at 3 minutes left and covers both.
    assert len(notes) == 1 and "About 3 minutes are left" in notes[0][1]


def test_cli_flag_and_result(tmp_path):
    import os

    from fake_anthropic import FakeAnthropic, text_turn
    home, project = tmp_path / "home", tmp_path / "p"
    home.mkdir()
    project.mkdir()
    with FakeAnthropic([text_turn("hi")]) as server:
        (home / "config.toml").write_text(f'[models.fake]\nmodel = "m"\nbase_url = "{server.url}"\n'
                                          'api_key_env = "K"\nprompt_cache = false\n')
        proc = subprocess.run([sys.executable, "-m", "wren.cli.main", "-p", "hi", "-m", "fake", "--time-limit", "30",
                               "--no-final-check", "--output-format", "json"], cwd=project,
                              env={**os.environ, "WREN_HOME": str(home), "K": "k"},
                              capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "done"
    log = next((home / "sessions").glob("*.jsonl")).read_text().splitlines()
    assert any(json.loads(line) == {**json.loads(line), "kind": "time_limit", "minutes": 30.0} for line in log)
