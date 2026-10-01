"""Scheduled runs: cron expressions, the tick, running a job, the installers, the tool."""

import json
import plistlib
import subprocess
import sys
from datetime import datetime, timedelta

import pytest
from conftest import RecordingUI, ScriptedProvider, reply
from fake_anthropic import FakeAnthropic, text_turn

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.cli.commands import parse_interval
from wren.cli.schedule_cmd import schedule_main
from wren.config import ModelConfig
from wren.credentials import read_env_file
from wren.cron import Cron, CronError, describe_next
from wren.llm.types import Message, Response, ToolResultBlock, ToolUseBlock, Usage
from wren.schedules import (
    CRON_MARKER,
    Job,
    ScheduleError,
    Scheduler,
    Schedules,
    command,
    crontab_line,
    execute,
    launchd_plist,
    tick,
)
from wren.tools.schedule import ScheduleTool

T = datetime(2026, 9, 25, 10, 0)  # a Friday


# --- cron --------------------------------------------------------------------------

@pytest.mark.parametrize("expr, after, expected", [
    ("0 9 * * 1-5", T, [datetime(2026, 9, 28, 9), datetime(2026, 9, 29, 9)]),        # skips the weekend
    ("*/20 * * * *", datetime(2026, 1, 1, 23, 50), [datetime(2026, 1, 2, 0, 0), datetime(2026, 1, 2, 0, 20)]),
    ("0 0 29 2 *", T, [datetime(2028, 2, 29), datetime(2032, 2, 29)]),                # leap days
    ("0 0 31 * *", T, [datetime(2026, 10, 31), datetime(2026, 12, 31)]),              # short months skipped
    ("0 12 1 * mon", datetime(2026, 9, 1), [datetime(2026, 9, 1, 12), datetime(2026, 9, 7, 12)]),  # either day
    ("30 8 * jan,jul sun", T, [datetime(2027, 1, 3, 8, 30), datetime(2027, 1, 10, 8, 30)]),
    ("0 0 * * 7", T, [datetime(2026, 9, 27), datetime(2026, 10, 4)]),                 # 7 is Sunday too
    ("5/15 * * * *", T, [datetime(2026, 9, 25, 10, 5), datetime(2026, 9, 25, 10, 20)]),
    ("@daily", T, [datetime(2026, 9, 26), datetime(2026, 9, 27)]),
    ("@hourly", T, [datetime(2026, 9, 25, 11), datetime(2026, 9, 25, 12)]),
])
def test_next_runs(expr, after, expected):
    assert describe_next(Cron.parse(expr), after, 2) == expected


@pytest.mark.parametrize("expr, error", [
    ("* * * *", "5 fields"), ("60 * * * *", "outside 0-59"), ("* * * foo *", "not a number"),
    ("5-1 * * * *", "backwards"), ("*/0 * * * *", "at least 1"), ("* * 0 * *", "outside 1-31"),
])
def test_invalid_cron(expr, error):
    with pytest.raises(CronError, match=error):
        Cron.parse(expr)


def test_never_matching_day():
    assert Cron.parse("0 0 30 2 *").next_after(T) is None


# --- jobs and the tick -------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return Schedules(tmp_path / "home")


def job(store, tmp_path, **kw):
    fields = dict(id="check-ci", cron="0 9 * * *", prompt="check CI", cwd=str(tmp_path),
                  created=datetime(2026, 9, 24, 12).isoformat())
    return store.add(Job(**{**fields, **kw}))


def test_store_round_trip_and_validation(store, tmp_path):
    job(store, tmp_path)
    assert store.get("check-ci").cron == "0 9 * * *"
    with pytest.raises(ScheduleError, match="already exists"):
        job(store, tmp_path)
    for bad, error in [({"id": "Bad Id"}, "lowercase"), ({"id": "x", "cron": "nope"}, "5 fields"),
                       ({"id": "y", "mode": "yolo"}, "mode"), ({"id": "z", "cwd": "/nonexistent"}, "exist")]:
        with pytest.raises(ScheduleError, match=error):
            job(store, tmp_path, **bad)
    assert store.new_id("Check the CI, please!") == "check-the-ci"
    store.remove("check-ci")
    assert store.jobs() == []


def test_tick_runs_due_jobs_once(store, tmp_path):
    job(store, tmp_path)
    started = []
    assert tick(store, datetime(2026, 9, 25, 8, 59), started.append) == []
    assert tick(store, datetime(2026, 9, 25, 9, 0), started.append) == [("check-ci", "started")]
    assert tick(store, datetime(2026, 9, 25, 9, 1), started.append) == []  # not twice
    assert [j.id for j in started] == ["check-ci"]


def test_missed_runs_catch_up_once_or_are_skipped(store, tmp_path):
    job(store, tmp_path)
    job(store, tmp_path, id="strict", catch_up=False)
    started = []
    # The machine slept through two 9:00 runs; wakes at 13:00 on the 27th.
    result = tick(store, datetime(2026, 9, 27, 13, 0), started.append)
    assert result == [("check-ci", "started"), ("strict", "missed")]
    assert store.runs("strict")[-1] == {"status": "missed", "scheduled": "2026-09-25T09:00:00"}
    # A few minutes late still counts as on time.
    assert tick(store, datetime(2026, 9, 28, 9, 3), started.append) == [
        ("check-ci", "started"), ("strict", "started")]


def test_paused_jobs_dont_run_and_resume_forward(store, tmp_path):
    job(store, tmp_path)
    store.set_paused("check-ci", True)
    started = []
    assert tick(store, datetime(2026, 9, 25, 9, 0), started.append) == []
    store.set_paused("check-ci", False)  # doesn't run what came due while paused
    assert tick(store, datetime(2026, 9, 25, 9, 30), started.append) == []
    assert tick(store, datetime(2026, 9, 26, 9, 0), started.append) == [("check-ci", "started")]


# --- running a job -----------------------------------------------------------------

def test_command_flags(tmp_path):
    j = Job("x", "@daily", "go", str(tmp_path), mode="accept_edits", model="qwen", memory=True,
            trust_project=True)
    cmd = command(j)
    assert cmd[cmd.index("-p") + 1] == "go" and cmd[cmd.index("--mode") + 1] == "accept_edits"
    assert {"--memory", "--trust-project-hooks", "--trust-project-mcp", "--output-format"} <= set(cmd)
    assert cmd[-cmd[::-1].index("-m")] == "qwen"  # the last -m (the first is python -m)


def test_env_file(store):
    store.home.mkdir(parents=True)
    (store.home / "env").write_text("# keys\nexport A=1\nB = 'two'\n\nnot a line\n")
    assert read_env_file(store.home) == {"A": "1", "B": "two"}


def test_execute_runs_wren_headless_and_records(store, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    store.home.mkdir(parents=True)
    (store.home / "env").write_text("SCHED_KEY=from-env-file\n")
    monkeypatch.delenv("SCHED_KEY", raising=False)
    notes = []
    with FakeAnthropic([text_turn("CI is green.")]) as server:
        (store.home / "config.toml").write_text(
            f'default_model = "fake"\n[models.fake]\nmodel = "m"\nbase_url = "{server.url}"\n'
            'api_key_env = "SCHED_KEY"\nprompt_cache = false\nprice = { input = 1.0, output = 2.0 }\n')
        j = job(store, project, cwd=str(project))
        entry = execute(store, j, notify_fn=lambda title, text: notes.append((title, text)))
    assert entry["status"] == "done" and entry["result"] == "CI is green."
    assert entry["session_id"] and entry["cost_usd"] == pytest.approx(0.0011)
    assert store.runs("check-ci")[-1] == entry
    assert notes == [("wren: check-ci done", "CI is green.")]
    body = server.requests[0]["body"]
    assert body["messages"][0]["content"][0]["text"] == "check CI"
    assert "Plan mode is on" in json.dumps(body["messages"][0])       # plan mode by default
    assert {k.lower(): v for k, v in server.requests[0]["headers"].items()}["x-api-key"] == "from-env-file"


def test_execute_skips_while_the_previous_run_holds_the_lock(store, tmp_path):
    import fcntl
    j = job(store, tmp_path)
    (store.dir / "check-ci").mkdir(parents=True)
    with (store.dir / "check-ci" / "lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        entry = execute(store, j, notify_fn=lambda *_: None)
    assert entry["status"] == "skipped"


# --- the system scheduler ----------------------------------------------------------

def test_launchd_plist_and_crontab_line(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin")
    plist = plistlib.loads(launchd_plist(tmp_path))
    assert plist["ProgramArguments"] == [sys.executable, "-m", "wren.cli.main", "schedule", "tick"]
    assert plist["StartInterval"] == 60 and plist["EnvironmentVariables"]["PATH"] == "/opt/homebrew/bin:/usr/bin"
    assert plist["EnvironmentVariables"]["WREN_HOME"] == str(tmp_path)
    line = crontab_line(tmp_path)
    assert line.startswith("* * * * * PATH=/opt/homebrew/bin:/usr/bin ") and line.endswith(CRON_MARKER)
    assert "schedule tick" in line


def test_crontab_install_keeps_other_entries(tmp_path):
    crontab = {"text": "0 3 * * * backup.sh\n"}

    def fake_run(cmd, input=None, **kw):
        if cmd == ["crontab", "-l"]:
            return subprocess.CompletedProcess(cmd, 0, crontab["text"], "")
        crontab["text"] = input
        return subprocess.CompletedProcess(cmd, 0, "", "")

    scheduler = Scheduler(tmp_path, platform="linux", run=fake_run)
    assert not scheduler.installed()
    scheduler.install()
    scheduler.install()  # idempotent
    lines = crontab["text"].splitlines()
    assert lines[0] == "0 3 * * * backup.sh" and len(lines) == 2 and scheduler.installed()
    scheduler.uninstall()
    assert crontab["text"] == "0 3 * * * backup.sh\n"


# --- CLI and tool ------------------------------------------------------------------

def test_cli_add_list_pause_rm(store, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("wren.cli.schedule_cmd.Schedules", lambda: store)
    monkeypatch.setattr("wren.cli.schedule_cmd.Scheduler", lambda: Scheduler(tmp_path, platform="linux",
                        run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "")))
    from rich.console import Console
    console = Console(width=200, force_terminal=False)
    assert schedule_main(["add", "@daily", "summarize open issues", "--cwd", str(tmp_path)], console) == 0
    assert "the scheduler isn't running yet" in capsys.readouterr().out
    assert schedule_main(["pause", "summarize-open-issues"], console) == 0
    assert schedule_main(["list"], console) == 0
    assert "paused" in capsys.readouterr().out
    assert schedule_main(["rm", "summarize-open-issues"], console) == 0
    assert schedule_main(["rm", "summarize-open-issues"], console) == 1


def test_tool_always_asks_even_in_auto_mode(ctx, store, tmp_path):
    def use(call_id, **input):
        return Response(Message("assistant", [ToolUseBlock(call_id, "schedule", input)]), "tool_use",
                        Usage(10, 5))

    turns = [use("c1", action="create", cron="0 9 * * 1-5", prompt="check CI", id="ci"),
             use("c2", action="create", cron="@daily", prompt="again", id="denied"),
             use("c3", action="list"), use("c4", action="delete", id="ci"), reply("done")]
    ui = RecordingUI([Decision(allow=True, remember=True), Decision(allow=False, feedback="not that one"),
                      Decision(allow=True)])
    agent = Agent(ScriptedProvider(turns), ModelConfig(name="f", model="f"), ctx, ui, Permissions(mode="auto"))
    agent.tools["schedule"] = ScheduleTool(store, Scheduler(tmp_path, platform="linux",
                                           run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "")))
    agent.run("check CI every weekday at 9")
    confirms = [e for e in ui.events if e[0] == "confirm"]
    assert [c[1] for c in confirms] == ["schedule", "schedule", "schedule"]  # create, create, delete; not list
    assert "next:" in confirms[0][2] and "check CI" in confirms[0][2]
    results = {b.tool_use_id: b for m in agent.messages for b in m.content if isinstance(b, ToolResultBlock)}
    assert "Scheduled 'ci'" in results["c1"].content and "wren schedule install" in results["c1"].content
    assert results["c2"].is_error                          # denied, despite "always" before
    assert "- ci:" in results["c3"].content
    assert store.jobs() == []                              # deleted


def test_parse_interval():
    assert parse_interval("10m") == 600 and parse_interval("2h") == 7200 and parse_interval("1d") == 86400
    assert parse_interval("30s") is None and parse_interval("5") is None and parse_interval("90s") == 90
    assert timedelta(seconds=parse_interval("1m") or 0) == timedelta(minutes=1)
