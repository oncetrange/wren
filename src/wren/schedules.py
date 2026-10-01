"""Scheduled runs: prompts that run headless on a cron schedule.

The system scheduler (launchd on macOS, crontab elsewhere) runs
`wren schedule tick` every minute. A tick reads the jobs, and starts each job
that came due since the last tick as a detached `wren schedule exec <id>`,
which runs `wren -p` in the job's directory and records the result.

  ~/.wren/schedules.json              the jobs (hand-editable)
  ~/.wren/schedules/state.json        when each job was last checked
  ~/.wren/schedules/<id>/runs.jsonl   one line per run (or skipped/missed run)
  ~/.wren/schedules/<id>/output.log   the runs' progress output
  ~/.wren/env                         KEY=value lines for scheduled runs, which
                                      don't see your shell's environment

A run that comes due while the machine sleeps runs once when it wakes
(`catch_up`, on by default; otherwise runs over LATE_LIMIT late are skipped).
A run that comes due while the previous one is still going is skipped.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from wren.config import CONFIG_DIR
from wren.credentials import read_env_file
from wren.cron import Cron, CronError

LATE_LIMIT = timedelta(minutes=5)
MAX_RESULT_CHARS = 4000
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MODES = ("plan", "accept_edits", "auto")
LAUNCHD_LABEL = "com.wren.schedule"
CRON_MARKER = "# wren-schedule"


class ScheduleError(Exception):
    pass


def now() -> datetime:
    return datetime.now().replace(second=0, microsecond=0)


@dataclass
class Job:
    id: str
    cron: str
    prompt: str
    cwd: str
    # Permission mode: plan is read-only; accept_edits may edit files; auto runs anything.
    mode: str = "plan"
    model: str | None = None
    max_turns: int = 50
    memory: bool = False
    # Pass --trust-project-hooks / --trust-project-mcp.
    trust_project: bool = False
    catch_up: bool = True
    notify: bool = True
    paused: bool = False
    created: str = field(default_factory=lambda: now().isoformat())

    def validate(self) -> None:
        if not ID_RE.match(self.id) or len(self.id) > 40:
            raise ScheduleError(f"id must be lowercase letters, digits and hyphens, got {self.id!r}")
        try:
            Cron.parse(self.cron)
        except CronError as e:
            raise ScheduleError(str(e)) from None
        if not self.prompt.strip():
            raise ScheduleError("the prompt is empty")
        if self.mode not in MODES:
            raise ScheduleError(f"mode must be one of {', '.join(MODES)}, got {self.mode!r}")
        if not Path(self.cwd).is_dir():
            raise ScheduleError(f"directory {self.cwd} doesn't exist")
        if self.max_turns < 1:
            raise ScheduleError("max_turns must be at least 1")

    def next_run(self, after: datetime | None = None) -> datetime | None:
        return Cron.parse(self.cron).next_after(after or now())


def describe_job(job: Job) -> str:
    """One line: when it runs next, how and where."""
    nxt = job.next_run()
    when = f"next {nxt:%a %Y-%m-%d %H:%M}" if nxt else "never runs"
    return f"{job.cron} ({when}) · {job.mode} · {job.cwd}"


class Schedules:
    """The job list and the scheduler's bookkeeping, under `home`."""

    def __init__(self, home: Path | None = None):
        self.home = home or CONFIG_DIR
        self.file = self.home / "schedules.json"
        self.dir = self.home / "schedules"

    # --- jobs ----------------------------------------------------------------------

    def jobs(self) -> list[Job]:
        try:
            raw = json.loads(self.file.read_text())
        except FileNotFoundError:
            return []
        except ValueError as e:
            raise ScheduleError(f"{self.file}: {e}") from None
        known = {f.name for f in fields(Job)}
        return [Job(**{k: v for k, v in entry.items() if k in known}) for entry in raw.get("jobs", [])]

    def get(self, job_id: str) -> Job:
        for job in self.jobs():
            if job.id == job_id:
                return job
        raise ScheduleError(f"no scheduled job {job_id!r}")

    def add(self, job: Job) -> Job:
        job.validate()
        jobs = self.jobs()
        if any(j.id == job.id for j in jobs):
            raise ScheduleError(f"a job named {job.id!r} already exists")
        self._save([*jobs, job])
        return job

    def remove(self, job_id: str) -> None:
        self.get(job_id)
        self._save([j for j in self.jobs() if j.id != job_id])

    def set_paused(self, job_id: str, paused: bool) -> None:
        jobs = self.jobs()
        for job in jobs:
            if job.id == job_id:
                job.paused = paused
                # Resuming doesn't run what came due while paused.
                self._set_checked({job_id: now()})
                self._save(jobs)
                return
        raise ScheduleError(f"no scheduled job {job_id!r}")

    def new_id(self, prompt: str) -> str:
        words = re.findall(r"[a-z0-9]+", prompt.lower())[:3] or ["job"]
        base, taken = "-".join(words)[:30].strip("-") or "job", {j.id for j in self.jobs()}
        candidate, n = base, 2
        while candidate in taken:
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def _save(self, jobs: list[Job]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"jobs": [asdict(j) for j in jobs]}, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(self.file)

    # --- bookkeeping ---------------------------------------------------------------

    def checked(self) -> dict[str, datetime]:
        try:
            raw = json.loads((self.dir / "state.json").read_text())
        except (OSError, ValueError):
            return {}
        return {k: datetime.fromisoformat(v) for k, v in raw.get("checked", {}).items()}

    def _set_checked(self, updates: dict[str, datetime]) -> None:
        state = {**self.checked(), **updates}
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "state.json").write_text(json.dumps(
            {"checked": {k: v.isoformat() for k, v in state.items()}}, indent=2) + "\n")

    def record(self, job_id: str, entry: dict[str, Any]) -> None:
        d = self.dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        with (d / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def runs(self, job_id: str, last: int = 10) -> list[dict[str, Any]]:
        path = self.dir / job_id / "runs.jsonl"
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines()[-last:]:
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        return entries


# --- the tick ----------------------------------------------------------------------


def tick(store: Schedules, at: datetime, launch: Callable[[Job], None]) -> list[tuple[str, str]]:
    """Start every job that came due since it was last checked. Returns (id, what happened)."""
    checked = store.checked()
    done: list[tuple[str, str]] = []
    updates: dict[str, datetime] = {}
    for job in store.jobs():
        updates[job.id] = at
        if job.paused:
            continue
        since = checked.get(job.id) or datetime.fromisoformat(job.created)
        try:
            due = job.next_run(after=since)
        except CronError:
            continue
        if due is None or due > at:
            continue
        if at - due > LATE_LIMIT and not job.catch_up:
            store.record(job.id, {"status": "missed", "scheduled": due.isoformat()})
            done.append((job.id, "missed"))
            continue
        launch(job)
        done.append((job.id, "started"))
    store._set_checked(updates)
    return done


def launch_detached(store: Schedules) -> Callable[[Job], None]:
    """Start `wren schedule exec <id>` in the background, outliving the tick."""
    def launch(job: Job) -> None:
        d = store.dir / job.id
        d.mkdir(parents=True, exist_ok=True)
        with (d / "output.log").open("a") as log:
            subprocess.Popen([sys.executable, "-m", "wren.cli.main", "schedule", "exec", job.id],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                             env={**os.environ, "WREN_HOME": str(store.home)})
    return launch


# --- running a job -----------------------------------------------------------------


def command(job: Job) -> list[str]:
    cmd = [sys.executable, "-m", "wren.cli.main", "-p", job.prompt, "--output-format", "json",
           "--mode", job.mode, "--max-turns", str(job.max_turns),
           "--memory" if job.memory else "--no-memory"]
    if job.model:
        cmd += ["-m", job.model]
    if job.trust_project:
        cmd += ["--trust-project-hooks", "--trust-project-mcp"]
    return cmd


def execute(store: Schedules, job: Job, notify_fn: Callable[[str, str], None] | None = None,
            timeout: float = 4 * 3600) -> dict[str, Any]:
    """Run the job once, unless its previous run is still going; record and return the result."""
    import fcntl

    d = store.dir / job.id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            entry = {"status": "skipped", "started": now().isoformat(),
                     "error": "the previous run was still going"}
            store.record(job.id, entry)
            return entry
        started = datetime.now()
        env = {**read_env_file(store.home), **os.environ, "WREN_HOME": str(store.home)}
        try:
            proc = subprocess.run(command(job), cwd=job.cwd, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=timeout)
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout, stderr, code = "", f"timed out after {timeout:.0f}s\n{e.stderr or ''}", -1
        except OSError as e:
            stdout, stderr, code = "", str(e), -1
        with (d / "output.log").open("a", encoding="utf-8") as log:
            log.write(f"--- {started:%Y-%m-%d %H:%M:%S} ---\n{stderr}\n")
        entry = _result_entry(stdout, stderr, code, started)
    store.record(job.id, entry)
    if job.notify:
        (notify_fn or notify)(f"wren: {job.id} {entry['status']}",
                              (entry.get("result") or entry.get("error") or "")[:200])
    return entry


def _result_entry(stdout: str, stderr: str, code: int, started: datetime) -> dict[str, Any]:
    entry: dict[str, Any] = {"started": started.isoformat(timespec="seconds"),
                             "finished": datetime.now().isoformat(timespec="seconds"), "exit_code": code}
    try:
        out = json.loads(stdout)
    except ValueError:
        out = None
    if isinstance(out, dict):
        entry.update(status=out.get("status", "error"), result=(out.get("result") or "")[:MAX_RESULT_CHARS],
                     session_id=out.get("session_id"), turns=out.get("turns"), cost_usd=out.get("cost_usd"))
    else:
        entry.update(status="error")
    if code != 0 and entry["status"] == "done":
        entry["status"] = "error"
    if entry["status"] != "done":
        entry["error"] = "\n".join(stderr.strip().splitlines()[-5:])
    return entry


def notify(title: str, message: str) -> None:
    """A desktop notification, where there's a way to send one."""
    try:
        if sys.platform == "darwin":
            script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", title, message], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


# --- the system scheduler ----------------------------------------------------------


def tick_command() -> list[str]:
    return [sys.executable, "-m", "wren.cli.main", "schedule", "tick"]


def launchd_plist(home: Path) -> bytes:
    log = str(home / "logs" / "schedule.log")
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": tick_command(),
        "StartInterval": 60,
        "RunAtLoad": True,
        # Scheduled runs need the tools you use (git, uv, npx...) on PATH.
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "WREN_HOME": str(home)},
        "StandardOutPath": log,
        "StandardErrorPath": log,
    })


def crontab_line(home: Path) -> str:
    env = f"PATH={shlex.quote(os.environ.get('PATH', '/usr/bin:/bin'))} WREN_HOME={shlex.quote(str(home))}"
    log = shlex.quote(str(home / "logs" / "schedule.log"))
    return f"* * * * * {env} {shlex.join(tick_command())} >> {log} 2>&1 {CRON_MARKER}"


class Scheduler:
    """Installs `wren schedule tick` in the system scheduler."""

    def __init__(self, home: Path | None = None, platform: str = sys.platform,
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.home = home or CONFIG_DIR
        self.platform, self._run = platform, run

    @property
    def plist(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"

    def installed(self) -> bool:
        if self.platform == "darwin":
            return self.plist.exists()
        return CRON_MARKER in self._crontab()

    def install(self) -> str:
        (self.home / "logs").mkdir(parents=True, exist_ok=True)
        if self.platform == "darwin":
            self.plist.parent.mkdir(parents=True, exist_ok=True)
            self.plist.write_bytes(launchd_plist(self.home))
            domain = f"gui/{os.getuid()}"
            self._run(["launchctl", "bootout", domain, str(self.plist)], capture_output=True)
            proc = self._run(["launchctl", "bootstrap", domain, str(self.plist)], capture_output=True, text=True)
            if proc.returncode != 0:
                raise ScheduleError(f"launchctl bootstrap failed: {(proc.stderr or '').strip()}")
            return f"installed a launchd agent ({self.plist}) that runs every minute"
        if not shutil.which("crontab") and self._run is subprocess.run:
            raise ScheduleError("no crontab command: install cron, or run `wren schedule tick` every "
                                "minute some other way")
        lines = [l for l in self._crontab().splitlines() if CRON_MARKER not in l]
        self._write_crontab("\n".join([*lines, crontab_line(self.home)]) + "\n")
        return "added a crontab entry that runs every minute"

    def uninstall(self) -> str:
        if self.platform == "darwin":
            if self.plist.exists():
                self._run(["launchctl", "bootout", f"gui/{os.getuid()}", str(self.plist)], capture_output=True)
                self.plist.unlink()
            return "removed the launchd agent"
        lines = [l for l in self._crontab().splitlines() if CRON_MARKER not in l]
        self._write_crontab("\n".join(lines) + ("\n" if lines else ""))
        return "removed the crontab entry"

    def _crontab(self) -> str:
        try:
            proc = self._run(["crontab", "-l"], capture_output=True, text=True)
        except OSError:
            return ""
        return proc.stdout if proc.returncode == 0 else ""

    def _write_crontab(self, text: str) -> None:
        proc = self._run(["crontab", "-"], input=text, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ScheduleError(f"crontab failed: {(proc.stderr or '').strip()}")
