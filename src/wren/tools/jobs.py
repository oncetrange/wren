"""Background commands: dev servers, watchers and long builds that keep running.

`bash` with `run_in_background` starts one and returns at once; `bash_output`
reads what it printed since the last read (optionally waiting for more or for
it to finish) and `kill_job` stops it. Each job runs in its own process group
with its output in a temporary file. wren stops every job when it exits (and a
subagent's jobs when the subagent finishes), so nothing outlives the session.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wren.tools.base import MAX_OUTPUT_CHARS, Tool, ToolContext, ToolError, ToolOutput

MAX_WAIT = 120  # seconds bash_output may wait
STARTUP_WAIT = 1.5  # seconds to watch a new job, so immediate failures show up


@dataclass
class Job:
    id: str
    command: str
    proc: subprocess.Popen
    log: Path
    started: float = field(default_factory=time.monotonic)
    read_pos: int = 0
    reported: bool = False  # the model has been told it finished

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    @property
    def status(self) -> str:
        code = self.proc.poll()
        return "running" if code is None else f"exited with code {code}"

    @property
    def runtime(self) -> str:
        seconds = int(time.monotonic() - self.started)
        return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class Jobs:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._dir: Path | None = None
        self._next = 1
        atexit.register(self.kill_all)  # in case the session ends without cleaning up

    def start(self, command: str, cwd: Path, env: dict[str, str]) -> Job:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="wren-jobs-"))
        job_id = f"bg{self._next}"
        self._next += 1
        log = self._dir / f"{job_id}.log"
        with log.open("wb") as out:
            proc = subprocess.Popen(["bash", "-c", command], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        job = Job(job_id, command, proc, log)
        self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id.strip())
        if job is None:
            known = ", ".join(self.jobs) or "none"
            raise ToolError(f"no background job {job_id!r} (jobs: {known})")
        return job

    def read(self, job: Job, limit: int = MAX_OUTPUT_CHARS) -> str:
        """Output since the last read; only the end of it if there's a lot."""
        with job.log.open("rb") as f:
            f.seek(job.read_pos)
            data = f.read()
        job.read_pos += len(data)
        text = data.decode(errors="replace")
        if len(text) > limit:
            text = f"[{len(text) - limit} earlier characters skipped]\n" + text[-limit:]
        return text

    def kill(self, job: Job) -> None:
        if not job.running:
            return
        for sig, grace in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
            try:
                os.killpg(job.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                job.proc.wait(grace)
                return
            except subprocess.TimeoutExpired:
                continue

    def kill_all(self) -> None:
        for job in self.jobs.values():
            self.kill(job)

    def running(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.running]

    def newly_finished(self) -> list[Job]:
        """Jobs that ended since the model was last told; marks them told."""
        done = [j for j in self.jobs.values() if not j.running and not j.reported]
        for j in done:
            j.reported = True
        return done


def start_in_background(command: str, ctx: ToolContext, env: dict[str, str]) -> ToolOutput:
    job = ctx.jobs.start(command, ctx.bash_cwd, env)
    deadline = time.monotonic() + STARTUP_WAIT
    while job.running and time.monotonic() < deadline:
        time.sleep(0.1)
    output = ctx.jobs.read(job)
    if not job.running:
        job.reported = True
        text = f"Background job {job.id} already {job.status}.\n{output}".rstrip()
        return ToolOutput(text, is_error=job.proc.returncode != 0, summary=f"{job.id} {job.status}")
    text = (f"Started background job {job.id} (it keeps running; read its output with "
            f"bash_output, stop it with kill_job).")
    if output.strip():
        text += f"\nOutput so far:\n{output}"
    return ToolOutput(text, summary=f"started {job.id}")


class BashOutput(Tool):
    name = "bash_output"
    description = (
        "Read new output from a background job started with bash run_in_background: what it "
        "printed since the last read, and whether it is still running. `wait` waits up to that "
        f"many seconds (max {MAX_WAIT}) for the job to finish, or until output matching `until` "
        "(a regex) appears, e.g. a server's 'listening on' line."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The job id, e.g. bg1"},
            "wait": {"type": "integer", "description": "Seconds to wait (default 0)"},
            "until": {"type": "string", "description": "Stop waiting once new output matches this regex"},
        },
        "required": ["id"],
    }
    read_only = True

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("id", "") + (f" (wait {args['wait']}s)" if args.get("wait") else "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        job = ctx.jobs.get(args["id"])
        try:
            pattern = re.compile(args["until"]) if args.get("until") else None
        except re.error as e:
            raise ToolError(f"invalid `until` regex: {e}") from None
        deadline = time.monotonic() + min(max(args.get("wait", 0), 0), MAX_WAIT)
        seen = ""
        while job.running and time.monotonic() < deadline:
            if pattern:
                seen = _peek(job)
                if pattern.search(seen):
                    break
            time.sleep(0.2)
        output = ctx.jobs.read(job)
        if not job.running:
            job.reported = True
        lines = output.count("\n") + (1 if output and not output.endswith("\n") else 0)
        text = f"[{job.id} {job.status} after {job.runtime}]\n{output or '(no new output)'}"
        return ToolOutput(text, summary=f"{job.status}, {lines} new line{'s' if lines != 1 else ''}")


def _peek(job: Job) -> str:
    with job.log.open("rb") as f:
        f.seek(job.read_pos)
        return f.read().decode(errors="replace")


class KillJob(Tool):
    name = "kill_job"
    description = "Stop a background job started with bash run_in_background (and anything it started)."
    input_schema = {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "The job id, e.g. bg1"}},
        "required": ["id"],
    }
    read_only = True  # it only stops what wren started

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("id", "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        job = ctx.jobs.get(args["id"])
        was = job.status
        ctx.jobs.kill(job)
        job.reported = True
        tail = ctx.jobs.read(job, limit=2000)
        text = f"{job.id} was {was}; now {job.status}." + (f"\nLast output:\n{tail}" if tail.strip() else "")
        return ToolOutput(text, summary=f"stopped {job.id}" if was == "running" else was)
