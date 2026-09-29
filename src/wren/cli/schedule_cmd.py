"""`wren schedule ...`: manage scheduled runs (see wren/schedules.py)."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from wren.config import ConfigError, load_config
from wren.schedules import (
    MODES,
    Job,
    ScheduleError,
    Scheduler,
    Schedules,
    describe_job,
    execute,
    launch_detached,
    load_env_file,
    now,
    tick,
)


def schedule_main(argv: list[str], console: Console | None = None) -> int:
    console = console or Console(highlight=False)
    parser = argparse.ArgumentParser(prog="wren schedule", description="Run prompts on a cron schedule.")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list scheduled jobs")
    add = sub.add_parser("add", help="schedule a prompt")
    add.add_argument("cron", help='a cron expression, e.g. "0 9 * * 1-5" or @daily')
    add.add_argument("prompt")
    add.add_argument("--id", help="a name for the job (default: from the prompt)")
    add.add_argument("--cwd", default=".", help="the directory to run in (default: here)")
    add.add_argument("--mode", choices=MODES, default="plan",
                     help="plan: read-only (default); accept_edits: may edit files; auto: runs anything")
    add.add_argument("-m", "--model", help="model name from the config (default: default_model)")
    add.add_argument("--max-turns", type=int, default=50)
    add.add_argument("--memory", action="store_true", help="use long-term memory")
    add.add_argument("--trust-project", action="store_true",
                     help="run the project's hooks and MCP servers")
    add.add_argument("--no-catch-up", action="store_true",
                     help="skip runs missed while the machine was off, instead of running once")
    add.add_argument("--no-notify", action="store_true", help="no desktop notification after each run")
    for name, text in [("rm", "delete a job"), ("pause", "stop a job running"), ("resume", "start it again"),
                       ("run", "run a job now, in the foreground")]:
        sub.add_parser(name, help=text).add_argument("id")
    logs = sub.add_parser("logs", help="a job's recent runs")
    logs.add_argument("id")
    logs.add_argument("-n", type=int, default=5)
    sub.add_parser("install", help="run the scheduler every minute (launchd or crontab)")
    sub.add_parser("uninstall", help="remove the scheduler")
    sub.add_parser("status", help="whether the scheduler is installed")
    sub.add_parser("tick", help=argparse.SUPPRESS)
    sub.add_parser("exec", help=argparse.SUPPRESS).add_argument("id")
    args = parser.parse_args(argv)

    store, scheduler = Schedules(), Scheduler()
    try:
        match args.action:
            case "list":
                print_jobs(console, store)
            case "add":
                job = store.add(Job(id=args.id or store.new_id(args.prompt), cron=args.cron, prompt=args.prompt,
                                    cwd=str(Path(args.cwd).resolve()), mode=args.mode, model=args.model,
                                    max_turns=args.max_turns, memory=args.memory,
                                    trust_project=args.trust_project, catch_up=not args.no_catch_up,
                                    notify=not args.no_notify))
                console.print(f"scheduled [bold]{job.id}[/]: {describe_job(job)}")
                if not scheduler.installed():
                    console.print("[yellow]the scheduler isn't running yet: run `wren schedule install`[/]")
            case "rm":
                store.remove(args.id)
                console.print(f"deleted {args.id}")
            case "pause" | "resume":
                store.set_paused(args.id, args.action == "pause")
                console.print(f"{args.id} {'paused' if args.action == 'pause' else 'resumed'}")
            case "run":
                console.print(f"[dim]running {args.id}…[/]")
                entry = execute(store, store.get(args.id), notify_fn=lambda *_: None)
                print_run(console, entry)
            case "logs":
                store.get(args.id)
                runs = store.runs(args.id, args.n)
                if not runs:
                    console.print("[dim]no runs yet[/]")
                for entry in runs:
                    print_run(console, entry)
                console.print(f"[dim]full output: {store.dir / args.id / 'output.log'}[/]")
            case "install":
                console.print(scheduler.install())
                check_keys(console, store)
            case "uninstall":
                console.print(scheduler.uninstall())
            case "status":
                state = "installed" if scheduler.installed() else "not installed (wren schedule install)"
                console.print(f"scheduler: {state}")
            case "tick":
                for job_id, what in tick(store, now(), launch_detached(store)):
                    print(f"{datetime.now():%Y-%m-%d %H:%M} {job_id}: {what}")
            case "exec":
                execute(store, store.get(args.id))
    except ScheduleError as e:
        console.print(f"[bold red]error:[/] {escape(str(e))}")
        return 1
    return 0


def print_jobs(console: Console, store: Schedules) -> None:
    jobs = store.jobs()
    if not jobs:
        console.print("[dim]no scheduled jobs; add one with `wren schedule add CRON PROMPT`[/]")
        return
    for job in jobs:
        runs = store.runs(job.id, 1)
        last = f" · last: {runs[-1]['status']} {runs[-1].get('started', runs[-1].get('scheduled', ''))[:16]}" \
            if runs else ""
        paused = " [yellow]paused[/]" if job.paused else ""
        console.print(f"  [bold]{job.id}[/]{paused} [dim]· {escape(describe_job(job))}{escape(last)}[/]")
        console.print(f"    {escape(job.prompt if len(job.prompt) <= 100 else job.prompt[:99] + '…')}")


def print_run(console: Console, entry: dict) -> None:
    color = {"done": "green", "skipped": "yellow", "missed": "yellow"}.get(entry.get("status", ""), "red")
    when = entry.get("started") or entry.get("scheduled", "")
    extra = []
    if entry.get("cost_usd") is not None:
        extra.append(f"${entry['cost_usd']:.4f}")
    if entry.get("session_id"):
        extra.append(f"resume with wren -r {entry['session_id']}")
    console.print(f"[{color}]{entry.get('status')}[/] {when}" + (f" [dim]· {' · '.join(extra)}[/]" if extra else ""))
    text = entry.get("result") or entry.get("error")
    if text:
        console.print(f"  {escape(text.strip())}")


def check_keys(console: Console, store: Schedules) -> None:
    """Scheduled runs don't see the shell's environment: point out missing API keys."""
    try:
        config = load_config()
        names = {j.model for j in store.jobs()} | {None}
        needed = {config.model(n).key_env for n in names} - {""}
    except (ConfigError, ScheduleError):
        return
    available = load_env_file(store.home)
    missing = sorted(k for k in needed if k not in available)
    if missing:
        console.print(f"[yellow]scheduled runs don't see your shell's environment: put "
                      f"{', '.join(missing)} in {store.home / 'env'} (KEY=value lines, chmod 600)[/]")
        in_shell = [k for k in missing if os.environ.get(k)]
        if in_shell:
            console.print(f"[dim]they're set in this shell, e.g.: "
                          f"printf '{in_shell[0]}=%s\\n' \"${in_shell[0]}\" >> {store.home / 'env'}[/]")
