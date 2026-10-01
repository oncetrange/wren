"""Slash commands. Each is declared once, with `@command`: its name and help
line (for /help and the slash menu) next to what it does. Registration order
is the order they are listed in."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule

from wren.agent import shell_hooks
from wren.agent.events import EventName
from wren.agent.plans import reminder
from wren.agent.remember import REMEMBER_REQUEST
from wren.agent.session import SessionLog, list_sessions
from wren.agent.subagents import TaskTool
from wren.agent.todos import format_todos, progress
from wren.checkpoint import CheckpointError
from wren.cli.app import run_prompt
from wren.cli.pickers import confirm, pick
from wren.cli.schedule_cmd import print_jobs, print_run
from wren.cli.terminal import detect_background, shift_enter_help
from wren.cli.text import one_line, session_options
from wren.cli.ui import fmt_tokens
from wren.config import CONFIG_FILE, ConfigError
from wren.llm.factory import create_provider
from wren.llm.types import LLMError
from wren.mcp_servers import PROJECT_MCP
from wren.schedules import Scheduler, Schedules, launch_detached
from wren.settings import Theme

if TYPE_CHECKING:
    from wren.cli.repl import Repl

Handler = Callable[["Repl", str], "str | None"]


@dataclass(frozen=True)
class Command:
    name: str  # with the slash
    help: str
    run: Handler
    aliases: tuple[str, ...] = ()


COMMANDS: list[Command] = []


def command(name: str, help: str, *aliases: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        COMMANDS.append(Command(name, help, fn, aliases))
        return fn
    return register


def find(name: str) -> Command | None:
    return next((c for c in COMMANDS if name == c.name or name in c.aliases), None)


# --- history -----------------------------------------------------------------------


@command("/undo", "undo the last turn (files + conversation) or the last compaction")
def _undo(repl: Repl, arg: str) -> None:
    timeline = repl.agent.conv.timeline
    if not timeline:
        repl.console.print("[dim]nothing to undo[/]")
    else:
        rewind(repl, len(timeline) - 1)


@command("/rewind", "pick an earlier point to go back to")
def _rewind(repl: Repl, arg: str) -> None:
    timeline = repl.agent.conv.timeline
    if not timeline:
        repl.console.print("[dim]nothing to rewind to[/]")
        return
    options = []
    for i, point in enumerate(timeline):
        if point.kind == "turn":
            options.append((i, f"before: {one_line(point.label)}"))
        else:
            options.append((i, "⟲ undo compaction (bring back the full history)"))
    index = pick("Go back to which point?", options, default=len(timeline) - 1)
    if index is not None:
        rewind(repl, index)


def rewind(repl: Repl, index: int) -> None:
    point = repl.agent.conv.timeline[index]
    if point.kind == "compaction":
        if not confirm("Undo the compaction and restore the full conversation?", default=True):
            return
        repl.agent.rewind(index)
        repl.console.print("[dim]full conversation restored; files unchanged[/]")
        return

    try:
        changed = repl.agent.changed_files(point)
    except CheckpointError as e:
        repl.ui.error(str(e))
        return
    repl.console.print(f"going back to before: [bold]{escape(one_line(point.label))}[/]")
    if point.commit is None:
        repl.console.print("[yellow]no file snapshot for this turn; only the conversation is rewound[/]")
    elif changed:
        repl.console.print("files to restore:")
        for line in changed[:20]:
            status, _, path = line.partition(" ")
            what = {"A": "delete", "D": "recreate", "M": "revert"}.get(status, status)
            repl.console.print(f"  [dim]{what:<8}[/] {escape(path)}")
        if len(changed) > 20:
            repl.console.print(f"  [dim]… and {len(changed) - 20} more[/]")
    else:
        repl.console.print("[dim]no file changes to restore[/]")
    if not confirm("Proceed?"):
        return
    try:
        repl.agent.rewind(index)
    except CheckpointError as e:
        repl.ui.error(str(e))
        return
    repl.console.print("[dim]restored; the prompt is back in the input box[/]")
    repl.prefill = point.label


@command("/compact", "summarize the conversation to free up context")
def _compact(repl: Repl, arg: str) -> None:
    try:
        repl.agent.compact()
    except LLMError as e:
        repl.ui.error(str(e))
    except KeyboardInterrupt:
        repl.ui.notice("compaction cancelled")


# --- sessions ----------------------------------------------------------------------


@command("/resume", "switch to another session in this directory")
def _resume(repl: Repl, arg: str) -> None:
    agent = repl.agent
    sessions = list_sessions(agent.ctx.cwd)
    if not sessions:
        repl.console.print("[dim]no sessions in this directory yet[/]")
        return
    state = pick("Switch to which session?", session_options(sessions, agent.log.path))
    if state is None or state.path == agent.log.path:
        return
    repl.save_memories()
    agent.log = SessionLog(path=state.path)
    agent.restore(state)
    agent.start_session("resume")
    repl.console.clear()
    repl.ui.render_history(agent.messages, agent.tools, agent.ctx)


@command("/clear", "start a new session")
def _clear(repl: Repl, arg: str) -> None:
    repl.save_memories()
    old = repl.agent.log.id
    repl.agent.new_session(SessionLog())
    repl.agent.start_session("clear")
    repl.console.print(f"[dim]new session started · the previous one can be resumed "
                       f"with /resume or wren -r {old}[/]")


@command("/todos", "show the current task list")
def _todos(repl: Repl, arg: str) -> None:
    todos = repl.agent.conv.todos
    if todos:
        repl.console.print(f"[bold]Tasks[/] [dim]({progress(todos)})[/]")
        repl.ui.print_todos_text(format_todos(todos))
    else:
        repl.console.print("[dim]no task list[/]")


# --- what the agent can use --------------------------------------------------------


@command("/skills", "list available skills (run one with /<name> [arguments])")
def _skills(repl: Repl, arg: str) -> None:
    skills = sorted(repl.agent.skills.values(), key=lambda s: s.name)
    if not skills:
        repl.console.print("[dim]no skills; add one as ~/.wren/skills/<name>/SKILL.md "
                           "or .wren/skills/<name>/SKILL.md[/]")
        return
    for s in skills:
        notes = [s.source]
        if not s.model_invocable:
            notes.append("manual only")
        if s.name in BUILTIN_NAMES:
            notes.append("shadowed by a built-in command")
        hint = f" {escape(s.argument_hint)}" if s.argument_hint else ""
        repl.console.print(f"  [bold]/{s.name}[/]{hint} [dim]· {' · '.join(notes)}[/]")
        repl.console.print(f"    {escape(one_line(s.description, 100))}")


@command("/agents", "list subagent types the model can delegate to")
def _agents(repl: Repl, arg: str) -> None:
    task = repl.agent.tools.get("task")
    if not isinstance(task, TaskTool):
        repl.console.print("[dim]subagents are off[/]")
        return
    for t in sorted(task.types.values(), key=lambda t: t.name):
        tools = "all tools" if t.tools is None else ", ".join(sorted(t.tools)) or "no tools"
        notes = [t.source, tools] + (["read-only"] if t.read_only else [])
        if t.model:
            notes.append(f"model {t.model}")
        repl.console.print(f"  [bold]{t.name}[/] [dim]· {escape(' · '.join(notes))}[/]")
        repl.console.print(f"    {escape(one_line(t.description, 100))}")
    repl.console.print("[dim]define more in ~/.wren/agents/<name>.md or .wren/agents/<name>.md[/]")


@command("/memory", "view, edit or delete memories · /memory on|off|auto")
def _memory(repl: Repl, arg: str) -> None:
    memories, settings = repl.memories, repl.settings
    if memories is None:
        repl.ui.error("memory is not available")
        return
    match arg:
        case "on" | "off":
            memories.set_enabled(arg == "on")
            live = (repl.agent.memory is not None) == (arg == "on")
            repl.console.print(f"[dim]memory {arg} for this project"
                               + ("" if live else "; takes effect when wren next starts") + "[/]")
            return
        case "auto":
            settings.memory_auto = repl.agent.auto_memory = not settings.memory_auto
            settings.save()
            state = "on" if settings.memory_auto else "off"
            repl.console.print(f"[dim]reviewing sessions for memories (at compaction and "
                               f"session end) is {state}[/]")
            return
        case "":
            pass
        case _:
            repl.ui.error("usage: /memory [on|off|auto]")
            return
    state = "on" if memories.enabled else "off (/memory on enables it)"
    auto = "on" if settings.memory_auto else "off"
    repl.console.print(f"[bold]Memory[/] {state} · review at session end {auto} (/memory auto)")
    entries = [(store, m) for store in (memories.user, memories.project) for m in store.all()]
    for store in (memories.user, memories.project):
        repl.console.print(f"  [dim]{store.scope}: {store.dir}[/]")
    if not entries:
        repl.console.print("[dim]no memories yet[/]")
        return
    choice = pick("Which memory?", [(i, f"{store.scope}/{m.name} · {one_line(m.description, 70)}")
                                    for i, (store, m) in enumerate(entries)])
    if choice is None:
        return
    store, memory = entries[choice]
    repl.console.print(Panel(Markdown(memory.body, code_theme=repl.ui.code_theme),
                             title=f"{store.scope}/{memory.name} · {memory.type} · {memory.updated}",
                             title_align="left", border_style="dim", padding=(0, 1)))
    action = pick("Do what with it?", [("keep", "Keep"), ("edit", "Edit in $EDITOR"),
                                        ("delete", "Delete")], default="keep")
    if action == "delete" and confirm(f"Delete {store.scope}/{memory.name}?"):
        store.delete(memory.name)
        repl.console.print("[dim]deleted[/]")
    elif action == "edit":
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        subprocess.run([*shlex.split(editor), str(store.path(memory.name))])
        store.reindex()


@command("/remember", "have the model remember something across sessions")
def _remember(repl: Repl, arg: str) -> None:
    if repl.agent.memory is None:
        repl.ui.error("memory is off in this session (see /memory)")
    elif not arg:
        repl.ui.error("usage: /remember <what to remember>")
    else:
        repl.agent.run(arg, attachments=[reminder(REMEMBER_REQUEST)])


@command("/mcp", "MCP servers: status and tools")
def _mcp(repl: Repl, arg: str) -> None:
    mcp = repl.agent.mcp
    if mcp is None:
        repl.console.print("[dim]no MCP servers; add one as [mcp.<name>] in "
                           f"{CONFIG_FILE} or in the project's {PROJECT_MCP}[/]")
        return
    for state in mcp.servers.values():
        cfg = state.config
        color = {"connected": "green", "failed": "red"}.get(state.status, "yellow")
        target = cfg.url if cfg.url else " ".join([cfg.command or "", *cfg.args])
        repl.console.print(f"  [bold]{cfg.name}[/] [{color}]{state.status}[/] "
                           f"[dim]· {cfg.transport} · {cfg.source} · {escape(target)}[/]")
        if state.error:
            repl.console.print(f"    [red]{escape(state.error)}[/]")
        if cfg.transport == "stdio":
            repl.console.print(f"    [dim]log: {mcp.log_dir / f'mcp-{cfg.name}.log'}[/]")
        names = [t.name + (" (read-only)" if t.read_only else "") for t in state.tools]
        if names:
            repl.console.print(f"    {escape(', '.join(names))}")


@command("/jobs", "background commands: output, or stop one")
def _jobs(repl: Repl, arg: str) -> None:
    jobs = repl.agent.ctx.jobs
    if not jobs.jobs:
        repl.console.print("[dim]no background jobs; the model starts them with bash run_in_background[/]")
        return
    for job in jobs.jobs.values():
        color = "green" if job.running else "dim"
        repl.console.print(f"  [bold]{job.id}[/] [{color}]{job.status}[/] [dim]· {job.runtime} ·[/] "
                           f"{escape(job.command if len(job.command) <= 80 else job.command[:79] + '…')}")
    job_id = pick("Which job?", [(j.id, f"{j.id} · {j.status}") for j in jobs.jobs.values()])
    if job_id is None:
        return
    job = jobs.jobs[job_id]
    options = [("tail", "Show its latest output")] + ([("kill", "Stop it")] if job.running else [])
    action = pick("Do what?", options, default="tail")
    if action == "tail":
        text = job.log.read_text(errors="replace")[-4000:]
        repl.console.print(escape(text.rstrip()) or "[dim](no output)[/]")
    elif action == "kill":
        jobs.kill(job)
        repl.console.print(f"[dim]{job_id} {job.status}[/]")


@command("/schedule", "scheduled runs: list, run now, pause or delete")
def _schedule(repl: Repl, arg: str) -> None:
    store, scheduler = Schedules(), Scheduler()
    state = "running" if scheduler.installed() else "[yellow]not installed[/] (wren schedule install)"
    repl.console.print(f"[bold]Scheduled runs[/] · scheduler {state}")
    print_jobs(repl.console, store)
    jobs = store.jobs()
    if not jobs:
        repl.console.print("[dim]ask me to schedule something, or: wren schedule add CRON PROMPT[/]")
        return
    job_id = pick("Which job?", [(j.id, f"{j.id} · {j.cron}") for j in jobs])
    if job_id is None:
        return
    job = store.get(job_id)
    action = pick("Do what with it?", [
        ("logs", "Show recent runs"), ("run", "Run it now, in the background"),
        ("pause", "Resume it" if job.paused else "Pause it"), ("delete", "Delete it")], default="logs")
    if action == "logs":
        runs = store.runs(job_id, 5)
        for entry in runs:
            print_run(repl.console, entry)
        if not runs:
            repl.console.print("[dim]no runs yet[/]")
    elif action == "run":
        launch_detached(store)(job)
        repl.console.print(f"[dim]started; see /schedule → logs, or {store.dir / job_id / 'output.log'}[/]")
    elif action == "pause":
        store.set_paused(job_id, not job.paused)
        repl.console.print(f"[dim]{job_id} {'resumed' if job.paused else 'paused'}[/]")
    elif action == "delete" and confirm(f"Delete {job_id}?"):
        store.remove(job_id)
        repl.console.print("[dim]deleted[/]")


@command("/loop", "repeat a prompt at an interval: /loop 10m <prompt>")
def _loop(repl: Repl, arg: str) -> None:
    """Run a prompt now and then every interval, until Ctrl-C."""
    interval, _, prompt = arg.partition(" ")
    seconds = parse_interval(interval)
    if seconds is None or not prompt.strip():
        repl.ui.error("usage: /loop <interval> <prompt>, e.g. /loop 10m check the deploy "
                      "(s, m, h or d; at least 1m)")
        return
    repl.console.print(f"[dim]running every {interval}, Ctrl-C stops[/]")
    run = 0
    try:
        while True:
            run += 1
            repl.console.print(Rule(f"loop run {run} · {datetime.now():%H:%M}", style="dim"))
            run_prompt(repl.agent, prompt.strip(), BUILTIN_NAMES)
            repl.usage_line()
            if repl.agent.status == "interrupted":
                break
            deadline = time.monotonic() + seconds
            with repl.console.status("") as status:
                while (left := deadline - time.monotonic()) > 0:
                    status.update(f"next run in {int(left) // 60}:{int(left) % 60:02d} · Ctrl-C stops")
                    time.sleep(min(1.0, left))
    except KeyboardInterrupt:
        pass
    repl.console.print(f"[dim]loop stopped after {run} run{'s' if run != 1 else ''}[/]")


def parse_interval(text: str) -> int | None:
    """"30m", "2h", "1d" -> seconds (at least a minute); None if it isn't one."""
    m = re.fullmatch(r"(\d+)([smhd])", text.strip())
    if not m:
        return None
    seconds = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return seconds if seconds >= 60 else None


@command("/hooks", "list active hooks and built-in policies")
def _hooks(repl: Repl, arg: str) -> None:
    labels: dict[EventName, str] = {
        "session_start": "SessionStart", "prompt": "UserPromptSubmit", "pre_tool": "PreToolUse",
        "post_tool": "PostToolUse", "stop": "Stop", "notification": "Notification", "session_end": "SessionEnd"}
    for event, label in labels.items():
        regs = repl.agent.hooks.registrations(event)
        if not regs:
            continue
        repl.console.print(f"[bold]{label}[/]")
        for reg in regs:
            if isinstance(reg.fn, shell_hooks.ShellHook):
                spec = reg.fn.spec
                matcher = f" [dim]({spec.matcher})[/]" if spec.matcher else ""
                repl.console.print(f"  {escape(spec.command)}{matcher} [dim]· {spec.source}[/]")
            else:
                repl.console.print(f"  [dim]{reg.name} · built-in[/]")
    repl.console.print(f"[dim]user hooks: {CONFIG_FILE} · project hooks: {shell_hooks.PROJECT_HOOKS}[/]")


# --- settings ----------------------------------------------------------------------


@command("/model", "show or switch the model")
def _model(repl: Repl, name: str) -> None:
    if not name:
        picked = pick("Model", [(m.name, f"{m.name}  {m.model}" + ("" if m.has_key() else
                                         f"  (no key: ${m.key_env}; wren setup saves one)"))
                                for m in repl.config.models.values()],
                      default=repl.agent.model.name)
        if picked is None:
            return
        name = picked
    try:
        model = repl.config.model(name)
        repl.agent.set_model(create_provider(model), model)
    except ConfigError as e:
        repl.ui.error(str(e))
        return
    repl.console.print(f"switched to [cyan]{model.name}[/] ({model.model}) "
                       f"[dim]· models are configured in {CONFIG_FILE}[/]")


@command("/theme", "dark / light / auto-detected colors")
def _theme(repl: Repl, arg: str) -> None:
    detected = detect_background()
    options: list[tuple[Theme, str]] = [
        ("auto", f"auto (detected: {detected or 'unknown, using dark'})"),
        ("dark", "dark terminal background"),
        ("light", "light terminal background"),
    ]
    theme: Theme | None = pick("Color theme", options, default=repl.settings.theme)
    if theme is None:
        return
    repl.settings.theme = theme
    repl.settings.save()
    repl.ui.background = theme if theme != "auto" else (detected or "dark")
    repl.console.print(f"[dim]theme: {theme} ({repl.ui.background})[/]")


@command("/suggest", "turn next-prompt suggestions on or off")
def _suggest(repl: Repl, arg: str) -> None:
    repl.settings.suggestions = not repl.settings.suggestions
    repl.settings.save()
    if not repl.settings.suggestions:
        repl.predictor.clear()
    state = "on" if repl.settings.suggestions else "off"
    repl.console.print(f"[dim]next-prompt suggestions {state} (Tab accepts one)[/]")


@command("/keys", "how to make Shift+Enter insert a newline in your terminal")
def _keys(repl: Repl, arg: str) -> None:
    repl.console.print(escape(shift_enter_help()))


@command("/cost", "token usage and cost so far")
def _cost(repl: Repl, arg: str) -> None:
    u, cost = repl.agent.usage, repl.agent.cost
    shown = f"${cost:.4f}" if cost is not None else "unknown (no price configured)"
    repl.console.print(f"input {fmt_tokens(u.input_tokens)} · output {fmt_tokens(u.output_tokens)} · "
                       f"cache read {fmt_tokens(u.cache_read_tokens)} · cache write "
                       f"{fmt_tokens(u.cache_write_tokens)} · cost {shown}")


@command("/help", "this help")
def _help(repl: Repl, arg: str) -> None:
    repl.console.print(HELP)


@command("/exit", "quit (or Ctrl-D)", "/quit")
def _exit(repl: Repl, arg: str) -> str:
    repl.save_memories()
    return "exit"


# --- derived -----------------------------------------------------------------------

MENU = [(c.name, c.help) for c in COMMANDS]
BUILTIN_NAMES = frozenset(n[1:] for c in COMMANDS for n in (c.name, *c.aliases))
HELP = "[bold]Commands[/]\n" + "\n".join(f"  {c.name:<15} {c.help}" for c in COMMANDS) + """

[bold]Keys[/]
  Enter submits · Shift+Enter (see /keys), Esc Enter or Ctrl-J inserts a newline
  Shift+Tab switches mode: ask before edits → accept edits → plan (read-only)
  Typing / lists commands and skills in the bottom line: ↑/↓ choose, Tab completes, Enter runs
  @path attaches a file (@path#L10-20 some lines, @dir/ a listing); typing @ completes paths
  After an answer, a predicted next prompt shows as grey text: Tab or → accepts it
  Ctrl-C interrupts the agent
  ↑/↓ and Enter in pickers, Esc cancels"""
