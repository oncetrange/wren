from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule

from wren import __version__, mentions
from wren.agent import shell_hooks
from wren.agent.events import EventName
from wren.agent.loop import Agent
from wren.agent.permissions import LABELS, Mode, Permissions
from wren.agent.plans import reminder
from wren.agent.predict import Predictor
from wren.agent.remember import REMEMBER_REQUEST
from wren.agent.session import SESSIONS_DIR, SessionLog, SessionState, list_sessions, load_session
from wren.agent.subagents import TaskTool, discover_agent_types
from wren.agent.todos import format_todos, progress
from wren.checkpoint import CheckpointError, Checkpoints
from wren.cli.completion import STYLES as MENU_STYLES
from wren.cli.completion import SlashMenu
from wren.cli.keys import newline_bindings
from wren.cli.pickers import confirm, pick
from wren.cli.schedule_cmd import print_jobs, print_run
from wren.cli.terminal import (
    detect_background,
    distinguish_shift_enter,
    register_shift_enter,
    shift_enter_help,
)
from wren.cli.ui import RichUI, fmt_tokens
from wren.config import CONFIG_DIR, CONFIG_FILE, Config, ConfigError, ModelConfig, load_config
from wren.llm.base import Provider
from wren.llm.factory import create_provider
from wren.llm.types import LLMError
from wren.mcp_servers import PROJECT_MCP, McpServers, ProjectMcp, load_project_mcp
from wren.memory import Memories
from wren.schedules import Scheduler, Schedules, launch_detached
from wren.settings import Settings, Theme
from wren.skills import discover
from wren.skills import expand as expand_skill
from wren.tools import ToolContext
from wren.tools.schedule import ScheduleTool
from wren.tools.web import SearchConfig, WebFetch, WebSearch

TOOLBAR_STYLE = Style.from_dict({"bottom-toolbar": "noreverse", **MENU_STYLES})

COMMANDS: list[tuple[str, str]] = [
    ("/undo", "undo the last turn (files + conversation) or the last compaction"),
    ("/rewind", "pick an earlier point to go back to"),
    ("/compact", "summarize the conversation to free up context"),
    ("/resume", "switch to another session in this directory"),
    ("/clear", "start a new session"),
    ("/todos", "show the current task list"),
    ("/skills", "list available skills (run one with /<name> [arguments])"),
    ("/agents", "list subagent types the model can delegate to"),
    ("/memory", "view, edit or delete memories · /memory on|off|auto"),
    ("/remember", "have the model remember something across sessions"),
    ("/mcp", "MCP servers: status and tools"),
    ("/jobs", "background commands: output, or stop one"),
    ("/schedule", "scheduled runs: list, run now, pause or delete"),
    ("/loop", "repeat a prompt at an interval: /loop 10m <prompt>"),
    ("/hooks", "list active hooks and built-in policies"),
    ("/model", "show or switch the model"),
    ("/theme", "dark / light / auto-detected colors"),
    ("/suggest", "turn next-prompt suggestions on or off"),
    ("/keys", "how to make Shift+Enter insert a newline in your terminal"),
    ("/cost", "token usage and cost so far"),
    ("/help", "this help"),
    ("/exit", "quit (or Ctrl-D)"),
]
BUILTIN_NAMES = {name[1:] for name, _ in COMMANDS} | {"quit"}

HELP = "[bold]Commands[/]\n" + "\n".join(f"  {name:<15} {desc}" for name, desc in COMMANDS) + """

[bold]Keys[/]
  Enter submits · Shift+Enter (see /keys), Esc Enter or Ctrl-J inserts a newline
  Shift+Tab switches mode: ask before edits → accept edits → plan (read-only)
  Typing / lists commands and skills in the bottom line: ↑/↓ choose, Tab completes, Enter runs
  @path attaches a file (@path#L10-20 some lines, @dir/ a listing); typing @ completes paths
  After an answer, a predicted next prompt shows as grey text: Tab or → accepts it
  Ctrl-C interrupts the agent
  ↑/↓ and Enter in pickers, Esc cancels"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["schedule"]:
        from wren.cli.schedule_cmd import schedule_main

        return schedule_main(argv[1:])
    parser = argparse.ArgumentParser(prog="wren", description="A coding agent for your terminal.",
                                     epilog="wren schedule --help: run prompts on a cron schedule")
    parser.add_argument("-p", "--print", dest="prompt", metavar="PROMPT",
                        help="run a single request non-interactively and exit ('-' reads stdin)")
    parser.add_argument("--output-format", choices=["text", "json"], default="text",
                        help="with -p: 'json' prints one JSON result object to stdout and "
                             "sends progress output to stderr")
    parser.add_argument("--max-turns", type=int, default=100, metavar="N",
                        help="stop after N model calls per request (default 100)")
    parser.add_argument("--final-check", action=argparse.BooleanOptionalAction, default=None,
                        help="before finishing a request that changed files, have the model "
                             "re-check the request's explicit instructions; also warns when "
                             "--max-turns is nearly used up (default: on with -p, off otherwise)")
    parser.add_argument("--trust-project-hooks", action="store_true",
                        help="run the project's .wren/hooks.toml without asking (needed with -p)")
    parser.add_argument("--trust-project-mcp", action="store_true",
                        help="start the project's .mcp.json servers without asking (needed with -p)")
    parser.add_argument("--no-web", action="store_true", help="don't offer web_search and web_fetch")
    parser.add_argument("--mask-at", type=int, metavar="TOKENS",
                        help="clear old tool outputs past this prompt size (0: never); overrides the model's")
    parser.add_argument("--compact-at", type=int, metavar="TOKENS",
                        help="summarize the conversation past this prompt size; overrides the model's")
    parser.add_argument("--no-subagents", action="store_true", help="don't offer the task tool")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=None,
                        help="use long-term memory: the model reads and keeps memories across "
                             "sessions (default: on, off with -p)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="don't snapshot the workspace before each prompt")
    parser.add_argument("-m", "--model", help="model name from the config (default: config default_model)")
    parser.add_argument("-c", "--continue", dest="continue_", action="store_true",
                        help="continue the most recent session in this directory")
    parser.add_argument("-r", "--resume", nargs="?", const="", metavar="ID",
                        help="resume a session by id, or pick one from a list")
    parser.add_argument("--mode", choices=["ask", "accept_edits", "plan", "auto"], default="ask",
                        help="permission mode to start in (Shift+Tab switches in a session)")
    parser.add_argument("--plan", action="store_true", help="start in plan mode (same as --mode plan)")
    parser.add_argument("--yolo", action="store_true",
                        help="run every tool without asking (same as --mode auto)")
    parser.add_argument("--version", action="version", version=f"wren {__version__}")
    args = parser.parse_args(argv)

    json_output = bool(args.prompt) and args.output_format == "json"
    if args.prompt == "-":
        args.prompt = sys.stdin.read()

    settings = Settings.load()
    # In JSON mode stdout carries only the result object; everything else goes to stderr.
    console = Console(highlight=False, stderr=json_output)
    ui = RichUI(console, background=_background(settings))
    cwd = Path.cwd().resolve()
    try:
        config = load_config()
        state = _session_to_resume(args, cwd)
        name = args.model
        if name is None and state and state.model in config.models:
            name = state.model
        model = config.model(name)
        if args.mask_at is not None or args.compact_at is not None:  # for experiments
            model = dataclasses.replace(
                model, mask_at=model.mask_at if args.mask_at is None else args.mask_at,
                compact_at=model.compact_at if args.compact_at is None else args.compact_at)
        provider = create_provider(model)
        if ignored := model.ignored_options():
            ui.notice(f"model {model.name!r}: {', '.join(ignored)} not used with the "
                      f"{model.provider} provider")
    except ConfigError as e:
        ui.error(str(e))
        return 1
    skills, skill_warnings = discover(cwd)
    agent_types, agent_warnings = discover_agent_types(cwd)
    memories = Memories.for_project(cwd)
    for warning in skill_warnings + agent_warnings:
        ui.notice(warning)
    try:
        mcp = _connect_mcp(config, cwd, args, ui)
    except ConfigError as e:
        ui.error(str(e))
        return 1

    agent = Agent(
        provider,
        model,
        ToolContext(cwd=cwd),
        ui,
        permissions=Permissions(mode=_mode(args), allow_auto=_mode(args) == "auto"),
        log=SessionLog(path=state.path) if state else SessionLog(),
        checkpoints=None if args.no_checkpoints else Checkpoints(cwd),
        max_turns=args.max_turns,
        final_check=bool(args.prompt) if args.final_check is None else args.final_check,
        skills=skills,
        agent_types={} if args.no_subagents else agent_types,
        memory=memories if (not args.prompt if args.memory is None else args.memory) else None,
        mcp=mcp,
    )
    agent.auto_memory = settings.memory_auto
    if not args.prompt:  # scheduled jobs are made in sessions, not by (scheduled) headless runs
        agent.tools["schedule"] = ScheduleTool()
    if config.web.enabled and not args.no_web:
        search = SearchConfig(config.web.search, config.web.api_key_env)
        agent.tools.update({"web_search": WebSearch(search), "web_fetch": WebFetch()})
    agent.resolve_model = lambda name: _provider_for(config.model(name))
    if state:
        agent.restore(state)
        ui.render_history(agent.messages, agent.tools, agent.ctx)
    try:
        project_hooks = shell_hooks.load_project_hooks(cwd)
    except ConfigError as e:
        ui.error(str(e))
        if mcp:
            mcp.close()
        return 1
    hooks = config.hooks
    if project_hooks and _trust_project_hooks(project_hooks, args, ui):
        hooks = hooks + project_hooks.specs
    shell_hooks.install(agent, hooks)
    agent.start_session("resume" if state else "startup")

    try:
        if args.prompt:
            start = time.monotonic()
            result = run_prompt(agent, args.prompt)
            if json_output:
                print(json.dumps(_result_json(agent, result, time.monotonic() - start),
                                 ensure_ascii=False))
            return 0 if agent.status == "done" else 1
        return Repl(agent, ui, config, settings, memories).loop()
    finally:
        agent.end_session()
        if mcp:
            mcp.close()


def _connect_mcp(config: Config, cwd: Path, args: argparse.Namespace, ui: RichUI) -> McpServers | None:
    """Connect to the user's MCP servers and the project's trusted ones. Raises ConfigError."""
    servers = list(config.mcp)
    project = load_project_mcp(cwd)
    if project and project.servers and _trust_project_mcp(project, args, ui):
        taken = {s.name for s in servers}
        servers += [s for s in project.servers if s.name not in taken]
    if not servers:
        return None
    mcp = McpServers(servers, cwd)
    with ui.console.status(f"connecting to MCP server{'s' if len(servers) > 1 else ''}…"):
        mcp.connect()
    for state in mcp.servers.values():
        if state.status != "connected":
            ui.notice(f"MCP server {state.config.name!r} unavailable: {state.error} (see /mcp)")
    return mcp


def _trust_project_mcp(project: ProjectMcp, args: argparse.Namespace, ui: RichUI) -> bool:
    """A project's MCP servers start only once the file's exact content is trusted."""
    if shell_hooks.is_trusted(project):
        return True
    if args.trust_project_mcp:
        shell_hooks.trust(project)
        return True
    if args.prompt or not ui.interactive:
        ui.notice(f"skipping untrusted MCP servers in {PROJECT_MCP} "
                  "(pass --trust-project-mcp to start them)")
        return False
    ui.console.print(f"[bold]This project defines MCP servers[/] in {PROJECT_MCP}; "
                     "they run programs or connect to services:")
    for s in project.servers:
        target = s.url if s.url else " ".join([s.command or "", *s.args])
        ui.console.print(f"  [cyan]{s.name}[/]: {escape(target)}")
    if pick("Start these servers?", [(True, "Yes, trust this file (asks again if it changes)"),
                                     (False, "No, skip them this time")], default=False):
        shell_hooks.trust(project)
        return True
    return False


def parse_interval(text: str) -> int | None:
    """"30m", "2h", "1d" -> seconds (at least a minute); None if it isn't one."""
    m = re.fullmatch(r"(\d+)([smhd])", text.strip())
    if not m:
        return None
    seconds = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return seconds if seconds >= 60 else None


def _provider_for(model: ModelConfig) -> tuple[Provider, ModelConfig]:
    return create_provider(model), model


def run_prompt(agent: Agent, text: str) -> str:
    """Run a prompt, expanding `/skill-name arguments` into the skill and
    attaching the files it @-mentions."""
    attachments = []
    if invoked := expand_skill(text, agent.skills, BUILTIN_NAMES):
        skill, arguments = invoked
        agent.log.record("skill", name=skill.name, by="user")
        attachments.append(skill.invocation(arguments))
    if found := mentions.find(text, agent.ctx.cwd):
        files, shown = mentions.attach(found, agent.ctx)
        attachments += files
        show = getattr(agent.ui, "attached", None)
        for label, summary in shown:
            if show:
                show(label, summary)
        agent.log.record("mentions", paths=[label for label, _ in shown])
    return agent.run(text, attachments=attachments or None)


def _result_json(agent: Agent, result: str, seconds: float) -> dict:
    u = agent.usage
    return {
        "status": agent.status,
        "result": result,
        "model": agent.model.name,
        "model_id": agent.model.model,
        "session_id": agent.log.id,
        "session_log": str(agent.log.path) if agent.log.path else None,
        "turns": agent.turns,
        "tool_calls": agent.tool_calls,
        "tool_errors": agent.tool_errors,
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_write_tokens": u.cache_write_tokens,
        },
        "cost_usd": agent.cost,
        "subagents": agent.subagent_runs,
        "todos": [t.to_dict() for t in agent.conv.todos],
        "plan": agent.plan_text,
        "plan_file": str(agent.plan_file) if agent.plan_file else None,
        "duration_s": round(seconds, 1),
    }


def _trust_project_hooks(project: shell_hooks.ProjectHooks, args: argparse.Namespace,
                         ui: RichUI) -> bool:
    """Project hooks run only once their exact content is trusted."""
    if shell_hooks.is_trusted(project):
        return True
    if args.trust_project_hooks:
        shell_hooks.trust(project)
        return True
    if args.prompt or not ui.interactive:
        ui.notice(f"skipping untrusted project hooks in {shell_hooks.PROJECT_HOOKS} "
                  "(pass --trust-project-hooks to run them)")
        return False
    ui.console.print(f"[bold]This project defines hooks[/] in {shell_hooks.PROJECT_HOOKS}; "
                     "they run shell commands on your machine:")
    for spec in project.specs:
        matcher = f" [dim]({spec.matcher})[/]" if spec.matcher else ""
        ui.console.print(f"  [cyan]{spec.event}[/]{matcher}: {escape(spec.command)}")
    if pick("Run these hooks?", [(True, "Yes, trust this file (asks again if it changes)"),
                                  (False, "No, skip them this time")], default=False):
        shell_hooks.trust(project)
        return True
    return False


def _mode(args: argparse.Namespace) -> Mode:
    return "auto" if args.yolo else "plan" if args.plan else args.mode


def _background(settings: Settings) -> str:
    if settings.theme != "auto":
        return settings.theme
    return detect_background() or "dark"


def _session_to_resume(args: argparse.Namespace, cwd: Path) -> SessionState | None:
    if args.resume:
        matches = list(SESSIONS_DIR.glob(f"*-{args.resume}.jsonl"))
        if not matches:
            raise ConfigError(f"no session with id {args.resume!r}")
        return load_session(matches[0])
    if not args.continue_ and args.resume is None:
        return None
    sessions = list_sessions(cwd)
    if not sessions:
        raise ConfigError(f"no previous sessions in {cwd}")
    if args.continue_:
        return sessions[0]
    state = pick("Resume which session?", _session_options(sessions))
    if state is None:
        raise ConfigError("no session selected")
    return state


def _session_options(sessions: list[SessionState], current: Path | None = None):
    options = []
    for s in sessions:
        when = datetime.fromtimestamp(s.updated).strftime("%m-%d %H:%M")
        mark = " (current)" if s.path == current else ""
        options.append((s, f"{when}  {_one_line(s.first_prompt, 50)}  · {len(s.messages)} messages{mark}"))
    return options


def _one_line(text: str, width: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1] + "…"


class Repl:
    def __init__(self, agent: Agent, ui: RichUI, config: Config, settings: Settings,
                 memories: Memories | None = None):
        self.agent, self.ui, self.config, self.settings = agent, ui, config, settings
        self.memories = memories
        self.console = ui.console
        self._prefill = ""  # text to pre-fill the next prompt with (e.g. after /undo)

    def loop(self) -> int:
        self.console.print(
            f"[bold]wren[/] {__version__} · model [cyan]{self.agent.model.name}[/] "
            f"({self.agent.model.model}) · {self.agent.ctx.cwd}\n"
            "[dim]/help for commands · Ctrl-D to quit[/]"
        )
        cps = self.agent.checkpoints
        if cps is not None and not cps.enabled:
            self.console.print(f"[yellow]file checkpoints disabled ({cps.disabled_reason}); "
                               "/undo only rewinds the conversation[/]")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        register_shift_enter()
        kb = newline_bindings()

        @kb.add("s-tab")
        def _(event) -> None:
            self.agent.permissions.cycle()
            event.app.invalidate()

        self.menu = SlashMenu(self.completions, mentions.FileIndex(self.agent.ctx.cwd))
        self.predictor = Predictor(self.agent, on_ready=self._prediction_ready)
        self.session = session = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "history")),
            key_bindings=merge_key_bindings([kb, self.menu.bindings()]),
            bottom_toolbar=self._toolbar, style=TOOLBAR_STYLE,
        )
        self.menu.attach(session)
        while True:
            prefill, self._prefill = self._prefill, ""
            self.menu.prediction = self.predictor.take()
            try:
                with distinguish_shift_enter():
                    text = session.prompt("\n› ", default=prefill, pre_run=self._show_prediction).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                self.save_memories()
                return 0
            if not text:
                continue
            self.predictor.clear()  # a new turn: the old guess no longer applies
            if text.startswith("/") and not expand_skill(text, self.agent.skills, BUILTIN_NAMES):
                try:
                    if self.command(text) == "exit":
                        return 0
                except (KeyboardInterrupt, EOFError):
                    self.console.print("[dim]cancelled[/]")
                continue
            run_prompt(self.agent, text)
            self.ui.usage_line(self.agent.estimated_context(), self.agent.model.context_window,
                               self.agent.usage.output_tokens, self.agent.cost)
            if self.settings.suggestions and self.agent.status == "done":
                self.predictor.start()

    def _prediction_ready(self) -> None:
        """From the predictor's thread: show the guess if the prompt is waiting."""
        app = self.session.app
        if app.is_running and app.loop is not None:
            app.loop.call_soon_threadsafe(self._show_prediction)

    def _show_prediction(self) -> None:
        self.menu.show_prediction(self.predictor.take(), self.session.default_buffer)
        self.session.app.invalidate()

    def completions(self) -> list[tuple[str, str]]:
        skills = [(f"/{s.name}", (f"{s.argument_hint} · " if s.argument_hint else "") + s.description)
                  for s in sorted(self.agent.skills.values(), key=lambda s: s.name)
                  if s.name not in BUILTIN_NAMES]
        return COMMANDS + skills

    def _toolbar(self):
        """The slash menu while typing a command, otherwise the mode line."""
        if line := self.menu.toolbar():
            return line
        mode = self.agent.permissions.mode
        icon = {"ask": "⏵", "accept_edits": "⏵⏵", "plan": "⏸", "auto": "⏵⏵⏵"}[mode]
        color = {"ask": "ansigray", "accept_edits": "ansigreen", "plan": "ansicyan", "auto": "ansired"}[mode]
        running = len(self.agent.ctx.jobs.running())
        jobs = f" · {running} background job{'s' if running != 1 else ''} (/jobs)" if running else ""
        return HTML(f"  <{color}>{icon} {LABELS[mode]}</{color}>"
                    f"<ansigray> · shift+tab to switch · {self.agent.model.name}{jobs}</ansigray>")

    def command(self, text: str) -> str | None:
        name, _, arg = text.partition(" ")
        arg = arg.strip()
        match name:
            case "/exit" | "/quit":
                self.save_memories()
                return "exit"
            case "/help":
                self.console.print(HELP)
            case "/clear":
                self.save_memories()
                old = self.agent.log.id
                self.agent.new_session(SessionLog())
                self.agent.start_session("clear")
                self.console.print(f"[dim]new session started · the previous one can be resumed "
                                   f"with /resume or wren -r {old}[/]")
            case "/resume":
                self.resume()
            case "/cost":
                u = self.agent.usage
                cost = f"${self.agent.cost:.4f}" if self.agent.cost is not None else "unknown (no price configured)"
                self.console.print(
                    f"input {fmt_tokens(u.input_tokens)} · output {fmt_tokens(u.output_tokens)} · "
                    f"cache read {fmt_tokens(u.cache_read_tokens)} · cache write "
                    f"{fmt_tokens(u.cache_write_tokens)} · cost {cost}"
                )
            case "/model":
                self.switch_model(arg)
            case "/hooks":
                self.show_hooks()
            case "/skills":
                self.show_skills()
            case "/agents":
                self.show_agents()
            case "/mcp":
                self.show_mcp()
            case "/jobs":
                self.jobs_command()
            case "/schedule":
                self.schedule_command()
            case "/loop":
                self.loop_command(arg)
            case "/todos":
                todos = self.agent.conv.todos
                if todos:
                    self.console.print(f"[bold]Tasks[/] [dim]({progress(todos)})[/]")
                    self.ui.print_todos_text(format_todos(todos))
                else:
                    self.console.print("[dim]no task list[/]")
            case "/undo":
                timeline = self.agent.conv.timeline
                if not timeline:
                    self.console.print("[dim]nothing to undo[/]")
                else:
                    self.rewind(len(timeline) - 1)
            case "/rewind":
                self.pick_rewind()
            case "/compact":
                try:
                    self.agent.compact()
                except LLMError as e:
                    self.ui.error(str(e))
                except KeyboardInterrupt:
                    self.ui.notice("compaction cancelled")
            case "/theme":
                self.pick_theme()
            case "/suggest":
                self.settings.suggestions = not self.settings.suggestions
                self.settings.save()
                if not self.settings.suggestions:
                    self.predictor.clear()
                state = "on" if self.settings.suggestions else "off"
                self.console.print(f"[dim]next-prompt suggestions {state} (Tab accepts one)[/]")
            case "/keys":
                self.console.print(escape(shift_enter_help()))
            case "/memory":
                self.memory_command(arg)
            case "/remember":
                if self.agent.memory is None:
                    self.ui.error("memory is off in this session (see /memory)")
                elif not arg:
                    self.ui.error("usage: /remember <what to remember>")
                else:
                    self.agent.run(arg, attachments=[reminder(REMEMBER_REQUEST)])
            case _:
                self.ui.error(f"unknown command {name}; see /help")
        return None

    # --- memory ----------------------------------------------------------------

    def save_memories(self) -> None:
        """Before this conversation is left behind: let the model keep what it learned."""
        if self.agent.memory is None or not self.agent.auto_memory:
            return
        if self.agent._memory_upto >= len(self.agent.messages):
            return
        self.console.print("[dim]reviewing the session for memories… (Ctrl-C skips)[/]")
        try:
            saved = self.agent.extract_memories("the session is ending")
        except KeyboardInterrupt:
            self.console.print("[dim]skipped[/]")
            return
        except LLMError as e:
            self.ui.error(f"could not save memories: {e}")
            return
        if not saved:
            self.console.print("[dim]nothing new to remember[/]")

    def memory_command(self, arg: str) -> None:
        memories = self.memories
        if memories is None:
            self.ui.error("memory is not available")
            return
        match arg:
            case "on" | "off":
                memories.set_enabled(arg == "on")
                live = (self.agent.memory is not None) == (arg == "on")
                self.console.print(f"[dim]memory {arg} for this project"
                                   + ("" if live else "; takes effect when wren next starts") + "[/]")
                return
            case "auto":
                self.settings.memory_auto = self.agent.auto_memory = not self.settings.memory_auto
                self.settings.save()
                state = "on" if self.settings.memory_auto else "off"
                self.console.print(f"[dim]reviewing sessions for memories (at compaction and "
                                   f"session end) is {state}[/]")
                return
            case "":
                pass
            case _:
                self.ui.error("usage: /memory [on|off|auto]")
                return
        state = "on" if memories.enabled else "off (/memory on enables it)"
        auto = "on" if self.settings.memory_auto else "off"
        self.console.print(f"[bold]Memory[/] {state} · review at session end {auto} (/memory auto)")
        entries = [(store, m) for store in (memories.user, memories.project) for m in store.all()]
        for store in (memories.user, memories.project):
            self.console.print(f"  [dim]{store.scope}: {store.dir}[/]")
        if not entries:
            self.console.print("[dim]no memories yet[/]")
            return
        choice = pick("Which memory?", [
            (i, f"{store.scope}/{m.name} · {_one_line(m.description, 70)}")
            for i, (store, m) in enumerate(entries)])
        if choice is None:
            return
        store, memory = entries[choice]
        self.console.print(Panel(Markdown(memory.body, code_theme=self.ui.code_theme),
                                 title=f"{store.scope}/{memory.name} · {memory.type} · {memory.updated}",
                                 title_align="left", border_style="dim", padding=(0, 1)))
        action = pick("Do what with it?", [("keep", "Keep"), ("edit", "Edit in $EDITOR"),
                                            ("delete", "Delete")], default="keep")
        if action == "delete" and confirm(f"Delete {store.scope}/{memory.name}?"):
            store.delete(memory.name)
            self.console.print("[dim]deleted[/]")
        elif action == "edit":
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
            subprocess.run([*shlex.split(editor), str(store.path(memory.name))])
            store.reindex()

    # --- sessions ------------------------------------------------------------

    def resume(self) -> None:
        sessions = list_sessions(self.agent.ctx.cwd)
        if not sessions:
            self.console.print("[dim]no sessions in this directory yet[/]")
            return
        state = pick("Switch to which session?", _session_options(sessions, self.agent.log.path))
        if state is None or state.path == self.agent.log.path:
            return
        self.save_memories()
        self.agent.log = SessionLog(path=state.path)
        self.agent.restore(state)
        self.agent.start_session("resume")
        self.console.clear()
        self.ui.render_history(self.agent.messages, self.agent.tools, self.agent.ctx)

    # --- undo / rewind ---------------------------------------------------------

    def pick_rewind(self) -> None:
        timeline = self.agent.conv.timeline
        if not timeline:
            self.console.print("[dim]nothing to rewind to[/]")
            return
        options = []
        for i, point in enumerate(timeline):
            if point.kind == "turn":
                options.append((i, f"before: {_one_line(point.label)}"))
            else:
                options.append((i, "⟲ undo compaction (bring back the full history)"))
        index = pick("Go back to which point?", options, default=len(timeline) - 1)
        if index is not None:
            self.rewind(index)

    def rewind(self, index: int) -> None:
        point = self.agent.conv.timeline[index]
        if point.kind == "compaction":
            if not confirm("Undo the compaction and restore the full conversation?", default=True):
                return
            self.agent.rewind(index)
            self.console.print("[dim]full conversation restored; files unchanged[/]")
            return

        try:
            changed = self.agent.changed_files(point)
        except CheckpointError as e:
            self.ui.error(str(e))
            return
        self.console.print(f"going back to before: [bold]{escape(_one_line(point.label))}[/]")
        if point.commit is None:
            self.console.print("[yellow]no file snapshot for this turn; only the conversation is rewound[/]")
        elif changed:
            self.console.print("files to restore:")
            for line in changed[:20]:
                status, _, path = line.partition(" ")
                what = {"A": "delete", "D": "recreate", "M": "revert"}.get(status, status)
                self.console.print(f"  [dim]{what:<8}[/] {escape(path)}")
            if len(changed) > 20:
                self.console.print(f"  [dim]… and {len(changed) - 20} more[/]")
        else:
            self.console.print("[dim]no file changes to restore[/]")
        if not confirm("Proceed?"):
            return
        try:
            self.agent.rewind(index)
        except CheckpointError as e:
            self.ui.error(str(e))
            return
        self.console.print("[dim]restored; the prompt is back in the input box[/]")
        self._prefill = point.label

    # --- settings ------------------------------------------------------------

    def pick_theme(self) -> None:
        detected = detect_background()
        options: list[tuple[Theme, str]] = [
            ("auto", f"auto (detected: {detected or 'unknown, using dark'})"),
            ("dark", "dark terminal background"),
            ("light", "light terminal background"),
        ]
        theme: Theme | None = pick("Color theme", options, default=self.settings.theme)
        if theme is None:
            return
        self.settings.theme = theme
        self.settings.save()
        self.ui.background = theme if theme != "auto" else (detected or "dark")
        self.console.print(f"[dim]theme: {theme} ({self.ui.background})[/]")

    def show_skills(self) -> None:
        skills = sorted(self.agent.skills.values(), key=lambda s: s.name)
        if not skills:
            self.console.print("[dim]no skills; add one as ~/.wren/skills/<name>/SKILL.md "
                               "or .wren/skills/<name>/SKILL.md[/]")
            return
        for s in skills:
            notes = [s.source]
            if not s.model_invocable:
                notes.append("manual only")
            if s.name in BUILTIN_NAMES:
                notes.append("shadowed by a built-in command")
            hint = f" {escape(s.argument_hint)}" if s.argument_hint else ""
            self.console.print(f"  [bold]/{s.name}[/]{hint} [dim]· {' · '.join(notes)}[/]")
            self.console.print(f"    {escape(_one_line(s.description, 100))}")

    def show_agents(self) -> None:
        task = self.agent.tools.get("task")
        if not isinstance(task, TaskTool):
            self.console.print("[dim]subagents are off[/]")
            return
        for t in sorted(task.types.values(), key=lambda t: t.name):
            tools = "all tools" if t.tools is None else ", ".join(sorted(t.tools)) or "no tools"
            notes = [t.source, tools] + (["read-only"] if t.read_only else [])
            if t.model:
                notes.append(f"model {t.model}")
            self.console.print(f"  [bold]{t.name}[/] [dim]· {escape(' · '.join(notes))}[/]")
            self.console.print(f"    {escape(_one_line(t.description, 100))}")
        self.console.print("[dim]define more in ~/.wren/agents/<name>.md or .wren/agents/<name>.md[/]")

    # --- schedules -------------------------------------------------------------

    def schedule_command(self) -> None:
        store, scheduler = Schedules(), Scheduler()
        state = "running" if scheduler.installed() else "[yellow]not installed[/] (wren schedule install)"
        self.console.print(f"[bold]Scheduled runs[/] · scheduler {state}")
        print_jobs(self.console, store)
        jobs = store.jobs()
        if not jobs:
            self.console.print("[dim]ask me to schedule something, or: wren schedule add CRON PROMPT[/]")
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
                print_run(self.console, entry)
            if not runs:
                self.console.print("[dim]no runs yet[/]")
        elif action == "run":
            launch_detached(store)(job)
            self.console.print(f"[dim]started; see /schedule → logs, or {store.dir / job_id / 'output.log'}[/]")
        elif action == "pause":
            store.set_paused(job_id, not job.paused)
            self.console.print(f"[dim]{job_id} {'resumed' if job.paused else 'paused'}[/]")
        elif action == "delete" and confirm(f"Delete {job_id}?"):
            store.remove(job_id)
            self.console.print("[dim]deleted[/]")

    def loop_command(self, arg: str) -> None:
        """Run a prompt now and then every interval, until Ctrl-C."""
        interval, _, prompt = arg.partition(" ")
        seconds = parse_interval(interval)
        if seconds is None or not prompt.strip():
            self.ui.error("usage: /loop <interval> <prompt>, e.g. /loop 10m check the deploy "
                          "(s, m, h or d; at least 1m)")
            return
        self.console.print(f"[dim]running every {interval}, Ctrl-C stops[/]")
        run = 0
        try:
            while True:
                run += 1
                self.console.print(Rule(f"loop run {run} · {datetime.now():%H:%M}", style="dim"))
                run_prompt(self.agent, prompt.strip())
                self.ui.usage_line(self.agent.estimated_context(), self.agent.model.context_window,
                                   self.agent.usage.output_tokens, self.agent.cost)
                if self.agent.status == "interrupted":
                    break
                deadline = time.monotonic() + seconds
                with self.console.status("") as status:
                    while (left := deadline - time.monotonic()) > 0:
                        status.update(f"next run in {int(left) // 60}:{int(left) % 60:02d} · Ctrl-C stops")
                        time.sleep(min(1.0, left))
        except KeyboardInterrupt:
            pass
        self.console.print(f"[dim]loop stopped after {run} run{'s' if run != 1 else ''}[/]")

    def jobs_command(self) -> None:
        jobs = self.agent.ctx.jobs
        if not jobs.jobs:
            self.console.print("[dim]no background jobs; the model starts them with bash run_in_background[/]")
            return
        for job in jobs.jobs.values():
            color = "green" if job.running else "dim"
            self.console.print(f"  [bold]{job.id}[/] [{color}]{job.status}[/] [dim]· {job.runtime} ·[/] "
                               f"{escape(job.command if len(job.command) <= 80 else job.command[:79] + '…')}")
        job_id = pick("Which job?", [(j.id, f"{j.id} · {j.status}") for j in jobs.jobs.values()])
        if job_id is None:
            return
        job = jobs.jobs[job_id]
        options = [("tail", "Show its latest output")] + ([("kill", "Stop it")] if job.running else [])
        action = pick("Do what?", options, default="tail")
        if action == "tail":
            text = job.log.read_text(errors="replace")[-4000:]
            self.console.print(escape(text.rstrip()) or "[dim](no output)[/]")
        elif action == "kill":
            jobs.kill(job)
            self.console.print(f"[dim]{job_id} {job.status}[/]")

    def show_mcp(self) -> None:
        mcp = self.agent.mcp
        if mcp is None:
            self.console.print("[dim]no MCP servers; add one as [mcp.<name>] in "
                               f"{CONFIG_FILE} or in the project's {PROJECT_MCP}[/]")
            return
        for state in mcp.servers.values():
            cfg = state.config
            color = {"connected": "green", "failed": "red"}.get(state.status, "yellow")
            target = cfg.url if cfg.url else " ".join([cfg.command or "", *cfg.args])
            self.console.print(f"  [bold]{cfg.name}[/] [{color}]{state.status}[/] "
                               f"[dim]· {cfg.transport} · {cfg.source} · {escape(target)}[/]")
            if state.error:
                self.console.print(f"    [red]{escape(state.error)}[/]")
            if cfg.transport == "stdio":
                self.console.print(f"    [dim]log: {mcp.log_dir / f'mcp-{cfg.name}.log'}[/]")
            names = [t.name + (" (read-only)" if t.read_only else "") for t in state.tools]
            if names:
                self.console.print(f"    {escape(', '.join(names))}")

    def show_hooks(self) -> None:
        labels: dict[EventName, str] = {"session_start": "SessionStart", "prompt": "UserPromptSubmit",
                  "pre_tool": "PreToolUse", "post_tool": "PostToolUse", "stop": "Stop",
                  "notification": "Notification", "session_end": "SessionEnd"}
        for event, label in labels.items():
            regs = self.agent.hooks.registrations(event)
            if not regs:
                continue
            self.console.print(f"[bold]{label}[/]")
            for reg in regs:
                if isinstance(reg.fn, shell_hooks.ShellHook):
                    spec = reg.fn.spec
                    matcher = f" [dim]({spec.matcher})[/]" if spec.matcher else ""
                    self.console.print(f"  {escape(spec.command)}{matcher} [dim]· {spec.source}[/]")
                else:
                    self.console.print(f"  [dim]{reg.name} · built-in[/]")
        self.console.print(f"[dim]user hooks: {CONFIG_FILE} · project hooks: "
                           f"{shell_hooks.PROJECT_HOOKS}[/]")

    def switch_model(self, name: str) -> None:
        if not name:
            picked = pick("Model", [(m.name, f"{m.name}  {m.model}") for m in self.config.models.values()],
                          default=self.agent.model.name)
            if picked is None:
                return
            name = picked
        try:
            model = self.config.model(name)
            self.agent.set_model(create_provider(model), model)
        except ConfigError as e:
            self.ui.error(str(e))
            return
        self.console.print(f"switched to [cyan]{model.name}[/] ({model.model}) "
                           f"[dim]· models are configured in {CONFIG_FILE}[/]")


def _key_bindings() -> KeyBindings:
    return newline_bindings()


if __name__ == "__main__":
    sys.exit(main())
