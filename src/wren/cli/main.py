from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape

from wren import __version__
from wren.agent.loop import Agent
from wren.agent.permissions import LABELS, Permissions
from wren.agent.todos import format_todos, progress
from wren.agent.session import SESSIONS_DIR, SessionLog, SessionState, list_sessions, load_session
from wren.checkpoint import CheckpointError, Checkpoints
from wren.cli.keys import newline_bindings
from wren.cli.pickers import confirm, pick
from wren.cli.terminal import (
    detect_background,
    distinguish_shift_enter,
    register_shift_enter,
    shift_enter_help,
)
from wren.cli.ui import RichUI, fmt_tokens
from wren.config import CONFIG_DIR, CONFIG_FILE, Config, ConfigError, load_config
from wren.llm.factory import create_provider
from wren.llm.types import LLMError
from wren.settings import Settings
from wren.tools import ToolContext

TOOLBAR_STYLE = Style.from_dict({"bottom-toolbar": "noreverse"})

HELP = """\
[bold]Commands[/]
  /undo           undo the last turn (files + conversation) or the last compaction
  /rewind         pick an earlier point to go back to
  /compact        summarize the conversation to free up context
  /resume         switch to another session in this directory
  /clear          start a new session
  /todos          show the current task list
  /model (name)   show or switch the model
  /theme          dark / light / auto-detected colors
  /keys           how to make Shift+Enter insert a newline in your terminal
  /cost           token usage and cost so far
  /help           this help
  /exit           quit (or Ctrl-D)

[bold]Keys[/]
  Enter submits · Shift+Enter (see /keys), Esc Enter or Ctrl-J inserts a newline
  Shift+Tab switches mode: ask before edits → accept edits → plan (read-only)
  Ctrl-C interrupts the agent · ↑/↓ and Enter in pickers, Esc cancels"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wren", description="A coding agent for your terminal.")
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
        provider = create_provider(model)
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
    )
    if state:
        agent.restore(state)
        ui.render_history(agent.messages, agent.tools, agent.ctx)

    if args.prompt:
        start = time.monotonic()
        result = agent.run(args.prompt)
        if json_output:
            print(json.dumps(_result_json(agent, result, time.monotonic() - start),
                             ensure_ascii=False))
        return 0 if agent.status == "done" else 1
    return Repl(agent, ui, config, settings).loop()


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
        "todos": [t.to_dict() for t in agent.conv.todos],
        "plan": agent.plan_text,
        "plan_file": str(agent.plan_file) if agent.plan_file else None,
        "duration_s": round(seconds, 1),
    }


def _mode(args: argparse.Namespace) -> str:
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
    def __init__(self, agent: Agent, ui: RichUI, config: Config, settings: Settings):
        self.agent, self.ui, self.config, self.settings = agent, ui, config, settings
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

        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "history")), key_bindings=kb,
            bottom_toolbar=self._toolbar, style=TOOLBAR_STYLE,
        )
        while True:
            prefill, self._prefill = self._prefill, ""
            try:
                with distinguish_shift_enter():
                    text = session.prompt("\n› ", default=prefill).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                return 0
            if not text:
                continue
            if text.startswith("/"):
                try:
                    if self.command(text) == "exit":
                        return 0
                except (KeyboardInterrupt, EOFError):
                    self.console.print("[dim]cancelled[/]")
                continue
            self.agent.run(text)
            self.ui.usage_line(self.agent.estimated_context(), self.agent.model.context_window,
                               self.agent.usage.output_tokens, self.agent.cost)

    def _toolbar(self) -> HTML:
        mode = self.agent.permissions.mode
        icon = {"ask": "⏵", "accept_edits": "⏵⏵", "plan": "⏸", "auto": "⏵⏵⏵"}[mode]
        color = {"ask": "ansigray", "accept_edits": "ansigreen", "plan": "ansicyan", "auto": "ansired"}[mode]
        return HTML(f"  <{color}>{icon} {LABELS[mode]}</{color}>"
                    f"<ansigray> · shift+tab to switch · {self.agent.model.name}</ansigray>")

    def command(self, text: str) -> str | None:
        name, _, arg = text.partition(" ")
        arg = arg.strip()
        match name:
            case "/exit" | "/quit":
                return "exit"
            case "/help":
                self.console.print(HELP)
            case "/clear":
                old = self.agent.log.id
                self.agent.new_session(SessionLog())
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
            case "/keys":
                self.console.print(escape(shift_enter_help()))
            case _:
                self.ui.error(f"unknown command {name}; see /help")
        return None

    # --- sessions ------------------------------------------------------------

    def resume(self) -> None:
        sessions = list_sessions(self.agent.ctx.cwd)
        if not sessions:
            self.console.print("[dim]no sessions in this directory yet[/]")
            return
        state = pick("Switch to which session?", _session_options(sessions, self.agent.log.path))
        if state is None or state.path == self.agent.log.path:
            return
        self.agent.log = SessionLog(path=state.path)
        self.agent.restore(state)
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
        theme = pick("Color theme", [
            ("auto", f"auto (detected: {detected or 'unknown, using dark'})"),
            ("dark", "dark terminal background"),
            ("light", "light terminal background"),
        ], default=self.settings.theme)
        if theme is None:
            return
        self.settings.theme = theme
        self.settings.save()
        self.ui.background = theme if theme != "auto" else (detected or "dark")
        self.console.print(f"[dim]theme: {theme} ({self.ui.background})[/]")

    def switch_model(self, name: str) -> None:
        if not name:
            name = pick("Model", [(m.name, f"{m.name}  {m.model}") for m in self.config.models.values()],
                        default=self.agent.model.name)
            if name is None:
                return
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
