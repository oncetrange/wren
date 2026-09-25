from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.markup import escape

from wren import __version__
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SESSIONS_DIR, SessionLog, SessionState, list_sessions, load_session
from wren.checkpoint import Checkpoint, CheckpointError, Checkpoints
from wren.cli.ui import RichUI, fmt_tokens
from wren.config import CONFIG_DIR, CONFIG_FILE, Config, ConfigError, load_config
from wren.llm.factory import create_provider
from wren.llm.types import LLMError
from wren.tools import ToolContext

HELP = """\
[bold]Commands[/]
  /model [name]   show or switch the model
  /undo           revert the last turn: files and conversation
  /rewind         pick an earlier turn to go back to
  /compact        summarize the conversation to free up context
  /clear          start a fresh conversation
  /cost           token usage and cost so far
  /help           this help
  /exit           quit (or Ctrl-D)

[bold]Keys[/]
  Enter submits · Esc Enter or Ctrl-J inserts a newline · Ctrl-C interrupts the agent"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wren", description="A coding agent for your terminal.")
    parser.add_argument("-p", "--print", dest="prompt", metavar="PROMPT",
                        help="run a single request non-interactively and exit")
    parser.add_argument("-m", "--model", help="model name from the config (default: config default_model)")
    parser.add_argument("-c", "--continue", dest="continue_", action="store_true",
                        help="continue the most recent session in this directory")
    parser.add_argument("-r", "--resume", nargs="?", const="", metavar="ID",
                        help="resume a session by id, or pick one from a list")
    parser.add_argument("--yolo", action="store_true", help="run every tool without asking for approval")
    parser.add_argument("--version", action="version", version=f"wren {__version__}")
    args = parser.parse_args(argv)

    ui = RichUI()
    cwd = Path.cwd().resolve()
    try:
        config = load_config()
        state = _session_to_resume(args, cwd, ui)
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
        permissions=Permissions(mode="auto" if args.yolo else "ask"),
        log=SessionLog(path=state.path) if state else SessionLog(),
        checkpoints=Checkpoints(cwd),
    )
    if state:
        agent.restore(state)
        ui.console.print(f"[dim]resumed session {agent.log.id} · {len(state.messages)} messages · "
                         f"first prompt: {escape(_one_line(state.first_prompt))}[/]")

    if args.prompt:
        agent.run(args.prompt)
        return 0
    return Repl(agent, ui, config).loop()


def _session_to_resume(args: argparse.Namespace, cwd: Path, ui: RichUI) -> SessionState | None:
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
    for i, s in enumerate(sessions, 1):
        when = datetime.fromtimestamp(s.updated).strftime("%m-%d %H:%M")
        ui.console.print(f" [bold]{i:>2}[/]  {when}  {escape(_one_line(s.first_prompt))} "
                         f"[dim]({len(s.messages)} messages)[/]")
    choice = ui.console.input("resume which session? [1] ").strip() or "1"
    if not choice.isdigit() or not 1 <= int(choice) <= len(sessions):
        raise ConfigError(f"invalid choice {choice!r}")
    return sessions[int(choice) - 1]


def _one_line(text: str, width: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1] + "…"


class Repl:
    def __init__(self, agent: Agent, ui: RichUI, config: Config):
        self.agent, self.ui, self.config = agent, ui, config
        self.console = ui.console
        self._prefill = ""  # text to pre-fill the next prompt with (e.g. after /undo)

    def loop(self) -> int:
        self.console.print(
            f"[bold]wren[/] {__version__} · model [cyan]{self.agent.model.name}[/] "
            f"({self.agent.model.model}) · {self.agent.ctx.cwd}\n"
            "[dim]/help for commands · Ctrl-D to quit[/]"
        )
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "history")), key_bindings=_key_bindings()
        )
        if not self.agent.checkpoints or not self.agent.checkpoints.enabled:
            reason = self.agent.checkpoints.disabled_reason if self.agent.checkpoints else "off"
            self.console.print(f"[yellow]checkpoints disabled ({reason}); /undo is unavailable[/]")
        while True:
            prefill, self._prefill = self._prefill, ""
            try:
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

    def command(self, text: str) -> str | None:
        name, _, arg = text.partition(" ")
        arg = arg.strip()
        match name:
            case "/exit" | "/quit":
                return "exit"
            case "/help":
                self.console.print(HELP)
            case "/clear":
                self.agent.clear()
                self.console.print("[dim]conversation cleared[/]")
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
            case "/undo":
                cps = self._checkpoints()
                if cps:
                    self.rewind(cps[-1])
            case "/rewind":
                self.pick_rewind()
            case "/compact":
                try:
                    self.agent.compact()
                except LLMError as e:
                    self.ui.error(str(e))
                except KeyboardInterrupt:
                    self.ui.notice("compaction cancelled")
            case _:
                self.ui.error(f"unknown command {name}; see /help")
        return None

    def _checkpoints(self) -> list[Checkpoint]:
        cps = self.agent.checkpoints
        if cps is None or not cps.enabled:
            self.ui.error(f"checkpoints are disabled ({cps.disabled_reason if cps else 'off'})")
            return []
        if not cps.history:
            self.console.print("[dim]nothing to undo[/]")
        return cps.history

    def pick_rewind(self) -> None:
        cps = self._checkpoints()
        if not cps:
            return
        for i, cp in enumerate(cps, 1):
            self.console.print(f" [bold]{i:>2}[/]  {escape(_one_line(cp.prompt))}")
        choice = self.console.input(f"go back to before which turn? [{len(cps)}] ").strip() or str(len(cps))
        if not choice.isdigit() or not 1 <= int(choice) <= len(cps):
            self.console.print("[dim]cancelled[/]")
            return
        self.rewind(cps[int(choice) - 1])

    def rewind(self, cp: Checkpoint) -> None:
        assert self.agent.checkpoints is not None
        try:
            changed = self.agent.checkpoints.changed_files(cp)
        except CheckpointError as e:
            self.ui.error(str(e))
            return
        self.console.print(f"going back to before: [bold]{escape(_one_line(cp.prompt))}[/]")
        if changed:
            self.console.print("files to restore:")
            for line in changed[:20]:
                status, _, path = line.partition(" ")
                what = {"A": "delete", "D": "recreate", "M": "revert"}.get(status, status)
                self.console.print(f"  [dim]{what:<8}[/] {escape(path)}")
            if len(changed) > 20:
                self.console.print(f"  [dim]… and {len(changed) - 20} more[/]")
        else:
            self.console.print("[dim]no file changes to restore[/]")
        if cp.message_index is None:
            self.console.print("[yellow]the conversation was compacted or cleared since; only files are restored[/]")
        if self.console.input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            self.console.print("[dim]cancelled[/]")
            return
        try:
            self.agent.rewind(cp)
        except CheckpointError as e:
            self.ui.error(str(e))
            return
        self.console.print("[dim]restored; the prompt is back in the input box[/]")
        self._prefill = cp.prompt

    def switch_model(self, name: str) -> None:
        if not name:
            for m in self.config.models.values():
                mark = "●" if m.name == self.agent.model.name else " "
                self.console.print(f" {mark} [bold]{m.name}[/]  {escape(m.model)}  [dim]{m.base_url or ''}[/]")
            self.console.print(f"[dim]models are configured in {CONFIG_FILE}[/]")
            return
        try:
            model = self.config.model(name)
            self.agent.set_model(create_provider(model), model)
        except ConfigError as e:
            self.ui.error(str(e))
            return
        self.console.print(f"switched to [cyan]{model.name}[/] ({model.model})")


def _key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    return kb


if __name__ == "__main__":
    sys.exit(main())
