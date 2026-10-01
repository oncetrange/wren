"""The interactive session: the input line, its menu and toolbar, and the loop
that sends prompts or runs slash commands (see commands.py)."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import merge_key_bindings
from prompt_toolkit.styles import Style

from wren import __version__, mentions
from wren.agent.loop import Agent
from wren.agent.permissions import LABELS
from wren.agent.predict import Predictor
from wren.cli import commands
from wren.cli.app import run_prompt
from wren.cli.completion import STYLES as MENU_STYLES
from wren.cli.completion import SlashMenu
from wren.cli.keys import newline_bindings
from wren.cli.terminal import distinguish_shift_enter, register_shift_enter
from wren.cli.ui import RichUI
from wren.config import CONFIG_DIR, Config
from wren.llm.types import LLMError
from wren.memory import Memories
from wren.settings import Settings
from wren.skills import expand as expand_skill

TOOLBAR_STYLE = Style.from_dict({"bottom-toolbar": "noreverse", **MENU_STYLES})


class Repl:
    def __init__(self, agent: Agent, ui: RichUI, config: Config, settings: Settings,
                 memories: Memories | None = None):
        self.agent, self.ui, self.config, self.settings = agent, ui, config, settings
        self.memories = memories
        self.console = ui.console
        self.prefill = ""  # text to pre-fill the next prompt with (e.g. after /undo)

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
            prefill, self.prefill = self.prefill, ""
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
            if text.startswith("/") and not expand_skill(text, self.agent.skills, commands.BUILTIN_NAMES):
                try:
                    if self.command(text) == "exit":
                        return 0
                except (KeyboardInterrupt, EOFError):
                    self.console.print("[dim]cancelled[/]")
                continue
            run_prompt(self.agent, text, commands.BUILTIN_NAMES)
            self.usage_line()
            if self.settings.suggestions and self.agent.status == "done":
                self.predictor.start()

    def command(self, text: str) -> str | None:
        name, _, arg = text.partition(" ")
        found = commands.find(name)
        if found is None:
            self.ui.error(f"unknown command {name}; see /help")
            return None
        return found.run(self, arg.strip())

    def usage_line(self) -> None:
        self.ui.usage_line(self.agent.estimated_context(), self.agent.model.context_window,
                           self.agent.usage.output_tokens, self.agent.cost)

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

    # --- the input line ----------------------------------------------------------

    def completions(self) -> list[tuple[str, str]]:
        skills = [(f"/{s.name}", (f"{s.argument_hint} · " if s.argument_hint else "") + s.description)
                  for s in sorted(self.agent.skills.values(), key=lambda s: s.name)
                  if s.name not in commands.BUILTIN_NAMES]
        return commands.MENU + skills

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

    def _prediction_ready(self) -> None:
        """From the predictor's thread: show the guess if the prompt is waiting."""
        app = self.session.app
        if app.is_running and app.loop is not None:
            app.loop.call_soon_threadsafe(self._show_prediction)

    def _show_prediction(self) -> None:
        self.menu.show_prediction(self.predictor.take(), self.session.default_buffer)
        self.session.app.invalidate()
