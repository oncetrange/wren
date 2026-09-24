from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.markup import escape

from wren import __version__
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SessionLog
from wren.cli.ui import RichUI, fmt_tokens
from wren.config import CONFIG_DIR, CONFIG_FILE, Config, ConfigError, load_config
from wren.llm.factory import create_provider
from wren.tools import ToolContext

HELP = """\
[bold]Commands[/]
  /model [name]   show or switch the model
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
    parser.add_argument("--yolo", action="store_true", help="run every tool without asking for approval")
    parser.add_argument("--version", action="version", version=f"wren {__version__}")
    args = parser.parse_args(argv)

    ui = RichUI()
    try:
        config = load_config()
        model = config.model(args.model)
        provider = create_provider(model)
    except ConfigError as e:
        ui.error(str(e))
        return 1

    agent = Agent(
        provider,
        model,
        ToolContext(cwd=Path.cwd().resolve()),
        ui,
        permissions=Permissions(mode="auto" if args.yolo else "ask"),
        log=SessionLog(),
    )

    if args.prompt:
        agent.run(args.prompt)
        return 0
    return Repl(agent, ui, config).loop()


class Repl:
    def __init__(self, agent: Agent, ui: RichUI, config: Config):
        self.agent, self.ui, self.config = agent, ui, config
        self.console = ui.console

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
        while True:
            try:
                text = session.prompt("\n› ").strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                return 0
            if not text:
                continue
            if text.startswith("/"):
                if self.command(text) == "exit":
                    return 0
                continue
            self.agent.run(text)
            self.ui.usage_line(self.agent.context_tokens, self.agent.usage.output_tokens, self.agent.cost)

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
            case _:
                self.ui.error(f"unknown command {name}; see /help")
        return None

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
