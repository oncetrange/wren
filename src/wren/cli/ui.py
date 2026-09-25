"""Terminal rendering of agent events."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.markup import escape
from rich.padding import Padding
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from wren.agent.compact import is_summary_note
from wren.agent.permissions import Decision
from wren.cli.markdown import MarkdownStream
from wren.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from wren.tools import Tool, ToolContext, ToolOutput

MAX_DIFF_LINES = 80


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class RichUI:
    def __init__(self, console: Console | None = None, interactive: bool | None = None,
                 background: str = "dark"):
        self.console = console or Console(highlight=False)
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.background = background  # "dark" | "light": picks syntax colors
        self._status: Status | None = None
        self._stream_kind: str | None = None  # "text" | "thinking" while streaming
        self._markdown: MarkdownStream | None = None
        self._previewed = False

    # --- model stream ------------------------------------------------------

    def model_started(self) -> None:
        self._spin("Thinking…")

    def text_delta(self, text: str) -> None:
        self._stream(text, "text", "")

    def thinking_delta(self, text: str) -> None:
        self._stream(text, "thinking", "dim italic")

    def tool_call_started(self, name: str) -> None:
        self._end_stream()
        self._spin(f"Preparing {name}…")

    def model_finished(self) -> None:
        self._stop_spin()
        self._end_stream()

    # --- tools -------------------------------------------------------------

    def tool_started(self, name: str, label: str) -> None:
        self._stop_spin()
        self._previewed = False
        line = Text("● ", style="cyan")
        line.append(name, style="bold")
        if label:
            first, *rest = label.splitlines()
            line.append(" " + first + (" …" if rest else ""))
        line.truncate(self.console.width * 2, overflow="ellipsis")
        self.console.print(line)

    def confirm(self, tool: Tool, args: dict[str, Any], label: str, preview: str | None) -> Decision:
        if preview:
            self._print_diff(preview)
            self._previewed = True
        elif "\n" in label:
            self.console.print(Padding(Syntax(label, "bash", theme=self.code_theme), (0, 0, 0, 4)))
        if not self.interactive:
            self.console.print("  [yellow]denied (no terminal to ask for approval; use --yolo)[/]")
            return Decision(allow=False)

        key = tool.permission_key(args)
        scope = key.removeprefix("bash:") if key.startswith("bash:") else key
        if len(scope) > 40:
            scope = "this exact command"
        self.console.print(
            f"  [bold]Allow?[/] [green]y[/]es · [green]a[/]lways for [bold]{escape(scope)}[/] "
            f"· [red]n[/]o · or type what to do instead"
        )
        try:
            answer = self.console.input("  [bold]›[/] ").strip()
        except EOFError:
            answer = "n"
        match answer.lower():
            case "" | "y" | "yes":
                return Decision(allow=True)
            case "a" | "always":
                return Decision(allow=True, remember=True)
            case "n" | "no":
                return Decision(allow=False)
        return Decision(allow=False, feedback=answer)

    def tool_finished(self, name: str, output: ToolOutput) -> None:
        style = "red" if output.is_error else "dim"
        summary = output.summary or ("error" if output.is_error else "done")
        self.console.print(Text(f"  ⎿ {summary}", style=style))
        if output.diff and not self._previewed:
            self._print_diff(output.diff)
        if output.is_error and name == "bash":
            tail = output.content.strip().splitlines()[-8:]
            self.console.print(Padding(Text("\n".join(tail), style="dim"), (0, 0, 0, 4)))

    @property
    def code_theme(self) -> str:
        return "ansi_light" if self.background == "light" else "ansi_dark"

    # --- history -----------------------------------------------------------

    def render_history(self, messages: list[Message], tools: dict[str, Tool], ctx: ToolContext,
                       max_messages: int = 60) -> None:
        """Print a past conversation the way it looked while it happened."""
        hidden = max(0, len(messages) - max_messages)
        if hidden:
            self.console.print(Text(f"… {hidden} earlier messages not shown", style="dim"))
        results = {b.tool_use_id: b for m in messages for b in m.content
                   if isinstance(b, ToolResultBlock)}
        for m in messages[hidden:]:
            for b in m.content:
                if isinstance(b, TextBlock) and m.role == "user":
                    if is_summary_note(b.text):
                        self.console.print(Rule("earlier conversation compacted", style="dim"))
                    else:
                        self.console.print(Text("\n› " + b.text, style="bold"))
                elif isinstance(b, TextBlock) and b.text.strip():
                    self.console.print(Markdown(b.text, code_theme=self.code_theme))
                elif isinstance(b, ToolUseBlock):
                    tool = tools.get(b.name)
                    self.tool_started(b.name, tool.describe(b.input, ctx) if tool else "")
                    result = results.get(b.id)
                    if result is not None and result.is_error:
                        first = (result.content.strip().splitlines() or ["error"])[0]
                        self.console.print(Text(f"  ⎿ {first}", style="red"))
        self.console.print(Rule("resumed", style="dim"))

    # --- misc --------------------------------------------------------------

    def notice(self, text: str) -> None:
        self._stop_spin()
        self.console.print(f"[yellow]{escape(text)}[/]")

    def error(self, text: str) -> None:
        self._stop_spin()
        self.console.print(f"[bold red]error:[/] {escape(text)}")

    def usage_line(self, context_tokens: int, context_window: int, output_tokens: int,
                   cost: float | None) -> None:
        pct = 100 * context_tokens / context_window if context_window else 0
        parts = [f"context {fmt_tokens(context_tokens)}/{fmt_tokens(context_window)} ({pct:.0f}%)",
                 f"output {fmt_tokens(output_tokens)} total"]
        if cost is not None:
            parts.append(f"${cost:.4f}")
        self.console.print(Text("  " + " · ".join(parts), style="dim"))

    # --- internals ---------------------------------------------------------

    def _print_diff(self, diff: str) -> None:
        lines = diff.splitlines()
        body = "\n".join(lines[:MAX_DIFF_LINES])
        if len(lines) > MAX_DIFF_LINES:
            body += f"\n... ({len(lines) - MAX_DIFF_LINES} more lines)"
        self.console.print(Padding(Syntax(body, "diff", theme=self.code_theme), (0, 0, 0, 4)))

    def _stream(self, text: str, kind: str, style: str) -> None:
        self._stop_spin()
        if self._stream_kind != kind:
            self._end_stream()
            self._stream_kind = kind
            if kind == "text":
                self._markdown = MarkdownStream(self.console, self.code_theme)
        if self._markdown is not None:
            self._markdown.feed(text)
        else:
            self.console.print(text, end="", style=style, markup=False, soft_wrap=True)

    def _end_stream(self) -> None:
        if self._markdown is not None:
            self._markdown.close()
            self._markdown = None
        elif self._stream_kind is not None:
            self.console.print()
        self._stream_kind = None

    def _spin(self, message: str) -> None:
        if not self.console.is_terminal:
            return
        if self._status is None:
            self._status = self.console.status(message, spinner="dots")
            self._status.start()
        else:
            self._status.update(message)

    def _stop_spin(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
