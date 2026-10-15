"""Terminal rendering of agent events."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterator
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.status import Status
from rich.text import Text

from wren.agent.permissions import Decision
from wren.agent.plans import PlanDecision
from wren.agent.ui import AgentUI
from wren.cli import render
from wren.cli.markdown import MarkdownStream
from wren.cli.screen import Screen
from wren.cli.typeahead import TypeAhead
from wren.llm.types import Message
from wren.tools import Tool, ToolContext, ToolOutput


class RichUI(AgentUI):
    def __init__(self, console: Console | None = None, interactive: bool | None = None,
                 background: str = "dark"):
        self.console = console or Console(highlight=False)
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.background = background  # "dark" | "light": picks syntax colors
        self.screen = Screen(self.console, self.interactive)
        self._status: Status | None = None
        self._stream_kind: str | None = None  # "text" | "thinking" while streaming
        self._markdown: MarkdownStream | None = None
        self._previewed = False
        # Inside a subagent's run: its tool calls print as compact, indented lines.
        self._depth = 0
        # While a prompt runs in the REPL: what the user is typing meanwhile, and
        # how many messages are waiting for the model (see typing_ahead).
        self.typeahead: TypeAhead | None = None
        self.queued: Callable[[], int] = lambda: 0

    @contextlib.contextmanager
    def typing_ahead(self, ahead: TypeAhead, queued: Callable[[], int]) -> Iterator[None]:
        self.typeahead = self.screen.typeahead = ahead
        self.queued = queued
        try:
            with ahead:
                yield
        finally:
            self.typeahead = self.screen.typeahead = None
            self.queued = lambda: 0

    @contextlib.contextmanager
    def nested(self) -> Iterator[None]:
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

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
        self._print(render.tool_line(name, label, self.console.width, self._depth))
        if self.typeahead is not None and not self._depth:
            self._spin(Text(f"Running {name}…", style="dim"))

    def confirm(self, tool: Tool, args: dict[str, Any], label: str, preview: str | None,
                key: str | None = None) -> Decision:
        self._stop_spin()
        if preview:
            self._print(render.diff(preview, self.code_theme, self._depth))
            self._previewed = True
        elif "\n" in label:
            self._print(render.command(label, self.code_theme))
        if not self.interactive:
            # Leaving the sandbox asks even with --yolo, so don't suggest it then.
            hint = "" if label.endswith("(outside the sandbox)") else "; use --yolo"
            self._print(f"  [yellow]denied (no terminal to ask for approval{hint})[/]")
            return Decision(allow=False)

        return self._decision(render.approval_question(key or tool.permission_key(args)))

    def allow_host(self, host: str) -> Decision | None:
        self._stop_spin()
        if not self.interactive:
            return None
        return self._decision(render.host_question(host))

    def _decision(self, question: str) -> Decision:
        try:
            answer = self.screen.ask_text(question).strip()
        except EOFError:
            answer = "n"
        match answer.lower():
            case "" | "y" | "yes":
                return Decision(allow=True)
            case "a" | "always":
                return Decision(allow=True, remember=True)
            case "p" | "project":
                return Decision(allow=True, remember=True, persist=True)
            case "n" | "no":
                return Decision(allow=False)
        return Decision(allow=False, feedback=answer)

    def review_plan(self, plan: str) -> PlanDecision | None:
        self._stop_spin()
        self._print(render.plan(plan, self.code_theme))
        if not self.interactive:
            return None
        choice = self.screen.pick("Approve this plan?", [
            ("accept_edits", "Yes, and auto-accept edits"),
            ("ask", "Yes, and ask before each edit"),
            ("revise", "No, keep planning"),
        ], default="accept_edits")
        if choice in ("accept_edits", "ask"):
            return PlanDecision(True, mode=choice)
        if choice == "revise":
            question = "  [bold]What should change?[/] [dim](Enter to skip)[/]"
            return PlanDecision(False, feedback=self.screen.ask_text(question).strip())
        return PlanDecision(False)

    def tool_finished(self, name: str, output: ToolOutput) -> None:
        self._stop_spin()
        for shown in render.tool_result(name, output, self.code_theme, self._depth, self._previewed):
            self._print(shown)

    @property
    def code_theme(self) -> str:
        return render.code_theme(self.background)

    # --- history -----------------------------------------------------------

    def render_history(self, messages: list[Message], tools: dict[str, Tool], ctx: ToolContext) -> None:
        """Show a past conversation the way it looked while it happened."""
        for shown in render.history(messages, tools, ctx, self.code_theme, self.console.width):
            self._print(shown)

    def waiting(self, text: str | None) -> None:
        if text is None:
            self._stop_spin()
        else:
            self._spin(Text(text, style="dim"))

    def progress(self, lines: list[str] | None) -> None:
        """Live status lines for subagents running concurrently; None ends it."""
        if lines is None:
            self._stop_spin()
            return
        self._spin(render.subagent_progress(lines, self.console.width))

    # --- misc --------------------------------------------------------------

    def steered(self, text: str) -> None:
        self._stop_spin()
        self._end_stream()
        self._print(render.steered(text))

    def attached(self, label: str, summary: str) -> None:
        """A file @-mentioned in the prompt, and what was attached."""
        self._print(render.attached(label, summary))

    def hook_ran(self, name: str, status: str) -> None:
        self._stop_spin()
        self._print(render.hook(name, status, self._depth))

    def notice(self, text: str) -> None:
        self._stop_spin()
        self._print(render.notice(text, self._depth))

    def error(self, text: str) -> None:
        self._stop_spin()
        self._print(render.error(text, self._depth))

    def usage_line(self, context_tokens: int, context_window: int, output_tokens: int,
                   cost: float | None) -> None:
        self._print(render.usage(context_tokens, context_window, output_tokens, cost))

    # --- internals ---------------------------------------------------------

    def _print(self, *renderables: Any, **kwargs: Any) -> None:
        """Everything the UI shows goes out here."""
        self.console.print(*renderables, **kwargs)

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
            self._print(text, end="", style=style, markup=False, soft_wrap=True)

    def _end_stream(self) -> None:
        if self._markdown is not None:
            self._markdown.close()
            self._markdown = None
        elif self._stream_kind is not None:
            self._print()
        self._stream_kind = None

    def _spin(self, message: str | Text) -> None:
        if not self.console.is_terminal:
            return
        shown = _WithTyping(self, message) if self.typeahead is not None else message
        if self._status is None:
            self._status = self.console.status(shown, spinner="dots")
            self._status.start()
        else:
            self._status.update(shown)

    def typing_lines(self) -> list[Text]:
        """Under the spinner while a prompt runs: messages waiting, and the one being typed."""
        lines = []
        if n := self.queued():
            lines.append(Text(f"⏎ {n} message{'s' if n != 1 else ''} for wren after this step",
                              style="dim"))
        if self.typeahead is not None and self.typeahead.draft:
            lines.append(Text("› " + self.typeahead.draft + "▏", style="bold"))
        return lines

    def _stop_spin(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None


class _WithTyping:
    """A spinner message re-rendered on every refresh, so what is typed shows at once."""

    def __init__(self, ui: RichUI, message: str | Text):
        self.ui, self.message = ui, Text(message) if isinstance(message, str) else message

    def __rich__(self) -> RenderableType:
        lines = self.ui.typing_lines()
        return Group(self.message, *lines) if lines else self.message
