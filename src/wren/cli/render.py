"""How things look: the renderables for tool calls, diffs, plans, notices and
past conversations. Nothing here prints; the UIs decide where these go."""

from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from wren.agent.compact import is_summary_note
from wren.agent.plans import is_reminder
from wren.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from wren.tools import Tool, ToolContext, ToolOutput

MAX_DIFF_LINES = 80


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def code_theme(background: str) -> str:
    return "ansi_light" if background == "light" else "ansi_dark"


def indent(renderable: Any, depth: int) -> Any:
    """Inside a subagent's run (depth > 0), output is indented under its call."""
    return Padding(renderable, (0, 0, 0, 4 * depth)) if depth else renderable


# --- tool calls ----------------------------------------------------------------------


def tool_line(name: str, label: str, width: int, depth: int = 0) -> RenderableType:
    """`● name label`: a call as it starts (a subagent's dimmed)."""
    line = Text("● ", style="cyan")
    line.append(name, style="bold")
    if label:
        first, *rest = label.splitlines()
        line.append(" " + first + (" …" if rest else ""))
    if depth:
        line.stylize("dim")
    line.truncate(width * 2 - 4 * depth, overflow="ellipsis")
    return indent(line, depth)


def tool_result(name: str, output: ToolOutput, theme: str, depth: int = 0,
                previewed: bool = False) -> list[RenderableType]:
    """What a finished call shows: `⎿ summary`, its diff (unless the approval
    prompt showed it already), a task list, the end of a failed command."""
    shown: list[RenderableType] = []
    style = "red" if output.is_error else "dim"
    summary = output.summary or ("error" if output.is_error else "done")
    # A subagent's successful calls are one line each; errors and diffs still show.
    if not (depth and not output.is_error):
        shown.append(indent(Text(f"  ⎿ {summary}", style=style), depth))
    if output.diff and not previewed:
        shown.append(diff(output.diff, theme, depth))
    if output.display:
        shown.append(todos(output.display))
    if output.is_error and name == "bash":
        tail = output.content.strip().splitlines()[-8:]
        shown.append(Padding(Text("\n".join(tail), style="dim"), (0, 0, 0, 4 + 4 * depth)))
    return shown


def diff(text: str, theme: str, depth: int = 0) -> RenderableType:
    """A unified diff, colored, cut at MAX_DIFF_LINES."""
    lines = text.splitlines()
    body = "\n".join(lines[:MAX_DIFF_LINES])
    if len(lines) > MAX_DIFF_LINES:
        body += f"\n... ({len(lines) - MAX_DIFF_LINES} more lines)"
    return Padding(Syntax(body, "diff", theme=theme), (0, 0, 0, 4 + 4 * depth))


def command(text: str, theme: str) -> RenderableType:
    """A multi-line shell command, shown whole before it is approved."""
    return Padding(Syntax(text, "bash", theme=theme), (0, 0, 0, 4))


def approval_question(key: str) -> str:
    """The question under a call that needs approval; `key` is what "always" allows."""
    scope = key.removeprefix("bash:") if key.startswith("bash:") else key
    if len(scope) > 40:
        scope = "this exact command"
    return (f"  [bold]Allow?[/] [green]y[/]es · [green]a[/]lways for [bold]{escape(scope)}[/] "
            f"· [green]p[/]: always in this project · [red]n[/]o · or type what to do instead")


def todos(text: str) -> RenderableType:
    """A formatted task list (agent/todos.py): done struck through, the current one bold."""
    styled = Text()
    for line in text.splitlines():
        style = {"✓": "dim strike", "▸": "bold cyan"}.get(line[:1], "")
        styled.append(line + "\n", style=style)
    styled.rstrip()
    return Padding(styled, (0, 0, 0, 4))


def plan(text: str, theme: str) -> RenderableType:
    return Panel(Markdown(text, code_theme=theme), title="Plan", title_align="left",
                 border_style="cyan", padding=(0, 1))


def patch(text: str, theme: str) -> list[RenderableType]:
    """A whole `git diff --stat --patch`: the stat in bold, the patch colored."""
    stat, sep, body = text.partition("\ndiff --git ")
    shown: list[RenderableType] = [Text(stat.rstrip("\n"), style="bold")]
    if sep:
        shown.append(Syntax("diff --git " + body.rstrip("\n"), "diff", theme=theme))
    return shown


def subagent_progress(lines: list[str], width: int) -> Text:
    """The live lines of subagents running at the same time."""
    status = Text(f"{len(lines)} subagents")
    for line in lines:
        row = Text("\n  " + line, style="dim")
        row.truncate(max(20, width - 6) + 3, overflow="ellipsis")
        status.append_text(row)
    return status


# --- everything else -----------------------------------------------------------------


def prompt(text: str) -> Text:
    return Text("\n› " + text, style="bold")


def steered(text: str) -> Text:
    line = Text("› " + text, style="bold")
    line.append("  (sent while wren worked)", style="dim")
    return line


def attached(label: str, summary: str) -> Text:
    style = "yellow" if summary.startswith("not attached") else "dim"
    return Text(f"  @ {label} · {summary}", style=style)


def hook(name: str, status: str, depth: int = 0) -> RenderableType:
    style = "dim" if status.startswith("ok") else "yellow"
    return indent(Text(f"  ◆ {name} · {status}", style=style), depth)


def notice(text: str, depth: int = 0) -> RenderableType:
    return indent(Text(text, style="yellow"), depth)


def error(text: str, depth: int = 0) -> RenderableType:
    return indent(Text.assemble(("error:", "bold red"), " " + text), depth)


def usage(context_tokens: int, context_window: int, output_tokens: int, cost: float | None) -> Text:
    pct = 100 * context_tokens / context_window if context_window else 0
    parts = [f"context {fmt_tokens(context_tokens)}/{fmt_tokens(context_window)} ({pct:.0f}%)",
             f"output {fmt_tokens(output_tokens)} total"]
    if cost is not None:
        parts.append(f"${cost:.4f}")
    return Text("  " + " · ".join(parts), style="dim")


def history(messages: list[Message], tools: dict[str, Tool], ctx: ToolContext, theme: str,
            width: int, max_messages: int = 60) -> list[RenderableType]:
    """A past conversation the way it looked while it happened."""
    shown: list[RenderableType] = []
    hidden = max(0, len(messages) - max_messages)
    if hidden:
        shown.append(Text(f"… {hidden} earlier messages not shown", style="dim"))
    results = {b.tool_use_id: b for m in messages for b in m.content if isinstance(b, ToolResultBlock)}
    for m in messages[hidden:]:
        for b in m.content:
            if isinstance(b, TextBlock) and m.role == "user":
                if is_reminder(b.text):
                    continue
                if is_summary_note(b.text):
                    shown.append(Rule("earlier conversation compacted", style="dim"))
                else:
                    shown.append(prompt(b.text))
            elif isinstance(b, TextBlock) and b.text.strip():
                shown.append(Markdown(b.text, code_theme=theme))
            elif isinstance(b, ToolUseBlock):
                tool = tools.get(b.name)
                shown.append(tool_line(b.name, tool.describe(b.input, ctx) if tool else "", width))
                result = results.get(b.id)
                if result is not None and result.is_error:
                    first = (result.content.strip().splitlines() or ["error"])[0]
                    shown.append(Text(f"  ⎿ {first}", style="red"))
    shown.append(Rule("resumed", style="dim"))
    return shown
