"""Subagents: the task tool hands a self-contained job to a fresh agent.

A subagent is an ordinary `Agent` with its own conversation, a narrower tool
set and a system prompt for working unattended. Only its final report goes
back to the main agent, so the files it read and the searches it ran never
enter the main context.

It shares the main agent's provider, permissions (mode and "always allow"
list) and tool hooks. It takes no checkpoints of its own: its changes happen
inside the main agent's turn, so that turn's restore point already covers them.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wren.agent.events import Hooks, PreToolUse, Verdict
from wren.agent.prompt import build_system_prompt
from wren.agent.session import SESSIONS_DIR, SessionLog
from wren.llm.types import ToolSpec
from wren.skills import prompt_section
from wren.tools import Tool, ToolContext, ToolError, ToolOutput, default_tools
from wren.tools.readonly import is_read_only_command

if TYPE_CHECKING:
    from wren.agent.loop import Agent, AgentUI

SUBAGENTS_DIR = SESSIONS_DIR / "subagents"
DEFAULT_MAX_TURNS = 30
# Never given to a subagent: no nesting, and the task list and plan approval
# belong to the conversation with the user.
EXCLUDED_TOOLS = frozenset({"task", "todo_write", "exit_plan_mode"})

SUBAGENT_BASE = """\
You are a subagent of Wren, a coding agent working in the user's terminal. The main agent \
gave you one task. You work on it alone with your tools: nobody reads your intermediate \
messages, and nobody can answer questions, so make sensible assumptions and state them.

When you are done, reply with your final report. It is the only thing the main agent \
receives, so make it complete on its own: the answer or outcome first, then the supporting \
details it will need (file paths as path:line, names, short code excerpts). Leave out your \
process and anything you checked that turned out irrelevant.

# How to work
- Locate code with grep/glob and read it; don't guess at file contents or APIs you haven't seen.
- When independent pieces of information are needed, request several tool calls at once.
- Stay within the task. Don't commit, push or run destructive commands.
"""

EXPLORE_PROMPT = """\
# Your role: explore
Answer a question about the codebase by searching and reading. You can't modify anything: \
bash only runs read-only commands (ls, git log/diff/show, find, wc and the like). Be thorough \
where it matters, efficient elsewhere: search broadly first, then read the relevant parts. \
Report where things are (path:line) and how they work, precisely enough that the main agent \
doesn't have to repeat your search.
"""

GENERAL_PROMPT = """\
# Your role: general
Complete the task, including changing files when it asks for that. Make the smallest change \
that fully solves it, matching the surrounding code's style. Verify your changes when you can \
(run the relevant tests). Report what you changed (files and what each change does), how you \
verified it, and anything left unfinished or uncertain.
"""


@dataclass
class AgentType:
    name: str
    description: str  # when the main agent should pick it (shown in the task tool)
    prompt: str       # appended to the subagent system prompt
    tools: frozenset[str] | None = None  # None: every tool not excluded
    read_only: bool = False
    max_turns: int = DEFAULT_MAX_TURNS
    source: str = "built-in"

    def allows(self, tool: str) -> bool:
        return tool not in EXCLUDED_TOOLS and (self.tools is None or tool in self.tools)


def builtin_agent_types() -> dict[str, AgentType]:
    return {
        "explore": AgentType(
            "explore",
            "Read-only: find where and how things are in the codebase (search, read, git "
            "history). Use it for broad searches and questions spanning many files.",
            EXPLORE_PROMPT,
            tools=frozenset({"read_file", "grep", "glob", "bash"}),
            read_only=True,
        ),
        "general": AgentType(
            "general",
            "Can read, edit files and run commands: a self-contained piece of work you can "
            "describe completely (e.g. a change in one module, a failing test to investigate).",
            GENERAL_PROMPT,
        ),
    }


def read_only_guard(e: PreToolUse) -> Verdict | None:
    """For read-only subagents: reads run freely, read-only commands too, the rest is refused."""
    if e.tool.name == "bash":
        if is_read_only_command(e.call.input.get("command", "")):
            return Verdict("allow")
        return Verdict("deny", "This subagent is read-only: bash only runs commands that read "
                               "(ls, cat, grep, find, wc, git log/diff/show/status and the like), "
                               "without redirections into files or command substitution.")
    if not e.tool.read_only:
        return Verdict("deny", f"This subagent is read-only and can't use {e.tool.name}.")
    return None


class SubagentUI:
    """Shows a subagent's tool calls nested under the task call; its text isn't streamed."""

    def __init__(self, ui: AgentUI):
        self.ui = ui

    @contextlib.contextmanager
    def _nested(self) -> Iterator[None]:
        nested = getattr(self.ui, "nested", None)
        with nested() if nested else contextlib.nullcontext():
            yield

    def model_started(self) -> None: self.ui.model_started()
    def text_delta(self, text: str) -> None: pass
    def thinking_delta(self, text: str) -> None: pass
    def tool_call_started(self, name: str) -> None: self.ui.tool_call_started(name)
    def model_finished(self) -> None: self.ui.model_finished()

    def tool_started(self, name: str, label: str) -> None:
        with self._nested():
            self.ui.tool_started(name, label)

    def confirm(self, tool: Tool, args: dict[str, Any], label: str, preview: str | None):
        with self._nested():
            return self.ui.confirm(tool, args, label, preview)

    def review_plan(self, plan: str):
        return None  # subagents have no exit_plan_mode

    def tool_finished(self, name: str, output: ToolOutput) -> None:
        with self._nested():
            self.ui.tool_finished(name, output)

    def hook_ran(self, name: str, status: str) -> None:
        with self._nested():
            self.ui.hook_ran(name, status)

    def notice(self, text: str) -> None:
        with self._nested():
            self.ui.notice(text)

    def error(self, text: str) -> None:
        with self._nested():
            self.ui.error(text)


class TaskTool(Tool):
    name = "task"
    description = (
        "Delegate a self-contained task to a subagent. It works in its own context with its own "
        "tools and returns only its final report, so its searching and reading don't fill "
        "yours. Use it for broad searches (\"where/how is X handled?\"), for several independent "
        "questions (call it several times in one response), or for a well-defined piece of work "
        "you can describe completely. Don't use it for a file you already know or a single "
        "grep: do those yourself.\n"
        "The subagent can't see this conversation: the prompt must contain everything it needs "
        "(the goal, relevant paths and findings so far, constraints, what to report back). Its "
        "report isn't shown to the user; pass on what matters."
    )
    read_only = True  # its subagent's tools ask for permission themselves

    def __init__(self, parent: Agent, types: dict[str, AgentType]):
        self.parent = parent
        self.types = types

    @property
    def input_schema(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "type": "object",
            "properties": {
                "description": {"type": "string",
                                "description": "A short (3-5 word) label for the task"},
                "prompt": {"type": "string",
                           "description": "The complete task for the subagent"},
                "agent": {"type": "string", "enum": sorted(self.types),
                          "description": "The kind of subagent (default: general)"},
            },
            "required": ["description", "prompt"],
        }

    def spec(self) -> ToolSpec:
        kinds = "\n".join(f"- {t.name}: {t.description}" for t in self.types.values())
        return ToolSpec(self.name, f"{self.description}\n\nSubagents:\n{kinds}", self.input_schema)

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"{args.get('description', '')} ({args.get('agent') or 'general'})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        kind = self.types.get(args.get("agent") or "general")
        if kind is None:
            raise ToolError(f"no subagent type {args['agent']!r}; available: {', '.join(sorted(self.types))}")
        if self.parent.permissions.mode == "plan" and not kind.read_only:
            read_only = ", ".join(sorted(t.name for t in self.types.values() if t.read_only))
            raise ToolError("Plan mode is on, so only read-only subagents can run"
                            + (f" ({read_only})." if read_only else "."))
        return run_subagent(self.parent, kind, args["description"], args["prompt"])


def run_subagent(parent: Agent, kind: AgentType, description: str, prompt: str) -> ToolOutput:
    from wren.agent.loop import Agent

    hooks = Hooks()
    if kind.read_only:
        hooks.on("pre_tool", "read_only_guard", read_only_guard)
    # The parent's tool hooks (plan_guard, user hooks) apply to the subagent's tools too.
    for event in ("pre_tool", "post_tool", "notification"):
        for reg in parent.hooks.registrations(event):
            hooks.on(event, reg.name, reg.fn)

    skills = parent.skills if kind.allows("skill") else {}
    log = _log(parent)
    child = Agent(
        parent.provider,
        parent.model,
        # Its own read tracking: a file it read hasn't been seen by the parent.
        ToolContext(cwd=parent.ctx.cwd),
        SubagentUI(parent.ui),
        permissions=parent.permissions,
        log=log,
        tools=[t for t in default_tools() if kind.allows(t.name)],
        max_turns=kind.max_turns,
        skills=skills,
        system=build_system_prompt(parent.ctx.cwd, prompt_section(skills),
                                   base=f"{SUBAGENT_BASE}\n{kind.prompt}"),
        hooks=hooks,
    )
    parent.log.record("subagent", agent=kind.name, description=description,
                      log=str(log.path) if log.path else None)
    report = child.run(prompt)

    parent.add_usage(child.usage, child.cost, child.model.name, purpose="subagent")
    parent.changed = parent.changed or child.changed
    tokens = child.usage.input_tokens + child.usage.cache_read_tokens + child.usage.cache_write_tokens
    run = {"agent": kind.name, "description": description, "status": child.status,
           "turns": child.turns, "tool_calls": child.tool_calls, "cost": child.cost}
    parent.subagent_runs.append(run)
    parent.log.record("subagent_end", **run)
    if child.status == "interrupted":
        raise KeyboardInterrupt  # stop the main agent's turn too

    stats = (f"{child.turns} turn{'s' if child.turns != 1 else ''} · {child.tool_calls} tool "
             f"call{'s' if child.tool_calls != 1 else ''} · {tokens / 1000:.1f}k tokens in")
    if child.status == "done" and report.strip():
        return ToolOutput(report, summary=f"{kind.name} · {stats}")
    why = {"max_turns": f"it ran out of turns ({kind.max_turns})",
           "error": "a model request failed",
           "refusal": "the model declined to continue"}.get(child.status, "it finished without a report")
    partial = f"\n\nIts last message:\n{report}" if report.strip() else ""
    return ToolOutput(f"The subagent stopped before finishing: {why}.{partial}", is_error=True,
                      summary=f"{kind.name} stopped: {why} · {stats}")


def _log(parent: Agent) -> SessionLog:
    """The subagent's own log, next to the parent's; the parent records where it is."""
    if parent.log.path is None:
        return SessionLog(directory=None)
    child_id = f"{parent.log.id}-{uuid.uuid4().hex[:6]}"
    SUBAGENTS_DIR.mkdir(parents=True, exist_ok=True)
    log = SessionLog(path=SUBAGENTS_DIR / f"{child_id}.jsonl")
    log.id = child_id  # also names its transcript archive, if it compacts
    return log
