"""wren's own policies, expressed as event handlers (see events.py)."""

from __future__ import annotations

from wren.agent.events import Hooks, PreToolUse, PromptSubmit, Stop, Verdict
from wren.agent.plans import FINAL_CHECK, PLAN_MODE_OFF, PLAN_MODE_ON, UNFINISHED_TODOS
from wren.agent.todos import format_todos


def plan_guard(e: PreToolUse) -> Verdict | None:
    """Plan mode is read-only. Registered first, so no hook can allow past it."""
    reason = e.agent.permissions.blocked(e.tool, e.call.input)
    return Verdict("deny", reason) if reason else None


def plan_reminder(e: PromptSubmit) -> Verdict | None:
    """Tell the model about plan mode: on every prompt while it's on, once when it ends."""
    agent = e.agent
    mode, told = agent.permissions.mode, agent.told_mode
    agent.told_mode = mode
    if mode == "plan":
        return Verdict(context=PLAN_MODE_ON)
    if told == "plan":
        return Verdict(context=PLAN_MODE_OFF)
    return None


def unfinished_todos(e: Stop) -> Verdict | None:
    """Once per request: finishing with open task-list items is often a slip."""
    if "unfinished_todos" in e.blocked_by or e.agent.permissions.mode == "plan":
        return None  # planning ends with a plan, not with finished tasks
    open_items = format_todos([t for t in e.agent.conv.todos if t.status != "completed"])
    return Verdict("block", UNFINISHED_TODOS.format(items=open_items)) if open_items else None


def final_check(e: Stop) -> Verdict | None:
    """Once per request that changed files: re-check the request's explicit instructions."""
    if "final_check" in e.blocked_by or not (e.agent.final_check and e.agent.changed):
        return None
    return Verdict("block", FINAL_CHECK)


def register_builtins(hooks: Hooks) -> None:
    hooks.on("pre_tool", "plan_guard", plan_guard)
    hooks.on("prompt", "plan_reminder", plan_reminder)
    hooks.on("stop", "unfinished_todos", unfinished_todos)
    hooks.on("stop", "final_check", final_check)
