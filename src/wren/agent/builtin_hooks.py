"""wren's own policies, expressed as event handlers (see events.py)."""

from __future__ import annotations

import hashlib
import json

from wren.agent.events import Hooks, PostToolUse, PreToolUse, PromptSubmit, Stop, Verdict
from wren.agent.plans import (
    FAILURE_STREAK,
    FINAL_CHECK,
    PLAN_MODE_OFF,
    PLAN_MODE_ON,
    REPEATED_FAILURE,
    UNFINISHED_TODOS,
    reminder,
)
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


def finished_jobs(e: PostToolUse | PromptSubmit) -> Verdict | None:
    """Tell the model when background jobs it started have ended (once per job)."""
    done = e.agent.ctx.jobs.newly_finished()
    if not done:
        return None
    lines = [f"- {j.id} (`{j.command[:80]}`) {j.status}" for j in done]
    return Verdict(context=reminder("Background jobs finished:\n" + "\n".join(lines)
                                    + "\nRead their final output with bash_output if needed."))


# Say it when a failure repeats this often, and again at twice that.
REPEATS, STREAK = 3, 5


def repeated_failures(e: PostToolUse) -> Verdict | None:
    """Point out loops: the same call failing the same way, or a run of failures.

    Rerunning a failing test while fixing it isn't a loop (its output changes),
    so a call counts as repeated only with identical input and output."""
    agent = e.agent
    if not e.output.is_error:
        agent.failure_streak = 0
        return None
    agent.failure_streak += 1
    key = hashlib.sha256(json.dumps([e.call.name, e.call.input, e.output.content], sort_keys=True,
                                    default=str).encode()).hexdigest()
    agent.failure_counts[key] = n = agent.failure_counts.get(key, 0) + 1
    if n in (REPEATS, 2 * REPEATS):
        agent.log.record("reminder", reason="repeated_failure", tool=e.call.name, count=n)
        return Verdict(context=reminder(REPEATED_FAILURE.format(n=n)))
    if agent.failure_streak in (STREAK, 2 * STREAK):
        agent.log.record("reminder", reason="failure_streak", count=agent.failure_streak)
        return Verdict(context=reminder(FAILURE_STREAK.format(n=agent.failure_streak)))
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
    hooks.on("post_tool", "finished_jobs", finished_jobs)
    hooks.on("post_tool", "repeated_failures", repeated_failures)
    hooks.on("prompt", "plan_reminder", plan_reminder)
    hooks.on("prompt", "finished_jobs", finished_jobs)
    hooks.on("stop", "unfinished_todos", unfinished_todos)
    hooks.on("stop", "final_check", final_check)
