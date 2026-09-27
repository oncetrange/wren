"""Plan mode: reminders for the model, plan review decisions, plan files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# Marks text wren adds to a user message for the model; hidden in transcripts.
REMINDER_TAG = "<wren-reminder>"

PLAN_MODE_ON = f"""{REMINDER_TAG}
Plan mode is on. Investigate with read-only tools (read_file, grep, glob; bash only for \
commands that change nothing) and do not modify any files. When you understand the task, \
call exit_plan_mode with a concrete plan: the files to change, what changes in each, and \
how you will verify the result. The user will approve it or ask for changes.
</wren-reminder>"""

PLAN_MODE_OFF = f"""{REMINDER_TAG}
Plan mode is off: you may modify files again.
</wren-reminder>"""


UNFINISHED_TODOS = """Your task list still has unfinished items:
{items}
If they're done, blocked, waiting on the user, or no longer needed, say so and update the \
list with todo_write. Otherwise continue with the next one."""

FINAL_CHECK = """Before you finish, re-read the user's request and check that every explicit \
instruction in it is satisfied: for example git operations they asked for (branch, commit), \
tests to run, or where and in what form to deliver the work. Also remove scratch files and \
build outputs you created that are not part of the requested change (and make sure none of \
them were committed). Do whatever is missing. If everything is done, reply with your final \
summary."""

TURN_BUDGET = """You have {left} model calls left before you are stopped. Wrap up: make \
sure the work is saved the way the user asked (e.g. committed if they asked for a commit), \
then give your final summary."""


def reminder(*parts: str) -> str:
    """Wrap reminder text for the model (hidden in transcripts)."""
    return f"{REMINDER_TAG}\n" + "\n\n".join(parts) + "\n</wren-reminder>"


def is_reminder(text: str) -> bool:
    return text.startswith(REMINDER_TAG)


WREN_DIR_GITIGNORE = "*\n!hooks.toml\n"


@dataclass
class PlanDecision:
    approved: bool
    # Permission mode to continue in once approved.
    mode: Literal["accept_edits", "ask"] = "ask"
    # When not approved: what the user wants changed ("" = just stop).
    feedback: str = ""


def save_plan(cwd: Path, plan: str) -> Path:
    """Write the plan to <cwd>/.wren/plans/<timestamp>-<slug>.md.

    .wren/ gets a .gitignore on creation that ignores everything but hooks.toml
    (which is meant to be shared), so plans stay out of git status and workspace
    checkpoints; edit it to version your plans.
    """
    root = cwd / ".wren"
    if not root.exists():
        root.mkdir()
        (root / ".gitignore").write_text(WREN_DIR_GITIGNORE)
    plans = root / "plans"
    plans.mkdir(exist_ok=True)
    title = next((l.strip("# ").strip() for l in plan.splitlines() if l.strip()), "plan")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "plan"
    path = plans / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
    path.write_text(plan.rstrip() + "\n", encoding="utf-8")
    return path
