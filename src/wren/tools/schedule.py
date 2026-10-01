from __future__ import annotations

from pathlib import Path
from typing import Any

from wren.cron import Cron, CronError, describe_next
from wren.schedules import MODES, Job, ScheduleError, Scheduler, Schedules, describe_job, now
from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput


class ScheduleTool(Tool):
    """Scheduled runs: prompts that run headless on a cron schedule, in this
    directory. Creating or deleting one always asks the user, whatever the
    permission mode, since a job keeps running while nobody watches."""
    subagents = "never"

    name = "schedule"
    description = (
        "Manage scheduled runs: prompts that wren runs by itself on a cron schedule (in local "
        "time), headless, even when no session is open. Use it when the user asks for something "
        "to happen regularly or at a set time. Actions:\n"
        "- create: cron (5 fields, e.g. \"0 9 * * 1-5\", or @daily), prompt (complete and "
        "self-contained: the run starts with no conversation), optional id and mode (plan: "
        "read-only, the default; accept_edits: may edit files; auto: may run any command; only "
        "use more than plan when the task needs it)\n"
        "- list: the scheduled jobs\n"
        "- delete: remove a job by id\n"
        "The user must approve every create and delete."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "delete"]},
            "cron": {"type": "string"},
            "prompt": {"type": "string"},
            "id": {"type": "string", "description": "lowercase-kebab-case name"},
            "mode": {"type": "string", "enum": list(MODES)},
        },
        "required": ["action"],
    }

    def __init__(self, store: Schedules | None = None, scheduler: Scheduler | None = None):
        self.store = store or Schedules()
        self.scheduler = scheduler or Scheduler()

    def always_confirm(self, args: dict[str, Any]) -> bool:
        return args.get("action") in ("create", "delete")

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if args.get("action") == "create":
            return f"create {args.get('cron', '')}: {args.get('prompt', '')}"
        return f"{args.get('action', '')} {args.get('id', '')}".strip()

    def preview(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        if args.get("action") != "create":
            return None
        try:
            times = describe_next(Cron.parse(args.get("cron", "")), now(), 3)
        except CronError as e:
            return f"invalid schedule: {e}"
        lines = [f"cron:   {args.get('cron')}", f"mode:   {args.get('mode') or 'plan'}",
                 f"dir:    {ctx.cwd}", "next:   " + ", ".join(f"{t:%a %m-%d %H:%M}" for t in times),
                 "prompt:", *("  " + line for line in str(args.get("prompt", "")).splitlines())]
        return "\n".join(lines)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            match args["action"]:
                case "list":
                    jobs = self.store.jobs()
                    text = "\n".join(f"- {j.id}{' (paused)' if j.paused else ''}: {describe_job(j)}\n"
                                     f"  prompt: {j.prompt}" for j in jobs) or "(no scheduled jobs)"
                    return ToolOutput(text, summary=f"{len(jobs)} jobs")
                case "delete":
                    job_id = _need(args, "id")
                    self.store.remove(job_id)
                    return ToolOutput(f"Deleted scheduled job {job_id!r}.", summary=f"deleted {job_id}")
                case "create":
                    prompt = _need(args, "prompt")
                    job = self.store.add(Job(id=args.get("id") or self.store.new_id(prompt),
                                             cron=_need(args, "cron"), prompt=prompt,
                                             cwd=str(Path(ctx.cwd).resolve()), mode=args.get("mode") or "plan"))
                    text = f"Scheduled {job.id!r}: {describe_job(job)}."
                    if not self.scheduler.installed():
                        text += (" The scheduler isn't installed yet, so it won't run until the user "
                                 "runs `wren schedule install`; tell them.")
                    return ToolOutput(text, summary=f"scheduled {job.id}")
        except ScheduleError as e:
            raise ToolError(str(e)) from None
        raise ToolError(f"unknown action {args['action']!r}")


def _need(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{args['action']} needs {key!r}")
    return value
