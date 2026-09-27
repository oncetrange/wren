from __future__ import annotations

from typing import Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput


class ExitPlanMode(Tool):
    """Presents the plan for approval. The agent handles the call itself,
    since it needs the user; `run` is never reached through the agent."""

    name = "exit_plan_mode"
    description = (
        "Only in plan mode: present your implementation plan to the user for approval, once "
        "you have investigated enough. Write it in markdown: a title line, the files to change "
        "and what changes in each, and how you will verify the result. If approved, plan mode "
        "ends and you carry out the plan; otherwise revise it according to the user's message."
    )
    input_schema = {
        "type": "object",
        "properties": {"plan": {"type": "string", "description": "The plan, in markdown"}},
        "required": ["plan"],
    }
    read_only = True

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        plan = args.get("plan", "")
        return next((l.strip("# ").strip() for l in plan.splitlines() if l.strip()), "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        raise ToolError("exit_plan_mode must be handled by the agent")
