from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput

if TYPE_CHECKING:
    from wren.skills import Skill


class SkillTool(Tool):
    name = "skill"
    description = (
        "Load a skill: task-specific instructions (and the files that come with them) listed "
        "under Skills in the system prompt. Load it before starting a task it applies to, then "
        "follow it. Its files can be read with read_file and scripts run with bash."
    )
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The skill's name"}},
        "required": ["name"],
    }
    read_only = True

    def __init__(self, skills: dict[str, Skill]):
        self.skills = {n: s for n, s in skills.items() if s.model_invocable}

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("name", "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        skill = self.skills.get(args["name"])
        if skill is None:
            raise ToolError(f"no skill named {args['name']!r}; available: {', '.join(sorted(self.skills))}")
        return ToolOutput(skill.content(), summary=f"loaded from {skill.source}")
