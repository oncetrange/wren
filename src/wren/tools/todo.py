from __future__ import annotations

from typing import Any

from wren.agent.todos import STATUSES, format_todos, parse_todos, progress
from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput


class TodoWrite(Tool):
    subagents = "never"
    name = "todo_write"
    description = (
        "Maintain your task list for the current request. Use it for work with three or more "
        "steps, or when the user gives several things to do; skip it for simple tasks. Send the "
        "complete list every time (it replaces the previous one). Mark an item in_progress "
        "before starting it (only one at a time) and completed as soon as it is done; add "
        "items you discover along the way and remove ones that no longer apply. Include every "
        "explicit instruction from the user as its own item (e.g. 'commit the changes', 'run "
        "the test suite'), so none is forgotten."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The full task list, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to do, as a short imperative"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    }
    read_only = True  # only changes the agent's own bookkeeping

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        todos = args.get("todos")
        return f"{len(todos)} items" if isinstance(todos, list) else ""

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            items = parse_todos(args["todos"])
        except ValueError as e:
            raise ToolError(str(e)) from None
        current = next((t.content for t in items if t.status == "in_progress"), None)
        text = f"Task list updated ({progress(items)})."
        if current:
            text += f" In progress: {current}"
        elif items and all(t.status == "completed" for t in items):
            text += " All items are completed."
        return ToolOutput(text, summary=progress(items) if items else "cleared",
                          display=format_todos(items) or None, todos=items)
