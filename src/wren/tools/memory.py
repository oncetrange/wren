from __future__ import annotations

from typing import Any

from wren.memory import InvalidMemory, Memories, Memory
from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Your long-term memory across sessions (see Memory in the system prompt). Actions:\n"
        "- write: create or replace a memory (scope, name, type, description, content)\n"
        "- read: a memory's full text (scope, name)\n"
        "- delete: remove a memory that is wrong or outdated (scope, name)\n"
        "- list: every memory's name and description (optional scope)"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["write", "read", "delete", "list"]},
            "scope": {"type": "string", "enum": ["user", "project"],
                      "description": "user: true in every project; project: this project only"},
            "name": {"type": "string", "description": "Short kebab-case id, e.g. prefers-pytest"},
            "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
            "description": {"type": "string",
                            "description": "One line saying what it is about, used to judge "
                                           "relevance from the index"},
            "content": {"type": "string",
                        "description": "The fact. For feedback and project memories, follow it "
                                       "with 'Why:' and 'How to apply:' lines"},
        },
        "required": ["action"],
    }
    # Writes only wren's own memory directories, never the workspace: no approval needed.
    read_only = True

    def __init__(self, memories: Memories):
        self.memories = memories

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        target = f"{args.get('scope', '')}/{args['name']}" if args.get("name") else args.get("scope", "")
        return f"{args.get('action', '')} {target}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        action = args["action"]
        try:
            if action == "list":
                return self._list(args.get("scope"))
            store = self.memories.store(self._need(args, "scope"))
            name = self._need(args, "name")
            if action == "read":
                memory = store.get(name)
                if memory is None:
                    raise ToolError(f"no {store.scope} memory named {name!r}")
                return ToolOutput(memory.to_text(), summary=memory.description)
            if action == "delete":
                if not store.delete(name):
                    raise ToolError(f"no {store.scope} memory named {name!r}")
                return ToolOutput(f"Deleted {store.scope} memory {name!r}.",
                                  summary=f"forgot {store.scope}/{name}")
            if action == "write":
                memory = Memory(name=name, type=self._need(args, "type"),
                                description=" ".join(self._need(args, "description").split()),
                                body=self._need(args, "content"))
                replaced = store.write(memory)
                verb = "updated" if replaced else "saved"
                return ToolOutput(f"Memory {verb}: {store.scope}/{name}.",
                                  summary=f"{verb} {store.scope}/{name}: {memory.description}")
        except InvalidMemory as e:
            raise ToolError(str(e)) from None
        raise ToolError(f"unknown action {action!r}")

    def _list(self, scope: str | None) -> ToolOutput:
        stores = [self.memories.store(scope)] if scope else [self.memories.user, self.memories.project]
        parts = [f"{s.scope} memories:\n" + ("\n".join(s.index_lines()) or "(none)") for s in stores]
        count = sum(len(s.index_lines()) for s in stores)
        return ToolOutput("\n\n".join(parts), summary=f"{count} memories")

    @staticmethod
    def _need(args: dict[str, Any], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"{args['action']} needs {key!r}")
        return value
