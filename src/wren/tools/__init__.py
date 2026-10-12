from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput
from wren.tools.files import EditFile, ReadFile, WriteFile
from wren.tools.jobs import BashOutput, KillJob
from wren.tools.plan import ExitPlanMode
from wren.tools.search import Glob, Grep
from wren.tools.shell import Bash, SandboxedBash
from wren.tools.todo import TodoWrite


def default_tools(sandboxed: bool = False) -> list[Tool]:
    return [ReadFile(), WriteFile(), EditFile(), SandboxedBash() if sandboxed else Bash(), BashOutput(), KillJob(), Grep(), Glob(), TodoWrite(),
            ExitPlanMode()]


__all__ = ["Tool", "ToolContext", "ToolError", "ToolOutput", "default_tools"]
