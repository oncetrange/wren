from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from wren.llm.types import ToolSpec

if TYPE_CHECKING:
    from wren.tools.jobs import Jobs

MAX_OUTPUT_CHARS = 30_000


@dataclass
class ToolContext:
    cwd: Path
    # Files the model has read, mapped to their mtime at read time. Edits are
    # refused on files that were never read or changed on disk since.
    read_files: dict[Path, int] = field(default_factory=dict)
    # Where the bash tool's shell currently is; `cd` persists across calls.
    shell_cwd: Path | None = None
    # Commands running in the background (see tools/jobs.py).
    jobs: Jobs = field(default_factory=lambda: _jobs())

    @property
    def bash_cwd(self) -> Path:
        if self.shell_cwd is not None and self.shell_cwd.is_dir():
            return self.shell_cwd
        return self.cwd

    def resolve(self, path: str) -> Path:
        return (self.cwd / Path(path).expanduser()).resolve()

    def display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)

    def mark_read(self, path: Path) -> None:
        self.read_files[path] = path.stat().st_mtime_ns


@dataclass
class ToolOutput:
    content: str
    is_error: bool = False
    # What the terminal shows the user; the model always gets `content`.
    summary: str = ""
    diff: str | None = None
    # Extra text shown under the tool line (e.g. the task list).
    display: str | None = None
    # A new task list, for todo_write; the agent stores it (list[TodoItem]).
    todos: list[Any] | None = None


class ToolError(Exception):
    """Raised inside a tool to return an error result to the model."""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]
    read_only: ClassVar[bool] = False
    # Writes files in the workspace (as opposed to running arbitrary commands).
    edits_files: ClassVar[bool] = False
    # Check arguments against input_schema before running (MCP tools leave it to their server).
    strict_args: ClassVar[bool] = True
    # Whether subagents get this tool (the parent's instance): "inherit" (if their
    # type allows it), "writers" (only subagents that may change things: the tool
    # reaches outside the machine) or "never" (it belongs to the conversation with
    # the user, like the task list, or would let subagents nest).
    subagents: ClassVar[Literal["inherit", "writers", "never"]] = "inherit"

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput: ...

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """One-line label for the call, e.g. the file path or command."""
        return ""

    def permission_key(self, args: dict[str, Any]) -> str:
        """Granularity of an "always allow" decision."""
        return self.name

    def always_confirm(self, args: dict[str, Any]) -> bool:
        """Ask the user before this call whatever the permission mode or hooks say
        (and never remember the answer). For calls with lasting effects, like
        creating a scheduled run."""
        return False

    def concurrent_safe(self, args: dict[str, Any]) -> bool:
        """Whether this call may run alongside others: it must never ask for
        approval and never change anything. Only read-only subagents so far."""
        return False

    def preview(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        """A diff shown before asking permission, if the tool can compute one."""
        return None


def _jobs() -> Jobs:
    from wren.tools.jobs import Jobs

    return Jobs()


def validate_args(schema: dict[str, Any], args: Any) -> str | None:
    """Minimal JSON-schema check covering what our tool schemas use."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    types = {"string": str, "integer": int, "boolean": bool, "object": dict, "array": list}
    for key, value in args.items():
        if key not in props:
            return f"unknown argument {key!r}; expected: {', '.join(props)}"
        expected = types.get(props[key].get("type", ""))
        if expected and (
            not isinstance(value, expected)
            or (expected is int and isinstance(value, bool))
        ):
            return f"argument {key!r} must be of type {props[key]['type']}"
    return None


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n... [{omitted} characters truncated] ...\n\n{tail}"


def read_text(path: Path) -> str:
    """Read a text file preserving its line endings."""
    with path.open(encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
