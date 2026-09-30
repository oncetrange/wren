from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from wren.tools.base import Tool, ToolContext, ToolError, ToolOutput, truncate
from wren.tools.jobs import start_in_background

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
_COMPOUND = ("&&", "||", ";", "|", "`", "$(", ">", "<", "\n")


class Bash(Tool):
    name = "bash"
    description = (
        "Run a shell command with bash in the working directory and return its combined "
        "stdout/stderr and exit code. The working directory persists between calls (cd sticks), "
        "other shell state such as variables does not. "
        "Commands are non-interactive: stdin is closed, so avoid anything that prompts. "
        "Use read_file/grep/glob instead of cat/grep/find for reading and searching files. "
        "For commands that keep running (dev servers, watchers, long builds), set "
        "run_in_background: the call returns a job id at once, bash_output reads the job's "
        "output and kill_job stops it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})",
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Start it as a background job instead of waiting for it",
            },
        },
        "required": ["command"],
    }

    def describe(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return args.get("command", "")

    def permission_key(self, args: dict[str, Any]) -> str:
        command = args.get("command", "").strip()
        # "Always allow" on a simple command covers its program (`bash:pytest`);
        # compound commands can hide anything, so they are only ever allowed verbatim.
        if any(tok in command for tok in _COMPOUND):
            return f"bash:{command}"
        try:
            program = shlex.split(command)[0]
        except (ValueError, IndexError):
            return f"bash:{command}"
        return f"bash:{program}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if args.get("run_in_background"):
            env = {**os.environ, "PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"}
            return start_in_background(args["command"], ctx, env)
        timeout = min(max(args.get("timeout", DEFAULT_TIMEOUT), 1), MAX_TIMEOUT)
        cwd_file = Path(tempfile.mkstemp(prefix="wren-cwd-")[1])
        env = {**os.environ, "PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
               "WREN_CWD_FILE": str(cwd_file)}
        # The EXIT trap records where the shell ended up, even if the command calls `exit`.
        script = f'trap \'pwd -P > "$WREN_CWD_FILE"\' EXIT\n{args["command"]}'
        try:
            return self._run(script, timeout, env, cwd_file, ctx)
        finally:
            cwd_file.unlink(missing_ok=True)

    def _run(self, script: str, timeout: int, env: dict[str, str], cwd_file: Path,
             ctx: ToolContext) -> ToolOutput:
        start_dir = ctx.bash_cwd
        proc = subprocess.Popen(
            ["bash", "-c", script],
            cwd=start_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so we can kill children too
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, _ = proc.communicate()
            raise ToolError(
                f"command timed out after {timeout}s\n{truncate(out.decode(errors='replace'))}"
            ) from None
        except BaseException:
            _kill_group(proc)
            raise

        text = truncate(out.decode(errors="replace").rstrip())
        code = proc.returncode
        content = f"{text}\n\n[exit code {code}]" if text else f"[exit code {code}]"
        new_dir = cwd_file.read_text().strip() if cwd_file.exists() else ""
        if new_dir and Path(new_dir) != start_dir:
            ctx.shell_cwd = Path(new_dir)
            content += f"\n[working directory is now {new_dir}]"
        lines = text.count("\n") + 1 if text else 0
        return ToolOutput(
            content,
            is_error=code != 0,
            summary=f"exit {code}, {lines} line{'s' if lines != 1 else ''} of output",
        )


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
