"""User hooks: shell commands run on lifecycle events.

Protocol (close to Claude Code's, so existing scripts port easily):

- The event is passed as JSON on stdin, plus environment variables
  WREN_EVENT, WREN_PROJECT_DIR, WREN_SESSION_ID, and for tool events
  WREN_TOOL_NAME and WREN_FILE (the path argument, when there is one).
- Exit 0: success. For UserPromptSubmit and SessionStart, stdout is added as
  context for the model.
- Exit 2: blocking. PreToolUse: the call is refused with stderr as the reason.
  PostToolUse: stderr is shown to the model. UserPromptSubmit: the prompt is
  rejected. Stop: the model is told stderr and keeps going.
- Other codes: a non-blocking error, shown to the user only.
- On exit 0, stdout may instead be a JSON object: {"decision": "allow" |
  "deny" | "block", "reason": "...", "additional_context": "..."}.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wren.agent.events import (
    Notification,
    PostToolUse,
    PreToolUse,
    PromptSubmit,
    SessionEnd,
    SessionStart,
    Stop,
    Verdict,
)
from wren.config import HookSpec

if TYPE_CHECKING:
    from wren.agent.loop import Agent

# Our event names for each user-facing hook event.
EVENTS = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "prompt",
    "PreToolUse": "pre_tool",
    "PostToolUse": "post_tool",
    "Stop": "stop",
    "Notification": "notification",
    "SessionEnd": "session_end",
}
MAX_OUTPUT = 10_000


@dataclass
class HookRun:
    code: int | None  # None: timed out
    stdout: str
    stderr: str
    seconds: float


def run_command(spec: HookSpec, payload: dict[str, Any], agent: Agent) -> HookRun:
    env = {
        **os.environ,
        "WREN_EVENT": spec.event,
        "WREN_PROJECT_DIR": str(agent.ctx.cwd),
        "WREN_SESSION_ID": agent.log.id,
    }
    if "tool_name" in payload:
        env["WREN_TOOL_NAME"] = payload["tool_name"]
        path = payload.get("tool_input", {}).get("path")
        if isinstance(path, str):
            env["WREN_FILE"] = str(agent.ctx.resolve(path))
    start = time.monotonic()
    proc = subprocess.Popen(["bash", "-c", spec.command], cwd=agent.ctx.cwd, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(json.dumps(payload, ensure_ascii=False), timeout=spec.timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, err = proc.communicate()
        code = None
    except BaseException:
        _kill(proc)
        raise
    return HookRun(code, out[:MAX_OUTPUT].strip(), err[:MAX_OUTPUT].strip(),
                   round(time.monotonic() - start, 2))


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class ShellHook:
    """Adapts one HookSpec to an event handler."""

    def __init__(self, spec: HookSpec):
        self.spec = spec
        self.name = f"{spec.event} hook: {spec.command.splitlines()[0][:60]}"
        self._matcher = re.compile(spec.matcher) if spec.matcher else None

    def __call__(self, event: Any) -> Verdict | None:
        payload = self._payload(event)
        if payload is None:  # tool not matched
            return None
        agent: Agent = event.agent
        run = run_command(self.spec, payload, agent)
        verdict = self._verdict(run)
        agent.log.record("hook", event=self.spec.event, command=self.spec.command, source=self.spec.source,
                         code=run.code, seconds=run.seconds,
                         decision=verdict.decision if verdict else None,
                         stderr=run.stderr[:500] or None)
        agent.ui.hook_ran(self.name, self._status(run, verdict))
        return verdict

    # --- mapping ---------------------------------------------------------------

    def _payload(self, e: Any) -> dict[str, Any] | None:
        base = {"event": self.spec.event, "session_id": e.agent.log.id,
                "cwd": str(e.agent.ctx.cwd), "permission_mode": e.agent.permissions.mode}
        match e:
            case PreToolUse() | PostToolUse():
                if self._matcher and not self._matcher.fullmatch(e.call.name):
                    return None
                base |= {"tool_name": e.call.name, "tool_input": e.call.input}
                if isinstance(e, PostToolUse):
                    base["tool_output"] = {"content": e.output.content, "is_error": e.output.is_error}
            case PromptSubmit():
                base["prompt"] = e.prompt
            case Stop():
                base |= {"stop_hook_active": bool(e.blocked_by), "last_message": e.final_text}
            case SessionStart():
                base["source"] = e.source
            case Notification():
                base["message"] = e.message
            case SessionEnd():
                pass
        return base

    def _verdict(self, run: HookRun) -> Verdict | None:
        event = self.spec.event
        if run.code == 2:
            reason = run.stderr or f"blocked by {self.name}"
            match event:
                case "PreToolUse" | "UserPromptSubmit":
                    return Verdict("deny", reason)
                case "Stop":
                    return Verdict("block", reason)
                case "PostToolUse":
                    return Verdict(context=f"[{self.name}]\n{reason}")
            return None
        if run.code != 0:
            return None  # non-blocking error: shown to the user, not the model
        data = _json_object(run.stdout)
        if data is not None:
            decision = data.get("decision")
            if event == "Stop" and decision == "deny":
                decision = "block"
            if decision not in ("allow", "deny", "block"):
                decision = None
            return Verdict(decision, str(data.get("reason", "")), str(data.get("additional_context", "")))
        if run.stdout and event in ("UserPromptSubmit", "SessionStart"):
            return Verdict(context=run.stdout)
        return None

    def _status(self, run: HookRun, verdict: Verdict | None) -> str:
        if run.code is None:
            return f"timed out after {self.spec.timeout}s"
        if verdict and verdict.decision in ("deny", "block"):
            return f"{verdict.decision}: {verdict.reason.splitlines()[0] if verdict.reason else ''}"
        if run.code not in (0, 2):
            return f"exit {run.code}: {run.stderr.splitlines()[0] if run.stderr else ''}"
        return f"ok ({run.seconds}s)"


def _json_object(text: str) -> dict[str, Any] | None:
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def install(agent: Agent, specs: list[HookSpec]) -> None:
    """Register each spec's handler after the built-ins, in configuration order."""
    for spec in specs:
        hook = ShellHook(spec)
        agent.hooks.on(EVENTS[spec.event], hook.name, hook)
