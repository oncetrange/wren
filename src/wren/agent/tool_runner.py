"""Running the tool calls of one model response: checks, hooks, approval, the
calls themselves (read-only subagents side by side), plan approval, and keeping
the history valid when a turn is cut short."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

from wren.agent.events import PostToolUse, PreToolUse, allowed, denial
from wren.agent.permissions import Decision
from wren.agent.plans import save_plan
from wren.agent.subagents import Progress
from wren.llm.types import INVALID_JSON_KEY, ContentBlock, TextBlock, ToolResultBlock, ToolUseBlock
from wren.tools import Tool, ToolError, ToolOutput
from wren.tools.base import validate_args

if TYPE_CHECKING:
    from wren.agent.loop import Agent

# Most tool calls (read-only subagents) run at the same time.
MAX_CONCURRENT = 4


class ToolRunner:
    def __init__(self, agent: Agent):
        self.agent = agent
        # Calls of the current turn that still need a result, and the results so
        # far: used to keep the history valid when the user interrupts mid-turn.
        self._pending: list[ToolUseBlock] = []
        self._results: list[ToolResultBlock] = []

    def run(self, calls: list[ToolUseBlock], truncated: bool) -> bool:
        """Execute a turn's tool calls. Returns False if the loop should stop."""
        self._pending, self._results = list(calls), []
        denial: Decision | None = None
        i = 0
        while i < len(calls):
            call = calls[i]
            group = self._concurrent_group(calls[i:len(calls) - truncated]) if denial is None else []
            if len(group) > 1:
                self._results += self._execute_concurrently(group)
                for c in group:
                    self._pending.remove(c)
                i += len(group)
                continue
            if truncated and call is calls[-1]:
                result = ToolResultBlock(
                    call.id,
                    "Not executed: your response hit the output token limit while writing "
                    "this call, so its arguments are incomplete. Split the work into smaller "
                    "steps (e.g. create a file in parts, or use edit_file).",
                    is_error=True,
                )
            elif denial is not None:
                result = ToolResultBlock(
                    call.id, "Not run: the user rejected an earlier tool call in this turn.", True)
            else:
                self.agent.check_cancel()
                result, denial = self._execute(call)
            self._results.append(result)
            self._pending.remove(call)
            i += 1
        blocks: list[ContentBlock] = list(self._results)
        if denial is not None and denial.feedback:
            # What the user typed is a message from them, not tool output: it
            # goes in as their own text after the tool results.
            blocks.append(TextBlock(denial.feedback))
        steered = self.agent.take_steering()
        self.agent.add_user(blocks + steered)
        self._results = []
        # A bare "no" hands control back to the user; with a message, the model responds to it.
        return denial is None or bool(denial.feedback) or bool(steered)

    def _execute(self, call: ToolUseBlock) -> tuple[ToolResultBlock, Decision | None]:
        """Run one call. Returns its result, and the user's decision if they rejected it."""
        tool = self._check(call)
        if isinstance(tool, ToolResultBlock):
            return tool, None

        label = tool.describe(call.input, self.agent.ctx)
        if self.agent.permissions.leaves_sandbox(tool, call.input):
            label = f"{label} (outside the sandbox)"
        self.agent.ui.tool_started(call.name, label)
        if call.name == "exit_plan_mode":
            return self._exit_plan_mode(call)

        verdicts = self.agent.hooks.run("pre_tool", PreToolUse(self.agent, tool, call))
        if blocked := denial(verdicts):
            return self._error(call, blocked.reason), None

        must_ask = tool.always_confirm(call.input) or self.agent.permissions.must_ask(tool, call.input)
        if must_ask or (not allowed(verdicts) and self.agent.permissions.needs_approval(tool, call.input)):
            self.agent.notify(f"wren needs your approval: {call.name} {label}".strip())
            decision = self.agent.ui.confirm(tool, call.input, label, tool.preview(call.input, self.agent.ctx),
                                       key=self.agent.permissions.key(tool, call.input))
            if not decision.allow:
                text = "The user rejected this tool call, so it was not run."
                if decision.feedback:
                    text += " Their message follows."
                self.agent.ui.tool_finished(call.name, ToolOutput(text, is_error=True, summary="denied"))
                self.agent.log.record("tool", name=call.name, input=call.input, denied=True)
                return ToolResultBlock(call.id, text, True), decision
            if (decision.remember or decision.persist) and not must_ask:
                self.agent.permissions.remember(tool, call.input, persist=decision.persist)
                if decision.persist and self.agent.permissions.project_file is not None:
                    self.agent.ui.notice(f"always allowed in this project: {self.agent.permissions.key(tool, call.input)} "
                                   f"(saved in {self.agent.permissions.project_file})")

        started = time.monotonic()
        output = self._run_tool(tool, call)
        if self.agent.permissions.runs_sandboxed(tool, call.input) and not self.agent.read_only:
            self._ask_about_hosts(started, output)
        return self._finish(call, tool, output), None

    def _ask_about_hosts(self, since: float, output: ToolOutput) -> None:
        """Ask the user about the hosts the sandbox refused during the command, if it
        asks, and tell the model what they said."""
        sandbox, permissions = self.agent.ctx.sandbox, self.agent.permissions
        if sandbox is None or not sandbox.asks:
            return
        for host in sandbox.refused_since(since):
            if permissions.host_decision(host) is not None:  # a deny rule, or answered already
                continue
            self.agent.notify(f"wren needs your approval: connect to {host}")
            decision = self.agent.ui.allow_host(host)
            if decision is None:
                continue
            permissions.answer_host(host, decision.allow, persist=decision.persist)
            self.agent.log.record("host", host=host, allowed=decision.allow, persist=decision.persist)
            if decision.allow:
                note = f"The user allowed {host} in the sandbox: run the command again if it needs it."
                if decision.persist and permissions.project_file is not None:
                    self.agent.ui.notice(f"always allowed in this project: net:{host} "
                                         f"(saved in {permissions.project_file})")
            else:
                note = f"The user didn't allow connecting to {host}; don't try to reach it another way."
                if decision.feedback:
                    note += f" They said: {decision.feedback}"
            output.content = f"{output.content}\n\n[{note}]"

    def _check(self, call: ToolUseBlock) -> Tool | ToolResultBlock:
        """The call's tool, or an error result if there is no such tool or the arguments are wrong."""
        tool = self.agent.tools.get(call.name)
        if tool is None:
            return self._error(call, f"unknown tool {call.name!r}; available: {', '.join(self.agent.tools)}")
        if INVALID_JSON_KEY in call.input:
            raw = str(call.input[INVALID_JSON_KEY])
            return self._error(call, f"invalid arguments for {call.name}: not a JSON object "
                                     f"({len(raw)} characters, starting {raw[:80]!r}). "
                                     "Call it again with valid JSON arguments.")
        problem = validate_args(tool.input_schema, call.input) if tool.strict_args else (
            None if isinstance(call.input, dict) else "arguments must be a JSON object")
        if problem:
            return self._error(call, f"invalid arguments for {call.name}: {problem}")
        return tool

    def _run_tool(self, tool: Tool, call: ToolUseBlock) -> ToolOutput:
        self.agent.ctx.read_only_shell = self.agent.read_only or self.agent.permissions.mode == "plan"
        self.agent.ctx.can_leave_sandbox = not self.agent.read_only
        try:
            return tool.run(call.input, self.agent.ctx)
        except ToolError as e:
            return ToolOutput(str(e), is_error=True, summary=str(e).splitlines()[0])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # a bug in a tool must not kill the session
            return ToolOutput(f"internal error in {call.name}: {type(e).__name__}: {e}", True,
                              summary=f"{type(e).__name__}: {e}")

    def _finish(self, call: ToolUseBlock, tool: Tool, output: ToolOutput) -> ToolResultBlock:
        if output.todos is not None:
            self.agent.conv.todos = output.todos
            self.agent.log.record("todos", items=[t.to_dict() for t in output.todos])
        if not tool.read_only and not output.is_error:
            self.agent.changed = True
        for v in self.agent.hooks.run("post_tool", PostToolUse(self.agent, tool, call, output)):
            if v.context:
                output.content = f"{output.content}\n\n{v.context}"
        self.agent.ui.tool_finished(call.name, output)
        self.agent.tool_calls += 1
        self.agent.tool_errors += output.is_error
        self.agent.log.record("tool", name=call.name, input=call.input, is_error=output.is_error,
                        error=output.content[:500] if output.is_error else None)
        return ToolResultBlock(call.id, output.content or "(no output)", output.is_error)

    # --- concurrent calls ----------------------------------------------------

    def _concurrent_group(self, calls: list[ToolUseBlock]) -> list[ToolUseBlock]:
        """The leading calls that may run at the same time (read-only subagents):
        they never ask for approval and change nothing, so order doesn't matter."""
        group = []
        for call in calls:
            tool = self.agent.tools.get(call.name)
            if (tool is None or validate_args(tool.input_schema, call.input)
                    or not tool.concurrent_safe(call.input)):
                break
            group.append(call)
        return group

    def _execute_concurrently(self, calls: list[ToolUseBlock]) -> list[ToolResultBlock]:
        """Run the calls on threads, showing their progress; print each call
        with its result once all are done, in the order they were made."""
        results: dict[str, ToolResultBlock] = {}
        runnable: list[tuple[ToolUseBlock, Tool]] = []
        for call in calls:
            tool = self.agent.tools[call.name]
            if blocked := denial(self.agent.hooks.run("pre_tool", PreToolUse(self.agent, tool, call))):
                self.agent.ui.tool_started(call.name, tool.describe(call.input, self.agent.ctx))
                results[call.id] = self._error(call, blocked.reason)
            else:
                runnable.append((call, tool))
        progress = Progress(self.agent.ui)
        self.agent.progress = progress
        try:
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
                futures = {call.id: pool.submit(self._run_tool, tool, call) for call, tool in runnable}
                try:
                    wait(futures.values())
                except KeyboardInterrupt:
                    progress.cancel.set()  # the workers stop at their next step
                    raise
        finally:
            self.agent.progress = None
            progress.close()
        for call, tool in runnable:
            self.agent.ui.tool_started(call.name, tool.describe(call.input, self.agent.ctx))
            error = futures[call.id].exception()
            if isinstance(error, KeyboardInterrupt):
                raise error
            results[call.id] = self._finish(call, tool, futures[call.id].result())
        return [results[c.id] for c in calls]

    def _exit_plan_mode(self, call: ToolUseBlock) -> tuple[ToolResultBlock, Decision | None]:
        """Show the plan for approval. Returns a rejection Decision when the
        turn should stop (no approval) or continue with the user's feedback."""
        if self.agent.permissions.mode != "plan":
            return self._error(call, "Plan mode is not on: no approval step is needed, go ahead."), None
        plan = call.input["plan"]
        self.agent.notify("wren has a plan for you to review")
        decision = self.agent.ui.review_plan(plan)
        self.agent.tool_calls += 1

        if decision is None or decision.approved:
            self.agent.plan_text, self.agent.plan_file = plan, save_plan(self.agent.ctx.cwd, plan)
            where = self.agent.ctx.display_path(self.agent.plan_file)
        if decision is None:  # nobody to ask (headless): the plan is the result
            text = f"Plan saved to {where}. Stopping here: it needs the user's approval first."
            summary, outcome = f"saved to {where}", Decision(allow=False)
        elif decision.approved:
            self.agent.permissions.mode = self.agent.told_mode = decision.mode
            text = (f"The user approved the plan (saved to {where}). Plan mode is off: you can "
                    "modify files now. Track the plan's steps with todo_write, then carry it out.")
            summary, outcome = f"approved · {decision.mode.replace('_', ' ')}", None
        else:
            text = "The user did not approve the plan. " + (
                "Their message follows: revise the plan and present it again with exit_plan_mode."
                if decision.feedback else "Wait for their instructions.")
            summary, outcome = "changes requested" if decision.feedback else "not approved", \
                Decision(allow=False, feedback=decision.feedback)
        self.agent.log.record("plan", plan=plan, approved=bool(decision and decision.approved),
                        file=str(self.agent.plan_file) if decision is None or decision.approved else None)
        self.agent.ui.tool_finished(call.name, ToolOutput(text, summary=summary))
        return ToolResultBlock(call.id, text), outcome

    def _error(self, call: ToolUseBlock, text: str) -> ToolResultBlock:
        self.agent.ui.tool_finished(call.name, ToolOutput(text, is_error=True, summary=text))
        self.agent.tool_calls += 1
        self.agent.tool_errors += 1
        self.agent.log.record("tool", name=call.name, input=call.input, is_error=True, error=text[:500])
        return ToolResultBlock(call.id, text, is_error=True)

    def abandon_pending(self, reason: str) -> None:
        """The turn was cut short: answer the calls that didn't get to run."""
        self.abandon(self._pending, reason)

    def abandon(self, calls: list[ToolUseBlock], reason: str) -> None:
        """Answer every unanswered tool call so the history stays valid."""
        results: list[ContentBlock] = [*self._results,
                                       *(ToolResultBlock(c.id, reason, is_error=True) for c in calls)]
        if results:
            self.agent.add_user(results)
        self._pending, self._results = [], []
