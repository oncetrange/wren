"""The agent loop: call the model, run the tools it asks for, repeat."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from wren.agent.compact import (
    MASK_MIN_CHARS,
    append_archive,
    estimate_tokens,
    plan_compaction,
    plan_mask,
    summarize,
    summary_note,
)
from wren.agent.conversation import Conversation, RestorePoint
from wren.agent.permissions import Decision, Permissions
from wren.agent.plans import (
    FINAL_CHECK,
    PLAN_MODE_OFF,
    PLAN_MODE_ON,
    TURN_BUDGET,
    UNFINISHED_TODOS,
    PlanDecision,
    reminder,
    save_plan,
)
from wren.agent.prompt import build_system_prompt
from wren.agent.session import SessionLog, SessionState
from wren.agent.todos import format_todos
from wren.checkpoint import CheckpointError, Checkpoints
from wren.config import CONFIG_DIR, ModelConfig
from wren.llm.base import Provider
from wren.llm.types import (
    Completed,
    ContentBlock,
    LLMError,
    Message,
    Response,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolCallStarted,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from wren.tools import Tool, ToolContext, ToolError, ToolOutput, default_tools
from wren.tools.base import validate_args


TRANSCRIPTS_DIR = CONFIG_DIR / "transcripts"


class AgentUI(Protocol):
    def model_started(self) -> None: ...
    def text_delta(self, text: str) -> None: ...
    def thinking_delta(self, text: str) -> None: ...
    def tool_call_started(self, name: str) -> None: ...
    def model_finished(self) -> None: ...
    def tool_started(self, name: str, label: str) -> None: ...
    def confirm(self, tool: Tool, args: dict[str, Any], label: str, preview: str | None) -> Decision: ...
    def review_plan(self, plan: str) -> PlanDecision | None: ...
    def tool_finished(self, name: str, output: ToolOutput) -> None: ...
    def notice(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...


class Agent:
    def __init__(
        self,
        provider: Provider,
        model: ModelConfig,
        ctx: ToolContext,
        ui: AgentUI,
        permissions: Permissions | None = None,
        log: SessionLog | None = None,
        tools: list[Tool] | None = None,
        checkpoints: Checkpoints | None = None,
        max_turns: int = 100,
        final_check: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.ctx = ctx
        self.ui = ui
        self.permissions = permissions or Permissions()
        self.log = log or SessionLog(directory=None)
        self.tools = {t.name: t for t in (tools if tools is not None else default_tools())}
        self.checkpoints = checkpoints
        self.max_turns = max_turns
        # Before finishing a request that changed things, ask the model once to
        # check the request's explicit instructions (on by default when headless).
        self.final_check = final_check
        self.system = build_system_prompt(ctx.cwd)
        self.conv = Conversation()
        self.usage = Usage()
        self.cost: float | None = 0.0 if model.price else None
        # Prompt size as of the last response, and how many messages it covered.
        self.context_tokens = 0
        self._billed_upto = 0
        # How the last run() ended: done | max_turns | error | interrupted | refusal
        self.status = "done"
        self.turns = self.tool_calls = self.tool_errors = 0
        # Plan mode: the mode the model was last told about, and the last saved plan.
        self._told_mode = "ask"
        self.plan_text: str | None = None
        self.plan_file: Path | None = None
        # Tool calls of the current turn that still need a result; used to keep
        # the history valid when the user interrupts mid-turn.
        self._pending: list[ToolUseBlock] = []
        self._results: list[ToolResultBlock] = []
        self.log.record("session_start", model=model.name, cwd=str(ctx.cwd), system=self.system)

    @property
    def messages(self) -> list[Message]:
        return self.conv.messages

    # --- public API --------------------------------------------------------

    def run(self, prompt: str) -> str:
        """Run one user request to completion. Returns the final assistant text."""
        final_text = ""
        self.status = "done"
        nudged = checked = warned = False
        self._changed = False  # set once a tool that can modify files ran successfully
        try:
            # Manage context before adding the prompt so it stays verbatim.
            self._manage_context(extra=len(prompt) // 4)
            self._start_turn(prompt)
            self._add_user([TextBlock(prompt), *self._mode_reminder()])
            for turn in range(self.max_turns):
                self._manage_context()
                response = self._call_model()
                if not response.message.content:
                    self.ui.notice("model returned an empty response")
                    return final_text
                self._append(response.message)
                self._billed_upto = len(self.messages)
                final_text = response.message.text() or final_text

                calls = response.message.tool_uses()
                if response.stop_reason == "refusal":
                    self.ui.notice("the model declined to continue")
                    self.status = "refusal"
                    self._abandon_pending(calls, "Not executed: the response was a refusal.")
                    return final_text
                if not calls:
                    if response.stop_reason == "max_tokens":
                        self.ui.notice("response was cut off at the max_tokens limit")
                    else:
                        # Each at most once per request, sent together in one message.
                        parts = []
                        if not nudged and (open_items := self._unfinished_todos()):
                            nudged = True
                            parts.append(UNFINISHED_TODOS.format(items=open_items))
                        if self.final_check and self._changed and not checked:
                            checked = True
                            parts.append(FINAL_CHECK)
                        if parts:
                            self._add_user([TextBlock(reminder(*parts))])
                            continue
                    return final_text
                if not self._run_tools(calls, truncated=response.stop_reason == "max_tokens"):
                    return final_text
                left = self.max_turns - turn - 1
                if self.final_check and not warned and 0 < left <= max(3, self.max_turns // 10):
                    warned = True
                    self._add_user([TextBlock(reminder(TURN_BUDGET.format(left=left)))])
            self.ui.notice(f"stopped after {self.max_turns} turns")
            self.status = "max_turns"
        except KeyboardInterrupt:
            self._abandon_pending(self._pending, "Interrupted by the user.")
            self.ui.notice("interrupted")
            self.status = "interrupted"
        except LLMError as e:
            self.log.record("error", error=str(e))
            self.ui.error(str(e))
            self.status = "error"
        return final_text

    def estimated_context(self) -> int:
        """Tokens the next request will send: last billed size plus what was added since."""
        return self.context_tokens + estimate_tokens(self.messages[self._billed_upto :])

    def _unfinished_todos(self) -> str:
        if self.permissions.mode == "plan":
            return ""  # planning ends with a plan, not with finished tasks
        return format_todos([t for t in self.conv.todos if t.status != "completed"])

    def _mode_reminder(self) -> list[TextBlock]:
        """Tell the model about plan mode: on every prompt while it's on, and once when it ends."""
        mode, told = self.permissions.mode, self._told_mode
        self._told_mode = mode
        if mode == "plan":
            return [TextBlock(PLAN_MODE_ON)]
        if told == "plan":
            return [TextBlock(PLAN_MODE_OFF)]
        return []

    def _manage_context(self, extra: int = 0) -> None:
        """L1 masking, then L2 compaction if the prompt is still too large."""
        if self.model.mask_at and self.estimated_context() + extra > self.model.mask_at:
            self.mask()
        if self.estimated_context() + extra > self.model.compact_threshold:
            self.compact()

    def mask(self) -> None:
        """Mask old tool traffic, down well below the threshold in one batch."""
        keep, freed = plan_mask(self.messages, self.estimated_context(), self.model.mask_at)
        if freed == 0:
            return
        self.conv.mask(keep, MASK_MIN_CHARS)
        self.log.record("mask", keep_turns=keep, min_chars=MASK_MIN_CHARS, freed_chars=freed)
        self._reestimate()
        self.ui.notice(f"cleared old tool outputs (~{freed // 4000}k tokens); "
                       f"context now ~{self.estimated_context() // 1000}k")

    def compact(self) -> None:
        """Summarize all but the most recent turns. Raises LLMError."""
        if not any(m.role == "assistant" for m in self.messages):
            return  # nothing new since the last compaction

        kept_from = plan_compaction(self.messages, self._fixed_tokens(), self.model.compact_threshold)
        span = self.messages[:kept_from]
        before = self.estimated_context()
        self.ui.notice(f"compacting conversation (~{before // 1000}k tokens)…")
        self.ui.model_started()
        try:
            summary = summarize(self.provider, self.system, span,
                                [t.spec() for t in self.tools.values()])
        finally:
            self.ui.model_finished()
        archive = self._archive(span)
        note = summary_note(summary, archive, format_todos(self.conv.todos))
        self.conv.compacted(summary, note, kept_from)
        self.ctx.read_files.clear()
        self._reestimate()
        self.log.record("compact", summary=summary, message=note.to_dict(), kept_from=kept_from,
                        archive=str(archive) if archive else None)
        self.ui.notice(f"compacted to ~{self.estimated_context() // 1000}k tokens "
                       "(undo with /rewind)")

    def _archive(self, span: list[Message]) -> Path | None:
        if self.log.path is None:
            return None
        path = TRANSCRIPTS_DIR / f"{self.log.id}.md"
        n = sum(1 for p in self.conv.timeline if p.kind == "compaction") + 1
        try:
            append_archive(path, span, f"Compaction {n} ({datetime.now():%Y-%m-%d %H:%M})")
        except OSError as e:
            self.ui.notice(f"could not write the transcript archive: {e}")
            return None
        return path

    def _fixed_tokens(self) -> int:
        """Estimated size of the system prompt and tool definitions."""
        return (len(self.system) + len(json.dumps([vars(t.spec()) for t in self.tools.values()]))) // 4

    def _reestimate(self) -> None:
        """Estimate the full prompt size after history changed underneath the last bill."""
        self.context_tokens = self._fixed_tokens() + estimate_tokens(self.messages)
        self._billed_upto = len(self.messages)

    def changed_files(self, point: RestorePoint) -> list[str]:
        """Files a rewind to `point` would restore. Raises CheckpointError."""
        if point.commit is None or not (self.checkpoints and self.checkpoints.enabled):
            return []
        return self.checkpoints.changed_files(point.commit)

    def rewind(self, index: int) -> RestorePoint:
        """Go back to `self.conv.timeline[index]`: files and conversation as
        before that turn, or the history as before that compaction.
        Raises CheckpointError (and changes nothing) if files can't be restored."""
        point = self.conv.timeline[index]
        if point.commit is not None and self.checkpoints and self.checkpoints.enabled:
            self.checkpoints.restore(point.commit)
        self.conv.rewind(index)
        self.ctx.read_files.clear()
        self.context_tokens, self._billed_upto = 0, 0
        self.log.record("rewind", index=index)
        return point

    def restore(self, state: SessionState) -> None:
        """Continue a logged session (its log must be `self.log`)."""
        self.conv = state.conversation
        self.usage = state.usage
        self.cost = state.cost if state.cost is not None else (0.0 if self.model.price else None)
        self.context_tokens, self._billed_upto = 0, 0
        self.ctx.read_files.clear()
        # A session that crashed mid-turn can end with unanswered tool calls.
        if self.messages and self.messages[-1].role == "assistant":
            self._abandon_pending(self.messages[-1].tool_uses(), "Interrupted: the session ended.")
        self.log.record("resume", model=self.model.name)

    def new_session(self, log: SessionLog) -> None:
        """Start over with an empty conversation in a new log."""
        self.log = log
        self.conv = Conversation()
        self.usage = Usage()
        self.cost = 0.0 if self.model.price else None
        self.context_tokens, self._billed_upto = 0, 0
        self.ctx.read_files.clear()
        self.log.record("session_start", model=self.model.name, cwd=str(self.ctx.cwd),
                        system=self.system)

    def set_model(self, provider: Provider, model: ModelConfig) -> None:
        # Thinking blocks are signed by the model that produced them and other
        # providers may reject them, so drop them when switching models.
        self.conv.strip_thinking()
        self.provider, self.model = provider, model
        if model.price and self.cost is None:
            self.cost = 0.0
        self.log.record("model_switch", model=model.name)

    def _start_turn(self, prompt: str) -> None:
        commit = None
        if self.checkpoints and self.checkpoints.enabled:
            try:
                commit = self.checkpoints.snapshot(f"before: {prompt}")
            except CheckpointError as e:
                self.ui.notice(f"checkpoint skipped, files can't be restored to this point: {e}")
        self.conv.start_turn(prompt, commit)
        self.log.record("checkpoint", commit=commit, prompt=prompt)

    # --- model -------------------------------------------------------------

    def _call_model(self) -> Response:
        self.ui.model_started()
        response: Response | None = None
        try:
            for event in self.provider.stream(
                system=self.system,
                messages=self.messages,
                tools=[t.spec() for t in self.tools.values()],
            ):
                match event:
                    case TextDelta(text):
                        self.ui.text_delta(text)
                    case ThinkingDelta(text):
                        self.ui.thinking_delta(text)
                    case ToolCallStarted(name):
                        self.ui.tool_call_started(name)
                    case Completed(r):
                        response = r
        finally:
            self.ui.model_finished()
        if response is None:
            raise LLMError("stream ended without a final message")

        self.usage += response.usage
        self.turns += 1
        self.context_tokens = response.usage.context_tokens + response.usage.output_tokens
        turn_cost = self.model.cost(response.usage)
        if turn_cost is not None and self.cost is not None:
            self.cost += turn_cost
        self.log.record(
            "usage",
            model=self.model.name,
            stop_reason=response.stop_reason,
            usage=vars(response.usage),
            cost=turn_cost,
        )
        return response

    # --- tools -------------------------------------------------------------

    def _run_tools(self, calls: list[ToolUseBlock], truncated: bool) -> bool:
        """Execute a turn's tool calls. Returns False if the loop should stop."""
        self._pending, self._results = list(calls), []
        denial: Decision | None = None
        for call in calls:
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
                result, denial = self._execute(call)
            self._results.append(result)
            self._pending.remove(call)
        blocks: list[ContentBlock] = list(self._results)
        if denial is not None and denial.feedback:
            # What the user typed is a message from them, not tool output: it
            # goes in as their own text after the tool results.
            blocks.append(TextBlock(denial.feedback))
        self._add_user(blocks)
        self._results = []
        # A bare "no" hands control back to the user; with a message, the model responds to it.
        return denial is None or bool(denial.feedback)

    def _execute(self, call: ToolUseBlock) -> tuple[ToolResultBlock, Decision | None]:
        """Run one call. Returns its result, and the user's decision if they rejected it."""
        tool = self.tools.get(call.name)
        if tool is None:
            return self._error(call, f"unknown tool {call.name!r}; available: {', '.join(self.tools)}"), None
        problem = validate_args(tool.input_schema, call.input)
        if problem:
            return self._error(call, f"invalid arguments for {call.name}: {problem}"), None

        label = tool.describe(call.input, self.ctx)
        self.ui.tool_started(call.name, label)
        if call.name == "exit_plan_mode":
            return self._exit_plan_mode(call)

        reason = self.permissions.blocked(tool, call.input)
        if reason:
            return self._error(call, reason), None

        if self.permissions.needs_approval(tool, call.input):
            decision = self.ui.confirm(tool, call.input, label, tool.preview(call.input, self.ctx))
            if not decision.allow:
                text = "The user rejected this tool call, so it was not run."
                if decision.feedback:
                    text += " Their message follows."
                self.ui.tool_finished(call.name, ToolOutput(text, is_error=True, summary="denied"))
                self.log.record("tool", name=call.name, input=call.input, denied=True)
                return ToolResultBlock(call.id, text, True), decision
            if decision.remember:
                self.permissions.remember(tool, call.input)

        try:
            output = tool.run(call.input, self.ctx)
        except ToolError as e:
            output = ToolOutput(str(e), is_error=True, summary=str(e).splitlines()[0])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # a bug in a tool must not kill the session
            output = ToolOutput(f"internal error in {call.name}: {type(e).__name__}: {e}", True,
                                summary=f"{type(e).__name__}: {e}")
        if output.todos is not None:
            self.conv.todos = output.todos
            self.log.record("todos", items=[t.to_dict() for t in output.todos])
        self.ui.tool_finished(call.name, output)
        self.tool_calls += 1
        self.tool_errors += output.is_error
        if not tool.read_only and not output.is_error:
            self._changed = True
        self.log.record("tool", name=call.name, input=call.input, is_error=output.is_error,
                        error=output.content[:500] if output.is_error else None)
        return ToolResultBlock(call.id, output.content or "(no output)", output.is_error), None

    def _exit_plan_mode(self, call: ToolUseBlock) -> tuple[ToolResultBlock, Decision | None]:
        """Show the plan for approval. Returns a rejection Decision when the
        turn should stop (no approval) or continue with the user's feedback."""
        if self.permissions.mode != "plan":
            return self._error(call, "Plan mode is not on: no approval step is needed, go ahead."), None
        plan = call.input["plan"]
        decision = self.ui.review_plan(plan)
        self.tool_calls += 1

        if decision is None or decision.approved:
            self.plan_text, self.plan_file = plan, save_plan(self.ctx.cwd, plan)
            where = self.ctx.display_path(self.plan_file)
        if decision is None:  # nobody to ask (headless): the plan is the result
            text = f"Plan saved to {where}. Stopping here: it needs the user's approval first."
            summary, outcome = f"saved to {where}", Decision(allow=False)
        elif decision.approved:
            self.permissions.mode = self._told_mode = decision.mode
            text = (f"The user approved the plan (saved to {where}). Plan mode is off: you can "
                    "modify files now. Track the plan's steps with todo_write, then carry it out.")
            summary, outcome = f"approved · {decision.mode.replace('_', ' ')}", None
        else:
            text = "The user did not approve the plan. " + (
                "Their message follows: revise the plan and present it again with exit_plan_mode."
                if decision.feedback else "Wait for their instructions.")
            summary, outcome = "changes requested" if decision.feedback else "not approved", \
                Decision(allow=False, feedback=decision.feedback)
        self.log.record("plan", plan=plan, approved=bool(decision and decision.approved),
                        file=str(self.plan_file) if decision is None or decision.approved else None)
        self.ui.tool_finished(call.name, ToolOutput(text, summary=summary))
        return ToolResultBlock(call.id, text), outcome

    def _error(self, call: ToolUseBlock, text: str) -> ToolResultBlock:
        self.ui.tool_finished(call.name, ToolOutput(text, is_error=True, summary=text))
        self.tool_calls += 1
        self.tool_errors += 1
        self.log.record("tool", name=call.name, input=call.input, is_error=True, error=text[:500])
        return ToolResultBlock(call.id, text, is_error=True)

    def _abandon_pending(self, calls: list[ToolUseBlock], reason: str) -> None:
        """Answer every unanswered tool call so the history stays valid."""
        results = self._results + [ToolResultBlock(c.id, reason, is_error=True) for c in calls]
        if results:
            self._add_user(results)
        self._pending, self._results = [], []

    # --- history -----------------------------------------------------------

    def _append(self, message: Message) -> None:
        self.conv.append(message)
        self.log.record("message", **message.to_dict())

    def _add_user(self, blocks: list[ContentBlock]) -> None:
        self.conv.add_user(blocks)
        self.log.record("message", **Message("user", list(blocks)).to_dict())
