"""The agent loop: call the model, run the tools it asks for, repeat."""

from __future__ import annotations

from typing import Any, Protocol

from wren.agent.compact import COMPACT_AT, compacted_history, estimate_tokens, summarize
from wren.agent.permissions import Decision, Permissions
from wren.agent.prompt import build_system_prompt
from wren.agent.session import SessionLog, SessionState
from wren.checkpoint import Checkpoint, CheckpointError, Checkpoints
from wren.config import ModelConfig
from wren.llm.base import Provider
from wren.llm.types import (
    Completed,
    ContentBlock,
    LLMError,
    Message,
    Response,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallStarted,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from wren.tools import Tool, ToolContext, ToolError, ToolOutput, default_tools
from wren.tools.base import validate_args


class AgentUI(Protocol):
    def model_started(self) -> None: ...
    def text_delta(self, text: str) -> None: ...
    def thinking_delta(self, text: str) -> None: ...
    def tool_call_started(self, name: str) -> None: ...
    def model_finished(self) -> None: ...
    def tool_started(self, name: str, label: str) -> None: ...
    def confirm(self, tool: Tool, args: dict[str, Any], label: str, preview: str | None) -> Decision: ...
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
        self.system = build_system_prompt(ctx.cwd)
        self.messages: list[Message] = []
        self.usage = Usage()
        self.cost: float | None = 0.0 if model.price else None
        # Prompt size as of the last response, and how many messages it covered.
        self.context_tokens = 0
        self._billed_upto = 0
        # Tool calls of the current turn that still need a result; used to keep
        # the history valid when the user interrupts mid-turn.
        self._pending: list[ToolUseBlock] = []
        self._results: list[ToolResultBlock] = []
        self.log.record("session_start", model=model.name, cwd=str(ctx.cwd), system=self.system)

    # --- public API --------------------------------------------------------

    def run(self, prompt: str) -> str:
        """Run one user request to completion. Returns the final assistant text."""
        final_text = ""
        try:
            # Compact before adding the prompt so it stays verbatim, not summarized.
            if self._context_full(extra=len(prompt) // 4):
                self.compact()
            self._checkpoint(prompt)
            self._add_user([TextBlock(prompt)])
            for _ in range(self.max_turns):
                if self._context_full():
                    self.compact()
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
                    self._abandon_pending(calls, "Not executed: the response was a refusal.")
                    return final_text
                if not calls:
                    if response.stop_reason == "max_tokens":
                        self.ui.notice("response was cut off at the max_tokens limit")
                    return final_text
                if not self._run_tools(calls, truncated=response.stop_reason == "max_tokens"):
                    return final_text
            self.ui.notice(f"stopped after {self.max_turns} turns")
        except KeyboardInterrupt:
            self._abandon_pending(self._pending, "Interrupted by the user.")
            self.ui.notice("interrupted")
        except LLMError as e:
            self.log.record("error", error=str(e))
            self.ui.error(str(e))
        return final_text

    def estimated_context(self) -> int:
        """Tokens the next request will send: last billed size plus what was added since."""
        return self.context_tokens + estimate_tokens(self.messages[self._billed_upto :])

    def _context_full(self, extra: int = 0) -> bool:
        return self.estimated_context() + extra > self.model.context_window * COMPACT_AT

    def compact(self) -> None:
        """Replace the conversation with a summary of it. Raises LLMError."""
        if not self.messages:
            return
        before = self.estimated_context()
        self.ui.notice(f"compacting conversation (~{before // 1000}k tokens)…")
        self.ui.model_started()
        try:
            summary = summarize(self.provider, self.system, self.messages,
                                [t.spec() for t in self.tools.values()])
        finally:
            self.ui.model_finished()
        self.messages = compacted_history(summary)
        self.ctx.read_files.clear()
        self.context_tokens, self._billed_upto = 0, 0
        if self.checkpoints:
            self.checkpoints.forget_conversation()
        self.log.record("compact", summary=summary, message=self.messages[0].to_dict())
        self.ui.notice(f"compacted to ~{self.estimated_context() // 1000}k tokens")

    def rewind(self, cp: Checkpoint) -> None:
        """Restore files, and the conversation when possible, to before `cp`'s turn.
        Raises CheckpointError."""
        assert self.checkpoints is not None
        index = self.checkpoints.history.index(cp)
        self.checkpoints.restore(cp)
        if cp.message_index is not None:
            self.messages = self.messages[: cp.message_index]
            self.context_tokens, self._billed_upto = 0, 0
        self.ctx.read_files.clear()
        self.log.record("rewind", commit=cp.commit, message_index=cp.message_index,
                        checkpoint_index=index)

    def restore(self, state: SessionState) -> None:
        """Continue a logged session."""
        self.messages = state.messages
        self.usage = state.usage
        if state.cost is not None:
            self.cost = state.cost
        if self.checkpoints and self.checkpoints.enabled:
            self.checkpoints.history = state.checkpoints
        # A session that crashed mid-turn can end with unanswered tool calls.
        if self.messages and self.messages[-1].role == "assistant":
            self._abandon_pending(self.messages[-1].tool_uses(), "Interrupted: the session ended.")
        self.log.record("resume", model=self.model.name)

    def set_model(self, provider: Provider, model: ModelConfig) -> None:
        # Thinking blocks are signed by the model that produced them and other
        # providers may reject them, so drop them when switching models.
        for m in self.messages:
            m.content = [b for b in m.content if not isinstance(b, ThinkingBlock)]
        self.messages = [m for m in self.messages if m.content]
        self.provider, self.model = provider, model
        if model.price and self.cost is None:
            self.cost = 0.0
        self.log.record("model_switch", model=model.name)

    def clear(self) -> None:
        self.messages.clear()
        self.ctx.read_files.clear()
        self.context_tokens, self._billed_upto = 0, 0
        if self.checkpoints:
            self.checkpoints.forget_conversation()
        self.log.record("clear")

    def _checkpoint(self, prompt: str) -> None:
        if not (self.checkpoints and self.checkpoints.enabled):
            return
        try:
            cp = self.checkpoints.snapshot(len(self.messages), prompt)
        except CheckpointError as e:
            self.ui.notice(f"checkpoint skipped: {e}")
            return
        if cp:
            self.log.record("checkpoint", commit=cp.commit, message_index=cp.message_index,
                            prompt=prompt)

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
        keep_going = True
        for call in calls:
            if truncated and call is calls[-1]:
                result = ToolResultBlock(
                    call.id,
                    "Not executed: your response hit the output token limit while writing "
                    "this call, so its arguments are incomplete. Split the work into smaller "
                    "steps (e.g. create a file in parts, or use edit_file).",
                    is_error=True,
                )
            elif not keep_going:
                result = ToolResultBlock(call.id, "Skipped: the user denied an earlier tool call.", True)
            else:
                result, keep_going = self._execute(call)
            self._results.append(result)
            self._pending.remove(call)
        self._add_user(self._results)
        self._results = []
        return keep_going

    def _execute(self, call: ToolUseBlock) -> tuple[ToolResultBlock, bool]:
        tool = self.tools.get(call.name)
        if tool is None:
            return self._error(call, f"unknown tool {call.name!r}; available: {', '.join(self.tools)}"), True
        problem = validate_args(tool.input_schema, call.input)
        if problem:
            return self._error(call, f"invalid arguments for {call.name}: {problem}"), True

        label = tool.describe(call.input, self.ctx)
        self.ui.tool_started(call.name, label)

        if self.permissions.needs_approval(tool, call.input):
            decision = self.ui.confirm(tool, call.input, label, tool.preview(call.input, self.ctx))
            if not decision.allow:
                text = "The user denied this tool call."
                if decision.feedback:
                    text += f" Their feedback: {decision.feedback}"
                output = ToolOutput(text, is_error=True, summary="denied")
                self.ui.tool_finished(call.name, output)
                self.log.record("tool", name=call.name, input=call.input, denied=True)
                # Without feedback there is nothing for the model to act on; hand
                # control back to the user.
                return ToolResultBlock(call.id, text, True), bool(decision.feedback)
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
        self.ui.tool_finished(call.name, output)
        self.log.record("tool", name=call.name, input=call.input, is_error=output.is_error)
        return ToolResultBlock(call.id, output.content or "(no output)", output.is_error), True

    def _error(self, call: ToolUseBlock, text: str) -> ToolResultBlock:
        self.ui.tool_finished(call.name, ToolOutput(text, is_error=True, summary=text))
        self.log.record("tool", name=call.name, input=call.input, is_error=True)
        return ToolResultBlock(call.id, text, is_error=True)

    def _abandon_pending(self, calls: list[ToolUseBlock], reason: str) -> None:
        """Answer every unanswered tool call so the history stays valid."""
        results = self._results + [ToolResultBlock(c.id, reason, is_error=True) for c in calls]
        if results:
            self._add_user(results)
        self._pending, self._results = [], []

    # --- history -----------------------------------------------------------

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        self.log.record("message", **message.to_dict())

    def _add_user(self, blocks: list[ContentBlock]) -> None:
        # Consecutive user content (e.g. tool results followed by a new prompt
        # after an interrupt) is merged into one message.
        if self.messages and self.messages[-1].role == "user":
            self.messages[-1].content.extend(blocks)
            self.log.record("message", **Message("user", blocks).to_dict())
        else:
            self._append(Message("user", list(blocks)))
