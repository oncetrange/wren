"""The agent loop: call the model, run the tools it asks for, repeat."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from wren.agent.builtin_hooks import register_builtins
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
from wren.agent.events import (
    Hooks,
    Notification,
    PromptSubmit,
    SessionEnd,
    SessionStart,
    Stop,
    denial,
)
from wren.agent.permissions import Permissions
from wren.agent.plans import (
    CUT_OFF,
    STEERED,
    TIME_CHECK,
    TIME_UP,
    TURN_BUDGET,
    is_reminder,
    reminder,
)
from wren.agent.prompt import build_system_prompt
from wren.agent.remember import extract_memories
from wren.agent.session import SessionLog, SessionState
from wren.agent.subagents import AgentType, Progress, TaskTool
from wren.agent.todos import format_todos
from wren.agent.tool_runner import ToolRunner
from wren.agent.ui import AgentUI
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
    ThinkingBlock,
    ThinkingDelta,
    ToolCallStarted,
    Usage,
)
from wren.mcp_servers import McpServers
from wren.memory import Memories
from wren.memory import prompt_section as memory_section
from wren.sandbox import Sandbox
from wren.skills import Skill, prompt_section
from wren.tools import Tool, ToolContext, default_tools
from wren.tools.mcp import mcp_tools
from wren.tools.memory import MemoryTool
from wren.tools.shell import Bash, SandboxedBash
from wren.tools.skill import SkillTool

TRANSCRIPTS_DIR = CONFIG_DIR / "transcripts"
# How often stop handlers may keep the model going within one request.
MAX_STOP_BLOCKS = 3
# Responses in a row cut off at max_tokens without acting, before the run gives up.
MAX_CUT_OFFS = 3
# Seconds to wait before each retry of a model request that failed in a way
# that may pass; on top of the SDK's own retries, which stop once a stream has begun.
MODEL_RETRY_DELAYS = (5, 15, 45, 90)

# How a run ended (`status` in the headless JSON result): finished normally, or why not.
RunStatus = Literal[
    "done",            # the model finished
    "blocked",         # a UserPromptSubmit hook refused the prompt
    "empty_response",  # the model returned nothing
    "refusal",         # the model declined to continue
    "max_tokens",      # responses kept being cut off at the output limit before acting
    "max_turns",       # --max-turns used up
    "time_limit",      # --time-limit reached
    "interrupted",     # Ctrl-C
    "error",           # the model request failed for good
]


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
        skills: dict[str, Skill] | None = None,
        agent_types: dict[str, AgentType] | None = None,
        system: str | None = None,
        hooks: Hooks | None = None,
        memory: Memories | None = None,
        mcp: McpServers | None = None,
        extra_tools: list[Tool] | None = None,
        auto_memory: bool = False,
        time_limit: float | None = None,
        resolve_model: Callable[[str], tuple[Provider, ModelConfig]] | None = None,
    ):
        """`tools` replaces the default tools; `extra_tools` adds to them (e.g. web
        search). `time_limit` is in minutes (see set_time_limit); `auto_memory`
        reviews the conversation for memories before compacting; `resolve_model`
        looks up other configured models by name, for subagents that name one."""
        self.provider = provider
        self.model = model
        self.ctx = ctx
        self.ui = ui
        self.permissions = permissions or Permissions()
        # Messages the user sent while the agent worked (see steer).
        self._inbox: deque[str] = deque()
        self.log = log or SessionLog(directory=None)
        self.tools = {t.name: t for t in (tools if tools is not None else default_tools(sandboxed=ctx.sandbox is not None))}
        # Skills: listed in the system prompt, loaded by the model with the skill tool.
        self.skills = skills or {}
        if any(s.model_invocable for s in self.skills.values()):
            self.tools["skill"] = SkillTool(self.skills)
        # Long-term memory: the model keeps it with the memory tool; the indexes
        # go into the system prompt.
        self.memory = memory if memory is not None and memory.enabled else None
        if self.memory is not None:
            self.tools["memory"] = MemoryTool(self.memory)
        # Save memories before context is lost (compaction, session end).
        self.auto_memory = auto_memory
        self._memory_upto = 0  # messages already reviewed for memories
        # MCP servers' tools, as mcp__<server>__<tool>.
        self.mcp = mcp
        if mcp is not None:
            for tool in mcp_tools(mcp):
                self.tools.setdefault(tool.name, tool)
        for tool in extra_tools or []:
            self.tools[tool.name] = tool
        # Subagents: the task tool hands a self-contained job to a fresh agent.
        if agent_types:
            self.tools["task"] = TaskTool(self, agent_types)
        self.checkpoints = checkpoints
        self.max_turns = max_turns
        # A wall-clock limit (set_time_limit): the model is reminded to save its
        # work as it nears, and the run stops when it passes.
        self.time_limit: float | None = None  # seconds
        self.deadline: float | None = None  # time.monotonic()
        self._time_notes: set[str] = set()  # reminders already given
        # Before finishing a request that changed things, ask the model once to
        # check the request's explicit instructions (on by default when headless).
        self.final_check = final_check
        # Shell commands can't write even in the project (a read-only subagent);
        # plan mode has the same effect while it's on.
        self.read_only = False
        self._own_system = system is None
        self.system = system or self._build_system()
        self.conv = Conversation()
        self.usage = Usage()
        self.cost: float | None = self._new_cost()
        # Prompt size as of the last response, and how many messages it covered.
        self.context_tokens = 0
        self._billed_upto = 0
        self.status: RunStatus = "done"
        self.turns = self.tool_calls = self.tool_errors = 0
        # Plan mode: the mode the model was last told about, and the last saved plan.
        self.told_mode = "ask"
        self.plan_text: str | None = None
        self.plan_file: Path | None = None
        self.tool_runner = ToolRunner(self)
        # Whether a tool that can modify files ran successfully in this request.
        self.changed = False
        # Failed calls in this request (by call and result) and the current run of
        # failures, for noticing when the model is going in circles.
        self.failure_counts: dict[str, int] = {}
        self.failure_streak = 0
        # Given hooks (a subagent's) come ready-made; otherwise start with wren's own.
        if hooks is None:
            hooks = Hooks()
            register_builtins(hooks)
        self.hooks = hooks
        # One summary per subagent run in this session (see agent/subagents.py).
        self.subagent_runs: list[dict[str, Any]] = []
        # Looks up another configured model by name, for subagents that name one.
        self.resolve_model = resolve_model
        # Set while calls run concurrently: where subagents report their progress.
        self.progress: Progress | None = None
        # Set from another thread to stop this agent at its next step (a subagent
        # running concurrently, when the user interrupts).
        self.cancel: threading.Event | None = None
        # Context from session-start handlers, added to the next prompt.
        self._session_context: list[str] = []
        self._log_session_start()
        if time_limit:
            self.set_time_limit(time_limit)

    @property
    def messages(self) -> list[Message]:
        return self.conv.messages

    # --- public API --------------------------------------------------------

    def run(self, prompt: str, attachments: list[str] | None = None) -> str:
        """Run one user request to completion. Returns the final assistant text.

        `attachments` are extra text blocks sent with the prompt (e.g. an invoked skill)."""
        final_text = ""
        self.status = "done"
        self.changed = warned = False
        self.failure_counts, self.failure_streak = {}, 0
        stopped_by: set[str] = set()
        stop_blocks = cut_offs = 0
        try:
            verdicts = self.hooks.run("prompt", PromptSubmit(self, prompt))
            if blocked := denial(verdicts):
                self.ui.notice(f"prompt blocked by {blocked.source}: {blocked.reason}")
                self.status = "blocked"
                return final_text
            contexts = self._session_context + [v.context for v in verdicts if v.context]
            self._session_context = []
            extra = [TextBlock(c if is_reminder(c) else reminder(c)) for c in contexts]
            # Manage context before adding the prompt so it stays verbatim.
            self._manage_context(extra=len(prompt) // 4)
            self._start_turn(prompt)
            self.add_user([TextBlock(prompt), *(TextBlock(a) for a in attachments or []), *extra])
            for turn in range(self.max_turns):
                self._manage_context()
                response = self._call_model()
                if not response.message.content:
                    self.ui.notice("model returned an empty response")
                    self.status = "empty_response"
                    return final_text
                calls = response.message.tool_uses()
                final_text = response.message.text() or final_text
                cut_off = response.stop_reason == "max_tokens" and not calls
                if cut_off:
                    # Thinking cut off mid-way may lack its signature, which APIs reject when
                    # it's sent back; a turn without tool calls doesn't need it.
                    response.message.content = [b for b in response.message.content
                                                if not isinstance(b, ThinkingBlock)] or \
                        [TextBlock("[response cut off at the output limit]")]
                self._append(response.message)
                self._billed_upto = len(self.messages)

                if response.stop_reason == "refusal":
                    self.ui.notice("the model declined to continue")
                    self.status = "refusal"
                    self.tool_runner.abandon(calls, "Not executed: the response was a refusal.")
                    return final_text
                if cut_off:
                    cut_offs += 1
                    if cut_offs > MAX_CUT_OFFS:
                        self.ui.notice(f"stopped: {cut_offs} responses in a row were cut off at the "
                                       "max_tokens limit before acting")
                        self.status = "max_tokens"
                        return final_text
                    self.ui.notice("response was cut off at the max_tokens limit; asking the model to go on")
                    self.log.record("reminder", reason="cut_off", count=cut_offs)
                    self.add_user([TextBlock(reminder(CUT_OFF))])
                    continue
                cut_offs = 0
                if not calls:
                    if steered := self.take_steering():
                        self.add_user(steered)
                        continue
                    if stop_blocks < MAX_STOP_BLOCKS:
                        blocks = [v for v in self.hooks.run("stop", Stop(self, final_text, stopped_by))
                                  if v.decision == "block"]
                        if blocks:
                            # All reasons in one message, so one extra model call.
                            stop_blocks += 1
                            stopped_by.update(v.source for v in blocks)
                            self.add_user([TextBlock(reminder(*(v.reason for v in blocks)))])
                            continue
                    return final_text
                if not self.tool_runner.run(calls, truncated=response.stop_reason == "max_tokens"):
                    return final_text
                left = self.max_turns - turn - 1
                if self.final_check and not warned and 0 < left <= max(3, self.max_turns // 10):
                    warned = True
                    self.add_user([TextBlock(reminder(TURN_BUDGET.format(left=left)))])
                if self._out_of_time():
                    self.ui.notice(f"stopped at the {(self.time_limit or 0) / 60:.0f}-minute time limit")
                    self.status = "time_limit"
                    return final_text
            self.ui.notice(f"stopped after {self.max_turns} turns")
            self.status = "max_turns"
        except KeyboardInterrupt:
            self.tool_runner.abandon_pending("Interrupted by the user.")
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

        if self.auto_memory:
            try:
                self.extract_memories("the conversation is being compacted")
            except LLMError as e:  # memories are a bonus; compaction must go on
                self.ui.notice(f"could not save memories before compacting: {e}")
        kept_from = plan_compaction(self.messages, self._fixed_tokens(), self.model.compact_threshold)
        span = self.messages[:kept_from]
        before = self.estimated_context()
        self.ui.notice(f"compacting conversation (~{before // 1000}k tokens)…")
        self.ui.model_started()
        try:
            summary, usage = self._retrying(lambda: summarize(
                self.provider, self.system, span, [t.spec() for t in self.tools.values()]))
        finally:
            self.ui.model_finished()
        self.record_side_usage(usage, purpose="compaction")
        archive = self._archive(span)
        note = summary_note(summary, archive, format_todos(self.conv.todos))
        self.conv.compacted(summary, note, kept_from)
        self.ctx.read_files.clear()
        self._reestimate()
        self._memory_upto = len(self.messages)
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

    def _build_system(self) -> str:
        return build_system_prompt(
            self.ctx.cwd, prompt_section(self.skills),
            memory=memory_section(self.memory) if self.memory else "",
            mcp=self.mcp.instructions() if self.mcp else "",
            sandbox=self.ctx.sandbox.describe() if self.ctx.sandbox else "")

    def set_sandbox(self, sandbox: Sandbox | None) -> None:
        """Turn the sandbox for shell commands on or off, mid-session: bash gains or
        loses its way out, and the system prompt says what applies now."""
        self.ctx.sandbox = self.permissions.sandbox = sandbox
        if "bash" in self.tools:
            self.tools["bash"] = SandboxedBash() if sandbox else Bash()
        if self._own_system:
            self.system = self._build_system()
        self.log.record("sandbox", enabled=sandbox is not None, system=self.system)
        self._reestimate()

    def changed_files(self, point: RestorePoint) -> list[str]:
        """Files a rewind to `point` would restore. Raises CheckpointError."""
        if point.commit is None or not (self.checkpoints and self.checkpoints.enabled):
            return []
        return self.checkpoints.changed_files(point.commit)

    def diff(self, point: RestorePoint, paths: tuple[str, ...] = ()) -> str:
        """Changes to files since the snapshot taken before `point`, as a patch.
        Raises CheckpointError."""
        if point.commit is None or not (self.checkpoints and self.checkpoints.enabled):
            return ""
        return self.checkpoints.diff(point.commit, paths)

    def rewind(self, index: int) -> RestorePoint:
        """Go back to `self.conv.timeline[index]`: files and conversation as
        before that turn, or the history as before that compaction.
        Raises CheckpointError (and changes nothing) if files can't be restored."""
        point = self.conv.timeline[index]
        if point.commit is not None and self.checkpoints and self.checkpoints.enabled:
            self.checkpoints.restore(point.commit)
        self.conv.rewind(index)
        self._reset_context()
        self.log.record("rewind", index=index)
        return point

    def start_session(self, source: Literal["startup", "resume", "clear"]) -> None:
        """Fire session_start ("startup", "resume" or "clear")."""
        for v in self.hooks.run("session_start", SessionStart(self, source)):
            if v.context:
                self._session_context.append(v.context)

    def end_session(self) -> None:
        self.hooks.run("session_end", SessionEnd(self))
        self.ctx.jobs.kill_all()  # background commands don't outlive the session

    def notify(self, message: str) -> None:
        """Fire notification: the agent is waiting for the user."""
        self.hooks.run("notification", Notification(self, message))

    def restore(self, state: SessionState) -> None:
        """Continue a logged session (its log must be `self.log`)."""
        self.conv = state.conversation
        self.usage = state.usage
        self.cost = state.cost if state.cost is not None else self._new_cost()
        self._reset_context()
        # A session that crashed mid-turn can end with unanswered tool calls.
        if self.messages and self.messages[-1].role == "assistant":
            self.tool_runner.abandon(self.messages[-1].tool_uses(), "Interrupted: the session ended.")
        self._memory_upto = len(self.messages)  # reviewed when that session ended
        self.log.record("resume", model=self.model.name)

    def new_session(self, log: SessionLog) -> None:
        """Start over with an empty conversation in a new log."""
        self.log = log
        self.conv = Conversation()
        self.usage = Usage()
        self.cost = self._new_cost()
        self._reset_context()
        self._memory_upto = 0
        self._log_session_start()

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

    def memories_pending(self) -> bool:
        """Whether there is conversation the model hasn't reviewed for memories yet."""
        return self.memory is not None and any(
            m.role == "assistant" for m in self.messages[self._memory_upto:])

    def extract_memories(self, reason: str) -> int:
        """Have the model save what the conversation taught it, in a side
        request that doesn't enter the history. Returns how many memories it
        wrote. Raises LLMError."""
        if not self.memories_pending():
            return 0
        saved = extract_memories(self, reason)
        self._memory_upto = len(self.messages)
        return saved

    def set_time_limit(self, minutes: float) -> None:
        """Limit the wall-clock time of everything from now on (for headless runs
        whose harness stops them)."""
        self.time_limit = minutes * 60
        self.deadline = time.monotonic() + self.time_limit
        self.log.record("time_limit", minutes=minutes)

    def _out_of_time(self) -> bool:
        """Past the deadline: True. Nearing it: remind the model (once each)."""
        if self.deadline is None or self.time_limit is None:
            return False
        left = self.deadline - time.monotonic()
        if left <= 0:
            return True
        minutes, total = max(round(left / 60), 1), round(self.time_limit / 60)
        if left <= max(180, self.time_limit * 0.1) and "up" not in self._time_notes:
            self._time_notes.update(("up", "check"))
            self.add_user([TextBlock(reminder(TIME_UP.format(left=minutes)))])
        elif left <= self.time_limit * 0.25 and "check" not in self._time_notes:
            self._time_notes.add("check")
            self.add_user([TextBlock(reminder(TIME_CHECK.format(left=minutes, total=total)))])
        return False

    def record_side_usage(self, usage: Usage, purpose: str) -> None:
        """Account for a request outside the conversation (e.g. a prediction):
        it counts toward tokens and cost, not turns or the context size."""
        self.add_usage(usage, self.model.cost(usage), self.model.name, purpose)

    def add_usage(self, usage: Usage, cost: float | None, model: str, purpose: str) -> None:
        """Account for side requests whose cost is already known (a subagent's)."""
        self._charge(usage, cost, model=model, purpose=purpose)

    def _charge(self, usage: Usage, cost: float | None, **record: Any) -> None:
        """Count tokens and cost (unknown once any part of it is) and log them."""
        self.usage += usage
        if cost is not None and self.cost is not None:
            self.cost += cost
        self.log.record("usage", **record, usage=vars(usage), cost=cost)

    def _new_cost(self) -> float | None:
        """Cost starts at zero when the model has a price, else it's unknown."""
        return 0.0 if self.model.price else None

    def _reset_context(self) -> None:
        """After the history was replaced: nothing billed for it yet, no file seen."""
        self.context_tokens, self._billed_upto = 0, 0
        self.ctx.read_files.clear()

    def _log_session_start(self) -> None:
        self.log.record("session_start", model=self.model.name, cwd=str(self.ctx.cwd), system=self.system,
                        mask_at=self.model.mask_at, compact_at=self.model.compact_threshold)

    # --- model -------------------------------------------------------------

    def _call_model(self) -> Response:
        return self._retrying(self._stream_model)

    def _retrying[T](self, request: Callable[[], T]) -> T:
        """Run a model request, retrying after a pause when it fails in a way that
        may pass (a connection dropped or timed out mid-stream, an overloaded
        server) rather than ending a run that may be hours in."""
        for delay in MODEL_RETRY_DELAYS:
            try:
                return request()
            except LLMError as e:
                if not e.retryable:
                    raise
                self.log.record("error", error=str(e), retry_in=delay)
                self.ui.notice(f"{e}; retrying in {delay}s")
                if self.cancel is not None:
                    self.cancel.wait(delay)
                else:
                    time.sleep(delay)
                self.check_cancel()
        return request()

    def _stream_model(self) -> Response:
        self.ui.model_started()
        response: Response | None = None
        try:
            for event in self.provider.stream(
                system=self.system,
                messages=self.messages,
                tools=[t.spec() for t in self.tools.values()],
            ):
                self.check_cancel()
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
            raise LLMError("stream ended without a final message", retryable=True)

        self.turns += 1
        self.context_tokens = response.usage.context_tokens + response.usage.output_tokens
        self._charge(response.usage, self.model.cost(response.usage), model=self.model.name,
                     stop_reason=response.stop_reason)
        return response

    # --- tools -------------------------------------------------------------

    def check_cancel(self) -> None:
        """Stop here (KeyboardInterrupt) if another thread asked this agent to."""
        if self.cancel is not None and self.cancel.is_set():
            raise KeyboardInterrupt

    # --- history -----------------------------------------------------------

    def _append(self, message: Message) -> None:
        self.conv.append(message)
        self.log.record("message", **message.to_dict())

    def steer(self, text: str) -> None:
        """Pass the model a message while it works, from any thread: it goes in
        after the current tool calls, or as the next prompt if the model is done."""
        self._inbox.append(text)

    def unsent(self) -> list[str]:
        """Messages passed to steer that the model hasn't seen yet; takes them back."""
        texts = []
        while self._inbox:
            texts.append(self._inbox.popleft())
        return texts

    @property
    def queued(self) -> int:
        return len(self._inbox)

    def take_steering(self) -> list[ContentBlock]:
        """The messages sent with steer, as blocks for the next user message (shown as they go)."""
        texts = self.unsent()
        for text in texts:
            self.ui.steered(text)
        return [TextBlock(reminder(STEERED)), *(TextBlock(t) for t in texts)] if texts else []

    def add_user(self, blocks: list[ContentBlock]) -> None:
        """Add a user message to the history (and the log)."""
        self.conv.add_user(blocks)
        self.log.record("message", **Message("user", list(blocks)).to_dict())
