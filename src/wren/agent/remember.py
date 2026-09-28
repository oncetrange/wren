"""Saving memories before the conversation leaves the context.

Models rarely stop mid-task to write a memory, so at compaction and at the
end of a session the model gets one side request: the conversation so far (the
same prefix as its last request, so mostly a cache read) plus an instruction to
review it for memories. It may only use the memory tool, for a few rounds.
Nothing of this enters the history; its tool calls are shown and logged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wren.agent.plans import reminder
from wren.llm.types import Completed, LLMError, Message, TextBlock, ToolResultBlock
from wren.tools.base import ToolError, ToolOutput, validate_args

if TYPE_CHECKING:
    from wren.agent.loop import Agent

EXTRACT_REQUEST = """Before the conversation above leaves your context ({reason}), review it \
for anything worth keeping in long-term memory, following the Memory guidelines in your \
system prompt: the user's corrections and confirmed preferences, facts about the user, and \
project context that isn't in the code or git history. Check the memory index first and \
update or delete memories rather than adding duplicates. Most conversations teach nothing \
new: save only what will clearly help in a future session. Use only the memory tool. When \
done, or if there is nothing to save, reply with just DONE."""
MAX_ROUNDS = 4
# Sent with /remember <text>.
REMEMBER_REQUEST = """The user used /remember: their message above is something to keep in \
long-term memory. Save it now with the memory tool, following the Memory guidelines (pick the \
scope and type; update an existing memory if one covers it), then confirm in one short line."""


def extract_memories(agent: Agent, reason: str) -> int:
    """One side conversation of up to MAX_ROUNDS requests. Returns the number of writes."""
    tool = agent.tools["memory"]
    request = [Message(m.role, list(m.content)) for m in agent.messages]
    ask = TextBlock(reminder(EXTRACT_REQUEST.format(reason=reason)))
    if request and request[-1].role == "user":
        request[-1].content.append(ask)
    else:
        request.append(Message("user", [ask]))
    writes = 0
    for _ in range(MAX_ROUNDS):
        agent.ui.model_started()
        response = None
        try:
            # Every tool stays declared (the history has calls to them, and
            # the request keeps its cached prefix); only memory calls run.
            for event in agent.provider.stream(system=agent.system, messages=request,
                                               tools=[t.spec() for t in agent.tools.values()]):
                if isinstance(event, Completed):
                    response = event.response
        finally:
            agent.ui.model_finished()
        if response is None:
            raise LLMError("stream ended without a final message")
        agent.record_side_usage(response.usage, purpose="memory")
        request.append(response.message)
        calls = response.message.tool_uses()
        if not calls:
            break
        results = []
        for call in calls:
            if call.name != "memory":
                results.append(ToolResultBlock(call.id, "Only the memory tool is available now.", True))
                continue
            if problem := validate_args(tool.input_schema, call.input):
                results.append(ToolResultBlock(call.id, f"invalid arguments: {problem}", True))
                continue
            label = tool.describe(call.input, agent.ctx)
            try:
                output = tool.run(call.input, agent.ctx)
            except ToolError as e:
                output = ToolOutput(str(e), is_error=True, summary=str(e).splitlines()[0])
            action = call.input.get("action")
            if action in ("write", "delete") or output.is_error:
                # Reads and lists are bookkeeping; changes are worth showing.
                agent.ui.tool_started("memory", label)
                agent.ui.tool_finished("memory", output)
            writes += action in ("write", "delete") and not output.is_error
            agent.log.record("tool", name="memory", input=call.input, is_error=output.is_error,
                             purpose="memory", error=output.content[:500] if output.is_error else None)
            results.append(ToolResultBlock(call.id, output.content, output.is_error))
        request.append(Message("user", results))
    return writes
