You are digesting one trajectory of a coding agent (wren) on a benchmark task, for
an engineer improving the agent's harness who will rely on your digest instead of
reading the trace. The trace is the file `$trace` in this directory: the agent's
turns, its tool calls (`[call N]`) and their results (`[result N]`), then the
verifier's output. Task: $task. Lens: $lens.
$questions
Read efficiently: the grade and the verifier's output at the end first, then grep
for anchors and read only the slices that matter. Don't modify anything.

Claims must be anchored in the trace: quote short exact excerpts and say where
(the call number). No speculation, and don't blame the model or the harness;
describe what happened.

If you were given questions, answer them in an extra "answers" field.

Reply with only one JSON object, at most 6000 characters, of this shape:

- lens "failure": {"blocker": "what stopped the task passing, in one sentence",
  "narrative": "how the run went, in a few sentences",
  "evidence": [{"where": "call 12", "quote": "..."}],
  "verifier_evidence": "what the verifier says failed",
  "capability_note": "what the agent lacked: knowledge, a tool, context, time...",
  "needed_instead": "what would have had to happen instead"}
- lens "capability_gap": {"wanted": "what the agent was trying to accomplish",
  "why_couldnt": "what stopped it: a tool's limits, missing information, a dead end...",
  "evidence": [{"where": "call 12", "quote": "..."}],
  "workaround_seen": "any partial workaround it attempted"}
- lens "success": {"habits": [{"habit": "what the agent did that worked", "where": "calls 3-9"}],
  "risk_if_removed": "which harness behavior this run depended on"}
