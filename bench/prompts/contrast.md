You are comparing two trajectories of a coding agent (wren) on the same benchmark
task, for an engineer improving the agent's harness who will rely on your digest
instead of reading the traces. Task: $task.

- `$inc_trace`: the incumbent harness, which $inc_outcome.
- `$cand_trace`: an earlier candidate harness, $candidate, which $cand_outcome.
  The candidate was $verdict.
- `$diff`: the candidate's change to the incumbent's source. Its declared edits:

$edits

The traces are files in this directory: the agent's turns, its tool calls
(`[call N]`) and their results (`[result N]`), then the verifier's output. The
harnesses differ, so the runs diverge from the start: don't compare them step by
step. Find whether and where the candidate's mechanism acted (its trigger
condition is declared above), and what the run did differently after it did.

With a few trials per task, a single task coming out differently is often
chance: the policy model samples, and the same harness can pass and fail the
same task. Say "no" or "unclear" when the candidate's mechanism never acted, or
acted without visibly changing the course of the run; that is as useful as a
"yes". Read efficiently: the grades and the verifier's output first, then grep
for the mechanism's anchors. Don't modify anything.

Claims must be anchored in the traces: quote short exact excerpts and say which
trace and where (the call number). No speculation.

Reply with only one JSON object, at most 6000 characters:

{"decisive_difference": "what made one run pass and the other fail, in one sentence",
 "caused_by_edit": "yes | no | unclear",
 "which_edit": "the edit id it came from, or null",
 "evidence": [{"trace": "incumbent | candidate", "where": "call 12", "quote": "..."}],
 "standalone": "whether the effect stands on the edit alone or leans on the candidate's other edits",
 "generalizes": "whether the mechanism is generic or only fits this task"}
