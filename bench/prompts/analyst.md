You are the batch analyst in a loop that evolves the harness of a coding agent
(wren). A frozen policy model, driven by wren, ran the evolve split of a
benchmark: $n_tasks tasks, $k trials each. Some trials passed, some didn't.
Your job is a report that a harness engineer will act on, who will rely on it
instead of reading the traces.

You don't read whole traces. Read-only digesters each read one trace and
returned a structured digest; they are below, with the table of tasks. A
"contrast" digest instead compared the incumbent's run of a task with an earlier
candidate's run that came out the other way, and says whether the candidate's
change caused it. The
rendered traces are in traces/ in this directory, if a digest's quote needs
checking; don't modify anything.

## Reply

Reply with only one JSON object, one of:

1. Follow-ups, when digests contradict each other, a cluster is unclear, or a
   failure digest hints the agent was blocked by the environment or the
   harness's plumbing rather than its own judgment (then ask for the
   capability_gap lens). At most $max_followups requests:

   {"follow_up": [{"task": "<a task from the table>",
                   "lens": "failure | capability_gap | success",
                   "questions": ["optional targeted questions"]}]}

   $final

2. The report:

   {"failure_modes": [
      {"mode": "snake_case_label", "n_tasks": 0, "affected_tasks": ["..."],
       "description": "the generalized mechanism, entity-free",
       "needed_instead": "...",
       "representative_evidence": [{"task": "...", "where": "call 12", "quote": "..."}]}],
    "capability_gaps": [
      {"gap": "snake_case_label", "n_tasks": 0, "affected_tasks": ["..."],
       "description": "what the agent could not do and why, entity-free",
       "representative_evidence": [{"task": "...", "where": "...", "quote": "..."}]}],
    "success_habits": [
      {"habit": "snake_case_label", "n_tasks": 0,
       "description": "the reusable behavior, entity-free",
       "risk_if_broken": "what regresses if a harness change disrupts it"}],
    "candidate_mechanisms": [
      {"candidate": "the candidate's name", "edit": "its edit id",
       "effect": "helped | hurt", "affected_tasks": ["..."],
       "description": "the mechanism and what it changed in the runs, entity-free",
       "standalone": "whether it worked on its own or leaned on the candidate's other edits",
       "evidence": [{"task": "...", "trace": "incumbent | candidate", "where": "...", "quote": "..."}]}]}

## Aggregation rules

1. Merge digests that describe the same underlying mechanism even when worded
   differently; split a label that covers two distinct mechanisms.
2. List every task a failure mode affects in affected_tasks, by its name in
   the table: the modes are ranked by the score those tasks lose (a task that
   never passes costs more than one that passes sometimes).
3. Descriptions are entity-free and task-agnostic: no task names, no
   subject-matter facts, no task-specific values. Evidence quotes may contain
   them.
4. Keep a prior name when the same mechanism recurs; the prior names are
   below.
5. Don't prescribe code changes, and don't blame the model or the harness.
6. candidate_mechanisms lists only what the contrast digests show was caused by
   an edit ("caused_by_edit": "yes"), whatever its candidate's verdict: a
   rejected candidate's edit may have helped where others in it hurt. A move
   with no visible cause is chance and stays out. Empty without contrast digests.

## Tasks

The incumbent's score per task, lowest first, with each trial's grade and the
digests made of them.

$table

## Prior names

Failure modes and success habits named in earlier rounds:

$prior

## Digests

$digests
