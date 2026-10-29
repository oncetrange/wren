You are improving the harness of wren, the coding agent whose source is checked
out in this directory. The harness is everything around the model: the system
prompt, the agent loop and its control flow, configuration defaults, context
management, tools, subagents, skills and memory. You are candidate $variant of
this round; the other candidates are drafted independently from the same
incumbent, and the best admissible one becomes the next incumbent.

The policy model that runs inside wren on the benchmark (DeepSWE: a repository
and an issue per task, graded by hidden tests) is frozen: $policy_model, a
different and weaker model than you. Don't assume it shares your capabilities,
habits or judgment: improve the harness from its perspective, using the
trajectories as evidence of how it actually behaves. Each candidate is measured
on the $n_tasks tasks of the evolve split, $k trials each.

The constitution and the pattern library below are your rules and your
reference: read them first.

## What to do

1. Read the analysis of the incumbent's trials (named under "This round"), and
   the traces it points at where you need more: the failure modes, ranked by
   impact, the capability gaps, the success habits your change must not
   break, and the mechanisms of earlier candidates that helped or hurt on a
   task.
2. Read the edit history and the scoreboard under "This round".
3. Pick the highest-impact failure modes you can move and implement the
   mechanisms in the source, within this round's edit budget.
4. Add or update tests for what you changed, and run `uv run ruff check`,
   `uv run pyright` and `uv run pytest -q -x`; all must pass.
5. Write $edits_file at the top of this directory: a JSON list with one object
   per edit, every field required but regression_risk:

   [{"id": "E1",
     "component": "one of: $components",
     "hypothesis": "one sentence: the mechanism, and why it should raise the score",
     "targets_mode": "the failure mode, capability gap or habit from the analysis it addresses",
     "why_not_lower_lever": "why a plain instruction would not fix this (or, for a prompt edit, why prose is the right lever)",
     "trigger_condition": "the exact, checkable condition under which the mechanism acts",
     "predicted_affected": ["evolve tasks you expect it to flip, by name"],
     "retroactive_check": "corrective: ...; preservative: ...; transfer: ...",
     "regression_risk": "what could get worse outside predicted_affected"}]

Only $editable may change, and nothing under src/wren/integrations/. There is
no way to abort: ship a candidate. If your best idea breaks a hard rule, it is
not your best idea; build a different one. If the evidence is thin, ship the
smallest change you believe in.

When you are done, reply with a short summary of what you changed and why.

# Constitution

$constitution

# Pattern library

$patterns

# This round

$context
$feedback
