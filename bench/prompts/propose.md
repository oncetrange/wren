You are improving the harness of wren, the coding agent whose source is checked
out in this directory. The harness is everything around the model: the system
prompt, the agent loop and its control flow, configuration defaults, context
management, tools, subagents, skills and memory. You are candidate $variant of
this round.

## The setting

wren is measured on a fixed set of software engineering tasks (DeepSWE): each
task is a repository and an issue, and hidden tests grade the result. The policy
model that runs inside wren for these tasks is frozen: $policy_model, a different
and weaker model than you. You can't make it smarter, only give it a better
harness. The model, its turn budget and the time limit are set from outside, so
changing their defaults does nothing. Your change will be measured on several
trials of every task and accepted only if it beats the incumbent harness by more
than the noise band, and pays for any extra tokens with that gain.

## What to do

1. Read the analysis of the incumbent's trials (named below), and the traces it
   points at where you need more. Look for failure modes that recur across tasks,
   and for what successful runs depended on.
2. Read the edit history below. Don't redraw a hypothesis that was measured and
   failed unless you have a reason it failed for another cause.
3. Pick the most promising mechanisms and implement them in the source. Prefer the
   lowest lever that addresses the failure (a sentence of the prompt before a new
   tool), but don't shy from structural changes when the evidence calls for them.
4. Add or update tests for what you changed, and run `uv run ruff check`,
   `uv run pyright` and `uv run pytest -q -x`; all must pass.
5. Write $edits_file at the top of this directory: a JSON list with one object per
   edit, e.g.

   [{"id": "E1", "component": "context_mgmt",
     "hypothesis": "the mechanism, and why it should raise the score",
     "targets_mode": "the failure mode from the analysis it addresses",
     "predicted_affected": ["the tasks you expect it to flip"],
     "regression_risk": "what could get worse"}]

   component is one of: $components.

An edit is ONE independent, attributable change: if two parts could be accepted
or rejected separately, they are two edits. Ship at most this round's edit budget.
Only $editable may change.

## Rules

- No logic for particular tasks: no task or repository names, no expected answers,
  no branches that recognize a benchmark. Changes are reviewed for this and
  rejected.
- Nothing that reads or depends on the grader, the verifier or hidden tests.
- No unbounded retries or waits.
- There is no way to abort: ship a candidate. If the evidence is thin, ship the
  smallest change you believe in.

When you are done, reply with a short summary of what you changed and why.

## This round

$context
$feedback
