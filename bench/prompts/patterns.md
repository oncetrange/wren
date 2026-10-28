# Pattern library (reference, not an allowlist)

Known families of mechanisms for coding-agent harnesses, each with a generic
trap. Adopt, adapt or ignore them, or invent something that isn't here. Each
one is a hypothesis to be measured, not a known win for this policy.

## 1. Verification before finishing

Examples: a bounded pass that reruns the project's own tests (found from its
CI configuration or test scripts) once the change is made; having the model
reproduce the issue before fixing it, so it has a check that should flip;
re-reading the issue's explicit requirements against the diff once before
stopping. wren already has `final_check`, a stop handler that fires once per
request that changed files: refine it before adding a parallel one.

Trap: an unbounded "check again" loop, or a check that can't pass (a test
suite that needs the network) and so blocks finishing. Every check needs a
give-up path, and runs out of turns or time are still graded as they stand.

## 2. Localization and exploration budget

Examples: guidance to find the code an issue concerns with grep/glob before
reading whole files; noticing a long stretch of reads with no edit and
nudging toward a hypothesis; using the read-only explore subagent for broad
searches so the main context stays small.

Trap: forcing an edit too early turns a slow success into a fast failure.
Nudge at a checkable threshold, once.

## 3. Edit and tool-error recovery

Examples: error messages that say what to do next (the closest match when an
edit_file string isn't found, the right flag when a command fails); noticing
the same call failing the same way and saying so (`repeated_failures` does
this today); recovering from a test runner or build that isn't installed.

Trap: auto-fixing a call can change what the model meant. Explain and point
the way rather than silently rewriting.

## 4. Long-running commands

Examples: running builds and test suites as background jobs and polling them
with bash_output; timeouts that fit what the command is; saying in the
timeout message whether the process is still running.

Trap: polling forever on a hung process. Pair any waiting with a bounded
number of checks, then kill it.

## 5. Output plumbing

Examples: truncation that keeps the lines that matter (the failing test and
its traceback, the compiler error) rather than a blind head and tail;
collapsing progress-bar and download spam; showing the end of a long test
run, where the summary is.

Trap: raising the limits wholesale fills the context on every long run.
Preserve selectively.

## 6. Context budget

Examples: which old tool results are masked and which survive (the issue
text, the reproduction, the latest test output); what the compaction summary
must keep for a headless task (the issue, what was tried and why it failed,
the current state of the working tree); re-reading files after a compaction.

Trap: a summary of a summary loses the task. Regenerate what matters from
pinned originals, not from the previous summary.

## 7. Getting unstuck

Examples: noticing that recent turns repeat the same calls and outputs and
telling the model to change approach; when a run has gone wrong, a fresh
start of the same policy with a clean context, the issue, and the earlier
diff offered as an unverified lead it may inspect, apply or discard, with the
instruction to reproduce the issue first.

Trap: successful runs also repeat themselves (rerunning a test while fixing
it), so a stuck detector needs a threshold that rarely fires on runs that
pass. Stopping a run without recovering it only turns it into a failure.

## 8. Bounded extra calls

Examples: one extra call of the policy with a tight "return only X" brief: a
plan critique once the issue is understood, a pre-finish check that lists
each requirement as met or unmet.

Cost: every extra call runs on the same frozen policy and costs tokens and
time that the gain has to pay for. Never delegate the whole task.
