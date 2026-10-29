# Recursive self-improvement

wren can improve its own harness: everything around the model, from the system prompt and the
agent loop to context management, tools and subagents. The model itself is frozen. Each round, wren
reads how the current harness failed on a benchmark, proposes and implements changes to its own
source, and a statistical rule decides which change, if any, replaces the harness. The loop follows
[RRSI](https://github.com/google-research/rrsi).

Two scripts make up the loop:

- `bench/evolve.py` measures a harness on the benchmark and judges one measurement against another.
  Hand-made changes go through it the same way.
- `bench/rsi.py` runs the loop: analysis, proposals, screening, then `evolve.py` to measure and
  select.

```
               ┌─────────────────────── rsi.py: round t ───────────────────────┐
  incumbent ──►│ analyze ─► propose (×2, in parallel) ─► screen ─► measure+select │──► incumbent
               └───────────────────────────────────────────────┬────────────────┘
                                                               │
               evolve.py: measure on the evolve split (Pier on Modal), judge by the selection rule
```

How to run it is in [benchmarks.md](benchmarks.md). This page explains how it works.

## Terms

- **Harness**: a wren commit, pushed (the task containers install wren from GitHub), plus the agent
  kwargs it runs with (`time_limit`, `max_turns`). The policy model is recorded with it, so
  measurements of different models are never compared.
- **Split**: `bench/split.json` divides the 113 DeepSWE tasks into 30 **evolve** tasks, used to
  choose, and 83 **held-out** tasks, used only to report a result. The split never changes.
- **Measurement**: one run of a harness on a split, k trials per task (k = 2: 60 trials on evolve).
- **S**: the share of trial slots that pass. **C**: mean policy tokens (input + output) per trial,
  over the trials that report them.
- **Incumbent**: the harness to beat. **S\***: the best score an incumbent has had in the run.
- **δ (delta)**: the noise band, how much S moves between two measurements of the same harness.
- **Candidate**: a proposed harness. **Edit**: one independent mechanism in a candidate; a
  candidate holds one or more. **Component**: what an edit changes, one of `prompt`,
  `control_flow`, `config`, `context_mgmt`, `tools`, `subagent`, `skill`, `memory`.

## Measuring a harness

### A measurement

A measurement runs as one Pier job, `jobs/bench-<name>`, and is saved as
`bench/runs/evals/<name>.json`, trial by trial. A trial without a grade counts as a failure, so a
harness can't look better by crashing on the tasks it finds hard. Yet if more than a fifth of the
trials have no grade (`invalid_missing_frac`), the infrastructure failed, not the harness: the
measurement is not used for anything until it is resumed and complete.

Trials that end in an agent or verifier timeout, or a verifier that produced no reward, are the
trial's own result. Any other exception (a spend limit, a vanished sandbox) is an infrastructure
failure, and resuming the measurement runs that trial again. A job is resumed only under the
harness and the sandbox sizing it started with, and two measurements are compared only if they ran
the same tasks with the same policy model.

### The noise band

A task that passes once can fail the next time, so two measurements of one harness differ. δ is
how much they differ, and a change has to beat it to count.

The same harness is measured twice and `calibrate` computes δ = z · sd, with z = 2. The sd comes
from the trials, not from the two scores. The two measurements are pooled task by task (four trials
per task at k = 2), and resampling trials within each task gives the spread of S that trial-to-trial
chance alone produces. Tasks that always pass or always fail contribute nothing, and every task
that passes only sometimes contributes. The estimate is scaled to the difference of two
measurements.

The difference of the two scores is a single draw, too noisy to set δ by, but it is the one
estimate that sees noise shared across trials, such as a provider slowing down for a whole
measurement. `calibrate` reports it and warns when it is more than twice what the trials predict.

### The selection rule

A candidate's S' and C' are compared with the incumbent's S and C. ΔS = S' − S, and ΔC is the
relative change in tokens.

| Case | Accepted when |
|---|---|
| S' < S\* − δ | never: below the floor |
| ΔS > δ | ΔC ≤ β0 + β1 · ΔS |
| otherwise | w_s · min(ΔS, 0) − w_c · ΔC > 0 |

- **The floor** keeps a series of small losses, each within the noise, from walking the score
  down: S\* is the best score so far, not the incumbent's.
- **A gain above δ** may cost tokens: β0 = 0.10 is free, and β1 = 5 lets a gain of one trial in 20
  buy 25% more tokens.
- **Inside the band** a gain counts for nothing, since it may be noise. Only a token saving gets a
  candidate in, and it must also pay for any score lost (w_s = 100, w_c = 15: one trial fewer
  passing out of 60 needs 11% fewer tokens). Otherwise the cheapest way in would be an agent that
  gives up sooner.

Among the admissible candidates of a round, the highest S wins, and fewer tokens break a tie.

### Records

Every verdict, rejections included, is appended to `bench/runs/ledger.jsonl`. The incumbent and S\*
are in `bench/runs/frontier.json`, and δ in `bench/runs/calibration.json`. These belong to the
benchmark, not to one run: an accepted change, by hand or by a run, moves the same incumbent, and a
later run starts from wherever it is, under the same S\*.

## A round

A run starts from the current incumbent (`evolve.py baseline` makes the first one) and needs a
calibrated δ. `start` points the branch `evolve/<run>` at the incumbent's commit, and the run goes
for T rounds (T = 8). Each round has four steps.

### 1. Analyze

The incumbent's trials are rendered to text: the grade, every session's model turns and tool calls
before any compaction or masking, the tail of wren's stderr and of the verifier's output, up to
150,000 characters.

**Which trials.** Up to `max_digests` = 12 per round, of the trials whose logs were kept:

1. on tasks that pass only sometimes, a failure and a success, since the contrast says most;
2. one failure of each task that never passes;
3. the longest run (by tokens) of up to `stable_digests` = 3 tasks that always pass. A long run
   leans on the harness (compaction, recovery, long commands), and what it leans on is what a
   change must not break;
4. further failures, as room allows.

**Contrasts.** Before those, up to `contrast_digests` = 3 slots go to tasks on which an earlier
candidate of the run, rejected or not, came out the other way from the incumbent. Only strong
moves count: all of one side's trials passed and none of the other's, or two or more candidates
moved the task the same way. With two trials per task, a single partial move is mostly chance. The
report says how many strong moves there were. Slots without one go back to the other trials.

**Digests.** A read-only wren reads each trace and returns structured JSON with quoted evidence,
under one of these lenses:

- failure: what blocked the task, how the run went, what the verifier says, what the agent lacked,
  what was needed instead;
- capability gap: what the agent tried to do, what stopped it (a tool's limits, missing
  information), any workaround it attempted;
- success: the habits that worked, and which harness behavior the run depended on;
- contrast: reads both runs with the candidate's diff and declared edits, and says what decided the
  outcome, whether the candidate's edit caused it (yes, no or unclear), which edit, and whether
  the effect stands on its own.

**The analyst.** Another read-only wren reads the table of tasks and all the digests. It may first
ask for up to `max_followups` = 6 more digests, choosing the task and lens and adding questions of
its own, for example when digests contradict each other or a failure looks like the harness's
plumbing. Then it reports:

- failure modes, each with the tasks it affects. They are ranked by the score those tasks lose:
  each task's share of failing trials, summed, so a task that never passes weighs twice one that
  passes half the time;
- capability gaps: these often need a tool or plumbing fix rather than an instruction;
- success habits a change must not break;
- candidate mechanisms: edits of earlier candidates that the contrasts showed helped or hurt.

Descriptions are generic, without task names. The names of failure modes and habits carry over
from round to round when a mechanism recurs. `report.md` holds the task table, the analyst's report
and every digest.

### 2. Propose

Each round drafts `candidates` = 2 candidates in parallel, each in its own git worktree and each
from the incumbent, independently of the others. A
writing wren (`--yolo`) edits wren's source there and adds tests. It may change `src/wren/`,
`tests/` and `docs/`, but not `src/wren/integrations/`, the benchmark adapter that also reports the
tokens C is made of.

The proposer reads:

- **the constitution** (`bench/prompts/constitution.md`): how candidates are judged; the
  overfitting trap; hard rules (no task-specific content, never touch the grader, don't break the
  CLI or the session log format, keep safety mechanisms, bound every check or retry, which
  settings are injected by the benchmark and so can't be changed by editing them); wren's levers
  file by file, with the component each one is; and the proposal discipline below;
- **the pattern library** (`bench/prompts/patterns.md`): known families of harness mechanisms
  (verification before finishing, localization, edit recovery, long commands, output plumbing,
  context budget, getting unstuck, bounded extra calls), each with its typical trap. It is a
  reference, not an allowlist;
- **the analysis** of this round;
- **this round's directives**:
  - the **edit budget** b_t, the most independent edits one candidate may hold. It anneals on a
    cosine from `b_max` = 3 in the first round to `b_min` = 1 in the last (3, 3, 3, 3, 2, 2, 2, 1
    at T = 8). Early rounds may bundle coordinated edits; late ones measure a single mechanism,
    which credits the gain to it;
  - **exploration**: after `stall_window` = 2 rounds in which the incumbent's score rose by no
    more than δ, the first candidate must include an edit on a component no measured edit has
    touched. Memory and skill are never required: each trial starts with an empty `~/.wren`,
    memory is off headless, and no skill reaches a trial unless the harness ships one;
  - **pruning**: components with accepted machinery whose edits gained nothing in the last
    `n_prune` = 4 rounds are listed, and removing that machinery is a legitimate edit;
  - **the edit history**: the latest 40 edits with their measured ΔS and ΔC and verdicts, of
    which at most 4 that were never measured (dropped in screening), so failed drafts don't crowd
    out measurements. A rejected mechanism is not to be redrawn unchanged, unless the candidate
    mechanisms show it helped on its own; then it may come back isolated or refined;
  - **the scoreboard**: for each of the latest 20 measured edits, which of its predicted tasks
    improved, every task its candidate improved, and the tasks it fell on without predicting them.

It declares each edit in `RSI_EDITS.json`:

| Field | What it says |
|---|---|
| `id` | a name for the edit |
| `component` | what it changes |
| `hypothesis` | the mechanism, and why it should raise the score |
| `targets_mode` | the failure mode, capability gap or habit from the analysis it addresses |
| `why_not_lower_lever` | why a plain instruction wouldn't do (or, for a prompt edit, why prose is the right lever) |
| `trigger_condition` | the exact, checkable condition under which it acts |
| `predicted_affected` | the evolve tasks it should flip |
| `retroactive_check` | which cited failures it would have fixed, which success habits it could break, and why it transfers to other repositories |
| `regression_risk` | optional: what could get worse elsewhere |

### 3. Screen

A candidate is checked before any measurement is spent on it:

1. **Its declaration**: present, within the budget, every field filled, a known component,
   predictions that name evolve tasks only, and only allowed files changed.
2. **Its components, from the diff.** Each changed file shows a component by its path
   (`agent/prompt.py` is prompt, `agent/compact.py` context_mgmt, `tools/` tools, and so on). A
   change only to strings and comments is prompt, and a new value for a constant is config. When
   the diff shows an edit's declared component, the declaration stands. Otherwise the edit is
   recorded under what the diff shows. An exploration slot is met only by what the diff shows, so
   a system-prompt edit can't pass as subagent work.
3. **The repo's checks**: `ruff check`, `pyright`, `pytest`.
4. **A scan of the added lines** for task names (evolve and held-out), grader files and
   credentials.
5. **A critic**, a read-only wren that rejects:
   - logic aimed at particular tasks;
   - a no-op, or a safety mechanism removed without a replacement;
   - a gamed grader;
   - changes no edit declares, or one edit that hides several mechanisms;
   - state that carries task data across trials;
   - a check, retry or wait without a bound.

A candidate that fails a step goes back to the proposer with the objections, up to
`repair_rounds` = 2 times. The proposer fixes it in the same worktree, starting over only if the
objections are to the idea itself. If it still fails, it is dropped, and the drop is recorded.

### 4. Measure and select

Candidates that pass screening are committed (without `RSI_EDITS.json`) and pushed as
`evolve-cand/<run>/t<t>-v<i>`, then measured on the evolve split in parallel. A measurement whose
first three finished trials all crashed the harness (wren exiting non-zero in the container, or
failing to install) is stopped there, since the rest would crash too, and its candidate is
dropped.

The selection rule judges each measured candidate against the round's incumbent, and the winner,
if any, becomes the incumbent: `frontier.json` is updated and the branch `evolve/<run>` moves to
it. Every edit goes into the run's history with its candidate's ΔS and ΔC, its verdict, and how
the tasks moved against the incumbent. Every measured candidate goes into the ledger.

After the last round, the starting harness and the final one (`evolve/<run>`, pushed to the
remote; fetch it first) are both measured on the held-out tasks, which the loop never saw. The difference between those two measurements is
the run's result: the evolve scores were used to choose, so they overstate it.

## Keeping it honest

The harness is evolved on the same tasks that score it, so the loop guards against changes that
score better without being better:

- **Held-out tasks** never take part in a choice, and the final result is reported on them.
- **The grader and the adapter are out of reach**: changes under `src/wren/integrations/` are
  refused, and the scan and the critic look for anything that reads hidden tests, reward files or
  the verifier's output.
- **No task-specific content**: task names are scanned for, and the critic rejects logic that
  recognizes a task, its repository or the benchmark's environment. The question it asks: would the
  change still help on an unfamiliar repository and issue?
- **Nothing carries over between trials**: each starts from a fresh container and `~/.wren`, and the
  critic rejects state that would carry task data.
- **Noise can't buy a place**: gains inside δ count for nothing, losses inside δ must be paid for in
  tokens, and the floor stops a walk downward.
- **Components are what the diff shows**, so the history, the exploration rule and pruning see
  what was actually changed.
- **Separate models**: the proposer, digesters, analyst and critic are the installed `wren`, not the
  one being changed, running a model other than the frozen policy model (`start` warns when the
  proposer's is the same).

## Running unattended

A run is meant to go for days without anyone watching, and to survive being cut off.

- **Everything is saved as it is made**: traces, digests, each analyst turn, `report.md`, each
  candidate's `candidate.json`, each measurement. Running `rsi.py run` again skips what exists and
  goes on from there.
- **The round's incumbent is pinned** in `state.json` when the round starts, so a round cut off after
  its winner reached the frontier is finished against the same incumbent. The history is written
  once per round.
- **A failure of the machinery is never recorded as a verdict on an idea.** A role whose wren can't
  run (its model request failed for good, it crashed or hung) is tried again after a pause (60 s,
  then 120 s). If it still fails, the round is cut off with nothing of that step saved, so it is
  done again later. The proposer isn't retried in place, since a failed run may have left half an
  edit; the rerun starts its worktree over from the base.
- **A reply that can't be read is still an answer.** When a role's wren finishes but its reply
  isn't the JSON asked for, nothing is retried: a digest is kept as unparsed text, which the analyst
  still reads; an analyst without a report leaves the proposer to work from the digests, and
  `report.md` says so; a critic that can't be read three times rejects the candidate.
- **Cut-off rounds are run again**: a failed `git push` is retried after 30 s. A round cut off by
  git, by trials the infrastructure failed, or by a wren that couldn't run is run again after
  `retry_wait` = 10 minutes. After `round_retries` = 3 such failures in a row, the run stops.
  Anything else, such as a bug, stops it at once.
- **Measurements resume** where they stopped, running only the trials without a result, and a trial
  whose session log goes quiet for 20 minutes is failed as stalled and run again. See
  [benchmarks.md](benchmarks.md) for this and for spreading measurements over several Modal accounts.
- **`STOP`**: touching `bench/runs/rsi/<run>/STOP` stops the run before its next round.
- **`--no-eval`** analyzes and proposes without measuring, to look at the candidates before
  spending on them.

## Reference

### Files

```
bench/
  config.json                 settings: the measurement and selection rule, and "rsi" for the loop
  split.json                  the evolve and held-out tasks
  prompts/                    constitution, patterns, propose, repair, critic, digest, contrast, analyst
  runs/
    evals/<name>.json         measurements, trial by trial
    calibration.json          δ
    frontier.json             the incumbent and S*
    ledger.jsonl              every verdict
    rsi/<run>/
      state.json              T, the next round t, the incumbent's S after each round, the pinned incumbent
      history.jsonl           every edit tried
      analysis.json           the failure modes and habits named so far
      STOP                    stop before the next round
      t<t>/
        traces/               rendered trials (and earlier candidates' trials and diffs, for contrasts)
        digests/              one JSON per digest
        analyst-<step>.json   the analyst's turns
        analysis.json         the analyst's report
        report.md             what the proposer reads
        context.md            the round's directives
        v<i>/                 candidate.json, diff.patch, critic-<attempt>.json, propose.log, critic.log
~/.wren/rsi/<run>/t<t>-v<i>/  each candidate's worktree, removed when its round ends
```

Branches: `evolve-cand/<run>/t<t>-v<i>` for each candidate, `evolve/<run>` for the run's incumbent.

### A history record

One line per edit in `history.jsonl`:

| Field | |
|---|---|
| `t`, `variant`, `edit_id` | where the edit came from |
| `component`, `declared_component` | what the diff shows, and what was declared if it differed |
| `hypothesis`, `predicted_affected` | as declared |
| `bundle`, `sha` | how many edits shared the candidate, and its commit |
| `delta_S`, `delta_C`, `S` | the candidate's measurement against the incumbent; empty if never measured |
| `accepted`, `outcome` | `accepted`, `admissible`, `rejected`, `invalid`, or how it was dropped (in screening, or in the container) |
| `detail` | the verdict's reason, or why the candidate was dropped |
| `predicted_hit`, `hit_rate` | the predicted tasks that improved, and their share |
| `improved`, `unpredicted_regressions` | every task the candidate improved, and those it fell on (by at least one trial in k) without predicting them |

### Settings

In `bench/config.json`. Measurement and selection:

| Setting | Default | |
|---|---|---|
| `model` | `deepseek-v4.1-flash-1017` | the frozen policy model |
| `agent_kwargs` | `time_limit` 170, `max_turns` 400 | how wren runs in each trial |
| `k` | 2 | trials per task |
| `invalid_missing_frac` | 0.2 | a measurement with more ungraded trials than this is not used |
| `delta` | null | the noise band; null takes it from `calibration.json` |
| `delta_z` | 2.0 | z in δ = z · sd |
| `beta0`, `beta1` | 0.10, 5.0 | the token budget of a gain above δ |
| `w_s`, `w_c` | 100, 15 | the rule inside the band |
| `resources` | `limit` | Pier's sandbox sizing; measurements that are compared must share it |

The loop, under `"rsi"`:

| Setting | Default | |
|---|---|---|
| `T` | 8 | rounds |
| `candidates` | 2 | candidates per round |
| `b_max`, `b_min` | 3, 1 | the edit budget in the first and the last round |
| `stall_window` | 2 | rounds without a gain above δ before a candidate must explore |
| `n_prune` | 4 | rounds a component's recent gains are looked back over |
| `max_digests` | 12 | digests per round before the analyst's follow-ups |
| `stable_digests` | 3 | of those, at most, long runs of tasks that always pass |
| `contrast_digests` | 3 | of those, at most, contrasts with earlier candidates |
| `max_followups`, `analyst_rounds` | 6, 2 | the analyst's follow-up digests, and its turns |
| `repair_rounds` | 2 | attempts to fix a candidate that fails screening |
| `proposer_model`, `digest_model`, `analyst_model`, `critic_model` | null | null: wren's default model; keep them apart from the policy model |
| `proposer_time_limit`, `digest_time_limit`, `analyst_time_limit` | 45, 8, 15 | minutes |
| `wren_retries` | 2 | retries of a read-only role whose wren couldn't run |
| `push_retries` | 3 | retries of a failed push |
| `round_retries`, `retry_wait` | 3, 600 | cut-off rounds run again in a row, and the pause before each, in seconds |
