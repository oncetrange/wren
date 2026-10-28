# Benchmarks (Pier / Harbor)

wren ships a Pier/Harbor installed-agent adapter, so it runs on
[DeepSWE](https://github.com/datacurve-ai/deep-swe) and other Harbor-format benchmarks:

```bash
uv tool install datacurve-pier --with git+https://github.com/oncetrange/wren
git clone https://github.com/datacurve-ai/deep-swe
pier run -p deep-swe/tasks --env modal --n-tasks 10 --sample-seed 0 \
    --agent-import-path wren.integrations.pier_agent:WrenAgent \
    -m moonshot/kimi-k2.7-code --ae MOONSHOT_API_KEY=$MOONSHOT_API_KEY \
    --ak version=main --ak max_turns=150
```

`-m` takes a builtin model or one from your `~/.wren/config.toml` (by name or model id): the
adapter reads the definition on your machine and writes just that model into the container's
config, without keys; pass the key with `--ae KEY_ENV=...` as above. Pier imports the adapter
from the wren installed alongside it (`--with`), so update that install to get adapter changes.

The same experiment options are agent kwargs: `--ak mask_at=24000`, `--ak compact_at=100000`,
`--ak no_subagents=true`, and `--ak time_limit=MINUTES` (a little under the agent timeout).

Inside each task container the adapter installs wren from this repository at the given git ref,
runs it headless with network access limited to the model's API host, and reports tokens, cost,
steps and compactions back to Pier. Session logs are kept under the trial's `agent/wren/` logs.

## Measuring a harness change

`bench/evolve.py` decides whether a change to wren (a commit, or agent kwargs such as `mask_at`)
actually helps, using the selection rules of [RRSI](https://github.com/google-research/rrsi). One
trial says little: a task that passes once can fail the next time. So changes are measured on a
fixed evolve split of DeepSWE (`bench/split.json`, 30 tasks) with k trials per task, against a noise
band measured by running the same harness twice:

```bash
python bench/evolve.py eval main                  # twice: two measurements of the same harness
python bench/evolve.py calibrate <name> <name>    # noise band delta
python bench/evolve.py baseline <name>            # the incumbent
python bench/evolve.py eval my-branch             # a pushed commit; or: eval main --ak mask_at=80000
python bench/evolve.py judge <name> --component context_mgmt --hypothesis "..."
python bench/evolve.py status                     # incumbent, delta, ledger
```

The noise band comes from the trials of both measurements, pooled per task: how much S would move
if each task's trials came out differently, which is far steadier than the gap between the two
scores, a single draw. `calibrate` still reports that gap, and warns when it is more than twice
what the trials predict, a sign that something hit many trials of one measurement at once.

A trial without a grade counts as a failure. A candidate is accepted only if it doesn't fall
below the best score so far minus delta, and pays for any extra tokens with a gain larger than
delta. Inside the band a gain counts for nothing, since it may be noise: only a token saving gets
in, and it must also pay for any score it loses, or the cheapest way in would be an agent that
gives up sooner. Every verdict, rejections included, goes into
a ledger in `bench/runs/`. The other 83 tasks are held out: `eval <ref> --set heldout` reports a
result but is never used to choose. Settings are in `bench/config.json`, including the frozen policy
model the benchmark runs (`deepseek-v4.1-flash-1017`).

A measurement runs as the Pier job `jobs/bench-<name>`. If it is cut off (a dropped connection, a
closed laptop), the same `eval` command resumes that job, running only the trials without a result;
`rsi.py run` does the same for its candidates, and finishes a round cut off while being recorded.
Left alone, it also runs again a round cut off by trials the infrastructure failed or by a git
error (a push failing on the network), after 10 minutes and up to three times in a row.
The first trials of a measurement to finish double as a smoke test: if the first three all crashed
the harness (wren exiting non-zero in the container, or not installing), the job is stopped and
never resumed, since the rest would only crash too. A crash of the infrastructure doesn't count.
A trial whose session log goes quiet for 20 minutes (a model request that never returns, as when
the provider's quota runs out) fails as `AgentStalledError`, which a resume runs again.

Sandboxes are sized with Pier's `limit` policy (`resources` in the config): a trial reserves almost
nothing, may burst up to the task's spec, and is billed for what it uses, about a seventh of the
default, with the same rewards and tokens in a trial run. Measurements that are compared must use
the same policy, so a job started under another one is never resumed into a measurement. To spread
measurements over several Modal accounts, list their profile names in `bench/runs/accounts.json`
as `{"profiles": [...]}`: each job runs on one account, and moves to the next when that one hits
its spend limit.

## Letting wren improve its own harness

`bench/rsi.py` closes the loop: instead of you proposing each change, wren proposes, implements
and screens changes to its own source, and `evolve.py`'s rule decides which ones stay. It follows
the RRSI loop:

```bash
python bench/rsi.py start r1           # from the incumbent; needs `baseline` and `calibrate` first
python bench/rsi.py run r1             # rounds until T; resumable; touch bench/runs/rsi/r1/STOP to stop
python bench/rsi.py run r1 --rounds 1 --no-eval   # analyze and propose only, to look at the candidates
python bench/rsi.py status r1          # the incumbent's score per round and every edit tried
```

Each round:

1. **Analyze.** Failed trials of the incumbent are rendered to text, along with a passing trial on
   tasks that pass only sometimes, since the contrast says the most, and the longest runs of a few
   tasks that always pass: what those lean on is what a change must not break. A read-only wren digests each
   one into a blocker, quoted evidence and what was needed instead. An analyst, another read-only
   wren, then clusters the digests into failure modes ranked by impact, capability gaps (what the
   agent tried to do but couldn't) and success habits a change must not break. It may first ask
   for follow-up digests with questions of its own, and keeps the names of earlier rounds when a
   mechanism recurs. The report and the digests are in `bench/runs/rsi/<run>/t<t>/report.md`.
2. **Propose.** A few candidates are drafted, each in its own git worktree. wren edits its source
   there, adds tests and declares every independent edit in `RSI_EDITS.json`: its component, its
   hypothesis, the failure mode it targets, why a plain instruction wouldn't do, the condition it
   acts on, the evolve tasks it should flip, and a retroactive check (which cited failures it
   would have fixed, which success habits it could break, why it generalizes). Its rules are
   `bench/prompts/constitution.md`: how candidates are judged, hard rules (nothing task-specific,
   the grader and `src/wren/integrations/` untouched, which settings are injected from outside
   and so can't be changed), and wren's levers file by file. `bench/prompts/patterns.md` lists
   known mechanism families with their traps. The proposer also reads the edit history, so it
   doesn't redraw a hypothesis that was measured and failed, and a scoreboard of how past
   predictions fared: which predicted tasks improved and which tasks an edit broke without
   predicting them. Other limits:
   - an edit budget that shrinks over the run, from `b_max` edits per candidate to `b_min`;
   - after `stall_window` rounds without a gain above the noise band, an edit on a component
     never tried (other than memory and skill, which no trial can reward today);
   - an offer to remove accepted machinery that stopped paying.
3. **Screen.** A candidate is checked before it is measured:
   - its declaration. Each edit's component is then read from the diff: one the diff doesn't
     show is re-tagged to what it does, and an exploration slot counts only what the diff shows
     (a system-prompt edit can't pass as subagent work);
   - the repo's checks (ruff, pyright, pytest);
   - a scan of the diff for task names, grader files and credentials;
   - a critic that rejects logic aimed at particular tasks, a gamed grader, or changes it doesn't
     declare.

   A candidate that fails gets `repair_rounds` attempts to fix what was objected to, and is
   otherwise dropped. The drop is still recorded.
4. **Measure and select.** Survivors are pushed (as `evolve-cand/<run>/t<t>-v<i>`) and measured
   on the evolve split; one whose first trials crash the harness is dropped there. The best admissible one becomes the incumbent, and `evolve/<run>` moves to
   it. Every edit goes into `bench/runs/rsi/<run>/history.jsonl` with its measured gain and the
   tasks that moved, and every measured candidate goes into the ledger.

The proposer, digesters, analyst and critic are the installed `wren`, not the one being changed.
They run with `proposer_model`, `digest_model`, `analyst_model` and `critic_model` from the `rsi`
settings in `bench/config.json`. Keep these models apart from the frozen policy model the benchmark runs.
When the run ends, `evolve.py eval evolve/<run> --set heldout` gives the result on tasks the
loop never saw.
