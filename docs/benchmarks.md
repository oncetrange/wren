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
actually helps. One trial says little: a task that passes once can fail the next time. So changes
are measured on a fixed evolve split of DeepSWE (`bench/split.json`, 30 tasks) with k trials per
task, and judged against a noise band measured by running the same harness twice:

```bash
python bench/evolve.py eval main                  # twice: two measurements of the same harness
python bench/evolve.py calibrate <name> <name>    # noise band delta
python bench/evolve.py baseline <name>            # the incumbent
python bench/evolve.py eval my-branch             # a pushed commit; or: eval main --ak mask_at=80000
python bench/evolve.py judge <name> --component context_mgmt --hypothesis "..."
python bench/evolve.py status                     # incumbent, delta, ledger
python bench/evolve.py eval <ref> --set heldout   # the other 83 tasks: reported, never used to choose
```

A candidate is accepted only if it doesn't fall below the best score so far minus delta, and pays
for any extra tokens with a gain larger than delta; inside the band only a token saving gets in,
and it must also pay for any score lost. A trial without a grade counts as a failure, but a
measurement with more than a fifth of its trials ungraded had an infrastructure problem, and isn't
used until it is resumed. Every verdict goes into a ledger in `bench/runs/`. Settings are in
`bench/config.json`, including the frozen policy model the benchmark runs
(`deepseek-v4.1-flash-1017`). How the noise band and the rule work is in
[rsi.md](rsi.md#measuring-a-harness).

A measurement runs as the Pier job `jobs/bench-<name>`. If it is cut off (a dropped connection, a
closed laptop), the same `eval` command resumes that job, running only the trials without a result.
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
and screens changes to its own source, and `evolve.py`'s rule decides which ones stay. Each round
it analyzes how the incumbent failed, drafts a few candidates in their own worktrees, screens them
(their declaration, the repo's checks, a critic) and measures the survivors. [rsi.md](rsi.md)
explains the loop in full.

```bash
python bench/rsi.py start r1           # from the incumbent; needs `baseline` and `calibrate` first
python bench/rsi.py run r1             # rounds until T; resumable; touch bench/runs/rsi/r1/STOP to stop
python bench/rsi.py run r1 --rounds 1 --no-eval   # analyze and propose only, to look at the candidates
python bench/rsi.py status r1          # the incumbent's score per round and every edit tried
git fetch origin evolve/r1                             # when it ends, measure the start and the result
python bench/evolve.py eval <start> --set heldout      # on unseen tasks: their difference is the
python bench/evolve.py eval origin/evolve/r1 --set heldout   # run's result
```

`run` can be left alone: it resumes where it stopped, and runs again a round cut off by git, by
the infrastructure or by a wren that couldn't run, after 10 minutes and up to three times in a row.
The proposer, digesters, analyst and critic are the installed `wren`, not the one being changed,
with the models set under `rsi` in `bench/config.json`; keep those apart from the policy model.
Each round's analysis is in `bench/runs/rsi/<run>/t<t>/report.md`.
