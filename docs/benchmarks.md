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
fixed evolve split of DeepSWE (`bench/split.json`, 10 tasks) with k trials per task, against a noise
band measured by running the same harness twice:

```bash
python bench/evolve.py eval main                  # twice: two measurements of the same harness
python bench/evolve.py calibrate <name> <name>    # noise band delta
python bench/evolve.py baseline <name>            # the incumbent
python bench/evolve.py eval my-branch             # a pushed commit; or: eval main --ak mask_at=80000
python bench/evolve.py judge <name> --component context_mgmt --hypothesis "..."
python bench/evolve.py status                     # incumbent, delta, ledger
```

A trial without a grade counts as a failure. A candidate is accepted only if it doesn't fall
below the best score so far minus delta, and pays for any extra tokens with a gain larger than
delta (inside the band, only a token saving gets in). Every verdict, rejections included, goes into
a ledger in `bench/runs/`. The other 103 tasks are held out: `eval <ref> --set heldout` reports a
result but is never used to choose. Settings are in `bench/config.json`.
