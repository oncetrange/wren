# Headless and scheduled runs

## Headless use

```bash
wren -p "fix the failing test" --yolo --output-format json   # one JSON object on stdout
echo "task text" | wren -p - --yolo --max-turns 50            # prompt from stdin
```

The exit code is 0 when the agent finished normally and 1 otherwise. `status` in the JSON says
how the run ended:

| status | meaning |
|---|---|
| `done` | the model finished |
| `blocked` | a `UserPromptSubmit` hook refused the prompt |
| `empty_response` | the model returned nothing |
| `refusal` | the model declined to continue |
| `max_tokens` | responses kept being cut off at the output limit before acting |
| `max_turns` | `--max-turns` was used up |
| `time_limit` | `--time-limit` was reached |
| `interrupted` | Ctrl-C |
| `error` | the model request failed for good, after retries |

Headless runs also have the model re-check the request's explicit
instructions (commit, run tests, ...) once before finishing a request that changed files, and warn
it when `--max-turns` is nearly used up (`--no-final-check` disables both; `--final-check` enables
them interactively). `-m` also accepts a model id such as `moonshot/kimi-k2.7-code`, and
`WREN_MODEL` sets the default.

`--time-limit MINUTES` bounds a run's wall-clock time: with a quarter of it left the model is
reminded to save its progress (e.g. commit what works), near the end to wrap up, and at the limit
wren stops (`status: time_limit`). Set it a little under a harness's own timeout, so the work is
committed before the harness kills the run.

For experiments, `--mask-at TOKENS` and `--compact-at TOKENS` override the model's context
thresholds and `--no-subagents` removes the task tool; the thresholds used are recorded at the
start of the session log.

## Scheduled runs

wren can run prompts on a cron schedule, headless, even when no session is open: "every weekday
at 9, check CI on main and summarize failures". Ask for one in a session (the model proposes the
job and you approve it; creating or deleting a job always asks, whatever the permission mode), or:

```bash
wren schedule add "0 9 * * 1-5" "check CI on main and summarize failures"   # runs in this directory
wren schedule add @daily "run the tests; fix failures on a branch" --mode accept_edits --id nightly
wren schedule install     # once: runs `wren schedule tick` every minute (launchd on macOS, else crontab)
wren schedule list | logs <id> | run <id> | pause <id> | resume <id> | rm <id> | status | uninstall
```

- Cron expressions have five fields in local time (`*/15`, `1-5`, `mon-fri`, `jan,jul`) or a
  shortcut (`@hourly`, `@daily`, `@weekly`, `@monthly`).
- Jobs run in plan mode (read-only) unless given `--mode accept_edits` or `--mode auto`; anything
  that would need approval is refused, since nobody is there to give it.
- Each run is a normal `wren -p` session: `wren schedule logs <id>` shows its result and cost,
  and `wren -r <session>` continues it interactively. A desktop notification reports each run.
- A run missed while the machine was asleep runs once when it wakes (`--no-catch-up` skips it
  instead); a run due while the previous one is still going is skipped.
- Scheduled runs don't see your shell's environment: put API keys in `~/.wren/env` as `KEY=value`
  lines (`chmod 600` it); `wren schedule install` tells you which are missing. Jobs are stored in
  `~/.wren/schedules.json`, runs in `~/.wren/schedules/<id>/`.

In a session, `/schedule` lists jobs to run now, pause or delete, and `/loop 10m <prompt>` repeats
a prompt every ten minutes while the session is open (Ctrl-C stops it).
