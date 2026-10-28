"""Measure harness changes on DeepSWE with RRSI's selection rules.

A harness is a wren commit plus the agent kwargs it runs with. Changes are judged
on the fixed evolve split (bench/split.json) against the incumbent, never on a
single trial, and every verdict goes into a ledger, rejected ideas included.

    python bench/evolve.py eval main                  # run Pier on the evolve set, score it
    python bench/evolve.py eval main                  # again: a second measurement of the same harness
    python bench/evolve.py calibrate <name> <name>    # noise band delta from the trials of two measurements
    python bench/evolve.py baseline <name>            # make a measurement the incumbent
    python bench/evolve.py eval my-branch --ak mask_at=80000
    python bench/evolve.py judge <name> --component context_mgmt --hypothesis "..."
    python bench/evolve.py eval <ref> --set heldout   # report only, never used to choose
    python bench/evolve.py score <name> jobs/<dir>... # score Pier jobs run by hand
    python bench/evolve.py status

The commit must be on GitHub (pushed), since the container installs wren from
there. Settings are in bench/config.json; measurements, the ledger and the
incumbent are kept in bench/runs/.

On Modal, measurements can be spread over several accounts: list their profile
names (from `modal token set --profile`) in bench/runs/accounts.json as
{"profiles": [...]}. A job stays on its account until that one hits its spend
limit, then resumes on the next; the file also records which job ran where and
which accounts are spent this month.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
RUNS = BENCH / "runs"
EVALS = RUNS / "evals"
LEDGER = RUNS / "ledger.jsonl"
FRONTIER = RUNS / "frontier.json"
CALIBRATION = RUNS / "calibration.json"
ACCOUNTS = RUNS / "accounts.json"
AGENT = "wren.integrations.pier_agent:WrenAgent"

# What a change touches, so the ledger shows which parts of the harness have paid off.
COMPONENTS = ["prompt", "control_flow", "config", "context_mgmt", "tools", "subagent", "skill", "memory"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, data: dict) -> None:
    """Write `path` whole or not at all, so a run cut off mid-write can be resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


# --- reading Pier jobs ---

def read_trial(trial_dir: Path) -> dict | None:
    """One trial's grade, tokens and harness, or None if Pier left no result."""
    path = trial_dir / "result.json"
    if not path.exists():
        return None
    r = load_json(path)
    agent = (r.get("config") or {}).get("agent") or {}
    kwargs = dict(agent.get("kwargs") or {})
    usage = r.get("agent_result") or {}
    tokens = (usage.get("n_input_tokens") or 0) + (usage.get("n_output_tokens") or 0)
    task_path = (r.get("task_id") or {}).get("path") or r.get("task_name") or ""
    return {
        "task": task_path.rstrip("/").split("/")[-1],
        "version": kwargs.pop("version", None),
        "kwargs": kwargs,
        "model": agent.get("model_name"),
        "rewards": (r.get("verifier_result") or {}).get("rewards"),
        "tokens": tokens or None,
        "exception": (r.get("exception_info") or {}).get("exception_type"),
    }


def collect(job_dirs: list[Path], tasks: list[str]) -> tuple[dict, dict]:
    """Trials per task from Pier job directories, and the harness they all ran.
    Trials of tasks outside `tasks` are ignored; trials of different harnesses
    can't be mixed into one measurement."""
    per_task: dict[str, list[dict]] = {t: [] for t in tasks}
    harness = None
    for job in job_dirs:
        for trial_dir in sorted(p for p in job.iterdir() if p.is_dir()):
            trial = read_trial(trial_dir)
            if trial is None or trial["task"] not in per_task:
                continue
            h = {"sha": trial["version"], "kwargs": trial["kwargs"], "model": trial["model"]}
            if harness is None:
                harness = h
            elif h != harness:
                raise SystemExit(f"{trial_dir}: ran {h}, other trials ran {harness}")
            rewards = trial["rewards"]
            per_task[trial["task"]].append({
                "scores": None if rewards is None else
                {m: rewards.get(m) for m in ("reward", "partial", "f2p", "p2p")},
                "tokens": trial["tokens"],
                "exception": trial["exception"],
                "dir": str(trial_dir),
            })
    if harness is None:
        raise SystemExit(f"no trials of these tasks in {[str(j) for j in job_dirs]}")
    return per_task, harness


# --- the estimate: S and C ---

def task_values(trials: list[dict], k: int, metric: str) -> list[float]:
    """One value per trial slot: at least k slots, an ungraded or missing trial
    is 0, so a harness can't look better by crashing on the tasks it finds hard."""
    values = [float((t["scores"] or {}).get(metric) or 0.0) for t in trials]
    return values + [0.0] * max(0, k - len(values))


def n_missing(trials: list[dict], k: int) -> int:
    return sum(1 for t in trials if t["scores"] is None) + max(0, k - len(trials))


def aggregate(ev: dict, metric: str) -> dict:
    """S, C and missing trials of a measurement under `metric`."""
    values = [v for trials in ev["per_task"].values() for v in task_values(trials, ev["k"], metric)]
    tokens = [t["tokens"] for trials in ev["per_task"].values() for t in trials if t["tokens"]]
    return {
        "S": sum(values) / len(values) if values else 0.0,
        "C": sum(tokens) / len(tokens) if tokens else None,
        "n": len(values),
        "missing": sum(n_missing(trials, ev["k"]) for trials in ev["per_task"].values()),
    }


def bootstrap_se(evals: list[dict], metric: str, reps: int = 2000, seed: int = 7) -> float:
    """Standard error of S from trial-to-trial noise alone: trials are resampled
    within each task, the tasks themselves stay fixed (every harness runs the same ones)."""
    rng = random.Random(seed)
    tasks: dict[str, list[float]] = {}
    for ev in evals:
        for task, trials in ev["per_task"].items():
            tasks.setdefault(task, []).extend(task_values(trials, ev["k"], metric))
    samples = []
    for _ in range(reps):
        total = n = 0
        for values in tasks.values():
            total += sum(rng.choice(values) for _ in values)
            n += len(values)
        samples.append(total / n if n else 0.0)
    return statistics.pstdev(samples)


def calibrate(evals: list[dict], metric: str, z: float) -> dict:
    """delta = z * sd of the difference between two measurements of the same
    harness, bootstrapped from the trials of all of them pooled per task (one
    needs k >= 2). The spread of their scores is reported beside it but not
    used: two scores are a single draw of the difference, while every task that
    flips adds to the bootstrap. It is the one estimate that would see noise
    shared across trials (a slow provider), so a spread far above the bootstrap
    is flagged."""
    scores = [aggregate(ev, metric)["S"] for ev in evals]
    se = bootstrap_se(evals, metric)
    # The bootstrap sees each task's plug-in variance, low by n/(n-1) for n
    # trials per task: half the truth at k = 2. And the pooled se is for
    # len(evals) times the trials of one measurement.
    n = sum(ev["k"] for ev in evals)
    sd_boot = math.sqrt(2) * se * math.sqrt(len(evals)) * math.sqrt(n / (n - 1))
    sd_repeat = statistics.stdev(scores) * math.sqrt(2) if len(scores) >= 2 else None
    return {"delta": round(z * sd_boot, 4), "z": z, "metric": metric, "sd_null": round(sd_boot, 4),
            "sd_null_repeat": None if sd_repeat is None else round(sd_repeat, 4),
            "shared_noise_suspected": sd_repeat is not None and sd_repeat > 2 * sd_boot,
            "S_per_eval": [round(s, 4) for s in scores], "evals": [ev["name"] for ev in evals],
            "method": "bootstrap over trials"}


# --- selection: RRSI Algorithm 2 ---

def judge(cand: dict, inc: dict, s_star: float, delta: float, cfg: dict) -> dict:
    """Is the candidate admissible against the incumbent? cand and inc are
    aggregates ({S, C}); s_star is the best incumbent score so far."""
    d_s = cand["S"] - inc["S"]
    d_c = (cand["C"] - inc["C"]) / inc["C"] if cand["C"] and inc["C"] else 0.0
    out = {"delta_S": round(d_s, 4), "delta_C": round(d_c, 4), "admissible": False}
    if cand["S"] < s_star - delta:
        out["reason"] = f"below the floor: S' {cand['S']:.4f} < S* {s_star:.4f} - delta {delta:.4f}"
    elif d_s > delta:
        budget = cfg["beta0"] + cfg["beta1"] * d_s
        out["admissible"] = d_c <= budget
        out["reason"] = (f"gain {d_s:+.4f} > delta {delta:.4f}; tokens {d_c:+.1%} "
                         f"{'within' if out['admissible'] else 'over'} budget {budget:+.1%}")
    else:
        shaped = cfg["w_s"] * d_s - cfg["w_c"] * d_c
        out["admissible"] = shaped > 0
        out["reason"] = (f"gain {d_s:+.4f} within delta {delta:.4f}; "
                         f"{cfg['w_s']}*dS - {cfg['w_c']}*dC = {shaped:+.4f} "
                         f"{'>' if out['admissible'] else '<='} 0")
    return out


# --- state ---

def load_config() -> dict:
    return load_json(BENCH / "config.json")


def load_split() -> dict:
    return load_json(BENCH / "split.json")


def eval_path(name: str) -> Path:
    return EVALS / f"{name}.json"


def load_eval(name: str) -> dict:
    path = eval_path(name)
    if not path.exists():
        raise SystemExit(f"no measurement {name!r} in {EVALS}")
    return load_json(path)


def delta_of(cfg: dict) -> float:
    if cfg["delta"] is not None:
        return float(cfg["delta"])
    if CALIBRATION.exists():
        return float(load_json(CALIBRATION)["delta"])
    raise SystemExit("no noise band: run `calibrate` on two measurements of the same harness, "
                     "or set delta in bench/config.json")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def resolve_ref(ref: str) -> str:
    """The commit `ref` names, checked to be on GitHub (the container installs it from there)."""
    try:
        sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"unknown ref {ref!r}") from None
    if not git("branch", "-r", "--contains", sha):
        raise SystemExit(f"{sha[:7]} isn't on any remote branch: push it first")
    return sha


def unique_name(base: str) -> str:
    name, i = base, 2
    while eval_path(name).exists():
        name, i = f"{base}-{i}", i + 1
    return name


def describe(ev: dict, metric: str) -> str:
    a = aggregate(ev, metric)
    c = f"{a['C'] / 1e6:.2f}M" if a["C"] else "?"
    kw = " ".join(f"{k}={v}" for k, v in sorted(ev["harness"]["kwargs"].items()))
    return (f"{ev['name']}: {(ev['harness']['sha'] or '?')[:7]} {kw} | {ev['set']} {len(ev['per_task'])} tasks "
            f"x k={ev['k']} | {metric} S={a['S']:.4f} C={c} tokens/trial | missing {a['missing']}/{a['n']}")


def build_eval(name: str, job_dirs: list[Path], set_name: str, k: int) -> dict:
    tasks = load_split()[set_name]
    per_task, harness = collect(job_dirs, tasks)
    ev = {"name": name, "set": set_name, "k": k, "harness": harness,
          "jobs": [str(j.relative_to(ROOT) if j.is_relative_to(ROOT) else j) for j in job_dirs],
          "created": time.strftime("%Y-%m-%d %H:%M:%S"), "per_task": per_task}
    save_json(eval_path(name), ev)
    return ev


def print_tasks(ev: dict, metric: str) -> None:
    for task, trials in sorted(ev["per_task"].items()):
        values = task_values(trials, ev["k"], metric)
        marks = [("-" if t["scores"] is None else f"{(t['scores'].get(metric) or 0):.2f}") for t in trials]
        marks += ["-"] * max(0, ev["k"] - len(trials))
        errors = sorted({t["exception"] for t in trials if t["exception"]})
        print(f"  {task:42} {sum(values) / len(values):.2f}  [{' '.join(marks)}]"
              + (f"  {', '.join(errors)}" if errors else ""))


# --- commands ---

def job_harness(job_dir: Path) -> dict:
    """The harness a Pier job was started with, from its config.json."""
    agent = load_json(job_dir / "config.json")["agents"][0]
    kwargs = {k: str(v) for k, v in (agent.get("kwargs") or {}).items()}
    return {"sha": kwargs.pop("version", None), "kwargs": kwargs, "model": agent.get("model_name")}


# What Pier counts as a trial's own outcome (it never retries these); any other
# exception is the infrastructure failing (a spend limit, a sandbox that vanished).
OUTCOME_EXCEPTIONS = {"AgentTimeoutError", "VerifierTimeoutError", "VerifierOutputParseError",
                      "RewardFileEmptyError", "RewardFileNotFoundError"}


# The first trials of a measurement to finish double as a smoke test: when all of them
# crashed the harness itself (wren exiting non-zero, or not installing), the candidate is
# broken in the container and the rest of the job would be spent learning nothing.
SMOKE_TRIALS = 3
POLL_SECONDS = 30


class HarnessBroken(Exception):
    """A measurement stopped because the harness crashes in the container."""


def agent_crash(result: dict) -> str | None:
    """How a trial's harness crashed, or None if it didn't: Pier's error when the
    agent command exits non-zero, or when installing the agent fails."""
    exc = result.get("exception_info") or {}
    kind, msg = exc.get("exception_type"), exc.get("exception_message") or ""
    if kind == "NonZeroAgentExitCodeError" or (kind == "RuntimeError" and msg.startswith("Agent install failed")):
        return f"{kind}: {msg[:400]}"
    return None


def broken(job_dir: Path) -> list[str] | None:
    """The crashes, if the first SMOKE_TRIALS trials of the job to finish all crashed."""
    done = sorted((p / "result.json" for p in job_dir.iterdir() if (p / "result.json").exists())
                  if job_dir.exists() else [], key=lambda r: r.stat().st_mtime)[:SMOKE_TRIALS]
    crashes = [c for r in done if (c := agent_crash(load_json(r)))]
    return crashes if len(done) == SMOKE_TRIALS and len(crashes) == SMOKE_TRIALS else None


def run_pier(cmd: list[str], env: dict | None, job_dir: Path) -> None:
    """Run Pier, interrupting it (as Ctrl-C does, so it cancels its sandboxes) once
    the job shows the harness is broken."""
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    try:
        while proc.poll() is None:
            try:
                proc.wait(POLL_SECONDS)
            except subprocess.TimeoutExpired:
                if broken(job_dir):
                    proc.send_signal(signal.SIGINT)
                    proc.wait(300)
    except BaseException:
        proc.kill()
        raise


def job_progress(job_dir: Path) -> tuple[int, set[str]]:
    """Trials of a Pier job that finished (graded, or failed on their own), and
    the exception types of those the infrastructure failed."""
    done, infra = 0, set()
    for p in job_dir.iterdir() if job_dir.exists() else []:
        if not (p / "result.json").exists():
            continue
        exc = (load_json(p / "result.json").get("exception_info") or {}).get("exception_type")
        if exc and exc not in OUTCOME_EXCEPTIONS:
            infra.add(exc)
        else:
            done += 1
    return done, infra


def out_of_credit(job_dir: Path) -> bool:
    """Whether a trial of the job was refused because the Modal account hit its spend limit."""
    for p in job_dir.iterdir() if job_dir.exists() else []:
        if (p / "result.json").exists():
            exc = load_json(p / "result.json").get("exception_info") or {}
            if exc.get("exception_type") == "ResourceExhaustedError" and "spend limit" in (
                    exc.get("exception_message") or ""):
                return True
    return False


@contextlib.contextmanager
def accounts_state():
    """bench/runs/accounts.json, locked against measurements running at once:
    {"profiles": [...], "jobs": {job: profile}, "spent": {profile: "YYYY-MM"}}."""
    ACCOUNTS.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCOUNTS.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = {"profiles": [], "jobs": {}, "spent": {}}
        if ACCOUNTS.exists():
            state |= load_json(ACCOUNTS)
        yield state
        if ACCOUNTS.exists() or state["jobs"]:
            save_json(ACCOUNTS, state)


def pick_account(job: str) -> tuple[str | None, list[str]]:
    """The Modal profile a job runs on: the one it started on while that has credit
    this month, else the one with credit that was given the fewest jobs. None
    with no profiles listed (Modal's active one) or none left; then the list."""
    month = time.strftime("%Y-%m")
    with accounts_state() as state:
        live = [p for p in state["profiles"] if state["spent"].get(p) != month]
        if state["jobs"].get(job) not in live and live:
            load = Counter(state["jobs"].values())
            state["jobs"][job] = min(live, key=lambda p: load[p])
        return (state["jobs"][job] if live else None), state["profiles"]


def mark_spent(profile: str) -> None:
    with accounts_state() as state:
        state["spent"][profile] = time.strftime("%Y-%m")


def run_eval(sha: str, overrides: dict, set_name: str, k: int, name: str, cfg: dict,
             dry_run: bool = False) -> dict | None:
    """Run Pier on a split with the harness (sha, config kwargs + overrides) and
    save the measurement as `name`; None on a dry run.

    The job is jobs/bench-<name>. If it was cut off (no measurement saved yet),
    the same call resumes it: Pier runs only the trials without a result. One
    whose Modal account runs out of credit resumes on the next account. Raises
    HarnessBroken, and never resumes the job, when its first trials all crashed
    the harness."""
    kwargs = {**cfg["agent_kwargs"], **overrides}
    if "version" in kwargs:
        raise SystemExit("the version is the ref; don't pass it with --ak")
    tasks = load_split()[set_name]
    total = len(tasks) * k
    job_dir = ROOT / "jobs" / f"bench-{name}"
    # How Pier sizes sandboxes: "limit" reserves almost nothing and lets a trial
    # burst up to the task's spec, billing what it uses (about 1/7 of "auto").
    resources = {"cpus": "auto", "memory": "auto"} | cfg.get("resources", {})
    if crashes := broken(job_dir):
        raise HarnessBroken(f"{name}: the first {SMOKE_TRIALS} trials of {job_dir.relative_to(ROOT)} "
                            f"crashed the harness: {crashes[0]}")
    if (job_dir / "config.json").exists():
        want = {"sha": sha, "kwargs": {key: str(v) for key, v in kwargs.items()}, "model": cfg["model"]}
        if (ran := job_harness(job_dir)) != want:
            raise SystemExit(f"{job_dir} ran {ran}, not {want}: remove it or measure under another name")
        env = load_json(job_dir / "config.json").get("environment") or {}
        ran_res = {"cpus": env.get("cpu_enforcement_policy", "auto"),
                   "memory": env.get("memory_enforcement_policy", "auto")}
        if ran_res != resources:
            raise SystemExit(f"{job_dir} ran with resources {ran_res}, not {resources}: "
                             "remove it or measure under another name")
        done, infra = job_progress(job_dir)
        print(f"{name}: resuming {job_dir.relative_to(ROOT)}, {done}/{total} trials done"
              + (f", rerunning those that failed with {', '.join(sorted(infra))}" if infra else ""))
        cmd = resume_command(job_dir, infra)
    else:
        cmd = ["pier", "run", "-p", cfg["dataset"], "-e", cfg["env"], "-o", "jobs", "--job-name", job_dir.name,
               "-k", str(k), "-n", str(cfg["n_concurrent"]), "-y",
               "--cpus", resources["cpus"], "--memory", resources["memory"],
               "--agent-import-path", AGENT, "-m", cfg["model"], "--ak", f"version={sha}"]
        cmd += [x for key, v in kwargs.items() for x in ("--ak", f"{key}={v}")]
        # Pier resolves ${VAR} itself, so the key never lands in the job config.
        cmd += [x for var in cfg["agent_env"] for x in ("--ae", f"{var}=${{{var}}}")]
        cmd += [x for t in tasks for x in ("-i", t)]
        print(f"{name}: {len(tasks)} tasks x k={k} = {total} trials of {sha[:7]}")
    print(" ".join(cmd[:24]) + (" ..." if len(cmd) > 24 else ""))
    if dry_run:
        return None
    while True:
        env = None
        if cfg["env"] == "modal":
            profile, profiles = pick_account(job_dir.name)
            if profiles and profile is None:
                raise SystemExit(f"{name}: every Modal account ({', '.join(profiles)}) is spent this month; "
                                 f"drop one from \"spent\" in {ACCOUNTS.relative_to(ROOT)} once it has credit")
            if profile:
                print(f"{name}: on Modal account {profile}")
                env = os.environ | {"MODAL_PROFILE": profile}
        else:
            profile = None
        run_pier(cmd, env, job_dir)
        if crashes := broken(job_dir):
            raise HarnessBroken(f"{name}: the first {SMOKE_TRIALS} trials of {job_dir.relative_to(ROOT)} "
                            f"crashed the harness: {crashes[0]}")
        done, infra = job_progress(job_dir)
        if done >= total:
            return build_eval(name, [job_dir], set_name, k)
        if profile and out_of_credit(job_dir):
            mark_spent(profile)
            print(f"{name}: Modal account {profile} hit its spend limit at {done}/{total} trials")
            cmd = resume_command(job_dir, infra)
            continue
        failed = f" ({', '.join(sorted(infra))} on the rest)" if infra else ""
        raise SystemExit(f"{name}: {done}/{total} trials finished{failed}; the same command resumes "
                         f"{job_dir.relative_to(ROOT)} (or `score {name} {job_dir.relative_to(ROOT)}` "
                         "takes what is there)")


def resume_command(job_dir: Path, infra: set[str]) -> list[str]:
    """Pier resuming a job, running again the trials the infrastructure failed."""
    return ["pier", "job", "resume", "-p", str(job_dir)] + [x for e in sorted(infra) for x in ("-f", e)]


def cmd_eval(args, cfg: dict) -> None:
    sha = resolve_ref(args.ref)
    overrides = dict(item.partition("=")[::2] for item in args.ak)
    # The first name without a measurement: if its job was cut off, run_eval resumes it.
    name = unique_name(args.name or f"{sha[:7]}-{args.set}")
    try:
        ev = run_eval(sha, overrides, args.set, args.k or cfg["k"], name, cfg, args.dry_run)
    except HarnessBroken as e:
        raise SystemExit(f"{e}\nfix the harness and measure the fix; if the infrastructure caused "
                         "the crashes, remove the job to measure this one again") from None
    if ev is not None:
        print(describe(ev, cfg["metric"]))
        print_tasks(ev, cfg["metric"])


def cmd_score(args, cfg: dict) -> None:
    if eval_path(args.name).exists():
        raise SystemExit(f"measurement {args.name!r} exists")
    ev = build_eval(args.name, [Path(j).resolve() for j in args.jobs], args.set, args.k or cfg["k"])
    print(describe(ev, cfg["metric"]))
    print_tasks(ev, cfg["metric"])


def cmd_show(args, cfg: dict) -> None:
    ev = load_eval(args.name)
    metric = args.metric or cfg["metric"]
    print(describe(ev, metric))
    print_tasks(ev, metric)


def cmd_calibrate(args, cfg: dict) -> None:
    evals = [load_eval(n) for n in args.names]
    if len({json.dumps(ev["harness"], sort_keys=True) for ev in evals}) > 1:
        raise SystemExit("calibration needs measurements of one and the same harness")
    if len(evals) == 1 and evals[0]["k"] < 2:
        raise SystemExit("one measurement with k=1 says nothing about noise: measure again, or use k >= 2")
    cal = calibrate(evals, cfg["metric"], cfg["delta_z"])
    save_json(CALIBRATION, cal)
    print(f"delta = {cal['delta']:.4f} ({cal['method']}; S per measurement {cal['S_per_eval']})")
    if cal["shared_noise_suspected"]:
        print(f"warning: the measurements differ by more than twice the trial noise predicts "
              f"(sd {cal['sd_null_repeat']:.4f} vs {cal['sd_null']:.4f}): look for something that "
              f"hit many trials of one of them (provider slowdowns, timeouts), or measure again")


def cmd_baseline(args, cfg: dict) -> None:
    ev = load_eval(args.name)
    if ev["set"] != "evolve":
        raise SystemExit("the incumbent is measured on the evolve set")
    a = aggregate(ev, cfg["metric"])
    save_json(FRONTIER, {"incumbent": args.name, "S_star": a["S"], "metric": cfg["metric"]})
    append_ledger({"candidate": args.name, "harness": ev["harness"], "verdict": "BASELINE",
                   "S": round(a["S"], 4), "C": a["C"]})
    print(f"incumbent: {describe(ev, cfg['metric'])}")


def cmd_judge(args, cfg: dict) -> None:
    if not FRONTIER.exists():
        raise SystemExit("no incumbent: run `baseline` first")
    fr = load_json(FRONTIER)
    if fr["metric"] != cfg["metric"]:
        raise SystemExit(f"the incumbent was scored on {fr['metric']}, config says {cfg['metric']}")
    inc, cand = load_eval(fr["incumbent"]), load_eval(args.name)
    if cand["set"] != "evolve":
        raise SystemExit("only evolve-set measurements are judged; held-out ones are for reporting")
    if set(cand["per_task"]) != set(inc["per_task"]) or cand["harness"]["model"] != inc["harness"]["model"]:
        raise SystemExit("the candidate ran other tasks or another model than the incumbent")
    if args.component and args.component not in COMPONENTS:
        raise SystemExit(f"component must be one of {COMPONENTS}")
    delta = delta_of(cfg)
    a_inc, a_cand = aggregate(inc, cfg["metric"]), aggregate(cand, cfg["metric"])
    if a_cand["missing"] > cfg["invalid_missing_frac"] * a_cand["n"]:
        raise SystemExit(f"{a_cand['missing']}/{a_cand['n']} trials have no grade: an infrastructure "
                         f"problem, not a measurement. Run it again.")
    verdict = judge(a_cand, a_inc, fr["S_star"], delta, cfg)
    print(describe(inc, cfg["metric"]) + "  (incumbent)")
    print(describe(cand, cfg["metric"]))
    print(f"{'ACCEPTED' if verdict['admissible'] else 'REJECTED'}: {verdict['reason']}")
    if args.dry_run:
        return
    append_ledger({"candidate": args.name, "incumbent": fr["incumbent"], "harness": cand["harness"],
                   "component": args.component, "hypothesis": args.hypothesis,
                   "S": round(a_cand["S"], 4), "C": a_cand["C"], **verdict, "delta": delta,
                   "verdict": "ACCEPTED" if verdict["admissible"] else "REJECTED"})
    if verdict["admissible"]:
        save_json(FRONTIER, {"incumbent": args.name, "S_star": max(fr["S_star"], a_cand["S"]),
                             "metric": cfg["metric"]})


def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **record}, ensure_ascii=False) + "\n")


def cmd_status(args, cfg: dict) -> None:
    if FRONTIER.exists():
        fr = load_json(FRONTIER)
        print(f"incumbent: {describe(load_eval(fr['incumbent']), fr['metric'])}")
        print(f"S* = {fr['S_star']:.4f}")
    else:
        print("no incumbent yet")
    try:
        print(f"delta = {delta_of(cfg):.4f}")
    except SystemExit as e:
        print(e)
    if LEDGER.exists():
        print("ledger:")
        for line in LEDGER.read_text().splitlines()[-args.n:]:
            r = json.loads(line)
            gain = f" dS={r['delta_S']:+.4f} dC={r['delta_C']:+.1%}" if "delta_S" in r else ""
            print(f"  {r['ts'][:16]} {r['verdict']:8} {r['candidate']}{gain}"
                  f"  [{r.get('component') or '-'}] {r.get('hypothesis') or ''}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval", help="run Pier on a split and score it")
    p.add_argument("ref", help="git ref of wren to measure (must be pushed)")
    p.add_argument("--set", choices=["evolve", "heldout"], default="evolve")
    p.add_argument("--ak", action="append", default=[], metavar="KEY=VALUE", help="agent kwarg override")
    p.add_argument("--k", type=int, help="trials per task (default: config)")
    p.add_argument("--name", help="measurement name (default: <sha7>-<set>)")
    p.add_argument("--dry-run", action="store_true", help="print the Pier command only")
    p = sub.add_parser("score", help="score Pier jobs that were run by hand")
    p.add_argument("name")
    p.add_argument("jobs", nargs="+", help="Pier job directories (their trials are merged)")
    p.add_argument("--set", choices=["evolve", "heldout"], default="evolve")
    p.add_argument("--k", type=int, help="trials per task expected (default: config)")
    p = sub.add_parser("show", help="per-task results of a measurement")
    p.add_argument("name")
    p.add_argument("--metric", choices=["reward", "partial", "f2p", "p2p"])
    p = sub.add_parser("calibrate", help="noise band from measurements of one harness")
    p.add_argument("names", nargs="+")
    p = sub.add_parser("baseline", help="make a measurement the incumbent")
    p.add_argument("name")
    p = sub.add_parser("judge", help="accept or reject a measurement against the incumbent")
    p.add_argument("name")
    p.add_argument("--component", help=f"what the change touches: {', '.join(COMPONENTS)}")
    p.add_argument("--hypothesis", help="one sentence: the mechanism and why it should help")
    p.add_argument("--dry-run", action="store_true", help="show the verdict without recording it")
    p = sub.add_parser("status", help="incumbent, noise band and recent verdicts")
    p.add_argument("-n", type=int, default=15, help="ledger rows to show")
    args = ap.parse_args(argv)
    commands = {"eval": cmd_eval, "score": cmd_score, "show": cmd_show, "calibrate": cmd_calibrate,
                "baseline": cmd_baseline, "judge": cmd_judge, "status": cmd_status}
    commands[args.cmd](args, load_config())


if __name__ == "__main__":
    sys.exit(main())
