from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from bench import evolve

CFG = {"beta0": 0.10, "beta1": 5.0, "w_s": 0.0, "w_c": 15.0}


def write_trial(job: Path, name: str, task: str, reward: float | None, tokens: int = 1000,
                version: str = "abc", kwargs: dict | None = None) -> None:
    d = job / name
    d.mkdir(parents=True)
    result = {
        "task_id": {"path": f"deep-swe/tasks/{task}"},
        "config": {"agent": {"model_name": "m", "kwargs": {"version": version, **(kwargs or {})}}},
        "agent_result": {"n_input_tokens": tokens, "n_output_tokens": 0},
        "verifier_result": None if reward is None else {"rewards": {"reward": reward, "partial": 0.5}},
        "exception_info": None if reward is not None else {"exception_type": "RuntimeError"},
    }
    (d / "result.json").write_text(json.dumps(result))


def test_collect_counts_missing_trials_as_zero(tmp_path: Path):
    job = tmp_path / "job"
    write_trial(job, "a__1", "a", 1)
    write_trial(job, "a__2", "a", 1)
    write_trial(job, "b__1", "b", None)          # crashed before grading
    write_trial(job, "x__1", "x", 1)             # not in the split: ignored
    per_task, harness = evolve.collect([job], ["a", "b", "c"])
    ev = {"k": 2, "per_task": per_task}
    agg = evolve.aggregate(ev, "reward")
    # a: 1, 1 | b: ungraded + an absent slot | c: never ran -> 6 slots, 2 passes
    assert agg["S"] == pytest.approx(2 / 6)
    assert agg["missing"] == 4 and agg["n"] == 6
    assert harness == {"sha": "abc", "kwargs": {}, "model": "m"}
    assert evolve.aggregate(ev, "partial")["S"] == pytest.approx(1.0 / 6)


def test_collect_refuses_mixed_harnesses(tmp_path: Path):
    job = tmp_path / "job"
    write_trial(job, "a__1", "a", 1, kwargs={"mask_at": 1})
    write_trial(job, "a__2", "a", 1, kwargs={"mask_at": 2})
    with pytest.raises(SystemExit):
        evolve.collect([job], ["a"])


def test_judge_floor_cost_rule_and_band():
    inc = {"S": 0.50, "C": 100.0}
    # A real gain pays for its tokens: +0.2 buys up to 10% + 5 * 0.2 = 110%.
    assert evolve.judge({"S": 0.70, "C": 200.0}, inc, 0.50, 0.1, CFG)["admissible"]
    assert not evolve.judge({"S": 0.70, "C": 220.0}, inc, 0.50, 0.1, CFG)["admissible"]
    # Inside the band only a token saving gets in (w_s = 0).
    assert evolve.judge({"S": 0.55, "C": 90.0}, inc, 0.50, 0.1, CFG)["admissible"]
    assert not evolve.judge({"S": 0.55, "C": 100.0}, inc, 0.50, 0.1, CFG)["admissible"]
    # Never below the best incumbent so far minus the band, however cheap.
    v = evolve.judge({"S": 0.45, "C": 10.0}, inc, 0.60, 0.1, CFG)
    assert not v["admissible"] and "floor" in v["reason"]


def test_calibrate_from_repeated_and_single_measurements():
    def ev(name: str, rewards: dict[str, list[float]]) -> dict:
        return {"name": name, "k": 2, "per_task": {
            t: [{"scores": {"reward": r}, "tokens": 1, "exception": None} for r in rs]
            for t, rs in rewards.items()}}
    a = ev("a", {"t1": [1, 0], "t2": [1, 1]})
    b = ev("b", {"t1": [0, 0], "t2": [1, 1]})
    cal = evolve.calibrate([a, b], "reward", z=2.0)
    assert cal["S_per_eval"] == [0.75, 0.5]
    # Pooled, t1 has four trials, one passing: plug-in variance 3/16, corrected by 4/3 to 1/4;
    # one measurement's S has variance 2 * 1/4 / 4^2, the difference of two twice that.
    assert cal["sd_null"] == pytest.approx(math.sqrt(2 * 2 * 0.25 / 16), abs=0.01)
    assert cal["sd_null_repeat"] == pytest.approx(0.25) and not cal["shared_noise_suspected"]
    # Every task lost a pass in the second measurement: more than independent trials
    # explain, as when something hit all of them. Flagged; the band stays the bootstrap's.
    shocked = evolve.calibrate([ev("x", {f"t{i}": [1, 1] for i in range(20)}),
                                ev("y", {f"t{i}": [1, 0] for i in range(20)})], "reward", z=2.0)
    assert shocked["shared_noise_suspected"] and shocked["sd_null"] < shocked["sd_null_repeat"] / 2
    # One coin-flip task of two: the bootstrap's plug-in variance (0.25 per trial)
    # is corrected to the unbiased 0.5, so sd_null = sqrt(2) * sqrt(0.5 / 2) / 2.
    flip = ev("f", {"t1": [1, 0], "t2": [1, 1]})
    assert evolve.calibrate([flip], "reward", z=1.0)["sd_null"] == pytest.approx(0.3536, abs=0.01)
    # Tasks that always pass or always fail carry no trial noise.
    steady = ev("s", {"t1": [1, 1], "t2": [0, 0]})
    assert evolve.calibrate([steady], "reward", z=2.0)["delta"] == 0


def test_a_measurement_cut_off_is_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path)
    monkeypatch.setattr(evolve, "EVALS", tmp_path / "evals")
    monkeypatch.setattr(evolve, "ACCOUNTS", tmp_path / "accounts.json")
    monkeypatch.setattr(evolve, "load_split", lambda: {"evolve": ["a", "b"]})
    cfg = {"dataset": "tasks", "env": "modal", "n_concurrent": 2, "model": "m", "agent_env": [],
           "agent_kwargs": {"time_limit": 170}}
    job = tmp_path / "jobs" / "bench-x"
    calls: list[list[str]] = []

    def pier(cmd, cwd, check, env):
        calls.append(cmd)
        if cmd[1] == "run":   # one trial graded, then the sandbox provider fails the other
            job.mkdir(parents=True)
            (job / "config.json").write_text(json.dumps({"agents": [
                {"model_name": "m", "kwargs": {"version": "abc", "time_limit": 170}}]}))
            write_trial(job, "a__1", "a", 1, kwargs={"time_limit": 170})
            write_trial(job, "b__1", "b", None, kwargs={"time_limit": 170})   # RuntimeError
        else:                 # -f drops the failed trial, which then runs again
            shutil.rmtree(job / "b__1")
            write_trial(job, "b__2", "b", 0, kwargs={"time_limit": 170})

    monkeypatch.setattr(evolve, "run_pier", lambda cmd, env, job_dir: pier(cmd, evolve.ROOT, False, env))
    with pytest.raises(SystemExit, match=r"1/2 trials finished \(RuntimeError on the rest\)"):
        evolve.run_eval("abc", {}, "evolve", 1, "x", cfg)
    assert not evolve.eval_path("x").exists()
    # The same call goes on with the job instead of starting over.
    ev = evolve.run_eval("abc", {}, "evolve", 1, "x", cfg)
    assert calls[1] == ["pier", "job", "resume", "-p", str(job), "-f", "RuntimeError"]
    assert ev is not None and evolve.aggregate(ev, "reward")["S"] == 0.5
    # A job of another harness under the name is never resumed into this measurement.
    evolve.eval_path("x").unlink()
    with pytest.raises(SystemExit, match="ran"):
        evolve.run_eval("def", {}, "evolve", 1, "x", cfg)
    # Nor one sized another way.
    with pytest.raises(SystemExit, match="resources"):
        evolve.run_eval("abc", {}, "evolve", 1, "x", cfg | {"resources": {"cpus": "limit", "memory": "limit"}})
    assert not (tmp_path / "accounts.json").exists()   # no accounts listed: Modal's active profile


def crash(job: Path, name: str, task: str, kind: str = "NonZeroAgentExitCodeError",
          msg: str = "Command failed (exit 1): wren -p ...", age: float = 0) -> None:
    write_trial(job, name, task, None)
    r = job / name / "result.json"
    r.write_text(json.dumps(json.loads(r.read_text()) | {
        "exception_info": {"exception_type": kind, "exception_message": msg}}))
    os.utime(r, (time.time() - age, time.time() - age))


def test_broken_reads_the_first_trials_to_finish(tmp_path: Path):
    job = tmp_path / "job"
    assert evolve.broken(job) is None
    crash(job, "a__1", "a", age=30)
    crash(job, "b__1", "b", kind="RuntimeError", msg="Agent install failed: pip exited 1", age=20)
    assert evolve.broken(job) is None   # too few finished to say
    crash(job, "c__1", "c", age=10)
    assert len(evolve.broken(job) or []) == 3
    # One of the first three ran: an infrastructure hiccup, not a broken harness.
    healthy = tmp_path / "healthy"
    crash(healthy, "a__1", "a", age=30)
    write_trial(healthy, "b__1", "b", 0)
    os.utime(healthy / "b__1" / "result.json", (time.time() - 20, time.time() - 20))
    crash(healthy, "c__1", "c", age=10)
    crash(healthy, "d__1", "d")
    assert evolve.broken(healthy) is None
    # A sandbox that failed is the infrastructure's, whatever came first.
    infra = tmp_path / "infra"
    for i, t in enumerate("abc"):
        crash(infra, f"{t}__1", t, kind="RuntimeError", msg="sandbox vanished", age=10 * i)
    assert evolve.broken(infra) is None


def test_a_broken_harness_stops_its_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path)
    monkeypatch.setattr(evolve, "EVALS", tmp_path / "evals")
    monkeypatch.setattr(evolve, "ACCOUNTS", tmp_path / "accounts.json")
    monkeypatch.setattr(evolve, "load_split", lambda: {"evolve": ["a", "b", "c", "d"]})
    cfg = {"dataset": "tasks", "env": "local", "n_concurrent": 4, "model": "m", "agent_env": [],
           "agent_kwargs": {}}
    job = tmp_path / "jobs" / "bench-x"
    calls: list[list[str]] = []

    def pier(cmd, env, job_dir):
        calls.append(cmd)
        job.mkdir(parents=True)
        (job / "config.json").write_text(json.dumps({"agents": [{"model_name": "m", "kwargs": {"version": "abc"}}]}))
        for i, t in enumerate("abc"):
            crash(job, f"{t}__1", t, age=30 - 10 * i)

    monkeypatch.setattr(evolve, "run_pier", pier)
    with pytest.raises(evolve.HarnessBroken, match="crashed the harness: NonZeroAgentExitCodeError"):
        evolve.run_eval("abc", {}, "evolve", 1, "x", cfg)
    # Running it again doesn't resume the job: the crashes would only repeat.
    with pytest.raises(evolve.HarnessBroken):
        evolve.run_eval("abc", {}, "evolve", 1, "x", cfg)
    assert len(calls) == 1 and not evolve.eval_path("x").exists()


def test_run_pier_interrupts_a_broken_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(evolve, "POLL_SECONDS", 0.1)
    job = tmp_path / "job"
    for i, t in enumerate("abc"):
        crash(job, f"{t}__1", t, age=30 - 10 * i)
    start = time.monotonic()
    evolve.run_pier([sys.executable, "-c", "import time; time.sleep(60)"], None, job)
    assert time.monotonic() - start < 30


def test_a_spent_modal_account_hands_the_job_to_the_next(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path)
    monkeypatch.setattr(evolve, "EVALS", tmp_path / "evals")
    monkeypatch.setattr(evolve, "ACCOUNTS", tmp_path / "accounts.json")
    monkeypatch.setattr(evolve, "load_split", lambda: {"evolve": ["a", "b"]})
    (tmp_path / "accounts.json").write_text(json.dumps({"profiles": ["me", "ann", "bob"],
                                                        "jobs": {"bench-y": "me"}}))
    cfg = {"dataset": "tasks", "env": "modal", "n_concurrent": 2, "model": "m", "agent_env": [],
           "agent_kwargs": {}, "resources": {"cpus": "limit", "memory": "limit"}}
    job = tmp_path / "jobs" / "bench-x"
    calls: list[tuple[list[str], str]] = []

    def spend_limit(name: str) -> None:
        write_trial(job, name, "b", None)
        r = json.loads((job / name / "result.json").read_text())
        r["exception_info"] = {"exception_type": "ResourceExhaustedError",
                               "exception_message": "Workspace ac-1 has exceeded its spend limit"}
        (job / name / "result.json").write_text(json.dumps(r))

    def pier(cmd, cwd, check, env):
        calls.append((cmd, env["MODAL_PROFILE"]))
        if cmd[1] == "run":
            job.mkdir(parents=True)
            (job / "config.json").write_text(json.dumps({
                "agents": [{"model_name": "m", "kwargs": {"version": "abc"}}],
                "environment": {"cpu_enforcement_policy": "limit", "memory_enforcement_policy": "limit"}}))
            write_trial(job, "a__1", "a", 1)
            spend_limit("b__1")
        elif len(calls) == 2:   # the next account is spent too
            shutil.rmtree(job / "b__1")
            spend_limit("b__2")
        else:
            shutil.rmtree(job / "b__2")
            write_trial(job, "b__3", "b", 1)

    monkeypatch.setattr(evolve, "run_pier", lambda cmd, env, job_dir: pier(cmd, evolve.ROOT, False, env))
    ev = evolve.run_eval("abc", {}, "evolve", 1, "x", cfg)
    assert ev is not None and evolve.aggregate(ev, "reward")["S"] == 1.0
    # A new job goes to the account given the fewest jobs; each spent one hands it on.
    assert [p for _, p in calls] == ["ann", "bob", "me"]
    run = calls[0][0]
    assert run[run.index("--cpus"):][:4] == ["--cpus", "limit", "--memory", "limit"]
    assert calls[1][0][-2:] == ["-f", "ResourceExhaustedError"]
    state = json.loads((tmp_path / "accounts.json").read_text())
    assert state["jobs"]["bench-x"] == "me" and set(state["spent"]) == {"ann", "bob"}
    # With every account spent, a measurement stops before starting Pier.
    evolve.mark_spent("me")
    with pytest.raises(SystemExit, match="every Modal account"):
        evolve.run_eval("abc", {}, "evolve", 1, "z", cfg)
    assert len(calls) == 3
