from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench import evolve, rsi

CFG = {"beta0": 0.10, "beta1": 5.0, "w_s": 0.0, "w_c": 15.0, "invalid_missing_frac": 0.2,
       "metric": "reward", "k": 2, "model": "policy", "delta": 0.05}
# A declared edit with every required field.
EDIT = {"id": "E1", "component": "prompt", "hypothesis": "h", "targets_mode": "m",
        "why_not_lower_lever": "w", "trigger_condition": "c", "predicted_affected": ["b"],
        "retroactive_check": "corrective; preservative; transfer"}


def trials(*rewards: float | None, tokens: int = 100, dirs: list[str] | None = None) -> list[dict]:
    return [{"scores": None if r is None else {"reward": r}, "tokens": tokens, "exception": None,
             **({"dir": dirs[i]} if dirs else {})} for i, r in enumerate(rewards)]


def test_edit_budget_anneals_from_b_max_to_b_min():
    table = [rsi.edit_budget(t, 8, 1, 3) for t in range(8)]
    assert table[0] == 3 and table[-1] == 2 and rsi.edit_budget(8, 8, 1, 3) == 1
    assert table == sorted(table, reverse=True)


def test_stall_untried_and_prune():
    assert rsi.stalled([0.5, 0.5, 0.52], 2, window=2, delta=0.05)
    assert not rsi.stalled([0.5, 0.5, 0.6], 2, window=2, delta=0.05)
    assert not rsi.stalled([0.5], 1, window=2, delta=0.05)        # too early to tell
    history = [
        {"t": 0, "component": "prompt", "delta_S": 0.1, "accepted": True},
        {"t": 1, "component": "tools", "delta_S": None},           # dropped in screening
        {"t": 5, "component": "prompt", "delta_S": -0.02, "accepted": False},
        {"t": 5, "component": "context_mgmt", "delta_S": 0.2, "accepted": True},
    ]
    assert "tools" in rsi.untried(history) and "prompt" not in rsi.untried(history)
    assert "memory" not in rsi.untried(history)   # no trial can reward it: never forced
    # prompt's machinery is in the incumbent, and its edits stopped paying in the last 4 rounds.
    assert [p["component"] for p in rsi.prune_set(history, 6, n_prune=4)] == ["prompt"]


def test_history_keeps_few_unmeasured_edits():
    history = [{"t": i, "edit_id": f"E{i}", "component": "tools", "delta_S": None} for i in range(6)]
    history.append({"t": 6, "edit_id": "E6", "component": "prompt", "delta_S": 0.1})
    lines = rsi.render_history(history).splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["edit_id"] == "E6"
    assert json.loads(lines[0])["edit_id"] == "E2"


def test_render_trace(tmp_path: Path):
    d = tmp_path / "trial"
    (d / "agent" / "wren" / "sessions").mkdir(parents=True)
    (d / "verifier").mkdir()
    log = [
        {"kind": "message", "role": "user", "content": [{"type": "text", "text": "fix the bug"}]},
        {"kind": "message", "role": "assistant", "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "pytest"}}]},
        {"kind": "message", "role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "1 failed", "is_error": True}]},
        {"kind": "compact", "summary": "s"},
    ]
    (d / "agent" / "wren" / "sessions" / "s.jsonl").write_text("\n".join(map(json.dumps, log)))
    (d / "verifier" / "test-stdout.txt").write_text("FAILED test_x\n")
    text = rsi.render_trace({"dir": str(d), "scores": {"reward": 0}}, "t1")
    assert '[call 1] bash {"command": "pytest"}' in text
    assert "[result 1 ERROR] 1 failed" in text and "[context compacted]" in text
    assert "FAILED test_x" in text and '"reward": 0' in text


def test_choose_trials_puts_contrasts_first():
    ev = {"k": 2, "per_task": {
        "always": trials(1, 1, dirs=["a1", "a2"]),
        "never": trials(0, 0, dirs=["n1", "n2"]),
        "sometimes": trials(0, 1, dirs=["s1", "s2"]),
        "unlogged": trials(0, 0),
    }}
    picked = [(c["task"], c["lens"], c["trial"]["dir"]) for c in rsi.choose_trials(ev, "reward", 10, 0)]
    assert picked == [("sometimes", "failure", "s1"), ("sometimes", "success", "s2"),
                      ("never", "failure", "n1"), ("never", "failure", "n2")]
    assert len(rsi.choose_trials(ev, "reward", 2, 0)) == 2


def test_choose_trials_keeps_room_for_the_longest_stable_passes():
    def passes(*tokens: int, task: str) -> list[dict]:
        return [{"scores": {"reward": 1}, "tokens": n, "exception": None, "dir": f"{task}{i}"}
                for i, n in enumerate(tokens)]
    ev = {"k": 2, "per_task": {
        **{f"f{i}": trials(0, 0, dirs=[f"f{i}a", f"f{i}b"]) for i in range(5)},
        "short": passes(10, 20, task="short"), "long": passes(900, 300, task="long"),
        "mid": passes(100, 50, task="mid"),
        "half_missing": passes(5000, task="half_missing"),   # a trial missing: not a stable pass
    }}
    picked = [(c["task"], c["lens"], c["trial"]["dir"]) for c in rsi.choose_trials(ev, "reward", 6, 2)]
    # Four failures, then the two longest stable passes, each by its longest trial; the
    # second failures of each task make way.
    assert picked[:4] == [(f"f{i}", "failure", f"f{i}a") for i in range(4)]
    assert picked[4:] == [("long", "success", "long0"), ("mid", "success", "mid0")]


def test_analyze_asks_for_follow_ups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rsi, "RSI", tmp_path / "rsi")
    inc = {"name": "base", "k": 2, "per_task": {
        "sometimes": trials(0, 1, dirs=[str(tmp_path / "s0"), str(tmp_path / "s1")]),
        "never": trials(0, 0, dirs=[str(tmp_path / "n0"), str(tmp_path / "n1")])}}
    rc = rsi.rsi_config({"rsi": {"max_digests": 2}})
    calls: list[str] = []

    def fake_wren(rc, text, cwd, *, model, yolo, time_limit, log):
        calls.append(text)
        if "batch analyst" not in text:
            return {"status": "done", "result": '{"blocker": "b"}'}
        if "last turn" not in text:
            return {"status": "done", "result": json.dumps({"follow_up": [
                {"task": "never", "lens": "capability_gap", "questions": ["was grep cut off?"]},
                {"task": "zz"}]})}
        return {"status": "done", "result": json.dumps({
            "failure_modes": [{"mode": "small", "n_tasks": 1}, {"mode": "big", "n_tasks": 2}],
            "success_habits": [{"habit": "runs_tests", "description": "d"}]})}

    monkeypatch.setattr(rsi, "run_wren", fake_wren)
    report = rsi.analyze("r1", 0, inc, CFG, rc).read_text()
    digests = sorted(p.name for p in (tmp_path / "rsi" / "r1" / "t0" / "digests").iterdir())
    # The contrast pair first, then the follow-up the analyst asked for (an unknown task is skipped).
    assert digests == ["never-0-capability_gap.json", "sometimes-0-failure.json", "sometimes-1-success.json"]
    assert any("Questions to answer:\n- was grep cut off?" in c for c in calls)
    assert "capability_gap of trial 0" in calls[-1]
    # Failure modes ranked by impact; the names carry over to the next round.
    assert report.index('"big"') < report.index('"small"') and "must not break" in report
    assert evolve.load_json(tmp_path / "rsi" / "r1" / "analysis.json")["success_habits"][0]["habit"] == "runs_tests"
    calls.clear()
    rsi.analyze("r1", 1, inc, CFG, rc)
    assert '"habit": "runs_tests"' in next(c for c in calls if "batch analyst" in c)
    # A resumed round reuses what it has: no wren runs.
    calls.clear()
    (tmp_path / "rsi" / "r1" / "t1" / "report.md").unlink()
    rsi.analyze("r1", 1, inc, CFG, rc)
    assert calls == []


def test_check_declaration():
    good, tasks = [EDIT], ["a", "b"]
    assert rsi.check_declaration(good, ["src/wren/agent/prompt.py"], 1, [], tasks) == []
    problems = rsi.check_declaration(good * 2, ["bench/evolve.py"], 1, ["tools"], tasks)
    assert any("budget" in p for p in problems)
    assert any("untried" in p for p in problems)
    assert any("bench/evolve.py" in p for p in problems)
    assert rsi.check_declaration([{"id": "E1", "component": "vibes"}], ["src/wren/x.py"], 3, [], tasks)
    assert rsi.check_declaration(good, [], 1, [], tasks) == ["no files changed"]
    assert rsi.check_declaration({}, ["src/wren/x.py"], 1, [], tasks)
    # The adapter that runs wren in the container and reports its tokens is frozen.
    adapter = rsi.check_declaration(good, ["src/wren/integrations/pier_support.py"], 1, [], tasks)
    assert any("pier_support.py" in p for p in adapter)
    # Every required field, and predictions that name evolve tasks.
    terse = {k: EDIT[k] for k in ("id", "component", "hypothesis", "predicted_affected")}
    missing = rsi.check_declaration([terse], ["src/wren/x.py"], 1, [], tasks)
    assert any("retroactive_check" in p and "trigger_condition" in p for p in missing)
    unknown = rsi.check_declaration([EDIT | {"predicted_affected": ["b", "zz"]}], ["src/wren/x.py"], 1, [], tasks)
    assert any("zz" in p for p in unknown)
    assert rsi.check_declaration([EDIT | {"predicted_affected": "b"}], ["src/wren/x.py"], 1, [], tasks)


def diff(path: str, *added: str) -> str:
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -0,0 +1 @@\n"
            + "".join(f"+{line}\n" for line in added))


def test_tag_components():
    def tags(*components: str, diff_: str) -> list[str]:
        return rsi.tag_components([EDIT | {"component": c} for c in components], diff_)

    assert tags("tools", diff_=diff("src/wren/tools/shell.py", "    return out")) == ["tools"]
    # A prompt edit dressed up as subagent work, to fill an exploration slot, is recorded as prompt.
    assert tags("subagent", diff_=diff("src/wren/agent/prompt.py", '"Delegate to the task tool."')) == ["prompt"]
    # Text alone is prompt even in a file of another component; code there is that component.
    assert tags("subagent", diff_=diff("src/wren/agent/subagents.py", '    "Explore first."')) == ["prompt"]
    assert tags("subagent", diff_=diff("src/wren/agent/subagents.py", "    brief = task")) == ["subagent"]
    # plans.py holds reminders (prompt) and when they fire (control_flow).
    assert tags("control_flow", diff_=diff("src/wren/agent/plans.py", "if turns > 3:")) == ["control_flow"]
    # config is a constant's new value, wherever it lives.
    assert tags("config", diff_=diff("src/wren/agent/compact.py", "KEEP_TURNS = 6")) == ["config"]
    assert tags("config", diff_=diff("src/wren/agent/compact.py", "    keep = 6")) == ["context_mgmt"]
    # Two edits: the mislabelled one takes what the diff shows that the other doesn't claim.
    both = diff("src/wren/tools/files.py", "x = 1") + diff("src/wren/agent/compact.py", "    keep = 6")
    assert tags("tools", "subagent", diff_=both) == ["tools", "context_mgmt"]
    # Tests, docs and unmapped modules show nothing.
    assert tags("tools", diff_=diff("tests/test_tools.py", "x") + diff("src/wren/cli/app.py", "x")) == ["prompt"]


def test_attribute_and_scoreboard():
    inc = {"k": 2, "per_task": {"a": trials(1, 1), "b": trials(0, 0), "c": trials(0, 1), "d": trials(1, 0)}}
    cand = {"k": 2, "per_task": {"a": trials(0, 1), "b": trials(1, 0), "c": trials(1, 1), "d": trials(1, 0)}}
    rows = rsi.attribute([EDIT | {"predicted_affected": ["b", "d"]}, EDIT | {"predicted_affected": []}],
                         inc, cand, "reward")
    assert rows[0] == {"predicted_hit": ["b"], "hit_rate": 0.5, "improved": ["b", "c"],
                       "unpredicted_regressions": ["a"]}
    assert rows[1]["hit_rate"] is None and rows[1]["unpredicted_regressions"] == ["a"]
    history = [{"t": 0, "edit_id": "E1", "hypothesis": "h", **rows[0]},
               {"t": 0, "edit_id": "E2", "delta_S": None}]   # dropped in screening: not on the board
    board = [json.loads(line) for line in rsi.render_scoreboard(history).splitlines()]
    assert [r["edit_id"] for r in board] == ["E1"] and board[0]["hit_rate"] == 0.5
    assert rsi.render_scoreboard([]) == "(nothing measured yet)"


def test_precheck_catches_task_names_and_credentials():
    split = {"_doc": "x", "evolve": ["abs-module-cache-flags"], "heldout": ["adaptix-name-mapping-aliases"]}
    diff = "+++ b/src/wren/agent/prompt.py\n+if 'adaptix-name-mapping-aliases' in cwd:\n-old\n"
    assert any("task name" in h for h in rsi.precheck(diff, split))
    assert any("credential" in h for h in rsi.precheck("+key = 'sk-abcdefghijklmnopqrstu'\n", split))
    assert rsi.precheck("-'abs-module-cache-flags'\n+generic\n", split) == []   # removed lines are fine
    assert rsi.precheck("+'abs-module-cache-flags-v2'\n", split) == []          # whole names only


def test_parse_json_reply():
    assert rsi.parse_json_reply('{"verdict": "accept"}') == {"verdict": "accept"}
    assert rsi.parse_json_reply('Here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert rsi.parse_json_reply('I think {"a": {"b": 2}} fits') == {"a": {"b": 2}}
    assert rsi.parse_json_reply("no json") is None


def test_select_picks_the_best_admissible():
    inc = {"S": 0.5, "C": 100.0}
    ready = [{"variant": 1}, {"variant": 2}, {"variant": 3}]
    measured = [{"k": 2, "per_task": {"a": trials(1, 1), "b": trials(1, 0)}},       # S 0.75: admissible
                {"k": 2, "per_task": {"a": trials(1, 1), "b": trials(1, 1)}},       # S 1.0: best
                {"k": 2, "per_task": {"a": trials(None, None), "b": trials(1, 1)}}]  # half ungraded
    v = rsi.select(ready, measured, inc, 0.5, 0.05, CFG)
    assert v[2]["winner"] and v[2]["outcome"] == "accepted"
    assert v[1]["admissible"] and not v[1]["winner"]
    assert v[3]["outcome"] == "invalid"
    v = rsi.select(ready[:1], measured[:1], {"S": 0.9, "C": 100.0}, 0.9, 0.05, CFG)
    assert not v[1]["winner"] and v[1]["outcome"] == "rejected"


# --- a whole round, with wren and Pier faked ---

def sh(*cmd: str, cwd: Path) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    remote, root = tmp_path / "remote.git", tmp_path / "repo"
    sh("git", "init", "-q", "--bare", str(remote), cwd=tmp_path)
    sh("git", "init", "-q", "-b", "main", str(root), cwd=tmp_path)
    (root / "src" / "wren" / "agent").mkdir(parents=True)
    (root / "src" / "wren" / "agent" / "prompt.py").write_text("PROMPT = 'be good'\n")
    for cmd in (["git", "config", "user.email", "t@t"], ["git", "config", "user.name", "t"],
                ["git", "add", "-A"], ["git", "commit", "-q", "-m", "init"],
                ["git", "remote", "add", "origin", str(remote)], ["git", "push", "-q", "origin", "main"]):
        sh(*cmd, cwd=root)
    runs = tmp_path / "runs"
    monkeypatch.setattr(evolve, "ROOT", root)
    monkeypatch.setattr(evolve, "EVALS", runs / "evals")
    monkeypatch.setattr(evolve, "LEDGER", runs / "ledger.jsonl")
    monkeypatch.setattr(evolve, "FRONTIER", runs / "frontier.json")
    monkeypatch.setattr(rsi, "RSI", runs / "rsi")
    monkeypatch.setattr(rsi, "worktree_dir", lambda run, name: tmp_path / "wt" / run / name)
    monkeypatch.setattr(evolve, "load_split", lambda: {"evolve": ["a", "b"]})
    return root


@pytest.mark.parametrize("cut_off,broken", [(False, False), (True, False), (False, True)])
def test_a_round_end_to_end(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut_off: bool,
                            broken: bool):
    sha = sh("git", "rev-parse", "HEAD", cwd=repo)
    trial_dir = tmp_path / "trial-b"
    (trial_dir / "agent" / "wren" / "sessions").mkdir(parents=True)
    inc = {"name": "base", "set": "evolve", "k": 2, "harness": {"sha": sha, "kwargs": {}, "model": "policy"},
           "per_task": {"a": trials(1, 1), "b": trials(0, 0, dirs=[str(trial_dir)] * 2)}}
    evolve.save_json(evolve.eval_path("base"), inc)
    evolve.save_json(evolve.FRONTIER, {"incumbent": "base", "S_star": 0.5, "metric": "reward"})

    prompts: list[tuple[str, bool]] = []

    def fake_wren(rc, text, cwd, *, model, yolo, time_limit, log):
        prompts.append((text, yolo))
        if not yolo and "batch analyst" in text:
            return {"status": "done", "result": '{"failure_modes": [{"mode": "turn_exhaustion", "n_tasks": 1}]}'}
        if not yolo and "digesting" in text:
            return {"status": "done", "result": '{"blocker": "ran out of turns"}'}
        if not yolo:
            return {"status": "done", "result": '{"verdict": "accept", "reasons": []}'}
        variant = 1 if "candidate 1" in text else 2
        if variant == 1 and "previous attempt" not in text:   # first try: nothing declared
            return {"status": "done", "result": "done"}
        (cwd / "src" / "wren" / "agent" / "prompt.py").write_text(f"PROMPT = 'be good, v{variant}'\n")
        (cwd / rsi.EDITS_FILE).write_text(json.dumps([EDIT | {"hypothesis": f"variant {variant}"}]))
        return {"status": "done", "result": f"changed the prompt (v{variant})"}

    def fake_eval(sha, overrides, set_name, k, name, cfg, dry_run=False):
        if broken and "v1" in name:
            raise evolve.HarnessBroken(f"{name}: the first 3 trials crashed the harness: NonZeroAgentExitCodeError")
        b = [1, 1] if "v2" in name else [1, 0]
        ev = {"name": name, "set": "evolve", "k": 2, "harness": {"sha": sha, "kwargs": {}, "model": "policy"},
              "per_task": {"a": trials(1, 1), "b": trials(*b)}}
        evolve.save_json(evolve.eval_path(name), ev)
        return ev

    monkeypatch.setattr(rsi, "run_wren", fake_wren)
    monkeypatch.setattr(rsi, "run_checks", lambda wt, checks: None)
    monkeypatch.setattr(evolve, "run_eval", fake_eval)
    cfg = {**CFG, "rsi": {"T": 2, "candidates": 2, "repair_rounds": 1}}
    rsi.cmd_start(type("A", (), {"run": "r1", "T": None})(), cfg)
    args = type("A", (), {"run": "r1", "rounds": 1, "no_eval": False})()
    if cut_off:   # the round dies after its winner reached the frontier; `run` finishes it
        real_git = rsi.git

        def failing_git(*cmd: str) -> str:
            if any("evolve/r1" in c for c in cmd):
                monkeypatch.setattr(rsi, "git", real_git)
                raise RuntimeError("connection reset")
            return real_git(*cmd)

        monkeypatch.setattr(rsi, "git", failing_git)
        with pytest.raises(RuntimeError):
            rsi.cmd_run(args, cfg)
        assert evolve.load_json(evolve.FRONTIER)["incumbent"] == "r1-t0-v2"
    rsi.cmd_run(args, cfg)

    state = rsi.load_state("r1")
    assert state["t"] == 1 and state["trajectory"] == [0.5, 1.0]
    frontier = evolve.load_json(evolve.FRONTIER)
    assert frontier["incumbent"] == "r1-t0-v2" and frontier["S_star"] == 1.0
    history = rsi.read_history("r1")
    if broken:   # v1 never got a measurement; the round goes on with v2
        assert [(r["variant"], r["accepted"], r["outcome"], r["delta_S"]) for r in history] == [
            (1, False, "dropped in the container (smoke)", None), (2, True, "accepted", 0.5)]
        assert "crashed the harness" in history[0]["detail"]
        return
    assert [(r["variant"], r["accepted"], r["outcome"]) for r in history] == [
        (1, False, "admissible"), (2, True, "accepted")]
    # Both predicted b, and both moved it.
    assert [(r["predicted_hit"], r["hit_rate"], r["unpredicted_regressions"]) for r in history] == [
        (["b"], 1.0, []), (["b"], 1.0, [])]
    # The winner is committed without the declaration, pushed, and evolve/r1 points at it.
    winner = history[1]["sha"]
    remote_tip = sh("git", "ls-remote", "origin", "refs/heads/evolve/r1", cwd=repo).split()[0]
    assert remote_tip == winner
    assert rsi.EDITS_FILE not in sh("git", "show", "--name-only", "--format=", winner, cwd=repo)
    # The proposer saw the analysis and its rules, and was told why its first attempt failed.
    assert any("previous attempt" in p and "RSI_EDITS.json is missing" in p for p, _ in prompts)
    proposals = [p for p, yolo in prompts if yolo]
    assert all("# Proposer constitution" in p and "# Pattern library" in p for p in proposals)
    assert all("2 tasks of the evolve split, 2 trials each" in p for p in proposals)
    report = rsi.RSI / "r1" / "t0" / "report.md"
    assert "ran out of turns" in report.read_text() and "turn_exhaustion" in report.read_text()
    ledger = [json.loads(line) for line in evolve.LEDGER.read_text().splitlines()]
    assert [r["verdict"] for r in ledger] == ["REJECTED", "ACCEPTED"]
    assert not (tmp_path / "wt" / "r1" / "t0-v1").exists()


def test_git_says_what_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path)
    with pytest.raises(evolve.GitError, match="not a git repository"):
        evolve.git("status")


def test_push_tries_again(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rsi.time, "sleep", lambda s: None)
    tries: list[tuple[str, ...]] = []

    def flaky(*args: str) -> str:
        tries.append(args)
        if len(tries) < 3:
            raise evolve.GitError(128, ["git", *args], "", "fatal: could not read Username")
        return ""

    monkeypatch.setattr(rsi, "git", flaky)
    rc = rsi.rsi_config({})
    rsi.push(rc, "push", "origin", "x")
    assert len(tries) == 3
    tries.clear()
    with pytest.raises(evolve.GitError, match="could not read Username"):
        rsi.push(rc | {"push_retries": 1}, "push", "origin", "x")
    assert len(tries) == 2


def test_run_goes_on_after_a_round_is_cut_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rsi, "RSI", tmp_path)
    monkeypatch.setattr(rsi.time, "sleep", lambda s: None)
    rsi.save_state({"run": "r1", "T": 2, "t": 0, "trajectory": [0.5]})
    outcomes: list = [evolve.MeasurementIncomplete("r1-t0-v1: 50/60 trials finished")]

    def fake_round(state, cfg, rc, evaluate=True):
        out = outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        state["t"] += 1
        rsi.save_state(state)
        return out

    monkeypatch.setattr(rsi, "run_round", fake_round)
    args = type("A", (), {"run": "r1", "rounds": None, "no_eval": False})()
    # Without retries, a cut-off round stops the run.
    with pytest.raises(evolve.MeasurementIncomplete):
        rsi.cmd_run(args, {"rsi": {"round_retries": 0}})
    assert rsi.load_state("r1")["t"] == 0
    # With one, each cut-off round is run again and the run finishes; the count starts
    # over after a round that finishes.
    outcomes[:] = [evolve.MeasurementIncomplete("cut off"), True,
                   evolve.GitError(128, ["git", "push"], "", "x"), True]
    rsi.cmd_run(args, {"rsi": {"round_retries": 1}})
    assert rsi.load_state("r1")["t"] == 2 and outcomes == []
    # Two in a row are one too many.
    rsi.save_state({"run": "r1", "T": 3, "t": 2, "trajectory": [0.5]})
    outcomes[:] = [evolve.GitError(128, ["git", "push"], "", "x")] * 2
    with pytest.raises(evolve.GitError):
        rsi.cmd_run(args, {"rsi": {"round_retries": 1}})
    # Anything else stops the run at once.
    rsi.save_state({"run": "r1", "T": 3, "t": 2, "trajectory": [0.5]})
    outcomes[:] = [ValueError("a bug")]
    with pytest.raises(ValueError):
        rsi.cmd_run(args, {"rsi": {"round_retries": 3}})
