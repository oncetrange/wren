"""Recursive self-improvement of wren's harness, following RRSI.

Each round reads how the incumbent harness failed, has wren itself propose and
implement a few candidate changes to its own source, screens them, measures
them on the evolve split and keeps the best admissible one (evolve.py's
selection rule). The policy model the benchmark runs is frozen; only the
harness changes.

    python bench/rsi.py start <run> [--T 8]   # from the incumbent (evolve.py baseline)
    python bench/rsi.py run <run>             # run rounds until T, or until runs/rsi/<run>/STOP
    python bench/rsi.py run <run> --rounds 1 --no-eval   # analyze and propose only
    python bench/rsi.py status <run>

A round t:
  1. analyze: failed (and contrasting passed) trials of the incumbent are rendered
     to text and digested one at a time by a read-only wren; an analyst wren
     clusters the digests into a three-lens report (failure modes ranked by
     impact, capability gaps, success habits), asking for follow-up digests
     first if it needs them, and keeps the names stable across rounds;
  2. propose: in its own git worktree, wren edits its source under the rules of
     prompts/constitution.md, declaring each independent edit (component,
     hypothesis, trigger, predicted tasks, a retroactive check) in RSI_EDITS.json,
     within the annealed edit budget b_t and the exploration and prune directives
     drawn from the edit history;
  3. screen: the declaration is checked, the repo's checks run, and a critic
     rejects benchmark-specific logic; a failing candidate gets bounded repairs;
  4. measure the survivors (pushed, since the task containers install wren from
     GitHub), judge them against the incumbent and fast-forward evolve/<run> to the
     best admissible one.

Every edit goes into runs/rsi/<run>/history.jsonl with its measured gain and
which tasks moved (its predictions' hits, unpredicted regressions), so the
proposer doesn't redraw a falsified hypothesis and sees how well it predicts.
Settings are under "rsi" in bench/config.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import evolve  # noqa: E402
from bench.evolve import COMPONENTS, git, load_json, save_json  # noqa: E402

RSI = evolve.RUNS / "rsi"
PROMPTS = evolve.BENCH / "prompts"
EDITS_FILE = "RSI_EDITS.json"
# A candidate may change wren itself, its tests and its docs, never the benchmark tooling,
# nor the adapter that runs wren in the task container and reports the tokens C is made of.
EDITABLE = ("src/wren/", "tests/", "docs/")
FROZEN = ("src/wren/integrations/",)
EDIT_FIELDS = ("id", "component", "hypothesis", "targets_mode", "why_not_lower_lever",
               "trigger_condition", "predicted_affected", "retroactive_check")
# Which components a change to a file shows (RRSI's component_signals): an edit is recorded
# under its declared component only when the diff shows it. The first matching prefix wins;
# other files under src/wren/ (the CLI, providers, new modules) show nothing. A change that
# only touches strings and comments is prompt, and a constant's new value, in any file, config.
COMPONENT_SIGNALS = (
    ("src/wren/agent/prompt.py", ("prompt",)),
    ("src/wren/agent/plans.py", ("prompt", "control_flow")),
    ("src/wren/agent/compact.py", ("context_mgmt",)),
    ("src/wren/agent/conversation.py", ("context_mgmt",)),
    ("src/wren/agent/subagents.py", ("subagent",)),
    ("src/wren/agent/remember.py", ("memory",)),
    ("src/wren/agent/", ("control_flow",)),
    ("src/wren/tools/memory.py", ("memory",)),
    ("src/wren/tools/skill.py", ("skill",)),
    ("src/wren/tools/", ("tools",)),
    ("src/wren/memory.py", ("memory",)),
    ("src/wren/skills.py", ("skill",)),
)
CONSTANT = re.compile(r"^\+\s*[A-Z][A-Z0-9_]*\s*(?::[^=]+)?=", re.M)
TEXT_LINE = re.compile(r'^[+-]\s*(?:[frb]?["\']|#)')
# Never required by a stalled run's exploration slot: memory can't carry anything from one
# trial to the next (each starts with an empty ~/.wren, and memory is off headless), and no
# skill reaches a trial unless the harness ships one. Both can still be declared.
NOT_EXPLORED = ("memory", "skill")
# Candidates are drafted in parallel; git commands on the shared repo (worktrees, pushes) take turns.
_GIT_LOCK = threading.Lock()

DEFAULTS = {
    "T": 8,                    # rounds
    "candidates": 2,           # candidates drafted per round
    "b_min": 1, "b_max": 3,    # edit budget, annealed from b_max to b_min
    "stall_window": 2,         # rounds without a gain above delta before exploring
    "n_prune": 4,              # rounds a component's best recent gain is looked back over
    "max_digests": 12,         # trials digested per round before the analyst reads them
    "max_followups": 6,        # further digests the analyst may ask for per turn
    "analyst_rounds": 2,       # analyst turns: follow-ups, then the report
    "repair_rounds": 2,        # attempts to fix a candidate that fails screening
    "wren": "wren",            # the installed wren that does the work, not the one being changed
    "proposer_model": None,    # None: wren's default model; should not be the policy model
    "digest_model": None,
    "analyst_model": None,
    "critic_model": None,
    "proposer_time_limit": 45,
    "digest_time_limit": 8,
    "analyst_time_limit": 15,
    "checks": [["uv", "run", "ruff", "check"], ["uv", "run", "pyright"],
               ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider"]],
    "remote": "origin",
}


def rsi_config(cfg: dict) -> dict:
    return {**DEFAULTS, **cfg.get("rsi", {})}


# --- schedule and history: what the proposer is told ---

def edit_budget(t: int, T: int, b_min: int, b_max: int) -> int:
    """b_t: many edits per candidate early on, few late, on a cosine."""
    if T <= 0:
        return b_max
    t = max(0, min(t, T))
    v = b_min + (b_max - b_min) * 0.5 * (1 + math.cos(math.pi * t / T))
    return math.ceil(round(v, 9))


def read_history(run: str) -> list[dict]:
    path = RSI / run / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_history(run: str, records: list[dict]) -> None:
    path = RSI / run / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **r}, ensure_ascii=False) + "\n")


def stalled(trajectory: list[float], t: int, window: int, delta: float) -> bool:
    """sigma_t: the incumbent's score hasn't risen by more than delta in `window` rounds.
    trajectory[i] is the incumbent's S after round i - 1 (trajectory[0]: the start)."""
    if t < window or t >= len(trajectory):
        return False
    return trajectory[t] - trajectory[t - window] <= delta


def untried(history: list[dict]) -> list[str]:
    """Components no measured edit has touched yet, but those a benchmark trial can't reward."""
    tried = {r["component"] for r in history if r.get("delta_S") is not None}
    return [c for c in COMPONENTS if c not in tried and c not in NOT_EXPLORED]


def prune_set(history: list[dict], t: int, n_prune: int) -> list[dict]:
    """Components with machinery in the incumbent whose edits stopped paying:
    none measured in the last n_prune rounds gained anything."""
    accepted: dict[str, int] = {}
    for r in history:
        if r.get("accepted") and r.get("component"):
            accepted[r["component"]] = accepted.get(r["component"], 0) + 1
    out = []
    for component, n in accepted.items():
        gains = [r["delta_S"] for r in history if r.get("component") == component
                 and r.get("delta_S") is not None and t - r["t"] <= n_prune]
        best = max(gains) if gains else None
        if best is None or best <= 0:
            out.append({"component": component, "recent_best_gain": best, "accepted_edits": n})
    return out


def render_history(history: list[dict], n: int = 40) -> str:
    """The newest n edits for the proposer, oldest first; of the edits never
    measured (dropped in screening) only the latest few are kept."""
    keys = ("t", "variant", "edit_id", "component", "hypothesis", "delta_S", "delta_C",
            "accepted", "outcome", "detail")
    kept, unmeasured = [], 0
    for r in reversed(history):
        if r.get("delta_S") is None:
            unmeasured += 1
            if unmeasured > 4:
                continue
        kept.append({k: r[k] for k in keys if r.get(k) is not None})
        if len(kept) >= n:
            break
    if not kept:
        return "(no edits yet: this is the first round)"
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in reversed(kept))


def attribute(edits: list, inc: dict, cand: dict, metric: str) -> list[dict]:
    """Per declared edit, how its candidate's tasks moved against the incumbent:
    the predicted tasks that improved, every task that improved, and the tasks
    it fell on without predicting them (a drop of at least one trial in k)."""
    def mean(ev: dict, task: str) -> float:
        values = evolve.task_values(ev["per_task"][task], ev["k"], metric)
        return sum(values) / len(values)
    change = {t: mean(cand, t) - mean(inc, t) for t in inc["per_task"] if t in cand["per_task"]}
    drop = 1 / max(1, cand["k"]) - 1e-9
    improved = sorted(t for t, d in change.items() if d > 0)
    worse = sorted(t for t, d in change.items() if d <= -drop)
    rows = []
    for e in edits:
        predicted = [str(p) for p in e.get("predicted_affected") or [] if p in change]
        hit = [p for p in predicted if change[p] > 0]
        rows.append({"predicted_hit": hit,
                     "hit_rate": round(len(hit) / len(predicted), 2) if predicted else None,
                     "improved": improved[:12],
                     "unpredicted_regressions": [t for t in worse if t not in predicted][:12]})
    return rows


def render_scoreboard(history: list[dict], n: int = 20) -> str:
    """How the newest n measured edits' predictions fared, oldest first."""
    keys = ("t", "variant", "edit_id", "component", "predicted_affected", "predicted_hit",
            "hit_rate", "improved", "unpredicted_regressions")
    rows = [{k: r.get(k) for k in keys} | {"hypothesis": str(r.get("hypothesis"))[:160]}
            for r in history if "predicted_hit" in r][-n:]
    if not rows:
        return "(nothing measured yet)"
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


# --- analysis: traces and digests ---

def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters cut ...]\n{text[-half:]}"


def _tail(path: Path, lines: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def render_trace(trial: dict, task: str, limit: int = 150_000) -> str:
    """One trial as text: its grade, every model turn and tool call from the
    session logs (as sent, before masking or compaction), and what the verifier said."""
    d = Path(trial["dir"])
    scores = trial.get("scores")
    out = [f"# task {task}", f"grade: {json.dumps(scores) if scores is not None else 'none (no grade)'}"]
    if trial.get("exception"):
        out.append(f"exception: {trial['exception']}")
    for session in sorted((d / "agent" / "wren" / "sessions").glob("*.jsonl")):
        out.append(f"\n## session {session.name}")
        calls: dict[str, int] = {}
        for line in session.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            kind = e.get("kind")
            if kind == "compact":
                out.append("[context compacted]")
            elif kind == "message":
                for b in e.get("content") or []:
                    out.append(_render_block(e.get("role", "?"), b, calls))
    stderr = _tail(d / "agent" / "wren.log", 40)
    if stderr:
        out.append(f"\n## wren stderr (tail)\n{stderr}")
    for path in sorted((d / "verifier").glob("*.txt")):
        out.append(f"\n## verifier {path.name} (tail)\n{_tail(path, 150)}")
    return _clip("\n".join(out), limit)


def _render_block(role: str, b: dict, calls: dict[str, int]) -> str:
    match b.get("type"):
        case "text":
            return f"[{role}] {b.get('text', '')}"
        case "tool_use":
            n = calls.setdefault(b.get("id", ""), len(calls) + 1)
            return f"[call {n}] {b.get('name')} {_clip(json.dumps(b.get('input'), ensure_ascii=False), 3000)}"
        case "tool_result":
            n = calls.get(b.get("tool_use_id", ""), "?")
            flag = " ERROR" if b.get("is_error") else ""
            content = b.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return f"[result {n}{flag}] {_clip(text, 4000)}"
        case "thinking":
            return f"[{role} thinking] {_clip(b.get('thinking', ''), 1500)}"
        case _:
            return f"[{role} {b.get('type')}]"


LENSES = ("failure", "capability_gap", "success")


def choose_trials(ev: dict, metric: str, max_n: int) -> list[dict]:
    """Which trials to digest: on tasks that pass only sometimes a failure and a
    success (the contrast says most), then one failure of each task that never
    passes, then further failures, up to max_n. Trials without logs are skipped."""
    mixed, failing, more = [], [], []
    for task, trials in sorted(ev["per_task"].items()):
        trials = [t for t in trials if t.get("dir")]
        passed = [t for t in trials if ((t["scores"] or {}).get(metric) or 0) >= 1]
        failed = [t for t in trials if t not in passed]
        if not failed:
            continue
        if passed:
            mixed += [{"task": task, "trial": failed[0], "lens": "failure"},
                      {"task": task, "trial": passed[0], "lens": "success"}]
        else:
            failing.append({"task": task, "trial": failed[0], "lens": "failure"})
        more += [{"task": task, "trial": t, "lens": "failure"} for t in failed[1:]]
    return (mixed + failing + more)[:max_n]


def prompt(name: str, **values: object) -> str:
    return Template((PROMPTS / f"{name}.md").read_text()).substitute(
        {k: str(v) for k, v in values.items()})


def run_wren(rc: dict, text: str, cwd: Path, *, model: str | None, yolo: bool,
             time_limit: float | None, log: Path) -> dict:
    """Run the installed wren headless on `text` in `cwd`; its JSON result, or
    {"status": "error"} with what went wrong. Without yolo it is read-only:
    headless, nobody approves an edit or a command."""
    cmd = [rc["wren"], "-p", "-", "--output-format", "json", "--no-checkpoints", "--max-turns", "400"]
    cmd += ["--yolo"] if yolo else ["--mode", "ask"]
    if model:
        cmd += ["-m", model]
    if time_limit:
        cmd += ["--time-limit", str(time_limit)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as err:
        try:
            proc = subprocess.run(cmd, cwd=cwd, input=text, stdout=subprocess.PIPE, stderr=err, text=True,
                                  timeout=(time_limit or 60) * 60 + 300, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "error", "result": f"wren didn't finish in time; see {log}"}
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {"status": "error", "result": f"wren exited {proc.returncode}; see {log}"}


def parse_json_reply(text: str) -> dict | None:
    """The JSON object in a model's reply: the whole reply, a fenced block, or
    the outermost braces."""
    candidates = [text, *re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)]
    if "{" in text:
        candidates.append(text[text.index("{"):text.rindex("}") + 1])
    for c in candidates:
        try:
            value = json.loads(c.strip())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def trial_index(ev: dict, task: str, lens: str, metric: str, taken: set) -> int | None:
    """A logged trial of `task` that suits `lens` (a pass for success, else a
    failure), preferring one not digested yet; None if there is none."""
    fits = [j for j, t in enumerate(ev["per_task"].get(task) or [])
            if t.get("dir") and (((t["scores"] or {}).get(metric) or 0) >= 1) == (lens == "success")]
    return next((j for j in fits if (task, j, lens) not in taken), fits[0] if fits else None)


def task_table(ev: dict, metric: str, digests: list[dict]) -> str:
    rows = []
    for task, trials in ev["per_task"].items():
        values = evolve.task_values(trials, ev["k"], metric)
        made = [f"{d['lens']} of trial {d['trial']}" for d in digests if d["task"] == task]
        rows.append((sum(values) / len(values), f"- {task}: {sum(values) / len(values):.2f} "
                     f"(trials {', '.join(f'{v:g}' for v in values)}; digests: {', '.join(made) or 'none'})"))
    return "\n".join(r for _, r in sorted(rows))


def analyze(run: str, t: int, inc: dict, cfg: dict, rc: dict) -> Path:
    """Digest the incumbent's trials, have the analyst aggregate the digests
    into a three-lens report (asking for follow-up digests if it needs them),
    and write both to runs/rsi/<run>/t<t>/report.md."""
    rdir = RSI / run / f"t{t}"
    report_md = rdir / "report.md"
    if report_md.exists():
        return report_md
    traces, digests = rdir / "traces", rdir / "digests"
    traces.mkdir(parents=True, exist_ok=True)
    digests.mkdir(parents=True, exist_ok=True)
    metric = cfg["metric"]

    def digest(req: dict) -> dict:
        task, j, lens = req["task"], req["trial"], req["lens"]
        name = f"{task}-{j}-{lens}{req.get('suffix', '')}"
        out = digests / f"{name}.json"
        if out.exists():
            return load_json(out)
        trace = traces / f"{task}-{j}.txt"
        if not trace.exists():
            trace.write_text(render_trace(inc["per_task"][task][j], task))
        questions = req.get("questions") or []
        asked = "".join(f"- {q}\n" for q in questions)
        reply = run_wren(rc, prompt("digest", trace=trace.name, task=task, lens=lens,
                                    questions=f"\nQuestions to answer:\n{asked}" if asked else ""),
                         traces, model=rc["digest_model"], yolo=False,
                         time_limit=rc["digest_time_limit"], log=rdir / "logs" / "digest.log")
        d = parse_json_reply(reply.get("result") or "") or {"unparsed": reply.get("result")}
        d = {"task": task, "trial": j, "lens": lens, "trace": f"traces/{trace.name}",
             **({"questions": questions} if questions else {}), **d}
        save_json(out, d)
        return d

    def run_digests(reqs: list[dict]) -> list[dict]:
        with ThreadPoolExecutor(4) as pool:
            return list(pool.map(digest, reqs))

    done = run_digests([
        {"task": c["task"], "lens": c["lens"],
         "trial": next(j for j, x in enumerate(inc["per_task"][c["task"]]) if x is c["trial"])}
        for c in choose_trials(inc, metric, rc["max_digests"])])
    prior_path = RSI / run / "analysis.json"
    prior = load_json(prior_path) if prior_path.exists() else {}
    report: dict | None = None
    for step in range(rc["analyst_rounds"]):
        final = step == rc["analyst_rounds"] - 1
        cached = rdir / f"analyst-{step}.json"
        if cached.exists():
            reply = load_json(cached)
        else:
            text = prompt(
                "analyst", n_tasks=len(inc["per_task"]), k=inc["k"], max_followups=rc["max_followups"],
                final="This is your last turn: no follow-ups, send the report." if final else
                      "Each one costs a digester's run: ask only for what would change the report.",
                table=task_table(inc, metric, done),
                prior=json.dumps({k: [{label: m.get(label), "description": m.get("description")}
                                      for m in prior.get(k) or [] if isinstance(m, dict)]
                                  for k, label in (("failure_modes", "mode"), ("success_habits", "habit"))},
                                 ensure_ascii=False, indent=1),
                digests="\n\n".join(f"```json\n{json.dumps(d, indent=1, ensure_ascii=False)}\n```"
                                      for d in done))
            result = run_wren(rc, text, rdir, model=rc["analyst_model"], yolo=False,
                              time_limit=rc["analyst_time_limit"], log=rdir / "logs" / "analyst.log")
            reply = parse_json_reply(result.get("result") or "") or {}
            if reply:   # a failed turn is tried again when the round is resumed
                save_json(cached, reply)
        if "failure_modes" in reply:
            report = {k: reply.get(k) or [] for k in ("failure_modes", "capability_gaps", "success_habits")}
            report["failure_modes"].sort(key=lambda m: -(m.get("n_tasks") or 0))
            break
        taken = {(d["task"], d["trial"], d["lens"]) for d in done}
        reqs = []
        for r in (reply.get("follow_up") or [])[:rc["max_followups"]]:
            if not isinstance(r, dict) or r.get("task") not in inc["per_task"]:
                continue
            lens = str(r.get("lens")) if r.get("lens") in LENSES else "failure"
            j = trial_index(inc, r["task"], lens, metric, taken)
            if j is not None:
                questions = [str(q) for q in r.get("questions") or []][:5]
                reqs.append({"task": r["task"], "trial": j, "lens": lens, "questions": questions,
                             "suffix": f"-f{step + 1}" if (r["task"], j, lens) in taken else ""})
        if final or not reqs:
            break
        done += run_digests(reqs)

    a = evolve.aggregate(inc, metric)
    lines = [f"# Round {t}: how the incumbent {inc['name']} did",
             f"S = {a['S']:.4f} over {len(inc['per_task'])} tasks x k={inc['k']}; per task, lowest first:",
             "", task_table(inc, metric, done), "", "## Analysis", ""]
    if report:
        save_json(rdir / "analysis.json", report)
        save_json(prior_path, {k: report[k] for k in ("failure_modes", "success_habits")})
        lines += ["The analyst's three-lens report: failure modes ranked by impact, capability gaps "
                  "(these often need a tool or plumbing fix rather than an instruction), and success "
                  "habits, which your change must not break.", "",
                  f"```json\n{json.dumps(report, indent=1, ensure_ascii=False)}\n```"]
    else:
        lines.append("The analyst produced no report this round: work from the digests.")
    lines += ["", "## Digests", "",
              "One per trial, by a reader of the whole trace. The traces are in traces/.", ""]
    lines += [f"```json\n{json.dumps(d, indent=1, ensure_ascii=False)}\n```" for d in done]
    report_md.write_text("\n".join(lines) + "\n")
    return report_md


# --- proposing and screening a candidate ---

def worktree_dir(run: str, name: str) -> Path:
    return Path.home() / ".wren" / "rsi" / run / name


def changed_files(wt: Path, base: str) -> list[str]:
    tracked = git("-C", str(wt), "diff", "--name-only", base).splitlines()
    new = git("-C", str(wt), "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({f for f in tracked + new if f and f != EDITS_FILE})


def diff_text(wt: Path, base: str) -> str:
    git("-C", str(wt), "add", "-A", "--", ".", f":!{EDITS_FILE}")
    return subprocess.run(["git", "-C", str(wt), "diff", "--cached", base], check=True,
                          capture_output=True, text=True).stdout


def check_declaration(edits: object, files: list[str], budget: int, explore: list[str],
                      tasks: list[str]) -> list[str]:
    """What is wrong with a candidate's RSI_EDITS.json and changed files; [] if
    nothing. `tasks` are the evolve tasks predictions may name."""
    if not isinstance(edits, list) or not edits:
        return [f"{EDITS_FILE} must be a non-empty JSON list of edits"]
    problems = []
    if len(edits) > budget:
        problems.append(f"{len(edits)} edits declared; this round's budget is {budget}")
    for e in edits:
        if not isinstance(e, dict):
            problems.append(f"not an object: {e!r}")
            continue
        if missing := [f for f in EDIT_FIELDS if not e.get(f)]:
            problems.append(f"edit {e.get('id', '?')}: missing {', '.join(missing)}")
        if str(e.get("component", "")).lower() not in COMPONENTS:
            problems.append(f"edit {e.get('id', '?')}: component must be one of {', '.join(COMPONENTS)}")
        predicted = e.get("predicted_affected")
        if predicted and not isinstance(predicted, list):
            problems.append(f"edit {e.get('id', '?')}: predicted_affected must be a list of task names")
        elif unknown := [str(p) for p in predicted or [] if p not in tasks]:
            problems.append(f"edit {e.get('id', '?')}: predicted_affected names tasks outside the "
                            f"evolve split: {', '.join(unknown[:5])}")
    if explore and not any(isinstance(e, dict) and str(e.get("component", "")).lower() in explore
                           for e in edits):
        problems.append(f"this round reserves an edit for an untried component: one of {', '.join(explore)}")
    if not files:
        problems.append("no files changed")
    if outside := [f for f in files if not f.startswith(EDITABLE) or f.startswith(FROZEN)]:
        problems.append(f"only {', '.join(EDITABLE)} may change, and not {', '.join(FROZEN)}: "
                        f"not {', '.join(outside)}")
    return problems


def file_components(path: str) -> tuple[str, ...]:
    return next((comps for prefix, comps in COMPONENT_SIGNALS if path.startswith(prefix)), ())


def split_diff(diff: str) -> dict[str, str]:
    """A unified diff by file (its new path)."""
    out: dict[str, str] = {}
    for chunk in re.split(r"^(?=diff --git )", diff, flags=re.M):
        if m := re.match(r"diff --git a/\S+ b/(\S+)", chunk):
            out[m.group(1)] = chunk
    return out


def text_only(diff: str) -> bool:
    """Every changed line is a string literal or a comment: model-facing text."""
    changed = [line for line in diff.splitlines() if line[:1] in "+-"
               and not line.startswith(("+++", "---")) and line[1:].strip()]
    return bool(changed) and all(TEXT_LINE.match(line) for line in changed)


def evidenced(diff: str) -> list[str]:
    """The components a diff shows, in COMPONENTS order."""
    shown: set[str] = set()
    for f, d in split_diff(diff).items():
        if comps := file_components(f):
            shown |= {"prompt"} if text_only(d) else set(comps)
        if f.startswith("src/wren/") and CONSTANT.search(d):
            shown.add("config")
    return [c for c in COMPONENTS if c in shown]


def tag_components(edits: list, diff: str) -> list[str]:
    """The component each edit is recorded under: the declared one if the diff shows it,
    else one the diff shows and no other edit claims (prompt if it shows none). Like
    RRSI, a mislabel is re-tagged, not rejected: arguing labels costs a candidate."""
    shown = evidenced(diff)
    declared = [str(e.get("component", "")).lower() for e in edits]
    tags: list[str] = []
    for d in declared:
        free = [c for c in shown if c not in declared and c not in tags]
        tags.append(d if d in shown else (free or shown or ["prompt"])[0])
    return tags


def run_checks(wt: Path, checks: list[list[str]]) -> str | None:
    """The repo's own checks in the worktree: None if they pass, else the failing output."""
    for cmd in checks:
        proc = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, check=False, timeout=1800)
        if proc.returncode != 0:
            return f"$ {' '.join(cmd)}\n{_clip(proc.stdout + proc.stderr, 8000)}"
    return None


def denylist(split: dict) -> list[tuple[str, re.Pattern[str]]]:
    """Patterns no harness change has a reason to contain: credentials, task
    names (evolve and held-out), and the grader's files."""
    tasks = sorted({t for ids in split.values() if isinstance(ids, list) for t in ids})
    pats = [("credential", re.compile(r"(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}"
                                      r"|api_key\s*=\s*[\"'][^\"']{8,})")),
            ("grader file", re.compile(r"/logs/verifier|reward\.txt|verifier_result"))]
    if tasks:
        names = "|".join(re.escape(t) for t in tasks)
        pats.append(("benchmark task name", re.compile(rf"(?<![\w-])(?:{names})(?![\w-])")))
    return pats


def precheck(diff: str, split: dict) -> list[str]:
    added = "\n".join(line[1:] for line in diff.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    return [f"{label} in the diff: {m.group(0)[:60]!r}"
            for label, pat in denylist(split) if (m := pat.search(added))]


def critic(rc: dict, wt: Path, edits: list, diff: str, log: Path) -> dict:
    """An LLM review for benchmark-specific logic and other ways to win without
    a better harness. {"verdict": "accept" | "reject", "reasons": [...]}."""
    text = prompt("critic", edits=json.dumps(edits, indent=1, ensure_ascii=False), diff=_clip(diff, 120_000))
    for _ in range(3):
        reply = run_wren(rc, text, wt, model=rc["critic_model"], yolo=False, time_limit=10, log=log)
        v = parse_json_reply(reply.get("result") or "")
        if v and v.get("verdict") in ("accept", "reject"):
            return v
    return {"verdict": "reject", "reasons": ["the critic's reply couldn't be read"]}


def propose(run: str, t: int, variant: int, base: str, context: str, cfg: dict, rc: dict,
            split: dict, budget: int, explore: list[str]) -> dict:
    """Draft, implement and screen one candidate in its own worktree. Returns
    {"variant", "status": "ready" | "dropped", "edits", "sha"?, "detail"?}."""
    name = f"t{t}-v{variant}"
    vdir = RSI / run / f"t{t}" / f"v{variant}"
    saved = vdir / "candidate.json"
    if saved.exists():
        return load_json(saved)
    wt = worktree_dir(run, name)
    branch = f"evolve-cand/{run}/{name}"
    with _GIT_LOCK:
        if wt.exists():
            git("worktree", "remove", "--force", str(wt))
        git("worktree", "add", "--force", "-B", branch, str(wt), base)
    vdir.mkdir(parents=True, exist_ok=True)
    log = vdir / "propose.log"
    feedback = ""
    edits: list = []
    tags: list[str] = []
    out: dict = {"variant": variant, "status": "dropped", "edits": []}
    tasks = split["evolve"]
    for attempt in range(rc["repair_rounds"] + 1):
        text = prompt("propose", context=context, budget=budget, variant=variant,
                      edits_file=EDITS_FILE, components=", ".join(COMPONENTS), policy_model=cfg["model"],
                      editable=", ".join(EDITABLE), feedback=feedback, n_tasks=len(tasks), k=cfg["k"],
                      constitution=(PROMPTS / "constitution.md").read_text().strip(),
                      patterns=(PROMPTS / "patterns.md").read_text().strip())
        result = run_wren(rc, text, wt, model=rc["proposer_model"], yolo=True,
                          time_limit=rc["proposer_time_limit"], log=log)
        out["summary"] = result.get("result")
        try:
            edits = json.loads((wt / EDITS_FILE).read_text())
        except (OSError, ValueError) as e:
            edits = []
            problems = [f"{EDITS_FILE} is missing or not JSON ({e})"]
        else:
            problems = check_declaration(edits, changed_files(wt, base), budget, explore, tasks)
            if not problems:
                tags = tag_components(edits, diff_text(wt, base))
                if explore and not any(c in explore for c in tags):
                    problems = [f"this round reserves an edit for an untried component ({', '.join(explore)}), "
                                f"judged by the diff, which reads as {', '.join(dict.fromkeys(tags))}"]
        out["edits"] = edits if isinstance(edits, list) else []
        stage = "declaration"
        if not problems and (failure := run_checks(wt, rc["checks"])):
            stage, problems = "checks", [failure]
        if not problems:
            diff = diff_text(wt, base)
            (vdir / "diff.patch").write_text(diff)
            if hits := precheck(diff, split):
                stage, problems = "critic", hits
            else:
                review = critic(rc, wt, edits, diff, vdir / "critic.log")
                save_json(vdir / f"critic-{attempt}.json", review)
                if review["verdict"] == "reject":
                    stage, problems = "critic", [str(r) for r in review.get("reasons") or ["rejected"]]
        if not problems:
            break
        out["detail"] = f"{stage}: {'; '.join(problems)}"[:600]
        print(f"  v{variant}: attempt {attempt + 1} failed {out['detail'][:200]}")
        feedback = prompt("repair", stage=stage, problems="\n".join(f"- {p}" for p in problems))
    else:
        out["outcome"] = f"dropped in screening ({out['detail'].split(':')[0]})"
        save_json(saved, out)
        return out
    for e, tag in zip(edits, tags, strict=True):
        if tag != str(e.get("component", "")).lower():
            print(f"  v{variant}: edit {e.get('id')} declared {e.get('component')}, "
                  f"but the diff reads as {tag}: re-tagged")
            e["declared_component"], e["component"] = e.get("component"), tag
    git("-C", str(wt), "add", "-A", "--", ".", f":!{EDITS_FILE}")
    summary = "; ".join(f"{e.get('component')}: {e.get('hypothesis')}" for e in edits)
    git("-C", str(wt), "commit", "-q", "-m", f"rsi({run}): round {t}, variant {variant}\n\n{summary}")
    sha = git("-C", str(wt), "rev-parse", "HEAD")
    with _GIT_LOCK:
        git("-C", str(wt), "push", "-q", "--force", rc["remote"], f"HEAD:refs/heads/{branch}")
    out.update(status="ready", sha=sha, branch=branch)
    out.pop("detail", None)
    save_json(saved, out)
    return out


# --- a round ---

def load_state(run: str) -> dict:
    path = RSI / run / "state.json"
    if not path.exists():
        raise SystemExit(f"no run {run!r}: `start` it first")
    return load_json(path)


def save_state(state: dict) -> None:
    save_json(RSI / state["run"] / "state.json", state)


def context_for(state: dict, t: int, history: list[dict], delta: float, rc: dict) -> tuple[str, int, list[str]]:
    """The round's directives for the proposer, its edit budget and the
    components a candidate must explore (none unless the run has stalled)."""
    budget = edit_budget(t, state["T"], rc["b_min"], rc["b_max"])
    fresh = untried(history)
    stall = stalled(state["trajectory"], t, rc["stall_window"], delta)
    explore = fresh if stall and fresh else []
    prune = prune_set(history, t, rc["n_prune"])
    lines = [f"Round {t} of {state['T']}. Edit budget b_t = {budget}: at most {budget} independent edits."]
    if explore:
        lines.append(f"The run has stalled (no gain above the noise band delta = {delta:.4f} in "
                     f"{rc['stall_window']} rounds). At least one edit must be on a component never "
                     f"tried yet: {', '.join(explore)}.")
    elif fresh:
        lines.append(f"Components never tried yet: {', '.join(fresh)}.")
    if prune:
        lines.append("Components whose accepted machinery stopped paying; consider removing it "
                     "(a removal edit's hypothesis starts with \"prune: \"): "
                     + json.dumps(prune, ensure_ascii=False))
    lines += ["", "Edit history, oldest first (delta_S is the measured change in score, delta_C in "
              "tokens per trial; an edit bundled with others shares their measurement). A rejected "
              "mechanism is negative evidence: don't redraw it unchanged. An accepted one carries the "
              "gain it produced: refine what has known credit, not what merely preceded a rise.",
              render_history(history),
              "", "Attribution scoreboard, oldest first: per measured edit, which of its predicted tasks "
              "improved over the incumbent (hit_rate), every task its candidate improved, and the tasks "
              "it fell on without predicting them (unpredicted_regressions). With few trials per task, "
              "a single task's move is partly noise; a pattern across rounds is not.",
              render_scoreboard(history)]
    return "\n".join(lines), budget, explore


def run_round(state: dict, cfg: dict, rc: dict, evaluate: bool = True) -> bool:
    """One round; True if it finished (measured and judged)."""
    run, t = state["run"], state["t"]
    rdir = RSI / run / f"t{t}"
    rdir.mkdir(parents=True, exist_ok=True)
    # The round's incumbent is pinned when it starts: a round cut off after its
    # winner reached the frontier is finished again against the same incumbent.
    if state.get("round", {}).get("t") != t:
        fr = load_json(evolve.FRONTIER)
        state["round"] = {"t": t, "incumbent": fr["incumbent"], "S_star": fr["S_star"]}
        save_state(state)
    fr = state["round"]
    inc = evolve.load_eval(fr["incumbent"])
    base = inc["harness"]["sha"]
    delta = evolve.delta_of(cfg)
    history = read_history(run)
    split = evolve.load_split()
    print(f"round {t}: incumbent {fr['incumbent']} ({base[:7]}), S* = {fr['S_star']:.4f}, delta = {delta:.4f}")

    report = analyze(run, t, inc, cfg, rc)
    directives, budget, explore = context_for(state, t, history, delta, rc)
    context = (f"{directives}\n\nThe analysis of the incumbent's trials is in {report} "
               f"(read it first; it points at the rendered traces).")
    (rdir / "context.md").write_text(context + "\n")

    def draft(v: int) -> dict:
        # A reserved exploration slot binds the first variant only.
        return propose(run, t, v, base, context, cfg, rc, split, budget, explore if v == 1 else [])

    with ThreadPoolExecutor(rc["candidates"]) as pool:
        cands = list(pool.map(draft, range(1, rc["candidates"] + 1)))
    for c in cands:
        print(f"  v{c['variant']}: {c['status']} {c.get('sha', '')[:7]} {c.get('detail', '')[:120]}")
    if not evaluate:
        return False

    ready = [c for c in cands if c["status"] == "ready"]

    def measure(c: dict) -> dict:
        name = f"{run}-t{t}-v{c['variant']}"
        if evolve.eval_path(name).exists():
            return evolve.load_eval(name)
        ev = evolve.run_eval(c["sha"], {}, "evolve", cfg["k"], name, cfg)
        assert ev is not None
        return ev

    with ThreadPoolExecutor(max(1, len(ready))) as pool:
        measured = list(pool.map(measure, ready))
    a_inc = evolve.aggregate(inc, cfg["metric"])
    verdicts = select(ready, measured, a_inc, fr["S_star"], delta, cfg)

    records = []
    measured_by = {c["variant"]: ev for c, ev in zip(ready, measured, strict=True)}
    for c in cands:
        v = verdicts.get(c["variant"])
        edits = c["edits"] or [{"id": "C1", "component": None}]
        # An invalid measurement says nothing about which tasks moved.
        attribution = (attribute(edits, inc, measured_by[c["variant"]], cfg["metric"])
                       if v and v["outcome"] != "invalid" else [{}] * len(edits))
        for e, attr in zip(edits, attribution, strict=True):
            records.append({
                "t": t, "variant": c["variant"], "edit_id": e.get("id"), "component": e.get("component"),
                **({"declared_component": e["declared_component"]} if "declared_component" in e else {}),
                "hypothesis": e.get("hypothesis"), "predicted_affected": e.get("predicted_affected"),
                "bundle": len(c["edits"]), "sha": c.get("sha"),
                "delta_S": v["delta_S"] if v else None, "delta_C": v["delta_C"] if v else None,
                "S": v["S"] if v else None, "accepted": bool(v and v["winner"]),
                "outcome": v["outcome"] if v else c.get("outcome"),
                "detail": (v["reason"] if v else c.get("detail") or "")[:600],
                **attr,
            })
    # Recorded once: a rerun of a round cut off below doesn't record it twice.
    if not any(r["t"] == t for r in history):
        append_history(run, records)
    logged = ({json.loads(line).get("candidate") for line in evolve.LEDGER.read_text().splitlines()}
              if evolve.LEDGER.exists() else set())
    for c, ev in zip(ready, measured, strict=True):
        if ev["name"] in logged:
            continue
        v = verdicts[c["variant"]]
        evolve.append_ledger({
            "candidate": ev["name"], "incumbent": fr["incumbent"], "harness": ev["harness"],
            "component": ",".join(sorted({str(e.get("component")) for e in c["edits"]})),
            "hypothesis": "; ".join(str(e.get("hypothesis")) for e in c["edits"]),
            "S": v["S"], "C": v["C"], "delta_S": v["delta_S"], "delta_C": v["delta_C"],
            "reason": v["reason"], "delta": delta, "rsi": f"{run}/t{t}",
            "verdict": "ACCEPTED" if v["winner"] else "REJECTED"})

    winner = next((c for c in ready if verdicts[c["variant"]]["winner"]), None)
    if winner:
        ev = measured[ready.index(winner)]
        s = verdicts[winner["variant"]]["S"]
        save_json(evolve.FRONTIER, {"incumbent": ev["name"], "S_star": max(fr["S_star"], s),
                                    "metric": cfg["metric"]})
        git("push", "-q", rc["remote"], f"{winner['sha']}:refs/heads/evolve/{run}")
        print(f"round {t}: v{winner['variant']} accepted, S = {s:.4f}")
    else:
        print(f"round {t}: nothing accepted")
    state["trajectory"].append(evolve.aggregate(evolve.load_eval(
        load_json(evolve.FRONTIER)["incumbent"]), cfg["metric"])["S"])
    state["t"] = t + 1
    save_state(state)
    for c in cands:
        wt = worktree_dir(run, f"t{t}-v{c['variant']}")
        if wt.exists():
            git("worktree", "remove", "--force", str(wt))
    return True


def select(ready: list[dict], measured: list[dict], a_inc: dict, s_star: float, delta: float,
           cfg: dict) -> dict[int, dict]:
    """Each measured candidate's verdict, by variant; the winner is the
    admissible one with the best score (fewer tokens on a tie)."""
    out: dict[int, dict] = {}
    for c, ev in zip(ready, measured, strict=True):
        a = evolve.aggregate(ev, cfg["metric"])
        if a["missing"] > cfg["invalid_missing_frac"] * a["n"]:
            v = {"admissible": False, "delta_S": None, "delta_C": None, "outcome": "invalid",
                 "reason": f"{a['missing']}/{a['n']} trials have no grade: not a measurement"}
        else:
            v = evolve.judge(a, a_inc, s_star, delta, cfg)
            v["outcome"] = "admissible" if v["admissible"] else "rejected"
        out[c["variant"]] = {**v, "S": round(a["S"], 4), "C": a["C"], "winner": False}
    admissible = [k for k, v in out.items() if v["admissible"]]
    if admissible:
        best = max(admissible, key=lambda k: (out[k]["S"], -(out[k]["C"] or 0)))
        out[best]["winner"] = True
        out[best]["outcome"] = "accepted"
    return out


# --- commands ---

def cmd_start(args, cfg: dict) -> None:
    rc = rsi_config(cfg)
    if (RSI / args.run / "state.json").exists():
        raise SystemExit(f"run {args.run!r} exists: `run` continues it")
    if not evolve.FRONTIER.exists():
        raise SystemExit("no incumbent: measure a harness and make it the `baseline` with evolve.py first")
    evolve.delta_of(cfg)  # a noise band is needed to judge anything
    if rc["proposer_model"] and rc["proposer_model"] == cfg["model"]:
        print("warning: the proposer is the policy model; RRSI keeps them apart")
    fr = load_json(evolve.FRONTIER)
    inc = evolve.load_eval(fr["incumbent"])
    sha = inc["harness"]["sha"]
    git("push", "-q", rc["remote"], f"{sha}:refs/heads/evolve/{args.run}")
    state = {"run": args.run, "T": args.T or rc["T"], "t": 0, "start": fr["incumbent"], "start_sha": sha,
             "trajectory": [evolve.aggregate(inc, cfg["metric"])["S"]],
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_state(state)
    print(f"run {args.run}: {state['T']} rounds from {fr['incumbent']} ({sha[:7]}); "
          f"evolve/{args.run} tracks the incumbent")


def cmd_run(args, cfg: dict) -> None:
    rc = rsi_config(cfg)
    state = load_state(args.run)
    done = 0
    while state["t"] < state["T"] and (args.rounds is None or done < args.rounds):
        if (RSI / args.run / "STOP").exists():
            print(f"stopped: remove {RSI / args.run / 'STOP'} to go on")
            return
        if not run_round(state, cfg, rc, evaluate=not args.no_eval):
            print(f"candidates are in {RSI / args.run / f't{state["t"]}'}; `run {args.run}` measures them")
            return
        done += 1
    if state["t"] >= state["T"]:
        print(f"run {args.run} finished its {state['T']} rounds; "
              f"`evolve.py eval evolve/{args.run} --set heldout` reports the result out of sample")


def cmd_status(args, cfg: dict) -> None:
    state = load_state(args.run)
    print(f"run {args.run}: round {state['t']} of {state['T']}, from {state['start']} ({state['start_sha'][:7]})")
    print("incumbent S per round: " + " ".join(f"{s:.3f}" for s in state["trajectory"]))
    history = read_history(args.run)
    for r in history[-args.n:]:
        gain = f"dS={r['delta_S']:+.4f}" if r.get("delta_S") is not None else "unmeasured"
        print(f"  t{r['t']} v{r['variant']} {r.get('edit_id') or '-':4} {'*' if r.get('accepted') else ' '} "
              f"{gain:13} [{r.get('component') or '-'}] {(r.get('hypothesis') or '')[:80]}"
              f"  ({r.get('outcome')})")
    print(f"untried components: {', '.join(untried(history)) or 'none'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="start a run from the incumbent")
    p.add_argument("run", help="a name for the run")
    p.add_argument("--T", type=int, help="rounds (default: config)")
    p = sub.add_parser("run", help="run rounds (resumable)")
    p.add_argument("run")
    p.add_argument("--rounds", type=int, help="at most this many rounds now")
    p.add_argument("--no-eval", action="store_true", help="analyze and propose, but don't measure")
    p = sub.add_parser("status", help="where a run is, and its edit history")
    p.add_argument("run")
    p.add_argument("-n", type=int, default=20, help="history rows to show")
    args = ap.parse_args(argv)
    {"start": cmd_start, "run": cmd_run, "status": cmd_status}[args.cmd](args, evolve.load_config())


if __name__ == "__main__":
    sys.exit(main())
