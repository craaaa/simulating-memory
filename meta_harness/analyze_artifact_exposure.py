"""How much of each candidate's gain is credit for trials the benchmark failed to collect?

UNITS. Every score here is humanlikeness = 1 - Wasserstein-1 between the model's and the
humans' per-participant score distributions. Range 0-1, higher is more human-like, in units of
task proportion-correct: 0.03 means three percentage points of distributional distance.

THE DEFECT. Until commit eb3e96f the harness showed the model tool schemas on turns that
forbade tool calls, so the model typed the call out as text instead of answering. That text
carries no classification, so the trial was recorded as UNANSWERED -- and because the model is
more accurate than humans on most of these tasks, an unanswered trial lowers its score and
therefore RAISES its measured humanlikeness. Candidates that triggered the defect were
flattered by it.

THE COUNTERFACTUAL. Score only the trials the benchmark actually collected:

    n-back            correct / answered      instead of  correct / 14
    variable_mapping  correct / (10 - spoken) instead of  correct / 10

That is what the numbers would have been if the affected trials had never been presented. It
assumes the affected trials would have been answered at the same rate as the unaffected ones,
which is why it is an ESTIMATE OF THE ARTIFACT COMPONENT and not a prediction of the post-fix
run: the fix changes what the model sees, so its behaviour can shift for other reasons too.
Read it as "how much of this gain cannot survive proper collection".

Usage:
    python meta_harness/analyze_artifact_exposure.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import score as S  # noqa: E402

from meta_harness import nback_steps as NS  # noqa: E402

Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"
CALL_RE = re.compile(r"<tool_call>")

BASELINES = [f"meta_harness/runs/iter0/baseline/{Q}",
             f"meta_harness/runs/iter8repA/baseline/{Q}",
             f"meta_harness/runs/iter8repB/baseline/{Q}"]

ARMS = {
    "baseline": BASELINES,
    "respond_first": [f"meta_harness/runs/iter6/respond_first/{Q}",
                      f"meta_harness/runs/iter6rep/respond_first/{Q}",
                      f"meta_harness/runs/iter6rep2/respond_first/{Q}"],
    "respond_only": [f"meta_harness/runs/iter6/respond_only/{Q}",
                     f"meta_harness/runs/iter6rep/respond_only/{Q}",
                     f"meta_harness/runs/iter6rep2/respond_only/{Q}"],
    "evicting_reset": [f"meta_harness/runs/iter6/evicting_reset/{Q}",
                       f"meta_harness/runs/iter7repA/evicting_reset/{Q}",
                       f"meta_harness/runs/iter7repB/evicting_reset/{Q}"],
    "primacy": [f"meta_harness/runs/iter2/primacy/{Q}",
                f"meta_harness/runs/iter7repA/primacy/{Q}",
                f"meta_harness/runs/iter7repB/primacy/{Q}"],
    "chunk_limit": [f"meta_harness/runs/iter2/chunk_limit/{Q}",
                    f"meta_harness/runs/iter7repA/chunk_limit/{Q}",
                    f"meta_harness/runs/iter7repB/chunk_limit/{Q}"],
}


def _rows(run: Path, task: str) -> list[dict]:
    f = run / "tasks" / f"wm_{task}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def nback_scores(run: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(as-scored, collected-only, n_spoken_answer_turns, n_answer_turns).

    The counterfactual excludes ONLY the trials whose own answer turn contained a spoken tool
    call -- not every unanswered trial. `acc_over_answered` would have been easier to use and
    is wrong here: a model may decline to answer for reasons that have nothing to do with the
    defect, and crediting all of those to the defect overstates it. The step log's step i is
    trial i (step 0 is the instruction turn), so the mapping is exact, and `maintenance_text`
    is excluded because it is not an answer turn.
    """
    asis, coll = [], []
    spoken = total = 0
    for r in _rows(run, "nback"):
        a14 = r.get("acc_over_14")
        if a14 is None:
            continue
        asis.append(float(a14))
        n = int(r.get("n_level") or 1)
        per_trial = r.get("per_trial") or []
        n_trials = len(per_trial) or 14
        correct = sum(1 for t in per_trial if t.get("correct")) if per_trial else None
        if correct is None:
            correct = float(a14) * n_trials

        # which trials had a spoken call in their own answer turn?
        # per_trial's trial 1 is the first SCORED trial, which is step n+1: step 0 is the
        # instruction turn and steps 1..n are the lead-in letters where "no response" is the
        # correct reply. So scored trial k is step n+k, and anything past n+n_trials is not a
        # scored trial at all.
        bad = 0
        for st in NS.scored_answer_turns(r).values():
            v = st.get("text")
            if isinstance(v, str):
                total += 1
                if CALL_RE.search(v):
                    spoken += 1
                    bad += 1
        denom = max(1, n_trials - bad)
        coll.append(min(1.0, correct / denom))
    return np.asarray(asis), np.asarray(coll), spoken, total


def vm_scores(run: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    asis, coll = [], []
    spoken = total = 0
    for r in _rows(run, "variable_mapping"):
        logs = r.get("step_logs") or []
        n_spoken = 0
        for it in logs:
            v = it.get("answer_raw")
            if isinstance(v, str):
                total += 1
                if CALL_RE.search(v):
                    n_spoken += 1
        spoken += n_spoken
        correct = sum(1 for q in (r.get("questions") or []) if q.get("correct"))
        m = r.get("metrics") or {}
        raw = m.get("score")
        n_q = len(r.get("questions") or []) or 10
        if raw is not None:
            asis.append(float(raw) / n_q)
            correct = float(raw)
        else:
            asis.append(correct / n_q)
        denom = max(1, n_q - n_spoken)
        coll.append(min(1.0, correct / denom))
    return np.asarray(asis), np.asarray(coll), spoken, total


def hl(human: np.ndarray, model: np.ndarray) -> float | None:
    if not human.size or not model.size:
        return None
    return round(S.humanlikeness(human, model), 4)


def main() -> int:
    print(__doc__.split("Usage:")[0].rstrip())
    for task, fn in (("nback", nback_scores), ("variable_mapping", vm_scores)):
        human = S.human_scores(task)
        print(f"\n{'=' * 92}\n{task}   human mean score {human.mean():.4f}, n={human.size}")
        print(f"{'arm':<16}{'spoken/replies':>16}{'as-scored':>11}{'collected':>11}"
              f"{'artifact':>10}   per-run as-scored")
        base_asis = base_coll = None
        for arm, dirs in ARMS.items():
            a_vals, c_vals, per_run = [], [], []
            sp = tot = 0
            for d in dirs:
                run = ROOT / d
                if not run.exists():
                    continue
                a, c, s, t = fn(run)
                sp += s
                tot += t
                ha, hc = hl(human, a), hl(human, c)
                if ha is not None:
                    a_vals.append(ha)
                    per_run.append(ha)
                if hc is not None:
                    c_vals.append(hc)
            if not a_vals:
                print(f"{arm:<16}  MISSING")
                continue
            ma = round(float(np.mean(a_vals)), 4)
            mc = round(float(np.mean(c_vals)), 4) if c_vals else None
            art = round(ma - mc, 4) if mc is not None else None
            if arm == "baseline":
                base_asis, base_coll = ma, mc
            runs = " ".join(f"{v:.4f}" for v in per_run)
            print(f"{arm:<16}{f'{sp}/{tot}':>16}{ma:>11.4f}"
                  f"{(f'{mc:.4f}' if mc is not None else '-'):>11}"
                  f"{(f'{art:+.4f}' if art is not None else '-'):>10}   {runs}")
        if base_asis is not None:
            print(f"\n{'arm':<16}{'delta as-scored':>17}{'delta collected':>17}"
                  f"{'gain lost':>11}")
            for arm, dirs in ARMS.items():
                if arm == "baseline":
                    continue
                a_vals, c_vals = [], []
                for d in dirs:
                    run = ROOT / d
                    if not run.exists():
                        continue
                    a, c, _, _ = fn(run)
                    ha, hc = hl(human, a), hl(human, c)
                    if ha is not None:
                        a_vals.append(ha)
                    if hc is not None:
                        c_vals.append(hc)
                if not a_vals:
                    continue
                da = float(np.mean(a_vals)) - base_asis
                dc = (float(np.mean(c_vals)) - base_coll) if c_vals and base_coll else None
                lost = (da - dc) if dc is not None else None
                print(f"{arm:<16}{da:>+17.4f}"
                      f"{(f'{dc:+.4f}' if dc is not None else '-'):>17}"
                      f"{(f'{lost:+.4f}' if lost is not None else '-'):>11}")
    print("\n'gain lost' is how much of the candidate's advantage over the baseline exists "
          "only\nbecause trials went uncollected. It is an upper bound on the artifact, not "
          "a\nprediction of the post-fix run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
