"""Score a candidate run: humanlikeness per task, plus the axes and constraints.

This is the evaluation contract in one place. It produces the record the
proposer reads, so it deliberately reports the whole per-task vector rather than
a single number -- a mean alone lets a candidate buy its headline from Variable
Mapping, which holds 0.569 of the 2.25 total task headroom on opus.

Decision rules, from domain_spec.md:

  primary      mean humanlikeness over the search tasks
  floor        no task may regress more than FLOOR below the baseline
  A2 (axis)    word-recognition miss/false-alarm ratio, humans 6.09 -- the only
               axis with real headroom on qwen3-30b
  A1 (guard)   protocol-matched digit-span sub-span leak must stay in [0.05,0.12]
  A3 (guard)   story-recall BLEU < 0.02 and recall length in [100,175] words
  covariate    word-recognition trials attempted, reported so an A2 gain that
               came from surviving longer is visible rather than banked

Usage:
    python meta_harness/score_candidate.py <run_dir> [--baseline <run_dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import score as S  # noqa: E402

from meta_harness import error_structure as ES  # noqa: E402
from meta_harness import interference as IF  # noqa: E402
from meta_harness import protocol_match as PM  # noqa: E402

HUMAN_A2_RATIO = 6.094
HUMAN_A3_BLEU = 0.002
HUMAN_A3_WORDS = 137.2

FLOOR = 0.03                      # max allowed per-task regression vs baseline

# Two task families, and the distinction matters more than any single number
# here. `reset_messages` is never called by any task, and `WorkingMemoryAgent`
# maintains conversation history across `step()` calls -- but `recall()` builds a
# fresh single prompt from the KV store alone. So:
#
#   BOTTLENECKED  encode() then recall(): the 4-slot store is the only route from
#                 stimulus to answer. The bottleneck genuinely binds.
#   LEAKY         answers via step(): the full study history is still in context,
#                 so the store can be bypassed entirely.
#
# Measured consequence on variable_mapping (leaky): 674 of 1500 questions ask
# about a name the store had already evicted, and accuracy there is 0.985. When
# the store holds a STALE value contradicting the truth, accuracy is still 0.935 --
# the model overrides its own memory. That is why capacity 4, capacity 10 000 and
# random decay all produce exactly 0.992.
#
# This is why a leaky task's humanlikeness must not be treated as a memory-module
# score: no change to the store can move it.
BOTTLENECKED_TASKS = ["digit_span_forward", "digit_span_reverse", "word_recognition",
                      "semantic_story_recall", "craft_task", "narrative_qa"]
LEAKY_TASKS = ["nback", "variable_mapping"]

# Guards are expressed as distance-from-human, not as absolute bands.
#
# An absolute band cannot work: the first version used A1 leak in [0.05, 0.12],
# calibrated on the released OpenRouter run's 0.105, and the locally measured
# baseline came in at 0.1298 -- so the baseline failed its own guard, which is
# incoherent. Absolute bands are serving-stack-dependent for the same reason the
# released headroom figures were.
#
# What a guard is actually for: catching a candidate whose error structure drifts
# FURTHER from humans than the baseline already is. So the rule is relative --
# |candidate - human| must not exceed |baseline - human| + tolerance.
HUMAN_A1_LEAK = 0.087
HUMAN_BEST_SPAN = 6.88            # humans stop here; baseline reaches 18.4
A1_LEAK_TOLERANCE = 0.03
A3_BLEU_TOLERANCE = 0.02          # absolute; human BLEU is ~0 so this is a cap
A3_WORD_TOLERANCE = 40.0          # words, around the human 137.2

# Smallest per-task delta that is not sampling noise: 2 x the bootstrap SE of a
# DIFFERENCE between two runs. From `metric_noise.py`, which resamples the model
# side because that is what actually varies between two candidates.
#
# These replace the earlier human split-half figures, which captured uncertainty
# in the reference distribution only and so understated the real imprecision by up
# to 4x. The worst case is digit_span_forward: `search_set.yaml` cuts it to 190
# rows, which score.py resolves into just **10** model participants, giving a
# min credible delta of 0.140. A floor of 0.03 there was measuring nothing.
#
# The mean is much better conditioned than any single task -- sqrt(sum of squares)/8
# gives SE(mean delta) = 0.013, so a mean delta of ~0.026 is credible. That is why
# the primary objective stays the mean and no re-run was needed to fix precision.
NOISE_FLOOR = {
    "digit_span_forward": 0.140, "digit_span_reverse": 0.059, "nback": 0.060,
    "word_recognition": 0.121, "variable_mapping": 0.017,
    "narrative_qa": 0.030, "semantic_story_recall": 0.011, "craft_task": 0.025,
    # Held-out tasks were not bootstrapped (no local run yet); keep the old
    # split-half values, which are known to be optimistic.
    "factual_qa": 0.061, "map_task": 0.054,
}
MIN_CREDIBLE_MEAN_DELTA = 0.026

SEARCH_TASKS = [
    "digit_span_forward", "digit_span_reverse", "nback", "word_recognition",
    "variable_mapping", "narrative_qa", "semantic_story_recall", "craft_task",
]
HELDOUT_TASKS = ["map_task", "factual_qa"]


def humanlikeness_by_task(run_dir: Path, tasks: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for task in tasks:
        human = S.human_scores(task)
        model = S.llm_scores(task, run_dir, "compactor")
        out[task] = (round(S.humanlikeness(human, model), 4)
                     if human.size and model.size else None)
    return out


def axes(run_dir: Path) -> dict[str, Any]:
    res: dict[str, Any] = {}

    wr = run_dir / "tasks/wm_word_recognition.jsonl"
    if wr.exists():
        miss, fa, trials = ES.a2_model(str(run_dir))
        m, f = float(np.nanmean(miss)), float(np.nanmean(fa))
        ratio = m / f if f else float("nan")
        lo, hi = ES.a2_ratio_ci(miss, fa)
        res["A2"] = {
            "miss_rate": round(m, 4), "fa_rate": round(f, 4),
            "ratio": round(ratio, 4), "ratio_ci": [round(lo, 3), round(hi, 3)],
            "human_ratio": HUMAN_A2_RATIO,
            "distance": round(abs(ratio - HUMAN_A2_RATIO), 4),
            "trials_attempted": round(float(np.mean(trials)), 2),
        }

    ds = run_dir / "tasks/wm_digit_span_forward.jsonl"
    if ds.exists():
        recs = PM.model_participants(ds)
        if recs:
            summ = PM.summarize("candidate", recs)
            leak = summ["sub_span_fail"]
            # Caveat, so a future reader does not mistake a small A1 distance for
            # reassurance: sub_span_leak is only defined where the staircase
            # actually fails. `full_context` reached best_span 20.0 on a 19-span
            # schedule -- it never terminated, so there were no failures to leak
            # and A1 read 0.032, closer to human than the baseline's 0.130 purely
            # by being uninformative. best_span is reported alongside for exactly
            # this reason, and unlike the W_1 delta it is a scalar over all 190
            # trials rather than 10 pseudo-participants, so it is the sensitive
            # digit-span regression signal (it caught random_decay at 18.4 -> 2.0).
            res["A1"] = {
                "sub_span_leak": round(leak, 4),
                "best_span": round(summ["best_span"], 2),
                "human_leak": HUMAN_A1_LEAK,
                "human_best_span": HUMAN_BEST_SPAN,
                "best_span_distance": round(abs(summ["best_span"] - HUMAN_BEST_SPAN), 2),
                "distance": round(abs(leak - HUMAN_A1_LEAK), 4),
                "at_ceiling": bool(summ["best_span"] >= 19.0),
            }

    sr = run_dir / "tasks/wm_semantic_story_recall.jsonl"
    if sr.exists():
        vals = ES.a3_model(str(run_dir))
        if vals:
            bleu = float(np.nanmean([v[0] for v in vals]))
            words = float(np.nanmean([v[2] for v in vals]))
            res["A3"] = {
                "bleu": round(bleu, 4), "words": round(words, 1),
                "human_bleu": HUMAN_A3_BLEU, "human_words": HUMAN_A3_WORDS,
                "bleu_distance": round(abs(bleu - HUMAN_A3_BLEU), 4),
                "word_distance": round(abs(words - HUMAN_A3_WORDS), 1),
            }

    # A4 closes the one cell a candidate could otherwise win by pure noise:
    # variable_mapping has the largest headroom and the best precision, and A1/A2/A3
    # do not touch it. See meta_harness/interference.py for the human reference.
    if (run_dir / "tasks/wm_variable_mapping.jsonl").exists():
        res["A4"] = IF.a4(run_dir)
    return res


def evaluate(run_dir: Path, baseline_dir: Path | None) -> dict[str, Any]:
    per_task = humanlikeness_by_task(run_dir, SEARCH_TASKS + HELDOUT_TASKS)
    search_vals = [v for t, v in per_task.items() if t in SEARCH_TASKS and v is not None]

    rec: dict[str, Any] = {
        "run_dir": str(run_dir),
        "humanlikeness_by_task": per_task,
        "mean_humanlikeness_search": (round(float(np.mean(search_vals)), 4)
                                      if search_vals else None),
        "n_search_tasks_scored": len(search_vals),
        "axes": axes(run_dir),
    }

    if baseline_dir is not None:
        base = humanlikeness_by_task(baseline_dir, SEARCH_TASKS + HELDOUT_TASKS)
        rec["baseline_dir"] = str(baseline_dir)
        deltas, violations, noise = {}, [], []
        for task, val in per_task.items():
            bv = base.get(task)
            if val is None or bv is None:
                continue
            d = round(val - bv, 4)
            deltas[task] = d
            # A regression cannot be charged against a candidate if it is smaller
            # than the precision with which the task can be measured at all --
            # otherwise digit_span_forward (min credible delta 0.140) would reject
            # candidates for noise. So the effective floor is whichever is larger.
            eff = max(FLOOR, NOISE_FLOOR.get(task, 0.05))
            if task in SEARCH_TASKS and d < -eff:
                violations.append({"task": task, "delta": d, "floor": -eff})
            if abs(d) < NOISE_FLOOR.get(task, 0.05):
                noise.append(task)
        rec["delta_vs_baseline"] = deltas
        rec["floor_violations"] = violations
        rec["within_noise_floor"] = noise
        rec["passes_floor"] = not violations

    # Guards compare distance-from-human against the BASELINE's distance, so a
    # candidate is only flagged for drifting further from humans than the
    # baseline already sits. Without a baseline there is nothing to drift from,
    # so the guards are reported and not enforced.
    guards: list[str] = []
    base_axes = axes(baseline_dir) if baseline_dir is not None else {}
    for key, tol, field in (("A1", A1_LEAK_TOLERANCE, "distance"),
                            ("A3", A3_BLEU_TOLERANCE, "bleu_distance"),
                            ("A3", A3_WORD_TOLERANCE, "word_distance")):
        cand_ax, base_ax = rec["axes"].get(key), base_axes.get(key)
        if not cand_ax or not base_ax:
            continue
        cd, bd = cand_ax.get(field), base_ax.get(field)
        if cd is None or bd is None:
            continue
        if cd > bd + tol:
            guards.append(
                f"{key} {field}: {cd} vs baseline {bd} (tolerance {tol}) -- "
                f"drifted further from human"
            )
    # A4 is conditional, unlike the others. variable_mapping is leaky -- the store
    # can be bypassed -- so a candidate that does not move it owes no explanation.
    # But a candidate that DOES move it has manufactured errors, and then the
    # question is whether those errors are interference-shaped like a human's
    # (rc_ratio 1.386) or rate-like a noise injector's (rc_ratio -> 1.0).
    vm_delta = (rec.get("delta_vs_baseline") or {}).get("variable_mapping")
    a4 = rec["axes"].get("A4") or {}
    if vm_delta is not None and vm_delta > NOISE_FLOOR["variable_mapping"]:
        if not a4.get("trustworthy"):
            guards.append(
                f"A4: variable_mapping improved by {vm_delta} but only "
                f"{a4.get('n_errors')} errors were produced, too few to tell "
                f"interference from noise (need >=30)"
            )
        elif a4.get("rc_ratio") is not None and a4["rc_ratio"] < 1.15:
            guards.append(
                f"A4: variable_mapping improved by {vm_delta} with rc_ratio "
                f"{a4['rc_ratio']} -- errors are independent of interference load "
                f"(human 1.386, pure noise 1.0), so the gain is unstructured"
            )

    rec["guard_violations"] = guards
    rec["passes_guards"] = not guards
    rec["guards_enforced"] = baseline_dir is not None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--record", action="store_true",
                    help="append to logs/evolution_summary.jsonl, which is what "
                         "history.py and the proposer read")
    ap.add_argument("--id", default=None, help="candidate id for the record")
    ap.add_argument("--iteration", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rec = evaluate(run_dir, Path(args.baseline) if args.baseline else None)

    # Pull identity and provenance from the manifest the runner wrote, so a
    # record cannot disagree with the code that produced it.
    #
    # The run_dir passed in is bench's per-model output dir (.../<cand>/<model>/),
    # but run_candidate.py writes the manifest one level up, beside the candidate
    # source. Check both, or every record silently falls back to the MODEL name as
    # its id -- which collides across candidates and, with --record replacing rows
    # by id, would overwrite a previously scored candidate.
    manifest_path = next(
        (p for p in (run_dir / "manifest.json", run_dir.parent / "manifest.json")
         if p.exists()),
        None,
    )
    if manifest_path is not None:
        man = json.loads(manifest_path.read_text())
        rec["id"] = args.id or man.get("id") or run_dir.name
        rec["parent"] = man.get("parent")
        rec["capacity"] = man.get("capacity")
        rec["decay"] = man.get("decay")
        rec["role"] = man.get("role")
        rec["candidate_summary"] = man.get("summary")
        rec["injection"] = man.get("injection")
        rec["elapsed_seconds"] = man.get("elapsed_seconds")
    else:
        rec["id"] = args.id or run_dir.name
    rec["iteration"] = args.iteration

    text = json.dumps(rec, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)

    if args.record:
        summary = ROOT / "meta_harness/logs/evolution_summary.jsonl"
        summary.parent.mkdir(parents=True, exist_ok=True)
        # Replace any existing row for this id rather than appending a duplicate,
        # so re-scoring a candidate does not create two conflicting records.
        rows = []
        if summary.exists():
            for line in summary.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("id")) != str(rec["id"]):
                    rows.append(r)
        rows.append(rec)
        summary.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"\nrecorded {rec['id']} -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
