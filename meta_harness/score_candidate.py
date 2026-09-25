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
from meta_harness import protocol_match as PM  # noqa: E402

HUMAN_A2_RATIO = 6.094
HUMAN_A3_BLEU = 0.002
HUMAN_A3_WORDS = 137.2

FLOOR = 0.03                      # max allowed per-task regression vs baseline
A1_LEAK_BAND = (0.05, 0.12)       # protocol-matched digit-span sub-span leak
A3_BLEU_MAX = 0.02
A3_WORD_BAND = (100.0, 175.0)

# Split-half human noise floor per task; a delta under this is not a result.
NOISE_FLOOR = {
    "digit_span_forward": 0.036, "digit_span_reverse": 0.026, "nback": 0.025,
    "word_recognition": 0.075, "variable_mapping": 0.041, "factual_qa": 0.061,
    "narrative_qa": 0.053, "semantic_story_recall": 0.042, "map_task": 0.054,
    "craft_task": 0.041,
}

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
            res["A1"] = {
                "sub_span_leak": round(leak, 4),
                "best_span": round(summ["best_span"], 2),
                "band": list(A1_LEAK_BAND),
                "within_band": bool(A1_LEAK_BAND[0] <= leak <= A1_LEAK_BAND[1]),
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
                "bleu_ok": bool(bleu < A3_BLEU_MAX),
                "words_ok": bool(A3_WORD_BAND[0] <= words <= A3_WORD_BAND[1]),
            }
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
            if task in SEARCH_TASKS and d < -FLOOR:
                violations.append({"task": task, "delta": d, "floor": -FLOOR})
            if abs(d) < NOISE_FLOOR.get(task, 0.05):
                noise.append(task)
        rec["delta_vs_baseline"] = deltas
        rec["floor_violations"] = violations
        rec["within_noise_floor"] = noise
        rec["passes_floor"] = not violations

    guards = []
    a1 = rec["axes"].get("A1")
    if a1 and not a1["within_band"]:
        guards.append(f"A1 sub-span leak {a1['sub_span_leak']} outside {A1_LEAK_BAND}")
    a3 = rec["axes"].get("A3")
    if a3 and not (a3["bleu_ok"] and a3["words_ok"]):
        guards.append(f"A3 bleu={a3['bleu']} words={a3['words']} outside bounds")
    rec["guard_violations"] = guards
    rec["passes_guards"] = not guards
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rec = evaluate(Path(args.run_dir), Path(args.baseline) if args.baseline else None)
    text = json.dumps(rec, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
