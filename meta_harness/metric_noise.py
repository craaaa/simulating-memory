"""How precisely can we even measure humanlikeness, per task?

The noise floor in `score_candidate.py` came from splitting the HUMAN sample in
half. That captures uncertainty in the reference distribution and nothing else. It
misses the other source, which turns out to be the larger one: the model side is
also a finite sample, and for the digit-span tasks it is a *very* small one.

`search_set.yaml` cuts digit span from 1900 rows to 190 (19 spans x 10 sequences)
because forward and reverse together are 85% of all LLM turns in a full
evaluation. That was a deliberate and, for throughput, correct trade. But the
consequence for the metric was never checked: `score.py` resolves those 190 rows
into **10** model participants, which are then compared against 52 humans via
W_1. A W_1 estimated from 10 draws is not precise, and a per-task floor of 0.03
is far below that imprecision -- so a candidate could be rejected, or credited,
for sampling noise.

This bootstraps the model side to get the real per-task standard error, which is
what a floor has to respect. Run:

    python meta_harness/metric_noise.py
    python meta_harness/metric_noise.py --write   # update NOISE_FLOOR in place

Interpretation: a per-task delta smaller than ~2x the reported SE is not evidence
of anything, whatever the sign.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import score as S  # noqa: E402

N_BOOT = 2000
SEED = 20260925

TASKS = [
    "digit_span_forward", "digit_span_reverse", "nback", "word_recognition",
    "variable_mapping", "narrative_qa", "semantic_story_recall", "craft_task",
]


def bootstrap_task(task: str, run_dir: Path, n_boot: int = N_BOOT) -> dict | None:
    human = S.human_scores(task)
    model = S.llm_scores(task, run_dir, "compactor")
    if not human.size or not model.size:
        return None

    point = S.humanlikeness(human, model)
    rng = np.random.default_rng(SEED)
    h = np.asarray(human, float)
    m = np.asarray(model, float)

    # Resample BOTH sides. A candidate comparison is affected by model-side noise
    # every run, and by human-side noise once (the reference is fixed), so the
    # both-sides SE is the conservative figure to set a floor from.
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = S.humanlikeness(
            rng.choice(h, size=h.size, replace=True),
            rng.choice(m, size=m.size, replace=True),
        )

    # Model-side only: this is the part that actually varies between two runs of
    # two different candidates, so it is the right figure for a per-task DELTA.
    mboots = np.empty(n_boot)
    for i in range(n_boot):
        mboots[i] = S.humanlikeness(h, rng.choice(m, size=m.size, replace=True))

    se_delta = float(np.std(mboots)) * np.sqrt(2)  # two independent runs differenced
    return {
        "task": task,
        "n_human": int(h.size),
        "n_model": int(m.size),
        "humanlikeness": round(float(point), 4),
        "se_both": round(float(np.std(boots)), 4),
        "se_model_only": round(float(np.std(mboots)), 4),
        "se_of_delta": round(se_delta, 4),
        "min_credible_delta": round(2 * se_delta, 4),
        "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                 round(float(np.percentile(boots, 97.5)), 4)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default="meta_harness/runs/iter0/baseline/"
                            "Qwen_Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--json-out", default="meta_harness/logs/metric_noise.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = [r for t in TASKS
            if (r := bootstrap_task(t, run_dir, args.n_boot)) is not None]

    hdr = f"{'task':24s} {'n_h':>4s} {'n_m':>4s} {'score':>7s} {'SE':>7s} {'SE(d)':>7s} {'min|d|':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['task']:24s} {r['n_human']:4d} {r['n_model']:4d} "
              f"{r['humanlikeness']:7.3f} {r['se_model_only']:7.3f} "
              f"{r['se_of_delta']:7.3f} {r['min_credible_delta']:7.3f}")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.json_out}")
    print("\nNOISE_FLOOR = {")
    for r in rows:
        print(f'    "{r["task"]}": {r["min_credible_delta"]},')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
