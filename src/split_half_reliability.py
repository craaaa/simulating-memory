"""Split-half reliability ceiling for the humanlikeness (1 - W_1) metric.

Randomly splits each task's human sample into two halves many times, scores
humanlikeness of half A vs half B each time, and reports the mean/CI. This
is the noise ceiling induced by sample size alone -- no model can exceed it
in expectation, since even human-vs-human comparisons don't hit 1.0.

Usage:
    python src/split_half_reliability.py [--n-splits 2000] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

from score import (
    TASK_DISPLAY,
    TASKS,
    build_table,
    discover_models,
    human_scores,
    humanlikeness,
    llm_scores,
    wasserstein_1d,
)


def split_half_humanlikeness(scores: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    n = scores.size
    if n < 4:
        return np.asarray([], dtype=np.float64)
    half = n // 2
    out = np.empty(n_splits, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n_splits):
        rng.shuffle(idx)
        a, b = scores[idx[:half]], scores[idx[half:]]
        out[i] = humanlikeness(a, b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-model-dir", type=str, default=None,
                     help="Optional model_dir (e.g. runs/prompting/gpt-4.1) to show actual "
                          "humanlikeness alongside the split-half ceiling.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"{'task':<26}  {'n_human':>7}  {'mean_HL':>9}  {'sd_HL':>7}  {'95% CI':>17}")
    print("-" * 72)
    for t in TASKS:
        h = human_scores(t)
        dist = split_half_humanlikeness(h, args.n_splits, rng)
        if dist.size == 0:
            print(f"{TASK_DISPLAY[t]:<26}  {h.size:>7}  {'n/a':>9}")
            continue
        mean_hl = float(np.mean(dist))
        sd_hl = float(np.std(dist))
        lo, hi = np.percentile(dist, [2.5, 97.5])
        print(
            f"{TASK_DISPLAY[t]:<26}  {h.size:>7}  {mean_hl:>9.3f}  {sd_hl:>7.3f}  "
            f"[{lo:.3f}, {hi:.3f}]"
        )

    if args.compare_model_dir:
        from pathlib import Path

        model_dir = Path(args.compare_model_dir)
        print()
        print(f"# vs. model at {model_dir} (condition C2 / compactor)")
        print(f"{'task':<26}  {'model_HL':>9}  {'split_half_ceiling':>19}")
        print("-" * 60)
        is_compactor = "compactor" in str(model_dir)
        cond = "compactor" if is_compactor else "C2"
        for t in TASKS:
            h = human_scores(t)
            m = llm_scores(t, model_dir, cond)
            if h.size < 4 or m.size == 0:
                continue
            model_hl = humanlikeness(h, m)
            dist = split_half_humanlikeness(h, args.n_splits, rng)
            ceiling = float(np.mean(dist)) if dist.size else float("nan")
            print(f"{TASK_DISPLAY[t]:<26}  {model_hl:>9.3f}  {ceiling:>19.3f}")


if __name__ == "__main__":
    main()
