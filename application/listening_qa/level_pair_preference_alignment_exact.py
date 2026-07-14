"""Exact (non-Monte-Carlo) version of ``level_pair_preference_alignment.py``.

Same design -- within-topic level-pair preference agreement vs. human -- but
replaces "draw one random sample from each side, repeat 2000x, coin-flip
ties" with the exact closed-form probability of superiority (Cliff's-delta
style: for two finite samples, P(a>b), P(a<b), P(a==b) computed by full
pairwise count, O(n*m) per cell -- trivial at these sample sizes).

Why: the individual-draw method's coin-flip on ties injects Monte Carlo RNG
noise, which is severe when accuracy is discretized to few possible values
(e.g. gpt-4.1-mini's near-ceiling data is ~2 distinct values per cell, so
64-88% of draws tie and get a random coin flip). The exact method has no
sampling noise -- ties contribute their exact 0.5 credit deterministically,
matching what the Monte Carlo estimator converges to as
``n_samples_per_pair -> infinity``, computed directly instead of approximated.

Uncertainty comes from a bootstrap over the raw per-trial data (resample
each cell's raw trials with replacement, recompute the exact statistic),
not from a binomial CI on draw count -- there's no draw count here, so a
Wilson CI doesn't apply.

Usage:
    python -m application.listening_qa.level_pair_preference_alignment_exact \\
        --standalone-jsonl runs/prompting/openai_gpt-4.1-mini/tasks/application_listening_qa_full_grid.jsonl \\
        --wm-jsonl runs/compactor/openai_gpt-4.1-mini/tasks/wm_application_listening_qa_full_grid.jsonl \\
        --out-json application/comparisons/listening_level_pair_alignment_exact_gpt-4.1-mini.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .level_pair_preference_alignment import (
    CONDITIONS,
    DEFAULT_HUMAN_CSV,
    DEFAULT_STANDALONE_JSONL,
    DEFAULT_WM_JSONL,
    LEVELS,
    _rng_for_seed,
    enumerate_base_pairs,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)


def exact_prob_a_gt_b(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    """Exact (p_a_gt_b, p_b_gt_a, p_tie) over the full n*m cross-product."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n, m = len(a_arr), len(b_arr)
    if n == 0 or m == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a_arr[:, None] - b_arr[None, :]
    total = n * m
    p_gt = float(np.sum(diff > 0)) / total
    p_lt = float(np.sum(diff < 0)) / total
    p_tie = float(np.sum(diff == 0)) / total
    return p_gt, p_lt, p_tie


def side_prob_a(a: List[float], b: List[float]) -> float:
    """P(side == 'A') under the same tie-coin-flip convention as the Monte Carlo version,
    computed exactly: P(a>b) + 0.5 * P(a==b)."""
    p_gt, _, p_tie = exact_prob_a_gt_b(a, b)
    if np.isnan(p_gt):
        return float("nan")
    return p_gt + 0.5 * p_tie


def exact_agreement_for_pair(
    human_a: List[float], human_b: List[float], model_a: List[float], model_b: List[float]
) -> float:
    """P(human side == model side), treating each side's draw as independent."""
    h_probA = side_prob_a(human_a, human_b)
    m_probA = side_prob_a(model_a, model_b)
    if np.isnan(h_probA) or np.isnan(m_probA):
        return float("nan")
    return h_probA * m_probA + (1 - h_probA) * (1 - m_probA)


def _resample(values: List[float], rng: np.random.Generator) -> List[float]:
    if not values:
        return values
    idx = rng.integers(0, len(values), size=len(values))
    return [values[i] for i in idx]


def compute_condition_agreement(
    base_pairs: List[Tuple[str, str, str]],
    human: Dict[str, Dict[str, List[float]]],
    llm_cond: Dict[str, Dict[str, List[float]]],
) -> Tuple[float, List[Dict[str, object]]]:
    per_pair = []
    vals = []
    for topic, l1, l2 in base_pairs:
        ha = human.get(topic, {}).get(l1) or []
        hb = human.get(topic, {}).get(l2) or []
        ma = llm_cond.get(topic, {}).get(l1) or []
        mb = llm_cond.get(topic, {}).get(l2) or []
        if not ha or not hb or not ma or not mb:
            continue
        agree = exact_agreement_for_pair(ha, hb, ma, mb)
        per_pair.append({"topic": topic, "level_pair": f"{l1}_vs_{l2}", "agreement": agree})
        vals.append(agree)
    overall = float(np.mean(vals)) if vals else float("nan")
    return overall, per_pair


def bootstrap_ci(
    base_pairs: List[Tuple[str, str, str]],
    human: Dict[str, Dict[str, List[float]]],
    llm_cond: Dict[str, Dict[str, List[float]]],
    *,
    seed: int,
    label: str,
    n_boot: int = 2000,
) -> Tuple[float, float]:
    boot_means = []
    for b in range(n_boot):
        rng = _rng_for_seed(seed, f"{label}_boot_{b}")
        vals = []
        for topic, l1, l2 in base_pairs:
            ha = human.get(topic, {}).get(l1) or []
            hb = human.get(topic, {}).get(l2) or []
            ma = llm_cond.get(topic, {}).get(l1) or []
            mb = llm_cond.get(topic, {}).get(l2) or []
            if not ha or not hb or not ma or not mb:
                continue
            agree = exact_agreement_for_pair(
                _resample(ha, rng), _resample(hb, rng), _resample(ma, rng), _resample(mb, rng)
            )
            vals.append(agree)
        if vals:
            boot_means.append(float(np.mean(vals)))
    if not boot_means:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    human = load_human_topic_level_accuracies(args.human_csv)
    llm = load_llm_topic_level_condition(args.standalone_jsonl, args.wm_jsonl)
    topics = sorted(set(human.keys()) | {t for c in llm.values() for t in c.keys()})
    base_pairs = enumerate_base_pairs(topics)

    eval_conditions = ["C1", "C2", "C3", "C4", "WM"]
    results = {}
    print(f"{'condition':<10}{'agreement':>12}{'95% CI (bootstrap)':>22}")
    for cond in eval_conditions:
        overall, per_pair = compute_condition_agreement(base_pairs, human, llm.get(cond, {}))
        lo, hi = bootstrap_ci(base_pairs, human, llm.get(cond, {}), seed=args.seed, label=f"exact_{cond}", n_boot=args.n_boot)
        results[cond] = {"agreement": overall, "ci_lo": lo, "ci_hi": hi, "by_pair": per_pair}
        print(f"{cond:<10}{overall:>12.4f}   [{lo:.4f}, {hi:.4f}]")

    out = {
        "meta": {
            "method": "exact_cliffs_delta_no_monte_carlo",
            "n_base_pairs": len(base_pairs),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "standalone_jsonl": str(args.standalone_jsonl),
            "wm_jsonl": str(args.wm_jsonl),
            "human_csv": str(args.human_csv),
        },
        "results": results,
    }
    out_path = args.out_json or Path("application/comparisons/listening_level_pair_alignment_exact.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
