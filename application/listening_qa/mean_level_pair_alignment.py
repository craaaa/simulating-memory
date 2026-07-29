#!/usr/bin/env python3
"""Rank-based (Mann-Whitney U) level-pair preference alignment (listening QA).

Variant of ``level_pair_preference_alignment.py`` that compares two cells by
their FULL raw per-trial distributions via the Mann-Whitney U probability of
superiority, PS = U / (n_a * n_b) -- the probability a random draw from cell A
beats a random draw from cell B (ties count as 0.5 pairwise) -- instead of
either collapsing each cell to a single mean or drawing one random sample per
repetition (the Monte Carlo approximation used by
``level_pair_preference_alignment.py``). PS is the closed-form, exact version
of that same estimand, so it fully uses each cell's variance without ever
reducing to a mean and without resampling noise. Deterministic (no RNG) --
one score per (topic, level_pair, condition).

A tie (PS == 0.5 on either side) scores 0.5 rather than being coin-flipped or
excluded -- the standard convention for ties in concordance/agreement metrics.

Two scopes:
- ``within_topic`` (default): base pairs are C(4,2)=6 level pairs x 4 topics = 24,
  always comparing two levels of the SAME topic.
- ``all``: base pairs are all C(16,2)=120 pairs among the full 4 topics x 4 levels
  = 16 cells, including cross-topic pairs (e.g. astronomy:control vs fabrics:distractor).
  Superset of ``within_topic``.

Usage:
    python -m application.listening_qa.mean_level_pair_alignment [--scope within_topic|all]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu

from .level_pair_preference_alignment import (
    CONDITIONS,
    DEFAULT_HUMAN_CSV,
    LEVELS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STANDALONE_JSONL = (
    REPO_ROOT / "runs" / "prompting" / "gpt-4.1" / "20260716T155755Z"
    / "tasks" / "application_listening_qa_full_grid.jsonl"
)
DEFAULT_WM_JSONL = (
    REPO_ROOT / "runs" / "compactor" / "gpt-4.1" / "n20_stream_sentence_cap30"
    / "tasks" / "wm_application_listening_qa_full_grid.jsonl"
)


def mannwhitney_ps(vals_a: list[float] | None, vals_b: list[float] | None) -> float | None:
    """Probability of superiority PS = U / (n_a * n_b): the probability a random
    draw from ``vals_a`` exceeds a random draw from ``vals_b`` (ties = 0.5).
    ``None`` if either side has no data or either side has zero variance overlap
    that scipy can't rank (e.g. n=1 on both sides is still fine; only empty
    inputs return None)."""
    if not vals_a or not vals_b:
        return None
    u_a = mannwhitneyu(vals_a, vals_b, alternative="two-sided", method="auto").statistic
    return float(u_a / (len(vals_a) * len(vals_b)))


def mannwhitney_preference(vals_a: list[float] | None, vals_b: list[float] | None) -> str | None:
    """Which side the Mann-Whitney U probability of superiority prefers:
    ``"A"``, ``"B"``, or ``"TIE"`` (PS == 0.5). ``None`` if either side has no data."""
    ps = mannwhitney_ps(vals_a, vals_b)
    if ps is None:
        return None
    if ps > 0.5:
        return "A"
    if ps < 0.5:
        return "B"
    return "TIE"


def pair_score(human_pref: str, model_pref: str) -> float:
    """Agreement score for one (human_pref, model_pref) pair: 1.0 if both sides
    agree, 0.0 if they disagree, 0.5 if either side is a tie (no preference to
    (dis)agree with)."""
    if human_pref == "TIE" or model_pref == "TIE":
        return 0.5
    return 1.0 if human_pref == model_pref else 0.0


def enumerate_cell_pairs(
    topics: list[str], *, scope: str
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Base comparisons as pairs of (topic, level) cells.

    ``scope="within_topic"``: only pairs sharing the same topic (l1 != l2).
    ``scope="all"``: every pair among all topic x level cells, topic and/or
    level may differ (includes the within_topic pairs as a subset).
    """
    if scope == "within_topic":
        pairs = []
        for topic in topics:
            for l1, l2 in combinations(LEVELS, 2):
                pairs.append(((topic, l1), (topic, l2)))
        return pairs
    if scope == "all":
        cells = [(topic, level) for topic in topics for level in LEVELS]
        return list(combinations(cells, 2))
    raise ValueError(f"Unknown scope: {scope!r}")


def run_mean_alignment(
    *,
    llm: dict[str, dict[str, dict[str, list[float]]]],
    human: dict[str, dict[str, list[float]]],
    topics: list[str],
    eval_conditions: list[str],
    scope: str = "within_topic",
) -> dict[str, Any]:
    base_pairs = enumerate_cell_pairs(topics, scope=scope)
    per_condition: dict[str, dict[str, Any]] = {
        c: {"score_sum": 0.0, "n": 0, "missing_human": 0, "missing_model": 0,
            "ties_human": 0, "ties_model": 0, "by_pair": {}, "scores": []}
        for c in eval_conditions
    }

    for (ta, la), (tb, lb) in base_pairs:
        h_vals_a = (human.get(ta) or {}).get(la)
        h_vals_b = (human.get(tb) or {}).get(lb)
        h_side = mannwhitney_preference(h_vals_a, h_vals_b)
        h_ps = mannwhitney_ps(h_vals_a, h_vals_b)
        pair_key = f"{ta}:{la}_vs_{tb}:{lb}"
        for c in eval_conditions:
            m_stats = llm.get(c, {})
            m_vals_a = (m_stats.get(ta) or {}).get(la)
            m_vals_b = (m_stats.get(tb) or {}).get(lb)
            m_side = mannwhitney_preference(m_vals_a, m_vals_b)
            m_ps = mannwhitney_ps(m_vals_a, m_vals_b)
            row = {
                "human_ps": h_ps,
                "model_ps": m_ps,
                "human_pref": h_side,
                "model_pref": m_side,
            }
            if h_side is None:
                per_condition[c]["missing_human"] += 1
            elif m_side is None:
                per_condition[c]["missing_model"] += 1
            else:
                if h_side == "TIE":
                    per_condition[c]["ties_human"] += 1
                if m_side == "TIE":
                    per_condition[c]["ties_model"] += 1
                score = pair_score(h_side, m_side)
                per_condition[c]["n"] += 1
                per_condition[c]["score_sum"] += score
                per_condition[c]["scores"].append(score)
                row["score"] = score
            per_condition[c]["by_pair"][pair_key] = row

    for c in eval_conditions:
        n = per_condition[c]["n"]
        per_condition[c]["agreement"] = (per_condition[c]["score_sum"] / n) if n else None

    return {
        "method": f"mannwhitney_ps_{scope}",
        "scope": scope,
        "n_topics": len(topics),
        "n_base_comparisons": len(base_pairs),
        "per_condition": per_condition,
    }


def bootstrap_ci_agreement(
    scores: list[float], *, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on mean agreement, resampling the per-base-pair
    scores (0 / 0.5 / 1) with replacement. NaN bounds if too few scores to resample."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def mean_split_half_baseline(
    human_csv: Path, *, scope: str = "within_topic", n_splits: int = 20, base_seed: int = 42
) -> float:
    """Human split-half reliability ceiling for the Mann-Whitney PS method: split
    participants into two random halves, treat half A as ground truth and half B
    as a synthetic model, and run the same rank-based comparison. Averaged
    over ``n_splits`` independent random splits."""
    import random as _random

    from application.plot_listening_qa_pair_level_breakdown import _load_human_by_pid

    by_pid = _load_human_by_pid(human_csv)
    pids = sorted(by_pid.keys())

    def pool(pid_set: set[str]) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        for pid in pid_set:
            for topic, levels in by_pid[pid].items():
                for level, accs in levels.items():
                    out.setdefault(topic, {}).setdefault(level, []).extend(accs)
        return out

    agreements = []
    for split_idx in range(n_splits):
        seed = base_seed + split_idx
        rng = _random.Random(seed)
        shuffled = list(pids)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        half_a = pool(set(shuffled[:half]))
        half_b = pool(set(shuffled[half:]))
        topics = sorted(set(half_a.keys()) | set(half_b.keys()))
        result = run_mean_alignment(
            llm={"HalfB": half_b},
            human=half_a,
            topics=topics,
            eval_conditions=["HalfB"],
            scope=scope,
        )
        agr = result["per_condition"]["HalfB"]["agreement"]
        if agr is not None:
            agreements.append(agr)
    return sum(agreements) / len(agreements) if agreements else float("nan")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--scope", choices=["within_topic", "all"], default="within_topic")
    args = ap.parse_args()

    human = load_human_topic_level_accuracies(args.human_csv)
    llm = load_llm_topic_level_condition(args.standalone_jsonl, args.wm_jsonl)
    topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})

    result = run_mean_alignment(
        llm=llm, human=human, topics=topics, eval_conditions=list(CONDITIONS), scope=args.scope
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_json or (
        REPO_ROOT / "application" / "comparisons" / f"listening_mean_level_pair_alignment_{stamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    print(f"\n{result['n_base_comparisons']} base comparisons (scope={result['scope']}, {result['n_topics']} topics)")
    for c in CONDITIONS:
        row = result["per_condition"][c]
        agr = row["agreement"]
        agr_str = f"{agr:.3f}" if agr is not None else "na"
        print(
            f"  {c}: agreement={agr_str} (n={row['n']}, "
            f"missing_human={row['missing_human']}, missing_model={row['missing_model']}, "
            f"ties: human={row['ties_human']} model={row['ties_model']})"
        )


if __name__ == "__main__":
    main()
