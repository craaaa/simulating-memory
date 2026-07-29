#!/usr/bin/env python3
"""Bradley-Terry pooled level-strength estimation (listening QA).

The mean/Mann-Whitney level-pair alignment scripts estimate each of the 24
(topic, level_pair) comparisons independently -- with only ~3-13 raw samples
per cell, each comparison is individually noisy, which is why their bootstrap
CIs are wide. Bradley-Terry pooling fits ONE latent strength beta_level per
level (not per topic x level), using ALL 4 topics' comparisons for that level
pair as repeated, independent evidence for the same underlying beta_level_i -
beta_level_j difference. This borrows statistical power across topics instead
of estimating each pair in isolation.

Model: for topic t and levels i != j, compute the Mann-Whitney probability of
superiority PS_t(i,j) from the RAW per-trial samples (same PS estimand as
``mean_level_pair_alignment.mannwhitney_ps``). Bradley-Terry says
logit(P(i beats j)) = beta_i - beta_j, so fit

    logit(PS_t(i, j)) = beta_i - beta_j + noise

by weighted least squares across all (topic, level_pair) rows, with one
reference level pinned to beta=0 for identifiability. Weight per row is the
harmonic-mean-ish precision proxy n_i*n_j/(n_i+n_j) (more samples -> more
weight). Confidence intervals come from a nonparametric bootstrap: resample
the raw per-trial values within each (topic, level) cell with replacement,
recompute all PS values and refit, repeat many times, take percentiles.

Usage:
    python -m application.listening_qa.bradley_terry_level_strength
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
from .mean_level_pair_alignment import DEFAULT_STANDALONE_JSONL, DEFAULT_WM_JSONL

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_LEVEL = "control"
PS_CLIP = 1e-3  # keep logit finite


def _ps_and_weight(vals_a: list[float] | None, vals_b: list[float] | None) -> tuple[float, float] | None:
    """(PS, weight) for cell A vs cell B, or None if either side has no data."""
    if not vals_a or not vals_b:
        return None
    n_a, n_b = len(vals_a), len(vals_b)
    u_a = mannwhitneyu(vals_a, vals_b, alternative="two-sided", method="auto").statistic
    ps = float(u_a / (n_a * n_b))
    weight = (n_a * n_b) / (n_a + n_b)  # harmonic-mean-like precision proxy
    return ps, weight


def _logit(p: float) -> float:
    p = min(max(p, PS_CLIP), 1.0 - PS_CLIP)
    return float(np.log(p / (1.0 - p)))


def fit_bt_strengths(
    topic_level_values: dict[str, dict[str, list[float]]],
    topics: list[str],
    *,
    levels: tuple[str, ...] = LEVELS,
    reference_level: str = REFERENCE_LEVEL,
) -> dict[str, float] | None:
    """Pool within-topic level-pair comparisons across all ``topics`` into one
    beta_level per level (reference_level pinned to 0.0). Returns None if there
    isn't enough data to fit (fewer than 2 free levels with usable rows)."""
    free_levels = [lv for lv in levels if lv != reference_level]
    col_index = {lv: i for i, lv in enumerate(free_levels)}

    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_w: list[float] = []

    for topic in topics:
        cell = topic_level_values.get(topic) or {}
        for li, lj in combinations(levels, 2):
            out = _ps_and_weight(cell.get(li), cell.get(lj))
            if out is None:
                continue
            ps, weight = out
            x = np.zeros(len(free_levels), dtype=np.float64)
            if li in col_index:
                x[col_index[li]] += 1.0
            if lj in col_index:
                x[col_index[lj]] -= 1.0
            rows_x.append(x)
            rows_y.append(_logit(ps))
            rows_w.append(weight)

    if len(rows_x) < len(free_levels):
        return None

    X = np.stack(rows_x)
    y = np.asarray(rows_y, dtype=np.float64)
    w = np.sqrt(np.asarray(rows_w, dtype=np.float64))
    Xw = X * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = {reference_level: 0.0}
    for lv, i in col_index.items():
        out[lv] = float(beta[i])
    return out


def bootstrap_bt_strengths(
    topic_level_values: dict[str, dict[str, list[float]]],
    topics: list[str],
    *,
    levels: tuple[str, ...] = LEVELS,
    reference_level: str = REFERENCE_LEVEL,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Nonparametric bootstrap CI on each beta_level: resample raw per-trial
    values within each (topic, level) cell with replacement, refit, repeat."""
    rng = np.random.default_rng(seed)
    free_levels = [lv for lv in levels if lv != reference_level]
    draws: dict[str, list[float]] = {lv: [] for lv in free_levels}

    for _ in range(n_boot):
        resampled: dict[str, dict[str, list[float]]] = {}
        for topic in topics:
            cell = topic_level_values.get(topic) or {}
            resampled[topic] = {}
            for lv in levels:
                vals = cell.get(lv)
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                idx = rng.integers(0, arr.size, size=arr.size)
                resampled[topic][lv] = arr[idx].tolist()
        fit = fit_bt_strengths(resampled, topics, levels=levels, reference_level=reference_level)
        if fit is None:
            continue
        for lv in free_levels:
            draws[lv].append(fit[lv])

    out: dict[str, tuple[float, float]] = {reference_level: (0.0, 0.0)}
    for lv in free_levels:
        if not draws[lv]:
            out[lv] = (float("nan"), float("nan"))
            continue
        lo, hi = np.percentile(draws[lv], [2.5, 97.5])
        out[lv] = (float(lo), float(hi))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    human = load_human_topic_level_accuracies(args.human_csv)
    llm = load_llm_topic_level_condition(args.standalone_jsonl, args.wm_jsonl)
    topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})

    result: dict[str, Any] = {"topics": topics, "reference_level": REFERENCE_LEVEL, "conditions": {}}

    print("Fitting Bradley-Terry pooled level strengths (human)...")
    human_beta = fit_bt_strengths(human, topics)
    human_ci = bootstrap_bt_strengths(human, topics, n_boot=args.n_boot)
    result["conditions"]["human"] = {"beta": human_beta, "ci": human_ci}
    for lv in LEVELS:
        b = human_beta[lv]
        lo, hi = human_ci[lv]
        print(f"  {lv}: beta={b:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    for c in CONDITIONS:
        print(f"Fitting Bradley-Terry pooled level strengths ({c})...")
        cell = llm.get(c, {})
        beta = fit_bt_strengths(cell, topics)
        if beta is None:
            print(f"  {c}: insufficient data")
            result["conditions"][c] = {"beta": None, "ci": None}
            continue
        ci = bootstrap_bt_strengths(cell, topics, n_boot=args.n_boot)
        result["conditions"][c] = {"beta": beta, "ci": ci}
        for lv in LEVELS:
            b = beta[lv]
            lo, hi = ci[lv]
            print(f"  {lv}: beta={b:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_json or (
        REPO_ROOT / "application" / "comparisons" / f"listening_bt_level_strength_{stamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
