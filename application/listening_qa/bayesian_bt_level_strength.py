#!/usr/bin/env python3
"""Hierarchical (partial-pooling) Bayesian Bradley-Terry level strength.

``bradley_terry_level_strength.fit_bt_strengths`` does FULL pooling: one
beta_level shared exactly across all 4 topics, fit by weighted least squares,
with uncertainty from a nonparametric bootstrap bolted on afterward.

This module instead fits a proper hierarchical model with PARTIAL pooling:
each topic gets its own beta_topic_level, drawn from a shared Normal prior
centered on a global beta_level with learned spread tau --

    beta_global_level ~ Normal(0, 2)                      (per level, ref=0)
    tau               ~ HalfNormal(1)                      (shared)
    beta_topic_level  ~ Normal(beta_global_level, tau)      (per topic x level)
    logit(PS_topic(i,j)) ~ Normal(beta_topic_i - beta_topic_j, sigma_obs / sqrt(weight))

Topics with more/cleaner data pull tau's implied shrinkage less; sparse or
noisy topics get pulled harder toward the shared beta_global. tau itself is
estimated from the data, so the amount of pooling is adaptive rather than
fixed like the "full pooling" (one shared value) vs "no pooling" (independent
per-topic fits) extremes.

Returns posterior mean beta_global_level + 95% credible interval (highest
density interval isn't computed here for simplicity -- percentile interval).

Usage:
    python -m application.listening_qa.bayesian_bt_level_strength
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .bradley_terry_level_strength import PS_CLIP, REFERENCE_LEVEL, _ps_and_weight
from .level_pair_preference_alignment import (
    CONDITIONS,
    DEFAULT_HUMAN_CSV,
    LEVELS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)
from .mean_level_pair_alignment import DEFAULT_STANDALONE_JSONL, DEFAULT_WM_JSONL

REPO_ROOT = Path(__file__).resolve().parents[2]


def _logit(p: float) -> float:
    p = min(max(p, PS_CLIP), 1.0 - PS_CLIP)
    return float(np.log(p / (1.0 - p)))


def fit_bt_strengths_bayesian(
    topic_level_values: dict[str, dict[str, list[float]]],
    topics: list[str],
    *,
    levels: tuple[str, ...] = LEVELS,
    reference_level: str = REFERENCE_LEVEL,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 42,
    return_hyperparams: bool = False,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]] | tuple[dict[str, float], dict[str, tuple[float, float]], dict[str, float]] | None:
    """Partial-pooling hierarchical Bradley-Terry fit. Returns
    (posterior_mean_beta_global, 95%_credible_interval) per level, or None if
    there isn't enough data to build any comparison rows. If
    ``return_hyperparams``, also returns a dict with posterior means of
    ``tau`` (topic-to-topic spread) and ``sigma_obs`` (per-comparison noise
    scale) -- diagnostic for what's driving a wide/narrow credible interval."""
    free_levels = [lv for lv in levels if lv != reference_level]
    n_free = len(free_levels)
    ref_idx = levels.index(reference_level)

    topic_idx_list: list[int] = []
    i_idx_list: list[int] = []
    j_idx_list: list[int] = []
    y_list: list[float] = []
    w_list: list[float] = []

    topic_index = {t: ti for ti, t in enumerate(topics)}
    level_index = {lv: li for li, lv in enumerate(levels)}

    for topic in topics:
        cell = topic_level_values.get(topic) or {}
        for li, lj in combinations(levels, 2):
            out = _ps_and_weight(cell.get(li), cell.get(lj))
            if out is None:
                continue
            ps, weight = out
            topic_idx_list.append(topic_index[topic])
            i_idx_list.append(level_index[li])
            j_idx_list.append(level_index[lj])
            y_list.append(_logit(ps))
            w_list.append(weight)

    if not y_list:
        return None

    n_topics = len(topics)
    topic_idx_arr = np.asarray(topic_idx_list, dtype="int64")
    i_idx_arr = np.asarray(i_idx_list, dtype="int64")
    j_idx_arr = np.asarray(j_idx_list, dtype="int64")
    y_arr = np.asarray(y_list, dtype=np.float64)
    w_arr = np.asarray(w_list, dtype=np.float64)

    with pm.Model():
        beta_global_free = pm.Normal("beta_global_free", mu=0.0, sigma=2.0, shape=n_free)
        tau = pm.HalfNormal("tau", sigma=1.0)
        beta_topic_free = pm.Normal(
            "beta_topic_free", mu=beta_global_free, sigma=tau, shape=(n_topics, n_free)
        )

        cols = []
        free_i = 0
        for li in range(len(levels)):
            if li == ref_idx:
                cols.append(pt.zeros((n_topics, 1)))
            else:
                cols.append(beta_topic_free[:, free_i : free_i + 1])
                free_i += 1
        beta_topic_full = pt.concatenate(cols, axis=1)  # (n_topics, n_levels)

        diff = beta_topic_full[topic_idx_arr, i_idx_arr] - beta_topic_full[topic_idx_arr, j_idx_arr]
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        obs_sigma = sigma_obs / pt.sqrt(w_arr)
        pm.Normal("y_obs", mu=diff, sigma=obs_sigma, observed=y_arr)

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=0.9,
            progressbar=False, random_seed=seed,
        )

    post = idata.posterior["beta_global_free"].values.reshape(-1, n_free)
    means: dict[str, float] = {reference_level: 0.0}
    ci: dict[str, tuple[float, float]] = {reference_level: (0.0, 0.0)}
    for fi, lv in enumerate(free_levels):
        means[lv] = float(post[:, fi].mean())
        lo, hi = np.percentile(post[:, fi], [2.5, 97.5])
        ci[lv] = (float(lo), float(hi))
    if return_hyperparams:
        hyper = {
            "tau_mean": float(idata.posterior["tau"].values.mean()),
            "sigma_obs_mean": float(idata.posterior["sigma_obs"].values.mean()),
            "n_rows": len(y_list),
        }
        return means, ci, hyper
    return means, ci


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    human = load_human_topic_level_accuracies(args.human_csv)
    llm = load_llm_topic_level_condition(args.standalone_jsonl, args.wm_jsonl)
    topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})

    result: dict[str, Any] = {"topics": topics, "reference_level": REFERENCE_LEVEL, "conditions": {}}

    print("Fitting hierarchical Bayesian Bradley-Terry (human)...")
    fit = fit_bt_strengths_bayesian(human, topics)
    if fit is not None:
        beta, ci = fit
        result["conditions"]["human"] = {"beta": beta, "ci": ci}
        for lv in LEVELS:
            print(f"  {lv}: beta={beta[lv]:.3f}  95% CI=[{ci[lv][0]:.3f}, {ci[lv][1]:.3f}]")

    for c in CONDITIONS:
        print(f"Fitting hierarchical Bayesian Bradley-Terry ({c})...")
        fit = fit_bt_strengths_bayesian(llm.get(c, {}), topics)
        if fit is None:
            print(f"  {c}: insufficient data")
            result["conditions"][c] = {"beta": None, "ci": None}
            continue
        beta, ci = fit
        result["conditions"][c] = {"beta": beta, "ci": ci}
        for lv in LEVELS:
            print(f"  {lv}: beta={beta[lv]:.3f}  95% CI=[{ci[lv][0]:.3f}, {ci[lv][1]:.3f}]")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_json or (
        REPO_ROOT / "application" / "comparisons" / f"listening_bayesian_bt_level_strength_{stamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
