#!/usr/bin/env python3
"""Slope plot of hierarchical (partial-pooling) Bayesian Bradley-Terry level
strength (listening QA), all 6 models' WM condition + human -- same
MODELS/MODEL_COLORS/line style as ``plot_listening_qa_bt_level_strength_all_models.py``,
but the beta/CI come from ``bayesian_bt_level_strength.fit_bt_strengths_bayesian``
(partial pooling across topics via a shared hierarchical prior) instead of
the WLS point estimate + bootstrap CI (full pooling, one shared beta per level).

Usage:
    python -m application.plot_listening_qa_bt_level_strength_bayesian_slopeplot
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.listening_qa.bayesian_bt_level_strength import fit_bt_strengths_bayesian  # noqa: E402
from application.listening_qa.bradley_terry_level_strength import LEVELS, REFERENCE_LEVEL  # noqa: E402
from application.listening_qa.level_pair_preference_alignment import (  # noqa: E402
    CONDITIONS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)
from application.plot_listening_qa_alignment_slopeplots import (  # noqa: E402
    GRID,
    HUMAN_CSV,
    INK,
    MODEL_COLORS,
    MODELS,
)


def compute_all_model_betas_bayesian() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]]:
    human = load_human_topic_level_accuracies(HUMAN_CSV)
    betas: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}

    print("Fitting hierarchical Bayesian Bradley-Terry (human)...")
    topics_all = sorted(human.keys())
    fit = fit_bt_strengths_bayesian(human, topics_all)
    if fit is not None:
        betas["Human"], cis["Human"] = fit

    for name, prompt_jsonl, wm_jsonl in MODELS:
        print(f"Fitting hierarchical Bayesian Bradley-Terry ({name}, WM)...")
        llm = load_llm_topic_level_condition(prompt_jsonl, wm_jsonl)
        topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})
        fit = fit_bt_strengths_bayesian(llm.get("WM", {}), topics)
        if fit is None:
            print(f"  {name}: insufficient WM data, skipping")
            continue
        betas[name], cis[name] = fit
        for lv in LEVELS:
            b = betas[name][lv]
            lo, hi = cis[name][lv]
            print(f"  {lv}: beta={b:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    return betas, cis


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    betas, cis = compute_all_model_betas_bayesian()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(LEVELS), dtype=np.float64)
    series = ["Human"] + [m[0] for m in MODELS]
    present = [name for name in series if name in betas]
    n_series = len(present)
    jitter_width = 0.42

    for si, name in enumerate(present):
        color = "#2b2b2b" if name == "Human" else MODEL_COLORS.get(name, "#888888")
        offset = (si - (n_series - 1) / 2) * (jitter_width / max(n_series - 1, 1))
        xs = x + offset
        ys = [betas[name][lv] for lv in LEVELS]
        lo_err = [max(0.0, betas[name][lv] - cis[name][lv][0]) for lv in LEVELS]
        hi_err = [max(0.0, cis[name][lv][1] - betas[name][lv]) for lv in LEVELS]
        lw = 2.4 if name == "Human" else 1.8
        ls = "-" if name == "Human" else "--"
        ax.errorbar(
            xs, ys, yerr=[lo_err, hi_err], fmt="o" + ls, markersize=6, linewidth=lw,
            color=color, ecolor=color, elinewidth=1.2, capsize=3, alpha=1.0 if name == "Human" else 0.9,
            label=name, zorder=3 if name == "Human" else 2,
        )

    ax.axhline(0.0, color=INK, linestyle=":", linewidth=1, zorder=1)
    for boundary in (x[:-1] + 0.5):
        ax.axvline(boundary, color=GRID, linewidth=1.0, zorder=1)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel(f"Bayesian BT beta, partial pooling (log-strength, ref={REFERENCE_LEVEL}=0)")
    ax.set_title("Listening QA: hierarchical Bayesian Bradley-Terry level strength (WM), all models vs. human")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "listening_qa_bt_level_strength_bayesian_slopeplot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
