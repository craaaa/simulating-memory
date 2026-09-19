#!/usr/bin/env python3
"""Grouped bar chart of Bradley-Terry pooled level strengths (listening QA):
x = level, groups/color = condition (human, C1, C2, C3, WM), y = beta
(log-strength relative to the "control" reference level, beta=0).

Uses ``application.listening_qa.bradley_terry_level_strength`` -- pools each
level's Mann-Whitney evidence across all 4 topics into one strength per level,
instead of estimating each (topic, level_pair) comparison in isolation (which
is what gave the earlier mean/MWU alignment plots their wide CIs). Default
model: gemini-3.1-pro-preview. Colors/style match the other listening_qa comparison plots.

Usage:
    python -m application.plot_listening_qa_bt_level_strength_bar
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

from application.listening_qa.bradley_terry_level_strength import (  # noqa: E402
    LEVELS,
    REFERENCE_LEVEL,
    bootstrap_bt_strengths,
    fit_bt_strengths,
)
from application.listening_qa.level_pair_preference_alignment import (  # noqa: E402
    CONDITIONS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)
from application.listening_qa.mean_level_pair_alignment import (  # noqa: E402
    DEFAULT_HUMAN_CSV,
    DEFAULT_STANDALONE_JSONL,
    DEFAULT_WM_JSONL,
)
from application.plot_style import GRID, INK, SERIES_COLORS  # noqa: E402

SERIES = ["human", "C1", "C2", "C3", "WM"]


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    human = load_human_topic_level_accuracies(DEFAULT_HUMAN_CSV)
    llm = load_llm_topic_level_condition(DEFAULT_STANDALONE_JSONL, DEFAULT_WM_JSONL)
    topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})

    betas: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}

    print("Fitting Bradley-Terry pooled level strengths (human)...")
    betas["human"] = fit_bt_strengths(human, topics)
    cis["human"] = bootstrap_bt_strengths(human, topics)

    for c in CONDITIONS:
        print(f"Fitting Bradley-Terry pooled level strengths ({c})...")
        cell = llm.get(c, {})
        beta = fit_bt_strengths(cell, topics)
        if beta is None:
            betas[c] = {lv: 0.0 for lv in LEVELS}
            cis[c] = {lv: (float("nan"), float("nan")) for lv in LEVELS}
            continue
        betas[c] = beta
        cis[c] = bootstrap_bt_strengths(cell, topics)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    n_series = len(SERIES)
    width = 0.8 / n_series
    x = np.arange(len(LEVELS))

    for si, series in enumerate(SERIES):
        offset = (si - (n_series - 1) / 2) * width
        heights = [betas[series][lv] for lv in LEVELS]
        lo_err = [
            max(0.0, betas[series][lv] - cis[series][lv][0]) if np.isfinite(cis[series][lv][0]) else 0.0
            for lv in LEVELS
        ]
        hi_err = [
            max(0.0, cis[series][lv][1] - betas[series][lv]) if np.isfinite(cis[series][lv][1]) else 0.0
            for lv in LEVELS
        ]
        ax.bar(
            x + offset, heights, width=width, color=SERIES_COLORS[series],
            edgecolor=INK, linewidth=0.6, label=series, zorder=3,
            yerr=[lo_err, hi_err], capsize=2.5, ecolor=INK, error_kw={"elinewidth": 1.0, "zorder": 4},
        )

    ax.axhline(0.0, color=INK, linestyle=":", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel(f"Bradley-Terry beta (log-strength, ref={REFERENCE_LEVEL}=0)")
    ax.set_title("Listening QA: Bradley-Terry pooled level strength, human vs. gemini-3.1-pro-preview")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=len(SERIES), loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    fig.tight_layout()
    out_path = out_dir / "listening_qa_bt_level_strength_bar.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
