#!/usr/bin/env python3
"""Faceted bar chart of Bradley-Terry pooled level strengths (listening QA),
hierarchical Bayesian fit -- one small subplot per condition (human, C1, C2,
C3, WM), each with its OWN y-axis.

Why faceted instead of grouped bars on one axis (see
plot_listening_qa_bt_level_strength_bar_bayesian.py): each condition is fit
INDEPENDENTLY -- human data and LLM data never appear in the same pairwise
comparison, and each series pins its own control=0 separately. A grouped bar
chart puts all five series on one shared y-axis, which visually invites
comparing e.g. human-control's beta against WM-repeat_short's beta as if
they were on the same scale -- they are not. Faceting with independent
y-axes makes the non-comparability explicit while still letting you read
each condition's own within-series level ordering/magnitude.

Usage:
    python -m application.plot_listening_qa_bt_level_strength_facets_bayesian
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

from application.listening_qa.bayesian_bt_level_strength import (  # noqa: E402
    fit_bt_strengths_bayesian,
)
from application.listening_qa.bradley_terry_level_strength import (  # noqa: E402
    LEVELS,
    REFERENCE_LEVEL,
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
from application.plot_style import GRID, INK, LEVEL_COLORS, MUTED  # noqa: E402

SERIES = ["human", "C1", "C2", "C3", "WM"]
SERIES_TITLES = {
    "human": "Human",
    "C1": "C1 (TaskPr)",
    "C2": "C2 (HumPr)",
    "C3": "C3 (MemPr)",
    "WM": "WM (compactor)",
}


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    human = load_human_topic_level_accuracies(DEFAULT_HUMAN_CSV)
    llm = load_llm_topic_level_condition(DEFAULT_STANDALONE_JSONL, DEFAULT_WM_JSONL)
    topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})

    betas: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}

    print("Fitting hierarchical Bayesian Bradley-Terry (human)...")
    fit = fit_bt_strengths_bayesian(human, topics)
    betas["human"], cis["human"] = fit if fit is not None else (
        {lv: 0.0 for lv in LEVELS}, {lv: (float("nan"), float("nan")) for lv in LEVELS}
    )

    for c in CONDITIONS:
        print(f"Fitting hierarchical Bayesian Bradley-Terry ({c})...")
        cell = llm.get(c, {})
        fit = fit_bt_strengths_bayesian(cell, topics)
        if fit is None:
            betas[c] = {lv: 0.0 for lv in LEVELS}
            cis[c] = {lv: (float("nan"), float("nan")) for lv in LEVELS}
            continue
        betas[c], cis[c] = fit

    fig, axes = plt.subplots(1, len(SERIES), figsize=(4.2 * len(SERIES), 4.6), sharey=False)
    x = np.arange(len(LEVELS))

    for ax, series in zip(axes, SERIES):
        heights = [betas[series][lv] for lv in LEVELS]
        lo_err = [
            max(0.0, betas[series][lv] - cis[series][lv][0]) if np.isfinite(cis[series][lv][0]) else 0.0
            for lv in LEVELS
        ]
        hi_err = [
            max(0.0, cis[series][lv][1] - betas[series][lv]) if np.isfinite(cis[series][lv][1]) else 0.0
            for lv in LEVELS
        ]
        colors = [LEVEL_COLORS[lv] for lv in LEVELS]
        ax.bar(
            x, heights, width=0.6, color=colors, edgecolor=INK, linewidth=0.6, zorder=3,
            yerr=[lo_err, hi_err], capsize=3, ecolor=INK, error_kw={"elinewidth": 1.0, "zorder": 4},
        )
        ax.axhline(0.0, color=MUTED, linestyle=":", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lv.replace("_", "\n") for lv in LEVELS], fontsize=9)
        ax.set_title(SERIES_TITLES[series], fontsize=12, fontweight=650)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(f"BT beta (log-strength, ref={REFERENCE_LEVEL}=0)\n-- each panel its OWN scale, not cross-comparable --")

    fig.suptitle(
        "Listening QA: hierarchical Bayesian BT level strength, gemini-3.1-pro-preview\n"
        "(faceted — each condition fit independently; y-axes NOT shared, do not compare heights across panels)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    out_path = out_dir / "listening_qa_bt_level_strength_facets_bayesian.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
