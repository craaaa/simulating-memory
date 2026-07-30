#!/usr/bin/env python3
"""Faceted version of plot_listening_qa_bt_level_strength_bayesian_slopeplot.py --
one subplot per model (+ human), each with its OWN y-axis, instead of all
series overlaid on one shared axis.

Why: each model's WM fit (and human's) is INDEPENDENT -- fit_bt_strengths_bayesian
pins control=0 separately per series, and never runs a direct human-vs-model or
model-vs-model pairwise comparison. Overlaying them on one axis (as the original
slopeplot does) visually invites comparing e.g. Human's beta at repeat_short
against GPT-4.1's beta at repeat_short as if on the same scale -- they aren't.
Faceting with independent y-axes makes each series' own within-series level
ordering/magnitude legible without implying cross-series comparability.

Usage:
    python -m application.plot_listening_qa_bt_level_strength_bayesian_slopeplot_faceted
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

from application.listening_qa.bradley_terry_level_strength import LEVELS, REFERENCE_LEVEL  # noqa: E402
from application.plot_listening_qa_alignment_slopeplots import GRID, INK, MODEL_COLORS, MODELS  # noqa: E402
from application.plot_listening_qa_bt_level_strength_bayesian_slopeplot import (  # noqa: E402
    compute_all_model_betas_bayesian,
)


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    betas, cis = compute_all_model_betas_bayesian()

    series = ["Human"] + [m[0] for m in MODELS]
    present = [name for name in series if name in betas]
    x = np.arange(len(LEVELS), dtype=np.float64)

    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 4.6), sharey=False)
    if len(present) == 1:
        axes = [axes]

    for ax, name in zip(axes, present):
        color = "#2b2b2b" if name == "Human" else MODEL_COLORS.get(name, "#888888")
        ys = [betas[name][lv] for lv in LEVELS]
        lo_err = [max(0.0, betas[name][lv] - cis[name][lv][0]) for lv in LEVELS]
        hi_err = [max(0.0, cis[name][lv][1] - betas[name][lv]) for lv in LEVELS]
        ax.errorbar(
            x, ys, yerr=[lo_err, hi_err], fmt="o-", markersize=6, linewidth=2.0,
            color=color, ecolor=color, elinewidth=1.2, capsize=3, zorder=3,
        )
        ax.axhline(0.0, color=INK, linestyle=":", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lv.replace("_", "\n") for lv in LEVELS], fontsize=9)
        ax.set_title(name, fontsize=12, fontweight=650, color=color)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(f"Bayesian BT beta, partial pooling (log-strength, ref={REFERENCE_LEVEL}=0)\n-- each panel its OWN scale, not cross-comparable --")

    fig.suptitle(
        "Listening QA: hierarchical Bayesian BT level strength (WM), faceted by model\n"
        "(each series fit independently; y-axes NOT shared, do not compare heights across panels)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    out_path = out_dir / "listening_qa_bt_level_strength_bayesian_slopeplot_faceted.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
