#!/usr/bin/env python3
"""Forest plot of Bradley-Terry pooled level strengths (listening QA) -- one
panel per non-reference level (repeat_short, repeat_long, distractor), one
row per model (+ human) within each panel: a point at beta with a horizontal
95% CI whisker.

Replaces the overlapping-line slopeplot (7 series x 4 x-positions with wide
CIs was hard to read) with the standard small-multiples layout for pooled
effect-size-with-CI data (the same shape as a meta-analysis forest plot).
Sorted by beta within each panel so the human-agreement pattern reads at a
glance; a hollow marker means the 95% CI crosses zero (no reliable signal
either direction), filled means it doesn't.

Usage:
    python -m application.plot_listening_qa_bt_level_strength_forest
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

from application.listening_qa.bradley_terry_level_strength import REFERENCE_LEVEL  # noqa: E402
from application.plot_listening_qa_alignment_slopeplots import GRID, INK, MODEL_COLORS  # noqa: E402
from application.plot_listening_qa_bt_level_strength_all_models import (  # noqa: E402
    LEVELS,
    MODELS,
    compute_all_model_betas,
)

PANEL_LEVELS = [lv for lv in LEVELS if lv != REFERENCE_LEVEL]
ROW_ORDER = ["Human"] + [m[0] for m in MODELS]
ROW_COLORS = {"Human": INK, **MODEL_COLORS}


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    betas, cis = compute_all_model_betas()
    rows = [r for r in ROW_ORDER if r in betas]

    fig, axes = plt.subplots(1, len(PANEL_LEVELS), figsize=(4.0 * len(PANEL_LEVELS), 0.62 * len(rows) + 1.6), sharey=True)
    if len(PANEL_LEVELS) == 1:
        axes = [axes]

    y = np.arange(len(rows))[::-1]  # top row = first entry (Human)

    for ax, lv in zip(axes, PANEL_LEVELS):
        for yi, name in zip(y, rows):
            b = betas[name][lv]
            lo, hi = cis[name][lv]
            color = ROW_COLORS.get(name, "#888888")
            significant = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(yi, lo, hi, color=color, linewidth=1.6, zorder=2)
            ax.scatter(
                [b], [yi], s=46, facecolor=color if significant else "white",
                edgecolor=color, linewidth=1.6, zorder=3,
            )
        ax.axvline(0.0, color=INK, linestyle=":", linewidth=1, zorder=1)
        ax.set_title(lv, fontsize=11)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel(f"beta (vs. {REFERENCE_LEVEL}=0)")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows)
    for name, yi in zip(rows, y):
        axes[0].get_yticklabels()[list(y).index(yi)].set_color(ROW_COLORS.get(name, "#888888"))
        axes[0].get_yticklabels()[list(y).index(yi)].set_fontweight("bold" if name == "Human" else "normal")

    fig.suptitle(
        "Listening QA: Bradley-Terry level strength (WM), all models vs. human\n"
        "filled marker = 95% CI excludes 0 (reliable direction); hollow = CI crosses 0",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / "listening_qa_bt_level_strength_forest.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
