#!/usr/bin/env python3
"""Bar chart of mean-cell-ranking level-pair alignment (listening QA), with a
human split-half reliability baseline. Two panels: within-topic level pairs
(the original comparison) and all pairs (cross-topic included).

Uses ``application.listening_qa.mean_level_pair_alignment`` (compares cell
MEANS, tie=0.5, not individual sample draws). Default model: gpt-4.1.
Colors/style match ``plot_listening_qa_alignment_slopeplots.py``.

Usage:
    python -m application.plot_listening_qa_mean_level_pair_alignment_bar
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.listening_qa.level_pair_preference_alignment import CONDITIONS  # noqa: E402
from application.listening_qa.mean_level_pair_alignment import (  # noqa: E402
    DEFAULT_HUMAN_CSV,
    DEFAULT_STANDALONE_JSONL,
    DEFAULT_WM_JSONL,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
    mean_split_half_baseline,
    run_mean_alignment,
)
from application.plot_listening_qa_alignment_slopeplots import CHANCE, GRID, INK  # noqa: E402

# One color per condition, reusing the slopeplot palette's tone family
# (grey -> warm accent) rather than introducing a new scheme.
COND_COLORS = {
    "C1": "#6b6b6b",
    "C2": "#9e9e9e",
    "C3": "#d62728",
    "WM": "#eb6834",
}


def _bar_panel(ax, agreements: dict[str, float | None], baseline: float, title: str) -> None:
    xs = list(range(len(CONDITIONS)))
    heights = [agreements[c] if agreements[c] is not None else 0.0 for c in CONDITIONS]
    colors = [COND_COLORS[c] for c in CONDITIONS]
    ax.bar(xs, heights, color=colors, edgecolor=INK, linewidth=0.8, width=0.6, zorder=3)
    for x, c in zip(xs, CONDITIONS):
        v = agreements[c]
        label = f"{v:.3f}" if v is not None else "na"
        ax.text(x, (v or 0.0) + 0.015, label, ha="center", va="bottom", fontsize=9, color=INK)

    ax.axhline(CHANCE, color=INK, linestyle=":", linewidth=1, zorder=1)
    ax.text(
        len(CONDITIONS) - 1 + 0.35, CHANCE, "chance", va="bottom", ha="right",
        fontsize=8, color="#6b6b6b",
    )
    ax.axhline(baseline, color="#8b0000", linestyle="--", linewidth=1, zorder=1)
    ax.text(
        -0.35, baseline, f" human split-half reliability ({baseline:.3f})",
        va="bottom", ha="left", fontsize=8, color="#8b0000",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylim(0.0, min(1.05, baseline + 0.10))
    ax.set_ylabel("Mean-ranking agreement with human")
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    human = load_human_topic_level_accuracies(DEFAULT_HUMAN_CSV)
    llm = load_llm_topic_level_condition(DEFAULT_STANDALONE_JSONL, DEFAULT_WM_JSONL)
    topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for ax, scope, label in [
        (axes[0], "within_topic", "Within-topic (24 pairs)"),
        (axes[1], "all", "All pairs, incl. cross-topic (120 pairs)"),
    ]:
        print(f"Computing mean-cell-ranking agreement per condition (scope={scope})...")
        result = run_mean_alignment(
            llm=llm, human=human, topics=topics, eval_conditions=list(CONDITIONS), scope=scope
        )
        agreements = {c: result["per_condition"][c]["agreement"] for c in CONDITIONS}
        for c, a in agreements.items():
            print(f"  {c}: {a:.3f}" if a is not None else f"  {c}: na")

        print(f"Computing human split-half reliability baseline (scope={scope}, 20 splits)...")
        baseline = mean_split_half_baseline(DEFAULT_HUMAN_CSV, scope=scope)
        print(f"  Human split-half reliability: {baseline:.3f}")

        _bar_panel(ax, agreements, baseline, label)

    fig.suptitle("Listening QA: mean-cell-ranking agreement with human (gpt-4.1)")
    fig.tight_layout()
    out_path = out_dir / "listening_qa_mean_level_pair_alignment_bar.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
