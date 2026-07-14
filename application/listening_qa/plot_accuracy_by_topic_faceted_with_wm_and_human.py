"""Reproduce ``accuracy_by_topic_faceted_by_condition_with_wm_and_human.png`` for
any model: 6 columns (C1-C4, WM, Human) x 2 rows (top: pooled across topics per
level; bottom: by topic), bars = the 4 text levels.

Usage:
    python -m application.listening_qa.plot_accuracy_by_topic_faceted_with_wm_and_human \\
        --standalone-jsonl runs/prompting/openai_gpt-4.1-mini/tasks/application_listening_qa_full_grid.jsonl \\
        --wm-jsonl runs/compactor/openai_gpt-4.1-mini/tasks/wm_application_listening_qa_full_grid.jsonl \\
        --model-label openai/gpt-4.1-mini \\
        --out-dir runs/comparisons/wm_vs_standalone_listening_qa_gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from bench.core.plotting import save_fig

from .level_pair_preference_alignment import (
    DEFAULT_HUMAN_CSV,
    DEFAULT_STANDALONE_JSONL,
    DEFAULT_WM_JSONL,
    HUMAN_TOPIC_MAP,
    LEVELS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)

TOPIC_ORDER = ["martial_arts", "fruits", "astronomy", "fabrics"]
TOPIC_LABELS = {
    "martial_arts": "Martial Arts",
    "fruits": "Fruits",
    "astronomy": "Astronomy",
    "fabrics": "Fabrics",
}
LEVEL_LABELS = {
    "control": "Control",
    "repeat_short": "Repeat Short",
    "repeat_long": "Repeat Long",
    "distractor": "Distractor",
}
LEVEL_COLORS = {
    "control": "#8c8c8c",
    "repeat_short": "#f2a13a",
    "repeat_long": "#d1451b",
    "distractor": "#b39ddb",
}
COLUMNS = ["C1", "C2", "C3", "C4", "WM", "Human"]
COLUMN_TITLES = {
    "C1": "C1 (TaskPr)",
    "C2": "C2 (HumPr)",
    "C3": "C3 (MemPr)",
    "C4": "C4 (MemPr)",
    "WM": "WM compactor\n(4-slot, C2)",
}


def _mean_se(values: List[float]) -> Tuple[float, float, int]:
    if not values:
        return 0.0, 0.0, 0
    arr = np.asarray(values, dtype=float)
    p = float(arr.mean())
    n = len(arr)
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return p, se, n


def _n_human_participants(human_csv: Path) -> int:
    rows = list(csv.DictReader(human_csv.open()))
    return len({r["pid"] for r in rows if HUMAN_TOPIC_MAP.get(r["topic"]) is not None})


def _bar_panel(ax, level_stats: Dict[str, Tuple[float, float, int]], title: str) -> None:
    levels = LEVELS
    means = [level_stats[l][0] for l in levels]
    ses = [level_stats[l][1] for l in levels]
    colors = [LEVEL_COLORS[l] for l in levels]
    x = np.arange(len(levels))
    ax.bar(x, means, yerr=ses, capsize=3, edgecolor="black", color=colors)
    for i, m in enumerate(means):
        ax.text(i, m + ses[i] + 0.015, f"{m:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0 / 32.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[l] for l in levels], rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)


def plot(
    standalone_jsonl: Path,
    wm_jsonl: Path,
    human_csv: Path,
    model_label: str,
    out_dir: Path,
) -> Path:
    human = load_human_topic_level_accuracies(human_csv)
    llm = load_llm_topic_level_condition(standalone_jsonl, wm_jsonl)
    n_human = _n_human_participants(human_csv)

    topics = sorted(set(human.keys()) | {t for c in llm.values() for t in c.keys()})
    topics = [t for t in TOPIC_ORDER if t in topics] or topics

    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(3.2 * len(COLUMNS), 6.4), sharey=True)

    n_label_parts = []
    for col_idx, col in enumerate(COLUMNS):
        if col == "Human":
            per_level_all: Dict[str, List[float]] = {l: [] for l in LEVELS}
            for t in topics:
                for l in LEVELS:
                    per_level_all[l].extend(human.get(t, {}).get(l) or [])
            top_stats = {l: _mean_se(per_level_all[l]) for l in LEVELS}
            title = f"Human\n(Prolific, n={n_human})"
            n_label_parts.append(f"HUMAN: n={n_human}")
        else:
            per_level_all = {l: [] for l in LEVELS}
            for t in topics:
                for l in LEVELS:
                    per_level_all[l].extend(llm.get(col, {}).get(t, {}).get(l) or [])
            top_stats = {l: _mean_se(per_level_all[l]) for l in LEVELS}
            title = COLUMN_TITLES[col]
            n_top = top_stats[LEVELS[0]][2]
            n_label_parts.append(f"{col}: n={n_top}")

        _bar_panel(axes[0, col_idx], top_stats, title)

    axes[0, 0].set_ylabel("Proportion correct (exact match)\naveraged across all 4 topics")
    axes[1, 0].set_ylabel("Proportion correct (exact match)\nby topic")

    # Bottom row: by topic (grouped bars, one subplot per column, x-axis = topics, bars = levels)
    for col_idx, col in enumerate(COLUMNS):
        ax = axes[1, col_idx]
        x = np.arange(len(topics))
        width = 0.2
        for li, level in enumerate(LEVELS):
            means, ses = [], []
            for t in topics:
                if col == "Human":
                    vals = human.get(t, {}).get(level) or []
                else:
                    vals = llm.get(col, {}).get(t, {}).get(level) or []
                m, se, _ = _mean_se(vals)
                means.append(m)
                ses.append(se)
            ax.bar(
                x + (li - 1.5) * width,
                means,
                width=width,
                yerr=ses,
                capsize=2,
                edgecolor="black",
                color=LEVEL_COLORS[level],
            )
        ax.axhline(1.0 / 32.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([TOPIC_LABELS.get(t, t) for t in topics], rotation=20, ha="right", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LEVEL_COLORS[l], edgecolor="black") for l in LEVELS]
    fig.legend(handles, [LEVEL_LABELS[l] for l in LEVELS], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(
        "Accuracy by condition — standalone C1-C4 vs. WM compactor vs. Human (Prolific)\n"
        "top row: averaged across all 4 topics  ·  bottom row: by topic (n per bar shown below, aggregate row/level cell counts vary by source)\n"
        f"{model_label}  |  " + "  ·  ".join(n_label_parts),
        fontsize=11,
        y=1.08,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy_by_topic_faceted_by_condition_with_wm_and_human.png"
    save_fig(fig, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--model-label", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out = plot(args.standalone_jsonl, args.wm_jsonl, args.human_csv, args.model_label, args.out_dir)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
