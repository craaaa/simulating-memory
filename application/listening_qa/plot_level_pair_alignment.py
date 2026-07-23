"""Bar chart of ``level_pair_preference_alignment.py`` output: agreement with
human (individual-sample pairwise preference) per condition (C1-C4 + WM),
matching the style of ``application/comparisons/listening_level_pair_alignment.png``.

Usage:
    python -m application.listening_qa.plot_level_pair_alignment \\
        --json application/comparisons/listening_level_pair_alignment_gpt-4.1-mini.json \\
        --model-label gpt-4.1-mini \\
        --out-png application/comparisons/listening_level_pair_alignment_gpt-4.1-mini.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from bench.core.plotting import save_fig

CONDITION_IDS = ["C1", "C2", "C3", "C4", "WM"]
CONDITION_LABELS = {
    "C1": "C1 (TaskPr)",
    "C2": "C2 (HumPr)",
    "C3": "C3 (MemPr)",
    "C4": "C4 (MemPr)",
    "WM": "WM compactor\n(4-slot, C2)",
}
COLORS = {
    "C1": "#c6dbef",
    "C2": "#6baed6",
    "C3": "#2171b5",
    "C4": "#08306b",
    "WM": "#238b45",
}


def plot(summary_path: Path, model_label: str, out_png: Path) -> Path:
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = d["meta"]
    acc = d["result"]["accuracy_by_condition"]
    n_base_pairs = d["result"]["n_base_pairs"]
    n_samples_per_pair = meta["n_samples_per_pair"]

    # Conditions with no data (e.g. WM when there's no compactor run for this
    # model) report agreement=None; drop them rather than crash on the plot.
    condition_ids = [c for c in CONDITION_IDS if acc[c]["agreement"] is not None]

    means = [acc[c]["agreement"] for c in condition_ids]
    lo = [acc[c]["ci_lo"] for c in condition_ids]
    hi = [acc[c]["ci_hi"] for c in condition_ids]
    err_lo = [m - l for m, l in zip(means, lo)]
    err_hi = [h - m for m, h in zip(means, hi)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        [CONDITION_LABELS[c] for c in condition_ids],
        means,
        yerr=[err_lo, err_hi],
        capsize=4,
        edgecolor="black",
        color=[COLORS[c] for c in condition_ids],
    )
    for i, m in enumerate(means):
        ax.text(i, hi[i] + 0.001, f"{m:.3f}", ha="center", fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_ylim(min(0.45, min(lo) - 0.01), max(0.55, max(hi) + 0.02))
    ax.set_ylabel("Agreement with human (individual-sample pairwise preference)")
    ax.set_title(
        f"Within-topic level-pair preference alignment vs. human\n"
        f"{model_label}  |  {n_base_pairs} base pairs × {n_samples_per_pair} individual-sample draws each"
        f"  |  error bars = Wilson 95% CI"
    )
    ax.legend()

    save_fig(fig, out_png)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--model-label", type=str, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()

    out = plot(args.json, args.model_label, args.out_png)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
