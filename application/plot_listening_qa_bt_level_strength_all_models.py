#!/usr/bin/env python3
"""Bradley-Terry pooled level-strength MLE, run for all 6 models (WM condition)
plus human -- same MODELS list/colors as ``plot_listening_qa_alignment_slopeplots.py``.

x = level, one line per model (+ human), y = beta (log-strength vs "control").
Point estimates are the WLS/MLE fit; error bars are the bootstrap CI from
``bradley_terry_level_strength.bootstrap_bt_strengths``.

Usage:
    python -m application.plot_listening_qa_bt_level_strength_all_models
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
from application.plot_listening_qa_alignment_slopeplots import (  # noqa: E402
    CHANCE,  # noqa: F401  (kept for parity/reference with sibling slopeplots)
    GRID,
    HUMAN_CSV,
    INK,
    MODEL_COLORS,
    MODELS,
)


def compute_all_model_betas(
    *, n_boot: int = 1000
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]]:
    """MLE beta (+ bootstrap CI) per model's WM condition, plus human."""
    human = load_human_topic_level_accuracies(HUMAN_CSV)
    betas: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}

    print("Fitting Bradley-Terry MLE (human)...")
    topics_all = sorted(human.keys())
    betas["Human"] = fit_bt_strengths(human, topics_all)
    cis["Human"] = bootstrap_bt_strengths(human, topics_all, n_boot=n_boot)

    for name, prompt_jsonl, wm_jsonl in MODELS:
        print(f"Fitting Bradley-Terry MLE ({name}, WM)...")
        llm = load_llm_topic_level_condition(prompt_jsonl, wm_jsonl)
        topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})
        cell = llm.get("WM", {})
        beta = fit_bt_strengths(cell, topics)
        if beta is None:
            print(f"  {name}: insufficient WM data, skipping")
            continue
        betas[name] = beta
        cis[name] = bootstrap_bt_strengths(cell, topics, n_boot=n_boot)
        for lv in LEVELS:
            b = beta[lv]
            lo, hi = cis[name][lv]
            print(f"  {lv}: beta={b:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    return betas, cis


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    betas, cis = compute_all_model_betas()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(LEVELS))

    for name in ["Human"] + [m[0] for m in MODELS]:
        if name not in betas:
            continue
        color = "#2b2b2b" if name == "Human" else MODEL_COLORS.get(name, "#888888")
        ys = [betas[name][lv] for lv in LEVELS]
        lo_err = [max(0.0, betas[name][lv] - cis[name][lv][0]) for lv in LEVELS]
        hi_err = [max(0.0, cis[name][lv][1] - betas[name][lv]) for lv in LEVELS]
        lw = 2.4 if name == "Human" else 1.8
        ls = "-" if name == "Human" else "--"
        ax.errorbar(
            x, ys, yerr=[lo_err, hi_err], fmt="o" + ls, markersize=6, linewidth=lw,
            color=color, ecolor=color, elinewidth=1.0, capsize=3, alpha=1.0 if name == "Human" else 0.85,
            label=name, zorder=3 if name == "Human" else 2,
        )

    ax.axhline(0.0, color=INK, linestyle=":", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel(f"Bradley-Terry beta (log-strength, ref={REFERENCE_LEVEL}=0)")
    ax.set_title("Listening QA: Bradley-Terry pooled level strength (WM), all models vs. human")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "listening_qa_bt_level_strength_all_models.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
