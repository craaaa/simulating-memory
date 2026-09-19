"""Figures for a listening-QA STaR round.

Same principle as the digit-span figures: the human curve goes on the same axes as the
model's, because the claim being tested is "closer to human", not "higher". The x-axis
is the encoding level rather than span length -- that is the manipulation this study
varies, and it is where a model that has learned nothing about memory will sit flat
while the human curve moves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from bench.core.plotting import save_fig

from .evaluate import CATEGORIES

# Ordered as the manipulation is meant to be read: more support for encoding on the
# left, interference on the right.
LEVEL_ORDER = ["control", "repeat_short", "repeat_long", "distractor"]


def _levels(by_level: Dict[str, Any]) -> List[str]:
    known = [lv for lv in LEVEL_ORDER if lv in by_level]
    return known + sorted(set(by_level) - set(known))


def plot_by_level(
    eval_payload: Dict[str, Any], figures_dir: Path, *, round_n: int
) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    out: List[Path] = []
    base = eval_payload["base"].get("by_level_cell", {})
    tuned = eval_payload["tuned"].get("by_level_cell", {})
    levels = _levels(tuned)
    if not levels:
        return out

    x = np.arange(len(levels))
    width = 0.27

    fig = plt.figure(figsize=(max(6, len(levels) * 1.4), 4))
    plt.bar(x - width, [base.get(lv, {}).get("model_exact_rate", 0) for lv in levels],
            width, label="base model")
    plt.bar(x, [tuned[lv]["model_exact_rate"] for lv in levels], width,
            label=f"STaR round {round_n}")
    plt.bar(x + width, [tuned[lv]["human_exact_rate"] for lv in levels], width,
            label="human")
    plt.xticks(x, levels, rotation=15, ha="right")
    plt.ylabel("Exact match rate (vs. the correct answer)")
    plt.ylim(0, 1.05)
    plt.title(f"Listening QA: accuracy by level (round {round_n})")
    plt.legend()
    path = figures_dir / f"round{round_n}_accuracy_by_level.png"
    save_fig(fig, path)
    out.append(path)

    fig = plt.figure(figsize=(max(6, len(levels) * 1.4), 4))
    plt.bar(x - width / 2, [base.get(lv, {}).get("human_match_rate", 0) for lv in levels],
            width, label="base model")
    plt.bar(x + width / 2, [tuned[lv]["human_match_rate"] for lv in levels], width,
            label=f"STaR round {round_n}")
    plt.xticks(x, levels, rotation=15, ha="right")
    plt.ylabel("Match to the human's own selection")
    plt.ylim(0, 1.05)
    plt.title(f"Listening QA: human-match by level (round {round_n})")
    plt.legend()
    path = figures_dir / f"round{round_n}_human_match_by_level.png"
    save_fig(fig, path)
    out.append(path)

    return out


def plot_endorsement_profile(
    eval_payload: Dict[str, Any], figures_dir: Path, *, round_n: int
) -> Path:
    """Model vs. human distribution over the kind of option endorsed.

    Read this rather than the exact-match rate: which specific option a given human
    picks is stochastic, but "reaches for interference foils under the distractor
    passage" is a learnable property of the population.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tuned = eval_payload["tuned"]["overall"]
    base = eval_payload["base"]["overall"]
    cats = [
        c
        for c in CATEGORIES
        if tuned["error_profile_model"].get(c)
        or tuned["error_profile_human"].get(c)
        or base["error_profile_model"].get(c)
    ]

    x = np.arange(len(cats))
    width = 0.27
    fig = plt.figure(figsize=(max(6, len(cats) * 1.4), 4))
    plt.bar(x - width, [base["error_profile_model"].get(c, 0) for c in cats], width,
            label="base")
    plt.bar(x, [tuned["error_profile_model"].get(c, 0) for c in cats], width,
            label=f"STaR round {round_n}")
    plt.bar(x + width, [tuned["error_profile_human"].get(c, 0) for c in cats], width,
            label="human")
    plt.xticks(x, cats, rotation=30, ha="right")
    plt.ylabel("Share of endorsements")
    tv = tuned.get("error_profile_tv_distance")
    plt.title(
        f"Endorsements by option type, round {round_n} (TV vs. human: {tv:.3f})"
        if tv is not None
        else f"Endorsements by option type, round {round_n}"
    )
    plt.legend()
    path = figures_dir / f"round{round_n}_endorsement_profile.png"
    save_fig(fig, path)
    return path


def plot_round(eval_payload: Dict[str, Any], figures_dir: Path, *, round_n: int) -> List[Path]:
    paths = plot_by_level(eval_payload, figures_dir, round_n=round_n)
    paths.append(plot_endorsement_profile(eval_payload, figures_dir, round_n=round_n))
    return paths
