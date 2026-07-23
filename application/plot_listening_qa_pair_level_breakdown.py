#!/usr/bin/env python3
"""Per-level-pair breakdown of the WM compactor's agreement with human
preference (application/listening_qa/level_pair_preference_alignment.py's
``by_level_pair``), grouped by model — shows which specific level
comparisons (e.g. control vs. distractor) a model gets most wrong,
rather than just an overall agreement number.

Usage:
    python -m application.plot_listening_qa_pair_level_breakdown
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.listening_qa.level_pair_preference_alignment import (  # noqa: E402
    DEFAULT_HUMAN_CSV,
    HUMAN_TOPIC_MAP,
    LEVELS,
    enumerate_base_pairs,
    run_individual_samples,
)

# (display name, pair-alignment JSON path) — the 6 models that cleared the
# model-screening bar, matching MODELS in plot_listening_qa_alignment_slopeplots.py.
MODELS: list[tuple[str, Path]] = [
    ("GPT-4.1", REPO_ROOT / "application/comparisons/listening_level_pair_alignment_GPT-4.1.json"),
    ("Qwen2.5-72B-Instruct", REPO_ROOT / "application/comparisons/listening_level_pair_alignment_Qwen2.5-72B-Instruct.json"),
    ("Gemma-4-31B-it", REPO_ROOT / "application/comparisons/listening_level_pair_alignment_Gemma-4-31B-it.json"),
    ("Kimi-K2-0905", REPO_ROOT / "application/comparisons/listening_level_pair_alignment_kimi-k2-0905.json"),
    ("Qwen2.5-32B-Instruct", REPO_ROOT / "application/comparisons/listening_level_pair_alignment_Qwen2.5-32B-Instruct.json"),
    ("Command-A", REPO_ROOT / "application/comparisons/listening_level_pair_alignment_Command-A.json"),
]

MODEL_COLORS = {
    "GPT-4.1": "#6b6b6b",
    "Qwen2.5-72B-Instruct": "#eb6834",
    "Gemma-4-31B-it": "#1baf7a",
    "Kimi-K2-0905": "#4a3aa7",
    "Qwen2.5-32B-Instruct": "#d62728",
    "Command-A": "#bcbd22",
}

PAIR_ORDER = [
    "control_vs_repeat_short",
    "control_vs_repeat_long",
    "control_vs_distractor",
    "repeat_short_vs_repeat_long",
    "repeat_short_vs_distractor",
    "repeat_long_vs_distractor",
]
PAIR_LABELS = {
    "control_vs_repeat_short": "control vs\nrepeat-short",
    "control_vs_repeat_long": "control vs\nrepeat-long",
    "control_vs_distractor": "control vs\ndistractor",
    "repeat_short_vs_repeat_long": "repeat-short vs\nrepeat-long",
    "repeat_short_vs_distractor": "repeat-short vs\ndistractor",
    "repeat_long_vs_distractor": "repeat-long vs\ndistractor",
}

CHANCE = 0.5
GRID = "#d9d9d9"
INK = "#2b2b2b"

# One distinct color per level pair (categorical, tableau10-ish — 6 pairs, not
# reusing the 4-color per-level palette since a pair spans two levels).
PAIR_COLORS = {
    "control_vs_repeat_short": "#4e79a7",
    "control_vs_repeat_long": "#f28e2b",
    "control_vs_distractor": "#e15759",
    "repeat_short_vs_repeat_long": "#76b7b2",
    "repeat_short_vs_distractor": "#59a14f",
    "repeat_long_vs_distractor": "#af7aa1",
}


def _load_human_by_pid(human_csv: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """pid -> topic -> level -> list of per-trial exact-match accuracies (kept
    separate per pid so callers can split participants into halves). Same
    aggregation as level_pair_preference_alignment.load_human_topic_level_accuracies,
    just not collapsed across participants."""
    raw = list(csv.DictReader(human_csv.open()))
    by_trial: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in raw:
        topic = HUMAN_TOPIC_MAP.get(r["topic"])
        if topic is None:
            continue
        by_trial[(r["pid"], topic, r["condition"])].append(int(r["correct"]))

    by_pid: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (pid, topic, level), corrects in by_trial.items():
        if level not in LEVELS:
            continue
        exact = sum(corrects) / len(corrects)
        by_pid[pid][topic][level].append(exact)
    return {pid: dict(topics) for pid, topics in by_pid.items()}


def split_half_by_level_pair(
    human_csv: Path, *, seed: int = 42, n_samples_per_pair: int = 500
) -> dict[str, tuple[float, float, float]]:
    """Split participants into two random halves; treat half A as ground truth
    and half B as a synthetic 'model' condition, running the same individual-
    sample pairwise-preference machinery used for LLMs vs. human. This gives a
    split-half reliability estimate — the noise ceiling any model could hit."""
    by_pid = _load_human_by_pid(human_csv)
    pids = sorted(by_pid.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)
    half = len(pids) // 2
    half_a_pids, half_b_pids = set(pids[:half]), set(pids[half:])

    def pool(pid_set: set[str]) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for pid in pid_set:
            for topic, levels in by_pid[pid].items():
                for level, accs in levels.items():
                    out[topic][level].extend(accs)
        return {t: dict(ls) for t, ls in out.items()}

    human_a = pool(half_a_pids)
    human_b = pool(half_b_pids)
    topics = sorted(set(human_a.keys()) | set(human_b.keys()))
    base_pairs = enumerate_base_pairs(topics)
    result = run_individual_samples(
        llm={"HalfB": human_b},
        human=human_a,
        base_pairs=base_pairs,
        eval_conditions=["HalfB"],
        seed=seed,
        n_samples_per_pair=n_samples_per_pair,
    )
    by_lp = result["accuracy_by_condition"]["HalfB"]["by_level_pair"]
    return {
        lp: (row["agreement"], row["ci_lo"], row["ci_hi"])
        for lp, row in by_lp.items()
        if row["agreement"] is not None
    }


def plot_pair_level_breakdown(out_path: Path, condition: str = "WM") -> None:
    data: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, path in MODELS:
        d = json.loads(path.read_text(encoding="utf-8"))
        by_lp = d["result"]["accuracy_by_condition"][condition]["by_level_pair"]
        data[name] = {
            lp: (row["agreement"], row["ci_lo"], row["ci_hi"])
            for lp, row in by_lp.items()
            if row["agreement"] is not None
        }
    data["Human\n(split-half)"] = split_half_by_level_pair(DEFAULT_HUMAN_CSV)

    model_names = [name for name, _path in MODELS] + ["Human\n(split-half)"]
    n_models = len(model_names)
    n_pairs = len(PAIR_ORDER)
    width = 0.8 / n_pairs

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * n_models), 6))
    for j, lp in enumerate(PAIR_ORDER):
        vals = [data[name][lp][0] for name in model_names]
        los = [max(0.0, data[name][lp][0] - data[name][lp][1]) for name in model_names]
        his = [max(0.0, data[name][lp][2] - data[name][lp][0]) for name in model_names]
        offsets = [i + (j - (n_pairs - 1) / 2) * width for i in range(n_models)]
        ax.bar(
            offsets, vals, width, color=PAIR_COLORS[lp], label=PAIR_LABELS[lp].replace("\n", " "),
            yerr=[los, his], capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.6},
            zorder=3,
        )

    ax.axhline(CHANCE, color=INK, linestyle=":", linewidth=1, zorder=2, label="Chance (0.5)")
    ax.axvline(n_models - 1.5, color=INK, linestyle="-", linewidth=1, zorder=2, alpha=0.5)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.3, 0.75)
    ax.set_ylabel("Agreement with human (individual-sample pairwise preference)")
    ax.set_title(f"Listening QA: {condition} agreement with human, by level pair")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pair_level_breakdown(out_dir / "listening_qa_wm_pair_level_breakdown.png")
    print(f"Saved plot to {out_dir}")


if __name__ == "__main__":
    main()
