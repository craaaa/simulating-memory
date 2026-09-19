"""Reproduce the ``accuracy_by_condition_and_topic_with_wm`` figure for any model:
left panel = exact-match accuracy pooled across topic x level per condition
(C1-C4 standalone + WM compactor + Human); right panel = same metric broken out
by topic, pooled across level.

Usage:
    python -m application.listening_qa.wm_vs_standalone_plot \\
        --standalone-jsonl runs/prompting/openai_gpt-4.1-mini/tasks/application_listening_qa_full_grid.jsonl \\
        --wm-jsonl runs/compactor/openai_gpt-4.1-mini/tasks/wm_application_listening_qa_full_grid.jsonl \\
        --model-label "gpt-4.1-mini" \\
        --out-dir runs/comparisons/wm_vs_standalone_listening_qa_gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bench.core.plotting import save_fig
from .level_pair_preference_alignment import DEFAULT_HUMAN_CSV, load_human_topic_level_accuracies

CONDITION_IDS = ["C1", "C2", "C3", "C4"]
CONDITION_LABELS = {
    "C1": "C1 (TaskPr)",
    "C2": "C2 (HumPr)",
    "C3": "C3 (MemPr)",
    "C4": "C4 (MemPr)",
    "WM": "WM compactor (4-slot)",
    "Human": "Human (Prolific)",
}
COLORS = {
    "C1": "#c6dbef",
    "C2": "#6baed6",
    "C3": "#2171b5",
    "C4": "#08306b",
    "WM": "#238b45",
    "Human": "#d95f02",
}
TOPIC_ORDER = ["martial_arts", "fruits", "astronomy", "fabrics"]
TOPIC_LABELS = {
    "martial_arts": "Martial Arts",
    "fruits": "Fruits",
    "astronomy": "Astronomy",
    "fabrics": "Fabrics",
}


def _load_rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open()]


def _accuracy_values(rows: List[dict], topic: str | None = None) -> List[float]:
    out = []
    for r in rows:
        if topic is not None and str(r.get("topic_id") or "").strip() != topic:
            continue
        m = r.get("metrics") or {}
        acc = m.get("exact_match_accuracy")
        if acc is not None:
            out.append(float(acc))
    return out


def _mean_se(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    p = float(arr.mean())
    n = len(arr)
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return p, se


def _human_values(
    human: Dict[str, Dict[str, List[float]]], topic: str | None = None
) -> List[float]:
    out: List[float] = []
    for t, by_level in human.items():
        if topic is not None and t != topic:
            continue
        for vals in by_level.values():
            out.extend(vals)
    return out


def build_condition_series(
    standalone_rows: List[dict],
    wm_rows: List[dict],
    human: Dict[str, Dict[str, List[float]]],
    topic: str | None = None,
) -> Tuple[List[str], List[float], List[float], List[int]]:
    cond_ids = CONDITION_IDS + ["WM", "Human"]
    means, ses, ns = [], [], []
    for cid in cond_ids:
        if cid == "WM":
            vals = _accuracy_values(wm_rows, topic=topic)
        elif cid == "Human":
            vals = _human_values(human, topic=topic)
        else:
            rows = [r for r in standalone_rows if r.get("condition_id") == cid]
            vals = _accuracy_values(rows, topic=topic)
        m, se = _mean_se(vals)
        means.append(m)
        ses.append(se)
        ns.append(len(vals))
    return cond_ids, means, ses, ns


def plot(
    standalone_jsonl: Path,
    wm_jsonl: Path,
    out_dir: Path,
    model_label: str,
    human_csv: Path = DEFAULT_HUMAN_CSV,
    chance_level: float = 1.0 / 32.0,
) -> List[Path]:
    standalone_rows = _load_rows(standalone_jsonl)
    wm_rows = _load_rows(wm_jsonl)
    wm_rows = [r for r in wm_rows if r.get("condition_id") in ("C2", "C2-stream")]
    human = load_human_topic_level_accuracies(human_csv)

    topics = sorted(
        {str(r.get("topic_id") or "").strip() for r in standalone_rows} | set(TOPIC_ORDER)
    )
    topics = [t for t in TOPIC_ORDER if t in topics] or topics

    cond_ids, means, ses, ns = build_condition_series(standalone_rows, wm_rows, human)

    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5.2))

    ax_left.bar(
        [CONDITION_LABELS[c] for c in cond_ids],
        means,
        yerr=ses,
        capsize=4,
        edgecolor="black",
        color=[COLORS[c] for c in cond_ids],
    )
    for i, m in enumerate(means):
        ax_left.text(i, m + ses[i] + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax_left.axhline(chance_level, color="gray", linestyle="--", linewidth=1, label=f"Chance (1/32)")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("Proportion correct (exact match)")
    wm_idx, human_idx = cond_ids.index("WM"), cond_ids.index("Human")
    ax_left.set_title(
        f"{model_label}: accuracy by condition\n"
        f"(standalone C1-C4 pooled across topic×level, n={ns[0]} each;\n"
        f"WM n={ns[wm_idx]}, C2 only; Human n={ns[human_idx]}, Prolific; all pooled across topic×level)"
    )
    ax_left.legend()
    ax_left.tick_params(axis="x", rotation=10)

    width = 0.16
    x = np.arange(len(topics))
    n_per_cell: Dict[str, int] = {}
    for idx, cid in enumerate(cond_ids):
        t_means, t_ses = [], []
        for t in topics:
            if cid == "WM":
                vals = _accuracy_values(wm_rows, topic=t)
            elif cid == "Human":
                vals = _human_values(human, topic=t)
            else:
                rows = [r for r in standalone_rows if r.get("condition_id") == cid]
                vals = _accuracy_values(rows, topic=t)
            m, se = _mean_se(vals)
            t_means.append(m)
            t_ses.append(se)
            if t == topics[0]:
                n_per_cell[cid] = len(vals)
        ax_right.bar(
            x + (idx - (len(cond_ids) - 1) / 2) * width,
            t_means,
            width=width,
            yerr=t_ses,
            capsize=3,
            edgecolor="black",
            color=COLORS[cid],
            label=CONDITION_LABELS[cid],
        )

    ax_right.axhline(chance_level, color="gray", linestyle="--", linewidth=1)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([TOPIC_LABELS.get(t, t) for t in topics])
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_ylabel("Proportion correct")
    ax_right.set_title(
        f"{model_label}: accuracy by topic × condition\n"
        f"(pooled across level; standalone n={n_per_cell['C1']}/cell, "
        f"WM n={n_per_cell['WM']}/cell, Human n={n_per_cell['Human']}/cell)"
    )
    ax_right.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy_by_condition_and_topic_with_wm.png"
    save_fig(fig, out_path)
    return [out_path]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, required=True)
    ap.add_argument("--wm-jsonl", type=Path, required=True)
    ap.add_argument("--model-label", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    args = ap.parse_args()

    paths = plot(
        args.standalone_jsonl, args.wm_jsonl, args.out_dir, args.model_label, human_csv=args.human_csv
    )
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
