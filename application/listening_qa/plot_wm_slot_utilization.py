"""Final memory-store usage (WM compactor slot utilization = len(final_kv)/MAX_KEYS)
by topic and level. Left panel: pooled across topics per level. Right panel: by topic,
grouped by level.

Usage:
    python -m application.listening_qa.plot_wm_slot_utilization \\
        --wm-jsonl runs/compactor/openai_gpt-4.1-mini/tasks/wm_application_listening_qa_full_grid.jsonl \\
        --model-label "openai/gpt-4.1-mini" \\
        --out-png runs/comparisons/wm_vs_standalone_listening_qa_gpt-4.1-mini/slot_utilization_by_topic_and_level.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from bench.core.plotting import save_fig

LEVELS = ["control", "repeat_short", "repeat_long", "distractor"]
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
TOPIC_ORDER = ["martial_arts", "fruits", "astronomy", "fabrics"]
TOPIC_LABELS = {
    "martial_arts": "Martial Arts",
    "fruits": "Fruits",
    "astronomy": "Astronomy",
    "fabrics": "Fabrics",
}


def _mean_se(values: List[float]) -> Tuple[float, float, int]:
    if not values:
        return 0.0, 0.0, 0
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    n = len(arr)
    se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return m, se, n


def plot(wm_jsonl: Path, model_label: str, out_png: Path) -> Path:
    rows = [json.loads(line) for line in wm_jsonl.open()]
    rows = [r for r in rows if r.get("condition_id") == "C2"]

    by_topic_level: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_level: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        util = r.get("metrics", {}).get("slot_utilization")
        if util is None:
            continue
        by_topic_level[r["topic_id"]][r["level"]].append(float(util))
        by_level[r["level"]].append(float(util))

    topics = [t for t in TOPIC_ORDER if t in by_topic_level] or sorted(by_topic_level)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.2))

    means = [_mean_se(by_level[l])[0] for l in LEVELS]
    ses = [_mean_se(by_level[l])[1] for l in LEVELS]
    ns = [_mean_se(by_level[l])[2] for l in LEVELS]
    ax_left.bar(
        [LEVEL_LABELS[l] for l in LEVELS],
        means,
        yerr=ses,
        capsize=4,
        edgecolor="black",
        color=[LEVEL_COLORS[l] for l in LEVELS],
    )
    for i, m in enumerate(means):
        ax_left.text(i, m + ses[i] + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_ylabel("Slot utilization (final_kv keys used / 4)")
    ax_left.set_title(f"{model_label}: WM slot utilization by level\n(pooled across topics, n={ns[0]}/bar)")

    width = 0.2
    x = np.arange(len(topics))
    for li, level in enumerate(LEVELS):
        t_means, t_ses = [], []
        for t in topics:
            m, se, _ = _mean_se(by_topic_level[t][level])
            t_means.append(m)
            t_ses.append(se)
        ax_right.bar(
            x + (li - 1.5) * width,
            t_means,
            width=width,
            yerr=t_ses,
            capsize=3,
            edgecolor="black",
            color=LEVEL_COLORS[level],
            label=LEVEL_LABELS[level],
        )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([TOPIC_LABELS.get(t, t) for t in topics])
    ax_right.set_ylim(0.0, 1.05)
    ax_right.set_ylabel("Slot utilization (final_kv keys used / 4)")
    ax_right.set_title(f"{model_label}: WM slot utilization by topic x level")
    ax_right.legend()

    save_fig(fig, out_png)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm-jsonl", type=Path, required=True)
    ap.add_argument("--model-label", type=str, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()

    out = plot(args.wm_jsonl, args.model_label, args.out_png)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
