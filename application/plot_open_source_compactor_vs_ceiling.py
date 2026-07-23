#!/usr/bin/env python3
"""Plots for the open-source-model listening-QA pilot (see
application/listening_qa/OPEN_SOURCE_MODEL_COMPARISON.md):

1. Prompting-baseline (C1-C4) ceiling accuracy across every open-weight model
   tried, plus the gpt-4.1 reference.
2. Prompting ceiling vs. compactor accuracy for the models with a completed
   compactor run (gpt-4.1, Qwen2.5-72B-Instruct, Gemma-4-31B-it), broken down
   by reading level.

Reads directly from runs/prompting/<model>/tasks/*.jsonl and
runs/compactor/<model>/tasks/*_summary.json; no hand-entered numbers.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#d9d9d9"
INK = "#2b2b2b"
MUTED = "#6b6b6b"

# (display name, prompting run dir under runs/prompting/)
PROMPTING_RUNS: list[tuple[str, str]] = [
    ("GPT-4.1", "gpt-4.1/20260716T155755Z"),
    ("Gemma-4-31B-it", "google_gemma-4-31b-it"),
    ("Qwen2.5-72B-Instruct", "qwen_qwen-2.5-72b-instruct"),
    ("Mistral-Small-24B-2501", "mistralai_mistral-small-24b-instruct-2501"),
    ("WizardLM-2-8x22B", "microsoft_wizardlm-2-8x22b"),
    ("R1-Distill-Llama-70B", "deepseek_deepseek-r1-distill-llama-70b"),
    ("Gemma-3-27B-it", "google_gemma-3-27b-it"),
    ("DeepSeek-V3.2", "deepseek_deepseek-v3.2"),
    ("Llama-3.3-70B-Instruct", "meta-llama_llama-3.3-70b-instruct"),
    ("Hermes-4-70B", "nousresearch_hermes-4-70b"),
    ("Llama-3.1-70B-Instruct", "meta-llama_llama-3.1-70b-instruct"),
    ("Qwen3-32B", "qwen_qwen3-32b"),
    ("Nemotron-3-Super-120B", "nvidia_nemotron-3-super-120b-a12b"),
]

# (display name, compactor run dir under runs/compactor/)
COMPACTOR_RUNS: list[tuple[str, str]] = [
    ("GPT-4.1", "gpt-4.1/20260720T211617Z"),
    ("Qwen2.5-72B-Instruct", "Qwen_Qwen2.5-72B-Instruct"),
    ("Gemma-4-31B-it", "google_gemma-4-31B-it"),
]

LEVEL_ORDER = ("control", "repeat_short", "repeat_long", "distractor")


def prompting_accuracy(run_dir: str) -> float:
    path = REPO_ROOT / "runs" / "prompting" / run_dir / "tasks" / "application_listening_qa_full_grid.jsonl"
    rows = [json.loads(line) for line in path.open()]
    accs = [r["metrics"]["exact_match_accuracy"] for r in rows if r.get("metrics")]
    return sum(accs) / len(accs)


def compactor_accuracy_by_level(run_dir: str) -> dict[str, float]:
    path = REPO_ROOT / "runs" / "compactor" / run_dir / "tasks" / "wm_application_listening_qa_full_grid_summary.json"
    summary = json.load(path.open())
    by_level: dict[str, list[float]] = defaultdict(list)
    for key, cell in summary["cells"].items():
        if cell.get("exact_match_mean") is None:
            continue
        level = key.split(":")[1]
        by_level[level].append(cell["exact_match_mean"])
    return {level: sum(v) / len(v) for level, v in by_level.items()}


def plot_ceiling_bar(out_path: Path) -> None:
    rows = [(name, prompting_accuracy(run_dir)) for name, run_dir in PROMPTING_RUNS]
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    accs = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = range(len(names))
    colors = [BLUE if name != "GPT-4.1" else MUTED for name in names]
    bars = ax.barh(list(y), accs, color=colors, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Prompting-baseline (C1-C4) exact-match accuracy")
    ax.set_title("Open-weight models: prompting-baseline ceiling vs. gpt-4.1")
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for bar, acc in zip(bars, accs):
        ax.text(
            acc + 0.01, bar.get_y() + bar.get_height() / 2, f"{acc:.3f}",
            va="center", ha="left", fontsize=9, color=INK,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ceiling_vs_compactor(out_path: Path) -> None:
    models = [name for name, _ in COMPACTOR_RUNS]
    prompting_by_model = dict(PROMPTING_RUNS)
    ceiling = [prompting_accuracy(prompting_by_model[m]) for m in models]
    compactor_by_level = {name: compactor_accuracy_by_level(run_dir) for name, run_dir in COMPACTOR_RUNS}
    compactor_overall = [
        sum(compactor_by_level[m].values()) / len(compactor_by_level[m]) for m in models
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = range(len(models))
    width = 0.34
    b1 = ax.bar([i - width / 2 for i in x], ceiling, width, color=BLUE, label="Prompting (ceiling)", zorder=3)
    b2 = ax.bar([i + width / 2 for i in x], compactor_overall, width, color=ORANGE, label="Compactor (working memory)", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Prompting ceiling vs. compactor accuracy")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_compactor_by_level(out_path: Path) -> None:
    models = [name for name, _ in COMPACTOR_RUNS]
    by_level = {name: compactor_accuracy_by_level(run_dir) for name, run_dir in COMPACTOR_RUNS}

    fig, ax = plt.subplots(figsize=(8, 5))
    n_levels = len(LEVEL_ORDER)
    width = 0.8 / len(models)
    colors = [BLUE, ORANGE, "#1baf7a"]
    for i, model in enumerate(models):
        vals = [by_level[model].get(level, float("nan")) for level in LEVEL_ORDER]
        offsets = [j + (i - (len(models) - 1) / 2) * width for j in range(n_levels)]
        ax.bar(offsets, vals, width, color=colors[i % len(colors)], label=model, zorder=3)
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels([lvl.replace("_", " ") for lvl in LEVEL_ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Compactor exact-match accuracy")
    ax.set_title("Compactor accuracy by reading level")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_ceiling_bar(out_dir / "open_source_prompting_ceiling_bar.png")
    plot_ceiling_vs_compactor(out_dir / "open_source_ceiling_vs_compactor_bar.png")
    plot_compactor_by_level(out_dir / "open_source_compactor_by_level_bar.png")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
