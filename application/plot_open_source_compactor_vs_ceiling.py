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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.listening_qa.level_pair_preference_alignment import (  # noqa: E402
    DEFAULT_HUMAN_CSV,
    load_human_topic_level_accuracies,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#d9d9d9"
INK = "#2b2b2b"
MUTED = "#6b6b6b"

# Canonical per-level palette from the human study's own analysis script
# (application/prolific_study/multi_v6/analyze_v6.py: colors = [...], CONDS order
# from data_prep.py).
LEVEL_COLORS: dict[str, str] = {
    "control": "#94a3b8",
    "repeat_short": "#fb923c",
    "repeat_long": "#f97316",
    "distractor": "#a78bfa",
}

# Friendly display names for known slugs; anything not listed here falls back
# to an auto-formatted version of its runs/prompting/<slug> directory name.
DISPLAY_NAMES: dict[str, str] = {
    "gpt-4.1": "GPT-4.1",
    "openai_gpt-4.1-mini": "GPT-4.1-mini",
    "google_gemma-4-31b-it": "Gemma-4-31B-it",
    "google_gemma-4-26b-a4b-it": "Gemma-4-26B-A4B-it",
    "google_gemma-3-27b-it": "Gemma-3-27B-it",
    "qwen_qwen-2.5-72b-instruct": "Qwen2.5-72B-Instruct",
    "Qwen_Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "mistralai_mistral-small-24b-instruct-2501": "Mistral-Small-24B-2501",
    "microsoft_wizardlm-2-8x22b": "WizardLM-2-8x22B",
    "microsoft_phi-4": "Phi-4",
    "deepseek_deepseek-r1-distill-llama-70b": "R1-Distill-Llama-70B",
    "deepseek_deepseek-v3.2": "DeepSeek-V3.2",
    "meta-llama_llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
    "meta-llama_llama-3.1-70b-instruct": "Llama-3.1-70B-Instruct",
    "meta-llama_llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "meta-llama_Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "meta-llama_llama-4-maverick": "Llama-4-Maverick",
    "meta-llama_llama-4-scout": "Llama-4-Scout",
    "nousresearch_hermes-4-70b": "Hermes-4-70B",
    "nvidia_nemotron-3-super-120b-a12b": "Nemotron-3-Super-120B",
    "nvidia_nemotron-3-nano-30b-a3b": "Nemotron-3-Nano-30B",
    "nvidia_nemotron-3-ultra-550b-a55b": "Nemotron-3-Ultra-550B",
    "cohere_command-a": "Command-A",
    "moonshotai_kimi-k2-0905": "Kimi-K2-0905",
    "z-ai_glm-4.6": "GLM-4.6",
    "qwen_qwen3-32b": "Qwen3-32B",
    "qwen_qwen3-235b-a22b-2507": "Qwen3-235B-A22B-2507",
    "qwen_qwen3-30b-a3b-instruct": "Qwen3-30B-A3B-Instruct",
    "qwen_qwen3-30b-a3b-thinking": "Qwen3-30B-A3B-Thinking",
    "qwen_qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B-Instruct",
    "qwen_qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
    "qwen_qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
    "qwen_qwen3.6-27b": "Qwen3.6-27B",
}


def _display_name(slug: str) -> str:
    if slug in DISPLAY_NAMES:
        return DISPLAY_NAMES[slug]
    return slug.split("_", 1)[-1].replace("-", " ").title().replace(" ", "-")


def discover_prompting_runs() -> list[tuple[str, str]]:
    """Every runs/prompting/<slug>[/<timestamp>]/tasks/application_listening_qa_full_grid.jsonl,
    as (display name, path relative to runs/prompting/ up to but excluding /tasks/...jsonl)."""
    root = REPO_ROOT / "runs" / "prompting"
    rows: list[tuple[str, str]] = []
    for jsonl_path in sorted(root.glob("**/tasks/application_listening_qa_full_grid.jsonl")):
        rel = jsonl_path.relative_to(root)
        run_dir = str(rel.parent.parent)  # strip "/tasks/<file>.jsonl"
        slug = run_dir.split("/")[0]
        rows.append((_display_name(slug), run_dir))
    return rows


# (display name, prompting run dir under runs/prompting/) — auto-discovered.
PROMPTING_RUNS: list[tuple[str, str]] = discover_prompting_runs()

# (display name, compactor run dir under runs/compactor/) — models that cleared
# the model-screening bar (near-ceiling prompting accuracy, no severe persistent
# errors) and have a completed compactor run. Kept in sync with MODELS in
# plot_listening_qa_alignment_slopeplots.py.
COMPACTOR_RUNS: list[tuple[str, str]] = [
    ("GPT-4.1", "gpt-4.1/20260720T211617Z"),
    ("Qwen2.5-72B-Instruct", "Qwen_Qwen2.5-72B-Instruct"),
    ("Gemma-4-31B-it", "google_gemma-4-31B-it"),
    ("Kimi-K2-0905", "moonshotai_kimi-k2-0905"),
    ("Qwen2.5-32B-Instruct", "Qwen_Qwen2.5-32B-Instruct"),
    ("Command-A", "CohereLabs_c4ai-command-a-03-2025"),
]

LEVEL_ORDER = ("control", "repeat_short", "repeat_long", "distractor")


N_BOOT = 2000


def bootstrap_ci(vals: list[float], *, seed: int = 42, n_boot: int = N_BOOT) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def human_accuracy_by_level() -> dict[str, float]:
    by_topic_level = load_human_topic_level_accuracies(DEFAULT_HUMAN_CSV)
    pooled: dict[str, list[float]] = defaultdict(list)
    for levels in by_topic_level.values():
        for level, accs in levels.items():
            pooled[level].extend(accs)
    return {level: sum(v) / len(v) for level, v in pooled.items()}


def human_raw_by_level() -> dict[str, list[float]]:
    by_topic_level = load_human_topic_level_accuracies(DEFAULT_HUMAN_CSV)
    pooled: dict[str, list[float]] = defaultdict(list)
    for levels in by_topic_level.values():
        for level, accs in levels.items():
            pooled[level].extend(accs)
    return dict(pooled)


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


def compactor_raw_by_level(run_dir: str) -> dict[str, list[float]]:
    """Every individual trial's exact_match_accuracy, grouped by level (not
    collapsed to per-cell means) — needed for a real bootstrap CI."""
    path = REPO_ROOT / "runs" / "compactor" / run_dir / "tasks" / "wm_application_listening_qa_full_grid.jsonl"
    by_level: dict[str, list[float]] = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        level = r.get("level")
        acc = (r.get("metrics") or {}).get("exact_match_accuracy")
        if level is None or acc is None:
            continue
        by_level[level].append(float(acc))
    return dict(by_level)


def plot_ceiling_bar(out_path: Path) -> None:
    rows = [(name, prompting_accuracy(run_dir)) for name, run_dir in PROMPTING_RUNS]
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    accs = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(6, 0.32 * len(names))))
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

    fig, ax = plt.subplots(figsize=(max(7.5, 1.6 * len(models)), 5.5))
    x = range(len(models))
    width = 0.34
    b1 = ax.bar([i - width / 2 for i in x], ceiling, width, color=BLUE, label="Prompting (ceiling)", zorder=3)
    b2 = ax.bar([i + width / 2 for i in x], compactor_overall, width, color=ORANGE, label="Compactor (working memory)", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Prompting ceiling vs. compactor accuracy")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_compactor_by_level(out_path: Path) -> None:
    models = ["Human"] + [name for name, _ in COMPACTOR_RUNS]
    by_level = {name: compactor_accuracy_by_level(run_dir) for name, run_dir in COMPACTOR_RUNS}
    by_level["Human"] = human_accuracy_by_level()
    raw_by_level = {name: compactor_raw_by_level(run_dir) for name, run_dir in COMPACTOR_RUNS}
    raw_by_level["Human"] = human_raw_by_level()

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(models)), 5))
    n_models = len(models)
    n_levels = len(LEVEL_ORDER)
    width = 0.8 / n_levels
    for j, level in enumerate(LEVEL_ORDER):
        vals = [by_level[model].get(level, float("nan")) for model in models]
        cis = [
            bootstrap_ci(raw_by_level[model][level]) if raw_by_level[model].get(level) else (float("nan"), float("nan"))
            for model in models
        ]
        los = [max(0.0, v - c[0]) for v, c in zip(vals, cis)]
        his = [max(0.0, c[1] - v) for v, c in zip(vals, cis)]
        offsets = [i + (j - (n_levels - 1) / 2) * width for i in range(n_models)]
        ax.bar(
            offsets, vals, width, color=LEVEL_COLORS[level], label=level.replace("_", " "),
            yerr=[los, his], capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.6},
            zorder=3,
        )
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1, zorder=2)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Compactor accuracy by reading level (vs. human)")
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
