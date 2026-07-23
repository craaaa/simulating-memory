#!/usr/bin/env python3
"""Two slope/dot plots comparing prompting (C1-C4) vs. compactor (WM) conditions
across models, for the listening-QA task:

1. Pairwise reranking accuracy — per-condition agreement with human judgments of
   which of two (topic, level) cells scores higher, from
   application.listening_qa.level_pair_preference_alignment. Re-uses that module's
   individual-sample-draw methodology rather than recomputing it here.

2. Humanlikeness (1 - W1) — the paper's (arXiv:2605.25680) 1D Wasserstein-distance
   similarity measure between the human and model score distributions, pooled
   across topics/levels per condition. Uses the exact wasserstein_1d/humanlikeness
   implementation from src/score.py.

Usage:
    python -m application.plot_listening_qa_alignment_slopeplots
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from application.listening_qa.level_pair_preference_alignment import (  # noqa: E402
    CONDITIONS,
    enumerate_base_pairs,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
    run_individual_samples,
)
from score import humanlikeness  # noqa: E402

HUMAN_CSV = (
    REPO_ROOT
    / ".claude"
    / "worktrees"
    / "compare-task"
    / "application"
    / "prolific_study"
    / "multi_v6"
    / "longdata_strict.csv"
)

# (display name, prompting jsonl, compactor jsonl)
MODELS: list[tuple[str, Path, Path]] = [
    (
        "GPT-4.1",
        REPO_ROOT / "runs/prompting/gpt-4.1/20260716T155755Z/tasks/application_listening_qa_full_grid.jsonl",
        REPO_ROOT / "runs/compactor/gpt-4.1/20260720T211617Z/tasks/wm_application_listening_qa_full_grid.jsonl",
    ),
    (
        "Qwen2.5-72B-Instruct",
        REPO_ROOT / "runs/prompting/qwen_qwen-2.5-72b-instruct/tasks/application_listening_qa_full_grid.jsonl",
        REPO_ROOT / "runs/compactor/Qwen_Qwen2.5-72B-Instruct/tasks/wm_application_listening_qa_full_grid.jsonl",
    ),
    (
        "Gemma-4-31B-it",
        REPO_ROOT / "runs/prompting/google_gemma-4-31b-it/tasks/application_listening_qa_full_grid.jsonl",
        REPO_ROOT / "runs/compactor/google_gemma-4-31B-it/tasks/wm_application_listening_qa_full_grid.jsonl",
    ),
    (
        "Kimi-K2-0905",
        REPO_ROOT / "runs/prompting/moonshotai_kimi-k2-0905/tasks/application_listening_qa_full_grid.jsonl",
        REPO_ROOT / "runs/compactor/moonshotai_kimi-k2-0905/tasks/wm_application_listening_qa_full_grid.jsonl",
    ),
    (
        "Qwen2.5-32B-Instruct",
        REPO_ROOT / "runs/prompting/Qwen_Qwen2.5-32B-Instruct/tasks/application_listening_qa_full_grid.jsonl",
        REPO_ROOT / "runs/compactor/Qwen_Qwen2.5-32B-Instruct/tasks/wm_application_listening_qa_full_grid.jsonl",
    ),
    (
        "Command-A",
        REPO_ROOT / "runs/prompting/cohere_command-a/tasks/application_listening_qa_full_grid.jsonl",
        REPO_ROOT / "runs/compactor/CohereLabs_c4ai-command-a-03-2025/tasks/wm_application_listening_qa_full_grid.jsonl",
    ),
]

MODEL_COLORS = {
    "GPT-4.1": "#6b6b6b",
    "Qwen2.5-72B-Instruct": "#eb6834",
    "Gemma-4-31B-it": "#1baf7a",
    "Kimi-K2-0905": "#4a3aa7",
    "Qwen2.5-32B-Instruct": "#d62728",
    "Command-A": "#bcbd22",
}
CHANCE = 0.5
GRID = "#d9d9d9"
INK = "#2b2b2b"


N_BOOT = 2000


def bootstrap_ci_proportion(hits: list[bool], *, seed: int, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap CI on a proportion, resampling the individual hit/miss draws."""
    arr = np.asarray(hits, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def compute_pairwise_reranking(
    seed: int = 42, n_samples_per_pair: int = 500
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]]:
    human = load_human_topic_level_accuracies(HUMAN_CSV)
    out: dict[str, dict[str, float]] = {}
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    for name, prompt_jsonl, wm_jsonl in MODELS:
        llm = load_llm_topic_level_condition(prompt_jsonl, wm_jsonl)
        topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})
        base_pairs = enumerate_base_pairs(topics)
        result = run_individual_samples(
            llm=llm,
            human=human,
            base_pairs=base_pairs,
            eval_conditions=list(CONDITIONS),
            seed=seed,
            n_samples_per_pair=n_samples_per_pair,
        )
        rep_hits = result["rep_hits"]
        out[name] = {c: result["accuracy_by_condition"][c]["agreement"] for c in CONDITIONS}
        ci[name] = {}
        for c in CONDITIONS:
            hits = [h for h in rep_hits[c] if h is not None]
            ci[name][c] = bootstrap_ci_proportion(hits, seed=seed) if hits else (float("nan"), float("nan"))
    return out, ci


def load_human_pooled_accuracy() -> np.ndarray:
    """Every individual (pid, topic, level) trial's exact-match accuracy, pooled."""
    rows = list(csv.DictReader(HUMAN_CSV.open()))
    by_trial: dict[tuple[str, str, str], list[int]] = {}
    for r in rows:
        key = (r["pid"], r["topic"], r["condition"])
        by_trial.setdefault(key, []).append(int(r["correct"]))
    return np.asarray([sum(v) / len(v) for v in by_trial.values()], dtype=np.float64)


def load_model_pooled_accuracy(jsonl_path: Path, condition_ids: tuple[str, ...]) -> np.ndarray:
    vals = []
    for line in jsonl_path.open():
        r = json.loads(line)
        if r.get("condition_id") not in condition_ids:
            continue
        acc = (r.get("metrics") or {}).get("exact_match_accuracy")
        if acc is not None:
            vals.append(float(acc))
    return np.asarray(vals, dtype=np.float64)


def bootstrap_ci_humanlikeness(
    human: np.ndarray, model: np.ndarray, *, seed: int, n_boot: int = N_BOOT
) -> tuple[float, float]:
    """Percentile bootstrap CI on humanlikeness, resampling human and model arrays independently."""
    rng = np.random.default_rng(seed)
    n_h, n_m = len(human), len(model)
    boot_vals = np.empty(n_boot)
    for i in range(n_boot):
        h_idx = rng.integers(0, n_h, size=n_h)
        m_idx = rng.integers(0, n_m, size=n_m)
        boot_vals[i] = humanlikeness(human[h_idx], model[m_idx])
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(lo), float(hi)


def compute_humanlikeness(
    seed: int = 42,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]]:
    human_scores = load_human_pooled_accuracy()
    out: dict[str, dict[str, float]] = {}
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    for name, prompt_jsonl, wm_jsonl in MODELS:
        row: dict[str, float] = {}
        row_ci: dict[str, tuple[float, float]] = {}
        for c in ("C1", "C2", "C3", "C4"):
            scores = load_model_pooled_accuracy(prompt_jsonl, (c,))
            row[c] = humanlikeness(human_scores, scores)
            row_ci[c] = bootstrap_ci_humanlikeness(human_scores, scores, seed=seed)
        wm_scores = load_model_pooled_accuracy(wm_jsonl, ("C2", "C2-stream"))
        row["WM"] = humanlikeness(human_scores, wm_scores)
        row_ci["WM"] = bootstrap_ci_humanlikeness(human_scores, wm_scores, seed=seed)
        out[name] = row
        ci[name] = row_ci
    return out, ci


def slope_plot(
    data: dict[str, dict[str, float]],
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    chance_line: float | None = None,
    ci: dict[str, dict[str, tuple[float, float]]] | None = None,
    ylim: tuple[float, float] = (0.0, 1.0),
) -> None:
    columns = list(CONDITIONS)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = list(range(len(columns)))
    end_ys = sorted(
        ((name, data[name][columns[-1]]) for name, _p, _w in MODELS), key=lambda t: t[1]
    )
    min_gap = 0.035 * (ylim[1] - ylim[0])
    for i in range(1, len(end_ys)):
        prev_name, prev_y = end_ys[i - 1]
        name, y = end_ys[i]
        if y - prev_y < min_gap:
            end_ys[i] = (name, prev_y + min_gap)
    label_y = dict(end_ys)
    n_models = len(MODELS)
    jitter_width = 0.09
    for i, (name, _pjsonl, _wjsonl) in enumerate(MODELS):
        ys = [data[name][c] for c in columns]
        color = MODEL_COLORS[name]
        offset = (i - (n_models - 1) / 2) * (jitter_width / max(n_models - 1, 1))
        xs = [xi + offset for xi in x]
        if ci is not None:
            lo = [max(0.0, data[name][c] - ci[name][c][0]) for c in columns]
            hi = [max(0.0, ci[name][c][1] - data[name][c]) for c in columns]
            ax.errorbar(
                xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1.2,
                capsize=3, alpha=0.6, zorder=2,
            )
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2, color=color, label=name, zorder=3)
        ax.text(len(columns) - 1 + 0.08, label_y[name], name, va="center", ha="left", fontsize=10, color=color)
    if chance_line is not None:
        ax.axhline(chance_line, color=INK, linestyle=":", linewidth=1, zorder=1)
        ax.text(len(columns) - 1, chance_line, "chance ", va="bottom", ha="right", fontsize=8, color="#6b6b6b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(columns)
    ax.set_xlim(-0.15, len(columns) - 1 + 0.75)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing pairwise reranking accuracy (vs. human)...")
    reranking, reranking_ci = compute_pairwise_reranking()
    for name, row in reranking.items():
        print(f"  {name}: " + " ".join(f"{c}={row[c]:.3f}" for c in CONDITIONS))
    slope_plot(
        reranking,
        ylabel="Pairwise reranking accuracy (agreement with human)",
        title="Listening QA: model-vs-human pairwise level-preference agreement",
        out_path=out_dir / "listening_qa_pairwise_reranking_slopeplot.png",
        chance_line=CHANCE,
        ci=reranking_ci,
        ylim=(0.40, 0.60),
    )

    print("\nComputing humanlikeness (1 - W1)...")
    human_w1, human_w1_ci = compute_humanlikeness()
    for name, row in human_w1.items():
        print(f"  {name}: " + " ".join(f"{c}={row[c]:.3f}" for c in CONDITIONS))
    slope_plot(
        human_w1,
        ylabel="Humanlikeness = 1 − W₁(human, model)",
        title="Listening QA: human–model distributional alignment (Wasserstein)",
        out_path=out_dir / "listening_qa_humanlikeness_slopeplot.png",
        chance_line=None,
        ci=human_w1_ci,
    )

    print(f"\nSaved plots to {out_dir}")


if __name__ == "__main__":
    main()
