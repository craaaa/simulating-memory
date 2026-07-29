#!/usr/bin/env python3
"""Slope plot of mean-cell-ranking level-pair alignment (listening QA), across
all models -- same MODELS list, colors, and slope_plot machinery as
``plot_listening_qa_alignment_slopeplots.py``, but using the mean-ranking
(tie=0.5, within-topic) agreement metric from
``application.listening_qa.mean_level_pair_alignment`` instead of the
individual-sample-draw method.

Usage:
    python -m application.plot_listening_qa_mean_level_pair_alignment_slopeplot
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.listening_qa.level_pair_preference_alignment import (  # noqa: E402
    CONDITIONS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)
from application.listening_qa.mean_level_pair_alignment import (  # noqa: E402
    bootstrap_ci_agreement,
    mean_split_half_baseline,
    run_mean_alignment,
)
from application.plot_listening_qa_alignment_slopeplots import (  # noqa: E402
    CHANCE,
    HUMAN_CSV,
    MODELS,
    slope_plot,
)


def compute_mean_ranking_agreement(
    *, scope: str = "within_topic", seed: int = 42
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]]:
    human = load_human_topic_level_accuracies(HUMAN_CSV)
    out: dict[str, dict[str, float]] = {}
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    for name, prompt_jsonl, wm_jsonl in MODELS:
        llm = load_llm_topic_level_condition(prompt_jsonl, wm_jsonl)
        topics = sorted(set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()})
        result = run_mean_alignment(
            llm=llm, human=human, topics=topics, eval_conditions=list(CONDITIONS), scope=scope
        )
        out[name] = {c: result["per_condition"][c]["agreement"] or 0.0 for c in CONDITIONS}
        ci[name] = {
            c: bootstrap_ci_agreement(result["per_condition"][c]["scores"], seed=seed)
            for c in CONDITIONS
        }
    return out, ci


def main() -> None:
    out_dir = REPO_ROOT / "application" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scope, scope_label, fname_suffix, ylabel_suffix in [
        ("within_topic", "within-topic", "", "(within-topic)"),
        ("all", "all pairs incl. cross-topic", "_alltopics", "(all pairs, incl. cross-topic)"),
    ]:
        print(f"Computing mean-cell-ranking agreement (vs. human) for all models (scope={scope})...")
        agreement, ci = compute_mean_ranking_agreement(scope=scope)
        for name, row in agreement.items():
            print(f"  {name}: " + " ".join(f"{c}={row[c]:.3f}" for c in CONDITIONS))

        print(f"Computing human split-half reliability baseline (scope={scope}, 20 splits)...")
        baseline = mean_split_half_baseline(HUMAN_CSV, scope=scope)
        print(f"  Human split-half reliability: {baseline:.3f}")

        out_path = out_dir / f"listening_qa_mean_level_pair_alignment_slopeplot{fname_suffix}.png"
        slope_plot(
            agreement,
            ylabel=f"Mean-ranking agreement with human {ylabel_suffix}",
            title=f"Listening QA: model-vs-human mean-cell level-preference agreement ({scope_label})",
            out_path=out_path,
            chance_line=CHANCE,
            ci=ci,
            ylim=(0.0, 1.0),
            baseline_line=("Human split-half reliability", baseline),
        )
        print(f"Saved plot to {out_path}\n")


if __name__ == "__main__":
    main()
