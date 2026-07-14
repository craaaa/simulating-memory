"""Pearson and Spearman correlation between human and model cell-mean accuracy,
computed over the 16 (topic, level) cells, per condition (C1-C4, WM).

scipy is broken in this env (numpy/scipy ABI mismatch), so both are implemented
by hand: Pearson via np.corrcoef, Spearman via Pearson on average ranks.

Usage:
    python -m application.listening_qa.human_model_correlation \\
        --standalone-jsonl runs/prompting/openai_gpt-4.1-mini/tasks/application_listening_qa_full_grid.jsonl \\
        --wm-jsonl runs/compactor/openai_gpt-4.1-mini/tasks/wm_application_listening_qa_full_grid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

from .level_pair_preference_alignment import (
    DEFAULT_HUMAN_CSV,
    DEFAULT_STANDALONE_JSONL,
    DEFAULT_WM_JSONL,
    LEVELS,
    load_human_topic_level_accuracies,
    load_llm_topic_level_condition,
)

CONDITIONS = ["C1", "C2", "C3", "C4", "WM"]


def cell_means(
    human: Dict[str, Dict[str, List[float]]],
    llm: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> Tuple[List[str], List[float | None], Dict[str, List[float | None]]]:
    topics = sorted(set(human.keys()) | {t for c in llm.values() for t in c.keys()})
    cells = [f"{t}:{l}" for t in topics for l in LEVELS]

    h_vals: List[float | None] = []
    for t in topics:
        for l in LEVELS:
            v = human.get(t, {}).get(l) or []
            h_vals.append(mean(v) if v else None)

    m_vals: Dict[str, List[float | None]] = {c: [] for c in CONDITIONS}
    for c in CONDITIONS:
        for t in topics:
            for l in LEVELS:
                v = llm.get(c, {}).get(t, {}).get(l) or []
                m_vals[c].append(mean(v) if v else None)

    return cells, h_vals, m_vals


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: List[float], ys: List[float]) -> float:
    x, y = np.asarray(xs), np.asarray(ys)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: List[float], ys: List[float]) -> float:
    return pearson(_average_ranks(xs), _average_ranks(ys))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    human = load_human_topic_level_accuracies(args.human_csv)
    llm = load_llm_topic_level_condition(args.standalone_jsonl, args.wm_jsonl)
    cells, h_vals, m_vals = cell_means(human, llm)

    results = {}
    print(f"{'condition':<10}{'n':>4}{'pearson r':>12}{'spearman rho':>14}")
    for c in CONDITIONS:
        paired = [(h, m) for h, m in zip(h_vals, m_vals[c]) if h is not None and m is not None]
        if len(paired) < 3:
            results[c] = {"n": len(paired), "pearson_r": None, "spearman_rho": None}
            print(f"{c:<10}{len(paired):>4}{'na':>12}{'na':>14}")
            continue
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]
        r = pearson(xs, ys)
        rho = spearman(xs, ys)
        results[c] = {"n": len(paired), "pearson_r": None if np.isnan(r) else float(r), "spearman_rho": None if np.isnan(rho) else float(rho)}
        print(f"{c:<10}{len(paired):>4}{r:>12.3f}{rho:>14.3f}")

    if args.out_json:
        out = {
            "meta": {
                "cells": cells,
                "standalone_jsonl": str(args.standalone_jsonl),
                "wm_jsonl": str(args.wm_jsonl),
                "human_csv": str(args.human_csv),
            },
            "cell_means": {"Human": h_vals, **m_vals},
            "correlations": results,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
