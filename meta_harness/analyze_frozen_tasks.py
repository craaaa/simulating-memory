"""Which tasks can the objective actually see move, and why are some frozen?

On Hermes, craft_task reads 0.9340 for all seven arms and both digit-span tasks read
0.9479/0.9498 for all seven, to four decimal places. Humanlikeness is 1 - W_1 between the
model's and the humans' per-participant score distributions, so an identical figure has two
very different possible causes:

  (a) the arms produce identical MODEL SCORES -- the harness change is a no-op on that task,
      and the metric is working fine;
  (b) the scores differ but W_1 does not move -- the metric cannot see the difference,
      which would mean the objective is blind on that task.

These are distinguishable: print the raw score distribution alongside the humanlikeness. If
the sorted score vectors are identical across arms, it is (a). If they differ while
1 - W_1 does not, it is (b), and the mean over eight tasks is averaging in dead weight.

Usage:
    python meta_harness/analyze_frozen_tasks.py                  # Hermes held-out arms
    python meta_harness/analyze_frozen_tasks.py --substrate qwen  # Qwen search arms
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_harness import score_candidate as SC  # noqa: E402

S = SC.S  # the same top-level `score` module the contract itself uses

HERMES = "NousResearch_Hermes-4-70B"
QWEN = "Qwen_Qwen3-30B-A3B-Instruct-2507"

ARMS = {
    "hermes": [(a, f"meta_harness/runs/heldout/{a}/{HERMES}") for a in (
        "baseline", "baseline_rep2", "respond_first", "respond_first_rep2",
        "respond_first_rep3", "evicting_reset", "episodic_reset_v3")],
    "qwen": [
        ("baseline", f"meta_harness/runs/iter0/baseline/{QWEN}"),
        ("respond_first", f"meta_harness/runs/iter6/respond_first/{QWEN}"),
        ("respond_first_rep2", f"meta_harness/runs/iter6rep/respond_first/{QWEN}"),
        ("respond_only", f"meta_harness/runs/iter6/respond_only/{QWEN}"),
        ("evicting_reset", f"meta_harness/runs/iter6/evicting_reset/{QWEN}"),
        ("primacy", f"meta_harness/runs/iter2/primacy/{QWEN}"),
        ("chunk_limit", f"meta_harness/runs/iter2/chunk_limit/{QWEN}"),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=sorted(ARMS), default="hermes")
    args = ap.parse_args()

    arms = [(n, ROOT / p) for n, p in ARMS[args.substrate]]
    arms = [(n, p) for n, p in arms if p.exists()]
    tasks = SC.SEARCH_TASKS + (SC.HELDOUT_TASKS if args.substrate == "hermes" else [])

    print(f"substrate: {args.substrate}   arms: {len(arms)}")
    print(f"{'task':<24}{'n_human':>8}{'human mean':>12}{'model mean':>12}"
          f"{'1-W1':>9}{'distinct':>10}{'verdict':>26}")
    for task in tasks:
        human = S.human_scores(task)
        rows = []
        for name, p in arms:
            model = S.llm_scores(task, p, "compactor")
            if not model.size:
                continue
            hl = round(S.humanlikeness(human, model), 4) if human.size else None
            rows.append((name, tuple(sorted(model.tolist())), round(float(model.mean()), 4),
                         hl, len(model)))
        if not rows:
            print(f"{task:<24}  no model scores")
            continue
        vecs = {r[1] for r in rows}
        hls = {r[3] for r in rows}
        means = {r[2] for r in rows}
        if len(vecs) == 1:
            verdict = "no-op: scores identical"
        elif len(hls) == 1:
            verdict = "BLIND: scores move, 1-W1 does not"
        else:
            verdict = f"live: {len(hls)} distinct 1-W1"
        hmean = round(float(human.mean()), 4) if human.size else None
        print(f"{task:<24}{human.size:>8}{(hmean if hmean is not None else '-'):>12}"
              f"{rows[0][2]:>12}{(rows[0][3] if rows[0][3] is not None else '-'):>9}"
              f"{len(vecs):>10}{verdict:>26}")
        if len(vecs) > 1 and len(hls) == 1:
            for name, _, m, hl, n in rows:
                print(f"      {name:<24} model mean {m:<9} 1-W1 {hl}")

    # How much of a candidate's score vector is even distinguishable? For the frozen tasks,
    # show the score histogram: a metric cannot resolve differences inside one bin.
    print("\nscore histograms for the tasks whose 1-W1 never moved:")
    for task in tasks:
        human = S.human_scores(task)
        vecs = set()
        hls = set()
        first = None
        for name, p in arms:
            model = S.llm_scores(task, p, "compactor")
            if not model.size:
                continue
            vecs.add(tuple(sorted(model.tolist())))
            if human.size:
                hls.add(round(S.humanlikeness(human, model), 4))
            if first is None:
                first = model
        if len(hls) == 1 and first is not None:
            hm = Counter(round(float(v), 3) for v in first.tolist())
            hh = Counter(round(float(v), 3) for v in human.tolist()) if human.size else {}
            print(f"  {task}: model {dict(sorted(hm.items()))}")
            print(f"  {' ' * len(task)}  human {dict(sorted(hh.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
