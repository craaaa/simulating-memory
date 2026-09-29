"""How many of the 8 search tasks can any candidate actually move?

UNITS. Every score is humanlikeness = 1 - Wasserstein-1 between the model's and the humans'
per-participant score distributions. Range 0-1, higher is more human-like, in units of task
proportion-correct (0.03 = three percentage points of distributional distance). A DELTA is
candidate score minus baseline score in those same units and is the only quantity that can be
negative. All candidates and the baseline are averaged over their repeat runs.

WHY. Six iterations have produced one transferable finding, and every iteration reads as a
bugfix. The suspicion this script tests: the 8-task mean has very few live dimensions, and the
ones that move are the two tasks the benchmark's own audit says do NOT exercise the memory
module (`reset_messages()` is never called for `nback` and `variable_mapping`, so every
stimulus stays in context). If so, the search has been optimising harness plumbing, and the
treadmill is the expected behaviour of the objective rather than bad luck.

The SE of the 6-task mean delta is computed across the candidate's own repeat runs and
includes a baseline term, because a baseline measured N times still carries noise.

Usage:
    python meta_harness/analyze_live_dimensions.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_harness import score_candidate as SC  # noqa: E402

Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"
PLUMBING = ("nback", "variable_mapping")

BASELINE = ["iter0/baseline", "iter8repA/baseline", "iter8repB/baseline"]
ARMS = {
    "respond_first": ["iter6/respond_first", "iter6rep/respond_first",
                      "iter6rep2/respond_first"],
    "respond_only": ["iter6/respond_only", "iter6rep/respond_only",
                     "iter6rep2/respond_only"],
    "evicting_reset": ["iter6/evicting_reset", "iter7repA/evicting_reset",
                       "iter7repB/evicting_reset"],
    "primacy": ["iter2/primacy", "iter7repA/primacy", "iter7repB/primacy"],
    "chunk_limit": ["iter2/chunk_limit", "iter7repA/chunk_limit",
                    "iter7repB/chunk_limit"],
}


def per_task(dirs: list[str], baseline: Path | None) -> list[dict[str, float]]:
    out = []
    for d in dirs:
        run = ROOT / "meta_harness" / "runs" / d / Q
        if not run.exists():
            continue
        rec = SC.evaluate(run, baseline)
        out.append({t: v for t, v in (rec.get("humanlikeness_by_task") or {}).items()
                    if v is not None})
    return out


def main() -> int:
    print(__doc__.split("Usage:")[0].rstrip())
    base_runs = per_task(BASELINE, None)
    tasks = [t for t in base_runs[0] if t not in ("map_task", "factual_qa")]
    base_mean = {t: sum(r[t] for r in base_runs) / len(base_runs) for t in tasks}
    flat = [t for t in tasks if t not in PLUMBING]

    print(f"\nBaseline, {len(base_runs)} runs. Per-task humanlikeness (0-1, task "
          f"proportion-correct):")
    for t in tasks:
        spread = " ".join(f"{r[t]:.4f}" for r in base_runs)
        print(f"  {t:<24}{base_mean[t]:.4f}   runs: {spread}")

    print(f"\n{'=' * 96}\nHow much of each candidate's mean-over-8 delta comes from the two "
          f"plumbing tasks?")
    print(f"{'candidate':<18}{'mean-8 delta':>14}{'from nback+vm':>15}"
          f"{'from other 6':>14}{'SE of other 6':>15}{'|other6|/SE':>13}")
    for arm, dirs in ARMS.items():
        runs = per_task(dirs, None)
        if not runs:
            print(f"{arm:<18}  MISSING")
            continue
        cand = {t: sum(r[t] for r in runs) / len(runs) for t in tasks}
        d = {t: cand[t] - base_mean[t] for t in tasks}
        mean8 = sum(d.values()) / len(tasks)
        plumb = sum(d[t] for t in PLUMBING) / len(tasks)
        other = sum(d[t] for t in flat) / len(tasks)

        # SE of the 6-task mean, over this candidate's repeats, plus a baseline term.
        def six_mean(rec: dict[str, float]) -> float:
            return sum(rec[t] for t in flat) / len(tasks)

        c_vals = [six_mean(r) for r in runs]
        b_vals = [six_mean(r) for r in base_runs]
        se = 0.0
        if len(c_vals) >= 2:
            se += (st.stdev(c_vals) ** 2) / len(c_vals)
        if len(b_vals) >= 2:
            se += (st.stdev(b_vals) ** 2) / len(b_vals)
        se = se ** 0.5
        ratio = abs(other) / se if se > 0 else float("nan")
        print(f"{arm:<18}{mean8:>+14.4f}{plumb:>+15.4f}{other:>+14.4f}"
              f"{se:>15.4f}{ratio:>13.2f}")

    print(f"\n{'=' * 96}\nPer-task delta for every candidate. A column of exact zeros is an "
          f"inert task:")
    header = "".join(f"{t[:11]:>13}" for t in tasks)
    print(f"{'candidate':<18}{header}")
    for arm, dirs in ARMS.items():
        runs = per_task(dirs, None)
        if not runs:
            continue
        cand = {t: sum(r[t] for r in runs) / len(runs) for t in tasks}
        row = "".join(f"{cand[t] - base_mean[t]:>+13.4f}" for t in tasks)
        print(f"{arm:<18}{row}")

    print("\nRead: if 'from other 6' is inside 2 SE for every candidate, then six of the "
          "eight\nsearch tasks are not discriminating anything, and the whole search is "
          "being decided by\nnback and variable_mapping -- the two tasks whose stimuli never "
          "leave the context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
