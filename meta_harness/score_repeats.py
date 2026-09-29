"""Score a candidate from N repeat runs by AVERAGING per-task humanlikeness.

WHY THIS EXISTS
---------------
Single-run per-task verdicts cannot decide candidates on this benchmark. Measured on ten
episodic-family runs, `craft_task`'s delta against the baseline is

    -0.0435 -0.0373 -0.0342 -0.0326 -0.0280 -0.0280 -0.0248 0.0000 0.0000 0.0000

mean -0.0228, sd 0.0158, and 4 of 10 trip the 0.030 floor. Two runs of IDENTICAL code
gave 0.8534 and 0.8907 for `respond_first`, and 0.8565 and 0.8907 for `respond_only` --
and on craft those candidates are provable no-ops, with request sequences byte-identical
to the baseline's. So `respond_only` passed the whole contract in one run and failed it in
the other, while `respond_first` failed on craft in one run and on narrative_qa in the
other. Same code, opposite verdicts, different tasks.

WHY AVERAGING RATHER THAN "ALL REPEATS MUST PASS"
-------------------------------------------------
Requiring every repeat to pass compounds the noise instead of reducing it. At craft's
observed single-run failure rate of 0.40 it rejects a blameless candidate with probability
1-(1-0.40)^3 = 0.78. Averaging reduces the noise by sqrt(N) -- sd 0.0158 -> 0.0091 at
N=3 -- and compares one better-estimated number to the floor, which is what a repeat
actually buys.

WHAT AVERAGING DOES NOT FIX
---------------------------
craft's drop on this family is NOT pure noise: the distribution is centred on -0.0228,
not zero, so there is a real effect of about -0.023 sitting only 0.007 inside a 0.030
floor. At N=3 a typical episodic candidate still trips craft roughly one run-set in six.
Fixing that needs either a floor derived from same-family repeats (~0.04 for craft) or a
decision that a real -0.023 craft cost is acceptable for this family. Both are contract
decisions, not measurements, and this module deliberately does not take them: it reports
the per-run spread beside the average so the margin is always visible.

USAGE
-----
    python meta_harness/score_repeats.py RUN_DIR [RUN_DIR ...] --baseline DIR \
        [--baseline DIR ...] [--id NAME] [--iteration N] [--record]

Every RUN_DIR must be the same candidate. The averaged record carries `n_repeats`, the
per-run values and the per-task spread, so a reader can always see whether a verdict
rested on a margin inside the noise.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_harness import score_candidate as SC  # noqa: E402

# Same-family run-to-run spread, measured from identical-code and no-op pairs. Used only
# to FLAG a verdict that rests inside the noise, never to change the verdict.
SAME_FAMILY_SD = {
    "craft_task": 0.0158,
    "narrative_qa": 0.0093,
    "word_recognition": 0.0180,
    "digit_span_forward": 0.0152,
    "semantic_story_recall": 0.0038,
    "nback": 0.0056,
    "variable_mapping": 0.0016,
    "digit_span_reverse": 0.0000,
}


def average_records(run_dirs: list[Path],
                    baselines: list[Path] | None) -> dict[str, Any]:
    """Average per-task humanlikeness across repeats, then apply the contract once.

    The baseline may itself be several repeats. It has exactly the same run-to-run noise
    as the candidate, so comparing an averaged candidate to a single baseline run leaves
    half the noise in the delta; when repeats of the baseline exist they are averaged too.
    """
    baseline = baselines[0] if baselines else None
    recs = [SC.evaluate(d, baseline) for d in run_dirs]
    n = len(recs)

    tasks = sorted({t for r in recs for t in r["humanlikeness_by_task"]})
    per_task: dict[str, float | None] = {}
    per_run: dict[str, list[float | None]] = {}
    spread: dict[str, float | None] = {}
    for t in tasks:
        vals = [r["humanlikeness_by_task"].get(t) for r in recs]
        per_run[t] = vals
        good = [v for v in vals if v is not None]
        per_task[t] = round(st.mean(good), 4) if good else None
        spread[t] = round(max(good) - min(good), 4) if len(good) > 1 else None

    search = [per_task[t] for t in SC.SEARCH_TASKS if per_task.get(t) is not None]
    out: dict[str, Any] = {
        "run_dirs": [str(d) for d in run_dirs],
        "n_repeats": n,
        "humanlikeness_by_task": per_task,
        "per_run_humanlikeness": per_run,
        "per_task_spread": spread,
        "mean_humanlikeness_search": (round(st.mean(search), 4) if search else None),
        "mean_per_run": [r.get("mean_humanlikeness_search") for r in recs],
    }

    # Axes come from the FIRST run rather than being averaged: several are distances and
    # ratios whose average is not the statistic they name, and A4's ceiling depends on a
    # run's own error count. The per-run values are recorded so nothing is hidden.
    out["axes"] = recs[0].get("axes")
    out["axes_note"] = ("taken from the first repeat, not averaged: A2/A3 distances and "
                        "A4's normalized ratio are not linear in the runs, and A4's "
                        "ceiling is a function of each run's own error count")
    out["axes_per_run"] = [r.get("axes") for r in recs]

    # Floors, applied ONCE to the averaged deltas.
    bhl: dict[str, float | None] = {}
    bhl_per_run: list[dict[str, float | None]] = []
    if baselines:
        bhl_per_run = [SC.humanlikeness_by_task(b, SC.SEARCH_TASKS + SC.HELDOUT_TASKS)
                       for b in baselines]
        for t in {k for d in bhl_per_run for k in d}:
            vals = [d.get(t) for d in bhl_per_run]
            good = [v for v in vals if v is not None]
            bhl[t] = round(st.mean(good), 4) if good else None
        out["baseline_dirs"] = [str(b) for b in baselines]
        out["n_baseline_repeats"] = len(baselines)
        out["baseline_humanlikeness_by_task"] = bhl
        out["baseline_per_run_humanlikeness"] = {
            t: [d.get(t) for d in bhl_per_run] for t in sorted(bhl)}
    deltas: dict[str, float | None] = {}
    violations = []
    flagged = []
    for t, v in per_task.items():
        b = bhl.get(t)
        if v is None or b is None:
            deltas[t] = None
            continue
        d = round(v - b, 4)
        deltas[t] = d
        eff = max(SC.FLOOR, SC.NOISE_FLOOR.get(t, 0.05))
        if t in SC.SEARCH_TASKS and d < -eff:
            violations.append({"task": t, "delta": d, "floor": -eff})
            # Is the violation inside the noise? Use THIS candidate's own observed
            # spread when there are enough repeats, and fall back to the family-wide
            # figure only when there are not.
            #
            # The family-wide table alone is too crude and produced a misleading note:
            # primacy's craft violation of -0.0466 was flagged as "inside the noise"
            # using the family sd of 0.0158, when primacy's OWN three runs read 0.8456,
            # 0.8456 and 0.8410 -- a spread of 0.0046. Its craft cost is one of the most
            # robust effects measured, not a marginal one.
            fam = SAME_FAMILY_SD.get(t)
            own = [v for v in per_run[t] if v is not None]
            if len(own) >= 3:
                sd = st.stdev(own)
                src = f"this candidate's own {len(own)} repeats"
                # An observed sd of 0 from a handful of repeats does not mean the noise is
                # zero -- it means these repeats did not resolve it, and a zero SE would
                # declare every nonzero delta a real effect. Never go below the
                # family-wide figure.
                if fam is not None and sd < fam:
                    sd = fam
                    src = (f"the family-wide spread, which is larger than this "
                           f"candidate's own {len(own)} repeats produced")
            else:
                sd = fam
                src = "the family-wide spread (too few repeats for an own estimate)"
            if sd is not None:
                se = sd / (n ** 0.5)
                # The delta carries the BASELINE's noise as well as the candidate's, and
                # that term does not vanish when the baseline was run once: a single draw
                # has the full family variance, not zero. Treating it as zero understated
                # every single-baseline standard error in this project.
                bown = [d.get(t) for d in bhl_per_run]
                bown = [v for v in bown if v is not None]
                if len(bown) >= 3:
                    bsd = st.stdev(bown)
                    if fam is not None and bsd < fam:
                        bsd = fam
                    bsrc = f"the baseline's own {len(bown)} repeats"
                else:
                    bsd = fam
                    bsrc = (f"the family-wide spread for the baseline's {len(bown)} "
                            f"run(s)")
                if bsd is not None and bown:
                    se = (se ** 2 + (bsd ** 2) / len(bown)) ** 0.5
                    src += f" plus {bsrc}"
                if abs(d + eff) < 2 * se:
                    flagged.append(
                        f"{t}: violation of {d} against -{eff} is within 2 SE "
                        f"({2 * se:.4f}, from {src}) of the floor at n={n}, so it "
                        f"rests inside the noise and should not be read as a candidate "
                        f"defect on its own")
                else:
                    flagged.append(
                        f"{t}: violation of {d} against -{eff} is OUTSIDE 2 SE "
                        f"({2 * se:.4f}, from {src}), so it is a real effect at n={n}")
    out["delta_vs_baseline"] = deltas
    out["floor_violations"] = violations
    out["passes_floor"] = not violations
    out["violations_inside_noise"] = flagged
    out["guard_violations"] = recs[0].get("guard_violations")
    out["passes_guards"] = recs[0].get("passes_guards")
    out["guards_note"] = "guards evaluated on the first repeat, for the same reason as axes"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--baseline", nargs="+", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.run_dirs]
    missing = [d for d in dirs if not d.exists()]
    if missing:
        print(f"!!! missing run dirs: {[str(d) for d in missing]}")
        return 2
    if len(dirs) < 2 and len(args.baseline or []) < 2:
        print("!!! score_repeats needs repeats on at least one side: either 2+ run dirs, "
              "or 1 run dir against 2+ --baseline dirs")
        return 2

    bl = [Path(b) for b in (args.baseline or [])]
    missing_b = [b for b in bl if not b.exists()]
    if missing_b:
        print(f"!!! missing baseline dirs: {[str(b) for b in missing_b]}")
        return 2
    rec = average_records(dirs, bl or None)
    rec["id"] = args.id or dirs[0].parent.name
    rec["iteration"] = args.iteration

    nb = rec.get("n_baseline_repeats", 1)
    print(f"=== {rec['id']}  candidate averaged over {rec['n_repeats']} repeats, "
          f"baseline over {nb}")
    print("    all three score columns are humanlikeness = 1 - Wasserstein-1 between the "
          "model's and\n    the humans' per-participant score distributions: range 0-1, "
          "in units of task\n    proportion-correct. `delta` is candidate minus baseline "
          "in those same units, so it\n    is the only column that can be negative.")
    print(f"\n{'task':<26}{'baseline':>10}{'candidate':>11}{'delta':>9}{'spread':>9}"
          f"  candidate per-run")
    bhl = rec.get("baseline_humanlikeness_by_task") or {}
    for t in sorted(rec["humanlikeness_by_task"]):
        v = rec["humanlikeness_by_task"][t]
        if v is None:
            continue
        b = bhl.get(t)
        d = rec["delta_vs_baseline"].get(t)
        s = rec["per_task_spread"].get(t)
        runs = " ".join(f"{x:.4f}" if x is not None else "-"
                        for x in rec["per_run_humanlikeness"][t])
        print(f"{t:<26}{(f'{b:.4f}' if b is not None else '-'):>10}{v:>11.4f}"
              f"{(f'{d:+.4f}' if d is not None else '-'):>9}"
              f"{(f'{s:.4f}' if s is not None else '-'):>9}  {runs}")
    if rec.get("n_baseline_repeats", 1) > 1:
        print(f"\nbaseline averaged over {rec['n_baseline_repeats']} repeats "
              f"{rec['baseline_per_run_humanlikeness']}")
    print(f"\nmean over search tasks: {rec['mean_humanlikeness_search']}   "
          f"per run {rec['mean_per_run']}")
    print(f"passes_floor {rec['passes_floor']}   passes_guards {rec['passes_guards']}")
    for v in rec["floor_violations"]:
        print(f"  FLOOR {v}")
    for f in rec["violations_inside_noise"]:
        print(f"  NOTE  {f}")

    if args.record:
        summary = ROOT / "meta_harness/logs/evolution_summary.jsonl"
        rows = []
        if summary.exists():
            for line in summary.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(r.get("id")) != str(rec["id"]):
                        rows.append(r)
        rows.append(rec)
        summary.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"\nrecorded {rec['id']} -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
