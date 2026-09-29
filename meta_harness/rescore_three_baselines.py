"""Re-decide every pre-fix Qwen candidate against a THREE-run baseline.

Until now every Qwen verdict in this project compared an averaged candidate to a baseline
measured ONCE, and the standard error treated that single draw as carrying no noise. It
carries a full draw of the family variance. Jobs 18754201 and 18754204 add two more baseline
runs, collected under the same pre-fix code as the candidates they are the control for -- a
post-fix baseline would be the wrong comparison for a pre-fix candidate.

Usage:
    python meta_harness/rescore_three_baselines.py            # print
    python meta_harness/rescore_three_baselines.py --record    # and update the summary
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"

BASELINES = [f"meta_harness/runs/iter0/baseline/{Q}",
             f"meta_harness/runs/iter8repA/baseline/{Q}",
             f"meta_harness/runs/iter8repB/baseline/{Q}"]

CANDIDATES = [
    ("respond_first", 6, [f"meta_harness/runs/iter6/respond_first/{Q}",
                          f"meta_harness/runs/iter6rep/respond_first/{Q}",
                          f"meta_harness/runs/iter6rep2/respond_first/{Q}"]),
    ("respond_only", 6, [f"meta_harness/runs/iter6/respond_only/{Q}",
                         f"meta_harness/runs/iter6rep/respond_only/{Q}",
                         f"meta_harness/runs/iter6rep2/respond_only/{Q}"]),
    ("evicting_reset", 5, [f"meta_harness/runs/iter6/evicting_reset/{Q}",
                           f"meta_harness/runs/iter7repA/evicting_reset/{Q}",
                           f"meta_harness/runs/iter7repB/evicting_reset/{Q}"]),
    ("primacy", 2, [f"meta_harness/runs/iter2/primacy/{Q}",
                    f"meta_harness/runs/iter7repA/primacy/{Q}",
                    f"meta_harness/runs/iter7repB/primacy/{Q}"]),
    ("chunk_limit", 2, [f"meta_harness/runs/iter2/chunk_limit/{Q}",
                        f"meta_harness/runs/iter7repA/chunk_limit/{Q}",
                        f"meta_harness/runs/iter7repB/chunk_limit/{Q}"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    missing = [b for b in BASELINES if not (ROOT / b).exists()]
    if missing:
        print(f"!!! missing baselines: {missing}")
        return 2

    for cid, it, dirs in CANDIDATES:
        gone = [d for d in dirs if not (ROOT / d).exists()]
        if gone:
            print(f"\n{cid}: MISSING {gone}")
            continue
        cmd = [PY, "meta_harness/score_repeats.py", *dirs, "--baseline", *BASELINES,
               "--id", cid, "--iteration", str(it)]
        if args.record:
            cmd.append("--record")
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        keep = [ln for ln in p.stdout.splitlines()
                if not ln.startswith("baseline averaged")]
        print("\n" + "=" * 78)
        print("\n".join(keep))
        if p.returncode != 0:
            print(p.stderr[-500:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
