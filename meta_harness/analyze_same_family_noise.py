"""Measure craft_task's real run-to-run spread on the EPISODIC harness family.

Two runs of identical code for each of respond_first and respond_only. craft's request
sequences are provably identical to the baseline's on these candidates, so any spread
here is pure nondeterminism -- and it decides whether iteration 6's only blocker is a
candidate defect or an instrument artifact.

Prior craft noise measurements, all from OTHER harness families: 0.0000 (baseline
repeat), 0.0031 (primacy pair), and 0.0000 twice reported by iteration 5's proposer under
provable no-ops. If this pair shows ~0.03, those were unrepresentative and the floor is
too tight for this family.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
PY = "/Users/cl5625/simulating-memory/.venv/bin/python"
M = "Qwen_Qwen3-30B-A3B-Instruct-2507"
QB = f"meta_harness/runs/iter0/baseline/{M}"


def rec(rel, cid):
    p = subprocess.run([PY, "meta_harness/score_candidate.py", rel,
                        "--baseline", QB, "--id", cid],
                       cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"FAILED {cid}: {p.stderr[-500:]}")
        sys.exit(1)
    return json.loads(p.stdout[: p.stdout.rfind("}") + 1])


PAIRS = {
    "respond_first": (f"meta_harness/runs/iter6/respond_first/{M}",
                      f"meta_harness/runs/iter6rep/respond_first/{M}"),
    "respond_only": (f"meta_harness/runs/iter6/respond_only/{M}",
                     f"meta_harness/runs/iter6rep/respond_only/{M}"),
}

tasks = ["craft_task", "nback", "variable_mapping", "narrative_qa",
         "semantic_story_recall", "word_recognition",
         "digit_span_forward", "digit_span_reverse"]

print("TWO RUNS OF IDENTICAL CODE -- spread is pure run-to-run variation")
print("=" * 82)
recs = {}
for cand, (a, b) in PAIRS.items():
    ra, rb = rec(a, f"{cand}_A"), rec(b, f"{cand}_B")
    recs[cand] = (ra, rb)
    print(f"\n{cand}")
    print(f"  {'task':<24}{'run A':>9}{'run B':>9}{'spread':>9}{'floor':>8}  verdict")
    for t in tasks:
        va, vb = ra["humanlikeness_by_task"].get(t), rb["humanlikeness_by_task"].get(t)
        if va is None or vb is None:
            continue
        floor = max(0.03, {"digit_span_forward": 0.140, "digit_span_reverse": 0.059,
                           "nback": 0.060, "word_recognition": 0.121}.get(t, 0.03))
        s = abs(vb - va)
        flag = "EXCEEDS ITS OWN FLOOR" if s >= floor else ""
        print(f"  {t:<24}{va:>9.4f}{vb:>9.4f}{s:>9.4f}{floor:>8.3f}  {flag}")
    ma, mb = ra["mean_humanlikeness_search"], rb["mean_humanlikeness_search"]
    print(f"  {'MEAN':<24}{ma:>9.4f}{mb:>9.4f}{abs(mb - ma):>9.4f}")
    print(f"  floor/guards: A {ra.get('passes_floor')}/{ra.get('passes_guards')}   "
          f"B {rb.get('passes_floor')}/{rb.get('passes_guards')}")
    for r, lbl in ((ra, "A"), (rb, "B")):
        for v in r.get("floor_violations") or []:
            print(f"      {lbl} FLOOR {v}")

print("\n" + "=" * 82)
print("craft_task across ALL FOUR episodic-family runs of identical-or-no-op code:")
vals = []
for cand, (ra, rb) in recs.items():
    for lbl, r in (("A", ra), ("B", rb)):
        v = r["humanlikeness_by_task"].get("craft_task")
        d = (r.get("delta_vs_baseline") or {}).get("craft_task")
        vals.append(v)
        print(f"  {cand}_{lbl:<2} craft={v:.4f}  delta_vs_baseline={d:+.4f}")
print(f"  baseline craft = 0.8907")
print(f"  observed craft range across these runs: {min(vals):.4f} - {max(vals):.4f} "
      f"= {max(vals) - min(vals):.4f}")
print(f"  previously recorded craft run-to-run noise: 0.0000 - 0.0031")
