"""Three-way transfer table: does the leak closure stand ALONE on a second substrate?

episodic_reset_v3 = leak closure + response obligation, refusing store.
evicting_reset    = the same, plus eviction.

The held-out run said the leak closure transfers (+0.34 both models) and eviction reverses
n-back (-0.14). If v3 alone clears the held-out contract, it is the first portable
candidate, and the right thing to report even though it loses to evicting_reset on Qwen.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
PY = "/Users/cl5625/simulating-memory/.venv/bin/python"
H = "NousResearch_Hermes-4-70B"
Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"
HB = f"meta_harness/runs/heldout/baseline/{H}"
QB = f"meta_harness/runs/iter0/baseline/{Q}"


def rec(run_dir, base, cid):
    p = subprocess.run([PY, "meta_harness/score_candidate.py", run_dir,
                        "--baseline", base, "--id", cid],
                       cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"FAILED {cid}\n{p.stderr[-700:]}")
        sys.exit(1)
    return json.loads(p.stdout[: p.stdout.rfind("}") + 1])


qb = rec(QB, QB, "qb")
q3 = rec(f"meta_harness/runs/iter4/episodic_reset_v3/{Q}", QB, "q3")
qe = rec(f"meta_harness/runs/iter5/evicting_reset/{Q}", QB, "qe")
hb = rec(HB, HB, "hb")
h3 = rec(f"meta_harness/runs/heldout/episodic_reset_v3/{H}", HB, "h3")
he = rec(f"meta_harness/runs/heldout/evicting_reset/{H}", HB, "he")

print("DELTA vs that substrate's own baseline")
print("=" * 78)
print(f"{'task':<24}{'qwen v3':>10}{'qwen evict':>12}"
      f"{'herm v3':>10}{'herm evict':>12}{'v3 transfers?':>15}")
for t in sorted(k for k, v in qb["humanlikeness_by_task"].items() if v is not None):
    row = []
    for cand, base in ((q3, qb), (qe, qb), (h3, hb), (he, hb)):
        a, b = cand["humanlikeness_by_task"].get(t), base["humanlikeness_by_task"].get(t)
        row.append(None if a is None or b is None else a - b)
    q3d, qed, h3d, hed = row
    if q3d is None or h3d is None:
        verdict = "-"
    elif abs(q3d) < 0.02 and abs(h3d) < 0.02:
        verdict = "flat both"
    elif q3d > 0.02 and h3d > 0.02:
        verdict = "HOLDS"
    elif q3d < -0.02 and h3d < -0.02:
        verdict = "holds(neg)"
    else:
        verdict = "DIFFERS"
    print(f"{t:<24}" + "".join(f"{v:>+10.4f}" if v is not None else f"{'-':>10}"
                               for v in (q3d, qed))
          + "".join(f"{v:>+10.4f}" if v is not None else f"{'-':>10}"
                    for v in (h3d, hed))
          + f"{verdict:>15}")
print(f"\n{'MEAN':<24}"
      f"{q3['mean_humanlikeness_search'] - qb['mean_humanlikeness_search']:>+10.4f}"
      f"{qe['mean_humanlikeness_search'] - qb['mean_humanlikeness_search']:>+12.4f}"
      f"{h3['mean_humanlikeness_search'] - hb['mean_humanlikeness_search']:>+10.4f}"
      f"{he['mean_humanlikeness_search'] - hb['mean_humanlikeness_search']:>+12.4f}")

print("\nABSOLUTE means:")
for lbl, r in (("qwen baseline", qb), ("qwen v3", q3), ("qwen evicting", qe),
               ("hermes baseline", hb), ("hermes v3", h3), ("hermes evicting", he)):
    print(f"  {lbl:<18}{r['mean_humanlikeness_search']:.4f}")

print("\nCONTRACT STATUS on the held-out substrate")
print("=" * 78)
for lbl, r in (("episodic_reset_v3", h3), ("evicting_reset", he)):
    print(f"  {lbl:<20} floor={str(r.get('passes_floor')):<6} "
          f"guards={str(r.get('passes_guards')):<6}")
    for v in r.get("floor_violations") or []:
        print(f"      FLOOR {v}")
    for g in r.get("guard_violations") or []:
        print(f"      GUARD {g[:110]}")
    ax = r.get("axes") or {}
    a4 = ax.get("A4") or {}
    print(f"      A4 n_errors={a4.get('n_errors')} "
          f"rc_norm={a4.get('rc_ratio_normalized')}  "
          f"vm_matched={(ax.get('variable_mapping_matched') or {}).get('humanlikeness_matched')}")
