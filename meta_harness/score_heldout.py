"""Score the held-out arms on Hermes-4-70B, baseline vs evicting_reset.

Both arms ran in ONE job, so the comparison is free of cross-job run-to-run variation.
The question is whether evicting_reset's gains are substrate-specific: the mechanism is
that a denied tool call makes the model emit the call as text instead of answering, and
that is a behavioural fact about Qwen3-30B-A3B which Hermes-4-70B may not share.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
PY = "/Users/cl5625/simulating-memory/.venv/bin/python"
MODEL = "NousResearch_Hermes-4-70B"
QWEN = "Qwen_Qwen3-30B-A3B-Instruct-2507"

HB = f"meta_harness/runs/heldout/baseline/{MODEL}"
HE = f"meta_harness/runs/heldout/evicting_reset/{MODEL}"


def rec(run_dir, baseline, cid):
    p = subprocess.run([PY, "meta_harness/score_candidate.py", run_dir,
                        "--baseline", baseline, "--id", cid],
                       cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"FAILED {cid}: {p.stderr[-600:]}")
        sys.exit(1)
    return json.loads(p.stdout[: p.stdout.rfind("}") + 1])


hb = rec(HB, HB, "heldout_baseline")
he = rec(HE, HB, "heldout_evicting_reset")

# the search-substrate numbers, for the transfer comparison
sb = rec(f"meta_harness/runs/iter0/baseline/{QWEN}",
         f"meta_harness/runs/iter0/baseline/{QWEN}", "qwen_baseline")
se = rec(f"meta_harness/runs/iter5/evicting_reset/{QWEN}",
         f"meta_harness/runs/iter0/baseline/{QWEN}", "qwen_evicting_reset")

print("HELD-OUT: Hermes-4-70B, both arms in one job")
print("=" * 86)
print(f"{'task':<24}{'qwen base':>11}{'qwen evict':>12}{'qwen d':>9}"
      f"{'herm base':>11}{'herm evict':>12}{'herm d':>9}{'transfer':>10}")
tasks = [t for t in sb["humanlikeness_by_task"]
         if sb["humanlikeness_by_task"][t] is not None]
for t in sorted(tasks):
    qb = sb["humanlikeness_by_task"].get(t)
    qe = se["humanlikeness_by_task"].get(t)
    xb = hb["humanlikeness_by_task"].get(t)
    xe = he["humanlikeness_by_task"].get(t)
    if None in (qb, qe, xb, xe):
        print(f"{t:<24}" + "".join(f"{('-' if v is None else round(v,4)):>11}"
                                   for v in (qb, qe, xb, xe)))
        continue
    qd, xd = qe - qb, xe - xb
    # does the direction and rough size carry over?
    if qd > 0.02:
        tr = "HOLDS" if xd > 0.02 else ("partial" if xd > 0.005 else "FAILS")
    elif qd < -0.02:
        tr = "holds(neg)" if xd < -0.02 else "differs"
    else:
        tr = "flat both" if abs(xd) <= 0.02 else "NEW MOVE"
    print(f"{t:<24}{qb:>11.4f}{qe:>12.4f}{qd:>+9.4f}"
          f"{xb:>11.4f}{xe:>12.4f}{xd:>+9.4f}{tr:>10}")

print()
for label, a, b in (("mean_humanlikeness_search", sb, se), ("", hb, he)):
    pass
print(f"{'MEAN (8 search tasks)':<24}"
      f"{sb['mean_humanlikeness_search']:>11.4f}{se['mean_humanlikeness_search']:>12.4f}"
      f"{se['mean_humanlikeness_search'] - sb['mean_humanlikeness_search']:>+9.4f}"
      f"{hb['mean_humanlikeness_search']:>11.4f}{he['mean_humanlikeness_search']:>12.4f}"
      f"{he['mean_humanlikeness_search'] - hb['mean_humanlikeness_search']:>+9.4f}")

print("\nheld-out contract status for evicting_reset:")
print(f"  passes_floor  {he.get('passes_floor')}")
print(f"  passes_guards {he.get('passes_guards')}")
for v in he.get("floor_violations") or []:
    print(f"    FLOOR {v}")
for g in he.get("guard_violations") or []:
    print(f"    GUARD {g[:120]}")
ax = he.get("axes") or {}
print(f"  A4 n_errors {(ax.get('A4') or {}).get('n_errors')}  "
      f"rc_norm {(ax.get('A4') or {}).get('rc_ratio_normalized')}")
print(f"  vm matched  {(ax.get('variable_mapping_matched') or {}).get('humanlikeness_matched')}")
