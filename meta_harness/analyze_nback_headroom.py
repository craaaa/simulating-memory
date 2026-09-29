"""Is the n-back / leak-closure conflict structural, or just an unfixed failure?

I claimed closing the leak NECESSARILY costs n-back, because n-back is the one task where
the model is worse than humans and the leak was what made it good. The counter-question:
with the loophole closed, could a further change improve n-back AND the rest?

My own Qwen data already bears on this: evicting_reset closed the leak and reached n-back
0.9587, ABOVE the baseline's 0.7909. So on that substrate closing the leak and improving
n-back happened together, which contradicts "necessarily".

The question that decides it for Hermes: is its n-back loss a ceiling, or is it still
losing turns to a fixable failure? If turns are still being lost to tool-call-as-text with
zero refusals, that is route 3 -- the cumulative budget squeeze I diagnosed on Qwen and
explicitly declined to fix -- and there is headroom rather than a conflict.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "meta_harness"))
from bench.tasks.wm_nback import _parse_classification as pc  # noqa: E402
import nback_levels as NL  # noqa: E402
import nback_steps as NS  # noqa: E402

H = "NousResearch_Hermes-4-70B"
Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"
RUNS = {
    "qwen baseline": f"meta_harness/runs/iter0/baseline/{Q}",
    "qwen evicting": f"meta_harness/runs/iter5/evicting_reset/{Q}",
    "hermes baseline": f"meta_harness/runs/heldout/baseline/{H}",
    "hermes v3": f"meta_harness/runs/heldout/episodic_reset_v3/{H}",
    "hermes evicting": f"meta_harness/runs/heldout/evicting_reset/{H}",
}

print("n-back: where are the unanswered turns, and is anything still recoverable?")
print(f"{'run':<18}{'lvl':>4}{'answered':>10}{'of 14':>7}{'memfull':>9}"
      f"{'budget=0':>10}{'tc-as-text':>12}{'lost turns':>12}")
for name, rel in RUNS.items():
    p = ROOT / rel / "tasks/wm_nback.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    try:
        diag = NL.report(ROOT / rel).get("diagnostics", {})
    except Exception:  # noqa: BLE001
        diag = {}
    for lvl in (1, 2, 3):
        sel = [r for r in rows if r.get("n_level") == lvl]
        if not sel:
            continue
        t = mf = bz = tc = lost = 0
        for r in sel:
            # Tool state on the encode turn, reply on the answer turn; for pre-fix rows
            # these are the same entries. `step_log[1:]` and `step_log[1+lvl:]` assumed one
            # turn per letter and are wrong after the encode/answer split. See
            # nback_steps.py.
            enc = NS.encode_turns(r)
            ans = NS.answer_turns(r)
            for pos, st in sorted(enc.items()):
                t += 1
                for c in (st.get("tool_calls") or []):
                    if "memory is full" in str(c.get("result") or ""):
                        mf += 1
                if st.get("tool_call_budget_after") == 0:
                    bz += 1
                txt = str(ans.get(pos, st).get("text") or "")
                if "<tool_call>" in txt:
                    tc += 1
            # scored trials only -- the lead-in letters are not among the 14
            for st in NS.scored_answer_turns(r).values():
                if pc(re.sub(r"<tool_call>.*?</tool_call>", " ",
                             str(st.get("text") or ""), flags=re.S)) is None:
                    lost += 1
        a = (diag.get(lvl) or {}).get("answered")
        if t == 0:
            print(f"{name:<18}{lvl:>4}{str(a):>10}{'14':>7}"
                  f"{'-':>9}{'-':>10}{'-':>12}{'-':>12}  (no step_log)")
        else:
            print(f"{name:<18}{lvl:>4}{str(a):>10}{'14':>7}{mf / t:>9.3f}"
                  f"{bz / t:>10.3f}{tc / t:>12.3f}{lost / len(sel):>12.2f}")

print("\nThe decisive comparison, n-back humanlikeness:")
print("  qwen   baseline 0.7909  ->  evicting 0.9587   leak CLOSED and n-back ROSE")
print("  hermes baseline 0.8724  ->  evicting 0.7366   leak closed, n-back fell")
print("\nSo 'closing the leak necessarily costs n-back' is false: Qwen is a"
      "\ncounterexample from my own data. The question is whether Hermes' shortfall"
      "\nis a ceiling or an unfixed failure -- read `tc-as-text` and `lost turns`"
      "\nabove with memfull at 0.")
