"""Did the mechanism even ENGAGE on Hermes-4-70B?

evicting_reset works by removing the store's refusal, which on Qwen was firing on ~47%
of n>=2 turns. If Hermes writes fewer keys, the store never fills, the refusal never
fires, and eviction changes nothing -- in which case the n-back reversal is NOT the
mechanism backfiring, it is something else entirely. Four tasks showing exactly 0.0000
delta on Hermes is the hint.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "meta_harness"))
import nback_levels as NL  # noqa: E402
import nback_steps as NS  # noqa: E402

H = "NousResearch_Hermes-4-70B"
Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"
RUNS = {
    "qwen baseline": f"meta_harness/runs/iter0/baseline/{Q}",
    "qwen evicting": f"meta_harness/runs/iter5/evicting_reset/{Q}",
    "hermes baseline": f"meta_harness/runs/heldout/baseline/{H}",
    "hermes evicting": f"meta_harness/runs/heldout/evicting_reset/{H}",
}

print(f"{'run':<18}{'lvl':>4}{'answered':>10}{'keys':>7}"
      f"{'memfull/turn':>14}{'displ/turn':>12}{'tc-as-text':>12}{'turns':>7}")
for name, rel in RUNS.items():
    d = ROOT / rel
    p = d / "tasks/wm_nback.jsonl"
    if not p.exists():
        print(f"{name:<18}  (no nback file)")
        continue
    try:
        diag = NL.report(d).get("diagnostics", {})
    except Exception as e:  # noqa: BLE001
        diag = {}
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for lvl in (1, 2, 3):
        sel = [r for r in rows if r.get("n_level") == lvl]
        if not sel:
            continue
        t = mf = dp = tc = 0
        for r in sel:
            # Tool results on the encode turn, the reply on the answer turn. Identical
            # entries for pre-fix rows; distinct after the encode/answer split, where
            # `step_log[1:]` would count every letter twice. See nback_steps.py.
            enc = NS.encode_turns(r)
            ans = NS.answer_turns(r)
            for pos, st in sorted(enc.items()):
                t += 1
                for c in (st.get("tool_calls") or []):
                    res = str(c.get("result") or "")
                    if "memory is full" in res:
                        mf += 1
                    if "displaced" in res:
                        dp += 1
                if "<tool_call>" in str(ans.get(pos, st).get("text") or ""):
                    tc += 1
        a = (diag.get(lvl) or {}).get("answered")
        k = (diag.get(lvl) or {}).get("keys_held")
        if t == 0:
            print(f"{name:<18}{lvl:>4}{str(a):>10}{str(k):>7}"
                  f"{'-':>14}{'-':>12}{'-':>12}{0:>7}   (no step_log persisted)")
        else:
            print(f"{name:<18}{lvl:>4}{str(a):>10}{str(k):>7}"
                  f"{mf / t:>14.4f}{dp / t:>12.4f}{tc / t:>12.4f}{t:>7}")

# Store occupancy on a batch task, where 0.0000 deltas appeared.
print("\nstore occupancy on craft_task (where Hermes showed exactly 0.0000 delta):")
print(f"{'run':<18}{'rows':>6}{'mean keys':>11}{'rows overflowing':>18}"
      f"{'memfull results':>17}")
for name, rel in RUNS.items():
    p = ROOT / rel / "tasks/wm_craft_task.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    occ = over = mf = 0
    for r in rows:
        kv = r.get("final_kv") or {}
        occ += len(kv)
        written = set()
        for tc in ((r.get("encoding_log") or {}).get("tool_calls") or []):
            if tc.get("name") == "write_memory":
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if isinstance(args, dict) and args.get("key"):
                    written.add(str(args["key"]))
            if "memory is full" in str(tc.get("result") or ""):
                mf += 1
        if len(written) > 4:
            over += 1
    print(f"{name:<18}{len(rows):>6}{occ / max(1, len(rows)):>11.2f}"
          f"{over:>18}{mf:>17}")
