"""Is primacy_v2 functionally identical to primacy under SERIAL arrival?

The batch rule never fired in the live run (craft overflow occupancy 4.0), and for a
single eviction primacy_v2's tool-result string is byte-identical to primacy's. If the
eviction victim and the resulting store are also identical under serial arrival, then
the two candidates were the SAME harness in that run -- and every difference between
their scores is run-to-run variation, not mechanism.

That matters a lot: narrative_qa differed 0.9263 vs 0.9423 between them, which is 0.016
and about 2.5x the 0.0065 I measured from a baseline repeat. A second, independent
estimate of run-to-run noise would then exist, and it would be much larger than the
first on the task three candidates have been charged a floor violation on.

Replays the real write sequences from the live runs through BOTH classes under serial
semantics and compares stores, victims and returned strings step by step.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "meta_harness"))
import inject  # noqa: E402

pr = inject.load_candidate(
    str(ROOT / "meta_harness/candidates/primacy/harness.py"), name="mh_eq_primacy")
pv = inject.load_candidate(
    str(ROOT / "meta_harness/candidates/primacy_v2/harness.py"), name="mh_eq_primacy_v2")

PrimacyMem = pr.PrimacyMemory
V2Mem = pv.PrimacyV2Memory
print(f"classes: {PrimacyMem.__name__}  vs  {V2Mem.__name__}")


class SerialOwner:
    """One write_memory call per assistant message -- the arrival pattern the live
    run actually exhibits, proven by the delete that follows each refusal."""
    def __init__(self):
        self._messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant",
             "tool_calls": [{"function": {"name": "write_memory"}}]},
        ]


M = "Qwen_Qwen3-30B-A3B-Instruct-2507"
TASKS = ["wm_craft_task", "wm_semantic_story_recall", "wm_narrative_qa",
         "wm_word_recognition", "wm_digit_span_forward"]

total_rows = mismatch_rows = total_writes = 0
examples = []
for rel in ("iter2/primacy", "iter3/primacy_v2"):
    for task in TASKS:
        p = ROOT / "meta_harness/runs" / rel / M / f"tasks/{task}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            calls = (row.get("encoding_log") or {}).get("tool_calls") or []
            seq = []
            for tc in calls:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(args, dict):
                    continue
                seq.append((tc.get("name"), args))
            if not seq:
                continue
            a, b = PrimacyMem(owner=SerialOwner()), V2Mem(owner=SerialOwner())
            total_rows += 1
            bad = None
            for name, args in seq:
                if name == "write_memory":
                    total_writes += 1
                    ra = a.write_key(str(args.get("key")), str(args.get("value")))
                    rb = b.write_key(str(args.get("key")), str(args.get("value")))
                elif name in ("delete_key", "clear_key"):
                    ra = a.clear_key(str(args.get("key")))
                    rb = b.clear_key(str(args.get("key")))
                else:
                    continue
                if ra != rb or a.store != b.store:
                    bad = (name, args.get("key"), ra, rb,
                           dict(a.store), dict(b.store))
                    break
            if bad:
                mismatch_rows += 1
                if len(examples) < 3:
                    examples.append((rel, task, row.get("id"), bad))

print(f"\nrows replayed:      {total_rows}")
print(f"write calls:        {total_writes}")
print(f"rows that DIVERGED: {mismatch_rows}")
if examples:
    for rel, task, rid, bad in examples:
        print(f"\n  {rel} {task} {rid}")
        print(f"    on {bad[0]}({bad[1]!r})")
        print(f"    primacy   -> {bad[2]!r}  store={sorted(bad[4])}")
        print(f"    primacy_v2-> {bad[3]!r}  store={sorted(bad[5])}")
else:
    print("\n=> IDENTICAL under serial arrival, on every replayed row and call.")
    print("   So in the live run the two candidates were the SAME harness, and every")
    print("   difference between their scores is run-to-run variation.")
