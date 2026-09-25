"""Compare local vs released by participant id, not by file position.

Rows are written in completion order under 50-way parallelism, so index 0 in one
file is not the same participant as index 0 in the other. The earlier "stimuli
differ" result was that artifact.
"""
import json
from pathlib import Path

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
LOCAL = ROOT / "meta_harness/runs/gate/local_vllm_wm_word_recognition.jsonl"
REL = ROOT / "runs/compactor/qwen_qwen3-30b-a3b-instruct-2507/tasks/wm_word_recognition.jsonl"


def by_id(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        d[r["id"]] = r
    return d


L, R = by_id(LOCAL), by_id(REL)
print(f"ids: local {len(L)}  released {len(R)}  shared {len(set(L) & set(R))}")
print(f"local sample ids   : {sorted(L)[:3]}")
print(f"released sample ids: {sorted(R)[:3]}")

shared = sorted(set(L) & set(R))
if not shared:
    print("\nNO SHARED IDS -- cannot pair participants")
    raise SystemExit(1)

same_stim = 0
diffs = []
for i in shared:
    lw = tuple(t["word"] for t in (L[i].get("gold_trials") or []))
    rw = tuple(t["word"] for t in (R[i].get("gold_trials") or []))
    if lw == rw:
        same_stim += 1
    else:
        diffs.append((i, lw[:4], rw[:4]))

print(f"\nstimuli identical for {same_stim}/{len(shared)} paired participants")
for i, a, b in diffs[:3]:
    print(f"  {i}\n    local    {a}\n    released {b}")

if same_stim == len(shared):
    print("\n=> stimuli match; the difference is in the model's RESPONSES,")
    print("   i.e. genuinely the serving stack (vLLM bf16 vs OpenRouter).")
    # Paired per-participant score comparison is now meaningful.
    import numpy as np
    ls = np.array([L[i]["metrics"]["score"] for i in shared], float)
    rs = np.array([R[i]["metrics"]["score"] for i in shared], float)
    print(f"\n   paired score mean: local {ls.mean():.2f}  released {rs.mean():.2f}")
    print(f"   mean paired delta: {(ls - rs).mean():+.2f}")
    worse = int((ls < rs).sum())
    print(f"   local worse on {worse}/{len(shared)}, better on "
          f"{int((ls > rs).sum())}, tied on {int((ls == rs).sum())}")
