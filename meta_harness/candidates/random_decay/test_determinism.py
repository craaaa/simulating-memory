"""The random-decay control must be reproducible and must vary across
participants. Both properties are load-bearing: irreproducible, it is not a
trustworthy negative result; uniform, it does not widen the score distribution
and so is not an adversary at all.
"""
import sys
from pathlib import Path

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
sys.path.insert(0, str(ROOT))

import bench.cli  # noqa: F401,E402
from meta_harness.inject import apply, load_candidate  # noqa: E402
from meta_harness.verify_interface import StubLLM  # noqa: E402

cand = load_candidate(ROOT / "meta_harness/candidates/random_decay/harness.py")
apply(cand)
H = cand.WorkingMemoryAgent

def rate_for(content):
    h = H(llm=StubLLM(), condition_id="C2", temperature=0.0,
          debug=False, system_prompt_override=None)
    h.encode(content)
    return h._decay_rate

a1 = rate_for("participant A word list")
a2 = rate_for("participant A word list")
b1 = rate_for("participant B word list")

print(f"same content, run 1 : {a1:.6f}")
print(f"same content, run 2 : {a2:.6f}")
print(f"different content    : {b1:.6f}")
print()
print("reproducible across instances:", a1 == a2)
print("varies across participants   :", a1 != b1)

rates = sorted(rate_for(f"participant {i} stimuli") for i in range(30))
print(f"\n30 participants: min {rates[0]:.3f}  median {rates[15]:.3f}  max {rates[-1]:.3f}")
spread_ok = (rates[-1] - rates[0]) > 0.4
print("spread wide enough to widen the score distribution:", spread_ok)

ok = (a1 == a2) and (a1 != b1) and spread_ok
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
