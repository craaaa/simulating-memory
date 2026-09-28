"""Guard against the double-apply defect that doubled every episodic prompt delta.

`run_candidate.py` called `verify_interface.check()`, which applies the candidate, and
then applied again. Injection is not idempotent -- the second load resolves its base
class from the name the first apply already rebound -- so both overrides ran and the
control-state block was emitted TWICE on 2400 of 2400 non-first n-back turns, in
`episodic_reset_v3`, `evicting_reset` and both held-out arms.

Two things are asserted here:
  1. a second apply() of the same candidate file RAISES rather than stacking silently
  2. a real candidate's step() emits its prompt delta exactly once
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAND = ROOT / "meta_harness/candidates/episodic_reset_v3/harness.py"

checks: dict[str, bool] = {}

# --- 1. the guard fires --------------------------------------------------------
import bench.cli  # noqa: E402,F401
from meta_harness import inject  # noqa: E402

cand = inject.load_candidate(CAND)
inject.apply(cand)
try:
    cand2 = inject.load_candidate(CAND, name="mh_candidate_again")
    inject.apply(cand2)
    checks["second apply raises"] = False
    reason = "no exception"
except RuntimeError as e:
    checks["second apply raises"] = "not idempotent" in str(e)
    reason = str(e)[:80]
print(f"  second apply -> {reason}")

# allow_reapply must still work, for anyone who genuinely wants stacking
try:
    cand3 = inject.load_candidate(CAND, name="mh_candidate_third")
    inject.apply(cand3, allow_reapply=True)
    checks["allow_reapply still permitted"] = True
except Exception as e:  # noqa: BLE001
    checks["allow_reapply still permitted"] = False
    print(f"  allow_reapply failed: {type(e).__name__}: {e}")

# --- 2. a clean process emits the delta once ----------------------------------
# Run in a SUBPROCESS so this test's own applies cannot contaminate it.
probe = r"""
import sys, json
sys.path.insert(0, %r)
import bench.cli
from meta_harness import inject
from types import SimpleNamespace

cand = inject.load_candidate(%r)
inject.apply(cand)
from bench.core.wm_agent import WorkingMemoryAgent

class Stub:
    def __init__(self): self.seen = []
    def generate_with_tools(self, messages=None, tools=None, **kw):
        self.seen.append(messages[-1]["content"])
        return SimpleNamespace(content="Same", tool_calls=[], text="Same")
    def generate(self, *a, **kw):
        return SimpleNamespace(content="x", text="x", tool_calls=[])

llm = Stub()
a = WorkingMemoryAgent(llm=llm, condition_id="C2")
a.step("Next letter: K")
a.step("Next letter: Q")
marker = "[ongoing episode]"
counts = [m.count(marker) for m in llm.seen]
print(json.dumps({"counts": counts}))
""" % (str(ROOT), str(CAND))

p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                   cwd=str(ROOT))
line = [l for l in p.stdout.splitlines() if l.startswith("{")]
if not line:
    checks["delta emitted once in a clean process"] = False
    print(f"  probe failed: {p.stderr[-300:]}")
else:
    counts = eval(line[-1])["counts"]  # noqa: S307  -- our own json, one line
    print(f"  marker count per prompt: {counts}")
    # first step of an episode carries no block; the second carries exactly one
    checks["delta emitted once in a clean process"] = counts[1:] == [1] * len(counts[1:])

print("\n=== inject idempotence")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
