"""Request-level diff on craft between baseline and respond_only.

craft's user prompt is byte-identical, the system prompt is unchanged, the control-state
block is not prepended on its single encode step, and respond_only uses the baseline
refusing store. Yet it writes 622 times vs 600, deletes 28 vs 50, and is 0.029 more
accurate. Either the requests differ somewhere I have not looked, or this is
nondeterminism and craft's measured 0.0000 run-to-run noise was unlucky.

Drives the REAL craft encode path under both harnesses with a recording stub, and
compares the message lists sent, turn by turn.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")

PROBE = r'''
import sys, json
sys.path.insert(0, sys.argv[2])
import bench.cli
from types import SimpleNamespace

cand_path = sys.argv[1]
if cand_path != "BASELINE":
    from meta_harness import inject
    inject.apply(inject.load_candidate(cand_path))

from bench.core.wm_agent import WorkingMemoryAgent

SCRIPT = [
    [("write_memory", {"key": "rule1", "value": "A and B make C"}),
     ("write_memory", {"key": "rule2", "value": "C and D make E"})],
    [("write_memory", {"key": "rule3", "value": "B and D make A"}),
     ("write_memory", {"key": "rule4", "value": "E and A make F"})],
    [("write_memory", {"key": "rule5", "value": "F and C make G"})],
    [],
]

class Stub:
    def __init__(self):
        self.requests = []
        self.i = 0
    def _next(self):
        step = SCRIPT[self.i] if self.i < len(SCRIPT) else []
        self.i += 1
        if not step:
            return SimpleNamespace(content="done", tool_calls=[], text="done")
        tcs = [{"id": f"c{self.i}_{j}", "type": "function",
                "function": {"name": n, "arguments": json.dumps(a)}}
               for j, (n, a) in enumerate(step)]
        return SimpleNamespace(content=None, tool_calls=tcs, text="")
    def generate_with_tools(self, messages=None, tools=None, tool_choice=None, **kw):
        self.requests.append({"n_messages": len(messages),
                              "tool_choice": tool_choice,
                              "roles": [m.get("role") for m in messages],
                              "last": str(messages[-1].get("content"))[:120]})
        return self._next()
    def generate(self, *a, **kw):
        self.requests.append({"generate": True, "prompt": str(a[0])[:120] if a else None})
        return SimpleNamespace(content="answer", text="answer", tool_calls=[])

llm = Stub()
agent = WorkingMemoryAgent(llm=llm, condition_id="C2")
agent.encode("- A and B combine to form C.\n- C and D combine to form E.")
print("RESULT " + json.dumps({
    "requests": llm.requests,
    "final_kv": agent.wm.store,
    "n_steps": len(getattr(agent, "_step_log", [])),
}))
'''


def run(label, cand):
    p = subprocess.run([sys.executable, "-c", PROBE, cand, str(ROOT)],
                       capture_output=True, text=True, cwd=str(ROOT))
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(f"{label}: probe failed\n{(p.stderr or p.stdout)[-400:]}")
        return None
    return json.loads(line[-1][len("RESULT "):])


base = run("baseline", "BASELINE")
ro = run("respond_only", str(ROOT / "meta_harness/candidates/respond_only/harness.py"))
rf = run("respond_first", str(ROOT / "meta_harness/candidates/respond_first/harness.py"))
if not (base and ro):
    raise SystemExit(1)

for label, d in (("baseline", base), ("respond_only", ro), ("respond_first", rf)):
    if d is None:
        continue
    print(f"\n{label}: n_steps={d['n_steps']}  final_kv={sorted(d['final_kv'])}")
    for i, r in enumerate(d["requests"]):
        if r.get("generate"):
            print(f"  req{i}: generate() prompt={r['prompt'][:70]!r}")
        else:
            print(f"  req{i}: n_msgs={r['n_messages']} tool_choice={r['tool_choice']!r} "
                  f"roles={r['roles']}")

print("\n=== identical request sequence, baseline vs respond_only? ===")
same = base["requests"] == ro["requests"]
print(f"  {same}")
if not same:
    for i, (a, b) in enumerate(zip(base["requests"], ro["requests"])):
        if a != b:
            print(f"  first divergence at req{i}:")
            print(f"    baseline    : {a}")
            print(f"    respond_only: {b}")
            break
    if len(base["requests"]) != len(ro["requests"]):
        print(f"  request COUNT differs: baseline {len(base['requests'])} vs "
              f"respond_only {len(ro['requests'])}")
print(f"  final store same: {base['final_kv'] == ro['final_kv']}")
