"""Offline demonstration that refuse-on-full, not capacity, stalls n-back.

Free: a scripted stub LLM, no GPU, no API. Run from the repo root:

    python meta_harness/candidates/displacement/test_offline.py

The stub reproduces what the wave-0 n=3 traces imply the model does: each
letter-turn it tries to create a fresh `position_N` key holding that letter, and
if the store refuses the write it repairs with delete_key + write_memory. That
repair is forced -- the observed n=3 key sets are non-contiguous and high-index
(31 of 50 blocks end as position_2/3/8/10), which four occupied slots cannot
produce without a preceding delete.

Measured output (baseline vs this candidate, 16 letter-turns):

    baseline       budget pinned at 0 from turn 5, cap_hit on every later turn,
                   repairs truncated by tool_calls[:remaining] so the delete
                   lands and the replacement write is dropped,
                   final_kv = {position_1: C, position_4: C, position_15: A}
                   -- three stale, non-contiguous keys.

    displacement   cap_hit never fires, budget grows monotonically,
                   final_kv = {position_13: W, position_14: Z,
                               position_15: A, position_16: B}
                   -- the last four letters, contiguous.

`_tool_call_cap()` is identical in both. Displacement halves tool-call *demand*
(one write per letter instead of delete+write); it does not raise the supply.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import bench.cli  # noqa: F401,E402  import first so every binding exists
from bench.core.llm import LLM, LLMResponse, LLMToolResponse  # noqa: E402
from meta_harness.inject import apply, load_candidate  # noqa: E402

LETTERS = list("CXLCXLJXQLZMWZAB")
HERE = Path(__file__).resolve().parent


class NBackStub(LLM):
    """One new positional key per turn; repairs with delete+write on refusal."""

    def __init__(self) -> None:
        self.n = 0
        self.turn = 0

    def generate(self, prompt, *, system=None, **kw):
        return LLMResponse(text="stub")

    @staticmethod
    def _tc(cid, name, args):
        return {"id": cid, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}

    def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                            temperature=0.0, max_tokens=256, **kw):
        self.n += 1
        if tool_choice == "none":
            return LLMToolResponse(tool_calls=[], content="different",
                                   finish_reason="stop")
        last = messages[-1]
        if last.get("role") == "tool" and "memory is full" in str(last.get("content", "")):
            victim = f"position_{max(1, self.turn - 3)}"
            return LLMToolResponse(tool_calls=[
                self._tc(f"d{self.n}", "delete_key", {"key": victim}),
                self._tc(f"w{self.n}", "write_memory",
                         {"key": f"position_{self.turn}",
                          "value": LETTERS[self.turn - 1]}),
            ])
        if last.get("role") == "user":
            return LLMToolResponse(tool_calls=[
                self._tc(f"w{self.n}", "write_memory",
                         {"key": f"position_{self.turn}",
                          "value": LETTERS[self.turn - 1]})])
        return LLMToolResponse(tool_calls=[], content="different",
                               finish_reason="stop")


def run(label: str, harness_path: Path | None) -> dict:
    if harness_path is None:
        from bench.core.wm_agent import WorkingMemoryAgent as Agent
    else:
        cand = load_candidate(harness_path, name="cand_under_test")
        apply(cand)
        Agent = cand.WorkingMemoryAgent

    llm = NBackStub()
    agent = Agent(llm=llm, condition_id="C2", temperature=0.0, debug=False,
                  system_prompt_override="stub system")
    agent.step("This is a 3-back task.", allow_tools=False, max_tokens=64)

    print(f"\n===== {label} =====")
    cap_hits = 0
    for i, letter in enumerate(LETTERS, start=1):
        llm.turn = i
        agent.step(f"Next letter: {letter}", allow_tools=True, max_tokens=64)
        sl = agent.get_step_log()[-1]
        cap_hits += bool(sl["tool_call_cap_hit"])
        print(f"  t{i:2d} {letter}  budget_after={sl['tool_call_budget_after']:3d}"
              f"  cap_hit={str(sl['tool_call_cap_hit']):5s}"
              f"  calls={[c['name'] for c in sl['tool_calls']]}")
    store = agent.wm.store
    print(f"  -> cap_hit turns {cap_hits}/{len(LETTERS)}  final_kv={json.dumps(store)}")
    return {"cap_hits": cap_hits, "store": store}


def main() -> int:
    base = run("baseline (refuse-on-full)", None)
    cand = run("candidate (displacement)", HERE / "harness.py")

    failures = []
    if base["cap_hits"] < 8:
        failures.append("baseline did not stall on tool budget; premise not reproduced")
    if cand["cap_hits"] != 0:
        failures.append(f"displacement still stalls ({cand['cap_hits']} cap-hit turns)")
    if len(cand["store"]) != 4:
        failures.append(f"displacement holds {len(cand['store'])} keys, expected 4")
    tail = [LETTERS[-4:][i] for i in range(4)]
    if list(cand["store"].values()) != tail:
        failures.append(f"displacement store is not the last four letters: {cand['store']}")

    print()
    for f in failures:
        print(f"  FAIL: {f}")
    print("  PASS" if not failures else "  FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
