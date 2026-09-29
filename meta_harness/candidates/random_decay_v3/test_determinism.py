"""The v3 control must be reproducible, must vary per PARTICIPANT, and must fire on step().

Written because `random_decay/test_determinism.py` catches neither defect the audit found:

  * it hands every pseudo-participant a DISTINCT content string, so it never exercises
    the case the seeding defect is about -- several participants sharing one stimulus
    (measured: 4 distinct contents across 200 semantic_story_recall rows, 3 across 150
    craft_task rows). Under v1 those participants get ONE rate and the test still passes.
  * it drives only `encode()`, so it cannot see that three of the eight search tasks
    never call `encode()` at all.

So this test holds the content FIXED and varies only `participant_id`, and drives the
harness through `step()` rather than `encode()`.

Units: decay rates are proportions in [0, DECAY_MAX]; roll and key counts are integers.

Run: python meta_harness/candidates/random_decay_v3/test_determinism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import bench.cli  # noqa: F401,E402
from bench.core.llm import LLM, LLMResponse, LLMToolResponse  # noqa: E402
from meta_harness.inject import apply, load_candidate  # noqa: E402

cand = load_candidate(ROOT / "meta_harness/candidates/random_decay_v3/harness.py")
apply(cand)
H = cand.WorkingMemoryAgent
DECAY_MAX = cand.DECAY_MAX


class WritingStub(LLM):
    """Writes one fresh key on every tool-enabled turn, then stops calling tools.

    Deliberately per-INSTANCE state. A process-wide call counter is the trap that made an
    earlier instrumentation report "0 rolls changed the store" everywhere: after two calls
    every store was empty, so there was nothing left to drop and the zero was an artifact
    of the stub rather than a fact about the harness.
    """

    def __init__(self) -> None:
        self.turns = 0
        self.writes = 0

    def generate(self, prompt: str, *, system: str | None = None, **kw) -> LLMResponse:
        return LLMResponse(text="stub answer")

    def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                            temperature=0.0, max_tokens=256, **kw) -> LLMToolResponse:
        last_is_user = bool(messages) and messages[-1].get("role") == "user"
        if tools and last_is_user:
            self.turns += 1
            self.writes += 1
            return LLMToolResponse(tool_calls=[{
                "id": f"stub-{self.writes}",
                "type": "function",
                "function": {"name": "write_memory",
                             "arguments": json.dumps({"key": f"k{self.writes}",
                                                      "value": f"v{self.writes}"})},
            }])
        return LLMToolResponse(tool_calls=[], content="stub answer", finish_reason="stop")


FIXED_CONTENT = "the SAME stimulus every participant sees"
N_TURNS = 6


def drive(participant_id, *, n_turns: int = N_TURNS):
    """One agent driven through step() only -- never encode()."""
    h = H(llm=WritingStub(), condition_id="C2", temperature=0.0, debug=False,
          system_prompt_override=None, participant_id=participant_id)
    for i in range(n_turns):
        # An answer turn (tools OFF) then an encode turn (tools ON), the shape
        # wm_nback and wm_word_recognition actually use.
        h.step(f"answer turn {i}: {FIXED_CONTENT}", allow_tools=False)
        h.step(f"encode turn {i}: {FIXED_CONTENT}", allow_tools=True)
    return h


checks: dict[str, bool] = {}

# (a) identical content, different participant_id -> different rates.
a = drive("p1")
b = drive("p2")
print(f"identical content, participant_id p1 : rate {a._decay_rate:.6f}")
print(f"identical content, participant_id p2 : rate {b._decay_rate:.6f}")
checks["different participant_id on IDENTICAL content gives different rates"] = (
    a._decay_rate != b._decay_rate
)

# (d) same participant_id reproduces the same rate.
a_again = drive("p1")
print(f"participant_id p1 again              : rate {a_again._decay_rate:.6f}")
checks["same participant_id reproduces the same rate"] = (
    a._decay_rate == a_again._decay_rate
)

# (c) the rate is live after driving through step() only.
checks["_decay_rate != 0.0 after driving step() only"] = (
    a._decay_rate != 0.0 and b._decay_rate != 0.0
)
checks[f"rate within [0, {DECAY_MAX}]"] = all(
    0.0 <= h._decay_rate <= DECAY_MAX for h in (a, b, a_again)
)

# (b) the decay actually fired, once per TOOL-ENABLED turn and not on answer turns.
print(f"\np1: rolls {a.decay_rolls}  rolls that dropped {a.decay_rolls_that_dropped}"
      f"  keys dropped {a.decay_keys_dropped}")
print(f"p2: rolls {b.decay_rolls}  rolls that dropped {b.decay_rolls_that_dropped}"
      f"  keys dropped {b.decay_keys_dropped}")
checks[f"one roll per tool-enabled turn, {N_TURNS} of {2 * N_TURNS} steps"] = (
    a.decay_rolls == N_TURNS and b.decay_rolls == N_TURNS
)

# The population varies. 40 participants on ONE shared stimulus -- the case v1 fails.
rates = sorted(drive(f"p{i}", n_turns=1)._decay_rate for i in range(40))
print(f"\n40 participants, one shared stimulus: {len(set(rates))} distinct rates, "
      f"min {rates[0]:.3f} median {rates[20]:.3f} max {rates[-1]:.3f}")
checks["40 participants on one stimulus give 40 distinct rates"] = len(set(rates)) == 40
checks["spread covers most of the range"] = (rates[-1] - rates[0]) > 0.75 * DECAY_MAX

# participant_id=None must not raise: verify_interface.check() constructs without one.
none_agent = drive(None, n_turns=1)
print(f"\nparticipant_id=None : rate {none_agent._decay_rate:.6f}  "
      f"seeded_from_default {none_agent.seeded_from_default}")
checks["participant_id=None draws a rate rather than raising"] = (
    none_agent._decay_rate > 0.0 and none_agent.seeded_from_default is True
)

print()
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print("\nRESULT:", "PASS" if all(checks.values()) else "FAIL")
sys.exit(0 if all(checks.values()) else 1)
