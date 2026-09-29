"""respond_first_v2 must differ from respond_first only in how tools are offered.

The claim v2 exists to test is narrow: `respond_first` collapsed on the held-out model --
n-back `answered` 0.00 of 14 at every level, `"no response"` on 99.1% of the turns where an
answer was due -- and its ACT 1 request showed the model tool schemas while forbidding their
use, which is the defect just fixed in `bench/core/wm_agent.py`. 2417 of 4950 recorded
replies contained a spoken tool call.

If v2 recovers n-back, the collapse was a measurement artifact. That inference only holds if
v2 changes NOTHING ELSE, so this test pins the diff line by line, and checks behaviour
against a fake LLM rather than trusting the diff alone.

Run: python meta_harness/test_respond_first_v2_equivalence.py
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V1 = ROOT / "meta_harness/candidates/respond_first/harness.py"
V2 = ROOT / "meta_harness/candidates/respond_first_v2/harness.py"

# Every non-comment line the diff is allowed to touch, as a substring. Anything else fails.
ALLOWED = (
    "_strip_spoken_tool_calls",
    "from bench.core.wm_agent import _dispatch_tool",
    "tools=[]",
    "tools=TOOLS",
    "tool_choice=\"none\"",
    "tools=TOOLS if tools_available else []",
    "raw_response",
    "response_text",
    "spoken_calls",
    "self._messages.append({\"role\": \"assistant\", \"content\"",
)


def test_diff_is_confined_to_tool_offering() -> None:
    a = V1.read_text().splitlines()
    b = V2.read_text().splitlines()
    changed = [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0)
               if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    body = [ln for ln in changed if not re.match(r"^[+-]\s*#", ln)]
    offenders = [ln for ln in body if not any(tok in ln for tok in ALLOWED)]
    assert not offenders, "diff touches more than tool offering:\n" + "\n".join(offenders)
    print(f"ok  {len(body)} non-comment lines changed, all on the tool-offering path")


class FakeLLM:
    """Replies with a spoken tool call on ACT 1, then a real call on ACT 2."""

    SPOKEN = ('same\n<tool_call>\n{"name": "write_memory", "arguments": '
              '{"key": "a", "value": "1"}}\n</tool_call>')

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.n = 0
        self.wrote = False

    def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                            temperature=0.0, max_tokens=256, **kw):
        from bench.core.llm import LLMToolResponse
        self.requests.append({"tools": tools, "tool_choice": tool_choice})
        self.n += 1
        # Script on the REQUEST SHAPE rather than a call counter: the opening turn goes
        # through the parent's path, so a counter would mis-align with ACT 1 / ACT 2.
        if not tools:                        # a no-tools turn: answer, and speak a call
            return LLMToolResponse(tool_calls=[], content=self.SPOKEN,
                                   finish_reason="stop")
        if self.wrote:                       # ACT 2 already wrote once; stop calling
            return LLMToolResponse(tool_calls=[], content="", finish_reason="stop")
        self.wrote = True
        return LLMToolResponse(
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "write_memory",
                                      "arguments": '{"key": "a", "value": "1"}'}}],
            content=None, finish_reason="tool_calls")


def _agent(path: Path, llm):
    """Load the candidate and rebind it across bench, the way a real run does."""
    from meta_harness import inject
    mod = inject.load_candidate(path, name=f"cand_{path.parent.name}")
    inject.apply(mod)
    from bench.core import wm_agent
    return wm_agent.WorkingMemoryAgent(llm=llm, condition_id="C2")


def test_act1_sends_no_schemas_and_answer_is_clean() -> None:
    # v2 only: applying both candidates in one process would contaminate the rebinding,
    # so the v1 side of this comparison is the recorded run data, not a second apply.
    llm = FakeLLM()
    a = _agent(V2, llm)
    a.step("This is a 1-back task.")          # opening turn, parent's path
    before = len(llm.requests)
    out = a.step("Next letter: R")             # ordered turn -> ACT 1 + ACT 2
    act1 = llm.requests[before]
    assert act1["tools"] == [], act1
    assert "tool_call" not in out, repr(out)
    assert out.strip() == "same", repr(out)
    log = a._step_log[-1]
    assert log["ordered_turn"] is True, log
    assert log["spoken_tool_calls"], log
    assert "<tool_call>" in a._messages[2]["content"], a._messages[2]
    print("ok  ACT 1 sends no schemas; a spoken call is kept out of the answer, "
          "logged, and left in the transcript")


if __name__ == "__main__":
    test_diff_is_confined_to_tool_offering()
    test_act1_sends_no_schemas_and_answer_is_clean()
    print("\nall passed")
