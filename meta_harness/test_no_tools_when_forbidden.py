"""The two turn shapes that forbid tool calls must not send tool schemas.

This pins the collection fix in `bench/core/wm_agent.py`. Before it, every request carried
`tools=TOOLS`, including the `allow_tools=False` turn and the cap-exhausted turn, which sent
them with `tool_choice="none"`. The schemas render into the chat template either way, so the
model emits a `<tool_call>` block as ordinary content; nothing parses it back out when calls
are disallowed, so it was returned as the agent's reply and reached the task's answer parser,
which recorded the trial as unanswered. Measured on runs already collected: 554 of 1500
variable_mapping replies under evicting_reset, 2417 of 4950 n-back replies under
respond_first.

Because the model is MORE accurate than humans on most of these tasks, an unanswered trial
lowers its score and therefore RAISES its measured humanlikeness -- so this defect flattered
exactly the candidates that triggered it.

Run: python meta_harness/test_no_tools_when_forbidden.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.core.llm import LLMToolResponse  # noqa: E402
from bench.core.wm_agent import WorkingMemoryAgent, _strip_spoken_tool_calls  # noqa: E402

SPOKEN = '<tool_call>\n{"name": "write_memory", "arguments": {"key": "a", "value": "1"}}\n</tool_call>'


class RecordingLLM:
    """Records what each request was given, and replies as scripted."""

    def __init__(self, replies: list[LLMToolResponse]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                            temperature=0.0, max_tokens=256, **kw):
        self.requests.append({"tools": tools, "tool_choice": tool_choice})
        return self.replies.pop(0) if self.replies else LLMToolResponse(
            tool_calls=[], content="", finish_reason="stop")


def _call(name: str, key: str, cid: str) -> dict[str, Any]:
    return {"id": cid, "type": "function",
            "function": {"name": name,
                         "arguments": '{"key": "%s", "value": "v"}' % key}}


def test_allow_tools_false_sends_no_schemas() -> None:
    llm = RecordingLLM([LLMToolResponse(tool_calls=[], content="different",
                                        finish_reason="stop")])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    out = a.step("Next letter: R", allow_tools=False)
    assert out == "different", out
    assert llm.requests[0]["tools"] == [], llm.requests[0]
    print("ok  allow_tools=False sends no tool schemas")


def test_exhausted_budget_sends_no_schemas() -> None:
    # cap is max(6, interactions*1.5); spend the 6 on one turn, then the next turn's
    # request must carry no schemas.
    first = LLMToolResponse(
        tool_calls=[_call("write_memory", f"k{i}", f"c{i}") for i in range(6)],
        content=None, finish_reason="tool_calls")
    llm = RecordingLLM([first,
                        LLMToolResponse(tool_calls=[], content="ok", finish_reason="stop"),
                        LLMToolResponse(tool_calls=[], content="same", finish_reason="stop")])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    a.step("remember these")
    assert a._remaining_tool_calls() == 0, a._remaining_tool_calls()
    a.step("Next letter: R")            # allow_tools=True, but budget is gone
    last = llm.requests[-1]
    assert last["tools"] == [], last
    assert a._step_log[-1]["tool_call_cap_hit"] is True, a._step_log[-1]
    print("ok  exhausted budget sends no tool schemas, and still records cap_hit")


def test_spoken_call_does_not_become_the_answer() -> None:
    llm = RecordingLLM([LLMToolResponse(tool_calls=[], content=f"same\n{SPOKEN}",
                                        finish_reason="stop")])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    out = a.step("Next letter: R", allow_tools=False)
    assert out == "same", repr(out)
    assert a._step_log[-1]["spoken_tool_calls"], a._step_log[-1]
    # the transcript the model sees keeps the original content, so nothing is hidden from it
    assert SPOKEN in a._messages[-1]["content"], a._messages[-1]
    print("ok  a spoken tool call is stripped from the answer and recorded")


def test_refused_calls_are_recorded() -> None:
    first = LLMToolResponse(
        tool_calls=[_call("write_memory", f"k{i}", f"c{i}") for i in range(8)],
        content=None, finish_reason="tool_calls")
    llm = RecordingLLM([first,
                        LLMToolResponse(tool_calls=[], content="ok", finish_reason="stop")])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    a.step("remember these")
    refused = a._step_log[-1]["refused_tool_calls"]
    assert len(refused) == 2, refused          # 8 requested, cap 6
    assert refused[0]["reason"] == "tool_call_budget_exhausted", refused[0]
    print("ok  calls dropped by the cap are recorded instead of vanishing")


def test_strip_leaves_ordinary_text_alone() -> None:
    for s in ("different", "no response", "I will use write_memory next turn",
              "partial <tool_call> never closed"):
        assert _strip_spoken_tool_calls(s) == (s, []), s
    print("ok  ordinary replies, including a bare mention, are untouched")


if __name__ == "__main__":
    test_allow_tools_false_sends_no_schemas()
    test_exhausted_budget_sends_no_schemas()
    test_spoken_call_does_not_become_the_answer()
    test_refused_calls_are_recorded()
    test_strip_leaves_ordinary_text_alone()
    print("\nall passed")
