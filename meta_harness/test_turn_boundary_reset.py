"""The transcript must not cross a turn boundary, and the store must be readable.

Pins the instrument fix. Three claims:

1. `step()` starts every turn from the system prompt alone. Before the fix `_messages`
   accumulated every prior turn, so `wm_nback` and `wm_variable_mapping` -- the only two
   working-memory tasks presented turn-by-turn -- could be answered from the transcript
   without consulting the key-value store. They were also the only two tasks whose scores
   any harness candidate could move.

2. The batch tasks are byte-identical across the fix. Each calls `encode()` exactly once and
   `recall()` never reads `_messages`, so the reset cannot reach them. This test asserts the
   actual request sequence rather than the argument.

3. `wm_nback` now runs two turns per letter, matching `wm_variable_mapping`: encode (store +
   letter, tools ON), then answer (store + restated instructions, tools OFF, no letter). The
   store is injected on both, which is the only way the agent can read it -- `TOOLS` has no
   read tool.

Run: python meta_harness/test_turn_boundary_reset.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.core.llm import LLMResponse, LLMToolResponse  # noqa: E402
from bench.core.wm_agent import WorkingMemoryAgent  # noqa: E402


class RecordingLLM:
    """Records the full request for each call, and replies as scripted."""

    def __init__(self, replies: list[Any] | None = None) -> None:
        self.replies = list(replies or [])
        self.requests: list[dict[str, Any]] = []

    def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                            temperature=0.0, max_tokens=256, **kw):
        self.requests.append({"kind": "tools", "messages": [dict(m) for m in messages],
                              "tools": tools})
        return self.replies.pop(0) if self.replies else LLMToolResponse(
            tool_calls=[], content="", finish_reason="stop")

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512, **kw):
        self.requests.append({"kind": "plain", "prompt": prompt, "system": system})
        return LLMResponse(text="reply", raw=None)


def _reply(text: str) -> LLMToolResponse:
    return LLMToolResponse(tool_calls=[], content=text, finish_reason="stop")


def _call(name: str, key: str, cid: str) -> dict[str, Any]:
    return {"id": cid, "type": "function",
            "function": {"name": name,
                         "arguments": '{"key": "%s", "value": "v"}' % key}}


# ---------------------------------------------------------------------------
# 1. the transcript does not cross a turn boundary
# ---------------------------------------------------------------------------

def test_second_turn_does_not_see_the_first() -> None:
    llm = RecordingLLM([_reply("a"), _reply("b")])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    a.step("first letter: M", allow_tools=False)
    a.step("second letter: Z", allow_tools=False)

    first, second = llm.requests[0]["messages"], llm.requests[1]["messages"]
    assert [m["role"] for m in first] == ["system", "user"], first
    assert [m["role"] for m in second] == ["system", "user"], second
    assert second[1]["content"] == "second letter: Z", second[1]
    joined = " ".join(str(m.get("content")) for m in second)
    assert "first letter" not in joined, second
    print("ok  turn 2 sees the system prompt and its own message only")


def test_tool_loop_state_survives_within_a_turn() -> None:
    """The reset is at the turn boundary, not inside the loop -- tool results must persist."""
    llm = RecordingLLM([
        LLMToolResponse(tool_calls=[_call("write_memory", "k1", "c1")],
                        content=None, finish_reason="tool_calls"),
        _reply("different"),
    ])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    out = a.step("Next letter: R", allow_tools=True)
    assert out == "different", out
    roles = [m["role"] for m in llm.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"], roles
    print("ok  the tool-call loop keeps its own messages within one turn")


def test_store_survives_the_reset() -> None:
    llm = RecordingLLM([
        LLMToolResponse(tool_calls=[_call("write_memory", "position_1", "c1")],
                        content=None, finish_reason="tool_calls"),
        _reply("no response"),
        _reply("same"),
    ])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    a.step("Next letter: M", allow_tools=True)
    assert a.wm.store == {"position_1": "v"}, a.wm.store
    a.step("answer please", allow_tools=False)
    assert a.wm.store == {"position_1": "v"}, a.wm.store
    print("ok  the key-value store crosses the turn boundary, the transcript does not")


# ---------------------------------------------------------------------------
# 2. the batch tasks are untouched
# ---------------------------------------------------------------------------

def test_batch_path_request_sequence_unchanged() -> None:
    """encode() once then recall(): exactly two requests, the second with no history."""
    llm = RecordingLLM([
        LLMToolResponse(tool_calls=[_call("write_memory", "chunk_1", "c1")],
                        content=None, finish_reason="tool_calls"),
        _reply("stored"),
    ])
    a = WorkingMemoryAgent(llm=llm, condition_id="C2")
    a.encode("1. cat\n2. dog")
    a.recall(recall_prompt="Memory:\n{wm_contents}\nList them.")

    kinds = [r["kind"] for r in llm.requests]
    assert kinds == ["tools", "tools", "plain"], kinds
    # the encode turn: system + the material, nothing else
    enc = llm.requests[0]["messages"]
    assert [m["role"] for m in enc] == ["system", "user"], enc
    assert "cat" in enc[1]["content"], enc[1]
    # recall(): a bare generate, store injected, no transcript
    rec = llm.requests[2]
    assert rec["prompt"].startswith("Memory:\nchunk_1: v"), rec["prompt"]
    assert "cat" not in rec["prompt"], rec["prompt"]
    print("ok  batch path is encode(tool loop) + recall(plain), no history either way")


# ---------------------------------------------------------------------------
# 3. n-back's two turns per letter
# ---------------------------------------------------------------------------

def test_nback_encode_answer_split() -> None:
    from bench.tasks.nback import generate_block
    from bench.tasks.wm_nback import run_nback_block

    block = generate_block(n=2, rng=__import__("random").Random(0))
    n_letters = len(block.full_sequence)

    # every encode turn writes one key, every answer turn replies
    replies: list[Any] = []
    for i in range(n_letters):
        replies.append(LLMToolResponse(
            tool_calls=[_call("write_memory", f"position_{i + 1}", f"c{i}")],
            content=None, finish_reason="tool_calls"))
        replies.append(_reply("stored"))
        replies.append(_reply("different"))
    llm = RecordingLLM(replies)

    res = run_nback_block(llm=llm, block=block, condition_id="C2",
                          temperature=0.0, debug=False)

    assert len(res["encode_steps"]) == n_letters, res["encode_steps"]
    assert len(res["answer_steps"]) == n_letters, res["answer_steps"]
    # no standalone instruction turn: step 0 is the first letter's encode turn
    assert res["encode_steps"][0] == 0, res["encode_steps"]
    # the two interleave, encode before answer, for every letter
    assert res["encode_steps"] < res["answer_steps"], (res["encode_steps"],
                                                       res["answer_steps"])
    for e, a in zip(res["encode_steps"], res["answer_steps"]):
        assert e < a, (e, a)

    log = res["step_log"]
    enc_msg = log[res["encode_steps"][1]]["user_message"]
    ans_msg = log[res["answer_steps"][1]]["user_message"]

    # encode turn: store + the letter + a position index, tools allowed
    assert "Your working memory currently contains:" in enc_msg, enc_msg
    assert "New letter (position 2):" in enc_msg, enc_msg
    assert block.full_sequence[1] in enc_msg, enc_msg

    # answer turn: store + restated instructions + the count, NO letter presented
    assert "Your working memory currently contains:" in ans_msg, ans_msg
    assert "Original task instructions:" in ans_msg, ans_msg
    assert "Letters presented so far in this block: 2." in ans_msg, ans_msg
    assert "New letter" not in ans_msg, ans_msg
    # a lead-in turn must not presuppose a letter that does not exist yet
    lead = log[res["answer_steps"][0]]["user_message"]
    assert "Letters presented so far in this block: 1." in lead, lead
    assert "at position" not in lead, lead

    # the answer turn must carry no tool schemas
    ans_requests = [r for r in llm.requests if r["kind"] == "tools"
                    and "Original task instructions:" in r["messages"][1]["content"]]
    assert ans_requests, "no answer-turn request found"
    for r in ans_requests:
        assert r["tools"] == [], r["tools"]

    # scored trial k is answer_step_by_position[n + k], not step n + k
    by_pos = res["answer_step_by_position"]
    assert by_pos[block.n + 1] == res["answer_steps"][block.n], (by_pos, res)
    print("ok  n-back runs encode+answer per letter, answer turn is store-only, tools off")


def test_trial_accounting_survives_the_split() -> None:
    """Two step() calls per letter must not change the scoring denominator.

    Every prediction in logs/instrument_fix.md is stated in trials out of 14. The split
    doubles the number of turns, so `answered` and `acc_over_14` are the numbers most likely
    to be silently wrong -- 28 or 17 instead of 14.
    """
    import random

    from bench.tasks.nback import generate_block, score_block
    from bench.tasks.wm_nback import run_nback_block

    class AlwaysDifferent:
        def generate_with_tools(self, messages, tools, **kw):
            return LLMToolResponse(tool_calls=[], content="different",
                                   finish_reason="stop")

    for n in (1, 2, 3):
        block = generate_block(n=n, rng=random.Random(0))
        res = run_nback_block(llm=AlwaysDifferent(), block=block, condition_id="C2",
                              temperature=0.0, debug=False)
        sc = score_block(block, res["trial_map"])
        n_letters = len(block.full_sequence)
        assert len(res["step_log"]) == 2 * n_letters, (n, len(res["step_log"]))
        assert sc["answered"] == 14, (n, sc["answered"])
        assert len(sc["per_trial"]) == 14, (n, len(sc["per_trial"]))
        # all 14 answered, so acc_over_14 and acc_over_answered must agree
        assert abs(sc["accuracy_over_14"] - sc["accuracy_over_answered"]) < 1e-9, (n, sc)
        # the n lead-in turns are scored into buffer_map, never into the 14
        assert len(res["buffer_map"]) == n, (n, res["buffer_map"])
    print("ok  trial accounting is still 14 trials per block at every n level")


def test_variable_mapping_encode_shows_the_store() -> None:
    from bench.tasks.wm_variable_mapping import WM_ENCODE_PROMPT
    msg = WM_ENCODE_PROMPT.format(wm_contents="alice: paris", statements="bob -> rome")
    assert "Your working memory currently contains:" in msg, msg
    assert "alice: paris" in msg, msg
    assert "bob -> rome" in msg, msg
    print("ok  variable_mapping's encode turn shows the store")


if __name__ == "__main__":
    test_second_turn_does_not_see_the_first()
    test_tool_loop_state_survives_within_a_turn()
    test_store_survives_the_reset()
    test_batch_path_request_sequence_unchanged()
    test_nback_encode_answer_split()
    test_trial_accounting_survives_the_split()
    test_variable_mapping_encode_shows_the_store()
    print("\nall turn-boundary tests passed")
