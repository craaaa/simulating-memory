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

3. `wm_nback` runs two turns per letter: answer (store + restated instructions + the current
   letter, tools OFF), then encode (store + letter, tools ON). The store is injected on both,
   which is the only way the agent can read it -- `TOOLS` has no read tool. The order is
   answer-first because encoding first overwrote the comparison target at n=1;
   `wm_variable_mapping` keeps the opposite order because its question asks about stored
   content rather than the stimulus just presented.

4. `wm_word_recognition` runs the same answer-then-encode pair per word, with no `encode()`
   call and no re-printed list, and stops presenting at the third error as the human protocol
   does. Its reason for answering first is its own: a tool call must not be able to crowd out
   the reply.

Run: python meta_harness/test_turn_boundary_reset.py
"""
from __future__ import annotations

import json
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

    # Per letter, in the order the task now issues them: the ANSWER turn replies (one
    # request, tools off), then the ENCODE turn writes one key (two requests: the tool call,
    # then the follow-up).
    replies: list[Any] = []
    for i in range(n_letters):
        replies.append(_reply("different"))
        replies.append(LLMToolResponse(
            tool_calls=[_call("write_memory", f"position_{i + 1}", f"c{i}")],
            content=None, finish_reason="tool_calls"))
        replies.append(_reply("stored"))
    llm = RecordingLLM(replies)

    res = run_nback_block(llm=llm, block=block, condition_id="C2",
                          temperature=0.0, debug=False)

    assert len(res["encode_steps"]) == n_letters, res["encode_steps"]
    assert len(res["answer_steps"]) == n_letters, res["answer_steps"]
    # no standalone instruction turn: step 0 is the first letter's ANSWER turn
    assert res["answer_steps"][0] == 0, res["answer_steps"]
    # ANSWER precedes ENCODE for every letter. Encoding first overwrote the comparison
    # target at n=1 -- the model keeps one `previous_letter` key and destroyed it before
    # being asked. See logs/postfix_baseline_outcome.md.
    for a, e in zip(res["answer_steps"], res["encode_steps"]):
        assert a < e, (a, e)

    log = res["step_log"]
    enc_msg = log[res["encode_steps"][1]]["user_message"]
    ans_msg = log[res["answer_steps"][1]]["user_message"]

    # encode turn: store + the letter + a position index, tools allowed
    assert "Your working memory currently contains:" in enc_msg, enc_msg
    assert "New letter (position 2):" in enc_msg, enc_msg
    assert block.full_sequence[1] in enc_msg, enc_msg

    # answer turn: store + restated instructions + the count + THE CURRENT LETTER. Only the
    # letter n positions back must come from the store; the stimulus is on screen for a human.
    assert "Your working memory currently contains:" in ans_msg, ans_msg
    assert "Original task instructions:" in ans_msg, ans_msg
    assert "including this one: 2." in ans_msg, ans_msg
    assert f"Next letter: {block.full_sequence[1]}" in ans_msg, ans_msg
    # a lead-in turn must not presuppose a letter that does not exist yet
    lead = log[res["answer_steps"][0]]["user_message"]
    assert "including this one: 1." in lead, lead
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
    print("ok  n-back runs answer+encode per letter; answer turn shows the letter, tools off")


def test_nback_answer_sees_the_letter_n_back_at_n1() -> None:
    """The n=1 regression: a single-key store must still hold the PREVIOUS letter when asked.

    The first version of the encode/answer split encoded first, so a model keeping one
    `previous_letter` key overwrote it with the current letter before being asked about it.
    n=1 accuracy fell 0.9943 -> 0.4786 and 149 of 150 n=1 blocks ended with at most one key.
    This simulates exactly that model -- one key, always overwritten -- and asserts the
    answer turn can still see the letter one position back.
    """
    import random

    from bench.tasks.nback import generate_block
    from bench.tasks.wm_nback import run_nback_block

    class OneKeyModel:
        """Writes `previous_letter` = the letter in the encode prompt, nothing else."""

        def generate_with_tools(self, messages, tools, **kw):
            msg = messages[1]["content"]
            if "New letter (position" in msg:            # encode turn
                if any(m.get("role") == "tool" for m in messages):
                    return _reply("stored")
                letter = msg.split("New letter (position")[1].split(":")[1].split()[0]
                return LLMToolResponse(
                    tool_calls=[{"id": "c1", "type": "function",
                                 "function": {"name": "write_memory",
                                              "arguments": json.dumps(
                                                  {"key": "previous_letter",
                                                   "value": letter})}}],
                    content=None, finish_reason="tool_calls")
            return _reply("different")                    # answer turn

    block = generate_block(n=1, rng=random.Random(7))
    res = run_nback_block(llm=OneKeyModel(), block=block, condition_id="C2",
                          temperature=0.0, debug=False)
    log = res["step_log"]
    assert res["final_kv"] and len(res["final_kv"]) == 1, res["final_kv"]

    # For every letter after the first, the answer turn's store must show the PREVIOUS
    # letter, not the current one.
    checked = 0
    for pos in range(2, len(block.full_sequence) + 1):
        msg = log[res["answer_step_by_position"][pos]]["user_message"]
        store = msg.split("Original task")[0]
        prev, cur = block.full_sequence[pos - 2], block.full_sequence[pos - 1]
        assert f"previous_letter: {prev}" in store, (pos, prev, cur, store)
        if prev != cur:
            assert f"previous_letter: {cur}" not in store, (pos, cur, store)
        # and the current letter is present as the stimulus
        assert f"Next letter: {cur}" in msg, (pos, msg)
        checked += 1
    assert checked >= 10, checked
    print(f"ok  n=1 answer turn still sees the previous letter ({checked} letters checked)")


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


def test_nback_steps_handles_both_row_generations() -> None:
    """The step-log mapping helper must read pre-fix and post-fix rows correctly.

    Pre-fix: step 0 is the instruction turn, step i is letter i, one turn per letter.
    Post-fix: no instruction turn, two turns per letter, mapping carried in the row.
    Reading a post-fix row with the pre-fix rule doubles every per-turn count.
    """
    from meta_harness import nback_steps as NS

    pre = {
        "n_level": 2,
        "per_trial": [{"trial": i} for i in range(1, 15)],
        "step_log": [{"text": "instruction"}] + [{"text": f"t{i}"} for i in range(1, 17)],
    }
    assert not NS.is_split(pre)
    assert NS.counts(pre) == (16, 16), NS.counts(pre)
    # scored trial 1 is step n+1 = 3 -> "t3"
    assert NS.scored_answer_turns(pre)[1]["text"] == "t3", NS.scored_answer_turns(pre)[1]
    assert len(NS.scored_answer_turns(pre)) == 14

    post = {
        "n_level": 2,
        "per_trial": [{"trial": i} for i in range(1, 15)],
        "step_log": [{"text": f"{'e' if i % 2 == 0 else 'a'}{i // 2 + 1}"}
                     for i in range(32)],
        "encode_steps": list(range(0, 32, 2)),
        "answer_steps": list(range(1, 32, 2)),
        "answer_step_by_position": {str(p): 2 * p - 1 for p in range(1, 17)},
    }
    assert NS.is_split(post)
    assert NS.counts(post) == (16, 16), NS.counts(post)
    # every answer turn is an "a", every encode turn an "e" -- the kinds never mix
    assert all(v["text"].startswith("a") for v in NS.answer_turns(post).values())
    assert all(v["text"].startswith("e") for v in NS.encode_turns(post).values())
    # scored trial 1 is position n+1 = 3 -> step 5
    assert NS.scored_answer_turns(post)[1]["text"] == "a3", NS.scored_answer_turns(post)[1]
    assert len(NS.scored_answer_turns(post)) == 14
    print("ok  nback_steps maps both row generations, and never mixes turn kinds")


# ---------------------------------------------------------------------------
# 4. word_recognition's one word per turn
# ---------------------------------------------------------------------------

def _wr_trials(words: list[str], old_flags: list[bool]) -> list[dict[str, Any]]:
    return [{"trial_index": i + 1, "word": w, "is_old": o}
            for i, (w, o) in enumerate(zip(words, old_flags))]


def test_word_recognition_one_word_per_turn() -> None:
    """The stream is presented one word at a time, and never re-printed.

    Before this, `encode()` received all 100 trial lines and the recall prompt received the
    same 100 lines again, so Old/New was decidable from the prompt text without the store.
    """
    from bench.tasks.wm_word_recognition import run_recognition_stream

    words = ["ANCHOR", "BRIDGE", "ANCHOR", "CANDLE"]
    trials = _wr_trials(words, [False, False, True, False])
    # Per word: the ANSWER turn replies (one request, tools off), then the ENCODE turn
    # writes one key (two requests: the tool call, then the follow-up).
    replies: list[Any] = []
    for i, t in enumerate(trials):
        replies.append(_reply("old" if t["is_old"] else "new"))
        replies.append(LLMToolResponse(
            tool_calls=[_call("write_memory", f"seen_{i + 1}", "v")],
            content=None, finish_reason="tool_calls"))
        replies.append(_reply("stored"))
    llm = RecordingLLM(replies)

    res = run_recognition_stream(llm=llm, trials=trials, cond_id="C2",
                                 temperature=0.0, debug=False)

    assert res["trials_presented"] == 4, res["trials_presented"]
    assert res["stopped_at_third_error"] is None, res
    assert res["resp_map"] == {1: "New", 2: "New", 3: "Old", 4: "New"}, res["resp_map"]

    # ANSWER precedes ENCODE for every word. Unlike n-back the reason is not a destroyed
    # comparison target -- it is that a tool call must not be able to crowd out the reply.
    for a, e in zip(res["answer_steps"], res["encode_steps"]):
        assert a < e, (a, e)

    log = res["step_log"]
    ans = log[res["answer_step_by_position"][3]]["user_message"]
    enc = log[res["encode_steps"][2]]["user_message"]

    # answer turn: store + restated instructions + the count + THE CURRENT WORD, which is
    # the stimulus. Hiding it would test writing, not recognition.
    assert "Your working memory currently contains:" in ans, ans
    assert "Original task instructions:" in ans, ans
    assert "including this one: 3." in ans, ans
    assert "trial 3: ANCHOR" in ans, ans
    # encode turn: store + the word
    assert "Your working memory currently contains:" in enc, enc
    assert "New word (trial 3):" in enc, enc

    # NO turn may show a word the participant has not reached yet. This is the defect.
    for pos, idx in res["answer_step_by_position"].items():
        msg = log[idx]["user_message"]
        for later in words[pos:]:
            if later not in words[:pos]:
                assert later not in msg, (pos, later, msg)

    # the answer turn must carry no tool schemas
    ans_requests = [r for r in llm.requests if r["kind"] == "tools"
                    and "Original task instructions:" in r["messages"][1]["content"]]
    assert len(ans_requests) == 4, len(ans_requests)
    for r in ans_requests:
        assert r["tools"] == [], r["tools"]

    # and `encode()` is not used at all -- there is no study phase to encode
    assert not any("material to remember" in str(r.get("messages") or r.get("prompt"))
                   for r in llm.requests), llm.requests
    print("ok  word_recognition presents one word per turn, answer first, list never re-shown")


def test_word_recognition_stops_at_the_third_error() -> None:
    """Presentation ends on the third error, as the human protocol does.

    `trialsCompleted - correctResponses == 3` for 53 of 53 human records, so the human score
    is words survived. `score_game` already applies the rule analysis-side; stopping the loop
    means the trials a human would never have seen are also never presented.
    """
    from bench.tasks.word_recognition import score_game
    from bench.tasks.wm_word_recognition import run_recognition_stream

    words = ["ANCHOR", "BRIDGE", "CANDLE", "DAMSON", "ELIXIR", "FATHOM"]
    trials = _wr_trials(words, [False] * 6)
    replies: list[Any] = []
    for i in range(len(trials)):
        # every word is New; answering "old" is always wrong
        replies.append(_reply("old"))
        replies.append(_reply("nothing to store"))
    llm = RecordingLLM(replies)

    res = run_recognition_stream(llm=llm, trials=trials, cond_id="C2",
                                 temperature=0.0, debug=False)

    assert res["stopped_at_third_error"] == 3, res["stopped_at_third_error"]
    assert res["trials_presented"] == 3, res["trials_presented"]
    assert len(res["resp_map"]) == 3, res["resp_map"]

    scored = score_game(trials, res["resp_map"])
    assert scored["score"] == 0, scored
    assert len(scored["per_trial"]) == 3, scored["per_trial"]
    print("ok  word_recognition stops at the third error, and score_game agrees")


def test_word_recognition_unparsed_reply_is_not_an_error() -> None:
    """An unreadable reply must not count toward the three strikes.

    `score_game` records it as `correct: None` and neither counts it nor stops on it; the
    loop has to agree, or a garbage run would post a flattered score by stopping early.
    """
    from bench.tasks.wm_word_recognition import run_recognition_stream

    trials = _wr_trials(["ANCHOR", "BRIDGE", "CANDLE"], [False] * 3)
    replies: list[Any] = []
    for _ in range(3):
        replies.append(_reply("I am not sure about this one."))
        replies.append(_reply("nothing to store"))
    llm = RecordingLLM(replies)

    res = run_recognition_stream(llm=llm, trials=trials, cond_id="C2",
                                 temperature=0.0, debug=False)
    assert res["stopped_at_third_error"] is None, res
    assert res["trials_presented"] == 3, res["trials_presented"]
    assert res["resp_map"] == {}, res["resp_map"]
    print("ok  word_recognition: an unparsed reply is neither an error nor a stop")


def test_word_recognition_parser_refuses_ambiguous_replies() -> None:
    from bench.tasks.wm_word_recognition import _parse_old_new

    assert _parse_old_new("old") == "Old"
    assert _parse_old_new("New") == "New"
    assert _parse_old_new("  new\n") == "New"
    # a negated form names both words; guessing the first one reads this as Old
    assert _parse_old_new("not old, it is new") is None
    assert _parse_old_new("I cannot tell") is None
    assert _parse_old_new("") is None
    # substrings must not match
    assert _parse_old_new("the word is golden") is None
    print("ok  word_recognition parser refuses ambiguous and negated replies")


if __name__ == "__main__":
    test_second_turn_does_not_see_the_first()
    test_tool_loop_state_survives_within_a_turn()
    test_store_survives_the_reset()
    test_batch_path_request_sequence_unchanged()
    test_nback_encode_answer_split()
    test_nback_answer_sees_the_letter_n_back_at_n1()
    test_trial_accounting_survives_the_split()
    test_variable_mapping_encode_shows_the_store()
    test_nback_steps_handles_both_row_generations()
    test_word_recognition_one_word_per_turn()
    test_word_recognition_stops_at_the_third_error()
    test_word_recognition_unparsed_reply_is_not_an_error()
    test_word_recognition_parser_refuses_ambiguous_replies()
    print("\nall turn-boundary tests passed")
