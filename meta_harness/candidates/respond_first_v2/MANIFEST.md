# respond_first_v2

`respond_first`, with one thing changed: the turns that forbid tool calls no longer show the
model tool schemas. Everything else — the episodic reset, the evicting store, the two-act
turn, the budget bookkeeping, the probe-turn control — is byte-identical to
`meta_harness/candidates/respond_first/harness.py`, and
`meta_harness/test_respond_first_v2_equivalence.py` pins that: 13 non-comment lines differ,
all of them on the tool-offering path.

## Why

`respond_first` passed the search contract on Qwen (mean 0.8479 against baseline 0.7861,
3 repeats) and then failed held-out on Hermes-4-70B in the worst way available: n-back
`answered` 0.00 of 14 at all three levels in all 150 blocks, with `"no response"` on 99.1% of
the 2100 turns where an answer was due, against the baseline's 12.74 answered.

The same run shows **2417 of 4950 recorded replies containing a spoken `<tool_call>` block**.
ACT 1 — the act that produces the answer — issued its request with `tools=TOOLS,
tool_choice="none"`. The schemas render into the chat template regardless of `tool_choice`,
so a model that sees tools it may not call types the call out instead of answering, and that
text carries no classification, so the trial is recorded as unanswered.

That is the defect fixed in `bench/core/wm_agent.py` for the parent `step()`, but ACT 1 builds
its own request, so the fix could not reach it.

## The change

1. ACT 1 sends `tools=[]` and no `tool_choice`.
2. ACT 2's exhausted-budget branch does the same.
3. A complete spoken block is stripped from the ANSWER and recorded in
   `step_log["spoken_tool_calls"]`; the transcript the model sees keeps the original content,
   so nothing is hidden from the model, only from the answer parser.

## Prediction, registered before running

This is a **restoration** claim, not an ambition. If the collapse was the artifact:

- Hermes n-back humanlikeness returns to within the floor of its baseline 0.874, i.e.
  delta ≥ −0.06. (Under `respond_first` it was 0.1417, delta −0.7326.)
- Hermes n-back `answered` returns to the baseline's order of magnitude, ≥ 10 of 14 at n=1.
- `spoken_tool_calls` is empty on essentially every ordered turn.
- Hermes variable_mapping keeps its gain, ≥ +0.30 against baseline 0.3524.

If n-back does **not** recover, the collapse is a property of answering-before-remembering on
this model, the artifact explanation is dead, and no further ordering candidate is worth
running on Hermes.

On Qwen the prediction is weaker and worth stating anyway: the parent's Qwen n-back gain
(+0.1694) came partly from trials scored unanswered — 644 of 4950 replies there contained a
spoken call — so v2's Qwen n-back may come in **lower** than the parent's. A drop there is not
a defect in v2; it is the parent's number having been inflated.
