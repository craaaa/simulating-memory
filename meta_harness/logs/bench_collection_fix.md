# The collection fix in `bench/core` — what changed, and what it invalidates

Approved by the user 2026-09-28. Three changes, ~35 lines, one behavioural.

## The defect

`WorkingMemoryAgent.step()` sent `tools=TOOLS` on **every** request, including the two turn
shapes that forbid calling them: `allow_tools=False`, and the branch where the tool-call
budget is exhausted. Both sent the schemas with `tool_choice="none"`.

The schemas are rendered into the chat template regardless of `tool_choice`. A model that
sees them emits a `<tool_call>` block as ordinary content, and with calls disallowed nothing
parses it back out — so it stayed in `resp.content`, was returned by `step()` as the agent's
reply, and reached the task's answer parser, which found no classification and recorded the
trial as **unanswered**.

Direction of the bias matters: the model is *more* accurate than humans on most of these
tasks, so an unanswered trial lowers its score and therefore **raises** its measured
humanlikeness. The defect flattered exactly the candidates that triggered it.

Measured on runs already on disk (`meta_harness/analyze_spoken_calls.py`), share of recorded
replies containing a spoken call:

| task | baseline | baseline_rep2 | respond_first | evicting_reset |
|---|---|---|---|---|
| nback | 102/2550 | 63/2550 | **2417/4950** | 127/2550 |
| variable_mapping | 0/1500 | 0/1500 | 0/1500 | **554/1500** |
| the other 8 tasks | 0 | 0 | 0 | 0 |

Note what this corrects: I had told the user the *baseline* lost ~549 of 1500 variable_mapping
answers this way. It does not — it loses none. The losses are inside candidates.

## The changes

1. `bench/core/llm_openai.py`, `bench/core/llm_anthropic.py` — an **empty** `tools` list now
   means "no tools this turn", and both `tools` and `tool_choice` are left out of the request
   entirely. Previously an empty list was still sent as a parameter.
2. `bench/core/wm_agent.py` `step()` — the two no-tools branches (they were duplicate code)
   are merged into one that passes `tools=[]`. This is the behavioural change: those turns no
   longer show the model schemas it cannot use.
3. `bench/core/wm_agent.py` — `_strip_spoken_tool_calls()` removes complete
   `<tool_call>…</tool_call>` blocks from the returned reply, as a defence for the case where
   a model emits the syntax unprompted. The block is recorded in the step log under
   `spoken_tool_calls`, and the **transcript the model sees keeps the original content**, so
   nothing is hidden from the model — only from the answer parser. This rule previously lived
   in the scoring contract's contamination check, which was the wrong layer.
4. `bench/core/wm_agent.py` — calls dropped because the cap was hit are recorded in
   `refused_tool_calls` rather than vanishing. The transcript stays self-consistent (the model
   is resent only the calls that were kept, each with a result), so this is observability, not
   a behaviour change.

Pinned by `meta_harness/test_no_tools_when_forbidden.py` (5 tests, offline, fake LLM).

## What this invalidates

Every run on disk was collected under the defect. Baselines are nearly unaffected (n-back 4%,
everything else 0), but **`respond_first` and `evicting_reset` must be re-run before their
gains can be believed**: 37% of `evicting_reset`'s variable_mapping trials and 49% of
`respond_first`'s n-back replies contained a spoken call.

## The candidates carry the same defect internally

`meta_harness/candidates/respond_first/harness.py:654` issues its own ACT 1 request with
`tools=TOOLS, tool_choice="none"` — the defect, reproduced inside the candidate, where the
`bench/` fix cannot reach it. `episodic_reset_v3`, `evicting_reset` and `respond_only` do the
same.

This is a live hypothesis for the held-out failure: Hermes answers `"no response"` on 99.1% of
n-back turns under `respond_first`, and ACT 1 is precisely a turn that shows schemas while
forbidding them. The fix is one line per candidate (`tools=[]`, drop `tool_choice`), but it
changes what was measured, so it belongs in a **new** candidate rather than an edit to a
recorded one.
