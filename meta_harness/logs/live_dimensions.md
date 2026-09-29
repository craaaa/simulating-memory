# Why every iteration turns into a bugfix: the objective has two live dimensions

Measured 2026-09-28 with `meta_harness/analyze_live_dimensions.py`. Qwen3-30B-A3B search
substrate, 3-run candidates against the 3-run pre-fix baseline.

**Units.** Humanlikeness = 1 − Wasserstein-1 between the model's and the humans'
per-participant score distributions. Range 0–1, higher = more human-like, **in units of task
proportion-correct** (0.03 = three percentage points of distributional distance). A **delta** is
candidate score minus baseline score in the same units, and is the only quantity that can be
negative.

## The finding

| candidate | mean-over-8 delta | from nback + variable_mapping | from the other 6 tasks | SE of the other-6 part | \|other6\|/SE |
|---|---|---|---|---|---|
| respond_first | +0.0654 | +0.0645 | +0.0010 | 0.0019 | 0.52 |
| respond_only | +0.0604 | +0.0572 | +0.0032 | 0.0023 | 1.40 |
| evicting_reset | +0.0534 | +0.0538 | −0.0004 | 0.0020 | 0.22 |
| primacy | +0.0107 | +0.0201 | **−0.0094** | 0.0018 | **5.23** |
| chunk_limit | −0.0019 | −0.0013 | −0.0006 | 0.0016 | 0.36 |

For four of five candidates the six non-plumbing tasks contribute **nothing distinguishable
from zero** (inside 2 SE). The one candidate where they are outside the noise is `primacy`, and
there the contribution is **negative**. So across the whole search the other six tasks have
never been a source of gain — only, occasionally, a small penalty.

`digit_span_reverse` is **exactly +0.0000 for all five candidates** — fully inert on Qwen, as it
already was on Hermes. `craft_task` and `digit_span_forward` move by less than 0.005 for
everything except `primacy`.

Per-task deltas:

| candidate | dsp_fwd | dsp_rev | nback | word_rec | var_map | narr_qa | story | craft |
|---|---|---|---|---|---|---|---|---|
| respond_first | +0.0051 | +0.0000 | **+0.1755** | +0.0009 | **+0.3402** | −0.0005 | +0.0024 | +0.0000 |
| respond_only | +0.0051 | +0.0000 | **+0.1437** | +0.0019 | **+0.3137** | +0.0108 | +0.0072 | +0.0005 |
| evicting_reset | −0.0051 | +0.0000 | **+0.0888** | −0.0125 | **+0.3417** | +0.0086 | +0.0071 | −0.0016 |
| primacy | +0.0000 | +0.0000 | **+0.1620** | −0.0282 | −0.0011 | −0.0278 | +0.0041 | −0.0233 |
| chunk_limit | +0.0132 | +0.0000 | −0.0090 | −0.0061 | −0.0015 | −0.0116 | −0.0039 | +0.0036 |

## Why this explains the treadmill

`nback` and `variable_mapping` are precisely the two tasks the benchmark's own audit
(`HANDOFF.md`, defect 2) says **do not exercise the memory module**: `reset_messages()`
(`bench/core/wm_agent.py:161`) is never called for them, so every stimulus stays in the
conversation context. Their scores are therefore governed by harness plumbing — whether an
answer gets collected, whether prior stimuli are still visible — rather than by what the store
holds.

So the objective's only movable dimensions are the two where plumbing sets the score. A search
over harness code will find that gradient, and it did. Each iteration then reads as a bugfix
because each candidate exploits the next-largest instrument defect, and the response has been to
close that defect one at a time instead of treating the class. Five of the ten defects found in
this project were in the evaluation contract itself, not in the benchmark.

## What is *not* explained away by this

The two plumbing tasks are not equivalent.

- **`nback` dissolves into collection.** `respond_first` reaches 14 of 14 trials answered with
  accuracy ≈0.86 against a human 0.8657, and the unanswered baseline trials are empty-content
  bookkeeping turns (`logs/unanswered_cause.md`). The gain is "the harness collected the answer",
  not "the model remembered differently".
- **`variable_mapping` does not.** Closing the conversation-history leak changes what the agent
  can see, which is a substantive intervention, and it produced the project's closest match to
  human interference structure: on Hermes, A4 normalized rc_ratio **0.3751 against the human
  0.3728** over 845 errors, with the task score moving 0.3524 → 0.9533. It reproduces across two
  models and two candidate variants, at +0.34 to +0.60. This is the one real finding and it
  should not be filed with the treadmill.

## The choice this forces

Either

1. **Fix the instrument first** — require content on an answer turn, call `reset_messages()` for
   `nback` and `variable_mapping`, stop reprinting the studied list in `word_recognition` — then
   re-baseline and restart the search. **This voids the comparability of all twelve candidate
   verdicts**, including the accepted one.
2. **Keep searching and state the scope honestly** — "mean humanlikeness over 8 search tasks" is
   mostly a constant plus two plumbing-sensitive tasks, and report it that way.

A cheap measure that stops the class either way, and that reuses machinery that has already
fired correctly three times: a **pre-registered collection precondition** that VOIDs an arm
before scoring if it does not collect. It cuts both ways, which is the point — at 10.85 of 14
answered the **baseline itself fails** any threshold one would impose on a candidate.
