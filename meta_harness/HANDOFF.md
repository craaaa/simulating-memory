# Handoff — Meta-Harness on the compactor

Written 2026-09-27. Entry point for picking this up later. Read this, then
`PROPOSER.md`'s preface, then the iteration sections at the end of `WORKLOG.md`.

---

## Amendment, 2026-09-28: the held-out set is now repeated, and it changes the plan

Three things below are superseded.

1. **`respond_first` fails held-out, hard.** Three Hermes repeats against two baseline
   repeats (job 18719683, one job, so no cross-job variation): `nback` **-0.7326**, and it
   is not degradation but cessation — `answered` is 0.00 of 14 at all three levels in all
   150 blocks, and on the 2100 turns where an answer is due the reply is `"no response"`
   99.1% of the time, against 87.9% `"same"`/`"different"` for the baseline. Identical
   per-level figures across three repeats. The response-ordering block is necessary for the
   collapse (`evicting_reset`, the same harness without it, answers 14.0/8.88/9.72); whether
   it is *sufficient* is what the queued `respond_only` Hermes job decides.
2. **`evicting_reset` still fails held-out against the averaged baseline** (`nback`
   -0.1377), as does `episodic_reset_v3` (-0.2454). Averaging overturned two Qwen
   rejections and was worth re-running here; it overturns neither of these.
3. **"Route 3 is the only thing between `evicting_reset` and a both-substrate pass" is too
   optimistic.** Hermes's baseline `nback` humanlikeness is already **0.874**; Qwen's is
   0.791. There is almost no n-back headroom on the held-out model, so the n-back gains that
   drove the entire search (+0.08 to +0.17) were specific to Qwen's worse baseline. On Hermes
   the episodic framing *costs* n-back. A route-3 fix can plausibly recover the -0.1377, but
   it cannot turn n-back into a gain there, and the contract's floor is what has to be
   cleared.

What survives untouched, and is still the durable result: closing the history leak is worth
**+0.34 to +0.60 on `variable_mapping`** — on Hermes it moves 0.3524 to 0.9533, the largest
single-task gain measured anywhere in this project. I checked whether A4 could actually see
it, since the guard passed on the first repeat only and A4's ceiling depends on a run's own
error count — the shape of a defect already fixed once here. It can, and it agrees:

    arm                  errors   rc_raw  ceiling  rc_norm  dist from human (0.3728)
    baseline                 30   1.2221   1.2564   0.8662   0.4934
    respond_first           845   1.3170   1.8451   0.3751   0.0023
    respond_first_rep2      835   1.2948   1.8219   0.3587   0.0141
    respond_first_rep3      852   1.3081   1.8621   0.3574   0.0154
    evicting_reset         1136   0.9320   2.6742  -0.0406   0.4134

All well past the 30-error trust threshold. `respond_first` on Hermes is the closest match to
human interference structure measured anywhere in this project (0.0023), and it gets there
with 845 errors rather than by having too few to characterise. The baseline sits *exactly* at
the threshold (30), so its 0.8662 is the least trustworthy figure in that column. Note also
that `evicting_reset`'s ratio is *below* 1.0 — errors slightly ANTI-correlated with load —
which is a different failure from the baseline's, not a milder version of it.

Two measurement notes. `score_repeats.py` now takes several `--baseline` dirs and averages
them, and its standard error counts the baseline's noise even when the baseline ran once (a
single draw carries the full family variance, not zero) and never trusts an observed sd of 0.
Under that correction **primacy's craft violation is inside 2 SE, not outside** — I had
reported it as one of the most robust effects here, on a spread that describes the
candidate's three runs rather than the difference against a once-measured baseline. Two more
Qwen baseline runs are queued (18754201, 18754204) so that term stops being an assumption.

Finally: job 18719683's baseline arm **overwrote** the earlier Hermes baseline run, and
`runs/` is gitignored, so the originally recorded `evicting_reset` held-out comparison cannot
be reproduced from disk. The `heldout_evicting_reset` record in `evolution_summary.jsonl`
supersedes it.

## Where it stands in one paragraph

Five waves, twelve candidates, two substrates. **No candidate passes the contract on both
substrates.** One mechanism transfers robustly and is the durable result: closing the
conversation-history leak is worth **+0.32 to +0.34 on `variable_mapping` across two
models and two candidate variants**. The blocker in both cases is `nback`, and the reason
is a *known, measured, unfixed* failure with a named target — not a limit of the approach.

## The one thing to do next

**Fix route 3: the cumulative tool-call budget squeeze.** It is the only thing standing
between `evicting_reset` and a candidate that passes on both substrates.

The failure family, which every result in this project turns out to be an instance of:
**tools get denied, and the model emits the tool call as plain text instead of answering.**
That reply carries no classification, so the trial scores unanswered. Three routes reach it:

| route | cause | status |
|---|---|---|
| 1 | a refusing store burns the call budget (3 calls to change one slot) | **fixed** by eviction |
| 2 | `allow_tools=False` on a question turn | **fixed** by v3's response obligation (vm unparsed 38 → 0) |
| 3 | the cumulative budget runs dry from *ordinary* writing | **OPEN** |

Route 3, measured on both substrates with refusals already at zero:

    run              lvl   memfull  budget=0  tc-as-text  lost turns/block
    hermes evicting   n=2    0.000     0.399       0.114       1.82
    hermes evicting   n=3    0.000     0.151       0.042       0.72
    qwen   evicting   n=2    0.000     0.209       0.045       0.66

The mechanism, from one real n=2 block (`analyze_refusal_budget.py` prints these):

    step  cap  budget_in  calls  answered
       1    6          6      2      yes
       2    6          4      2      yes
       3    6          2      2      yes
       4    6          0      0      NO     <- budget already spent entering the turn
       5    7          1      1      yes
       6    9          2      1      yes    <- surplus builds, never fails again

`_tool_call_cap()` is `max(6, int(1.5 * turns_so_far))`. An agent that opens by writing two
keys per turn spends the floor of 6 by turn 4, loses that turn, then recovers permanently
as the 1.5/turn allowance outpaces its ~1/turn demand. It is a **transient**, worth about
one trial per block on Qwen and about two on Hermes.

**Every proposer so far was forbidden from touching `_tool_call_cap()`**, on the reasoning
that more budget lets the agent churn faster without holding more, and that
`variable_mapping`'s humanlike errors are capacity-bound. That reasoning still applies to
*raising the cap as a capacity change*. It does not apply to the transient: there is now
cross-substrate evidence that the floor of 6 is mis-shaped for the first four turns, and
fixing the shape is not the same as raising the ceiling. A candidate that reshapes the
early-turn allowance without increasing total budget is the obvious next move, and the
alternative — a response obligation strong enough that the model answers *before* writing —
is arguably more principled, since a human does not skip responding because they are busy
rehearsing.

## What is established, and what it cost to establish

**Benchmark defects found (these affect anyone using this benchmark, not just us):**

1. `variable_mapping`'s two sides are scored by **different formulas**. Human is
   `sum(q.correct)/10`; model is `relation_count` of the last consecutively-correct
   question, which saturates by question 5 — so **every model error after question 5 is
   invisible to its own score**. `displacement` erred on 25 runs and 24 still scored 1.0.
   Corrected analysis-side in `protocol_match.variable_mapping_scores`.
2. **Three of eight search tasks do not test the memory module.** `nback` and
   `variable_mapping` because `reset_messages()` (`wm_agent.py:161`) is never called, so
   every stimulus stays in context; `word_recognition` because the studied list is
   re-printed in the recall prompt. Measured: 36 of 50 word-recognition participants score
   a perfect 100/100 while their own store can decide at most 0.640 of old trials.
3. **Digit span and n-back are comparable only after protocol/granularity matching**
   (`protocol_match.py`, `nback_levels.py`). The human staircase terminates on double
   failure; the model ran all 19 spans. Human n-back is one pooled score per participant;
   the model's is per `(participant, level)`.
4. **The refusal loop distorts every n-back number the benchmark has produced.** A full
   store costs up to three tool calls to change one slot against a 1.5/turn budget;
   `tool_call_cap_hit` fires on 76% of n=3 turns. Arms that refuse answer 5.2–6.8 of 14;
   arms that evict answer exactly 14.0. The baseline's own 6.82 is an artifact.
5. `wm_nback.jsonl` did not persist `step_log`, so per-turn prompts and replies were
   unrecoverable. **Now fixed** (two lines, behaviour-preserving) and it immediately paid
   for itself — the whole route-1/route-3 diagnosis came from that field.

**Defects in my own evaluation contract, all corrected:**

- **A3 scored recall LENGTH, not verbatimness.** Pooled Spearman(length, BLEU) = 0.822, and
  all 121 rows under 60 words score *exactly* 0.0000. Replaced with brevity-penalty-free
  4-gram precision on medians; validated by still rejecting `full_context` (98.9% of
  4-grams lifted verbatim) while clearing displacement's move toward human length.
- **A4's `rc_ratio` has an error-count-dependent ceiling**, so its fixed 1.15 threshold
  could never reject anything — `displacement`'s 1.2575 at 35 errors was its arithmetic
  maximum. Now normalized against the run's own ceiling; **humans sit at 0.3728**, not near
  1.0. Its assignment window was also under-counted ~2× (`TURNS_PER_QUESTION`).
- **Noise floors were understated and are task-dependent.** Measured twice
  (`run_to_run_floor.json`): craft 0.0000–0.0031, word_recognition 0.0000, story 0.0012,
  nback 0.0023, narrative **0.0065–0.0160**, digit_span_forward 0.0152, 8-task mean 0.0001.
  Generated content varies run to run on every task even where the score does not, so **no
  prediction may demand bit-identity of text**.

## What to distrust in my own conclusions

I overclaimed five times, each time by generalising a clean mechanism into a sufficiency or
impossibility claim without checking the cell that would refute it. All five are retracted
in place in `WORKLOG.md` with the refuting data. The pattern is specific enough to guard
against: **when a mechanism explains a difference, check every arm that already varies
along that mechanism before claiming it is the whole story.** `analyze_store_vs_step.py`
exists for exactly that check.

The most recent instance is the one to read, because it nearly closed off the right next
step: I claimed closing the leak *necessarily* costs n-back, which my own Qwen data
contradicts (`evicting_reset`: leak closed, n-back 0.7909 → 0.9587).

## Practical notes

- **`evicting_reset` holds the Qwen frontier but fails held-out.** Held-out rows are in
  `evolution_summary.jsonl` marked `instrument` so they cannot enter frontier derivation
  (their means are not comparable — the Hermes *baseline* is 0.8111 vs Qwen's 0.7861).
  **Open decision:** whether held-out passage should gate frontier membership. It would
  demote `evicting_reset` immediately and costs ~35 min per arm.
- **Held-out model is Hermes-4-70B, not Llama-3.3.** Llama-3.3's correct parser
  (`llama3_json`) rejects parallel tool calls and this harness requires them. That is a
  real limitation on generality: the mechanism needs a stack that supports parallel calls.
- Runtime: a search arm is ~14 min on 1×H200; a held-out arm ~35 min on 2×H200. Both are
  cheap — spend them rather than reasoning about what a run would show.
- Tools: `history.py` (list/show/diff/frontier/regressions), `check_predictions.py` (all 11
  candidates, four verdicts including VOID), `score_heldout.py`, `analyze_transfer.py`,
  `analyze_refusal_budget.py`, `analyze_store_vs_step.py`, `analyze_heldout_engagement.py`.
- The proposer pattern that worked: pre-registered predictions with a **precondition row
  that voids the rest on failure**, pitched at *restoration* rather than ambition. Two
  candidates lost creditable gains to preconditions pitched at the ambition. Three
  candidates were voided by their own preconditions, which is the system working.
