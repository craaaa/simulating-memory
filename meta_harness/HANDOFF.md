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
   optimistic.** *(All `nback` humanlikeness figures in items 1–3 of this list are on the
   pre-2026-09-29 scoring shape and are not comparable with anything scored after Option D —
   see item 3 of the "comparability" list below. The qualitative point, that there is little
   n-back headroom, holds and is if anything stronger: n-back now scores 0.9633 against a
   mean-over-8 of 0.9167.)* Hermes's baseline `nback` humanlikeness is already **0.874**; Qwen's is
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

> **AMENDED 2026-09-29 — the "dist from human" column below is WITHDRAWN.** The
> `variable_mapping` humanlikeness gain in the paragraph above stands; the A4 agreement claimed
> here does not. **A4's human reference is void.** Human `variable_mapping` ends at the
> participant's FIRST error and `relationCount` is non-decreasing within a session, so every
> human "error trial" is simply their LAST question — necessarily the highest relation count
> they reached. 152 of 152 human records contain exactly one error, always their final question,
> and substituting "last answered question" for "error" reproduces the identical **1.3867**.
> The human 0.3728 measures the task's termination rule, not human interference structure.
>
> So the sentence below — "`respond_first` on Hermes is the closest match to human interference
> structure measured anywhere in this project (0.0023)" — is **retracted**. It compared the
> model's error distribution against a stopping rule. Nothing about `respond_first` is
> implicated; the reference was wrong, and the agreement was coincidence.
>
> The `errors` / `rc_raw` / `ceiling` / `rc_norm` columns are model-side and remain valid: the
> model answers every question, so its error trials are not selected by a stopping rule. They
> still support the weaker, internal claim that `respond_first`'s errors track interference load
> while `evicting_reset`'s do not. See `logs/a4_human_reference_invalid.md`, and prefer
> `report_error_shape.M1_variable_mapping_intrusion` for anything about mechanism — the stopping
> rule fixes *when* a human erred, not *what kind* of error it was, so the error-class shares
> are real observations.

    arm                  errors   rc_raw  ceiling  rc_norm  dist from human (0.3728)
    baseline                 30   1.2221   1.2564   0.8662   0.4934      <- column WITHDRAWN
    respond_first           845   1.3170   1.8451   0.3751   0.0023      <- column WITHDRAWN
    respond_first_rep2      835   1.2948   1.8219   0.3587   0.0141      <- column WITHDRAWN
    respond_first_rep3      852   1.3081   1.8621   0.3574   0.0154      <- column WITHDRAWN
    evicting_reset         1136   0.9320   2.6742  -0.0406   0.4134      <- column WITHDRAWN

All well past the 30-error trust threshold. ~~`respond_first` on Hermes is the closest match to
human interference structure measured anywhere in this project (0.0023)~~ — retracted, see the
amendment above — and it gets there with 845 errors rather than by having too few to
characterise. The baseline sits *exactly* at the threshold (30), so its 0.8662 is the least
trustworthy figure in that column. Note also that `evicting_reset`'s ratio is *below* 1.0 —
errors slightly ANTI-correlated with load — which is a different failure from the baseline's,
not a milder version of it. That last observation is model-internal and survives the amendment.

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
models and two candidate variants**.

**Amended 2026-09-29, and this reframes the whole paragraph.** Measured in
`analyze_live_dimensions.py` (commit dbd2ea3): **99–106% of every frontier candidate's
mean-over-8 delta came from `nback` and `variable_mapping`**, the other six tasks
contributed nothing outside 2 SE for four of five candidates, and `digit_span_reverse` was
exactly +0.0000 for all five. Those two tasks were exactly the ones that bypassed the memory
module. So the search was optimising the instrument, which is why the instrument was fixed
(commit 70befa7) rather than another candidate run. Everything above belongs to tag
`exp/compactor-prefix-v1`; **no number from before that tag is comparable to one after it.**
The `variable_mapping` gain in particular is now expected to become a *baseline* property,
since the baseline does what those candidates did.

## ~~The one thing to do next~~ RETRACTED 2026-09-29

> **Route 3 is dead. Do not spend an arm on it.** It is filed below under the account that
> "tools get denied → the model emits the tool call as text → the trial scores unanswered".
> The middle link is false: after the collection fix, spoken tool calls on n-back answer
> turns went from 553 of 2100 to **0 of 2100** while trials answered went **10.86 → 10.85 of
> 14**, and the baseline's own n-back humanlikeness did not move (0.7848 → 0.7838). The
> cumulative budget explains **6 of 965** unanswered trials.
>
> The live account: an n-back trial goes unanswered because the model spends that turn on
> bookkeeping and emits no label — 418 of 472 unanswered turns returned **empty assistant
> content**, the rest prose about the write. Every unanswered trial's own turn had
> `tool_call_cap_hit` set (964 of 965) and no turn without it was ever unanswered (0 of
> 2339). Evidence: `logs/unanswered_cause.md` and the OUTCOME section of
> `logs/postfix_prediction.md`.
>
> Routes 1 and 2 stand — they have independent measured effects (variable_mapping unparsed
> 38 → 0; answered 5.2–6.8 → 14.0 of 14).

**~~Fix route 3: the cumulative tool-call budget squeeze.~~** ~~It is the only thing standing
between `evicting_reset` and a candidate that passes on both substrates.~~

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
2. **Three of eight search tasks do not test the memory module** — **two are now fixed,
   2026-09-29, commit 70befa7.** `nback` and `variable_mapping` because `reset_messages()`
   was never called, so every stimulus stayed in context; `word_recognition` because the
   studied list is re-printed in the recall prompt. Measured: 36 of 50 word-recognition
   participants score a perfect 100/100 while their own store can decide at most 0.640 of
   old trials.
   - **`nback`, `variable_mapping`: FIXED.** `step()` clears the transcript at the turn
     boundary; both tasks now run an encode turn (store + stimulus, tools ON) and an answer
     turn (store + restated instructions, tools OFF). n-back had had **no read channel at
     all** — no read tool, no injection, the turn was the bare string `"Next letter: X"`, so
     its store was write-only.
   - **`word_recognition`: STILL OPEN**, fix planned, not landed. `word_list_text` and
     `trials_text` are built from the *same* 100 trial lines, so the sequence is printed
     twice. The task design is right (continuous recognition is study-equals-test); the
     presentation is wrong. Fix is one word per turn, answer-then-store, plus a decision on
     whether to replicate the human 3-strike stop (humans ~20 trials, model 100). Prerequisite:
     confirm `src/score.py` divides by trials attempted rather than a fixed 100.
3. **Digit span and n-back are comparable only after protocol/granularity matching**
   (`protocol_match.py`, `nback_levels.py`). The human staircase terminates on double
   failure; the model ran all 19 spans. Human n-back is one pooled score per participant;
   the model's is per `(participant, level)`.
   > **AMENDED 2026-09-29 [USER], "Option D" — the n-back half of this is now DONE, in
   > `src/score.py` itself rather than in a side module.** `human_scores("nback")` returns one
   > value per `(participant, n-level)`, matching the model, and the human's *n* **lead-in**
   > trials are dropped (no letter *n* back exists there; 112 of 318 are logged `target: true`,
   > which is impossible). `nback_levels.py` now delegates to `score.nback_human_by_level`
   > instead of carrying its own copy. Human *n* is 53 participants × 3 levels: 4 records have
   > `level: null` and their level is recovered from the block name. n-back humanlikeness moves
   > 0.9344 → **0.9622** on `iter11postfix` and 0.9340 → **0.9633** on `iter12stage2`, both
   > means over 3 arms; every n-back humanlikeness recorded before this date is on the old shape
   > and is **not** comparable. `score.nback_human_scores_legacy_pooled()` reproduces the old
   > shape. **The digit-span half is still open** — 1 trial/span on the model side against 2 on
   > the human side (audit M4). So is n-back's M9: the human keeps 14 − n scored trials per
   > block and the model 14, which M6 does not fix and which would need a `bench/` change.
   > See `logs/nback_denominator_decision.md`.
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
  maximum. Now normalized against the run's own ceiling; ~~**humans sit at 0.3728**, not near
  1.0~~ — **AMENDED 2026-09-29: strike the human figure. A4 has no valid human reference at
  all** (see the amendment earlier in this file and `logs/a4_human_reference_invalid.md`). The
  normalisation fix was real and stands; what it normalises against is a model-side ceiling, and
  A4 now survives only as an internal check that a candidate's manufactured errors track
  interference load. Its assignment window was also under-counted ~2× (`TURNS_PER_QUESTION`).
- **Noise floors were understated and are task-dependent.** Measured twice
  (`run_to_run_floor.json`): craft 0.0000–0.0031, word_recognition 0.0000, story 0.0012,
  nback 0.0023, narrative **0.0065–0.0160**, digit_span_forward 0.0152, 8-task mean 0.0001.
  Generated content varies run to run on every task even where the score does not, so **no
  prediction may demand bit-identity of text**.

  **AMENDED 2026-09-29: these figures were still too narrow, and craft_task's was the worst.**
  Re-measured over three identical repeats of `runs/iter11postfix/baseline` on the post-
  instrument-fix harness: craft **0.0357** (0.8907 / 0.8581 / 0.8550), story **0.0115**,
  narrative 0.0184, nback 0.0034, word_recognition 0.0199, variable_mapping 0.0033, both digit
  spans 0.0000. craft's true spread is over ten times the 0.0000–0.0031 recorded here and story's
  nearly ten times its 0.0012. `score_candidate.RUN_TO_RUN_SPREAD` now carries these and a delta
  must clear both them and the sampling floor. `word_recognition` 0.0000 was flagged in this file
  as unable to survive a task change, correctly — it is now 0.0199 and will change again once the
  one-word-per-turn rewrite is re-baselined.

  **AMENDED AGAIN 2026-09-29: every `nback` spread above is on the pre-Option-D scoring shape.**
  Re-measured under the new shape on the same six arms: **0.0060** (`iter11postfix`) and
  **0.0035** (`iter12stage2`), against the 0.0034 / 0.0061 recorded for the old shape. The kept
  constant, `score_candidate.RUN_TO_RUN_SPREAD["nback"] = 0.0061`, still bounds both, so nothing
  was retightened — the repo's rule is to keep the larger of two measurements. Note the two
  run-sets swapped which is the wider one, which is what three repeats pinning a spread only
  loosely looks like.

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

**Added 2026-09-29 — three more, all about the instrument rather than a candidate:**

- **I got the direction of the collection defect wrong twice**, then built a whole account on
  it. First: "the baseline loses ~549 of 1500 variable_mapping answers" — it loses **0**; that
  figure was `evicting_reset`'s. Then: "the losses are inside the candidates" — backwards for
  n-back, where the **baseline** lost 1091 of 4200 answer turns and `respond_first`/
  `respond_only` lost none. Cause: counting `maintenance_text` as an answer turn.
- **The artifact account itself was wrong** and I pre-registered a prediction that killed it:
  the post-fix baseline's n-back score was predicted to rise 0.7848 → ~0.87 and measured
  **0.7838**. Spoken tool calls were never the unanswered trials — the model was typing a
  call *alongside* an answer the parser already found. The `bench/` collection fix (eb3e96f)
  is correct as a measurement and changes **no score**; I oversold it.
- **`--baseline` in `score_repeats.py` silently kept only its last occurrence** (declared
  `nargs="+"` without `action="extend"`), so candidates reported as scored against two or
  three baselines were scored against one. Fixed 2026-09-29. `respond_first_v2`'s n-back
  delta moved −0.0784 → −0.0806 once three baselines actually applied; the verdict did not
  change, but the standard error it rested on was wrong.

The common shape: **four of the five things that moved a verdict in this project were
measurement defects, not candidate properties.** Three rejections were overturned by fixing
measurement (3-repeat averaging twice, the 3-run baseline once). Treat any single-run,
single-baseline, or single-generation number as provisional until it has survived a
measurement change.

## Practical notes

- **The frontier belongs to `exp/compactor-prefix-v1`, 2026-09-29.** Every membership claim
  below was measured on the leaky instrument, and the mechanism that put `respond_first`,
  `respond_only` and `evicting_reset` on it — closing the history leak — is now a baseline
  property. Read the frontier as history, and do not run a candidate against the post-fix
  baseline before checking it against the new turn structure: candidates carry their own
  `step()` copies, so the `bench/` fix does not reach them and several now close the leak
  twice. The open decision below (does held-out passage gate membership?) is moot until a
  post-fix frontier exists.
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
