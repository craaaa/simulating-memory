# Protocol mismatch audit: human test protocol vs model test protocol, task by task

Written 2026-09-29. **No code was changed for this document.** Every number below was measured
in this worktree from `runs/human/` and from the model runs named in each section.

## What is being audited, and in what units

The benchmark's headline number is **humanlikeness = 1 − W₁**, where W₁ is the Wasserstein-1
distance between the *human* per-participant score distribution and the *model* per-participant
score distribution for one task. It is dimensionless and bounded above by 1.0; because W₁ is in
the same units as the scores, humanlikeness is only meaningful when both score distributions are
in the *same* units on the *same* quantity. Three different kinds of number appear below and are
never mixed in a column:

| kind | unit | reference point |
|---|---|---|
| **score** | proportion of a task's maximum, in [0, 1] | the per-task denominator in `src/score.py:TASK_DENOM` |
| **trial count** | trials, questions, or spans | per participant, per side |
| **humanlikeness** | dimensionless, ≤ 1.0 | 1 − W₁ between the two score distributions |

Internal labels defined at first use:

- **A1 / `sub_span_leak`** (`meta_harness/protocol_match.py`) — the rate at which a participant
  fails a digit-span sequence *at or below* their own best span, over administered trials, in
  [0, 1]. Reference point: administered sub-ceiling trials, not all trials.
- **A2** — the word-recognition axis (`meta_harness/cluster/search_set.yaml` comment).
- **A3** — the story-recall verbatimness axis (4-gram precision on medians).
- **A4 / `rc_ratio`** — mean `relationCount` on `variable_mapping` error trials ÷ mean on correct
  trials; dimensionless, 1.0 = load-independent errors. `rc_ratio_normalized` rescales it to
  [0, 1] against the ceiling the run's own error count permits.
- **`acc_over_14`** (`bench/tasks/wm_nback.py`) — model n-back correct answers ÷ 14, the scored
  trials in one block. Unanswered trials count as wrong.
- **`best_span`** — highest sequence length at which a participant got at least one trial right,
  under the rule "stop at the first length where every administered trial failed". Unit: digits.

Model runs read: `meta_harness/runs/iter8repA/baseline/Qwen_Qwen3-30B-A3B-Instruct-2507`
(**qwenA**), `.../iter8repB/...` (**qwenB**), and
`meta_harness/runs/heldout/baseline/NousResearch_Hermes-4-70B` (**hermes**). `factual_qa` and
`map_task` have model rows only under `heldout/`, so **those two comparisons exist on one
substrate only.**

---

## Summary table

Severity: **INVALIDATING** = the two sides are not computing the same quantity, so the
comparison is void; **BIASING** = same quantity, known-direction distortion of a measurable
size; **COSMETIC** = real difference, no measured effect on a reported number.

| # | task | mismatch | severity | proposed fix |
|---|---|---|---|---|
| M1 | word_recognition | human score is a correct-**count over a fixed 100** while the human stops at 3 strikes after a median of 32 trials; model runs 100 so its score *is* a proportion. **And the human's proportion-correct is identically 1 − 3/n**, so no denominator fix makes it an accuracy | **INVALIDATING** | (b) analysis-side: report trials-to-3rd-error on both sides; the denominator fix is only the arithmetic floor |
| M2 | variable_mapping | human score is a correct-**count over a fixed 10** while the human terminates at the first error after a mean of 4.99 questions; model answers all 10. **And the human's proportion-correct is identically 1 − 1/n** | **INVALIDATING** | (b) analysis-side: report questions-to-1st-error on both sides; the denominator fix is only the arithmetic floor |
| M3 | variable_mapping | axis **A4** human reference is the stopping rule, not load-dependence | **INVALIDATING** (already retracted) | (c) documented; strike the human column |
| M4 | digit span (fwd + rev) | `best_span` is estimated from **1 trial/span** on the model side and **2 trials/span** on the human side; the estimator is strongly sensitive to this | **INVALIDATING** | (b) analysis-side: 2 trials/span on both, and re-baseline the digit-span humanlikeness figures |
| M5 | digit span (fwd + rev) | human staircase terminates on double failure; model administers all 19 spans (2–20) | BIASING (subsumed by M4 once the estimator is matched) | (b) analysis-side, as `protocol_match._administer` already does |
| M6 | n-back | human score divides by **all 42 non-practice trials** including the 6 lead-in trials where no answer is possible; model's `acc_over_14` divides by scored trials only | BIASING, −0.0088 score units | (b) analysis-side |
| M7 | n-back | scoring granularity: model = one observation per (participant, n-level); human = one pooled observation per participant | BIASING, +0.065 humanlikeness when corrected | (b) analysis-side (`meta_harness/nback_levels.py` already does it) |
| M8 | n-back | 112 of 318 human lead-in trials carry an impossible `target: true` | BIASING (inside M6) | (c) document; excluded once M6 is fixed |
| M9 | n-back | model gets 14 scored trials per block; human gets 14 − n (13/12/11) | BIASING, small | (c) document |
| M10 | narrative_qa | 49 of 520 human questions come from `data/narrative_QA_easy.json`, which the model never sees | BIASING | (c) document, or (b) drop the 4.5 affected human records |
| M11 | craft_task | 5 of 54 human records used `data/craft_task_old.json` (v2.0), which the model never sees | BIASING | (c) document, or (b) drop those 5 human records |
| M12 | craft_task, map_task | scoring granularity: model = one observation per 5-question trial; human = one pooled observation per 15 questions | BIASING, ±0.02 humanlikeness | (b) analysis-side |
| M13 | word_recognition | the model is shown the whole 100-word trial stream twice in full text (encode turn, then recall turn) and answers all 100 at once; humans saw one word per trial | **INVALIDATING** for the memory claim | (a) change `bench/` — costs a re-baseline of word_recognition |
| M14 | **all tasks** | "a participant" means a person on the human side and a seeded stimulus set on the model side; model between-participant variance is item difficulty, human is between-person | **INVALIDATING** for the interpretation of W₁ | (c) document — this is a design property, not a bug |
| M15 | all tasks | the model's system prompt states a 4-slot Cowan (2001) limit and instructs it to be "imperfect"; humans got only the task instructions | BIASING, unquantifiable here | (c) document |
| M16 | n-back, word_recognition, variable_mapping | humans were under real time pressure (measured); the model is untimed and turn-based | COSMETIC for scores, load-bearing for interpretation | (c) document |
| M21 | n-back | the human side has a **0% non-response rate by construction** (2618 of 2618 trials carry a response); the model's `acc_over_14` scores non-response as error, on 59 of 150 rows | BIASING, 0.157 proportion-correct on qwenA | (b) analysis-side: report `acc_over_answered` beside `acc_over_14` as a harness-health metric |
| M17 | semantic_story_recall | the human story mix is unequal (Eyespy 18, Pieman 15, Baseball 13, Oregon Trail 7); the model's is 50 each | BIASING | (b) analysis-side: reweight, or compare per story |
| M18 | semantic_story_recall | both sides report `summary.embeddingSimilarity` but only the model side's embedding model is verifiable from this repo | **UNVERIFIED** | (c) document as unverified |
| M19 | variable_mapping | 154 human records come from 54 distinct people (42 contributed 3 each); `src/score.py` treats each record as an independent participant | BIASING of the *n*, not the mean | (c) document |
| M20 | n-back | humans had an 8-trial 1-back practice block; the model gets a 5-letter worked example in-prompt at **every** level | COSMETIC | (c) document |

Tasks/dimensions checked and found **matched**, one line each:

- **Stimulus identity, word_recognition**: 173 distinct words on each side, all from
  `data/words.json`, overlap 173/173. Old-item rate human 0.486, model 0.502.
- **Stimulus identity, n-back**: 21-consonant alphabet identical on both sides; target rate
  0.357 at every level on both sides; 14 non-practice trials per level on both sides.
- **Stimulus identity, variable_mapping**: 10 names, 10 cities, both sides, all from
  `data/names.json` / `data/city.json`; 4 options on every question on both sides (human 759
  questions, model 1500).
- **Stimulus identity, factual_qa**: 100 questions, 4 options, identical question texts and
  identical answer letters on both sides (100/100) — the option ordering is not shuffled.
- **Stimulus identity, map_task**: 15 distinct question prompts, overlap 15/15; 2 options both
  sides.
- **Stimulus identity, narrative_qa** (within the shared bank): 100 common questions, identical
  answer letters 100/100.
- **Option counts**: craft_task 2 both sides; map_task 2 both sides; narrative_qa 4 both sides;
  factual_qa 4 both sides. No shuffling difference found on any of the four.
- **Digit sequences**: different random draws on the two sides (human 668 distinct presented
  sequences, model 190, overlap 10 at short spans) — same generative distribution, harmless.
- **Story recall length**: human recall mean 137.2 words (median 100, range 8–418), qwenA mean
  136.7 (median 109, range 75–397). Closely matched despite `recall_max_tokens: 16384`.

---

## 1. word_recognition

### M1 — the human score is a count over a denominator ~3× its own trial count

This is the largest single defect found and it was not on the original list in the form given.

**What the human did.** Continuous recognition, one word per trial, **stop at 3 strikes**.
Measured over all 53 records: `summary.strikesUsed == 3` for **53 of 53**, and
`payload.responses[-1].strikesAfterTrial == 3` for **53 of 53**;
`summary.exhaustedWordPool` is `false` for all 53. So every human terminated on the strike rule,
none by exhausting the pool. `summary.trialsCompleted`: mean **34.49 trials**, sd 24.24, median
**32**, range **4–102**; quartiles 15 / 32 / 46. `len(payload.responses) == trialsCompleted` for
all 53.

**What `src/score.py` does with it** (`src/score.py:174-176`):
`summary["correctResponses"] / 100.0`. `correctResponses` has mean 31.49 (range 1–99). So the
human "score" is **31.49 correct answers divided by 100**, a denominator ~3× the participant's
own trial count. It is not a proportion-correct.

**What the model did.** `bench/tasks/wm_word_recognition.py` has no strike rule:
`max_trials_per_game: 100` and `metrics.n_trials == 100` for **50 of 50** rows. `src/score.py:227`
reads `metrics["score"] / 100.0`, which for the model *is* a proportion-correct.

**Side by side** (qwenA; hermes in parentheses):

| quantity | unit | human | model |
|---|---|---|---|
| trials attempted per participant | trials | 34.49 mean, 32 median, 4–102 | 100, all rows |
| `src/score.py` value | "score" in [0,1] | **0.3149** (sd 0.2424) | **0.8164** (sd 0.3478) — hermes 0.6554 |
| proportion correct over own trials | proportion-correct | **0.8281** (sd 0.1854, range 0.250–0.971) | **0.8938** — hermes not recomputed |
| humanlikeness as reported | dimensionless | — | **0.4948** — hermes 0.6323 |
| humanlikeness with both sides as proportion-correct | dimensionless | — | **0.8659** — hermes **0.7411** |

**What the difference does.** It changes the human denominator, so the two distributions are in
different units and W₁ between them is meaningless. The reported word_recognition humanlikeness of
0.4948 is off by **+0.371** from the value the same runs give once both sides are
proportion-correct. Note the *direction*: the mismatch made the model look far *less* human than it
is on this task.

**And the denominator fix is not sufficient.** Because every human stopped on exactly the 3rd
strike, `trialsCompleted − correctResponses == 3` for **53 of 53** records, so the human
proportion-correct is **identically 1 − 3/n**, a monotone rescaling of how long the participant
survived. It carries no information about accuracy beyond trial count. This is the A4 defect class
(`meta_harness/logs/a4_human_reference_invalid.md`) applied to the *score* rather than to an axis:
the quantity being compared is the stopping rule.

**Severity: INVALIDATING.**

**Proposed fix: (b), analysis-side, in two parts.**

1. *Floor.* In `src/score.py`, divide the human branch by `summary["trialsCompleted"]` and the
   model branch by `metrics["n_trials"]`. This gives humanlikeness 0.8659 / 0.7411 above. Report
   it as the **arithmetic floor** on the error, not as a like-for-like comparison.
2. *The matched statistic is survival length.* Under a stop-on-*k*-errors rule, accuracy and
   duration are the same statistic, so the comparable quantity is **trials to the 3rd error** on
   both sides — already measured, no new computation: human **34.49** mean / **32** median /
   range 4–102 trials; model, emulating the strike rule on its own `per_trial` records,
   **82.92** mean / **100** median / range 3–100 trials.

Option (a) — putting a 3-strike rule into `bench/` — does not fix the score and the model errs too
rarely for it to bite (median 100 trials before a 3rd strike). The denominator, not the stopping
rule, is what breaks the arithmetic; the stopping rule is what makes even the fixed version not an
accuracy comparison.

### M13 — the model reads the test stream instead of remembering it

**What the human did.** One word per trial, decide old/new, no lookahead. First trial's
`expectedResponse` is `new` for 53 of 53.

**What the model did.** `bench/tasks/wm_word_recognition.py:100-110`: `word_list_text` is
`"\n".join(f"{t['trial_index']}: {t['word']}" for t in trials)` over **all 100 trials** and is
passed to `agent.encode(...)` in one turn. Then `trials_text` is built from *the same 100 trials
in the same order* and interpolated into the recall prompt. Verified on the released rows: the
`encoding_log.content` sequence equals the `gold_trials` word sequence exactly (100 = 100). The
model answers all 100 in one reply (`recall_raw` = `"trial 1: new\ntrial 2: new\n..."`).

**What the difference does.** A trial is "old" iff the word appeared earlier in the same printed
list, so the answer is derivable by reading the prompt; the 4-slot store is not needed.
Consistent with that: **19 of 50** qwenA rows and **21 of 50** hermes rows score exactly 100/100,
and score mass sits at {98: 5, 99: 12, 100: 19} against a 14-row tail of parse/refusal failures at
0–47. The task measures reading, not memory. This is `HANDOFF.md` defect 2's remaining open item
and it is confirmed here.

**Severity: INVALIDATING for any memory claim about this task**; the score comparison itself is
still arithmetic once M1 is fixed, it just is not about memory.

**Proposed fix: (a), change `bench/`** — one word per turn, answer before store, no re-print.
This costs a re-baseline of word_recognition on every model. It is the only mismatch here where
(a) is right, because no analysis-side transform can remove information the model already saw.

---

## 2. variable_mapping

### M2 — the human score is a count over a fixed 10 while the human stops at the first error

**What the human did.** Re-verified independently over all 154 records:

```
records 154 | no `questions` key 2 | answered nothing 0
errors per record: {1: 152}
the single error IS the last answered question: 152 of 152 | is NOT: 0
relationCount non-decreasing within participant: 152 of 152
questions asked per participant: 2:25 3:28 4:20 5:29 6:19 7:7 8:11 9:5 10:3 11:1 15:1 16:3
                                mean 4.99, sd 2.77, min 2, max 16
summary.strikesUsed: 1:54  2:48  3:50  (2 missing)
summary.taskEnded: False:102  True:50  (2 missing)
```

Every human has exactly one error and it is always their last answered question. The task
terminates there. `src/score.py:178-183` scores `sum(q["correct"]) / 10` — a correct **count**
over a fixed 10 — giving **0.3868** (sd 0.2352).

**What the model did.** `metrics["n_questions"] == 10` for **150 of 150** rows; the model answers
all ten regardless of errors (`metrics["first_error_at"]` = {3: 2, 8: 3, 9: 2, 10: 5, None: 138}).
`src/score.py:185` reads `metrics["score"] / 10`, which gives **0.9920** (sd 0.0690) over two
distinct values {1.0: 148, 0.4: 2}.

**Side by side** (qwenA; hermes in parentheses):

| quantity | unit | human | model |
|---|---|---|---|
| questions asked per participant | questions | 4.99 mean, range 2–16 | 10, all rows |
| `src/score.py` value | "score" in [0,1] | **0.3868** | **0.9920** (hermes 0.9947) |
| proportion correct over questions asked | proportion-correct | **0.7409** (sd 0.1294) | **0.9920** (hermes 0.9800) |
| humanlikeness as reported | dimensionless | — | **0.3554** (hermes 0.3524) |
| humanlikeness, both proportion-over-asked | dimensionless | — | **0.7653** (hermes 0.7765) |
| humanlikeness, both proportion-over-asked and model stopped at its own first error | dimensionless | — | **0.7686** (hermes 0.7798) |

**What the difference does.** Same shape as M1: a shifted human denominator. It moves the reported
humanlikeness by **+0.410**. Note also that applying the human *stopping rule* to the model on top
of the denominator fix adds only **+0.0033** — so the whole effect is the denominator, and the
termination rule per se is nearly free here.

**Note on the existing correction.** `meta_harness/protocol_match.variable_mapping_scores` is
right that the two sides used different *formulas* (the model's own `metrics["score"]` scores
`relation_count` of the last consecutively-correct question, which saturates by question 5) and it
makes the model compute `sum(correct)/10`. Measured: it moves the model from {1.0: 148, 0.4: 2} to
{1.0: 138, 0.9: 12}, mean 0.9920. That is a real improvement over the saturating formula, but it
**equalizes the formula without fixing the denominator** — both sides then compute count-over-10,
and the human's 10 is not the number of questions they were asked. This must not be read as
closing M2.

**And, as in M1, the denominator fix is not sufficient.** Every human record has exactly one error
and it is the last question asked, so the human proportion-correct is **identically 1 − 1/n**.
Confirmed arithmetically against the question-count distribution above: mean(1 − 1/n) = **0.7409**,
equal to the measured proportion-correct to four decimal places. The human "accuracy" on
variable_mapping is a monotone rescaling of how many questions they survived and nothing else.

**Severity: INVALIDATING.**

**Proposed fix: (b), analysis-side, in two parts.**

1. *Floor.* Divide each side by questions asked → humanlikeness 0.7653 / 0.7765 above. Report as
   the arithmetic floor.
2. *The matched statistic is survival length:* **questions to the 1st error**. Human mean
   **4.99**, range 2–16. Model: censored at 10 for 143 of 150 rows
   (`first_error_at` = {3: 2, 8: 3, 9: 2, 10: 5, None: 138}), so the model's survival length is
   right-censored by the schedule and a proper comparison needs the schedule extended past 10
   questions — which *is* an (a)-class change, digit-span-style, and the one place option (a) has
   a real argument on this task.

Option (a) in the form "make the model stop at its first error" is measurably nearly irrelevant to
the score (+0.0033 humanlikeness) and would cost a re-baseline for nothing.

### M3 — axis A4's human reference is the stopping rule

Re-verified, and the retraction in `meta_harness/logs/a4_human_reference_invalid.md` stands.
Because every human's single error is also their last and highest-`relationCount` trial,
"error trials" and "last answered trials" are the *same 152 trials*. Substituting position for
error reproduces the ratio exactly: `mean(rc | error) = 6.1842 (n=152)`,
`mean(rc | correct) = 4.4596 (n=607)`, ratio **1.3867**; the positional null gives
**1.3867**. The human A4 value contains zero information about load-dependence of human errors.

Supporting measurement from this pass: human questions carry `relationCount` mean **4.81**
(max 10) while the model's `relation_count` has mean **8.00** (max 10) — because the model
reaches question 10 and the human stops around question 5. The model's questions are
systematically higher-load, which is a second reason the two A4 values are not comparable.

**Severity: INVALIDATING; already retracted.** **Fix: (c).** Keep A4 as a one-sided guard on the
model's own error distribution (`score_candidate.py:383`); strike the human column.

### M19 — 154 human records, 54 people

`participant_id` multiplicity in `runs/human/working-memory-variable-mapping/`: 54 distinct ids
over 154 files — 42 people contributed 3 records, 8 contributed 2, 1 contributed 4, 1 contributed
6, 2 contributed 1. `src/score.py` treats each *record* as a participant, so the human n of 152
is ~2.8× the number of people and carries within-person correlation. The model side is structured
the same way (`n_runs_per_participant: 3`, 150 rows from 50 stimulus sets), so the *means* are
comparable, but neither side's n is a count of independent units. Every other human task
directory is one record per person (checked all 11 directories; only narrative-qa, craft, map,
story-recall and digit-span have a single duplicated id each).

**Severity: BIASING of the n, not of the mean. Fix: (c) document.**

---

## 3. Digit span (forward and reverse)

### M4 — the `best_span` estimator is run at different trials-per-span on the two sides

This is the most consequential unlisted finding after M1.

**What the human did.** Adaptive staircase, inferred from all 52 forward / 49 reverse records with
trials: **2 sequences per length**, ascending from length 2 in steps of 1, terminate as soon as
both sequences at one length fail. Verified: the outcome pair at each participant's highest
administered length is `(False, False)` for **52 of 52** forward and **49 of 49** reverse; the
sequences-per-length signature is `(2, 2, 2, …)` for every participant. Trials per participant:
forward 8–32 (mode 14), reverse 4–20 (mode 10 or 12). Human `best_span` mean **6.885** digits
forward (score 0.3442 = 6.885/20), **5.898** reverse (0.2949).

**What the model did.** All 19 lengths 2–20, 10 `sequence_index` values each = 190 rows, no
termination. `src/score.py:_digit_span_participant_scores` then splits on `sequence_index` (one
participant, many sequences), producing **10 "participants" each with exactly 1 trial per span**,
and applies the same "highest span with ≥1 correct, stop at the first all-failed span" estimator.

**The estimator is not invariant to trials per span.** Measured on qwenA's own rows, forward:

| grouping of the same 190 model rows | trials per span | n units | `best_span` mean, digits | /20 |
|---|---|---|---|---|
| `src/score.py` unit (1 sequence = 1 participant) | 1 | 10 | **8.70** (sd 3.02) | 0.4350 |
| `protocol_match` pairing (adjacent sequences) | 2 | **5** | **18.40** — ceiling-censored, see below | 0.9200 |
| all 10 sequences pooled | 10 | 1 | **20** (the schedule maximum) | 1.0000 |
| hermes, `protocol_match` pairing (better powered) | 2 | **50** | **10.08** | 0.5040 |
| — human, for reference | 2 | 52 | **6.885** | 0.3442 |

**Caveat on the 18.40.** It rests on **5** pseudo-participants (qwenA carries only 10
`sequence_index` values, so pairing yields 5) on a schedule whose maximum is 20 digits, so it is
censored at the ceiling — the same confound this audit criticises in the A1 section. Hermes, which
carries 50 pseudo-participants, gives the better-powered matched figure: `best_span` **10.08**
digits against the human **6.885**. The INVALIDATING call does not depend on the 18.40: 8.70 vs
18.40 vs 20 digits from *the same 190 rows* is the argument, and hermes's 10.08 vs `src/score.py`'s
own 0.2925 × 20 = 5.85 digits shows the same non-invariance at n=50.

Reverse span: 1 trial/span → 6.30 digits (0.3150); 2 trials/span → 8.00 (0.4000); 10 pooled →
19; human 5.898 (0.2949).

**What the difference does.** The reported digit-span humanlikeness values —
forward 0.9012 (qwenA) / 0.9479 (hermes), reverse 0.9666 / 0.9498 — are produced by giving the
model **half** the chances per span that the human got. At the human's own 2-trial/span protocol
the same model rows put `best_span` at 18.40 digits against the human's 6.885, i.e. the model is
nowhere near human on forward span and the current figure hides it. `HANDOFF.md` records
`digit_span_reverse` as exactly +0.0000 humanlikeness delta for all five candidates and
`analyze_live_dimensions.py` records both digit-span tasks as contributing nothing; the reason is
that an estimator this unstable, run at a non-matched trials-per-span, is not measuring the
model.

**Severity: INVALIDATING.**

**Proposed fix: (b), analysis-side**, and it requires a re-baseline of the digit-span
humanlikeness column even though no model is re-run. Use `protocol_match._administer` — 2 trials
per span, stop on double failure — on both sides, and accept the smaller model n (5 pseudo-
participants per 190-row run at `sequences_per_span: 10`). If a usable n is wanted, the right
change is (a): raise `sequences_per_span` so the pairing yields ~50 pseudo-participants, which is
a schedule change rather than a scoring change and costs a digit-span re-run only.

### M5 — the human staircase terminates; the model's schedule does not

Subsumed by M4 once the estimator is matched: `protocol_match._administer` applies the human stop
rule to model trials and discards everything above the first double failure, and
`supra_span_hit` is then 0 on both sides *by construction*, which is why it cannot be an axis.
The earlier "supra-span hit rate 0.574 vs 0.000" was a schedule artifact, as that module's
docstring already says. **Fix: (b), already implemented in `protocol_match.py`; `src/score.py`
does not use it.**

### A1 (`sub_span_leak`): the null-model test

**The human value, re-measured.** `protocol_match.human_participants()` over the 52 forward
records: A1 mean **0.0866**, sd 0.0756, median 0.0833, p10 0.0000, p90 0.1987; **34.6% of
participants have A1 exactly 0**; mean administered trials 13.77; pooled trial-weighted rate
0.0915 (56 failures of 612 sub-ceiling trials). (The cached 0.087 is confirmed.)

**The null.** A participant whose success probability declines smoothly with span,
p(correct | span) = logistic((θ − span)/s), with θ ~ N(μ, σ) across participants and *no*
psychological structure beyond that, run through the human rule exactly (2 trials per span,
ascend from 2, stop on double failure). For each within-participant slope *s*, μ and σ were fitted
so the simulated `best_span` **mean and sd** match the human ones (6.885 and 2.064 digits);
4000 simulated participants per cell.

```
s     mu     sigma   best_mean  best_sd   A1_mean   A1_sd    A1 p10..p90     rmse(spans)
0.25  7.10   2.05    6.900      2.052     0.0451    0.0624   [0.000, 0.125]  0.020
0.40  6.95   2.05    6.874      2.071     0.0699    0.0790   [0.000, 0.167]  0.013
0.60  6.90   2.00    6.901      2.077     0.0955    0.0931   [0.000, 0.200]  0.020
0.80  6.85   1.90    6.890      2.059     0.1152    0.1005   [0.000, 0.250]  0.007
1.00  6.85   1.80    6.894      2.060     0.1329    0.1053   [0.000, 0.250]  0.010
1.25  6.85   1.70    6.877      2.071     0.1513    0.1097   [0.000, 0.300]  0.010
1.50  6.95   1.50    6.912      2.063     0.1630    0.1096   [0.000, 0.300]  0.027
2.00  7.10   1.10    6.882      2.072     0.1829    0.1105   [0.000, 0.333]  0.009
HUMAN observed                            0.0866    0.0756   [0.000, 0.199]  --
```

**Result: A1's human value of 0.0866 is not distinguishable from this null.** Every row above
reproduces the human `best_span` mean *and* sd to within 0.03 digits — the calibration target is
satisfied throughout — while A1 sweeps **0.045 to 0.183**. The human 0.0866 sits in the middle of
that range, matched at s ≈ 0.55. A1 is, to the precision available, a reparameterization of the
within-participant slope of the success curve, and `best_span`'s mean and sd do not identify that
slope. The user's objection is correct: failing a short sequence after passing a longer one is
ordinary variability in a declining curve, and A1 measures how steep that curve is, not anything
specifically human. The per-participant spread makes this worse, not better: at ~13.8 administered
trials, a third of human participants have A1 exactly 0. Taking mean and sd jointly does narrow
the fit — the human pair (0.0866, 0.0756) is consistent with s ≈ 0.4–0.6 and not with s = 2.0
(null sd 0.1105) — but that is precisely the point: **A1 is a read-out of the within-participant
slope of the success curve**, which is a property of the declining curve, not a human signature,
and which `best_span` mean and sd cannot pin down.

**Independent confirmation from the model side.** qwenA's A1 is **0.1360** and qwenB's **0.1298**
while their `best_span` is **18.40** digits against the human 6.885 — an 11.5-digit ceiling error
that A1 reads as a 0.05 discrepancy. hermes: A1 0.1719, `best_span` 10.08. And the confound
`score_candidate.py:245-248` already concedes: `full_context` reached `best_span` 20.0 on a
19-span schedule, never terminated, had no sub-ceiling failures to leak, and so scored A1 0.032 —
*closer* to human than the baseline's 0.130 — purely by being uninformative.

**Recommendation.** A1 was demoted to report-only on 2026-09-29
(`score_candidate.py:351`); **retire it** rather than leave it reported, because a small A1
distance reads as reassurance and cannot be one. Replace with, in order of demonstrated
sensitivity:

1. **`best_span`** (digits, human 6.885 forward / 5.898 reverse) — already named in
   `score_candidate.py:363` as the sensitive regression signal, and the thing that caught
   `random_decay` collapsing 18.4 → 2.0. It must be computed at 2 trials/span on both sides
   (M4), or it inherits the same estimator instability it is meant to police.
2. **The serial-position curve and error typology** in `protocol_match.py`
   (`serial_position`, `span_typology`) — these are shape measures over administered trials and
   do not depend on the slope of the success curve. Caveat already in that module: reverse span
   will usually print `insufficient` at a 30-error minimum.

---

## 4. n-back

### M6 — human denominator includes the lead-in trials

**What the human did.** 8-trial `training-1-back` practice block (logged for 49 of 53 records
with trials; 4 records have 42 trials and no practice block), then 3 scored blocks of 14 trials at
n = 1, 2, 3. `payload.trials` phases: `scored` 2058, `practice` 392, unlabelled 168 (the 4
records above). `src/score.py:165-170` takes every trial with `phase != "practice"`, i.e. **all
42**, including the first *n* of each block where no answer is possible.

**What the model did.** `bench/tasks/wm_nback.py` presents *n* buffer letters (`buffer_letters`,
length = n, 50 rows each at n = 1/2/3) that feed `buffer_map` and are **excluded** from
`acc_over_14`, plus 14 scored trials (`trial_letters` length 14 for all 150 rows).

**Measured effect on the human reference:**

| human n-back score | unit | mean | sd | n |
|---|---|---|---|---|
| all 42 non-practice trials (what `src/score.py` uses) | proportion-correct | **0.8657** | 0.0820 | 53 |
| scored trials only (36 per participant; matches the model) | proportion-correct | **0.8569** | 0.0853 | 53 |

Delta **−0.0088 proportion-correct**, moving the human reference *down*. The model is *below* the
human on this task (qwenA 0.6867), so correcting this narrows the gap very slightly rather than
widening it. (The a4 log describes it as widening an over-accuracy gap; on these baseline runs the
model under-performs, so the sign of the consequence is the other way.) Human accuracy on the 318 lead-in
trials themselves is **0.9182**, so they are easy trials inflating the human mean, not noise.

**Severity: BIASING, small. Fix: (b), analysis-side.**

### M8 — 112 impossible lead-in targets

Re-verified: of the 318 human lead-in trials (53 participants × (1+2+3)), **112 carry
`target: true`**, which is impossible because there is no letter *n* back. They span **48 of 53**
participants, **14** of them on the very first trial of the 1-back block. Post-lead-in, the
logged `target` flag agrees with a flag recomputed from the letter stream on **1908 of 1908**
trials, so the defect is confined to the lead-in. **Fix: (c)** — these trials leave the
denominator anyway once M6 is fixed.

### M7 — scoring granularity

`src/score.py` scores the model at one observation per (participant, n-level) and the human at one
pooled observation per participant. Measured on qwenA:

| unit | n | mean, proportion-correct | sd |
|---|---|---|---|
| model, per (participant, level) — what `src/score.py` uses | 150 | 0.6867 | **0.3439** |
| model, pooled per participant over 3 levels | 50 | 0.6867 | **0.0865** |
| human, pooled per participant | 53 | 0.8657 | 0.0820 |

The model's per-row sd is 4× the human's purely because it spans the level effect the human's
pooling averages away (qwenA per level: n=1 **0.9943**, n=2 **0.7971**, n=3 **0.2686**; human per
level: **0.9474 / 0.8652 / 0.7844**). Correcting both M6 and M7 together:

| | qwenA | hermes |
|---|---|---|
| humanlikeness as reported | 0.7636 | 0.8662 |
| humanlikeness, model pooled + human scored-only | **0.8308** | **0.8961** |

**Severity: BIASING, +0.065 / +0.030 humanlikeness. Fix: (b)** —
`meta_harness/nback_levels.py` already implements both the pooling and the better
split-by-level alternative; `src/score.py` uses neither.

### M9 — different numbers of scored trials per block

Human scored trials per block = 14 − n, i.e. 13 / 12 / 11 (36 total). Model scored trials per
block = **14** at every level (42 total). The model therefore answers proportionally more
high-load trials. **Severity: BIASING, small. Fix: (c) document.**

### M20 — practice and worked examples

Humans practised **8 trials of 1-back only** and nothing at n = 2 or 3. The model gets no practice
trials but its task description embeds a 5-letter worked example *with the correct responses* at
**every** level (`bench/tasks/nback.py:44-75`, `TASK_DESC_BY_N` / `HUMAN_PROMPT_BY_N`). So the
human's practice advantage is concentrated at n=1 (where both sides are near ceiling: 0.947 vs
0.994) and the model's example advantage is uniform. The observed level profile — the model beats
the human at n=1 and collapses at n=3 — is not explained by either. **Severity: COSMETIC.
Fix: (c).**

### M16 — timing, and M21 — the non-response asymmetry

`bench/tasks/nback.py:33-36` defines `STIMULUS_MS = 500`, `ISI_MS = 2000` as "Match the HTML
constants". They are inert for the model, which is turn-based and untimed. But the human data does
**not** support reading 2000 ms as a response deadline: over 2618 human trials with `rtMs`, the
median is **1393 ms** and **23.6%** exceed 2000 ms, 15.9% exceed 2500 ms, 4.3% exceed 5000 ms,
max **32859 ms**. So the human protocol accepted late responses. **M16's severity is COSMETIC for
the scores** — no human trial is scored as missed because of the clock — and load-bearing only for
interpretation.

**M21 is the separate, larger item.** The `response` field is `target` or `nontarget` on every one
of 2618 human trials — **there are no missing human responses at all**, so the human non-response
rate is 0% by construction. The model's `acc_over_14` counts an unanswered trial as **wrong**. Measured on qwenA: `answered`
is 14 of 14 in only **91 of 150** rows; the rest range down to 2 (distribution: 2:9, 3:8, 4:16,
6:4, 7:5, 8:2, 9:1, 10:3, 12:8, 13:3, 14:91). `acc_over_14` mean 0.6867 against
`acc_over_answered` mean 0.8440 over rows with any answer. **So `acc_over_14`'s non-response
penalty has no human counterpart at all — a fourth n-back asymmetry, distinct from M6, M8 and
M9.** It interacts directly with the
"arms answer 5.2–6.8 of 14" family in `HANDOFF.md` — those numbers are a harness-engagement
failure being scored as a memory failure. **Severity: BIASING, large (0.157 proportion-correct on
qwenA). Fix: (b)** — report `acc_over_answered` beside `acc_over_14` and treat the gap as a
harness-health metric, not a score; do not silently switch to `acc_over_answered`, which would
reward a model for declining hard trials.

Human wall-clock, for the record (`duration_ms`, seconds, median / p10 / p90):
n-back 251.9 / 210.4 / 317.2 (≈5.0 s per trial including instructions);
word_recognition 51.6 / 24.7 / 103.7 (≈1.5 s per trial — real speed pressure);
variable_mapping 42.8 / 20.2 / 117.7; digit span 146.8; reverse 173.0;
story recall 594.2; narrative-qa 291.7; factual-qa 320.2; craft 308.9; map 358.8.

---

## 5. narrative_qa

### M10 — two human question banks, one model bank

**What the human did.** Each participant read one ~600-word story and answered 10 four-option
questions. Matching all 520 human question texts against the two banks on disk:

```
from data/narrative_QA.json      : 471 questions
from data/narrative_QA_easy.json :  49 questions
neither                          :   0
per-record: 47 records all-hard, 4 records all-easy, 1 record mixed, 2 records empty payload
bank overlap (identical question text in both files): 1
```

Two human titles — *The Unexpected Journey of the Broken Compass* and *The Unexpected Journey of
a Lost Letter* — occur only in the easy bank, independently confirming the split. The easy bank's
questions are tagged `"difficulty": "easy"` against the hard bank's `"medium"`.

**What the model did.** `search_set.yaml` sets `data_json_path: data/narrative_QA.json` only.
Model rows use 10 stories S01–S10 from that file, 4 options each, 500 questions, and the 100
shared question texts carry **identical answer letters** on both sides (100/100), so option
ordering matches.

**What the difference does.** 4.5 of 52 scored human participants (≈9%) took an easier test. Human
mean 0.7962, model qwenA 0.7620, reported humanlikeness 0.9378. The effect is a small upward bias
on the human mean; with 4–5 records it cannot be estimated precisely.

**Severity: BIASING, small. Proposed fix: (c) document**, or (b) drop the 5 affected human
records from the narrative_qa reference and re-baseline that one column. Option (a) is wrong: the
model should not be given the easy bank to match a data-collection accident.

Also noted: the model's story assignment is uneven (S10 drawn 12 times of 50, S07 once), as is
the human's (15 / 10 / 9 / 8 / 8 / 1 / 1 by title). Both are random draws from the same ten
stories; no correction proposed.

---

## 6. craft_task

### M11 — 5 human records used the retired stimulus file

`data/craft_task.json` is version 3.0 (`item_counts` easy 5 / medium 6 / hard 7, task ids
C2001–C2003); `data/craft_task_old.json` is version 2.0 (easy 4 / medium 5 / hard 6, ids
C1001–C1003). Each has 11 distinct question prompts, and 4 prompts are exclusive to each file.

The human records contain all four old-only prompts — *Which pair produces D?* (10 occurrences),
*Which item can be crafted using C?* (5), *Which item is a base material?* (5), *Which item is
not produced by any rule?* (5) — and **5 of 54** human records contain at least one of them. The
other 49 use only v3.0 prompts. The model saw only v3.0 (`task_id` C2001/C2002/C2003, 50 rows
each, 11 distinct prompts, all 11 present in the human set).

**Severity: BIASING, small — same shape as M10. Fix: (c) document**, or (b) drop those 5 human
records.

### M12 — scoring granularity (craft_task and map_task)

**What the human did.** 3 trials of 5 binary-choice questions = 15 questions; `src/score.py:197`
uses `summary["totalCorrect"] / summary["totalQuestions"]`, i.e. one observation per 15 questions.
Verified: `totalQuestions == 15` and `trialsCompleted == 3` for all 54 craft / 56 map records,
5 questions per trial throughout.

**What the model did.** One jsonl row per 5-question trial (150 rows = 50 `repeat_index` × 3
task ids), and `src/score.py:265` takes `metrics["accuracy"]` per row — one observation per **5**
questions.

| | unit | craft human | craft qwenA | craft hermes | map human | map hermes |
|---|---|---|---|---|---|---|
| per-participant (15 q) | proportion-correct | 0.8395, sd **0.1191** | 0.9627, sd **0.0334** | 0.8667, sd **0.0000** | 0.7321, sd **0.1600** | 0.7480, sd **0.0388** |
| per-trial (5 q) — `src/score.py`'s model unit | proportion-correct | 0.8395, sd 0.1753 | 0.9627, sd 0.0782 | 0.8667, sd 0.0946 | 0.7321, sd 0.2688 | 0.7480, sd 0.2311 |
| humanlikeness, as reported (per-row model) | dimensionless | — | 0.8565 | 0.9340 | — | 0.8918 |
| humanlikeness, model pooled to 15 q | dimensionless | — | **0.8690** | **0.9105** | — | **0.9081** |

A 5-question binomial has strictly more variance than a 15-question one at the same p, so the
per-row unit both coarsens the model's support (craft model takes only the two values
{0.8, 1.0}) and inflates its spread. The humanlikeness effect is ±0.02 and its **sign is not
constant** across substrates, which is exactly what makes it unsafe to leave in.

**Severity: BIASING, small. Fix: (b), analysis-side** — pool the model's 3 rows per
`repeat_index`, which `src/score.py` can do from fields already present.

Option counts and answer-key space match: craft 2 choices labelled `Choice 1` / `Choice 2` on both
sides (human key values seen: Choice 1 × 418, Choice 2 × 392); map 2 options labelled A / B on
both sides (human key: B × 504, A × 336).

---

## 7. semantic_story_recall

### M17 — unequal story mix

Human `summary.storyName`: Eyespy 18, Pieman 15, Baseball 13, Oregon Trail 7, missing 3. Model:
exactly 50 rows per story, all four stories (`repeats_per_stimulus: 50`). The four stories differ
substantially in length (`n_sentences` 61 / 115 / 144 / 162), so recall difficulty is not
exchangeable across them and the two sides average over different mixtures. **Severity: BIASING.
Fix: (b)** — reweight the model rows to the human story mix, or report four per-story comparisons.

### M18 — the two `embeddingSimilarity` numbers have different provenance

The model side is computed in `bench/tasks/semantic_story_recall.py:214-242`: cosine of
mean-pooled, normalized **all-MiniLM-L6-v2** embeddings of the story and the recall, each
truncated to 200 words, clipped to [0, 1]. Its docstring asserts this is "same as
semantic-memory-story-recall.html". The human side is read straight from
`summary.embeddingSimilarity`, produced by the web app, which is **not in this repository**. I
could not verify the human side's embedding model, its reference text, or its truncation.
**This is the one item in this audit I could not verify.** Treat the story-recall humanlikeness
(qwenA 0.9411, hermes 0.8788) as conditional on that assertion. Related but separate: `src/score.py:190-195`
already records that the app's BLEU is ~0 across participants, so `embeddingSimilarity` was chosen
because it "lines up with the published Figure 2 visually" — a visual match, not a verified one.

**Severity: UNVERIFIED. Fix: (c) document**, and check against the web app source when available.

Recall lengths match well and need no correction: human 137.2 words mean (median 100, range
8–418), qwenA 136.7 (median 109, range 75–397), hermes 100.3 (median 84, range 69–183).

---

## 8. factual_qa and map_task (held out)

Model rows exist only under `meta_harness/runs/heldout/baseline/NousResearch_Hermes-4-70B`, so
**every held-out comparison is single-substrate.**

`factual_qa` is the cleanest task in the benchmark. 100 questions, 4 options, 10 per participant;
`totalQuestions == 10` for 53 of 55 human records and `metrics["total"] == 10` for all 50 model
rows; the 100 question texts are shared and the answer letters agree **100 of 100**, so option
order is identical and unshuffled. Document mix differs by random draw only (11 titles human, 10
model). Human 0.7075 (sd 0.1930) vs hermes 0.8540 (sd 0.1388), humanlikeness 0.8392. **No protocol
mismatch found beyond M12's granularity issue, which does not apply (both sides are 10-question
units), M14 and M15.**

`map_task` matches on stimuli (15/15 prompts, 2 options) and differs only by M12 (granularity) and
the global items.

---

## 9. M14 — what "a participant" means on each side

Treating this as a first-class finding, as instructed.

**The claim to test.** `meta_harness/cluster/search_set.yaml` sets `temperature: 0.0` for every
task, and the 50 "participants" per task are 50 seeded *stimulus sets* drawn by one
`stimuli_seed` and run by one model. If the model were deterministic, its between-participant
variance would be pure item-difficulty variance, while the human's is between-person variance
(ability, strategy, attention, fatigue).

**What the data shows — and the yaml's version of the claim is not literally true.** Joining
qwenA and qwenB row-by-row on `id` (both are baseline runs of the same config at temperature 0.0):

| task | rows joined | identical per-row score | identical generated text |
|---|---|---|---|
| digit_span_forward | 190 | 189 (0.995) | 188 (0.989) |
| digit_span_reverse | 190 | 187 (0.984) | 179 (0.942) |
| variable_mapping | 150 | 149 (0.993) | — |
| craft_task | 150 | 149 (0.993) | 149 (0.993) |
| narrative_qa | 50 | 37 (0.740) | 35 (0.700) |
| word_recognition | 50 | 27 (0.540) | 26 (0.520) |
| nback | 150 | 78 (0.520) | — |
| semantic_story_recall | 200 | 11 (0.055) | 9 (0.045) |

So "the same deterministic model" holds approximately for the short-generation tasks and fails
badly for the long-generation ones (story recall reproduces 4.5% of its own texts). Task-level
means move by up to **0.036 score units** (word_recognition) and humanlikeness by up to
**0.038** (word_recognition, n-back) between two runs of the identical configuration. The
practical consequence is separate from the participant question: **a single run's humanlikeness
carries ~0.04 of run-to-run noise on the noisiest tasks**, consistent with the
`run_to_run_floor.json` figures in `HANDOFF.md` but larger than the 8-task mean of 0.0001 quoted
there.

**The structural point, measured.** Whatever residual generation noise exists, the model's
per-participant spread is *not* built from the same source as the human's. Per-participant score
sd (score units, both sides on `src/score.py`'s current definitions):

| task | human sd | qwenA sd | qwenB sd |
|---|---|---|---|
| digit_span_forward | 0.1032 | 0.1510 | 0.1907 |
| digit_span_reverse | 0.0745 | 0.1132 | 0.1132 |
| nback | 0.0820 | 0.3439 | 0.3237 |
| word_recognition | 0.2424 | 0.3478 | 0.3808 |
| variable_mapping | 0.2352 | 0.0690 | 0.0490 |
| factual_qa | 0.1930 | 0.1388 (hermes) | — |
| narrative_qa | 0.1620 | 0.1677 | 0.1525 |
| semantic_story_recall | 0.1294 | 0.0765 | 0.0838 |
| map_task | 0.1600 | 0.2311 (hermes) | — |
| craft_task | 0.1191 | 0.0782 | 0.0771 |

The clearest single demonstration: **hermes's craft_task, pooled to the human's 15-question
participant unit, has sd exactly 0.0000** — all 50 "participants" scored 13 of 15. That is a
distribution with zero between-participant variance, and the only thing generating any spread at
all in the per-row figure (sd 0.0946) is which of the three task ids a row happens to be. There is
no ability, strategy, attention or fatigue dimension on the model side to vary.

**What this means for W₁.** W₁ between the two distributions is being read as "how human-like is
this model", but it is the distance between a *between-person* distribution and a
*between-item* distribution. Two consequences that follow directly:

1. **A model can improve its humanlikeness by matching item-difficulty spread to human ability
   spread, which is not a psychological achievement.** Where the model's spread is too small
   (craft 0.078 vs 0.119; story 0.077 vs 0.129; variable_mapping 0.069 vs 0.235) the only way to
   widen it without changing the mean is to become more item-sensitive. Where it is too large
   (n-back 0.344 vs 0.082) the excess was a scoring-granularity artifact (M7), not behaviour.
2. **A perfect humanlikeness of 1.0 is not attainable in principle and not desirable in
   interpretation.** The right reading of the metric is "does the model's score distribution
   occupy the same interval as the human one", which is a much weaker claim than the name
   suggests.

**Severity: INVALIDATING for the interpretation of W₁ as a psychological measure**, not for its
arithmetic. **Proposed fix: (c), document.** Option (a) — giving the model a population of
simulated abilities — is a research programme, not a harness change, and option (b) cannot create
variance that was never sampled. Two cheap things are worth doing instead: report the two sds
beside every humanlikeness value so the reader can see which side is wider, and report the
run-to-run spread from the table above as the metric's noise floor.

---

## 10. M15 — instructions and framing

**What the human got.** The task instructions only (the same text that `bench` carries as each
task's `HUMAN_PROMPT` / `TASK_DESC`).

**What the model gets** (`bench/tasks/wm_prompt_parts.py:24-47`, condition C2, the only condition
in the compactor runs — `condition_name` is `"Human cognitive limits framing"` on every row of
every task file):

> You are simulating a human participant in a psychology experiment on working memory.
> You have a key-value memory store with at most 4 slots, reflecting the ~4-chunk limit of human
> short-term memory (Cowan, 2001). … When the task asks for verbatim retrieval of a sequence, a
> human will form meaningful chunks of 1–3 items, starting from the beginning. NEVER pack a long
> run of items into one slot. Once your slots are filled, accept that the rest will be lost. …
> behave as a real human would: imperfect and sensitive to what seems important.

**What the difference does.** Three things the human side has no counterpart for: (i) the
capacity limit is *stated as a number* rather than being a property of the participant;
(ii) the chunking strategy (1–3 items, from the beginning) is *prescribed*, which pre-empts
exactly the serial-position and chunking behaviour that `protocol_match.serial_position` is meant
to measure; (iii) "be imperfect" instructs the model to lower its own score, so the model's
distance from human accuracy is partly a compliance measurement. A humanlikeness gain under this
prompt cannot be attributed to a memory architecture.

**Severity: BIASING, and not quantifiable from the runs on disk** — there is no C1/C3 arm in the
compactor runs to contrast against. **Fix: (c) document**, and note that any future claim about
the memory module needs an arm with the Cowan sentence and the chunking prescription removed.

---

## Ranked by severity

**Invalidating (the two sides do not compute the same quantity):**

1. **M1** word_recognition denominator — humanlikeness off by +0.371 (qwenA) / +0.109 (hermes).
2. **M2** variable_mapping denominator — humanlikeness off by +0.410 (qwenA) / +0.424 (hermes).
3. **M4** digit-span `best_span` estimator run at 1 trial/span (model) vs 2 (human) — the same
   model rows give `best_span` 8.70 or 18.40 or 20 digits depending on the grouping.
4. **M13** word_recognition prints the test stream to the model in full, twice.
5. **M14** "participant" means a person on one side and a stimulus set on the other.
6. **M3** A4's human reference is the stopping rule (already retracted).

**Biasing (same quantity, measured distortion):** **M21** n-back non-response penalty with no human
counterpart (0.157 proportion-correct on qwenA); M7 n-back granularity (0.065 humanlikeness);
M6 n-back lead-in denominator (0.0088 score units); M12 craft/map granularity (±0.02
humanlikeness, sign not constant); M15 the Cowan framing; M17 story mix; M10 narrative_qa two
banks; M11 craft two banks; M5, M8, M9, M19.

**Cosmetic:** M16 timing (no human trial is scored as missed because of the clock); M20 practice
asymmetry.

**Unverified:** M18 — the human `embeddingSimilarity` provenance.

## Fixes: harness (a) vs analysis-side (b)

**Change `bench/` — accepts a re-baseline:** only **M13** (word_recognition one word per turn,
answer before store, no re-print). Optionally **M4**'s companion (raise
`sequences_per_span` so the matched 2-trial/span pairing yields a usable n) — a schedule change,
digit span only.

**Correct analysis-side:** M1, M2, M4, M5, M6, M7, M12, M17, M21 — all of them computable from
fields already on disk, no model re-run. M1, M2, M4, M7 and M12 change published humanlikeness
numbers and require a re-baseline of the *table*, not of the runs.

**Leave and document:** M3, M8, M9, M10, M11, M14, M15, M16, M18, M19, M20.

**Caveat carried forward on M1 and M2:** the analysis-side fix removes the arithmetic error but
does **not** produce a like-for-like accuracy comparison, because on both tasks the human's
proportion-correct is algebraically 1 − k/n. The comparable quantity is survival length, and on
variable_mapping the model's survival length is right-censored by the 10-question schedule, so a
complete fix there needs the schedule lengthened — an (a)-class change.

## What contradicts the framing this audit was given

1. **The n-back 500 ms / 2000 ms schedule was not a response deadline for humans.** 23.6% of
   human responses exceeded 2000 ms, the median was 1393 ms, the maximum 32859 ms, and no trial
   is missing a response. Human time pressure on n-back is real but far softer than stated, and
   the sharper asymmetry is the opposite one: the human side has a 0% non-response rate while the
   model's `acc_over_14` scores non-response as error on 59 of 150 rows.
2. **The model is not deterministic at temperature 0.0.** Row-level reproduction between two runs
   of the identical config ranges from 0.995 (digit span forward) to 0.055 (story recall).
   M14's structural point survives, but it has to be argued from the measured sds, not from the
   yaml.
3. **Neither word_recognition nor variable_mapping has a human accuracy to compare against,
   fixed denominator or not.** Because both human tasks stop on a fixed error count, the human
   proportion-correct is algebraically 1 − k/n: word_recognition
   `trialsCompleted − correctResponses == 3` for 53 of 53 records, and variable_mapping's
   mean(1 − 1/n) = 0.7409 reproduces the measured 0.7409. Two of the eight search tasks therefore
   score humans on persistence, not accuracy. This is the A4 defect class applied to the score
   itself, and it means the +0.371 / +0.410 humanlikeness corrections are the *arithmetic* value
   of a still-incommensurable comparison. It was not in the brief.
4. **Matching the stopping rule is not the lever on either task.** Emulating the 3-strike rule on
   the model's own word_recognition trials stops it after a median of 100 trials (it errs too
   rarely to accumulate 3 strikes); applying the human stop rule to variable_mapping adds +0.0033
   humanlikeness on top of the denominator fix. The stopping rule is the *cause* of the problem
   and not itself the thing to change in `bench/`.
5. **The human digit-span best span is 6.885 digits forward, not "≈6.88" as a loose figure** —
   and the reverse figure, 5.898, was not in the brief. Both need M4's estimator matching before
   they can be compared to anything.
6. **`protocol_match.variable_mapping_scores` does not fix variable_mapping.** It equalizes the
   formula and leaves the denominator wrong on both sides.
7. **The craft_task two-bank problem was not in the brief** and is the same defect as
   narrative_qa's.
