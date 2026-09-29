# Survival-length scoring for `word_recognition` and `variable_mapping`, 2026-09-29

User decision: **"Use survival length on both sides."** Landed in `src/score.py` as
`a3d0bc3`. This file records why, what changed numerically, and the two caveats that must
travel with the new numbers.

## Units, stated first

**Humanlikeness** = 1 − Wasserstein-1 between the model's and the humans' per-participant
score distributions. Range 0–1, higher is more human-like. A **delta** is one humanlikeness
minus another and is the only quantity here that can be negative.

The *per-participant score* is task proportion-correct on six of the eight tasks. On the two
tasks in this file it is now **survival length**: the number of items the participant was
presented before the protocol stopped them, divided by a fixed scale — **100 words** for
`word_recognition`, **16 questions** for `variable_mapping`. Survival past the scale clips
to 1.0.

> **The `variable_mapping` scale was 10 for the first hours of 2026-09-29 and is now pinned at
> 16** (`56eedda`). The two are not comparable, and the denominator-10 figures are already
> quoted in earlier commit messages, so every `variable_mapping` number below is given at both.
> See *The denominator is pinned, and why* below.

**Worked example.** A human in `variable_mapping` answered 4 questions correctly and got the
5th wrong; the task ended there. Their record holds 5 questions. Survival length = 5,
score = 5/16 = **0.3125**. Under the old scoring they had 4 correct out of a fixed denominator
of 10, score = **0.40**. A model run that answered questions 1–2 correctly and erred on
question 3 has survival length 3, score = 3/16 = **0.1875**; under the old scoring it banked
`relation_count` of question 2 and then answered the remaining 7 questions anyway, most often
scoring **1.00**.

## Why neither task has a human accuracy

Both human protocols end a session after a fixed number of mistakes. The error count is
therefore pinned by the protocol, and the only thing that varies between participants is
*when* the last mistake landed.

| task | evidence | so the old human score was |
|---|---|---|
| `word_recognition` | `trialsCompleted − correctResponses == 3` for **53 of 53** records | identically `(n_survived − 3)/100` |
| `variable_mapping` | exactly one error in **152 of 152** records, always the participant's LAST question | identically `(n_survived − 1)/10` |

Human `word_recognition` survival: mean **34.49** words, median 32, min 4, max 102.
Human `variable_mapping` survival: mean **4.99** questions, median 5, min 2, max 16.

An "accuracy" computed from either is a deterministic function of survival length, so the
project was already measuring survival length on the human side — under a name that invited
comparison against a model accuracy, which is what went wrong.

## What changed, measured on both 3-repeat baselines

Pre-fix = the three pre-instrument-fix baseline runs `iter0/baseline`, `iter8repA/baseline`,
`iter8repB/baseline`. Post-fix = `iter10postfix/baseline{,_rep2,_rep3}`. All numbers are
humanlikeness.

| task | scoring | pre-fix | post-fix | delta |
|---|---|---|---|---|
| `word_recognition` | old (correct count / 100) | 0.5075 | 0.4918 | −0.0158 |
| `word_recognition` | **survival / 100 — CURRENT** | **0.5232** | **0.5073** | **−0.0159** |
| `variable_mapping` | old | 0.3539 | 0.6801 | +0.3262 |
| `variable_mapping` | survival / 10 — superseded | 0.4621 | 0.9564 | +0.4943 |
| `variable_mapping` | **survival / 16 — CURRENT** | **0.6809** | **0.9625** | **+0.2816** |
| mean over 8 | old | 0.7824 | 0.8137 | +0.0313 |
| mean over 8 | with survival, vm at /10 | 0.7979 | 0.8502 | +0.0523 |
| mean over 8 | **with survival, vm at /16 — CURRENT** | **0.8253** | **0.8509** | **+0.0256** |

All of these are measured on runs where `bench` still asked **10** `variable_mapping`
questions. `N_QUESTIONS` is now 20, so the model's survival distribution itself will change at
the next re-baseline — the 5 of 150 rows piled on the old ceiling can now resolve anywhere up
to 20, clipped at 16. **These `variable_mapping` figures are transitional.**

`word_recognition` barely moves, and that is expected: survival is the old score plus a
near-constant 3 words, so the two distributions differ by an almost pure translation that
Wasserstein-1 mostly absorbs. The change there is conceptual, not numeric — it is what makes
the stage-2 re-baseline interpretable.

`variable_mapping` moves a lot, and the reason is in `protocol_match.variable_mapping_scores`:
the old model formula scored `relation_count` of the last consecutively correct question, and
`relation_count` saturates at 10 by question 5. So once a run answered five questions
correctly it had banked the maximum and **every later error was invisible**.

Measured on the post-fix baseline, the old score takes exactly four values, and they are a
lossy function of survival length:

| survival (first error at) | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| rows | 39 | 37 | 31 | 23 | 6 | 5 | 4 | 5 |
| old score | 0.4 | 0.6 | 0.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**43 of 150 rows collapse onto 1.0**, spanning first errors from question 6 to no error at
all. On the pre-instrument-fix released baseline the collapse was far worse — the
`variable_mapping_scores` docstring records `{1.0: 148, 0.4: 2}` there, with 10 of the 12 runs
that made an error still scoring 1.0.

## The finding this exposes, and what it is not

Under the matched statistic the model's first-error position is close to the humans':

```
survival   2   3   4   5   6   7   8   9  10  11  15  16
human     25  28  20  29  19   7  11   5   3   1   1   3      mean 4.99  median 5
model      -  39  37  31  23   6   5   4   5   -   -   -      mean 4.84  median 4
```

Verified independently of `metrics.first_error_at` by re-grading every question from
`parsed_answers` + `options` + `correct_city`: **150 of 150 rows agree**, 0 unparsed answers.

**What this is not.** Two bounded distributions on 2–10 with means 0.15 apart will score high
on 1 − W₁ almost regardless of mechanism, so 0.96 is not evidence that the model fails for
human reasons. It says the *rate* at which this model runs out of usable bindings resembles
the human rate. It says nothing about which binding it loses, and the project's pseudo-
participant problem is untouched: per the user's decision to continue with the current
procedure, the model's between-participant spread is still item difficulty while the humans'
is between-person ability.

It also means the instrument fix's variable_mapping gain is **smaller than the old formula
showed on one reading and larger on another** — +0.2816 at the pinned scale of 16 against
+0.3262 under the old formula and +0.4943 at a scale of 10. The scale-dependence is the point
of the next section, and it is why the scale is now fixed.

## The denominator is pinned, and why

`TASK_DENOM["variable_mapping"] = 16.0`, pinned, and it **must not track `N_QUESTIONS`.**

Wasserstein-1 scales linearly with the denominator. A denominator that followed the schedule
length would therefore make humanlikeness *rise* for a longer task at identical behaviour —
about +0.02 going 10 → 20 — which is a metric that improves when you lengthen the task.

The size of the effect is not hypothetical. Rescoring the same two baselines at the two scales:

| | pre-fix | post-fix | delta |
|---|---|---|---|
| scale 10 | 0.4621 | 0.9564 | +0.4943 |
| scale 16 | 0.6809 | 0.9625 | +0.2816 |

The pre-fix figure moves 0.22 and the post-fix figure 0.006. Rescaling shrinks a large
distance far more than a small one, so the *delta* nearly halves. Fixing the scale once, at a
value chosen from the human data rather than from the harness, is what stops this recurring.

**16 is the human maximum.** Over 152 human records the most questions anyone answered is 16;
no human record is censored, because every one ends in an error. Survival past 16 clips to 1.0.

## Why `bench` now asks 20 questions

Raising `variable_mapping.N_QUESTIONS` from 10 to 20 (`56eedda`) is a separate change from
pinning the scale, and it is about censoring rather than units.

| quantity | human app | bench before | bench now |
|---|---|---|---|
| questions offered | **≥16** (max answered 16, max turn 32) | 10 | **20** |
| turns per question | 2 | 2 | 2 |
| distinct relations | max exactly 10 | 10 | 10 |

`TARGET_RELATIONS` stays at 10 because the human `relationCount` maxes at exactly 10 across all
152 records — the human app also introduced 10 people and then only re-moved existing ones,
which is what bench's `turn <= cap` branch already does.

Cost: 2 LLM turns per question (one encode, one answer), so 150 runs × 20 questions × 2 =
**6000 turns, up from 3000**. Small against post-fix n-back (~5100) and stage-2
`word_recognition` (up to 10,000). The model is left answering all 20 rather than stopping at
its first error: under survival scoring the later questions do not affect the score, but they
are the trace the error-shape measures read.

## `word_recognition` needs no pinning

It is nearly symmetric already: the human list is 100 long and the model's is 100. One human
record reports `trialsCompleted: 102` and is clipped to 100.

## Consequences for the rest of the harness

- Every floor and noise figure for these two tasks is measured on the old quantity and must be
  re-measured: `run_to_run_floor.json` and `score_repeats.SAME_FAMILY_SD`.
- `evolution_summary.jsonl` entries written before `a3d0bc3` carry old-scoring humanlikeness
  for these two tasks. **Never compare across the boundary** without rescoring; `score_repeats`
  reads from run dirs, so rescoring is cheap and needs no model run.
- A4 was already retracted; nothing here revives it.
- `word_recognition`'s A2 (~~miss/FA ratio, human 6.094~~ — **AMENDED 2026-09-29 [USER]: A2's
  scalar is now miss − FA, a proportion, human **+0.2273**; the ratio is superseded and kept only
  as a legacy field) and `M4_word_recognition_lag` are
  unaffected by the survival-length change — they read `per_trial`, not the score.
- `bench`'s own `metrics.score` for `variable_mapping` is now **vestigial**. It is
  `relation_count` of the last consecutively correct question and saturates by question 5.
  Left in place so old rows stay readable; nothing new should be built on it.
- `interference.py`'s pooling note claimed "154 participants answer 10 questions each and make
  152 errors in total". Corrected: humans answer 2–16 questions (mean 4.99) because the task
  ends at their first error, so each record holds exactly one error, always its last question.
  The pooling conclusion stands for a stronger reason than the one recorded.
- `report_error_shape.py`'s "a human participant answers 10 questions" note was checked and is
  **not** affected — it describes `narrative_qa` / `factual_qa`, which really do have 10.
