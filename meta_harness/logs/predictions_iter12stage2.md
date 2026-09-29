# Predictions for job 18781213, registered before the run produced anything

Submitted 2026-09-29 from `/scratch/cl5625/mh-postfix` at commit **a14b59d**.
`CAND="baseline baseline baseline"`, `ITER=12stage2` → `runs/iter12stage2/baseline{,_rep2,_rep3}`.
Written and committed while the job was still `PD (Resources)`.

## Units

Humanlikeness = 1 − Wasserstein-1 between the model's and the humans' per-participant score
distributions; 0–1. The per-participant score is proportion-correct, except on
`word_recognition` and `variable_mapping` where it is **survival length** — items presented
before the stop rule fired — over a fixed scale (100 words, 16 questions). Accuracies, survival
lengths and error-shape distances are in their own units and are **not** humanlikeness.

## What changed since `iter11postfix`, and nothing else did

1. **`word_recognition`**: one word per turn, answer-then-store, `encode()` deleted, presentation
   stops at the third error (`dc14d0a`).
2. **`variable_mapping`**: `N_QUESTIONS` 10 → 20 (`56eedda`).
3. **Both digit spans**: `sequences_per_span` 10 → 40 (`4894c58`).

Scoring changes (survival length, `variable_mapping` scale pinned at 16) were already applied
when `iter11postfix` was scored, so the two runs are directly comparable.

## P1 — `word_recognition`: the score came from list visibility, not memory

**Hypothesis.** The model's near-perfect word_recognition came from the whole 100-word stream
being visible at judgement time. Removing that visibility should make its survival length fall
toward the human range and its humanlikeness rise.

**Reference.** Model survival 84.80 words (median 100, 38 of 50 never reaching 3 errors);
human survival 34.49 (median 32). humanlikeness 0.5364, 3-repeat spread 0.0199.

| | supports | rejects |
|---|---|---|
| model mean survival | **< 60 words** | **> 80 words** |
| humanlikeness | **> 0.60** | within 0.04 of 0.5364 |

A rejection would mean the presentation was not the cause and the gap comes from somewhere I
have not identified. Anything between the two bands is inconclusive and must be reported as such
rather than read as partial support.

## P2 — `word_recognition`: a lag effect should appear at all

**Hypothesis.** Judging from a 4-slot store, rather than from a visible list, makes accuracy
depend on how far back the word's first presentation was.

**Reference.** The model is currently flat: 0.9574 / 0.9952 / 0.9953 / 0.9966 / 1.0000 across
lag bins 1–2 / 3–5 / 6–10 / 11–20 / 21+, i.e. |max − min| = 0.043.

- **Supports:** |max − min| across the five bins **> 0.10**.
- **Rejects:** stays **< 0.05**, i.e. still no lag dependence.

**Direction is deliberately not predicted.** A store-based model should be *better* at short
lags (the word is more likely still held), which is the opposite of the human curve, where
accuracy *rises* with lag from 0.5059 to 0.9504. That human pattern is not explained by any
memory account I can state, and lag 21+ falls back to 0.8878 on only n=27, so I am not going to
predict a direction I cannot justify. The claim is that lag stops being irrelevant.

## P3 — A2 should move from the wrong side of human toward it

**A2** is the word_recognition miss/false-alarm ratio: humans **6.094**, i.e. humans miss old
words far more often than they falsely call a new word old. The model currently sits at
**0.3901** — the wrong side of 1.0 entirely, false-alarming more than it misses, which is what a
participant reading the answers off a list would do when it slips.

- **Supports:** ratio **> 1.0** (the model now misses more than it false-alarms).
- **Rejects:** ratio stays **< 0.6**.

`domain_spec.md` records A2 as "badly weakened" because its value came from the ~7 of 50
participants who consulted the store while 43 read the list. This is the prediction that says so.

## P4 — `variable_mapping` at 20 questions: censoring resolves, behaviour does not change

**Hypothesis.** Raising the schedule removes a ceiling artefact without changing where the model
actually fails.

**Reference.** With 10 questions: survival mean 4.84, median 4, and 5 of 150 rows censored at 10.
humanlikeness 0.9643, 3-repeat spread 0.0033. Human survival mean 4.99, median 5, max 16.

- **Supports:** survival median stays **4–5**; **≤ 3 of 150** rows reach 16 or beyond;
  humanlikeness within **0.02** of 0.9643.
- **Rejects:** many rows run past 16 — meaning the model is far more persistent than humans and
  the old 10-question ceiling was flattering it — or humanlikeness moves more than **0.05**.

Note the asymmetry this repairs: no human record is censored, and 5 of 152 humans ran to 11, 15
or 16 questions.

## P5 — digit span: the matched comparison becomes readable

**Hypothesis.** `sequences_per_span` sets the matched n directly, at `sequences_per_span / 2`.

- **Supports:** `M2_digit_span_forward_serial_position` reports **n_model = 20** (was 5), and
  `M2_digit_span_reverse_typology` is **no longer withheld** (needs ≥ 30 model errors; it had 20).
- **Rejects:** n_model ≠ 20.

**`best_span` is NOT comparable to any earlier run** — the estimator is not invariant to
trials-per-span (audit M4: the same 190 rows give 8.70 / 18.40 / 20 digits depending on
grouping). Do not report a `best_span` delta against `iter11postfix`.

Digit-span *humanlikeness* should stay inside noise (both spreads were 0.0000 over 3 repeats),
though a 4× larger row count may itself shift the pseudo-participant grouping `src/score.py`
uses, so a move here is not automatically a defect — it must be traced before being called one.

## P6 — precondition: the four untouched tasks must not move

No code touching `nback`, `narrative_qa`, `semantic_story_recall` or `craft_task` changed. Each
must stay inside **2× its measured 3-repeat spread**, doubled because three repeats pin a spread
only to about a third of itself:

| task | iter11postfix | spread | must stay within |
|---|---|---|---|
| nback | 0.9344 | 0.0034 | ±0.0068 |
| narrative_qa | 0.9444 | 0.0184 | ±0.0368 |
| semantic_story_recall | 0.9470 | 0.0115 | ±0.0230 |
| craft_task | 0.8679 | 0.0357 | ±0.0714 |

**If any of these fails, P1–P5 are void** and the first question is what leaked, exactly as the
stage-1 precondition was treated.

## P7 — cost, so a timeout is diagnosable rather than mysterious

`iter11postfix` took **1h06m** for three repeats on one H200. This run adds: word_recognition up
to 200 turns per participant instead of 2 (cut back by the third-error stop, so the true cost
depends on P1 — if P1 holds, the model stops early and this is cheap); variable_mapping 6000
turns instead of 3000; digit spans 3040 calls instead of 760.

Expect **2–4 hours**. `--time=08:00:00`. If it runs past ~5 hours the likely cause is P1 failing
in the specific way where the model neither errs nor stops, which is itself informative.
