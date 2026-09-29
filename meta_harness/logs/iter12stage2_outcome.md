# Stage-2 outcome — job 18781213: word_recognition +0.2914, and 8 of 9 predictions held

Three Qwen baseline repeats, `runs/iter12stage2/baseline{,_rep2,_rep3}`, run from
`/scratch/cl5625/mh-postfix` at commit **a14b59d**. **COMPLETED, exit 0, 1h32m** on one H200.
All three arms hold all 8 tasks at full row counts (6810 rows total).

Predictions were registered in `logs/predictions_iter12stage2.md` and the evaluator
(`meta_harness/check_iter12_predictions.py`) was written and smoke-tested **before the run
produced any output**, so the bands could not be tuned to the result.

## Units

Humanlikeness = 1 − Wasserstein-1 between the model's and the humans' per-participant score
distributions; 0–1, higher is more human-like. A **delta** is one humanlikeness minus another and
is the only quantity here that can be negative. The per-participant score is task
proportion-correct **except** on `word_recognition` and `variable_mapping`, where it is
**survival length** — items presented before the stop rule fired — over a fixed scale (100 words,
16 questions). Survival lengths, accuracies and error-shape distances are in their own units and
are **not** humanlikeness.

## What changed in this run, and nothing else did

1. `word_recognition`: one word per turn, answer-then-store, `encode()` deleted, presentation
   stops at the third error (`dc14d0a`).
2. `variable_mapping`: `N_QUESTIONS` 10 → 20 (`56eedda`).
3. Both digit spans: `sequences_per_span` 10 → 40 (`4894c58`).

## The table, 3 repeats each side

| task | iter11postfix | **iter12stage2** | delta | 3-repeat spread | comparable? |
|---|---|---|---|---|---|
| digit_span_forward | 0.9012 | 0.8712 | −0.0300 | **0.0000** | **NO** — estimator changed |
| digit_span_reverse | 0.9666 | 0.9568 | −0.0098 | **0.0000** | **NO** — estimator changed |
| nback | 0.9344 | 0.9340 | −0.0004 | 0.0061 | yes |
| **word_recognition** | 0.5364 | **0.8278** | **+0.2914** | 0.0122 | yes |
| variable_mapping | 0.9643 | 0.9662 | +0.0019 | 0.0012 | yes |
| narrative_qa | 0.9444 | 0.9522 | +0.0078 | 0.0140 | yes |
| semantic_story_recall | 0.9470 | 0.9510 | +0.0040 | 0.0077 | yes |
| craft_task | 0.8679 | 0.8451 | −0.0228 | 0.0342 | yes (delta inside spread) |
| **mean over 8** | 0.8828 | **0.9130** | +0.0302 | | mixes 2 incomparable columns |
| **mean over the 6 comparable tasks** | 0.8657 | **0.9127** | **+0.0470** | | **the honest figure** |

**Recorded as `gen4_stage2_baseline_iter12stage2` in `evolution_summary.jsonl`.**

The +0.0470 six-task gain is **almost entirely one task**: `word_recognition`'s +0.2914 alone
contributes +0.0486 to a six-task mean. Everything else is inside its own spread.

## The predictions, as registered

| | verdict | observed |
|---|---|---|
| **P1** word_recognition | **SUPPORTS** | survival 80.82 → **18.72** words (median 18); HL **0.8278** |
| **P2** lag effect exists | **SUPPORTS** | spread 0.043 → **0.4369** |
| **P3** A2 ratio > 1.0 | **INCONCL** | ratio **undefined** — zero false alarms; see below |
| **P4** variable_mapping | **SUPPORTS** | median 4, **0 of 450** rows censored at 16, HL 0.9662 (Δ +0.0019) |
| **P5** digit-span matched n | **SUPPORTS** | 5 → **20** participants; reverse errors 20 → **88** |
| **P6** nback | **SUPPORTS** | −0.0004 within ±0.0068 |
| **P6** narrative_qa | **SUPPORTS** | +0.0078 within ±0.0368 |
| **P6** semantic_story_recall | **SUPPORTS** | +0.0040 within ±0.0230 |
| **P6** craft_task | **SUPPORTS** | −0.0228 within ±0.0714 |

**8 SUPPORTS, 1 INCONCL, 0 REJECTS.** The precondition held on all four untouched tasks, so
P1–P5 stand.

## P1: the model went from far too good to somewhat too bad

| | pre-fix | **now** | human |
|---|---|---|---|
| survival length | 80.82 words | **18.72** (median 18) | **34.49** (median 32) |
| rows reaching 3 errors | 12 of 50 | **50 of 50** | 53 of 53 |
| humanlikeness | 0.5364 | **0.8278** | — |

It **overshot**: the model now survives about half as many words as humans, having previously
survived more than twice as many. The hypothesis registered in
`logs/instrument_fix_stage2_plan.md` — that the near-perfect scores came from list visibility
rather than memory — is confirmed, and the residual error is now in the opposite direction. With
`MAX_KEYS = 4` against 100 words, a model that under-survives is the expected shape.

## P3: A2 is over-satisfied and simultaneously broken as a statistic

Confusion matrix over 969 trials: **(new → old) = 0.** Zero false alarms in 487 new-word trials.
Miss rate 0.3752, false-alarm rate exactly 0.0000.

So miss/FA is **undefined**, not small. The registered band was "supports > 1.0", which an
undefined value cannot satisfy, so the verdict is INCONCL **and the band was not rewritten after
the fact**. In substance the axis moved from 0.3901 (too liberal, the wrong side of 1.0) straight
*past* the human 6.094 to absolutely conservative. Bounded form: miss − FA = **+0.375** against a
human **+0.227**.

**A2 now has no defined value on its own substrate.** The axis `domain_spec.md` calls "the axis"
needs a bounded replacement — miss − FA, or d′ — before it can guard anything again. That is a
consequence of fixing the task, not a reason to regret it.

## P2: a lag effect appeared, running opposite to the human curve

| lag bin | 1–2 | 3–5 | 6–10 | 11–20 |
|---|---|---|---|---|
| model **before** | 0.9574 | 0.9952 | 0.9953 | 0.9966 |
| **model now** | **0.7717** | 0.7657 | 0.5651 | **0.3348** |
| human | **0.5059** | 0.7975 | 0.9265 | **0.9504** |

Spread 0.043 → 0.4369. The model now **falls** with lag, which is what a 4-slot store should do —
a word seen two items ago is more likely still held. Humans **rise**. I deliberately declined to
predict the direction, and that was the right call: the model's curve is now mechanistically
sensible and **the human curve is the unexplained one**.

A confound I can name but not remove: both sides stop at 3 errors, so long-lag bins only exist
for participants who survived that long, and the human n per bin falls 53 / 51 / 44 / 40 / 27.
Part of the human rise is likely survivorship. The model's decline survives the same bias. The
measure's caveat text now records this (`1e030a9`).

## What the error shape says now

Full report in `logs/error_shape_iter12stage2.txt`.

- **P5's payoff: digit-span reverse typology is readable for the first time** (88 model errors
  against a 30 minimum; it was 20 and withheld). It agrees with forward, so the pattern is now
  well-powered on both: **humans substitute a wrong digit in the right place (0.4125 fwd /
  0.3550 rev) while the model omits or transposes** (omission 0.3737 fwd / 0.2955 rev,
  transposition 0.2727 rev). A consistent mechanism difference across both directions.
- **Serial position is the closest match anywhere in the benchmark**: forward recency-third
  distance **0.0039**, reverse middle-third 0.0105. So the model's *forgetting curve* is nearly
  human while its *error type* is not — the clearest case yet for why shape measures were needed.
- **`variable_mapping` at 20 questions moved M1 toward human**: the share of errors naming the
  person's own stale city went 0.0621 → **0.1718** against a human **0.2303** (distance 0.1682 →
  0.0585), on 1275 model errors instead of 515. More questions means more stale bindings exist to
  intrude, so the proactive-interference gap is smaller than it looked at 10 questions.
- **n-back is unchanged and still response-biased**: miss/FA 8.26 with a 95% CI of
  [6.55, 10.67] against the human 3.10 [2.41, 3.93] — still disjoint.

## Two things this run does NOT establish

**1. The digit-span columns cannot be differenced.** Both fell on tasks whose measured spread is
exactly 0.0000, and it is an estimator artefact, traced in `logs/digit_span_not_comparable.md`:
`score.py` makes one pseudo-participant per `sequence_index`, so 10 → 40 sequences moves the
sample from 10 to 40. Taking the **same** run restricted to its first 10 sequences gives 0.8857 /
0.9530 — matching neither column. Forward `best_span` sd goes 2.87 → 4.11: the small sample was
understating the model's dispersion, and W₁ is dispersion-sensitive. **The drop is the estimator
ceasing to flatter the model**, which is what decision D was for.

**2. craft_task's −0.0228 is noise, and its spread is quantised rather than Gaussian.** Per-run:
0.8565 / 0.8565 / 0.8223. Two arms identical, the third different — because, as
`logs/stimulus_variation_ceiling.md` shows, the task's entire variability is one question (C2003
Q4) flipping for some fraction of 50 identical pseudo-participants. The delta sits inside the
0.0342 spread.

## Re-measured run-to-run spread, on the current instrument

Supersedes the `iter11postfix` figures for the three changed tasks.

| task | iter11postfix spread | **iter12stage2 spread** |
|---|---|---|
| digit_span_forward | 0.0000 | **0.0000** (at 40 seq/span) |
| digit_span_reverse | 0.0000 | **0.0000** (at 40 seq/span) |
| nback | 0.0034 | 0.0061 |
| word_recognition | 0.0199 | **0.0122** |
| variable_mapping | 0.0033 | **0.0012** |
| narrative_qa | 0.0184 | 0.0140 |
| semantic_story_recall | 0.0115 | 0.0077 |
| craft_task | 0.0357 | 0.0342 |

`word_recognition`'s spread **fell** (0.0199 → 0.0122) despite the task becoming much harder,
and `variable_mapping`'s fell to 0.0012. craft_task reproduces at 0.0342 against 0.0357, which
confirms that figure was not an unlucky triple.

## What is now open

- **A2 needs a bounded reformulation** before it can be a guard (zero false alarms).
- **The human lag curve's rise is unexplained**, and survivorship is a testable candidate.
- **The n-back denominator decision** is still not taken (`logs/nback_denominator_decision.md`);
  it would erase the remaining n=3 deficit and needs a human call.
- **M4's grouping decision** for `best_span`, which now has four incompatible values on the same
  model (8.70 / 9.47 / 18.40 / 20.0 digits).
- **craft_task carries 1/8 of the objective with one bit of resolution**
  (`logs/stimulus_variation_ceiling.md`); four options are recorded, none taken.
