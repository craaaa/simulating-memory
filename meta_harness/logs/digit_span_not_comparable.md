# Digit-span humanlikeness is not comparable across a `sequences_per_span` change

Measured 2026-09-29 on arm 1 of job 18781213, while arms 2–3 were still running.

I warned in `logs/predictions_iter12stage2.md` (P5) that **`best_span` in digits** is not
invariant to trials-per-span. It turns out **humanlikeness on the digit-span tasks is not
invariant either**, and both tasks moved on the arm-1 table:

| task | iter11postfix (3 repeats) | iter12stage2 arm 1 | delta | measured 3-repeat spread |
|---|---|---|---|---|
| digit_span_forward | 0.9012 (identical all 3) | 0.8712 | **−0.0300** | **0.0000** |
| digit_span_reverse | 0.9666 (identical all 3) | 0.9568 | **−0.0098** | **0.0000** |

A −0.0300 move on a task whose measured run-to-run spread is exactly 0.0000 is not noise, so it
had to be traced before being called anything. **It is an estimator artefact, not a behaviour
change**, and the arm-1 mean-over-8 must be read with that in mind.

## Why the estimator moves

`src/score.py::_digit_span_participant_scores` makes **one pseudo-participant per
`sequence_index`** when a single participant carries many sequences. So `sequences_per_span`
10 → 40 (decision D, `4894c58`) takes the sample from **10 pseudo-participants to 40**, each
still with one trial per span. Each one's score is `best_span / 20`, where `best_span` is the
highest span passed before the first failure.

| | n | best_span mean | sd | humanlikeness |
|---|---|---|---|---|
| **digit_span_forward** | | | | |
| human | 52 | **6.88** digits | ~2.06 | — |
| iter11postfix, 10 seq/span | 10 | 8.70 | **2.87** | 0.9012 |
| iter12stage2, 40 seq/span | 40 | 9.47 | **4.11** | 0.8712 |
| iter12stage2, **same run, first 10 sequences only** | 10 | 9.10 | — | **0.8857** |
| **digit_span_reverse** | | | | |
| human | 49 | **5.90** digits | — | — |
| iter11postfix, 10 seq/span | 10 | 6.30 | 2.15 | 0.9666 |
| iter12stage2, 40 seq/span | 40 | 6.75 | 1.81 | 0.9568 |
| iter12stage2, **same run, first 10 sequences only** | 10 | 6.90 | — | 0.9530 |

The decisive row is the subsample: taking the **same** run and keeping only its first 10
sequences gives 0.8857 forward and 0.9530 reverse — neither the 10-sequence figure from the
earlier run (0.9012 / 0.9666) nor the 40-sequence figure from this one (0.8712 / 0.9568). So the
old numbers were not a stable property of the model; they were what a 10-sample estimate happened
to return.

**The mechanism is the spread, not the mean.** Forward `best_span` sd goes 2.87 → 4.11 as the
sample grows. The 10-sample estimate was *understating* the model's true dispersion, and
Wasserstein-1 is sensitive to dispersion, so the more honest estimate sits further from the human
distribution. **The −0.0300 is the estimator ceasing to flatter the model.** That is what
decision D was for; it just also means the two columns cannot be differenced.

The reverse task's sd moves the other way (2.15 → 1.81) and its humanlikeness still falls, which
is consistent — at n=10 both the mean and the spread were unreliable.

## `best_span` now has four values on the same model

Audit **M4** said the `best_span` estimator is not invariant to grouping; here is the current
count of incompatible figures for the *same* baseline model:

| figure | where it comes from |
|---|---|
| **8.70** digits | `src/score.py` grouping, 10 sequences/span |
| **9.47** digits | `src/score.py` grouping, 40 sequences/span |
| **18.40** digits | the figure quoted throughout `score_candidate.py`, `domain_spec.md`, `HANDOFF.md` |
| **20.0** digits | `full_context`, which never terminated on a 19-span schedule |

Human reference: **6.88** forward, **5.90** reverse.

### The 18.40 figure is load-bearing for an argument I wrote today, and at 9.47 that argument fails

`score_candidate.py`'s note on why `best_span` is **not** promoted to a guard — written by me this
morning, and repeated in `domain_spec.md` and the A1-retirement commit — turns on this arithmetic:

> the human best span is 6.88 digits and the baseline sits at 18.4, so a collapse to 2.0 gives
> distance 4.88 against the baseline's 11.52 and would read as an IMPROVEMENT.

At **18.40** that is correct: |2.0 − 6.88| = 4.88 < |18.40 − 6.88| = 11.52, so a catastrophic
collapse scores better than the baseline and the guard is wrongly signed.

At **9.47** it is false, and in the opposite direction: |9.47 − 6.88| = **2.59**, so a collapse to
2.0 gives 4.88 > 2.59 and would be flagged as **worse** — which is the correct behaviour. So
`best_span` may well be usable as a distance-from-human guard after all, and my stated reason for
refusing to promote it rests on a figure from a grouping the scorer does not use.

**I am not promoting it on that basis.** Two reasons: the argument's *conclusion* could still be
right for other reasons (the estimator's grouping-dependence is itself disqualifying until M4 is
settled — a guard whose reference value has four incompatible values cannot reject anything), and
reversing a just-made decision on one arm of one run is exactly the kind of move this project has
had to retract. But the justification currently in the code is **wrong as written** and is marked
here so it is corrected rather than inherited. The honest version is: *`best_span` is not a guard
because its value depends on a grouping nobody has fixed*, not *because it is wrongly signed*.

## What to do with the arm-1 table

Report the mean over 8 **and** the mean over the six comparable tasks, because two of eight are
measured differently than in the comparator:

| | iter11postfix | iter12stage2 arm 1 | delta |
|---|---|---|---|
| mean over 8 | 0.8828 | **0.9147** | +0.0319 |
| mean over the 6 tasks whose estimator did not change | 0.8657 | **0.9149** | **+0.0492** |

The six-task delta is the honest one, and it is **almost entirely `word_recognition`** (+0.2967
on its own, which is +0.0495 spread over six tasks). Everything else is inside its band.

## Consequence for the floors

`RUN_TO_RUN_SPREAD` records both digit spans at **0.0000**, measured over three repeats at 10
sequences/span. That figure describes a 10-sample estimator that happened to be deterministic, not
the task's precision. It must be re-measured from the three arms of this run at 40, and until then
a digit-span floor of 0.0000 should not be trusted to reject anything.
