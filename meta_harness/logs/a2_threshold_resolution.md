# A2's thresholds after the scalar change — P6, the new band, and the adversary's broken leg

**2026-09-29 [USER]: "do all 3".** Three separate things, decided together because they all follow
from A2's scalar changing from the miss/false-alarm ratio to `miss_rate − fa_rate`.

**APPLIED 2026-09-29.** Items 1 and 2 landed in `check_predictions.py` (a `P6-substance` FAIL row
plus the forward-looking `A2-band` row); item 3 landed in `domain_spec.md`'s adversary section.
This file is kept as the derivation, not as a to-do.

## Units

`diff` = `miss_rate − fa_rate`, a **proportion difference** in [−1, +1]. Human **+0.2273**,
population sd **0.2658**, n=53. The legacy `ratio` is `miss_rate / fa_rate`, unbounded above and
undefined at `fa_rate = 0`; human **6.094**. Neither is humanlikeness.

---

## Item 1 — P6 is FAIL on substance, and its numeric form is un-evaluable

The registered row in `check_predictions.py`:

> **P6 A2 distance > 4.0 (does not approach human)** — baseline 5.754; *disconfirms below 2.0*
> "closing the leak removes a mixture artifact; **it does not install a conservative response
> criterion**, so a value near 5.754 is a coincidence of the mixture and must not be read as
> 'A2 unchanged'"

**The substantive prediction is falsified.** Closing the presentation leak did install a
conservative response criterion, in the strongest form available: `fa_rate` went from **0.1426** to
**exactly 0.0000** — zero false alarms in 487 new-word trials, for all 50 of 50 participants. The
model moved from too liberal, through the human value, to the conservative bound.

| | miss_rate | fa_rate | ratio | diff |
|---|---|---|---|---|
| human | 0.2719 | 0.0446 | 6.094 | **+0.2273** |
| pre-stage-2 | 0.0440 | 0.1426 | 0.3086 | **−0.0577** |
| post-stage-2 | 0.3752 | **0.0000** | **undefined** | **+0.3752** |

**What to do.** Record P6 as **FAIL on substance**, with the `fa_rate = 0.0000` evidence, and note
separately that the registered numeric form cannot be evaluated because the statistic it names
became undefined. Do **not** re-point the row at `diff_distance` — see item 2 for why.

This is the honest outcome: P6 was a real prediction, it was wrong, and the metric dying does not
convert a wrong prediction into an inconclusive one.

## Item 2 — a new band on the bounded scale, registered for FUTURE runs

**Why the old threshold cannot be translated.** The two scales disagree about the quantity the
threshold is anchored to — how far the pre-stage-2 baseline was from human:

| scale | baseline's distance from human | as a fraction of the maximum possible |
|---|---|---|
| ratio | 5.785 | **0.949** (max 6.094, the distance to the ratio's floor at 0) |
| `diff` | 0.2850 | **0.232** (max 1.2273, the distance to the bound at −1) |

The ratio said the baseline was 95% of the way to the floor; the bounded statistic says 23%. Any
"equivalent" threshold inherits whichever distortion is chosen, and **the disagreement is the
ratio's distortion, not a change of units.** So `> 4.0` has no faithful conversion.

**The new band.** Anchor it to the human distribution's own spread rather than to an invented
constant:

> **|diff − 0.2273| ≤ 0.2658** — within one human population sd of the human mean.

Measured against it:

| run | diff | distance | verdict |
|---|---|---|---|
| `iter11postfix/baseline` (pre-stage-2) | −0.0577 | 0.2850 | **outside** |
| `iter12stage2/baseline` (post-stage-2) | +0.3752 | 0.1479 | **inside** |

**State explicitly when registering it that the current run passes.** It is registered for
candidates *going forward*; it is not evidence about the run that motivated it. A threshold
derived after seeing the data it first scores is not a pre-registration, and this project has
already had to retract results for less.

A2 remains **report-only**. This band is a reporting reference, not a guard. Promotion still needs
a measured run-to-run spread for `diff` itself, which does not exist yet.

## Item 3 — the adversary's A2 pass condition is unsatisfiable and must be replaced

`domain_spec.md` states the noise adversary's pass condition as: mean humanlikeness ≥ baseline +
0.05 **AND A2 distance ≥ 2× baseline's**. The second leg is broken three ways.

**1. The number is out of range.** 2 × 5.785 = **11.57**, which is **1.9× the entire distance from
human (6.094) to the ratio's floor (0)**. Only reachable by becoming wildly *more* conservative
than human.

**2. Noise moves A2 TOWARD human, not away — the leg asks noise to do the opposite of what noise
does.** Dropping stored keys means failing to recognise old words: misses rise, false alarms fall,
the ratio rises toward the human 6.094. Measured on the existing runs:

| | A2 distance | vs baseline 5.754 |
|---|---|---|
| `random_decay` | **5.4288** | −0.325, i.e. CLOSER to human |
| `random_decay_v2` | **5.4344** | −0.320, CLOSER |

Not 2× further. Closer, both times.

**3. The scalar changed**, so the leg now names a statistic that is undefined on current runs.

**Consequence, and it is the important part.** **A2 cannot serve as the anti-noise guard the spec
assigns it.** Random forgetting makes the model look *more* human on A2. Together with A1 retired
and A4's human reference void, that is three of four axes failed as adversary guards — and A2 is
the one the spec leaned on hardest.

**Replacement.** The condition must name a measure that random key-dropping actually breaks. Strike
the A2 leg and replace with: *the adversary must be caught by at least one named error-shape
measure, by more than that measure's own measured run-to-run spread.* Principled candidates:

- **M1, variable_mapping intrusion class** (`interference.py`, reported by
  `report_error_shape.py`). Human: `stale_same_name` **0.2303**, `intrusion_other_name` 0.4605,
  `novel_guess` 0.3092. Naming the person's own *superseded* city requires that stale binding to
  still be represented somewhere. Random key-dropping destroys the binding outright, so it should
  drive `stale_same_name` toward 0 and `novel_guess` up — a signature noise cannot fake. **This is
  the best candidate.**
- ~~**M2, digit-span error typology.** Humans substitute a wrong digit in the right place (0.4125
  forward, 0.3550 reverse); random dropping should produce truncation and omission instead.~~
  **WITHDRAWN 2026-09-29, same day, on measurement. M2 cannot discriminate.** The baseline
  *already* has the signature I attributed to random dropping: at `iter12stage2/baseline`,
  forward `truncation` **0.3131** and `omission` **0.3737** against `substitution` **0.0808**,
  where the human is `substitution` **0.4125**. So "the model omits and truncates where humans
  substitute" describes the baseline, not a difference between baseline and adversary. **M1 is
  the only live candidate.** My error, caught by the agent implementing v3.

**Neither has a measured run-to-run spread**, which is exactly why no error-shape measure has been
promoted. So the replacement condition is not yet evaluable, and `domain_spec.md` should say so in
its own existing language: without a named measure failing by more than its own spread, the
adversary's verdict is **"untested rather than validated"**, not a pass.

**This is why the `random_decay_v3` run must be 3 repeats, not 1** — the repeats are what establish
those spreads. Communicated to the agent building v3 on 2026-09-29.

## What this chain says about the project

The Wave 0 verdict — *"the axes discriminate. Search is meaningful. Proceeding."* — rested on an
adversary that was (a) inert on 3 of 8 search tasks, (b) limited to ≤4 ability levels on 3 more,
and (c) evaluated on an A2 leg that could not be satisfied in the direction its own mechanism
pushes. The single axis credited with catching it was A1, on `digit_span_forward` — the one task
where the adversary genuinely worked — and A1 has since been retired as reproducible by a null
model. **The validity check has never actually run.** That supersession belongs in
`meta_harness/WORKLOG.md` as a dated amendment, appended, not edited in place.
