# The human lag curve is not identifiable from stop-on-3-errors data

Measured 2026-09-29 after job 18781213. This closes the one scientific loose end I flagged in
`logs/iter12stage2_outcome.md` — and it closes it by showing the question cannot be answered from
this dataset, not by answering it.

## Units

All numbers are **proportion of OLD trials correct** (a word's repeat correctly called "Old"),
within a lag bin, where **lag = trials elapsed since that word's FIRST appearance** — the
project's own definition in `error_structure._lag_one_session`. Not humanlikeness.

## The anomaly

The human word-recognition lag curve **rises**:

| lag bin | 1–2 | 3–5 | 6–10 | 11–20 | 21+ |
|---|---|---|---|---|---|
| human | **0.506** | 0.798 | 0.926 | **0.950** | 0.888 |
| model (post-stage-2) | 0.772 | 0.766 | 0.565 | **0.335** | — |

Reproduced exactly from the cached reference. This is backwards for recognition memory: a word
seen two items ago should be *easier* to recognise than one seen twenty items ago, which is what
the model now does. I flagged survivorship as the likely cause. **That was too quick, and the
caveat text I wrote into `error_structure.py` saying so has been corrected.**

## Test 1 — survivorship: real, explains under half

Sessions stop at the 3rd error, so long-lag bins are populated only by participants who survived
long enough to *have* long-lag repeats. Conditioning on survival:

| subgroup | 1–2 | 3–5 | 6–10 | 11–20 | 21+ |
|---|---|---|---|---|---|
| all participants (n=53) | 0.506 | 0.798 | 0.926 | 0.950 | 0.888 |
| survived ≥ median (32 trials), n=27 | 0.688 | 0.946 | 0.946 | 0.948 | 0.919 |
| survived < median, n=26 | **0.317** | 0.643 | 0.895 | 0.955 | 0.500 |
| survived ≥ 50 trials, n=12 | **0.758** | 0.952 | 0.931 | **1.000** | 0.921 |

Rise from lag 1–2 to lag 11–20: **+0.4445** over everyone, **+0.2419** among those who survived
≥50 trials. So **46% of the rise is composition** — weak participants have most of their old
trials at short lag and drag that bin down. **54% is not.**

## Test 2 — within participant: the rise is still there

For the 40 participants who have both bins, computing each person's own
(lag 11–20 − lag 1–2) difference removes composition entirely:

> **mean +0.3571**, median +0.3333, sd 0.3752. **28 positive, 4 negative, 8 zero.**
> Wilcoxon signed-rank against 0: **p = 7.5 × 10⁻⁶**.

So it is not merely who sits in which bin.

## Test 3 — position in session: a second real effect, and the lag effect survives it

Short-lag repeats happen early by construction (a lag-21 repeat cannot occur before trial 21).
Old-trial accuracy by **absolute trial index**, ignoring lag entirely:

| trial index | 1–10 | 11–20 | 21–40 | 41+ |
|---|---|---|---|---|
| accuracy | **0.740** | 0.901 | 0.893 | 0.899 |

There is a distinct early-session deficit of about 0.16, consistent with a base-rate effect: in
the first few trials almost nothing has been presented yet, so "Old" is a priori unlikely and a
cautious participant answers "New". But the lag effect does **not** reduce to it — restricting to
trials at index ≥ 21:

| lag bin (index ≥ 21) | 1–2 | 3–5 | 6–10 | 11–20 | 21+ |
|---|---|---|---|---|---|
| accuracy | 0.680 | 0.860 | 0.867 | 0.915 | 0.932 |
| n trials | 25 | 43 | 75 | 117 | 191 |

Still rising, by +0.25.

## Why none of this identifies the effect

**The stopping rule censors trials non-randomly with respect to lag, and it does so WITHIN a
participant.** A long-lag trial can only exist if that participant had not yet made 3 errors by
the time it arrived. So long-lag trials are conditioned on earlier success, and short-lag trials
are not. Test 2 removes between-participant composition but **not** this — which is why the
within-participant estimate is still biased upward, and why the estimate shrinks as the survival
filter tightens (+0.4445 → +0.3571 → +0.2419) and shrinks again deep into sessions (at index ≥ 41
the curve is 0.800 → 0.931 on n=5 and n=29).

Every estimator available here inherits the bias, so **the human lag curve's slope is not
identifiable from this dataset**. It would take data without the 3-strike stop, or a model of the
stopping process, to separate memory from selection. Neither exists here.

## What this means for M4, and it is not all bad

- **Do not quote the human lag curve as a memory curve.** It is a memory curve convolved with a
  selection process, and the selection inflates its slope by an unknown amount between roughly a
  quarter and a half of the observed rise.
- **The model-vs-human comparison is more defensible than the human curve alone**, because since
  `dc14d0a` the model stops at 3 errors too, so both sides now carry the same censoring. That is an
  argument *for* having replicated the stopping rule in `bench` rather than truncating
  analysis-side.
- **But the sign of the difference is safe.** The bias inflates the *rise*, and the model's curve
  *falls* (0.772 → 0.335). No amount of upward bias on the human side turns a falling curve into a
  rising one, so "the model forgets with lag and the humans do not visibly do so" survives. The
  magnitudes do not.
- The early-session base-rate deficit (0.740 at trials 1–10 against ~0.90 later) is worth having
  on its own: it is a **response-bias** finding, and it lines up with the model's own zero
  false-alarm rate being the same kind of phenomenon at a different extreme.

## Correction to my own earlier claim

`error_structure.wr_lag_summary`'s post-fix caveat and `report_error_shape`'s human caveat, both
written by me earlier today (`1e030a9`), said the human rise "may be partly survivorship" and gave
the falling per-bin n as the evidence. That was right in direction and wrong in emphasis:
survivorship accounts for under half, the effect persists within participants at p = 7.5 × 10⁻⁶,
and the deeper problem is within-participant censoring, which no subgroup analysis can remove.
The caveats have been rewritten to say that.
