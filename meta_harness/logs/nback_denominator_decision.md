# n-back: the two sides count different trials, and fixing it erases the remaining deficit

> ## RESOLVED 2026-09-29 [USER] — **Option D: Option A *plus* the M7 per-level fix, as one change.**
>
> Landed in `src/score.py` (`nback_human_by_level`, `nback_human_scores`,
> `human_scores("nback")`). n-back is scored per `(participant, n-level)` on **both** sides,
> with the human **lead-in trials excluded**, so both sides count only trials where the task
> is well-defined. Neither Option A nor Option C alone was chosen; Option B was rejected for
> the reason given below. Labels, from `protocol_mismatch_audit.md`: **M6** = unmatched
> denominators (the lead-in), **M7** = unmatched granularity.
>
> **Outcome, in humanlikeness (= 1 − W₁ between the two score distributions, in [0,1]), mean
> over three arms per run-set, using the project's own `score.wasserstein_1d`:**
>
> | combination | `iter11postfix` | `iter12stage2` |
> |---|---|---|
> | legacy shape (pooled human, lead-in included) | 0.9344 | 0.9340 |
> | M6 only (pooled human, lead-in excluded) | — | 0.9419 |
> | M7 only (per-level both sides, lead-in included) | — | 0.9627 |
> | **M6 + M7 (landed)** | **0.9622** | **0.9633** |
>
> Mean over the 8 search tasks: `iter11postfix` 0.8828 → 0.8862, `iter12stage2`
> 0.9130 → 0.9167.
>
> **Two corrections to the recommendation below, both material.**
>
> 1. **M7 dominates and M6 is nearly free — the opposite of this file's emphasis.** M7 alone
>    is **+0.0287** humanlikeness; M6 adds **+0.0006** on top of it, which is *inside*
>    n-back's measured run-to-run spread of 0.0061, and its sign is **not identified** across
>    reasonable specifications (it is −0.0020 if the human pool is restricted to the 49
>    level-carrying records). The **0.0378** figure quoted under Option C below is per-level
>    **accuracy** at n=3, not humanlikeness, and this file conflated the two when it called
>    the denominator the big fix. M6 is a correctness fix, not a scoring gain, and the
>    worry that it "flatters the model" was misplaced: almost none of the movement is M6's.
>
> 2. **M6 makes the n=3 DISTRIBUTION match slightly worse even though it closes the mean
>    gap.** Excluding the lead-in widens the human n=3 spread (sd 0.1492 → 0.1657 at the
>    49-record pool; 0.1459 → 0.1618 at the 53-record pool actually used) while the model
>    sits at sd ≈ 0.096. Closing the mean gap therefore opens the **dispersion** gap. **Any
>    statement of the n=3 mean gap must carry this caveat**, or it reads as a stronger result
>    than it is.
>
> **The n=3 gap literal below is superseded.** "0.7429 against 0.7421, a gap of 0.0008" was
> computed on the 49-participant human pool against `iter11postfix/baseline`. The landed code
> uses **53** participants per level (see next paragraph), giving a human n=3 mean of
> **0.7496**; against `iter12stage2/baseline`'s model 0.7214 the gap is **−0.0282**
> proportion-correct, and against the three-arm model mean 0.7405 it is **−0.0091**. The
> qualitative claim — the n=3 mean gap is small — survives; the number does not, and the
> dispersion caveat above applies wherever it is quoted.
>
> **The human n is 53 participants per level, not 49.** 4 of the 57 records carry
> `level: null` on every trial; their level is recovered from the block name ("2-back"),
> exactly as `error_structure.nback_human_trials` already did for the M3 measure, and their
> blocks are structurally identical to the other 49 (14 trials indexed 1..14 at each of
> n=1,2,3). Keeping them means the legacy pooled path and the new per-level path use the
> **same** 53 participants, so the before/after above is not confounded by a sample change.
> The remaining 4 of the 57 have an empty `payload` **and** an empty `summary`, so they have
> no trials and no accuracy on any shape, legacy included; they were already absent from
> every n-back figure this project has ever reported. Consequently the lead-in totals differ
> by record scope: **294 trials / 106 impossible `target: true`** at the 49-record scope used
> throughout this file, **318 / 112** at the 53-record scope the landed code uses. Both are
> correct for their scope; `error_structure.py` and `test_error_shape.py` use 318/112.
>
> **The historical figures are preserved, not recomputed.** This file's closing instruction
> to recompute `logs/nback_turn_order_outcome.md` was **not** followed: that file and
> `logs/iter12stage2_outcome.md` are frozen records of what was observed at the time and are
> **annotated in place** instead. `score.nback_human_scores_legacy_pooled()` reproduces the
> old shape exactly — it returns the recorded 0.9344 for `iter11postfix` and the recorded
> run-to-run spreads 0.0034 / 0.0061, which is the check that the legacy path still works.

**NOT LANDED. This needs a decision, because applying it changes the headline n-back result in a
direction that flatters the model.** Measured 2026-09-29 against
`runs/iter11postfix/baseline`. *(Superseded by the RESOLVED block above — it is landed.)*

## Units

Everything in this file is **proportion of trials correct**, per participant then averaged — not
humanlikeness. Humanlikeness = 1 − Wasserstein-1 between the two per-participant distributions,
and is called out explicitly where used.

## What each side actually counts

A block at level *n* opens with *n* **lead-in** letters, where no letter *n* positions back exists
yet and the correct response is "no response".

| | lead-in trials | scoreable trials | where lead-in goes |
|---|---|---|---|
| **model** | *n*, presented separately | **14 at every level** | `buffer_map`; excluded from `acc_over_14` |
| **human** | *n*, inside the block | **14 − n** (13 / 12 / 11) | counted in `payload.trials` like any other |

Verified from the data rather than assumed: every human block has **14 non-practice trials total**
with `trial` indices starting at 1 (49 of 49 participants at each level), and the first *n* of them
are the lead-in. So the human's 14 *includes* the lead-in and the model's 14 *excludes* it.

`src/score.py:_score_human_record` and `nback_levels.human_by_level` both count every
non-practice trial, so **the human denominator contains the lead-in and the model's does not.**
This is audit item **M6**; the unequal scoreable count is audit item **M9**.

## The lead-in trials are not measurements

| level | lead-in trials | accuracy on them | `target: true` — impossible | post-lead-in accuracy |
|---|---|---|---|---|
| n=1 | 49 | 0.9184 | 14 (28.6%) | 0.9482 |
| n=2 | 98 | 0.9184 | 38 (38.8%) | 0.8520 |
| n=3 | 147 | 0.9184 | 54 (36.7%) | 0.7421 |

**106 of 294 human lead-in trials are flagged as targets**, which cannot happen — there is no
letter *n* back to match. Whatever the web app did on those trials, it was not the task. Lead-in
accuracy is also flat at 0.9184 across levels while real accuracy falls steeply (0.948 → 0.742),
which is what you would expect of trials that are not measuring the level.

> **A hypothesis of mine that turned out to be wrong, recorded so nobody re-derives it.** The
> identical 0.9184 at all three levels looked like a per-participant artefact — the same few
> participants getting every lead-in trial wrong. I tested it: of 49 participants, 32 have all
> lead-in trials correct, **17 are mixed**, and none is all-wrong. So it is not per-participant.
> The identical rate is simply 24 errors splitting proportionally over 49 / 98 / 147 trials
> (4 / 8 / 12), which is the *expected* split, not a coincidence. The observation was a red
> herring; the impossible `target` flags are the real evidence.

## What changes if the human denominator is matched to the model's

| level | human, lead-in INCLUDED (current) | human, lead-in EXCLUDED (matched) | model | current gap | matched gap |
|---|---|---|---|---|---|
| n=1 | 0.9461 | 0.9482 | 0.9929 | +0.0468 | +0.0447 |
| n=2 | 0.8615 | 0.8520 | 0.7786 | −0.0829 | −0.0734 |
| n=3 | **0.7799** | **0.7421** | **0.7429** | **−0.0370** | **+0.0008** |

**The n=3 deficit disappears.** 0.7429 against 0.7421 is a gap of 0.0008. The whole remaining
n-back story at the hardest level — "the model is slightly worse than humans at 3-back" — is a
denominator artefact, if this correction is right.

That is exactly why it is not landed. It is a correction that favours the model, derived from my
reading of a field the data itself shows to be corrupt, and the project has already had to
withdraw three results for less. It needs to be someone's decision, not a quiet commit.

## The choice to make

**Option A — exclude the human lead-in (matched denominators).** Both sides then score only trials
where the task is well-defined. Costs: the human n drops from 14 to 14−n trials per participant
per level, so the human side gets noisier at n=3 (11 trials), and it discards 294 trials of real
human responses on the grounds that 106 of them are impossible.

**Option B — include the model's lead-in too (matched the other way).** The model's lead-in
responses exist in `buffer_map` and `buffer_no_response_frac` is 1.0 at n=1 and n=2 and 0.9933 at
n=3, i.e. the model answers "no response" essentially always, which is *correct*. Adding them
would push model accuracy **up** at every level, widening the n=2 gap and reversing n=3 further.
This is worse: it compares the model's correct lead-in behaviour against human lead-in trials that
are 36% impossible.

**Option C — leave it, document it.** What the audit originally recommended, at an estimated
0.0088 score units. That estimate was for the *pooled* human score; per level it is up to 0.0378
at n=3, four times larger, and it lands exactly on the number the project would want to quote.

**My recommendation: Option A**, and re-state the n-back result as "the model matches humans at
3-back and is still 0.073 worse at 2-back", which is a stranger and more interesting result than
the current one. But it must be a decision, and whichever is chosen, the n-back humanlikeness
figures in `logs/nback_turn_order_outcome.md` need recomputing and the old ones marked superseded.

> **DECIDED 2026-09-29 [USER]: Option D — Option A *and* M7 together.** The re-statement above
> is directionally right at 53 participants (n=2 remains the worst-matched level: model 0.7800
> against human 0.8553, a gap of −0.0753) but the n=3 "matches" claim needs the dispersion
> caveat in the RESOLVED block. The instruction to recompute `nback_turn_order_outcome.md` was
> **overridden**: frozen outcome records are annotated, never recomputed in place.

## Also still open on n-back, unchanged by this

- **M7, granularity.** ~~Still open.~~ **RESOLVED 2026-09-29 [USER], landed with M6 as Option
  D — see the RESOLVED block at the top.** The model is scored per (participant, level) — 150
  points — and the human *was* pooled over their three levels — 53 points. `nback_levels.py`
  measures the artefact directly: per-row humanlikeness 0.9366 against pooled 0.9489 on this
  run. The human records carry a `level` field on 49 of 57 participants — **and the level is
  recoverable from the block name for 4 more, so the landed fix uses 53 per level, not 49.**
  Both sides are now per level. M7 is the fix that actually moved the number: +0.0287 of the
  +0.0293 total.
- **M9**, the unequal scoreable count (14 model vs 14−n human), is a consequence of the block
  structure and cannot be fixed analysis-side without also fixing M6.
- **M21** is now near-moot: model `n_no_answers` is 0 at every level, so scoring non-response as
  an error costs nothing on the current instrument. It cost 0.157 proportion-correct pre-fix.
