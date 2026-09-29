# n-back: the two sides count different trials, and fixing it erases the remaining deficit

**NOT LANDED. This needs a decision, because applying it changes the headline n-back result in a
direction that flatters the model.** Measured 2026-09-29 against
`runs/iter11postfix/baseline`.

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

## Also still open on n-back, unchanged by this

- **M7, granularity.** The model is scored per (participant, level) — 150 points — and the human
  pooled over their three levels — 53 points. `nback_levels.py` measures the artefact directly:
  per-row humanlikeness 0.9366 against pooled 0.9489 on this run. The human records carry a
  `level` field on 49 of 57 participants, so the matched fix is per-level on both sides, which
  also gives three comparisons instead of one. Interacts with the denominator choice above, so
  decide them together.
- **M9**, the unequal scoreable count (14 model vs 14−n human), is a consequence of the block
  structure and cannot be fixed analysis-side without also fixing M6.
- **M21** is now near-moot: model `n_no_answers` is 0 at every level, so scoring non-response as
  an error costs nothing on the current instrument. It cost 0.157 proportion-correct pre-fix.
