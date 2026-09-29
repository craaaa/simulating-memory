# A4's human reference is the termination rule, not interference

Found by the error-shape subagent 2026-09-29, verified independently here before acceptance.
**This retracts the project's headline structural result.**

**Units.** `rc_ratio` = mean `relationCount` on error trials ÷ mean `relationCount` on correct
trials. Dimensionless. 1.0 means errors are independent of memory load; above 1.0 means errors
concentrate on high-load items. `rc_ratio_normalized` rescales it against the ceiling a given
run's own error count permits, range 0–1. Humanlikeness is not involved in any number here.

## The finding

Human `variable_mapping` **terminates at the first error.** Verified over all 154 records:

```
records: 154   no questions key: 2   answered nothing: 0
errors per record: {1: 152}
first error IS last answered question: 152 | is NOT last: 0
relationCount non-decreasing within participant: 152/152
```

Every human participant has **exactly one error**, it is **always their last answered
question**, and `relationCount` never decreases within a participant. So "the error trial" and
"the last answered trial" are the *same 152 trials*, and the last trial is by construction the
highest-`relationCount` one that participant reached.

The decisive test — recompute the ratio with the last answered question substituted for the
error trial, regardless of whether it actually was an error:

```
A4 as computed:  mean(rc|error)=6.1842 (n=152)  mean(rc|correct)=4.4596 (n=607)  ratio=1.3867
NULL, position:  mean(rc|last) =6.1842 (n=152)  mean(rc|earlier)=4.4596 (n=607)  ratio=1.3867
```

**Identical to four decimal places**, because the two sets are the same set. The human
`rc_ratio` of 1.3867 — and the normalized 0.3728 derived from it — contains **zero**
information about whether human errors are load-dependent. It is the stopping rule plus a
monotone `relationCount`.

## What this retracts

`HANDOFF.md` and `WORKLOG.md` record that on Hermes, `respond_first` reaches normalized
`rc_ratio` **0.3751** against the human **0.3728** over 845 errors — "distance 0.0023, the
closest structural match to human interference structure measured anywhere in this project",
and the project's one transferable finding. **That comparison is void.** The model answers all
10 questions and its errors fall where they fall, so its `rc_ratio` measures something real
about the model; the human number it was matched against measures when the task stopped. Two
different quantities, agreeing to 0.0023 by coincidence.

This is the **fifth** measurement defect to overturn a conclusion in this project, and the
first to remove a positive result rather than a rejection.

## What survives

- **`variable_mapping`'s humanlikeness gain is untouched** — +0.34 to +0.60 across two models
  and four candidate variants. That is a score-distribution measure and has nothing to do with
  A4. Closing the history leak really does move it.
- **A4 as a guard on the model alone retains its purpose.** `score_candidate.py:383` fires when
  `variable_mapping` improves while `rc_ratio_normalized < 0.15`, i.e. when a candidate wins the
  task by making load-*independent* errors. That is a sanity check on the model's own error
  distribution and does not depend on the human reference being meaningful. Keep the guard;
  stop calling 0.3728 a target.
- **The error-count ceiling correction stands.** A4's original fixed 1.15 threshold could never
  reject anything because the ceiling depends on the run's own error count. That fix was real.

## What replaces it

**M1, the intrusion-type classification** (`interference.py`, built 2026-09-29): given that an
error occurred, *which* city was chosen — the stale binding for that person, a city currently
bound to someone else, or a novel guess. Human reference, pooled over 152 errors:

| class | human share of errors |
|---|---|
| `stale_same_name` | 0.2303 |
| `intrusion_other_name` | 0.4605 |
| `novel_guess` | 0.3092 |

The termination rule decides *when* a human errs, not *which* distractor they pick, so M1 is not
contaminated the way `rc_ratio` is. One caveat to carry: because the error is always the
participant's last and highest-load trial, the human distractor pool is systematically larger
than at a random trial, so the three shares describe errors made *under high load* rather than
errors in general. M1 is pooled-only by necessity — one error per participant means there is no
per-participant distribution to take a Wasserstein distance over.

## Required follow-ups

1. Amend `HANDOFF.md`'s A4 table and its "durable result" paragraph, and `WORKLOG.md` by dated
   amendment. Added to `logs/doc_audit.md` group 2.
2. Relabel A4 in `domain_spec.md`: a one-sided guard on the model's own error distribution, not
   a human-match axis. Its human column should be struck, not updated.
3. Do not use `rc_ratio` distance-from-human in any future report or pre-registration.

## Related, found in the same pass and much smaller

> **FIXED 2026-09-29 [USER] as "Option D"** — this is audit item M6, landed in `src/score.py`
> together with the granularity fix M7. See `logs/nback_denominator_decision.md`. The two
> measurements in the table below reproduce exactly under the landed code (pooled human
> 0.8657 → 0.8569 proportion-correct). One correction: the sentence "the model's over-accuracy
> gap is slightly larger than recorded" is wrong in sign on these baselines — the model is
> *below* the human on n-back (`iter12stage2` three-arm means 0.9943 / 0.7762 / 0.7405 against
> human 0.9492 / 0.8553 / 0.7496, so it is above only at n=1), so moving the human reference
> down *narrows* the gap at n=2 and n=3. `logs/protocol_mismatch_audit.md` §4 M6 already noted
> this discrepancy with this file; the audit is the one that is right.

Human n-back scores divide by **all** non-practice trials, including the *n* lead-in trials,
while the model's `acc_over_14` counts scored trials only. And 112 of 318 human lead-in trials
are logged `target: true`, which is impossible — there is no letter *n* back — affecting 48 of
57 participants, 14 of them on the very first trial of the 1-back block. Post-lead-in the logged
flag agrees with a recomputed one on 1908/1908 trials, so the defect is confined to the lead-in.

Impact on the human reference, measured:

| human n-back score | mean | sd | n |
|---|---|---|---|
| all non-practice trials (what `src/score.py` uses) | **0.8657** | 0.0820 | 53 |
| scored trials only (matches the model's denominator) | **0.8569** | 0.0853 | 53 |

A shift of **−0.0088 proportion-correct**, and it moves the human reference *down*, so the
model's over-accuracy gap is slightly larger than recorded. Real, small, worth fixing
analysis-side, and nothing like A4's problem in magnitude. Human accuracy on the lead-in trials
themselves is 0.9182, so they are not noise — they are easy trials inflating the human mean.
