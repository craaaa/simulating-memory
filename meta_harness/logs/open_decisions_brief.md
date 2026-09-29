# Four open decisions — full context, 2026-09-29

Written because the condensed version was not enough to decide from. Nothing here is urgent:
**the only enforced guard in `score_candidate.py` today is A3** (two fields), A4 is conditional,
and A2, `best_span` and every error-shape measure are report-only. A1 is retired. So none of these
four is repairing a live gate; they are about what the benchmark measures next.

## Units, stated once

- **Humanlikeness** = 1 − Wasserstein-1 between the model's and the humans' per-participant score
  distributions. Range 0–1, higher is more human-like. Only a **delta** (one humanlikeness minus
  another) can be negative.
- **Accuracy / miss rate / false-alarm rate / error-class share** are proportions of trials or of
  errors, in their own units. **Not** humanlikeness.
- **`best_span`** is in **digits**.
- **Survival length** is in **items presented** (words, or questions).

---

# Decision 1 — A2 has no defined value any more. What replaces it?

## What A2 is

**A2** is the `word_recognition` **miss / false-alarm ratio**, computed per participant then
averaged. A *miss* is an old word called "New"; a *false alarm* is a new word called "Old".

**Worked example.** A participant sees 20 words. 8 are repeats (old), 12 are first appearances
(new). They call 6 of the 8 old words "Old" and 2 of them "New" → miss rate 2/8 = 0.250. They call
1 of the 12 new words "Old" → false-alarm rate 1/12 = 0.083. A2 = 0.250 / 0.083 = **3.0**. A ratio
above 1 means the participant errs *conservatively* (fails to claim recognition); below 1 means
*liberally* (claims recognition it doesn't have).

Humans sit at **6.094** — strongly conservative. `domain_spec.md` calls A2 "the axis" because
every model tested was on the wrong side of it, so it was the one axis with real headroom.

> **AMENDED 2026-09-29 [USER]:** both sentences in this paragraph are now qualified. 6.094 is the
> ratio of the human population means, not a per-participant average (see the RESOLVED section
> below), and "the axis" framing is withdrawn in `domain_spec.md` — A2 is report-only.

## What happened

| | miss rate | false-alarm rate | A2 ratio |
|---|---|---|---|
| humans | 0.272 | 0.045 | **6.094** |
| model, pre-stage-2 | 0.044 | 0.143 | **0.309** — liberal, wrong side of 1 |
| **model, post-stage-2** | **0.375** | **0.0000** | **undefined** |

Post-fix confusion matrix over 969 trials: (old→old) 332, (old→new) 150, (new→new) 487,
**(new→old) 0**. The model never once called a new word "Old" across 487 new-word trials.

So A2 went from the wrong side of human, past human, to the limit — and the ratio is a division by
zero, not a small number. **The statistic has no value on its own substrate.**

## Why this is a formulation problem, not a result problem

A ratio of two rates is unbounded above and undefined at zero. That was tolerable when the model
false-alarmed constantly; it is not now. Note the mirror case already in `NOTES.md`:
`claude-opus-4-6` scored A2 = **0.00** (miss 0.000, FA 0.533) — also a degenerate endpoint, at the
other end. The statistic breaks at *both* extremes, and two of the models tested hit one.

## Options

**Option A — miss minus false alarm.** Bounded in [−1, 1], defined everywhere.
Humans **+0.227**; model now **+0.375**; model before **−0.099**; opus **−0.533**.
Cheap, no new machinery, and it orders all the cases sensibly. Loses the "how many times more"
intuition the ratio gave.

**Option B — d′ (signal-detection sensitivity), with a correction.**
d′ = z(hit rate) − z(false-alarm rate). It separates *sensitivity* from *bias*, which is what this
measure is really about, and bias would be a second number (criterion *c*). **But d′ is also
undefined at FA = 0** — z(0) = −∞ — so it needs the standard log-linear correction (add 0.5 to
each cell, divide by n+1). With that, the model's FA becomes ≈ 0.5/487 and d′ is finite but is
driven by an arbitrary constant precisely in the regime we are in. More principled, more moving
parts, and the correction does real work here rather than being cosmetic.

**Option C — report both rates and drop the composite.** Miss and FA separately, each with a
distance from human. Two numbers instead of one, no degeneracy, no information destroyed. Costs
the ability to say "A2 improved" in one scalar, which matters if it ever becomes an objective.

**Option D — leave A2 as-is and mark it inapplicable when FA = 0.** Zero work; the axis silently
stops being measurable exactly when the model becomes interesting.

## Recommendation

**Option A now, Option C alongside it.** Take miss − FA as the scalar (bounded, defined, orders
every observed case correctly) and report the two rates beside it so the composite can never hide
which side moved. Skip d′ until there is a reason to model sensitivity and bias separately — its
correction term would be doing the work in exactly the regime that matters.

**Cost:** an hour of analysis-side work, no re-run. The numbers already exist in `per_trial`.

## RESOLVED 2026-09-29 [USER] — Option A + Option C, implemented, A2 stays report-only

Taken as recommended. `score_candidate.axes()["A2"]` now leads with `diff` = miss_rate − fa_rate,
a **proportion** in [−1,+1], with `diff_ci` (paired participant bootstrap, `a2_diff_ci`),
`diff_sd`, `diff_distance` from human, `diff_one_minus_w1`, and `miss_rate` / `fa_rate` each with
their own `*_distance`. The ratio survives only as labelled legacy fields.

Four corrections to the analysis above, all measured through the project's own `a2_human()` /
`a2_model()`:

1. **The ratio was not merely undefined post-stage-2 — it was undefined for most humans all
   along.** Per-participant FA rate is exactly 0 for **31 of 53 humans (58%)**, so the human ratio
   existed for 22 of 53. "Computed per participant then averaged", in the description above, is
   not what the reported 6.094 ever was.
2. **6.094 is a ratio of two population means** (0.2719 / 0.0446). The mean of the 22 defined
   per-participant ratios is **1.513**. The worked example above is still a correct account of
   what a per-participant ratio *would* mean; it is not an account of 6.094.
3. **This, not the degeneracy, is the decisive argument.** The project's objective is distribution
   matching, and a pooled scalar has no distribution. `diff` is defined for every participant on
   both sides, so A2 is now *eligible* for the objective — and was deliberately **not** promoted.
   It stays report-only; only A3 is enforced. Promotion needs a measured run-to-run spread for
   `diff` itself.
4. **The Option A figures above are POOLED differences; the implementation reports the mean of
   per-participant differences, and they differ.** Humans agree at **+0.2273** (sd 0.2658, ddof=0,
   n=53). Pre-stage-2 `iter11postfix/baseline` is **−0.0577** as a mean of per-participant
   differences, not the −0.099 quoted above (0.044 − 0.143), because 3 of its 50 participants have
   no old-word trials and drop out. Opus is **−0.5037**, not −0.533, for the same reason (4 of 50).
   Post-stage-2 `iter12stage2/baseline` is **+0.3752**, which matches, since no participant there
   drops out.

Option B (d′) was evaluated and rejected on measurement, not on taste: with the log-linear
correction it gives −0.6205 as a `1 − W_1` against the human distribution on
`iter11postfix/baseline` — a negative "humanlikeness", because d′ is unbounded so its W_1 has no
[0,1] range — and at FA = 0 the correction constant sets the value.

As `1 − W_1` over per-participant `diff`, in proportion units and **not** humanlikeness:
`iter11postfix/baseline` **0.7286**, `iter12stage2/baseline` **0.8425**, using the project's
`src/score.wasserstein_1d`. `scipy.stats.wasserstein_distance` gives 0.7150 / 0.8333 on the same
values — the project's grid approximation clips its outer edges, so it reads slightly closer. The
project's own function is used, for comparability with every other `1 − W_1` in the repo.

---

# Decision 2 — matching the n-back denominators erases the remaining n=3 deficit

> ## DECISION 2 RESOLVED 2026-09-29 [USER] — **Option D = the recommendation below (Option A
> plus the M7 per-level fix), landed as one change.**
>
> Landed in `src/score.py`; see `logs/nback_denominator_decision.md` for the full RESOLVED
> block. Summary of what this section got right and what it got wrong:
>
> - **Right:** M6 and M7 had to be decided together; Option B was correctly rejected;
>   analysis-side only, no re-run.
> - **Wrong, and it matters:** this section's emphasis. **M7 is the fix that moves the number**
>   (+0.0287 humanlikeness on the three `iter12stage2` arms); M6 adds **+0.0006**, inside
>   n-back's 0.0061 run-to-run spread and with a sign that is not identified across
>   specifications. The **0.0378** quoted under "Why I did not just do it" is per-level
>   **accuracy** at n=3, **not** humanlikeness — this section compared it to a humanlikeness
>   estimate and concluded M6 was four times bigger than the audit thought. The worry that the
>   fix "flatters the model" was therefore misplaced: essentially none of the gain is M6's.
> - **Wrong:** the n=3 tables below are at the **49-participant** human pool. 4 more records
>   carry `level: null` but have recoverable levels (from the block name, as
>   `error_structure.py` already did), so the landed code uses **53** per level. At 53 the
>   human n=3 mean is **0.7496** (lead-in excluded), not 0.7421, and the lead-in totals are
>   **318 trials / 112 impossible targets**, not 294 / 106.
> - **Missing, and required wherever the n=3 gap is quoted:** excluding the lead-in **widens**
>   the human n=3 spread (sd 0.1492 → 0.1657 at the 49-pool, 0.1459 → 0.1618 at 53) while the
>   model sits near 0.096. So M6 closes the n=3 **mean** gap and *opens* the **dispersion**
>   gap. "The n=3 deficit disappears" is true of the mean only.
>
> Net effect: n-back humanlikeness 0.9340 → **0.9633** on `iter12stage2`, 0.9344 → **0.9622** on
> `iter11postfix`; mean over 8 search tasks +0.0037 and +0.0035.

## The situation

A block at n-back level *n* opens with *n* **lead-in** letters, where no letter *n* positions back
exists yet and the correct answer is "no response".

**Worked example, n=3.** Letters arrive X, W, M, Q, … The first three (X, W, M) have no letter
three back, so they are lead-in. Trial 4 (Q) is the first real trial: is Q the same as X?

| | lead-in trials | scoreable trials | where lead-in goes |
|---|---|---|---|
| **model** | 3, presented separately | **14** | `buffer_map`, excluded from `acc_over_14` |
| **human** | 3, inside the block | **11** (14 − n) | counted like any other trial |

Verified from the data: every human block has exactly **14 non-practice trials** with `trial`
indices from 1, for 49 of 49 participants at each level. So the human 14 *includes* the lead-in and
the model's 14 *excludes* it. `src/score.py` and `nback_levels.py` both count every human
non-practice trial. This is audit item **M6**; the unequal scoreable count is **M9**.

## The human lead-in trials are not measurements

| level | lead-in trials | accuracy on them | flagged `target: true` (impossible) | post-lead-in accuracy |
|---|---|---|---|---|
| n=1 | 49 | 0.9184 | 14 (28.6%) | 0.9482 |
| n=2 | 98 | 0.9184 | 38 (38.8%) | 0.8520 |
| n=3 | 147 | 0.9184 | 54 (36.7%) | 0.7421 |

**106 of 294 human lead-in trials are marked as targets**, which cannot happen — there is no
letter *n* back to match. Whatever the web app scored there, it was not the task.

## What changes

| level | human, lead-in INCLUDED (current) | human, lead-in EXCLUDED | model | current gap | matched gap |
|---|---|---|---|---|---|
| n=1 | 0.9461 | 0.9482 | 0.9929 | +0.0468 | +0.0447 |
| n=2 | 0.8615 | 0.8520 | 0.7786 | −0.0829 | −0.0734 |
| **n=3** | **0.7799** | **0.7421** | **0.7429** | **−0.0370** | **+0.0008** |

**The n=3 deficit disappears.** "The model is slightly worse than humans at 3-back" becomes "the
model matches humans at 3-back and is 0.073 worse at 2-back" — a stranger and more interesting
claim than the current one.

## Why I did not just do it

It is a correction that **flatters the model**, derived from my reading of a field the data itself
shows to be corrupt, and it lands exactly on the number the project would most like to report. The
audit's own size estimate for M6 was 0.0088 score units — that was for the *pooled* human score;
per level it is up to **0.0378**, four times larger.

## Options

**Option A — exclude the human lead-in (match the model).** Both sides then score only trials where
the task is well-defined. Costs: human trials per participant per level drop to 14 − n (11 at n=3),
so the human side gets noisier where it matters most, and it discards 294 real human responses
because 106 of them are impossible.

**Option B — include the model's lead-in too (match the other way).** The model's lead-in answers
exist: `buffer_no_response_frac` is 1.0 at n=1 and n=2 and 0.9933 at n=3, i.e. it answers "no
response" essentially always, which is **correct**. Adding them pushes model accuracy *up* at every
level, widening the n=2 gap. Worse: it would compare the model's correct lead-in behaviour against
human lead-in trials that are 36% impossible.

**Option C — leave it, document it.** What the audit recommended at its (understated) 0.0088.

**Interacts with M7** (granularity): the model is scored per (participant, level) — 150 points —
and the human pooled across their three levels — 53 points. `nback_levels.py` measures the artefact
directly: per-row humanlikeness 0.9366 against pooled 0.9489. The human records carry a `level`
field for 49 of 57 participants, so the matched fix is per-level on both sides. **Decide M6 and M7
together** — they touch the same numbers.

## Recommendation

**Option A plus the M7 per-level fix**, done as one change, with the before/after published and the
old n-back figures explicitly marked superseded. The current comparison is unmatched in a way I can
demonstrate; leaving it because the correction is convenient in our favour is not better science
than making it and saying so loudly.

**Cost:** analysis-side only, no re-run. Half a day including re-recording every n-back figure in
the logs.

> **ACCEPTED 2026-09-29 [USER] as "Option D".** One deviation from the plan: the old n-back
> figures in `logs/nback_turn_order_outcome.md` and `logs/iter12stage2_outcome.md` were
> **annotated, not re-recorded** — they are frozen records of what was observed at the time, and
> this project supersedes explicitly rather than rewriting. `score.nback_human_scores_legacy_pooled()`
> reproduces the old shape exactly so those figures stay checkable.

---

# Decision 3 — `best_span` has four values on the same model

## What it is

**`best_span`** is the longest digit sequence a participant reproduces exactly before failing,
in **digits**. Humans: **6.88** forward, **5.90** reverse.

**Worked example.** A participant gets 4 digits right, 5 right, 6 right, then fails 7 twice. Their
`best_span` is 6 digits.

## The problem

The same baseline model yields:

| figure | grouping |
|---|---|
| **8.70** digits | `src/score.py`, 10 sequences/span |
| **9.47** digits | `src/score.py`, 40 sequences/span (current config) |
| **18.40** digits | the figure quoted in `score_candidate.py`, `domain_spec.md`, `HANDOFF.md` |
| **20.0** digits | `full_context`, which never terminated on a 19-span schedule |

`protocol_match.py` uses a third grouping again (pairing adjacent sequences into 2-trials-per-span
blocks, giving 20 matched pseudo-participants at the current config). Audit **M4** is exactly this:
the estimator is not invariant to trials-per-span or to how rows are grouped into participants.

## Why it matters beyond tidiness

Two things depend on it.

**1. My stated reason for refusing to promote `best_span` to a guard is wrong.** The argument in
`score_candidate.py` is: humans 6.88, baseline 18.4, so a collapse to 2.0 gives distance
|2.0 − 6.88| = 4.88 against the baseline's 11.52 and **reads as an improvement** — a wrongly-signed
guard. At 18.40 that is correct. At **9.47** it fails: |9.47 − 6.88| = 2.59, so a collapse to 2.0
gives 4.88 > 2.59 and correctly reads as **worse**. So `best_span` may be promotable after all, and
the reason recorded in the code is not the real one.

**2. Digit-span humanlikeness is not comparable across a config change either** — established this
run. Both spans "fell" (−0.0300 forward, −0.0098 reverse) on tasks whose measured spread is exactly
0.0000. Traced: `score.py` makes one pseudo-participant per `sequence_index`, so 10 → 40 sequences
moved the sample from 10 to 40. Taking the **same run** restricted to its first 10 sequences gives
0.8857 / 0.9530 — matching neither column. Forward `best_span` sd goes 2.87 → 4.11: the small
sample was understating the model's dispersion, and Wasserstein-1 is dispersion-sensitive. The drop
is the estimator ceasing to flatter the model.

## Options

**Option A — declare `src/score.py`'s grouping canonical at a pinned `sequences_per_span`.**
One number, reproducible, and the config change that produced it is already made (40). Cost: every
`best_span` figure in the docs must be restated, and the pin has to be enforced or the problem
returns.

**Option B — declare `protocol_match.py`'s 2-trials-per-span staircase canonical**, since it is the
one that actually replicates the human administration (2 trials per span, ascend, stop on double
failure). More faithful; gives 20 matched participants at the current config. Cost: it is a
different number again, and it needs `sequences_per_span` to stay even.

**Option C — stop reporting `best_span` as a scalar** and keep only the serial-position curve and
error typology, which are grouping-independent and, on this run, the *best*-matching measures in
the benchmark (forward recency-third distance **0.0039**).

## Recommendation

**Option B**, because matching the human administration is the whole point of the measure, and then
promote nothing until it has been measured across three repeats at the pinned config. Correct the
code comment either way — its current justification is demonstrably wrong.

**Cost:** an hour to switch the reported figure and restate the docs. No re-run.

---

# Decision 4 — craft_task carries 1/8 of the objective with one bit of resolution

## What was found

A "participant" in this benchmark is a seeded stimulus set run by one deterministic model. For five
of eight tasks each participant gets its own stimulus. For three it does not:

| task | rows | distinct stimuli | rows per stimulus | distinct score values |
|---|---|---|---|---|
| narrative_qa | 50 | 10 | 5.0 | 6 |
| semantic_story_recall | 200 | 4 | **50.0** | 64 |
| **craft_task** | 150 | **3** | **50.0** | **2** |

**All 50 craft pseudo-participants see byte-identical questions** (verified by hashing) and in
`iter11postfix/baseline` all 50 produced the **identical** result: C2001 = 1.0, C2002 = 1.0,
C2003 = 0.8. One distinct signature, zero between-participant variance.

The banks are exhausted: `data/craft_task.json` holds exactly **3** items, `data/narrative_QA.json`
exactly **10**, story recall 4 transcripts. So this is a design ceiling, not a misconfiguration.

## Consequences, measured

**Its entire variability is one question flipping.** C2003 Q4 was wrong for 50/50 participants in
repeat 1, 29/50 in repeat 2, 27/50 in repeat 3. That is the whole of its 0.0357 run-to-run spread,
and it is why the spread is **quantised, not Gaussian** — in `iter12stage2` two arms gave exactly
0.8565 and the third 0.8223.

**At the human's 15-question unit the comparison is a point mass against a distribution:**

| | n | mean | sd | distinct values |
|---|---|---|---|---|
| human | 54 | 0.8395 | **0.1180** | **8** (0.40 … 1.00) |
| model, `iter11postfix/baseline` | 50 | 0.9333 | **0.0000** | **1** |

So audit **M12**'s proposal — pool the model to the human's 15-question unit — makes it *worse*,
verified: pooling gives a single value with sd exactly 0.0000, the state the audit itself called
pathological on hermes.

**Therefore craft_task can register that one question flipping and nothing else.** A candidate that
changes memory behaviour without touching C2003 Q4 is invisible to it; one that happens to flip it
moves an eighth of the primary objective.

## Options

**Option 1 — weight the objective by resolution, or report craft separately** instead of as 1/8 of
the mean. Honest, cheap, changes the headline metric.

**Option 2 — stop spending 150 rows on 3 stimuli.** Free: craft runs 50 participants to estimate
one flip rate; 10 would do nearly as well and returns ~90% of that task's compute. Same argument
for story recall's 200 rows over 4 stories. No information loss.

**Option 3 — write more stimuli.** The only fix that raises the ceiling, and the only one needing
new material rather than code. Three craft tasks is very few for something carrying 1/8 weight.

**Option 4 — treat craft_task as a pass/fail check** rather than a graded axis, which is roughly
what two score values already make it.

## Recommendation

**Option 2 immediately** (it is free and uncontroversial), then **Option 1 or 4** as the real
decision, with **Option 3** as the actual fix if this task is meant to carry weight. I would not
leave it at 1/8 of the objective as-is.

**Cost:** Option 2 is a config edit. Option 1/4 changes the headline metric and every recorded
mean. Option 3 is authoring work outside this codebase.

---

# Suggested order

1. **Decision 4 Option 2** — free, do it regardless.
2. **Decision 1 Option A+C** — unblocks the one axis with headroom; an hour, no re-run.
3. **Decision 2 Option A + M7** — biggest effect on a published number; half a day, no re-run.
4. **Decision 3 Option B** — smallest stakes now that nothing is gated on it.

None needs GPU time. Everything is analysis-side except Decision 4 Option 2 (a config edit that
saves compute) and Option 3 (new stimuli).
