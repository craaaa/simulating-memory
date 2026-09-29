# Documentation audit, 2026-09-29 — what is stale and when it can be fixed

Written because three changes in two days (the collection fix, the instrument fix, and the
live-dimensions finding) invalidated claims spread across nine files, and two of those claims
were load-bearing enough to make a future proposer repeat work that four candidates already did.

Group 1 is **done** (commit below). Groups 2 and 3 are pending on work that has not landed.

## Group 1 — done 2026-09-29

| file | what was wrong | what it says now |
|---|---|---|
| `PROPOSER.md` §"Two of the eight tasks…" | "`reset_messages()` is never called by any task" — **false** since 70befa7 | retitled to "One of the eight tasks"; per-regime status table; explains n-back had no read channel at all; states that pre-tag numbers are not comparable |
| `PROPOSER.md` §"What you may change" | listed surfaces bench now owns | new subsection naming the turn-boundary reset, the store read channel and the tool-schema fix as **no-ops** for a candidate, and saying the 17 candidate `MANIFEST.md` files are historical and superseded by it |
| `PROPOSER.md` L339 | "three of eight tasks bypass the memory module" | now one; says the `word_recognition` fix is a bench change, not a candidate |
| `HANDOFF.md` §"Where it stands" | no mention that two tasks carried every delta | amended with the 99–106% figure and the tag boundary |
| `HANDOFF.md` §"The one thing to do next" | "**fix route 3**" — dead | struck through with the retraction and the replacement account inline |
| `HANDOFF.md` defect 2 | flat "three of eight" | per-leak status: two FIXED, `word_recognition` STILL OPEN with its fix sketched |
| `HANDOFF.md` §"Practical notes" | frontier presented as current | frontier declared history, belonging to `exp/compactor-prefix-v1`; warns that candidates carry their own `step()` and now close the leak twice |
| `HANDOFF.md` §"What to distrust" | five overclaims listed | three more added, with the observation that four of the five things that moved a verdict in this project were measurement defects |
| `logs/bench_collection_fix.md` | oversold a fix that changes no score | amendment at the top withdrawing its "what it invalidates" claims |

## Group 2 — DONE 2026-09-29, except one item deliberately left

All items below are applied. Where they are done:

| item | where |
|---|---|
| `domain_spec.md` axis table | A1 struck through and marked RETIRED with the null-model reasoning; A4's human column struck; new **"Error-shape measures — REPORT ONLY"** table added with all eight measures, their units and their human references |
| `domain_spec.md` `frontier` | amended — it omitted A4, and is now also blind to A1's retirement and the eight new measures; recorded as a log rather than a live Pareto set |
| `domain_spec.md` acceptance | stated in the new table: promotion to guard or objective is a separate decision and needs a measured run-to-run spread **for the measure itself**, which none of them has yet |
| `NOTES.md` A2 conservative-bias note | amended (trials-attempted is now the score, not a covariate) and the n-back miss/FA analogue added beside it — **the sign is reversed between the two tasks**, which no single response-bias story covers |
| `HANDOFF.md` A4 table + durable result | "dist from human" column marked WITHDRAWN on every row, the "closest match to human interference structure" sentence retracted in place, model-side columns explicitly kept |
| `HANDOFF.md` noise floors | amended with the re-measured spreads; craft 0.0357 against the 0.0000–0.0031 recorded there |
| `WORKLOG.md` | one dated amendment appended (append-only file), covering all four reversals with the affected line numbers listed |
| `domain_spec.md` A4 row | relabelled one-sided and model-internal; human column struck rather than updated |

**~~Left undone on purpose~~ — DONE 2026-09-29 [USER], as part of "Option D".** The denominator
fix (M6) landed together with the granularity fix (M7) in `src/score.py`; see
`logs/nback_denominator_decision.md`. The condition set out below was met: it was done against a
settled baseline (`iter11postfix` and `iter12stage2`, three arms each), on its own, with the
before/after stated (n-back humanlikeness 0.9344 → 0.9622 and 0.9340 → 0.9633). One correction to
the framing below: the 0.0088 is a shift in the human **accuracy** reference, and M6's effect on
**humanlikeness** is +0.0006, inside the 0.0061 run-to-run spread — M7 supplied +0.0287 of the
+0.0293 total. The original note follows.

**Left undone on purpose: the human n-back denominator** (audit M6, ~0.0088 score units). It is
a real analysis-side correction — human scores divide by all non-practice trials including the 6
lead-in, the model's `acc_over_14` does not, and 112 of 318 human lead-in trials carry an
impossible `target: true`. It moves the human reference 0.8657 → 0.8569 proportion-correct.
Not applied because it changes the published n-back humanlikeness in the same hours as the
turn-order fix, and separating the two effects matters more than the 0.0088. Do it against a
settled baseline, on its own, with the before/after stated.

## ~~Group 2 — pending the error-shape work (agent running 2026-09-29)~~ — original list

| file | line | change needed |
|---|---|---|
| `domain_spec.md` | ~246 | the axis table lists A1–A4 only. Add the new error-shape measures as **report-only**, each with its own unit and human reference. None of them is humanlikeness. |
| `domain_spec.md` | ~446 | `frontier` is documented as over (humanlikeness, A2, A3) — omits A4 and the new measures |
| `domain_spec.md` | acceptance section | state explicitly that error-shape measures report and do **not** gate, and that promotion to guard or objective is a separate decision requiring a measured run-to-run spread per measure |
| `NOTES.md` | ~73 | the A2 conservative-bias note ("humans are conservative") should sit beside the n-back miss/FA analogue once measured |
| `HANDOFF.md` | A4 table + "durable result" paragraph | **A4's human reference is void** — `logs/a4_human_reference_invalid.md`. The "closest structural match anywhere in this project" (rc_norm 0.3751 vs human 0.3728 over 845 errors) compares the model's error distribution against the human task's *termination rule*. Strike the comparison; keep the `variable_mapping` humanlikeness gain, which is unaffected. |
| `WORKLOG.md` | A4 sections | same, by dated amendment only |
| `domain_spec.md` | A4 row | relabel as a one-sided guard on the model's own error distribution; strike the human column rather than updating it |
| `src/score.py` / analysis | human n-back denominator | **DONE 2026-09-29 [USER] as "Option D"**, together with the granularity fix (M7). human scores divide by all non-practice trials including lead-in, the model's `acc_over_14` does not; and 112 of 318 human lead-in trials carry an impossible `target: true`. Correcting it moves the human reference 0.8657 → 0.8569 proportion-correct. Small, real, analysis-side. |

## Group 3 — DONE 2026-09-29, the fix landed the same day

| item | where |
|---|---|
| `domain_spec.md` premise 3 | marked SUPERSEDED with the date, kept as the record of why the change was made; states explicitly that whether A2 becomes a clean target is **open**, pending job 18781213's prediction P3 |
| `PROPOSER.md` H2 withdrawal + "highest-value target" | amended: all three leaky tasks are now fixed in `bench`, do not propose any of them. Also corrects two of my own numbers there — the unsourced 0.5199 (the measured value is 0.5075) and the claim that 0.315 is a human accuracy, which it is not |
| `PROPOSER.md` n-back per-level table | superseded with the post-fix table; n=3 0.360 → 0.7429 and `n_no_answers` 0 at every level, so the "store saturates and the model stops answering" diagnosis described a harness with no way to answer |
| `HANDOFF.md` noise floors | done under group 2 above |
| `logs/instrument_fix.md` deferred list | `word_recognition` moved from deferred to DONE, with the two things the plan did not anticipate, and the pre-registered outcome explicitly marked NOT IN YET pending job 18781213 |

Also resolved there: the deferred **empty-content answer turn** item. It disappeared structurally
as hoped — `n_no_answers` is 0 at every n-level, against 418 of 472 unanswered turns returning
empty content pre-fix. No separate fix was needed.

## ~~Group 3 — pending the `word_recognition` fix~~ — original list

| file | line | change needed |
|---|---|---|
| `domain_spec.md` | 24–32 | premise 3, "A2 is the axis — badly weakened", becomes historical once the leak closes. Do not delete it; mark it superseded with the date, per the project's supersession rule. |
| `PROPOSER.md` | ~241, ~330–341 | the H2 withdrawal and "closing the word-recognition leak is the highest-value target remaining" |
| `HANDOFF.md` | ~196 | noise floors from `run_to_run_floor.json`: `word_recognition` 0.0000 will not survive a task change and must be re-measured |
| `logs/instrument_fix.md` | deferred list | move `word_recognition` from "deferred" to done, with its measured outcome against the pre-registered expectation |

## Deliberately not rewritten

- **`WORKLOG.md`** (1414 lines) — append-only historical record of what was believed when. It
  gets dated amendments, never an edit in place. Five overclaims are already retracted in place
  there with the refuting data.
- **The 17 candidate `MANIFEST.md` files** — each cites `wm_agent.py:161` and describes the
  history leak as open, because each was written against the instrument of its own iteration.
  Rewriting them would falsify the record of what each proposer knew. `PROPOSER.md`'s new
  subsection supersedes them for anyone writing a new candidate.
- **`README.md`** (32 lines) and **`wave_prompt.md`** (72 lines) — checked, no stale claims;
  `wave_prompt.md` only references `history.py` invocations.

## The rule this audit exists to enforce

Supersede explicitly, never silently rewrite: date it, tag the provenance, and leave the old
claim visible with what refuted it. Every one of the nine items above was found by grepping for
a claim I knew had changed. The ones that cost the most were the two that read as ordinary
background — `PROPOSER.md`'s "`reset_messages()` is never called" and `HANDOFF.md`'s "the one
thing to do next" — because a reader has no reason to doubt either.
