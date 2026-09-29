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

## Group 2 — pending the error-shape work (agent running 2026-09-29)

| file | line | change needed |
|---|---|---|
| `domain_spec.md` | ~246 | the axis table lists A1–A4 only. Add the new error-shape measures as **report-only**, each with its own unit and human reference. None of them is humanlikeness. |
| `domain_spec.md` | ~446 | `frontier` is documented as over (humanlikeness, A2, A3) — omits A4 and the new measures |
| `domain_spec.md` | acceptance section | state explicitly that error-shape measures report and do **not** gate, and that promotion to guard or objective is a separate decision requiring a measured run-to-run spread per measure |
| `NOTES.md` | ~73 | the A2 conservative-bias note ("humans are conservative") should sit beside the n-back miss/FA analogue once measured |
| `HANDOFF.md` | A4 table + "durable result" paragraph | **A4's human reference is void** — `logs/a4_human_reference_invalid.md`. The "closest structural match anywhere in this project" (rc_norm 0.3751 vs human 0.3728 over 845 errors) compares the model's error distribution against the human task's *termination rule*. Strike the comparison; keep the `variable_mapping` humanlikeness gain, which is unaffected. |
| `WORKLOG.md` | A4 sections | same, by dated amendment only |
| `domain_spec.md` | A4 row | relabel as a one-sided guard on the model's own error distribution; strike the human column rather than updating it |
| `src/score.py` / analysis | human n-back denominator | human scores divide by all non-practice trials including lead-in, the model's `acc_over_14` does not; and 112 of 318 human lead-in trials carry an impossible `target: true`. Correcting it moves the human reference 0.8657 → 0.8569 proportion-correct. Small, real, analysis-side. |

## Group 3 — pending the `word_recognition` fix (`logs/instrument_fix_stage2_plan.md`)

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
