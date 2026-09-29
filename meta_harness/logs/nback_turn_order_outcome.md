# n-back turn order corrected — job 18774002, the check passed

> **ANNOTATION 2026-09-29 [TOOL] — every n-back humanlikeness in this file is on the
> PRE-OPTION-D SCORING SHAPE, and is deliberately NOT recomputed here.** This is a frozen
> record of what was observed at the time. On 2026-09-29 the user approved "Option D"
> (`logs/nback_denominator_decision.md`): n-back is now scored per `(participant, n-level)` on
> both sides with the human lead-in trials dropped. Re-scored under that shape, this run's
> three arms give n-back **0.9622** (mean over arms, spread 0.0060) in place of the **0.9344**
> in the table below, and the mean over 8 becomes **0.8862** in place of **0.8828**. No other
> task moves. The per-level *accuracy* table further down (human 0.9461 / 0.8615 / 0.7799)
> is the lead-in-**included** human reference at the 49-participant pool; under Option D the
> human reference is 0.9492 / 0.8553 / 0.7496 at 53 participants per level.
> `score.nback_human_scores_legacy_pooled()` reproduces this file's figures exactly, which is
> how the 0.9344 and the 0.0034 spread were re-verified rather than assumed.

Three Qwen baseline repeats, `runs/iter11postfix/baseline{,_rep2,_rep3}`, run from
`/scratch/cl5625/mh-postfix` at commit **410ec2a**. Job `18774002`, COMPLETED, 1h06m on one
H200. All three repeats hold all 8 tasks at full row counts (150/190/190/50/150/200/150/50).

## The hypothesis this run tested

Recapping, because the run was registered before it existed. The stage-1 instrument fix gave
n-back two turns per letter but ran them **encode-then-answer**, which overwrote the
comparison target: at n=1 the model keeps a single `previous_letter` key, the encode turn
replaced it with the current letter, and the answer turn was then asked about a letter that no
longer existed anywhere. `logs/postfix_baseline_outcome.md` recorded this as my defect, not a
finding about the model.

**Hypothesis.** The n=1 collapse was caused by the turn order alone, and flipping to
**answer-then-encode** (with the current letter shown, as it is on screen for a human) restores
it without costing the n=3 gain that the same fix produced.

**Evidence that would support it.** n=1 accuracy returns to roughly its pre-fix 0.9943, n=3
stays near its post-fix 0.5867 or better, and non-response stays low.
**Evidence that would reject it.** n=1 stays depressed — in which case the cause is elsewhere
and the store, not the order, is the problem — or n=3 falls back toward its pre-fix 0.3248,
meaning the two levels trade off and the order cannot fix both.

## Result: supported, and n=3 improved rather than held

Accuracy is proportion of the 14 scored trials correct (`acc_over_14`); not humanlikeness.

| level | human | pre-fix | defective (encode-first) | **now (answer-first)** |
|---|---|---|---|---|
| n=1 | 0.9461 | 0.9943 | 0.4786 | **0.9929** |
| n=2 | 0.8615 | 0.7938 | 0.6881 | **0.7786** |
| n=3 | 0.7799 | 0.3248 | 0.5867 | **0.7429** |

n=1 recovered to 0.9929. n=3 did not merely hold at 0.5867 — it rose to **0.7429**, against a
human 0.7799, from a pre-fix 0.3248. So the two levels did not trade off.

**Non-response is gone.** `answered` of 14 is now 13.94 / 13.98 / 13.96 with **`n_no_answers`
= 0 at every level**. Pre-fix n=3 answered only 6.29 of 14, and 418 of 472 unanswered turns had
returned empty assistant content. That failure mode does not appear in this run.

## The single-key symptom persists and is now harmless

48 of 50 n=1 blocks still end with at most one key (mean 1.04 keys, against 3.24 at n=2 and
3.86 at n=3). Pre-fix that was the mechanism of the collapse. It is now irrelevant, and the
reason is checkable rather than argued: **the answer turn precedes the encode turn in 150 of
150 rows**, so the single `previous_letter` key is read before it is overwritten. Verified
directly on the rows, along with the answer turn carrying both the current letter and the
restated instructions, and 15 encode + 15 answer turns per n=1 block (30 step-log entries, no
standalone instruction turn).

## Humanlikeness, all 8 tasks, 3 repeats

Humanlikeness = 1 − Wasserstein-1 between the model's and the humans' per-participant score
distributions; 0–1. The per-participant score is proportion-correct except on
`word_recognition` and `variable_mapping`, where it is **survival length** over a fixed scale
(100 words, 16 questions) — see `logs/survival_length_scoring.md`.

| task | pre-fix | post-fix (defective n-back) | **iter11postfix** | spread over 3 repeats |
|---|---|---|---|---|
| digit_span_forward | 0.8911 | 0.9012 | 0.9012 | 0.0000 |
| digit_span_reverse | 0.9666 | 0.9666 | 0.9666 | 0.0000 |
| **nback** | 0.7848 | 0.7123 | **0.9344** | 0.0034 |
| word_recognition | 0.5232 | 0.5073 | 0.5364 | 0.0199 |
| variable_mapping | 0.6809 | 0.9625 | 0.9643 | 0.0033 |
| narrative_qa | 0.9426 | 0.9509 | 0.9444 | 0.0184 |
| semantic_story_recall | 0.9454 | 0.9472 | 0.9470 | 0.0115 |
| craft_task | 0.8674 | 0.8596 | 0.8679 | 0.0357 |
| **mean over 8** | 0.8253 | 0.8509 | **0.8828** | |

All three columns are scored with the CURRENT scorer (survival length, `variable_mapping`
scale pinned at 16), so they are comparable to each other. They are **not** comparable to any
figure quoted before `a3d0bc3`.

### Precondition: passed

Only `bench/tasks/wm_nback.py` changed between `iter10postfix` and `iter11postfix`. Every other
task must therefore sit inside its own run-to-run spread:

| task | iter10 → iter11 | spread over 3 repeats | verdict |
|---|---|---|---|
| digit_span_forward | +0.0000 | 0.0000 | inside |
| digit_span_reverse | +0.0000 | 0.0000 | inside |
| word_recognition | +0.0291 | 0.0199 / 0.0349 | inside the larger |
| variable_mapping | +0.0018 | 0.0033 | inside |
| narrative_qa | −0.0065 | 0.0184 | inside |
| semantic_story_recall | −0.0002 | 0.0115 | inside |
| craft_task | +0.0083 | 0.0357 | inside |

`word_recognition` moves most, and both runs' own spreads (0.0199 here, 0.0349 in
`iter10postfix`) exceed the movement. Nothing points at leakage from the n-back change.

## Re-measured run-to-run noise, from these 3 repeats

This supersedes `run_to_run_floor.json` (measured 2026-09-25, pre-instrument-fix) for every
task, and is the first noise estimate taken on the current instrument. Spread = max − min over
the three repeats, in humanlikeness units.

| task | new spread | `run_to_run_floor.json` | `SAME_FAMILY_SD` |
|---|---|---|---|
| digit_span_forward | 0.0000 | 0.0152 | — |
| digit_span_reverse | 0.0000 | 0.0000 | — |
| nback | 0.0034 | — | — |
| word_recognition | 0.0199 | 0.0000 | 0.0180 |
| variable_mapping | 0.0033 | — | — |
| narrative_qa | 0.0184 | 0.0065–0.0160 | — |
| semantic_story_recall | 0.0115 | 0.0012 | — |
| craft_task | 0.0357 | 0.0000–0.0031 | 0.0158 |

**This resolves the craft_task disagreement, against the narrow estimate.** The two available
figures were 0.0000–0.0031 and 0.0158; the observed spread over three identical repeats is
**0.0357** (0.8907 / 0.8581 / 0.8550). The narrow band was wrong and the `−0.0078` craft
movement flagged in `logs/postfix_baseline_outcome.md` was well inside noise after all. Any
craft_task effect below about **0.036** is not measurable with three repeats.

`semantic_story_recall` also moves ten times more than its recorded 0.0012.

**Consequence for the floors:** three repeats give a standard error of roughly spread/3 at best,
so craft_task and `word_recognition` cannot support claims below ~0.02 on this budget, and no
task's floor should be set tighter than its spread here.

## What this run does not settle

- `word_recognition` is still the **old** presentation (all 100 words shown at once). The
  stage-2 rewrite is committed locally but was not on the cluster for this run.
- `variable_mapping` still asked **10** questions here, not 20. Its 0.9643 is transitional.
- Both digit spans still ran at `sequences_per_span: 10`, i.e. 5 matched pseudo-participants.
- n-back's own remaining mismatches are untouched and analysis-side (audit M6, M7, M9, M21):
  the human denominator includes the 6 lead-in trials, the human side is pooled per participant
  rather than per (participant, level), and human non-response is 0% by construction while
  `acc_over_14` scores non-response as error. M21 mattered at 0.157 proportion-correct pre-fix;
  with `n_no_answers` = 0 at every level it is now near-moot on the model side.
  **AMENDED 2026-09-29: M6, M7 and M9 are now FIXED in `src/score.py` (Option D); M21 stays
  documented-only, and is moot on this instrument for the reason just given.**
