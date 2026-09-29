# Error-shape measures — definitions, units, human references, filters

**2026-09-29. REPORT ONLY.** Nothing here gates. No floors, no guards, no change to
`mean_humanlikeness_search`, and deliberately no entry in `score_candidate.axes()` —
anything added there lands in the record the proposer reads. Promotion is a later
decision, to be taken after one run gives a run-to-run spread per measure.

## Why these exist

`mean_humanlikeness_search` is `1 − W₁` between the model's and the humans'
per-participant score distributions, in units of task proportion-correct, averaged
over 8 tasks. It is **a scalar per task**: it measures how many items were right and
can say nothing about which ones or how. Two runs with identical score distributions
can fail in entirely different ways. The four existing axes cover three tasks (A1
digit span, A2 word recognition, A3 story recall) plus A4 on variable mapping; four
search tasks have no error-shape check at all. These six measures cover all eight,
plus the two held-out tasks.

**Every measure has its own unit and none of them is humanlikeness.** Where
per-participant values exist on both sides at matched granularity, the report also
prints `1 − W₁` over those values in that measure's own unit. Same functional form
as the acceptance metric, a different quantity, never averaged into the 8-task mean.

## Where the code lives

| what | where |
|---|---|
| M1 variable_mapping intrusion type | `meta_harness/interference.py` (beside A4) |
| M2 digit-span typology + serial position | `meta_harness/protocol_match.py` (beside A1) |
| M3 n-back, M4 word-recognition lag, M5 distractor, M6 gist | `meta_harness/error_structure.py` (beside A2/A3) |
| cached human references | `meta_harness/logs/human_error_shape.json` |
| the one CLI | `meta_harness/report_error_shape.py` |
| tests | `meta_harness/test_error_shape.py` |

```
python meta_harness/report_error_shape.py --rebuild-human-cache
python meta_harness/report_error_shape.py <run_dir>
python meta_harness/test_error_shape.py
```

No human reference is ever computed inline. `human_reference()` in
`report_error_shape.py` is the only producer; the report reads the JSON.
`test_error_shape.py` asserts that a fresh recomputation still equals all 453
cached reference fields, so a reference cannot drift silently — which is exactly how
A3 and A4 went wrong the first time.

---

## M1 — variable_mapping intrusion type

**Definition.** Classify each wrong answer by which city it picked, relative to the
assignment history visible at that question:

| collapsed class (A4's own names) | four-way refinement | meaning |
|---|---|---|
| `stale_same_name` | `own_stale` | a city previously assigned to *this* person |
| `intrusion_other_name` | `other_current` | the city currently bound to a *different* person |
| | `other_stale` | a city a different person held earlier but no longer holds |
| `novel_guess` | `novel` | a city never assigned to anyone |

The collapsed three-way keeps `_classify`'s exact names and precedence so it stays
commensurable with A4's `intrusion_share` / `stale_share`; the four-way is the
refinement A4 cannot express.

**Unit.** Share of that side's errors, in `[0,1]`; reference point is the side's own
**error total**, so the three collapsed classes sum to 1.000.

**Human reference** — 152 errors in 759 question trials over 154 records:

| class | human share | count | n (errors) |
|---|---|---|---|
| `stale_same_name` / `own_stale` | 0.2303 | 35 | 152 |
| `intrusion_other_name` | 0.4605 | 70 | 152 |
| — of which `other_current` | 0.4013 | 61 | 152 |
| — of which `other_stale` | 0.0592 | 9 | 152 |
| `novel_guess` / `novel` | 0.3092 | 47 | 152 |

These reproduce the three numbers already published in `interference.py`'s docstring
(23.0 / 46.1 / 30.9), which is the free drift check on this measure.

**Minimum n.** 30 errors, the same bar A4 uses for `trustworthy`. Below it the
measure prints `insufficient (n=k)` instead of a share.

**Filters, in order.** `runs/human/working-memory-variable-mapping/run-*.json` (154
records) → `payload.questions[]` with `selectedCity is not None` (759 trials) →
errors only (152). The classification window is the assignments with
`turn < question turn`; on the model side the window is
`turn <= TURNS_PER_QUESTION * question_index`, which is the fix already recorded in
`interference.py`.

**Pooled only, and this is forced by the protocol.** Every human
variable-mapping session **terminates at its first wrong answer**: all 152
participants with an error have exactly one, and it is always their final question
(verified: 152/152). A per-participant share is therefore one observation and a
`1 − W₁` over them would measure rounding. See *Findings* below — this also has a
consequence for A4.

---

## M2 — digit-span error typology and serial position

Both digit-span tasks have come out at exactly `+0.0000` humanlikeness delta for
every candidate measured so far, so the score metric is blind to them. If anything
about digit span discriminates, it is the shape of the wrong answers.

### M2a typology

**Definition.** Each administered error is put in exactly one class, most specific
first:

| class | rule |
|---|---|
| `reversal` | reverse span only: the **presented** order was reported verbatim |
| `transposition` | right digits, wrong order — same multiset, same length |
| `truncation` | a proper prefix of the target: stopped early |
| `omission` | a shorter subset of the target's digits, not a prefix |
| `substitution` | right length, at least one digit not in the target |
| `other` | anything else, including responses **longer** than the target. In the human records this is a mixed error: the length is wrong *and* the content is wrong (len-diff distribution forward: −3:2, −2:5, −1:15, +1:19). |

**Unit.** Share of that side's **administered errors**, six classes summing to
1.000. `reversal` has its own denominator: non-palindromic administered errors only,
because when `presented == reversed(presented)` (e.g. span-2 `33`) a forward-order
report is indistinguishable from a correct one and belongs in neither numerator nor
denominator.

**Human reference.**

| class | forward share | forward count | reverse share | reverse count |
|---|---|---|---|---|
| `reversal` | 0.0000 | 0 | 0.0237 | 4 |
| `transposition` | 0.1375 | 22 | 0.1598 | 27 |
| `truncation` | 0.0563 | 9 | 0.0888 | 15 |
| `omission` | 0.1375 | 22 | 0.1775 | 30 |
| `substitution` | 0.4125 | 66 | 0.3550 | 60 |
| `other` | 0.2562 | 41 | 0.1953 | 33 |
| **n (administered errors)** | **160** | | **169** | |
| n administered trials | 716 | | 578 | |
| n participants | 52 | | 49 | |
| `reversal` over non-palindromic errors | 0.0000 (n=160) | | 0.0240 (n=167) | |

**Minimum n.** 30 administered errors, pooled. Measured on the released
`meta_harness/runs/*` the matched staircase yields ~30 administered errors per model
run on forward and ~20 on reverse, so **reverse span will usually print
`insufficient`** — that is the honest outcome, not a reason to widen the filter. An
all-19-spans variant is printed alongside for a larger n and is labelled
**NOT human-comparable**, because those extra trials are ones no human was ever
administered.

### M2b serial position

**Definition.** Position `i` of the target counts as recalled if the response has
the right digit in place `i`. The curve is reported by **relative** third —
position `i` of a length-`L` target falls in bin `min(2, 3*i // L)` — because the
staircase administers different span lengths to different participants and pooling
absolute position 5 would mix "the last item of a 5-span" with "the middle of a
12-span". An absolute-position curve is reported alongside, pooled, as a descriptive
companion with its own per-position n.

**Unit.** Proportion of presented positions reported with the right digit in the
right place, in `[0,1]`; reference point is the trial's own length.

**Human reference.**

| third | forward | reverse |
|---|---|---|
| primacy (first third) | 0.9224 | 0.9020 |
| middle | 0.8489 | 0.7984 |
| recency (last third) | 0.7764 | 0.6930 |
| n participants | 52 | 49 |

Monotonically falling on both tasks — primacy without recency, which is what a
spoken-recall-in-order task should show.

**`1 − W₁` is computable here** (per-participant values exist on both sides) but is
**dominated by the model side**: the released runs carry 10 `sequence_index` values,
so the paired staircase yields **5** model pseudo-participants against 52 humans.
Note this contradicts `protocol_match.py`'s docstring, which says 100 sequences → 50
pseudo-participants; that count was true of `runs/compactor/`, not of
`meta_harness/runs/`.

**Filters.** `runs/human/working-memory-digit-span/run-*.json` (forward) or
`.../working-memory-reverse-digit-span/run-*.json` → `payload.trials[]` with
`length is not None` → **administered under the real staircase**: 2 trials per span
ascending from span 2, stop at the first span where both trials fail → participants
with a defined ceiling (`best_span > 0`). The staircase is `_administer`, the same
function A1 uses, so M2 and A1 see exactly the same trials; the test asserts A1's
human summary is bit-identical after that refactor (`sub_span_fail` 0.0866414835,
`best_span` 6.884615385).

---

## M3 — n-back miss / false alarm, n±1 lures, within-block position

**Definitions.** A *target* is a letter equal to the letter `n` positions back.

| quantity | definition | unit |
|---|---|---|
| `miss_rate` | P(respond "different" \| target) | proportion of answered target trials |
| `fa_rate` | P(respond "same" \| non-target) | proportion of answered non-target trials |
| `ratio` | `miss_rate / fa_rate`, reported the way A2 reports its ratio, with a bootstrap CI over (participant, level) cells | dimensionless; random responding → 1.0 |
| `unanswered_rate` | share of defined trials with no parseable label | proportion of defined trials |
| `lure_fa_rate` | `fa_rate` restricted to non-targets whose letter matches at lag `n−1` or `n+1` | proportion of answered lure trials |
| `position_third_k` | accuracy over answered trials in third `k` of the block's defined trials | proportion correct |

`unanswered_rate` is reported **separately and never folded into `miss_rate`**. The
model leaves a substantial fraction of trials unanswered; counting silence as a miss
would let a candidate move the ratio by going quiet.

**Human reference** — 1908 defined trials, 683 target / 1225 non-target, 159
(participant, level) cells:

| quantity | human | n | 
|---|---|---|
| `miss_rate` | 0.2533 | 683 target trials |
| `fa_rate` | 0.0816 | 1225 non-target trials |
| `ratio` | 3.1029, 95% bootstrap CI over cells **[2.411, 3.930]** | 159 cells |
| `unanswered_rate` | 0.0000 | 1908 defined trials |
| `lure_fa_rate` | 0.3158 | 76 lure trials |
| `nonlure_fa_rate` | 0.0661 | 1149 non-lure non-target trials |
| `position_third_1 / 2 / 3` | 0.8774 / 0.8428 / 0.8323 | 159 cells each |

Per level:

| n | `miss_rate` | `fa_rate` | `ratio` | `lure_fa_rate` (n lures) |
|---|---|---|---|---|
| 1 | 0.0916 | 0.0274 | 3.3446 | insufficient (n=18) |
| 2 | 0.2723 | 0.0752 | 3.6192 | insufficient (n=28) |
| 3 | 0.4279 | 0.1520 | 2.8150 | 0.4333 (n=30) |

The lure effect is large where it is measurable (0.4333 against a non-lure 0.1275 at
n=3) but **lures are rare by stimulus construction** — 76 of 1225 human non-target
trials — so `lure_fa_rate` is pooled-only and prints `insufficient` at most levels.

**Minimum n.** 30 lure trials for `lure_fa_rate`. The other quantities have their
full trial counts behind them.

**Filters, in order — this is the part that has been wrong before.**

1. `runs/human/working-memory-nback/run-*.json`, 57 records.
2. Drop `phase == "practice"` **and** any block named `training-*`.
3. Level is read from the **block name** (`"2-back"`), not `payload.trials[].level`,
   which is `null` in 4 of the 57 records.
4. **Drop the first `n` trials of every block on both sides** (lead-in). On the model
   side they are not in `per_trial` at all — they live in `buffer_letters` — so
   dropping them on the human side is what makes `position` mean the same thing:
   model scored trial `k` == human block trial `n+k`. This is not cosmetic; see
   *Findings*.
5. `target` is **recomputed** from the letter sequence on both sides rather than read
   from the log. Post-lead-in the recomputation agrees with the logged flag on
   1908/1908 human trials and 2100/2100 model trials, which is what licenses using
   it.
6. **Granularity: one cell per (participant, level).** Not pooled over a
   participant's three levels, and not per-row-over-all-levels. This is the same
   mismatch `nback_levels.py` exists to fix; a per-trial measure repeats it unless it
   is stated.
7. Lure = non-target whose letter matches at lag `n−1` or `n+1`, from the letter
   sequences on both sides.

Step-log indexing is not used by M3 — it reads `per_trial` and the letter arrays. Any
future step-log work must go through `meta_harness/nback_steps.py`.

---

## M4 — word-recognition lag curve  ⚠️ **the task is broken**

**Read this before using the number.** `word_recognition` does not isolate memory on
either side. All test words are presented in one visible list, and "Old" means
"this word appeared **earlier in this list**" (verified on both sides: 1828/1828
human trials and 5000/5000 model gold trials agree with that rule). A participant who
re-reads the list can answer correctly with no retention at all, so a lag curve here
measures **list inspection at least as much as memory decay**. It is computed because
it is the only lag structure the released data supports, and the report labels it
`task_is_broken: true`. A fix is planned in `logs/instrument_fix_stage2_plan.md`.

**Definition.** For each *old* trial, `lag` = trials elapsed since that word's
**first** appearance in the list. Bins: 1–2, 3–5, 6–10, 11–20, 21+.

**Unit.** Accuracy on old trials (proportion correct) within a lag bin, in `[0,1]`.

**Human reference** — 53 sessions:

| lag bin | human accuracy | n participants contributing |
|---|---|---|
| 1–2 | 0.5059 | 53 |
| 3–5 | 0.7975 | 51 |
| 6–10 | 0.9265 | 44 |
| 11–20 | 0.9504 | 40 |
| 21+ | 0.8878 | 27 |

The curve **rises** with lag over most of its range, which is the opposite of a
memory-decay curve and is itself evidence for the label above.

**Minimum n.** 10 participants contributing to a bin; below that the bin prints
`insufficient (n=k participants)`.

**Filters.** `runs/human/working-memory-word-recognition/run-*.json` (53 records) →
records with a non-empty `payload.responses` → old trials only. **Every bin is
computed per participant and then averaged, never pooled**, because human sessions
terminate at 3 strikes and run 4 to 102 trials (median 32) while the model runs 100,
so lag coverage differs per participant. `n_participants` per bin is reported because
a late bin is carried by the few long human sessions.

---

## M5 — which wrong option was picked

Two different measures, kept apart on purpose.

### M5a `distractor_choice` — narrative_qa, factual_qa (4 options)

**Definition.** Two probabilities, both with chance = 1/3 for three distractors:

* `within_side_agreement` = P(two errors drawn without replacement from the same
  question chose the **same** distractor)
  `= Σ_q Σ_d c_qd(c_qd − 1) / Σ_q C_q(C_q − 1)`.
  Computed this way rather than as a Herfindahl index because a plain Herfindahl is
  biased upward at small counts (with 2 errors its minimum is 0.5) and the per-question
  error counts here are 1–3.
* `cross_side_agreement` = P(a human error and a model error on the same question
  chose the same distractor) `= Σ_q Σ_d h_qd·m_qd / Σ_q H_q·M_q`. This is the "do the
  two sides concentrate on the same distractors" number. It has no "human value", so
  the report puts **chance** in the human column and leaves the distance column empty.

**Unit.** Probability in `[0,1]`; chance 0.3333.

**Human reference.**

| task | `within_side_agreement` | chance | n errors | n questions with ≥1 error | n questions with ≥2 errors | n ordered pairs |
|---|---|---|---|---|---|---|
| narrative_qa | 0.5172 | 0.3333 | 101 | 56 | 24 | 174 |
| factual_qa *(held out)* | 0.3916 | 0.3333 | 151 | 67 | 46 | 286 |

Humans concentrate above chance on narrative_qa (0.5172 vs 0.3333) and barely above
chance on factual_qa (0.3916 vs 0.3333).

**Minimum n.** 10 questions with ≥2 errors **and** 30 ordered pairs; below either,
`insufficient`.

**Filters.** `runs/human/narrative-qa/run-*.json` (or `factual-qa`) →
`payload.responses[]` whose question text is in the shared bank → errors only, with a
parseable chosen letter different from the bank answer.

* **narrative_qa has two banks.** 471 of the 520 human questions come from
  `data/narrative_QA.json` and 49 from `data/narrative_QA_easy.json`; the model only
  ever sees the former, so those 49 are **dropped**. factual_qa has no such split:
  all 530 human questions match `data/wikipedia_10docs_questions.json`.
* **Option letters are not shuffled.** The record's `correctAnswer` letter agrees
  with the bank's `answer` on 471/471 narrative and 530/530 factual questions, which
  is what makes letter-level comparison legitimate.
* **The model's chosen option is not persisted per question.** It is recovered by
  re-running that task's own `parse_answers` over `recall_raw`. Validated: the
  recovered choices reproduce the stored `metrics.correct` on 50/50 narrative_qa rows
  and 150/150 craft_task rows of `iter8repA/baseline`, asserted in
  `test_error_shape.py`. **Persisting the chosen letter is a one-line change in
  `bench/tasks/wm_*` and is worth making later**; it is deliberately not made here,
  since this work is evaluator-side only.

**Pooled only.** A human participant answers 10 questions and makes 1–3 errors, so a
per-participant distractor distribution does not exist and no `1 − W₁` is reported.

### M5b `error_index_profile` — craft_task, map_task (2 options)

**`distractor_choice` does not exist for these tasks.** They have exactly two
options, so picking the wrong one carries no information — distractor identity is
forced. Calling it a distractor measure there would be a category error. What the
data supports instead is **where in the 5-question block the error falls**.

**Definition / unit.** Error rate at each question index (proportion of that index's
questions answered wrong), per participant then averaged.

**Human reference.**

| question index | craft_task | map_task *(held out)* |
|---|---|---|
| 1 | 0.1173 | 0.1310 |
| 2 | 0.0432 | 0.2976 |
| 3 | 0.2407 | 0.3095 |
| 4 | 0.2840 | 0.3512 |
| 5 | 0.1173 | 0.2500 |
| n participants | 54 | 56 |

**Minimum n.** None beyond having participants at that index; the index n is printed.

**Filters.** `runs/human/procedure-memory-craft-task/run-*.json` (or
`procedure-memory-map-task`) → `payload.trials[].responses[]`, all blocks pooled
within a participant (3 blocks each), error rate per index, per participant then
averaged. `1 − W₁` **is** computable here (54/56 humans vs 50 model participants).

---

## M6 — story-recall gist similarity

Complements A3 rather than replacing it. A3 enforces that recall is **not** verbatim
(clipped 4-gram precision, human median 0.0192) and is roughly the human length
(137.2 words). Neither says whether the **content** survived.
`payload.evaluation.embeddingSimilarity` has been sitting unused in the human
records; this is that quantity, made comparable.

**Definition / unit.** Cosine similarity between mean-pooled `all-MiniLM-L6-v2`
embeddings of the story and the recall, both truncated to their first 200 words, in
`[0,1]` — the same embedder and truncation as
`bench.tasks.semantic_story_recall.embedding_similarity`.

**Human reference** — 53 usable records:

| column | human mean | sd | n |
|---|---|---|---|
| `recomputed` (the comparable one) | 0.5969 | 0.1305 | 53 |
| `stored` (web app, kept for continuity) | 0.6041 | — | 53 |

Per story, recomputed: Eyespy 0.6177 (n=18), Baseball 0.6068 (n=13), Pieman 0.6006
(n=15), Oregon Trail 0.5174 (n=7).

**Why recomputed.** The stored human number came from the web app, the model's from
bench. They are the **same embedder** — verified in `WORKLOG.md`: over these 53
records stored mean 0.6041 / sd 0.1282 against recomputed 0.5911 / sd 0.1220, mean
|diff| 0.0346, corr 0.9431 — but not the same preprocessing, so only the recomputed
column is computed identically on the two sides and only it is used for `1 − W₁`.
(The cache's 0.5969 differs slightly from WORKLOG's 0.5911 because this recomputation
uses `payload.storyFile`-resolved transcripts via `error_structure._transcript`; both
are recorded so the difference is visible rather than assumed away.)

**Minimum n.** None; the n is printed. Per-story means are printed because the story
mix differs — humans split 18 Eyespy / 15 Pieman / 13 Baseball / 7 Oregon Trail plus
3 records with no story, while the model runs 50 of each.

**Filters.** `runs/human/semantic-memory-story-recall/run-*.json` (56 records) →
records with a non-empty `payload.recallText` and a resolvable `payload.storyFile` in
`data/` (53) → both texts truncated to 200 words → mean-pooled MiniLM cosine.

---

## Which measures are worth keeping

Judgement, stated as judgement.

| measure | verdict |
|---|---|
| **M2a typology (forward)** | **Keep.** Well-defined classes, 160 human errors, and the one place a digit-span difference can show at all given the `+0.0000` humanlikeness delta. |
| **M2b serial position** | **Keep as descriptive.** A clean, monotone human curve on both tasks. But `1 − W₁` here rests on 5 model pseudo-participants, so the distance column is the usable part and the `1 − W₁` column is not. |
| **M3 miss/FA and position** | **Keep.** Large human sample (683 target / 1225 non-target trials), a CI over 159 cells, and `unanswered_rate` separates silence from error. |
| **M6 gist** | **Keep.** 53 humans, same embedder both sides, complements A3 along an axis A3 cannot see. |
| **M1 intrusion type** | **Keep, pooled, with the termination caveat below.** The classification is sound and reproduces A4's published numbers, but it is pooled-only on both sides and the human error set is 152 single terminal errors. |
| **M5b error-index profile** | **Keep as descriptive.** Cheap, 54/56 humans, per-participant on both sides. It is a *position* profile, not an error-shape signature; do not oversell it. |
| **M2a typology (reverse)** | **Marginal.** Prints `insufficient` at ~20 administered model errors per run. Keep the human reference; expect no model number until reverse span degrades. |
| **M5a distractor choice** | **Marginal, and noisier than it looks.** 101 human narrative errors spread over 56 questions is ~1.8 errors per question; only 24 questions have ≥2, and only 18 are shared with the model side. The `cross_side_agreement` number is the interesting one and it rests on 18 questions. Report it, do not act on it. |
| **M3 `lure_fa_rate`** | **Too thin to keep as a standalone.** 76 human lure trials in total, 18/28/30 by level, so it is `insufficient` at two of three levels. The effect is real where measurable but the stimulus generator does not produce enough lures. Either keep it as a pooled-over-levels footnote or change the stimuli. |
| **M4 lag curve** | **Keep only as evidence the task is broken.** The human curve *rises* with lag. It is not a memory measure and must never be read as one. |

---

## Findings that contradict assumptions

1. **Human variable_mapping terminates at the first error.** All 152 participants
   with an error have exactly one, and it is always their last question (152/152
   verified). Questions answered per participant: 2–16, mode 5. Two of the 154
   records answered nothing.
   *Consequence for A4, which is not changed here:* A4's human `rc_ratio` = 1.386 is
   `mean(relationCount | error) / mean(relationCount | correct)`, and on the human
   side the error trial is **by construction the participant's last and therefore
   highest-load trial**. Part of that 1.386 is the termination rule, not an
   interference effect. The model answers all 10 questions and its errors can occur
   anywhere, so the two sides' `rc_ratio` are not measuring the same thing. This
   deserves its own look before A4 is trusted as a *comparative* number; A4's
   normalised form partly but not fully addresses it.

2. **112 of the 318 human n-back lead-in trials are logged `target: true`.** There is
   no letter `n` positions back on those trials, so the label is unsatisfiable —
   14 of 53 participants have it on the very **first** trial of the 1-back block,
   where no predecessor exists at all. Post-lead-in the logged flag agrees with the
   recomputed one on 1908/1908 trials, so the defect is confined to the lead-in.
   Those trials are part of the human pooled n-back score, which means the human
   n-back reference the score metric compares against includes trials scored against
   an impossible label. M3 drops them; the score metric does not, and this was not
   changed.

3. **`protocol_match.py`'s docstring over-counts model pseudo-participants by 10×.**
   It describes 100 `sequence_index` values → 50 emulated participants. The
   `meta_harness/runs/*` rows carry 10, giving 5. The 100 figure was true of
   `runs/compactor/`. `error_structure.a1_model` groups by `sequence_index` alone,
   giving a third count (10). Nothing is wrong in the numbers A1 reports, but any
   per-participant statistic on the model digit-span side has n=5.

4. **narrative_qa humans saw two question banks.** 49 of 520 human questions come
   from `data/narrative_QA_easy.json`, which the model never sees. Anything comparing
   per-question behaviour on this task must restrict to the shared bank, as M5a does.

5. **`word_recognition`'s "Old" really does mean "earlier in this list"** on both
   sides — confirmed against all 1828 human and 5000 model gold trials. The task is
   broken in the way already recorded, and M4 is labelled accordingly.

---

## What was not done

* **No model numbers are written up as findings.** `bench/tasks/wm_nback.py` and
  `bench/tasks/wm_variable_mapping.py` changed in `70befa7` (pre-fix state tagged
  `exp/compactor-prefix-v1`) and the post-fix baseline is still running. The
  extractors were exercised on pre-fix runs (`iter8repA`, `iter8repB`, `iter0`) only
  to test that they work, and the report prints model columns for whatever run dir it
  is given.
* **Nothing is wired into `score_candidate.axes()`**, by instruction.
* **`bench/`, `data/`, `src/` and `runs/` were not modified.** The one-line change
  that would persist the model's chosen MCQ option is identified but not made.
