# Error shape, first model-side numbers — they discriminate, and they contradict the scores

Run: `runs/iter11postfix/baseline/Qwen_Qwen3-30B-A3B-Instruct-2507` (job 18774002, the corrected
n-back turn order). Full report saved verbatim as `logs/error_shape_iter11postfix.txt`; human
references cached in `logs/human_error_shape.json`. The measures are **report-only** — nothing
here gates a candidate.

## What the labels mean

Six measures, one per shape of error, named M1–M6 by the agent that built them. **These are not
the audit's M1–M21 protocol-mismatch items** — the two numbering schemes collide and mean
different things. Here:

| label | task | what it measures |
|---|---|---|
| **M1** | variable_mapping | which *kind* of wrong city was named: the person's own **stale** city, **another person's** city, or a city never mentioned |
| **M2** | both digit spans | serial-position curve (primacy / middle / recency thirds) and error typology (reversal, transposition, truncation, omission, substitution) |
| **M3** | n-back | miss rate vs false-alarm rate, lure vs non-lure false alarms, accuracy by position third |
| **M4** | word_recognition | accuracy on Old trials by **lag** — how many words back the first presentation was |
| **M5** | narrative_qa, factual_qa | when two errors land on the same question, how often they pick the **same** distractor (chance = 0.3333) |
| **M5b** | craft_task, map_task | error rate at each question index 1–5 |
| **M6** | semantic_story_recall | gist similarity, recomputed identically on both sides |

**`distance` is |human − model| in the measure's own unit. It is a difference, not a score, and
it is not humanlikeness.** `1−W1` appears only where both sides have per-participant values at
matched granularity.

## The headline: score and shape disagree, on the same tasks, in the same run

| task | humanlikeness (score) | worst shape distance | reading |
|---|---|---|---|
| narrative_qa | 0.9444 | **0.4668** on M5 same-distractor | score looks human, errors do not |
| craft_task | 0.8679 | **0.2407** on M5b q3 error rate | score looks human, errors do not |
| word_recognition | 0.5364 | **0.4515** on M4 lag 1–2 | both flag it, for the same reason |
| nback | 0.9344 | **0.1073** on M3 miss rate, CIs disjoint | score looks human, one axis does not |
| semantic_story_recall | 0.9470 | 0.0343 on M6 gist | agree |
| variable_mapping | 0.9643 | **0.1682** on M1 stale-name intrusion | score looks human, mechanism differs |

This is the answer to "are we measuring the shape of errors, or only their rate" — and the
answer is that the rate was hiding the shape on four of six tasks.

## The five findings worth acting on

### 1. narrative_qa: the model's errors are near-deterministic

When two errors land on the same question, they choose the **same** distractor
**98.4%** of the time (model, n=500 error pairs) against **51.7%** for humans (n=174).
Chance is 33.3%.

A human population disagrees about which wrong answer is tempting; this model does not. That is
the temperature-0 / 50-seeded-stimulus-sets structure showing up as a measurable artefact — the
audit's M14 objection, which the user chose to continue with, made concrete. It costs nothing on
the score, which reads 0.9444.

### 2. craft_task: all 50 pseudo-participants fail the same question and only that question

| question index | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| human error rate | 0.1173 | 0.0432 | 0.2407 | 0.2840 | 0.1173 |
| **model error rate** | **0.0000** | **0.0000** | **0.0000** | **0.3333** | **0.0000** |

Exactly zero at four of five indices. The audit found the same degeneracy on hermes craft
(per-participant sd exactly 0.0000, all 50 at 13/15); this is the qwen version of it. The score
reads 0.8679 because the *mean* is roughly right while the *distribution* is a point mass.

### 3. n-back: the model misses targets; humans false-alarm

| | human | model |
|---|---|---|
| miss rate (targets called different) | 0.2533 | **0.3606** |
| false-alarm rate (non-targets called same) | 0.0816 | **0.0482** |
| miss/FA ratio | 3.10, 95% CI [2.41, 3.93] | **7.48, 95% CI [5.78, 10.21]** |

**The confidence intervals are disjoint**, over 683/1225 human and 750/1350 model trials, so
this is well-powered rather than suggestive. The model errs conservatively — it declines to call
a match — where humans over-call.

Two things this rules out as explanations: position is *not* the difference (accuracy by
position third matches to within 0.0375, `1−W1` 0.927–0.979), and non-response is not the
difference (model unanswered rate 0.0029 against a human 0.0000 by construction). Lure
sensitivity is close, if anything slightly higher in the model (0.3596 vs 0.3158).

### 4. word_recognition: the model has no lag effect at all — the stage-2 prediction

| lag bin | 1–2 | 3–5 | 6–10 | 11–20 | 21+ |
|---|---|---|---|---|---|
| human accuracy on Old trials | 0.5059 | 0.7975 | 0.9265 | 0.9504 | 0.8878 |
| **model** | **0.9574** | **0.9952** | **0.9953** | **0.9966** | **1.0000** |

Humans are *worst* at short lags — a word seen two items ago is hard to distinguish from one
never seen, in a continuous-recognition stream. The model is flat and near-ceiling everywhere.
This is exactly what reading the answers off the prompt looks like, and the report labels the
task BROKEN for that reason.

**Pre-registered prediction for the stage-2 re-baseline.** Under one-word-per-turn presentation,
a model whose judgements come from a 4-slot store should show a **rising** lag curve. Support:
lag 1–2 accuracy falls well below lag 11–20. Rejection: the curve stays flat near 1.0, which
would mean the score gap came from something other than list visibility. Note the human curve
is non-monotonic at 21+ (0.8878, on n=27) so the prediction is about the 1–2 vs 11–20 contrast,
not about matching the whole shape.

### 5. variable_mapping: humans suffer proactive interference, the model does not

| error class | human | model |
|---|---|---|
| **stale_same_name** — the person's own previous city | **0.2303** | **0.0621** |
| intrusion_other_name — a different person's city | 0.4605 | 0.5961 |
| novel_guess — a city never mentioned | 0.3092 | 0.3417 |

Humans re-report the *superseded* binding 23% of the time; the model does so 6%. This is the one
measure here that looks like a mechanism claim rather than a calibration gap, and it is the kind
of thing the never-tested Group B candidates (`displacement`, `random_decay`) were actually
about. Pooled only — 152 human errors against 515 model errors, one error per human record, so
there is no per-participant version.

## What decision D fixes, visible here

`M2_digit_span_forward_serial_position` reports **n_model = 5** against n_human = 52, and
`M2_digit_span_reverse_typology` is **withheld entirely** (model n=20 errors, minimum 30). That
is the 5-matched-pseudo-participant problem. `sequences_per_span` is now 40 (`4894c58`), giving
20, so both should become readable at the next re-baseline.

Where the forward typology *is* readable it already disagrees sharply: the model
**truncates (0.375) and omits (0.469)** where humans **substitute (0.413)**; distances
0.32–0.38. The model drops the tail of the sequence; humans report a wrong digit in the right
place.

## 6. story recall: the aggregate agrees, the per-story structure does not

Added 2026-09-29 while waiting on job 18781213. M6's gist similarity is the one measure above that
*agrees* (0.5626 model vs 0.5969 human, distance 0.0343). Broken out by story, it does not:

| story | human n | human mean | model n | model mean | gap |
|---|---|---|---|---|---|
| Eyespy | 18 | 0.6185 | 50 | 0.6259 | **+0.0074** |
| Pieman | 15 | 0.6128 | 50 | 0.6077 | −0.0051 |
| Baseball | 13 | 0.6114 | 50 | **0.5178** | **−0.0936** |
| Oregon Trail | 7 | 0.5344 | 50 | 0.4991 | −0.0353 |

**Humans are flat across stories and the model is not.** Three of four human means sit within
0.007 of each other (0.611–0.619); the model spans 0.499–0.626. Almost the entire model deficit is
one story, Baseball, at −0.0936 — and on Eyespy the model is slightly *better* than humans. So
"the model recalls gist a bit less well than humans" is really "the model recalls one of these
four stories much less well, and matches humans on the rest".

**Caveat, and it is a real limit here:** human n per story is 7–18, so these means carry wide
intervals and Oregon Trail's 7 is too thin to lean on. The Baseball gap is the only one large
enough to survive that, and it is the one the claim rests on.

### The M17 reweighting is real but immaterial

Audit **M17** is that the human story mix is uneven (Eyespy 18 / Pieman 15 / Baseball 13 / Oregon
Trail 7, n=53) while the model runs 50 of each, so the pooled distributions are not comparable.
Reweighting the model to the human proportions — analysis-side, no re-run — moves story-recall
humanlikeness **0.9398 → 0.9506, i.e. +0.0109**, and the model's mean 0.5626 → 0.5775 against a
human 0.6041.

`RUN_TO_RUN_SPREAD["semantic_story_recall"]` is **0.0115**, so **+0.0109 is inside noise** and
M17 is not worth changing the scorer for on its own. Recorded so nobody re-derives it as a
priority. The per-story breakdown above is the part worth having, and it needs no reweighting.

## Recommendation, not applied

The user's stated rule was that the spread of these numbers decides whether they get promoted
out of report-only. **The spread is wide** — distances run 0.0003 to 0.4668, four of six tasks
disagree with their own score, and two of the measures (M3's ratio, M5's agreement) have
non-overlapping confidence intervals or n in the hundreds.

Candidates for promotion to enforced guards, in order of how well powered they are:

1. **M3 n-back miss/FA ratio** — disjoint CIs, thousands of trials, and it replaces the retired
   A1 as the one axis with real signal on this model.
2. **M5 narrative_qa same-distractor agreement** — n=500 model error pairs, distance 0.4668.
3. **M5b craft_task error index** — catches the point-mass degeneracy that the score cannot see.

Against promoting M1, M4 and M6 yet: M4's task is mid-rewrite, M1 is pooled-only with no
per-participant version, M6 already agrees. **Not promoting anything without the user's word** —
A4 and A1 were both promoted on reasoning that later failed, and the rule now is that a guard
has to survive a null model first.
