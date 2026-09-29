# Three of the eight tasks have almost no stimuli, and craft_task has one bit of resolution

Measured 2026-09-29 on `runs/iter11postfix/baseline` while job 18781213 queued. This is an
instrument-design finding, not fixable analysis-side, and it caps what the objective can ever
detect on three tasks.

## Units

"Distinct stimuli" is a count of unique stimulus payloads across a task's rows. "Score values" is
the count of distinct per-row scores the task produced. Neither is humanlikeness.

## The table

A "participant" in this benchmark is a seeded stimulus set run by one deterministic model. So the
first question is whether each participant actually gets a *different* stimulus.

| task | rows | distinct stimuli | rows per stimulus | distinct score values |
|---|---|---|---|---|
| digit_span_forward | 190 | 190 | **1.0** | 2 (0.0 / 1.0 per sequence) |
| digit_span_reverse | 190 | 190 | **1.0** | 2 (0.0 / 1.0 per sequence) |
| nback | 150 | 150 | **1.0** | 8, range 0.50–1.00 |
| word_recognition | 50 | 50 | **1.0** | 14, range 0–100 |
| variable_mapping | 150 | 150 | **1.0** | 4 (4 / 6 / 8 / 10) |
| narrative_qa | 50 | **10** | 5.0 | 6 (0.5–1.0) |
| semantic_story_recall | 200 | **4** | **50.0** | 64, range 0.42–0.74 |
| **craft_task** | 150 | **3** | **50.0** | **2 (0.8 / 1.0)** |

Five tasks give every participant its own stimulus. Three do not, and for `craft_task` and
`semantic_story_recall` the "50 participants" are **50 reruns of the identical input**.

**The banks are exhausted, so this is not a configuration mistake.** `data/craft_task.json`
contains exactly **3** items (C2001–C2003) and `data/narrative_QA.json` exactly **10**; story recall
has 4 transcripts. The harness is already using all of them. There is no more material to draw on.

## craft_task is degenerate, and completely so

Every one of the 50 pseudo-participants produced the **identical** result:

```
C2001 = 1.0    C2002 = 1.0    C2003 = 0.8      <- 50 of 50 participants, one distinct signature
```

Zero between-participant variance. All 50 see byte-identical question sets for each task (verified
by hashing `questions`), so there is nothing for them to differ about except serving
nondeterminism. The error-shape report's "model error rate 0.0000 at question indices 1, 2, 3, 5
and 0.3333 at index 4" is this: **one question, on one stimulus, wrong for everyone.**

### This fully explains craft_task's run-to-run spread

I measured craft_task's 3-repeat spread as **0.0357** humanlikeness earlier today and wired it into
`score_candidate.RUN_TO_RUN_SPREAD`, resolving a disagreement between two narrower estimates. The
mechanism is now visible, and it is not sampling noise:

| repeat | distinct signatures | participants with C2003 wrong | mean accuracy |
|---|---|---|---|
| baseline | 1 | **50 of 50** | 0.9333 |
| baseline_rep2 | 2 | 29 of 50 | 0.9613 |
| baseline_rep3 | 2 | 27 of 50 | 0.9640 |

**The entire variability of craft_task is the flip rate of a single question on a single
stimulus.** Nothing else ever varies. The floor I set is correct as a floor, but it should be
understood as "how often does C2003 Q4 flip this run", not as measurement precision in the usual
sense.

This is also why the protocol audit found hermes craft with per-participant sd **exactly 0.0000**
and all 50 rows at 13/15: a model that never flips produces a single point.

## What this means for the objective

`craft_task` carries **1/8 of the mean humanlikeness** — the primary objective — while producing
**two distinct score values** on this substrate. It can register that one question flipping and
nothing else. A candidate that changed memory behaviour in any way that does not touch C2003 Q4
is invisible to it, and a candidate that happens to flip that question moves 1/8 of the objective.

`semantic_story_recall` is intermediate: only 4 stimuli, but 64 distinct score values across 200
rows, because the *recall text* varies run to run even on a fixed story. It has real resolution;
what it lacks is stimulus breadth, which is why its per-story gaps are so uneven (Baseball
−0.0936, Eyespy +0.0074 — see `logs/error_shape_first_model_numbers.md`).

`narrative_qa` sits in between at 10 stimuli, 5 participants each.

**This sharpens the M14 pseudo-participant objection the user chose to continue with.** For five
tasks, model between-participant spread really is item difficulty over distinct items, which is at
least a coherent quantity. For `craft_task` and `semantic_story_recall` it is *not even that* — it
is vLLM batch nondeterminism on a fixed input, compared against whatever produces the human spread
— which for craft is 8 distinct values at sd 0.1180 over 54 people, and is *not* fully
characterised, since the human records carry no task id. Continuing with the current procedure is a
defensible choice; this is what it costs, stated in numbers.

## This also settles audit M12, in the opposite direction from the audit's reading

**M12** is that the model is scored per 5-question trial while the human is scored per 15
questions, and the audit proposed pooling the model to the human's 15-question unit
(rated ±0.02 humanlikeness, "sign not constant across substrates").

**Pooling makes the degeneracy total.** The model's 15 questions per participant are exactly
C2001 + C2002 + C2003. Verified by pooling all three repeats:

| repeat | pooled-to-15 values | sd |
|---|---|---|
| baseline | **{0.9333}** — one value | **0.0000** |
| baseline_rep2 | {0.9333, 1.0} | 0.0329 |
| baseline_rep3 | {0.9333, 1.0} | 0.0332 |

In `baseline` all 50 participants land on 14/15 with **sd exactly 0.0000**, which is precisely the
state the audit flagged as pathological on hermes. The other two repeats escape it only because
C2003 Q4 flipped for some participants — i.e. the only thing standing between this task and a
degenerate point mass is serving nondeterminism.

At that same 15-question unit, the human side is a real distribution:

| | n | mean | sd | distinct values |
|---|---|---|---|---|
| human | 54 | 0.8395 | **0.1180** | **8** (0.40, 0.60, 0.667, 0.733, 0.80, 0.867, 0.933, 1.00) |
| model, `baseline` | 50 | 0.9333 | **0.0000** | **1** |

**A correction to something I asserted earlier in this file.** I wrote that human craft variation
is "between-person ability on fixed material". I cannot verify the "fixed material" half: the human
records carry no task identifier (`taskId` is absent on all 162 human trials), so which craft tasks
each person saw is not recoverable from them, and the protocol audit separately found that **5 of
54 human records used `craft_task_old.json` (v2.0)** rather than the current bank. So the human side
may carry some stimulus variation of its own, and possibly across two different banks. What is
verified is the shape: 8 distinct values at sd 0.1180 against the model's 1 at sd 0.0000.

So M12's fix is correct about the granularity mismatch and would make the resulting distribution
*less* usable, not more. The mismatch is real; pooling is not the remedy. The remedy is more
stimuli (option 3 below).

## Options, none taken

1. **Leave it and weight the mean by resolution**, or report craft_task separately rather than as
   1/8 of the objective. Cheapest, and honest, but changes the headline metric.
2. **Stop spending 150 rows on 3 stimuli.** craft currently runs 50 participants to learn one flip
   rate; 10 would estimate it nearly as well and free ~90% of that task's compute. Same argument
   for story recall's 200 rows over 4 stories. This is a pure cost saving with no information loss.
3. **Write more stimuli.** The only fix that raises the ceiling, and the only one that needs new
   material rather than a code change. 3 craft tasks is very few for a task carrying 1/8 weight.
4. **Accept craft_task as a pass/fail check** rather than a graded axis, which is roughly what its
   two score values already make it.

My recommendation is 2 immediately (free) and 1 or 4 as the decision, with 3 as the real fix if
this task is meant to carry weight. **Not applied** — all four change either the objective or the
run configuration, and both are the user's call.
