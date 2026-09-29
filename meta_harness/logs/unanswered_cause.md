# What actually leaves an n-back trial unanswered

Measured 2026-09-28 with `meta_harness/analyze_unanswered_cause.py` on the post-fix Qwen runs
`iter9postA` / `iter9postB`, plus the pre-fix baselines. This replaces the account that died
the same day (see the OUTCOME section of `postfix_prediction.md`).

**Units.** Everything in this file is a **count or proportion of scored n-back trials**. No
number here is a humanlikeness score and no number here is a delta. "0.230 of trials
unanswered" means 23.0% of scored trials had no parsed label. For reference, the humanlikeness
figures live in `postfix_prediction.md`; humanlikeness = 1 − Wasserstein-1, range 0–1, in units
of task proportion-correct.

## The dead account

Standing story, and the one HANDOFF.md files **route 3** under: the tool-call budget runs dry →
the model types the tool call out as text instead of answering → that text carries no label →
trial recorded unanswered.

The middle link is false. Post-fix, spoken tool calls on n-back answer turns went **553 of 2100
→ 0 of 2100** while trials answered went **10.86 → 10.85 of 14**. And the cumulative-budget
link is false too:

| generation | scored trials | unanswered | of unanswered, `budget_before <= 0` |
|---|---|---|---|
| pre-fix baseline (iter0 + iter8repA/B) | 4200 | 862 (0.205) | **0 (0.000)** |
| post-fix baseline (iter9postA/B) | 4200 | 965 (0.230) | **6 (0.006)** |

The cumulative tool-call budget explains **6 of 965** unanswered trials. Route 3 as written —
reshape `_tool_call_cap()`'s early-turn allowance — cannot recover them, so an arm spent on it
would have been wasted.

## The live account: the model spends the turn on bookkeeping and never emits a label

Per-turn breakdown of the post-fix baseline's unanswered trials:

| run | scored trials | unanswered | of unanswered: made tool calls | of those, assistant content empty | made no calls |
|---|---|---|---|---|---|
| iter9postA/baseline | 2100 | 472 (0.225) | 469 (0.994) | 418 | 3 |
| iter9postB/baseline | 2100 | 493 (0.235) | 490 (0.994) | 432 | 3 |

Two facts pin the mechanism:

1. **Every unanswered trial's own answer turn had `tool_call_cap_hit` set** — 964 of 965
   (0.999) — and **no turn without it was ever unanswered** (0 of 2339). Perfect separation.
   `tool_call_cap_hit` is not the cumulative budget running out; it fires when the turn spends
   its remaining per-turn allowance, which happens at `budget_before = 2` with 2 calls.
2. Tool calls alone do not block an answer: **1598 and 1589 answered turns also made tool
   calls.** What distinguishes the unanswered ones is the *content*: 418 of 472 and 432 of 493
   returned **empty assistant content** — a pure tool-call turn — and the small remainder
   returned prose about the write instead of a label, e.g. `"I'll overwrite the oldest key,
   \`position_2\`, since it's no longer needed for the"` (truncated mid-sentence at the token
   cap).

It is load-dependent, which is what makes it look like a memory effect:

| n level | post-fix baseline unanswered / trials |
|---|---|
| n=1 | 0 / 1400 (0.000) |
| n=2 | 376 / 1400 (0.269) |
| n=3 | 589 / 1400 (0.421) |

## Why this matters for the direction of the search

- The model's n-back accuracy counting unanswered as wrong is **0.694**; humans are **0.8657**.
  So an unanswered trial pushes the model **below** the human distribution, not toward it. The
  old "an uncollected trial looks like humanlikeness" framing had the sign wrong for this task.
- `respond_first` reaches **14 of 14 answered** and accuracy ≈ 0.86 because its response
  obligation makes the label the first thing emitted, before any bookkeeping. That is why its
  n-back gain (+0.1755 in humanlikeness units, against the 3-run pre-fix baseline) is real.
- `respond_first_v2` (post-fix, schemas hidden during the answer act) also reaches **14 of 14**
  — `tool_call_cap_hit` on 1748 trials, 0 unanswered — yet its accuracy **falls to 0.588**. So
  collecting the answer and getting it right are separable, and what v2 lost is accuracy, not
  collection.
- Route 3 is dead. Routes 1 and 2 are untouched: they have independent measured effects
  (variable_mapping unparsed 38 → 0; answered 5.2–6.8 → 14.0 of 14).

## The bench question this raises

The benchmark accepts a turn with tool calls and no content as a non-answer and moves on. A
human participant is not allowed to skip a trial because they were busy rehearsing. Either the
benchmark should require content on an answer turn (re-prompt once, which changes every n-back
number it has produced), or "unanswered" has to be treated as a measured behaviour of the
harness rather than a property of the model. This is a decision for the user, not the
proposers, and it is larger than any single candidate: it decides whether n-back's headroom on
Qwen is real signal or a bench artifact of a different kind than the one ruled out today.
