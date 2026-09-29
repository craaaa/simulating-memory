# Post-fix baseline (job 18767306) — the precondition passed, and n-back has a defect I introduced

Three Qwen baseline repeats, `runs/iter10postfix/baseline{,_rep2,_rep3}`, run from the
`/scratch/cl5625/mh-postfix` worktree at commit 38a89a6. Predictions were registered in
`logs/instrument_fix.md` before the run existed.

**Units.** Humanlikeness = 1 − Wasserstein-1 between the model's and the humans'
per-participant score distributions; range 0–1, in units of task proportion-correct. "Accuracy"
and "answered" are raw counts/proportions of trials, not humanlikeness. A delta is a difference
of humanlikeness values.

## Scores, pre-fix vs post-fix baseline, 3 repeats each

| task | pre-fix | post-fix | delta | post per-run |
|---|---|---|---|---|
| digit_span_forward | 0.8911 | 0.9012 | +0.0101 | 0.9012 0.9012 0.9012 |
| digit_span_reverse | 0.9666 | 0.9666 | +0.0000 | identical |
| **nback** | 0.7848 | **0.7123** | **−0.0725** | 0.7197 0.7040 0.7131 |
| word_recognition | 0.5075 | 0.4918 | −0.0158 | 0.4795 0.4799 0.5159 |
| **variable_mapping** | 0.3539 | **0.6801** | **+0.3262** | 0.6784 0.6810 0.6810 |
| narrative_qa | 0.9426 | 0.9509 | +0.0083 | 0.9572 0.9447 0.9507 |
| semantic_story_recall | 0.9454 | 0.9472 | +0.0018 | 0.9474 0.9476 0.9466 |
| craft_task | 0.8674 | 0.8596 | −0.0078 | 0.8643 0.8581 0.8565 |
| **mean over 8** | 0.7824 | **0.8137** | **+0.0313** | |

## Prediction 5, the precondition: PASSED

The five intact batch tasks (`digit_span_forward`, `digit_span_reverse`,
`semantic_story_recall`, `narrative_qa`, `craft_task`) had to stay inside their measured
run-to-run noise, or the reset reached somewhere it should not have and predictions 1–4 are void.

| task | delta | noise band (`run_to_run_floor.json`) | verdict |
|---|---|---|---|
| digit_span_forward | +0.0101 | 0.0152 | inside |
| digit_span_reverse | +0.0000 | 0.0000 | inside |
| narrative_qa | +0.0083 | 0.0065–0.0160 | inside |
| semantic_story_recall | +0.0018 | 0.0012 | marginal, +0.0006 over |
| craft_task | −0.0078 | 0.0000–0.0031 | over the narrow band |

craft is the only real question, and the two available noise estimates disagree about it:
`run_to_run_floor.json` gives 0.0000–0.0031 while `score_repeats.SAME_FAMILY_SD` gives **0.0158**
for craft, measured over ten identical-code and provable-no-op runs that ranged 0.8534–0.8907.
−0.0078 is well inside the larger estimate. The batch tasks do not call `step()` more than once
and `recall()` never reads the message history, so there is no mechanism by which the reset could
reach them; the test in `test_turn_boundary_reset.py` asserts their request sequence directly.
**Treating the precondition as passed, with craft flagged and the noise-estimate disagreement
recorded as the thing to resolve.**

## Prediction 1, variable_mapping: substantively right, threshold wrong

Predicted ≥ 0.85; measured **0.6801**. But the threshold was wrong, not the claim. The prediction
said "rises into the band the candidates reached", and the Qwen candidates reached
**0.6885–0.6941** (`respond_first` 0.6941, `respond_first_v2` 0.6885). The post-fix baseline at
0.6801 is within 0.014 of them. I took 0.85 from the *Hermes* figure (0.9533), which is a
different model. **The substantive prediction holds: closing the leak is now a baseline property,
and the candidates' variable_mapping advantage is gone as expected.**

## Predictions 2 and 3, n-back: NOT INTERPRETABLE — the fix has a defect

| quantity | pre-fix baseline | post-fix baseline | predicted |
|---|---|---|---|
| mean accuracy (proportion of 14 correct) | 0.7043 | 0.5844 | < 0.65 |
| mean answered (of 14) | 11.20 | 12.48 | ≥ 13.0 |

Both look roughly as predicted in aggregate, and the aggregate is misleading. Per level:

| level | accuracy pre → post | answered pre → post |
|---|---|---|
| n=1 | 0.9943 → **0.4786** | 13.99 → 10.64 |
| n=2 | 0.7938 → 0.6881 | 13.32 → 13.51 |
| n=3 | 0.3248 → **0.5867** | 6.29 → 13.29 |

n=3 improved substantially. **n=1 collapsed, and 1-back with a readable store should be
trivial.** The cause is the turn order I chose, visible in a single block:

```
letter 1   ENCODE store: (memory is empty)      writes previous_letter=G
           ANSWER store: previous_letter: G     -> "no response"   (lead-in, correct)
letter 2   ENCODE store: previous_letter: G     writes previous_letter=R   <- overwrites G
           ANSWER store: previous_letter: R     -> "no response"   WRONG (expected Different)
letter 3   ENCODE store: previous_letter: R     writes previous_letter=R
           ANSWER store: previous_letter: R     -> "no response"   WRONG (expected Same)
letter 4   ENCODE store: previous_letter: R     writes previous_letter=Y
           ANSWER store: previous_letter: Y     -> "different"     correct by luck
```

The encode turn runs **before** the answer turn and overwrites the one thing the answer needs.
At n=1 the model keeps a single key and the letter one position back is destroyed before it is
ever asked about: **149 of 150 n=1 blocks end with ≤1 key** (mean 1.01), against mean 3.11 at
n=2 and 3.86 at n=3, where positional keys let history survive. So the n=1 comparison is
literally unanswerable, and "no response" is a reasonable thing for the model to emit.

**This is my error in the stage-1 fix, not a finding about the model.** My stated reason for
hiding the stimulus on the answer turn was that "the current letter must have been written to
the store for the comparison to be possible". That is wrong: the current letter is the
*stimulus*, and a human participant **sees it on screen** while judging. Hiding it was never
faithful to the human protocol.

## The correction

Flip n-back to **answer-then-encode**, the order I had already argued for in
`logs/instrument_fix_stage2_plan.md` for `word_recognition`:

```
answer turn (allow_tools=False)          encode turn (allow_tools=True)
  store contents                           store contents
  Original task instructions: ...          New letter (position P): X
  Letters presented so far: P              Update your memory as needed.
  Next letter: X
  Does it match the letter n back?
```

Then the store at answer time holds the letter *n* back, the current letter is present as the
stimulus, and nothing has been overwritten. `variable_mapping` does **not** need this change:
its question asks about stored content ("Where does X live?"), not about the stimulus just
presented, so writing before answering is correct there.

Worth recording: `respond_first`, the best-scoring candidate in the project, is precisely the
one that answered before writing. I built the instrument to do the opposite of the thing the
search had already found.

## Status of these numbers

`variable_mapping` (+0.3262), the five batch tasks, and the mean-over-8 are usable.
**The n-back column is not a clean measurement and must not be quoted as one** — it is an
instrument with a known defect at n=1. Re-run needed after the turn-order correction; the three
repeats cost about 68 minutes wall-clock on one H200.
