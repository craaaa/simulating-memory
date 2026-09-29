# Instrument fix stage 2 — `word_recognition`, planned 2026-09-29, NOT LANDED

Approved by the user as "plan now, land later". **Land after job `18767306` (the 3 post-fix
baseline repeats) has been measured**, so that baseline is not stale on arrival. Changing two
tasks between baselines is what made three of the twelve candidate verdicts un-comparable.

**Units.** Humanlikeness = 1 − Wasserstein-1 between the model's and the humans'
per-participant score distributions; range 0–1, in units of task proportion-correct. Raw task
scores below (e.g. "human mean 0.315") are **proportion correct**, not humanlikeness. A delta
is a difference of humanlikeness values in those units and is the only quantity that can be
negative.

## The defect, stated correctly

Two earlier descriptions of this circulated and the original one was right.
`HANDOFF.md` said "the studied list is re-printed in the recall prompt", which is literally
what happens:

```python
word_list_text = "\n".join(f"{t['trial_index']}: {t['word']}" for t in trials)      # :93
encoding_log   = agent.encode(word_list_text)                                       # :107
trials_text    = "\n".join(f"trial {t['trial_index']}: {t['word']}" for t in trials) # :110
```

Both `encode()` and the recall prompt receive the **same 100 trial lines**. Mid-session I
claimed this was a different mechanism from what `HANDOFF.md` recorded; that claim was wrong
and is withdrawn. The substantive fix is unchanged either way.

Continuous recognition is genuinely study-equals-test — a word is "Old" if it appeared earlier
in the same sequence — so **the task design is right and the presentation is wrong**. Two
independent problems:

1. **The whole sequence is visible at judgement time**, so Old/New is decidable from the
   prompt without consulting the store. Measured: 36 of 50 model participants score ≥ 0.98 and
   7 score ≤ 0.04; humans have 1 of 53 above 0.98 and a mean score of 0.315.
2. **The model runs 100 trials, humans about 20.** Humans stop at 3 strikes (`strikesUsed: 3`,
   `trialsCompleted: 20`). Exposure is unequal, and the strike rule is a *selection effect* —
   it stops a participant precisely when they are doing badly.

## The rewrite

Delete the `encode()` call. There is no study phase to encode. The task becomes N turns of one
word each in `step()` form, like `wm_nback` and `wm_variable_mapping` after stage 1.

**Turn order is answer-then-store, which is deliberately the opposite of n-back:**

```
answer turn  (allow_tools=False)        encode turn  (allow_tools=True)
  Your working memory currently           Your working memory currently
  contains: {wm_contents}                 contains: {wm_contents}

  Original task instructions: ...          trial 7: ANCHOR

  trial 7: ANCHOR                          Update your memory as needed.
  Has this word appeared earlier in
  this list? Answer Old or New.
```

Why the order differs: n-back hides the stimulus on its answer turn because the comparison
target is *n positions back*, so the current letter must have been written to the store for the
comparison to be possible. Here the judgement is about the word in front of the participant —
hiding it would test writing, not recognition. Answering before writing also means a tool call
can never crowd out the reply, which is the failure mode that cost the stage-1 baseline 3.15 of
14 n-back trials (418 of 472 unanswered turns returned empty assistant content). **Comment this
asymmetry in the code** so a later reader does not "fix" it into consistency with n-back.

## Prerequisite check, before writing any code

Confirm how `src/score.py` scores the **human** side of `word_recognition`. The human summary
is `correctResponses: 17, trialsCompleted: 20`, i.e. a proportion over *attempted*, with
attempted varying per participant. If the scorer divides by a fixed 100 on either side, then
replicating the strike rule silently changes the metric rather than matching it.

## Open decision that check feeds

| option | effect |
|---|---|
| **replicate the 3-strike stop in bench** (preferred, if the scorer handles a variable denominator) | reproduces the selection effect and matches the human protocol; run length becomes variable per participant |
| **run a fixed 100 and truncate to the first 20 analysis-side** | cheaper, fixed denominator, reuses the pattern `protocol_match.py` already applies to the digit-span staircase; cannot reproduce the selection effect |

## Expected consequence, pre-registered

- `word_recognition` humanlikeness is **0.5199** in the 3-run pre-fix baseline — the lowest of
  the eight tasks and therefore the largest headroom. It is low because the model scores near
  perfect while humans average 0.315 proportion correct.
- After the fix the model's **score** should fall and its **humanlikeness** should rise.
- **A2 becomes a real axis again.** `domain_spec.md` records it as "badly weakened" because
  A2's value is produced by the ~7 participants who consult the store while 43 read the list
  off the prompt. Closing this is what unweakens it.
- `run_to_run_floor.json` lists `word_recognition` noise as 0.0000; that figure will not
  survive the change and must be re-measured.

## Precedent worth noting

`serial_recognition` already prototyped masked one-at-a-time presentation, scoring 0.8134
against its deliberate unmasked ablation `serial_recognition_open` at 0.8631. That makes this
the **fourth** case of a candidate re-deriving a benchmark fix — after the turn-boundary reset
(four candidates), the store read channel, and the response obligation. The pattern is the
substance of the `live_dimensions.md` finding: a search over harness code finds instrument
defects because that is where the gradient is.
