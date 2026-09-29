# Instrument fix stage 2 — `word_recognition`, planned and LANDED 2026-09-29

> **STATUS: code landed.** The plan below is kept as written; the sections at the END of this
> file record what the prerequisite check returned, which of the plan's claims it withdrew,
> and what was actually built. Two figures in the plan body are wrong and are corrected there:
> the pre-fix humanlikeness (0.5199 → **0.5075**) and the human trial count
> ("about 20" → mean **34.49**, median **32**).

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

---

# What happened when it landed, 2026-09-29

## 0. First, the question that gated it: is `word_recognition` touched by stage 1?

**No, on three independent grounds.** Stage 1 was the turn-boundary reset in `step()` plus the
`generate_with_tools` sanitizer.

1. The working-memory arm called `encode()` **exactly once** and then `recall()`. `encode()`
   runs one `step()`, so `reset_messages()` fires on an already-empty `_messages` — the
   request is byte-identical. `recall()` calls `llm.generate` directly and never reads
   `_messages` at all. The summarizer arm is further removed still: its `encode()` also calls
   `llm.generate` directly.
2. The sanitizer rewrites only surrogate pairs, NUL and control characters. The word lists are
   ASCII, so on this task it is a literal no-op.
3. The measured movement is inside noise. Pre-fix **0.5075** → post-fix **0.4918**, a delta of
   **−0.0158** humanlikeness. `score_repeats.SAME_FAMILY_SD` for this task is **0.0180**, and
   the three post-fix repeats themselves ranged 0.4795–0.5159, a spread of **0.0364** — over
   twice the delta. The pre-fix triple spread 0.0382.

So the re-baseline this change forces is a re-baseline of a number that stage 1 did not move.

## 1. The 0.5199 / 0.5075 discrepancy, resolved

The plan body says the pre-fix humanlikeness is 0.5199; `postfix_baseline_outcome.md` says
0.5075. **0.5075 is right** and 0.5199 is unsourced — recomputed with
`score_repeats.py --id prefix_baseline_3rep_audit` over the three pre-fix baseline runs
`iter0/baseline`, `iter8repA/baseline`, `iter8repB/baseline`, which reproduces every other
figure in that file's pre-fix column (per-run 0.4948 0.4948 0.5330, spread 0.0382).
**The pre-stage-2 anchor for `word_recognition` is 0.5075, over that named triple.**

## 2. The prerequisite check: the scorer uses a fixed denominator on BOTH sides

`src/score.py` has `TASK_DENOM["word_recognition"] = 100.0`, applied to the human side
(`correctResponses / 100.0`, line 174) and to the model side (`metrics.score / 100.0`,
line 240). The plan said that if this were so, "replicating the strike rule silently changes
the metric rather than matching it". That inference was wrong, for a reason the plan did not
anticipate:

**`score_game` already applies the 3-strike rule analysis-side.** `MAX_ERRORS_BEFORE_STOP = 3`,
and the scoring loop breaks at the third error, so `metrics.score` is already "words survived
minus 3". Verified on the post-fix baseline: `score == len(per_trial) − 3` for all 12 of 50
rows that reached three errors. Both sides were therefore *already* measuring the same
quantity, and it was never an accuracy.

Independently confirmed on the human side: `trialsCompleted − correctResponses == 3` for
**53 of 53** human records, mean `trialsCompleted` **34.49**, median **32**, min 4, max 102.
Mean human score `correctResponses/100` = **0.3149**. So the human "proportion correct" is
algebraically `(n_survived − 3)/100` and `correct/attempted` is identically `1 − 3/n`. There
is **no human accuracy in this dataset at all**, and no denominator choice creates one.
This is what the protocol-mismatch audit records as **M1** (`protocol_mismatch_audit.md`);
its severity rating of INVALIDATING is right about the human side and **wrong about the model
side**, where it states "model runs 100 so its score *is* a proportion". `score_game`
truncates. The residual mismatch is right-censoring, not a different metric.

**Consequence for the open decision.** Both options in the plan's table are void. Option 2
("truncate to the first 20 analysis-side") was built on a single example record and would
censor most humans — the median is 32, not 20. Option 1 ("replicate the 3-strike stop") needed
no scorer change because the rule was already applied post hoc. What was left to do was to
stop *presenting* trials a human would never have reached, which is a loop condition.

## 3. Pre-registered prediction, measured before the code was written

**Hypothesis.** The near-perfect model scores come from the presentation — the whole 100-word
stream is visible at judgement time — and not from the model's memory.

**Evidence that would support it.** Simulating the human 3-strike stop over the existing
post-fix rows, the model's third error falls at or beyond trial 100 for most participants,
against a human median of 32 trials completed.
**Evidence that would reject it.** The model's third error already falls near trial 32, in
which case the presentation is not what produces the gap and M13 buys much less than claimed.

**Measured** (`runs/iter10postfix/baseline`, 50 participants, condition C2): mean **1.2 errors
per 100 trials**; the 3-strike rule **never fires for 38 of 50**; median third-error position
**101** (censored), mean 85.6. Human median 32. **Hypothesis supported.** This also disposes of
the audit's objection that "the model errs too rarely for a 3-strike rule to bite" — that was
measured under the defective presentation, i.e. it is the pre-condition, not the post-condition.

## 4. What was built

`bench/tasks/wm_word_recognition.py`, working-memory arm only:

- `encode()` deleted. There is no study phase in continuous recognition.
- New `run_recognition_stream()`: one turn pair per word — ANSWER (store + restated
  instructions + trial index + the word, `allow_tools=False`) then ENCODE (store + the word,
  `allow_tools=True`). The store is injected via `wm.to_recall_text()` on both, as in
  `wm_nback` and `wm_variable_mapping`; `TOOLS` has no read tool.
- Presentation **stops at the third error**, matching the human protocol and `score_game`.
  An unparsed reply is neither an error nor a stop, which is what `score_game` does with it.
- New `_parse_old_new()` for single-turn replies. A reply naming **both** words returns None
  rather than the first match, so "not old, it is new" is not read as Old. `parse_responses`
  is unchanged and still serves the summarizer arm, which still answers in bulk.
- Row fields: `step_log`, `answer_steps`, `encode_steps`, `answer_step_by_position`,
  `trials_presented`, `stopped_at_third_error` added; `encoding_log` and `recall_raw` removed
  (neither exists any more). `per_trial`, `metrics` and `gold_trials` keep their shape, so
  `error_structure.a2_model`, `wr_lag_model` and `score_candidate.axes` need no change.
- The summarizer arm (`evaluate_summarizer`, `SUMMARIZER_CONDITIONS`) is deliberately
  **unchanged**: it is an ablation with no store and no turn structure. It still re-prints the
  list, and any comparison against it must say so.

`meta_harness/check_predictions.py`: `_wr_summary` read coverage and old-rate out of
`recall_raw` over a denominator of 100. Post-rewrite rows have no `recall_raw` and usually far
fewer than 100 presented trials, so a fixed 100 would have read a participant stopped at trial
4 as 0.04 coverage — flagging the intended protocol as degeneracy. It now reads
`per_trial.model_response` over `trials_presented` for new rows and keeps the old path for old
ones. Verified both: the old generation still returns coverage 1.0, old_rate 0.5156,
ceiling_group 38 of 50; a synthetic new row returns coverage 1.0, old_rate 0.8.

`meta_harness/test_turn_boundary_reset.py`: four new tests (13 total, all passing) —
one-word-per-turn with the assertion that **no turn shows a word the participant has not
reached**, the third-error stop agreeing with `score_game`, an unparsed reply neither counting
nor stopping, and the parser refusing ambiguous and negated replies. The file's docstring
claim 3 was also stale: it still described n-back as encode-then-answer with the letter hidden.

## 5. What this does NOT fix, and must be said when the number is quoted

`word_recognition` humanlikeness still compares two distributions of **survival length**
rescaled by 100, not two accuracies. Closing M13 makes the model's survival length reflect its
memory instead of its ability to read the prompt; it does not turn the task's score into an
accuracy, because the human data contains none. The honest headline measures for this task are
**A2** (miss/FA ratio, human 6.094 — the presentation defect is exactly what
`domain_spec.md` blames for weakening it) and **M4_word_recognition_lag** (the lag-to-accuracy
curve in `report_error_shape.py`), both of which are shape comparisons and survive the
missing accuracy.

## 6. Re-baseline needed

- `run_to_run_floor.json` lists `word_recognition` noise as 0.0000 and
  `score_repeats.SAME_FAMILY_SD` lists 0.0180. Neither survives this change; both must be
  re-measured from repeats of the new code.
- The pre-stage-2 anchor is **0.5075** over the named pre-fix triple; the immediate comparator
  will be whatever job `18774002` reports for this task, which is still the OLD presentation.
- Wall-clock cost rises: 100 trials × 2 turns is up to 200 turns per participant against 2
  before, though the third-error stop cuts it back sharply once the model starts erring
  (human mean 34.49 trials ⇒ ~69 turns).
