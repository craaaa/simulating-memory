# Instrument fix, stage 1 — closing the two leaks and giving n-back a read channel

Written 2026-09-28, **before** any post-fix run exists. Pre-fix state tagged
`exp/compactor-prefix-v1`. No number produced before that tag is comparable to a number
produced after it.

**Units.** Humanlikeness = 1 − Wasserstein-1 between the model's and the humans'
per-participant score distributions. Range 0–1, higher = more human-like, **in units of task
proportion-correct** (0.03 = three percentage points of distributional distance). A **delta** is
candidate score minus baseline score in those same units and is the only quantity that can be
negative. Counts of trials are labelled as counts and are not humanlikeness.

## Why

`analyze_live_dimensions.py` (commit `dbd2ea3`) showed that 99–106% of every frontier
candidate's mean-over-8 delta came from `nback` and `variable_mapping`, with the other six
tasks contributing nothing outside 2 SE. Those two are exactly the tasks whose stimuli never
leave the conversation context, so their scores were set by harness plumbing rather than by the
key-value store. Six iterations of search therefore optimised the instrument.

Reading the code established the asymmetry precisely:

| | store read back? | transcript at answer time |
|---|---|---|
| 6 batch tasks (`encode()` → `recall()`) | yes, `{wm_contents}` | none — `recall()` is a bare `generate()` |
| `variable_mapping` (`step()` ×2 per question) | yes, at the question turn | **leaked** |
| `nback` (`step()` per letter) | **never** — no read tool, no injection | **leaked** |

`TOOLS` holds only `write_memory` and `delete_key`. n-back's store was **write-only**: the
agent wrote keys it could never read, and every answer came from the transcript.

## What changed, exactly

**1. `bench/core/wm_agent.py` — `step()` clears the transcript at the turn boundary.**
Two lines (`self.reset_messages()` before `_ensure_messages()`), plus the docstring. Every turn
now sees the system prompt, its own user message, and its own tool-call loop. The store is the
only thing that crosses a turn. The tool loop keeps its multi-message state *within* a turn,
which is why `step()` still exists rather than a bare `generate()`.

*Not* changed: `_tool_call_cap()` is untouched, and `_tool_interactions` / `_tool_calls_used`
still accumulate across turns, so the cap still grows at 1.5 per tool-bearing turn. Changing
capacity in the same commit as the leak fix would confound two variables. Route 3 is already
dead — the cumulative budget explained 6 of 965 unanswered trials
(`logs/unanswered_cause.md`).

**2. `bench/tasks/wm_nback.py` — two turns per letter, matching `variable_mapping`.**

```
encode turn  (allow_tools=True)        answer turn  (allow_tools=False)
  store contents                         store contents
  New letter (position P): X             Original task instructions: TASK_DESC_BY_N[n]
  Update your memory as needed.          answer for the letter at position P:
                                         Does it match the letter n position(s) back?
```

- The standalone instruction turn is **deleted**. With the transcript cleared it would have
  been visible to nothing. The model-phrased instructions are restated on each answer turn,
  which is what the other seven tasks do.
- This also fixes a latent bug: `TASK_DESC_BY_N[n]` was passed to `wm_system_prompt_for()` as
  `task_prompt`, and `wm_system_prompt()` accepts that parameter and **never uses it** — only
  `human_task_prompt` reaches the system prompt. n-back's model-phrased instructions therefore
  reached the model **nowhere** before this change.
- The position index is stated explicitly because clearing the transcript removes the agent's
  only cue to where it is in the block. A human watching letters appear has that cue for free;
  `variable_mapping` supplies the same thing as `Question {q_idx}`.
- The answer turn carries **no stimulus**, so the current letter must have been written to the
  store for the comparison to be possible at all.
- New row fields `encode_steps`, `answer_steps`, `answer_step_by_position`. **The old
  positional rule "scored trial k is step n+k" is now wrong** — the step log holds two entries
  per letter. Scored trial k is `answer_step_by_position[n + k]`.

**3. `bench/tasks/wm_variable_mapping.py` — the encode turn shows the store.**
New `WM_ENCODE_PROMPT` with `{wm_contents}`; the old `ENCODE_PROMPT` stays for the summarizer
ablation, which has a running summary rather than a KV store. Without this the agent, having
lost the transcript, would overwrite its own slots blind.

**4. `bench/core/llm_openai.py` — sanitize messages on the tool path.**
`generate()` applied `_sanitize_message_text()` to prompt and system text; `generate_with_tools()`
applied it to nothing, so surrogates, NULs and control characters reached the API on the tool
path only. New `_sanitize_messages()` copies rather than mutates, because the caller's list is
the agent's live `_messages`. Inert for letters and digits; it would have mattered for a story
or CRAFT item carrying a stray control character.

**5. `meta_harness/test_turn_boundary_reset.py` — six tests, offline, fake LLM.**
Pins: turn 2 cannot see turn 1; the tool loop keeps its own messages within a turn; the store
survives the reset; the batch path's request sequence is unchanged (asserted as an actual
request sequence, not argued); n-back's encode/answer split with a tool-free store-only answer
turn; vm's encode turn shows the store.

## Deferred, deliberately

- **Stage 2: migrating the six batch tasks from `recall()` to `step(..., allow_tools=False)`.**
  The wire requests are equivalent at temperature 0 (`generate()` sends `top_p=1.0` explicitly,
  `generate_with_tools` omits it and the server default is 1.0), and `step()` would give those
  tasks `_step_log` coverage they completely lack today. But it re-baselines six tasks that
  contribute nothing outside 2 SE to any candidate delta, in exchange for diagnostics no
  candidate can currently use. Worth doing when a candidate needs to see inside a batch
  read-out.
- **The empty-content answer turn.** 418 of 472 unanswered n-back turns returned empty
  assistant content — the model wrote memory and emitted no label. n-back's answer turn is now
  `allow_tools=False`, so no tool call can crowd the text out, and this may disappear as a
  structural consequence. If it persists, fix it separately with this run as the comparison.

  **RESOLVED 2026-09-29: it disappeared, structurally, as hoped.** On
  `runs/iter11postfix/baseline` (3 repeats) `n_no_answers` is **0 at every n-level** and
  `answered` is 13.94 / 13.98 / 13.96 of 14, against a pre-fix n=3 that answered 6.82.
  The remaining model unanswered rate is 0.0029 against a human 0.0000-by-construction.
  No separate fix was needed. `logs/nback_turn_order_outcome.md`.

- **`word_recognition`.** Its defect is not a history leak — `recall()` has no history. Its
  `trials_text` presents all 100 test words at once, and continuous recognition defines "Old"
  as *appeared earlier in this list*, so the answer is derivable from the prompt itself (36 of
  50 participants score 100/100). That needs a turn-by-turn rewrite of the task. ~~Known-broken,
  not fixed here.~~

  **DONE 2026-09-29, `dc14d0a` + `56eedda`.** Moved out of "deferred". The rewrite landed as
  planned in `logs/instrument_fix_stage2_plan.md`: one word per turn, answer-then-store,
  `encode()` deleted, and presentation stops at the third error to match the human protocol.

  Two things the plan did not anticipate, both recorded in that file's outcome sections:
  - **The prerequisite check returned the opposite of the plan's assumption and it did not
    matter.** `src/score.py` divides both sides by a fixed 100 — but `score_game` already
    applied the 3-strike rule analysis-side, so both sides were already measuring words
    survived (`score == len(per_trial) − 3` for all 12 of 50 rows that reached three errors).
    Both of the plan's decision options were void; only the loop condition was left to change.
  - **There is no human accuracy on this task at all.** `trialsCompleted − correctResponses == 3`
    for 53 of 53 human records, so the human "proportion correct" is algebraically `1 − 3/n`.
    Scoring is now survival length on both sides (`logs/survival_length_scoring.md`).

  **The measured outcome against the pre-registered expectation is NOT IN YET.** The plan
  predicted the model's score falls and its humanlikeness rises; that is prediction P1 of job
  18781213, registered in `logs/predictions_iter12stage2.md` before the run started, with
  explicit support and rejection bands (mean survival < 60 words and humanlikeness > 0.60 versus
  survival > 80 or humanlikeness within 0.04 of 0.5364). Pre-fix reference: model survival 84.80
  words against a human 34.49, and a lag curve flat at 0.957–1.000 where humans rise
  0.506 → 0.950.

## Pre-registered predictions for the post-fix baseline

Written before the run. The project has been wrong three times about measurement, so these are
committed in advance and will be reported as stated whether or not they hold.

| # | prediction | pre-fix value | predicted post-fix |
|---|---|---|---|
| 1 | `variable_mapping` baseline humanlikeness rises into the band the candidates reached, because the baseline now does what they did | 0.3539 | **≥ 0.85** |
| 2 | `nback` model mean accuracy **falls** — the store is now the only channel, with `MAX_KEYS = 4` slots against an n=3 comparison | 0.694 | **< 0.65**, direction relative to human 0.8657 unknown |
| 3 | `nback` trials answered rises, because the answer turn cannot be crowded out by a tool call | 10.85 of 14 | **≥ 13.0 of 14** |
| 4 | A4 normalized `rc_ratio` in the **baseline** approaches the human 0.3728, since the baseline now closes the leak the candidates closed | 0.8662 (30 errors, least trustworthy figure in that column) | **< 0.55** with ≥ 30 errors |
| 5 | the **five intact** batch tasks' humanlikeness is unchanged within their measured run-to-run noise | mean-8 0.7861 | per-task delta inside the `run_to_run_floor.json` spread |

Prediction 5 covers `digit_span_forward`, `digit_span_reverse`, `semantic_story_recall`,
`narrative_qa` and `craft_task`. **`word_recognition` is excluded**: its score does not depend
on the store at all (all 100 test words are shown at once and "Old" means "appeared earlier in
this list"), so a move there says nothing about whether the reset leaked somewhere it should
not have.

Prediction 2 is the one that matters and its direction is genuinely unknown: n-back could
overshoot below the human distribution, as `respond_first_v2` did (accuracy 0.588, humanlikeness
0.7054). If it does, n-back has no headroom in the humanlike direction and the task's usefulness
for this search is over — which would be a finding, not a failure.

Prediction 5 is the **precondition**: if a batch task moves outside its noise band, the reset
reached somewhere it should not have, and predictions 1–4 are void until that is explained.

## Consequence for the twelve existing candidates

Every candidate overrides `step()` with its own copy, so the `bench/` fix does not reach them.
`episodic_reset_v2`/`v3`, `evicting_reset`, `respond_first` and `respond_first_v2` each
reimplement `step()` largely to call `reset_messages()` at the turn boundary — which the
baseline now does. Post-fix, "candidate vs baseline" no longer measures the leak on these two
tasks. Candidates that keyed their behaviour on the old bare `"Next letter: X"` prompt will also
see a different message shape. **Do not run a candidate against the post-fix baseline before
checking it against the new turn structure.** The frontier as recorded belongs to
`exp/compactor-prefix-v1` and should be read as history.
