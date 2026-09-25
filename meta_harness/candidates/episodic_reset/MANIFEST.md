# `episodic_reset` — iteration 2c

Parent: **`baseline`** (plain, unmodified). Not `displacement`.

One method body differs from the baseline: `WorkingMemoryAgent.step()`.
`WorkingMemory`, `write_key`, the overflow policy, `recall()`, `TOOLS`,
`CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS` and `MAX_KEYS` are untouched.

Parent choice justified: building on `displacement` would require overriding
`write_key`, which proposer 2a owns, and would confound the leak result with the
overflow result. Clean single-mechanism attribution is worth more here than a
higher expected score, and §4 and §7 state plainly where that costs me.

---

## 1. The defect, with numbers

`step()` maintains the full conversation history across calls
(`bench/core/wm_agent.py:176`). `reset_messages()` exists at line 161 and is
called by **nothing** in `bench/` or `src/` (grepped). `recall()` is the contrast
case: it builds one fresh prompt from `wm.to_recall_text()` alone, which is why
the six tasks that answer through `recall()` are the six where the bottleneck
binds.

Two tasks call `step()` more than once — `nback` and `variable_mapping` — and they
are the two least humanlike tasks in the table (0.7909, 0.3554).

### The decisive number, which is new

Counting `final_kv` values that contain a bare capital letter — i.e. that carry
letter identity at all — in the baseline's n-back run:

| n | `final_kv` values | carry a letter | `acc_over_answered` | `keys_held` |
|---|---|---|---|---|
| 1 | 50 | 50 (1.000) | 0.9943 | 1.00 |
| 2 | 77 | 77 (1.000) | 0.8206 | 1.54 |
| 3 | 198 | **4 (0.020)** | **0.7372** | 3.96 |

At n=3 the store holds **no letters**. The keys are `position_2`, `position_3`,
`position_8`, `position_10` holding contentless labels ("seventeenth letter in
sequence"), and the agent is still 0.737 accurate on the trials it answers. A
store with no letter information cannot support a 3-back judgement. The answers
come from the dialogue history.

`variable_mapping`, already in the WORKLOG: 674 of 1500 questions ask about a name
the store has evicted and are answered at 0.985; where the store holds a *stale*
value contradicting the truth, the model overrides its own memory and is still
right 93.5% of the time.

---

## 2. Mechanism, and what survives a reset

`step()` rebuilds its message list at every turn boundary. What the agent sees on
turn *k*: the system prompt, the current contents of the four slots, and the
current stimulus. Nothing else.

"Call `reset_messages()` after every step" is the naive reading and it is wrong
for a reason visible in the code, not merely a psychological one: **`step()` never
shows the agent its own store.** The only read access `step()` ever had to `wm`
was the conversation history of its own past tool calls. Wipe that and the agent
is not memory-limited, it is blind to its own current mental contents — a state no
account of working memory posits, and one that would destroy n-back by preventing
any strategy at all rather than by imposing a capacity limit.

### The account: Oberauer (2002), the concentric model

Chosen over the two alternatives because it is the account that licenses exactly
three things and no fourth, and the three map onto structures that already exist
in the harness:

| component | capacity | harness object | kept? |
|---|---|---|---|
| focus of attention | 1 item | the incoming `user_message` | yes |
| region of direct access | ~4 chunks | the 4-slot store, `to_recall_text()` | yes, **and now rendered into the turn** |
| activated LTM / task set | not chunk-limited | the system prompt + `TOOLS` | yes |

There is **no component corresponding to a verbatim record of previously
presented, no-longer-attended stimuli.** That is exactly what the conversation
history is. That is why all of it goes.

`MAX_KEYS = 4` already cites Cowan (2001); Oberauer's region of direct access is
the same ~4-chunk limit with the access structure made explicit, so this is the
existing commitment extended, not a new one.

**Alternative rejected: Baddeley (2000), the episodic buffer and its LTM
interface.** It would license a gist-level residue of earlier stimuli living
*outside* the four slots. The harness has no representation for that, so
implementing it means adding a second store — capacity growth by the back door on
a search where `full_context` already proved capacity is the wrong direction.

**Task set exempted from the limit, per Monsell (2003) and Logan & Gordon
(2001):** instructions are procedural knowledge retrieved from LTM, not chunks
held in the store. Humans in these tasks do not forget what the task is.

### Item-by-item

| content | verdict | why |
|---|---|---|
| system prompt | **kept** | task set. Verified sufficient, not assumed: for n-back the C2 prompt embeds `HUMAN_PROMPT_BY_N[n]`, which states the full rule ("respond with 'no response' to the first *n* letter(s)", "same"/"different") plus a worked example with its explanation. The task's own instruction turn (`wm_nback.py:86`) is therefore redundant with it, so **nothing needs pinning** and the rule has no exceptions. |
| `TOOLS` schemas | **kept** | affordances, not episodic content. Text unchanged. |
| the four slots' contents | **kept**, now visible | a region of direct access the processing system cannot read is not one. |
| last *N* turns verbatim | **excluded, N = 0** | at n=1 one turn hands over the answer outright; at n=2/3 it hands over part of it, so the leak would **scale with n** and confound the per-level comparison (`nback_levels.py`) this candidate exists to make honest. And a verbatim recent window is a second memory store with unbounded fidelity and no declared capacity — `full_context` at small scale, and `full_context` is the least humanlike harness measured (0.6387 vs 0.7861). The "sense of recent context" a human retains *is* what the agent chose to put in its four slots. That is the compactor's whole claim. |
| agent's own prior replies | **excluded** | episodic self-record, and stimulus-correlated on n-back: a "same" at position *k* reveals letter *k* = letter *k−n*. Keeping them reintroduces the leak through the agent's own mouth. |
| agent's own prior tool calls | **excluded** | the encoding episode itself. |
| tool results *within* the current step | **kept** | the agent must see whether its write succeeded. The reset fires only at turn boundaries — clearing mid-loop would also leave an orphaned `tool` message with no preceding assistant `tool_calls` and the request would be rejected. |

---

## 3. Prompt delta, and why it was forced

One addition, on `step()` turns where the store is non-empty:

```
Your working memory currently contains:
<wm.to_recall_text()>

<the task's own user message>
```

* **Forced**, per §2: without it, `step()` gives the agent no read access to its
  own store, and the candidate would degrade by blindness rather than by capacity
  limit — a garbage-emitting failure mode that `interference.py` rightly scores as
  `novel_guess` errors.
* **Not new text.** `"Your working memory currently contains:"` is verbatim from
  `RECALL_PROMPT` (`bench/core/wm_agent.py`) and from `QUESTION_PROMPT`
  (`bench/tasks/wm_variable_mapping.py`); the renderer is the same
  `to_recall_text()` those prompts use.
* **Never duplicated.** Where the task's own message already contains that header
  — `variable_mapping`'s question turn does — the block is not prepended, so on
  those turns the prompt is byte-identical to the baseline's apart from the absent
  history. Verified offline (§8).
* The rendering is the **pre-turn** store state; writes made during the turn are
  confirmed by tool results and appear in the next turn's block.
* `recall()` is not touched, so nothing about word-recognition's trial-list
  presentation (proposer 2b) changes.

---

## 4. Not a capacity change, and not noise

* `MAX_KEYS` stays 4; `WorkingMemory` is not subclassed; the capacity invariant is
  literally the baseline's, plus a live-read assertion (`_capacity()`) so the
  candidate fails loudly rather than silently if combined with a 2a candidate that
  breaks it.
* The change is strictly **subtractive** in what the model can see, and adds no
  stochasticity. **There is no RNG in the file.** Deterministic and per-turn
  identical across runs — which is why the no-change control in §7 is
  *byte-identical* rather than merely within-noise.
* Not `random_decay_v2` with extra steps. That candidate bought the aggregate
  (best_span 18.4 → 8.2 against a human 6.88) while making the error structure
  twice as wrong (A1 leak 0.249 vs human 0.087), because it dropped content the
  agent had chosen to keep. This drops **nothing** the agent chose to keep, and on
  the six `recall()` tasks it changes literally nothing, so it cannot buy an
  aggregate anywhere except the two tasks whose defect it owns.
* The tool-call cap is untouched, and the direction of the compute change is
  *downward*: the context shrinks by roughly an order of magnitude on both affected
  tasks. That alone carries the "not extra compute" argument.
  There is a second, weaker reason to expect tool-call *demand* to fall — rendering
  the store makes *overwriting* an existing key discoverable, one call where the
  baseline's blind delete-then-write repair costs two against a budget of
  `max(6, 1.5 × steps)` — but it is marked here as **unverifiable from this run's
  outputs**: `wm_nback.jsonl` rows do not carry `step_log`, and
  `variable_mapping`'s `step_logs` do not carry `tool_call_cap_hit`. It is offered
  as a mechanism sketch, not as evidence, and no prediction rests on it.

---

## 5. The A4 prediction, and three corrections to the brief's premises

### Correction 1 — A4 has already fired. The brief's central claim is false.

> "A4 has never been able to fire on a real run, because no run has ever produced
> the >= 30 variable_mapping errors it needs to be trustworthy."

`displacement` (iteration 1) produced **35 errors, `trustworthy: true`, `rc_ratio`
1.2575, `rc_ratio_distance` 0.1285**. Measured by running `interference.a4()` on
`meta_harness/runs/iter1/displacement/`. So `n_errors >= 30 AND rc_ratio > 1.15`
is **already satisfied by the previous iteration** and passing it would prove
nothing about this mechanism. The bar has to be raised, and it is, below.

### Correction 2 — displacement's 1.2575 was the arithmetic maximum, so it is evidence of almost nothing.

`relation_count` is `min(2q, 10)`: the model's rc distribution over 1500 trials is
150 trials each at 2, 4, 6, 8 and 900 at 10. So `rc_ratio` is a monotone function
of the share of errors at q ≥ 5, and it is bounded above by the value you get when
*every* error sits at rc = 10:

| n_errors | max attainable `rc_ratio` |
|---|---|
| 12 | 1.2525 |
| 35 | **1.2575** |
| 50 | 1.2609 |
| 150 | 1.2879 |
| 400 | 1.3750 |

Displacement's `rc_mean_error` was **exactly 10.000** — it hit the ceiling for its
error count. The ceiling rises with error count, so 1.386 is *not* impossible; but
it requires several hundred errors. Concretely: **at a given error count,
`rc_ratio` carries very little information beyond "what share of errors were
late".** It still discriminates a uniform noise injector (which tends to 1.0), and
the `test_guards.py` fixtures confirm it is not an error-rate detector. But it
should not be read as a measure of interference sensitivity on this task, because
for the model `rc` is collinear with question index.

### Correction 3 — the two sides of `variable_mapping` are scored by **different formulas**. This is a fifth comparability defect and it is not in the audit table.

* human (`src/score.py:_score_human_record`): `sum(q.correct for q in questions) / 10` — a **count of correct answers**.
* model (`src/score.py:_score_llm_row` → `metrics["score"]` from `bench/tasks/variable_mapping.py:score_run`): **`relation_count` of the last question answered correctly before the first error**, `/ 10`. The game stops at the first error.

These are different quantities that happen to share a denominator. The
consequence is arithmetic and large: because `relation_count = min(2q, 10)`
saturates at q5, **an error at q6–q10 is invisible to the model's score.** The
baseline's `first_error_at` histogram is `{3: 2, 8: 4, 9: 2, 10: 4}` — 10 of the
12 runs with an error still score 1.0. That, not the 0.992 accuracy alone, is what
produces the "99% at 1.0, 2 unique values" point mass.

So the brief's framing of the ceiling point mass as a *memory* fact is partly a
*scoring* fact, and the prediction in §7 is stated on `first_error_at` accordingly.

### The central prediction

**P1. `n_errors >= 150` (`error_rate >= 0.10`) with `rc_ratio > 1.15`.**

Why 150 and not 30: 30 is already met by displacement. 150 is 12.5× the
baseline's 12 and 4.3× displacement's 35, and it is what the leak arithmetic
demands. 674 of 1500 questions (45%) ask about a name the store no longer holds.
If the store is the only route, those must be answered at or near the 4-option
chance rate of 0.25, giving an expected error rate of roughly
0.45 × 0.75 ≈ 0.34. Rendering the store each turn should let the agent manage it
better and raise the "key present" share, so I set the threshold well below the
point estimate at 0.10 (150 errors) and expect 0.15–0.40.

**Why this is the central test.** It separates "the store is now load-bearing"
from "the model got worse" without appealing to any score. A candidate that
produces 150+ errors with `rc_ratio` near 1.0 has made the model noisy, not
memory-limited. **I accept `rc_ratio <= 1.15` at `n_errors >= 150` as
disconfirming** — it would mean my errors are spread uniformly over question
index, which is not what a store that fills up predicts. Given Correction 2 I
also state the honest reading of a pass: `rc_ratio > 1.15` means errors are
concentrated late, which is necessary but not sufficient for an interference
account.

**Not** predicted on: `intrusion_share`. See §9 — there is a measurement bug in
`interference.py` that biases it downward, so a threshold there could fail on the
measurement rather than the mechanism. Parse integrity is checked instead via
unparsed-answer count (P8).

### Reconciling P1 with the score prediction

These are compatible, not in tension. The score depends on the **first** error
(early-weighted, a minimum); `rc_ratio` depends on **all** errors (late-weighted,
a mean). A run with its first error at q3 and five more at q6–q10 satisfies both.
Displacement had 25 runs with at least one error and 35 errors in total.

---

## 6. Does `variable_mapping` need protocol matching?

**Yes, and the correction is the score formula, not the 3-strike rule.** Do not
read this candidate's raw `variable_mapping` humanlikeness as a fact about memory
without it.

**Analysis-side rule (for `protocol_match.py`, not for the harness — a harness
must not know the scoring protocol):** recompute the model's per-run
`variable_mapping` score as

```
score = (number of questions answered correctly) / 10
```

i.e. the same formula `_score_human_record` applies to humans, replacing
`relation_count`-at-last-correct-before-first-error. Score both sides that way and
recompute humanlikeness.

Measured effect of the recomputation on the model side (no human data read):

| run | as scored now | as correct-count/10 |
|---|---|---|
| baseline | 148 @ 1.0, 2 @ 0.4 | 138 @ 1.0, 12 @ 0.9 |
| displacement | 149 @ 1.0, 1 @ 0.8 | 125 @ 1.0, 18 @ 0.9, 4 @ 0.8, 1 @ 0.6, 2 @ 0.5 |

The correct-count formula is monotone in error count, so it registers the errors
the current formula discards.

**The 3-strike rule is inert here and should not be applied.** Truncating model
runs at the third strike leaves 1495 of 1500 trials (mean 9.97 of 10 questions
answered) and moves `rc_ratio` from 1.2575 to 1.2586 — the model almost never
accumulates three strikes. This confirms the WORKLOG's finding for the score and
extends it to A4. Under this candidate, error rates should rise enough that the
3-strike rule starts to bite, so it is worth **reporting** the truncated question
count as a covariate; but it is the formula mismatch that has to be fixed for the
number to mean anything.

---

## 7. Pre-registered predictions

Baseline reference run:
`meta_harness/runs/iter0/baseline/Qwen_Qwen3-30B-A3B-Instruct-2507`.
Machine-readable form: `meta_harness/logs/pending_episodic_reset.json`.

### Primary — variable_mapping, the store becomes load-bearing

| # | quantity | source | baseline | threshold | disconfirmed if |
|---|---|---|---|---|---|
| # | quantity | baseline | threshold | **point estimate** | disconfirmed if |
|---|---|---|---|---|---|
| **P1** | `A4.n_errors` (`interference.a4`) | 12 | **>= 150** | **225–600** (error_rate 0.15–0.40) | < 75: the store was never needed, so the leak reading is wrong |
| **P2** | `A4.rc_ratio` (`interference.a4`) | 1.1682 (n=12, untrustworthy) | **> 1.15** at P1's error count | 1.20–1.30 | <= 1.15: errors are uniform over question index — noise, not a filling store |
| **P3** | share of runs with `first_error_at <= 5` (`metrics.first_error_at`) | **2/150 = 0.0133** (displacement 1/150) | **> 0.25** | **~0.87** (1 − 0.66^5 at a per-question error rate of 0.34) | <= 0.10: errors are all late, so the score's point mass cannot break up and the vm number stays meaningless |
| **P4** | share at score 1.0 / unique score values (`metrics.score`) | 0.9867 / **2** | **< 0.80** and **>= 5** | ~0.15 / 6 | point mass intact (>= 0.95 at ceiling): contradicts P1/P3 |
| **P5** | `variable_mapping` humanlikeness (`score_candidate`) | 0.3554 | **>= +0.05** (noise floor 0.017) | **+0.45 to +0.55** | delta < +0.05 with P1 passing: the score formula is absorbing the errors — §6. Also disconfirming in the *other* direction: a delta above +0.60 would overshoot the human mean, i.e. the model is now worse than humans |

The thresholds are deliberate lower bounds, well below the point estimates, because
the point estimates rest on a chance-rate argument rather than a measurement. The
point estimates are recorded so that a pass at 10× the bar reads differently from a
marginal one — three of these five would pass on almost any outcome consistent with
P1, and that should be visible rather than banked.

P5 triggers the A4 conditional guard in `score_candidate.py`. P1+P2 are the
pre-commitment to pass it.

### Primary — nback, the store must start carrying letters

| # | quantity | source | baseline | threshold | disconfirmed if |
|---|---|---|---|---|---|
| **P6** | n=3 share of `final_kv` values containing a bare capital letter | `wm_nback.jsonl` `final_kv` | **4/198 = 0.020** (n=1 1.000, n=2 1.000) | **> 0.40** | <= 0.20: the agent did not adapt, so the n=3 store still carries no task information and any accuracy is unexplained |
| **P7** | n=3 `keys_held` **direction**, read jointly with P6 | `nback_levels.diagnostics` | 3.96 | reported, not thresholded | — |

P7 is the branch discriminator and is stated in advance:

* **P6 passes and `keys_held` falls toward 1** → the agent adopted
  `full_context`'s rolling single-key strategy (`position_3_back`, 49/50 blocks,
  100% bare letters, 14/14 answered at 0.770). n-back may *improve*, and for the
  right reason. This is the outcome I consider most likely, because that strategy
  never overflows and so never meets the refusal.
* **P6 passes but `keys_held` stays ~4 and `answered` falls** → the
  refuse-on-full jam dominates. n-back regresses, and the diagnosis is that
  closing the leak requires 2a's overflow fix to be viable. Informative, not a
  refutation of the mechanism.

**Stated cost, up front.** nback's effective floor is −0.060 against a baseline of
0.7909. I do not predict nback will improve. I regard a floor violation there as
a live possibility and the honest price of closing the leak while refuse-on-full
is still in place, since I am forbidden from touching it. The n=3
`acc_over_answered` of 0.7372 is, on §1's evidence, a leak-derived number; if it
falls, that is the leak closing, not the model breaking — which is exactly what P9
is there to distinguish.

### Guard — this must fail if the mechanism merely confuses the model

| # | quantity | source | baseline | threshold | reads as |
|---|---|---|---|---|---|
| **P8** | variable_mapping unparsed answers (letter missing or out of range) | `parsed_answers` vs `questions` | **0 / 1500** | **< 30 / 1500** | output format intact. `interference.py` counts an unparsed answer as a `novel_guess` error, so a candidate that reached P1 by emitting garbage fails here |
| **P9** | n=1 `acc_over_answered` **and** `answered` | `nback_levels.diagnostics` | 0.9943, 13.98 / 14 | **>= 0.90** and **>= 13.0** | **the key test.** At n=1 a single overwritten key suffices and the baseline already uses exactly that (`previous_letter`, 50/50 blocks, 1.00 keys held). One item in four slots cannot be a capacity failure, so a drop here can only mean the reset broke instruction-following, phase tracking or output parsing. **A P9 failure invalidates P1–P7 regardless of their values.** |

### No-change control — six tasks, byte-identical by construction

**P10. `word_recognition`, `digit_span_forward`, `digit_span_reverse`,
`semantic_story_recall`, `craft_task` and `narrative_qa` each move by
|Δ| <= 0.005, and A1 `sub_span_leak` (0.1298), A1 `best_span` (18.4), A3 `bleu`
(0.0031), A3 `words` (121.4) and A2 `ratio` (0.340) / `trials_attempted` (82.92)
are unchanged.**

This is stronger than iteration 1's control and it is structural, not statistical.
`encode()` calls `step()` exactly once, and all six tasks construct a fresh agent
per trial (`wm_digit_span_forward.py:209`, `wm_digit_span_reverse.py:110`,
`wm_semantic_story_recall.py:165`, `wm_mcq_common.py:52`,
`wm_word_recognition.py:94`). On that single step there is no history to drop and
the store is empty, so no state block is prepended. **Verified offline, not
argued:** the request stream a stub LLM receives across `encode()` → `recall()` is
byte-for-byte equal between `baseline` and this candidate (§8).

So the expected delta on all six is **exactly 0**. Any nonzero movement is serving
nondeterminism, not a mechanism effect, and must not be narrated as one. Digit
span is included but is no longer the interesting control: word_recognition is,
because a move there under a mechanism that provably cannot reach it would mean
the run is not reproducible.

Corollary, and this candidate's main virtue: **the mean can only move via `nback`
and `variable_mapping`.** No headline can be bought from elsewhere.

### Mean

**P11. Conditioned on P7's branch, and derived from P5 rather than guessed.** An
unconditional range here would have been miscalibrated in exactly the way
iteration 1's word_recognition threshold was, because P5's point estimate alone is
worth +0.056 to +0.069 on the 8-task mean (+0.45 to +0.55 divided by 8):

* **branch (a)**, P6 passes with `keys_held` falling — nback also improves, toward
  `full_context`'s per-level profile, worth roughly a further +0.019 on the mean.
  Predicted mean delta **+0.05 to +0.10**.
* **branch (b)**, the refusal jam dominates and nback regresses to its floor or
  past it. Predicted mean delta **−0.04 to +0.05**.

Either way the primary claims are P1, P3, P6 and P9, not the headline mean. A
mean gain under branch (a) would be **the largest in the search so far**, and it
must not be read as a capability result: it is what happens when two tasks stop
being scored on a leak. The `variable_mapping` component of it is also not
interpretable until the §6 protocol match is applied.

---

## 8. Offline verification performed

`verify_interface.py` **PASS**: overrides `WorkingMemoryAgent`, `SummarizerAgent`,
`TOOLS`, `CONDITION_PROMPTS`, `MAX_KEYS`; injection rebinds `WorkingMemoryAgent`
in 8 modules and `MAX_KEYS` in 10; encode/recall round trip returns the expected
shapes; capacity enforced at 4.

Stub-LLM leak test, in-process, no GPU, no network — a recording stub that writes
one rolling key per tool-enabled turn, driven through an n-back-shaped and a
variable-mapping-shaped turn sequence. **All assertions pass.** Observed context
on the 5th letter turn of a 3-back block (letters K Q R K Z), in full:

```
[system]    SYSTEM: 3-back rules here
[user]      Your working memory currently contains:
            window: letter_4

            Next letter: Z
[assistant] (tool_call: write_memory)
[tool]      Key 'window' written.
```

Asserted and passing: system prompt survives; current stimulus present; **no
earlier stimulus string appears on any turn** (max count of `"Next letter:"` over
all turns = 1); no earlier assistant reply present; store rendered with its value;
the task's instruction turn is gone (the system prompt carries the rule); context
is one turn plus the within-step tool loop (4 messages).

The **first** letter turn of a block was inspected separately, because it is the
one turn at risk and the one place P9 can fail. The store is still empty there, so
the `if store` guard skips the state block and the agent sees exactly:

```
[system]    SYSTEM: 3-back rules here
[user]      Next letter: K
```

No state block, not even "(memory is empty)". **The guard is left as it is
deliberately:** rendering an empty store would change `encode()`'s single step and
destroy the byte-identical property that is this candidate's strongest control
(P10). The diagnostic consequence, stated in advance: if P9 fails and the failure
is localised to buffer-period turns, the reading is "the agent lost track of
sequence position", whose fix is pinning the instruction turn — a *different*
iteration-3 move from "the reset broke the model". `model_parsed_buffer` at n=1 is
a live diagnostic for this (0.92 baseline, 1.00 displacement, 0.00 full_context —
unlike n=3, where it is a fixed positional pattern and inert).

Variable-mapping shape, final question turn: no earlier statement block visible,
current question visible, store header appears **exactly once** (no duplication
against the task's own header), and the encode turn shows the store while hiding
earlier statements.

No-op check: `encode()` → `recall()` request streams identical between `baseline`
and `episodic_reset` (3 requests each, equal).

The test lives in the session scratchpad rather than this directory, because the
iteration brief restricted me to three files.

---

## 9. Two bugs found, reported not fixed

Both are analysis-side and I have not edited the files, per the brief.

**(a) `interference.py:model_trials` under-counts `assigned_before` by ~2×, biasing
`intrusion_share` downward.** It builds the set of previously assigned cities by
walking assignments while `a["turn"] < turn`, where `turn` falls back to
`q["question_index"]` because model question records carry no `turn` field
(verified: question keys are `correct_city`, `name`, `options`, `question_index`,
`relation_count`). But **two assignments are presented per question** —
assignment turns run 1..20 against question indices 1..10 — so before question
*k* there are 2*k* assignments presented while the filter keeps only *k*−1. Real
intrusions are therefore misclassified as `novel_guess`. Suggested fix: index
assignments by presentation block, `a["turn"] <= TURNS_PER_QUESTION * (k - 1)`.
Both reported `intrusion_share` figures (baseline 0.4167, displacement 0.5429) are
depressed by this, and so is `intrusion_chance`. This is why §5 declines to
pre-register a threshold on `intrusion_share`.

**(b) The `variable_mapping` score-formula mismatch** — §6, Correction 3. The
per-task comparability audit records two defects for this task; the formula
mismatch is a third and is the one that makes the point mass arithmetic.

---

## 10. Interaction with 2a and 2b

### 2b (word_recognition trial-list presentation in `recall()`) — the important one

**Under this candidate alone, word_recognition does not move at all** (P10, and
verified byte-identical in §8). I touch neither `recall()` nor `encode()`.

**Resolved by inspection, so this is not a guess.** `candidates/serial_recognition/`
overrides `recall()` only — it keeps word_recognition on the
`encode()` → `recall()` route rather than serializing into `step()` calls. So:

* **Actual case: orthogonal.** Combined behaviour on word_recognition equals 2b
  alone, because word_recognition still runs one `encode()` step per agent and my
  reset has nothing to remove there. Attribution stays clean and there is nothing
  to coordinate. Combining the two is safe.

* **The branch that would *not* have been safe**, recorded because it is a trap for
  any future iteration that reaches for serialization: if a candidate serializes
  the recognition trials into per-trial `step()` calls, then
  **2b alone does not close the leak, and needs this candidate to do so.** The
  `encode()` turn, whose user message is `word_list_text` = the entire studied list
  formatted `"{trial_index}: {word}"`, stays in the conversation history for every
  subsequent trial turn. So the studied list would remain visible verbatim even
  after the recall prompt stops re-presenting it, and word_recognition would still
  be leaky — quietly, and with the appearance of having been fixed. In that branch
  the two are **complementary and jointly necessary**, and each *alone* is
  insufficient. I would expect 2b-alone to look like a partial fix and 2b+reset to
  produce the real A2 recalibration that PROPOSER.md anticipates. That branch is
  not live for `serial_recognition` as written, but it is the reason a future
  serialization candidate must be combined with this one rather than run alone.

Either way, A2's *meaning* changes once the leak closes, and the axis needs
recalibration against a fresh baseline rather than a claimed delta.

### 2a (overflow policy, primacy protection)

**Expected complementary, plausibly necessary, on nback.** §7/P7 spells it out:
this candidate makes the store the only route at n=3, and refuse-on-full is still
in place. `displacement` showed that refusal suppresses responses, and
`full_context` showed the correct strategy (a single rolling letter key) is
reachable when writes never fail. `episodic_reset ∘ 2a` is my recommendation for
iteration 3, and the ordering of evidence matters: run them separately first (as
planned), because if this candidate's n-back regresses and 2a's does not, the
combination's n-back number is interpretable, whereas a combined-first run would
not have told you which half was load-bearing.

One combined-run hazard to note before anyone composes the two: `step()` here
raises `RuntimeError` if the store ever holds more than `_capacity()` keys. That
is dead code under this candidate alone (the baseline's `write_key` refuses), but
it is a hard crash mid-job rather than a degraded row if a 2a composition ever
transiently exceeds capacity. Deliberate — silent capacity growth is the failure
mode this search most needs to not have — but the composer should know it is there.

On `variable_mapping` I expect them to be **partly redundant**: displacement
already tripled the error count (12 → 35) by making the store lossier, and this
candidate makes the store consequential. Combined error counts should be
super-additive rather than additive, and P2's `rc_ratio` ceiling rises with error
count, so the combination is where a genuinely human-like `rc_ratio` near 1.386
first becomes arithmetically attainable.

### Unexplained and carried forward from iteration 1

Displacement's A3 BLEU rose 0.0031 → 0.0413 with recall length 121 → 146.
Nothing in this candidate can bear on it: story recall is on the byte-identical
list (P10), so this run supplies a second clean measurement of A3 at the baseline
value and nothing more.

---

## References

* Baddeley, A. D. (2000). The episodic buffer: a new component of working memory? *Trends in Cognitive Sciences* 4(11), 417–423. — the account considered and rejected (§2).
* Cowan, N. (2001). The magical number 4 in short-term memory. *Behavioral and Brain Sciences* 24(1), 87–114. — the existing `MAX_KEYS = 4` commitment.
* Logan, G. D., & Gordon, R. D. (2001). Executive control of visual attention in dual-task situations. *Psychological Review* 108(2), 393–434. — task set as a separately maintained control structure.
* Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences* 7(3), 134–140. — task set is procedural, retrieved from LTM, not held as chunks.
* Oberauer, K. (2002). Access to information in working memory: exploring the focus of attention. *JEP:LMC* 28(3), 411–421. — the concentric model; the account adopted.
