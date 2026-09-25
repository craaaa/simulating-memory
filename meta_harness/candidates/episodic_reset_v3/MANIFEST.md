# `episodic_reset_v3` — the history reset with the whole task set restored

Parent: `episodic_reset_v2` (iteration 3). Grandparent: `episodic_reset` (iteration 2).
Surface touched: `WorkingMemoryAgent.step()` only. `MAX_KEYS` 4, read live.
Written before the run. Nothing in `bench/`, `src/`, `data/` or `runs/` is modified.

---

## 1. What the run shows the n≥2 failure actually is

### The briefed hypothesis is refuted, four ways

I was asked to test first whether the prepended block competes with
`variable_mapping`'s "output ONLY one line … No extra text", with the competition
scaling in how much the block contains — which would explain why the failure tracks
store occupancy. **It does not survive the run data.** Each of the following kills it
on its own.

**(a) The block is byte-identical across levels on the turn where the failure is
already present.** On the first stimulus turn of an n-back block the agent is at
episode turn 2, presentations 1, store empty. Driving the *real* `run_nback_block`
offline at n=1, n=2 and n=3 under v2, the prepended block is **the same 278
characters at all three levels** (the whole `user_message` is 294). The share of rows
whose reply at that turn parses at all:

| level | parses at first stimulus turn | block |
|---|---|---|
| n=1 | 50/50 = **1.00** | 278 chars, identical |
| n=2 | 17/50 = **0.34** | 278 chars, identical |
| n=3 |  7/50 = **0.14** | 278 chars, identical |

A constant cannot explain a 7× difference.

**(b) There is no verbosity at all.** v2's reported reply length on
`variable_mapping` — mean 15.91 against baseline 13.10, max 147 — decomposes:

| arm | n | all replies | excluding `<tool_call>` replies |
|---|---|---|---|
| baseline | 1500 | mean 13.10, max 14 | 1500 rows, mean **13.10**, max 14 |
| `episodic_reset` | 1500 | mean 13.18, max 131 | 1499 rows, mean **13.10**, max 14 |
| `episodic_reset_v2` | 1500 | mean 15.91, max 147 | 1462 rows, mean **13.09**, max 14 |

Drop 38 rows and v2's mean reply length is **13.09 against the baseline's 13.10, with
the same maximum of 14.** The block induced exactly zero verbosity; the format
instruction was obeyed on every turn on which the model answered at all. **P7 failed
on a quantity that, decomposed, shows the opposite of what it was written to catch.**

**(c) All 38 of those rows are one thing, and it is not prose.** Every one is a
literal tool call emitted as plain text, e.g.

```
'<tool_call>\n{"name": "write_memory", "arguments": {"key": "Linda",
  "value": "Currently lives in Philadelphia"}}\n</tool_call>'
```

38 in v2, 1 in `episodic_reset`, 0 in the baseline. They occur on `variable_mapping`'s
**question** turns, which are `allow_tools=False`, i.e. the harness called the model
with `tool_choice="none"`. The model wanted to write to the store, was denied the
tool, and emitted the call **instead of the answer**. On n-back the same string
reaches `_parse_classification`, which returns `None`, and the trial is recorded as
unanswered.

**(d) The hazard is flat over turns, so it is neither an absorbing state nor
cumulative occupancy.** Rows parsing at each successive presentation (buffer
positions, then the 14 trials), out of 50:

```
baseline  n=3   50 50 50 |  7 14 36 49 37 44 21 15 21 20 20 15 21 21
v2        n=3    7 37  9 |  0  8 11  2  9  3 10  4 11  8 11 11  8 10
v2        n=2   17 48 23 |  9 22 23 22 25 20 27 22 25 34 27 32 29
v2        n=1   50 48 48 48 50 50 50 50 50 50 50 50 50 50 50
```

v2's n=3 rate sits at ≈0.18 from the first presentation to the last with no trend.
Answered-trial index sets: `scattered` 17/50, `contiguous-not-from-1` 21/50, `empty`
12/50, and the **first trial is unanswered in 50 of 50 rows**.
`episodic_reset`'s n=1 failure was a genuine absorbing state — 14 of 14 non-silent
rows answered a contiguous run *ending* at trial 14. This is the opposite shape.

The other alternatives the brief listed also leave the wrong traces. *"The store looks
like an answer and gets echoed"* and *"the agent reports the store contents"* would
produce long replies — refuted by (b). *"The turn ordinals make it believe it is still
in the buffer period"* would produce `"no response"`, which **parses** and is recorded
in `model_parsed_buffer`; at n=3 buffer position 3 v2 shows 41/50 `None` and **0**
`No response`, so the replies are unparseable, not compliant-but-early. *"The format
instruction is too far from the generation point"* cannot explain n=1 at 1.00 with the
same distance.

### What does set the rate: two things, one of them off limits

**(i) The `write_key` refusal loop — the baseline's own n=3 defect, and not mine to
fix.** Sorting every scored arm by whether `write_key` refuses when full gives a
**perfect separation** on n=3 `answered`, at essentially identical occupancy:

| refusing (baseline semantics) | n=3 answered | keys | evicting | n=3 answered | keys |
|---|---|---|---|---|---|
| baseline | 6.82 | 3.96 | `displacement` | **14.0** | 3.98 |
| `random_decay` | 6.56 | 3.94 | `primacy` | **14.0** | 4.00 |
| `random_decay_v2` | 5.26 | 3.92 | `primacy_v2` | **14.0** | 3.98 |
| `chunk_limit` | 5.82 | 3.86 | | | |
| `serial_recognition` | 5.22 | 3.86 | | | |

(`full_context` answers 14.0 at `keys_held` 1.06 — it never fills the store, so it is
not a counterexample.)

A refusing 4-slot store costs up to three tool calls to change one slot: the refused
write, a `delete_key`, then the write. `_tool_call_cap()` is
`max(6, int(1.5 * _tool_interactions))` — 1.5 calls per turn. Driving the real
`run_nback_block` offline with a stub that maintains an n-slot buffer,
`tool_call_cap_hit` (the base `step()`'s own flag for "budget exhausted **or** the
agent's tool calls truncated") fires on

```
n=1, 1 write/turn    0 of 16 steps   budget never reaches zero
n=2, 2 writes/turn  14 of 17 steps   zero budget on step 5
n=3, 3 writes/turn  16 of 18 steps   zero budget on steps 4 and 5
```

Perfect level-grading, the same ordering as the failure, and the state in which (c)
shows the model emits its tool call as text in place of the answer. Second reading,
worth recording on its own: **at n≥2 the harness is silently discarding part of the
agent's store update on most turns.**

**This is the honest ceiling on iteration 4.** The surface that separates 14.0 from
6.82 is the eviction surface, and the brief forbids it. So v3 cannot be expected to
reach `primacy`'s n=3 numbers, and the remaining gap after this candidate is
attributable to a defect it was not allowed to touch.

**(ii) What closing the leak does to (i), and what v2 added on top.** Closing the
leak forces the store to be genuinely maintained, so n=2 occupancy goes from the
baseline's 1.54 keys (5 of 50 rows at capacity) to v2's 3.58 (29 of 50) — and **v2's
n=2 `answered` of 6.80 is, to two decimals, the baseline's n=3 value of 6.82.** n=2
was pushed into the regime n=3 was already in. `episodic_reset` does that much too
(n=2 10.16 at 3.06 keys).

But v1 and v2 hold **the same 3.82 keys** at n=3 and answer **9.08 vs 2.12**. The
difference between them is three lines of text, and only one is an instruction:
`TASK_SET_REMINDER`, *"Update it each turn to track what you will need later."* It is
the last thing the agent reads before the stimulus, it names the store and nothing
else, and the reset has deleted the fourteen prior turns of its own replies that were
the only other place a per-turn response obligation was visible.

### Diagnosis

**The n≥2 failure is a task-set failure, not a memory failure.** At n=3, v2 is 0.6225
accurate on the trials it does answer — it can do the task. It omits the response.
And what it *was* told to do, it did: `keys_held` 3.82, letter identity 0.9843. It
complied with the one standing directive in view, and it was shown half of the task
set.

A note on key naming, since the brief's occupancy story invited it: v2's absolute
positional key names (`third_letter`, `seventh_letter`) look like an ordinal side
effect, but the **baseline** is worse — 198/198 of its n=3 keys are absolute
(`position_3`, `position_10`) against v2's 140/191. So the ordinals did not cause it
and it is not the mechanism.

---

## 2. The mechanism and its psychological justification

One rule, unchanged in scope from v2 and completed in content:

> **What crosses a turn boundary is the agent's control state — its position in the
> episode, its store, and the COMPLETE task set — and nothing about earlier stimuli.**

v2 stated half the task set. A task set is not a maintenance policy: it is the control
configuration that binds the current stimulus to a response (Logan & Gordon 2001,
executive control settings; Monsell 2003, task-set reconfiguration), and in a task
with a concurrent memory requirement it comprises **both** the response mapping and
the maintenance requirement. v2 carried the maintenance requirement across the
boundary, dropped the response requirement, and deleted the transcript that was the
response requirement's only other carrier.

Three deltas, all inside the block, nothing else anywhere:

1. the no-transcript notice and the maintenance imperative are **separated**. v2
   welded them into one sentence, which made the only imperative in the block a store
   imperative;
2. a **response obligation** is stated, task-agnostically — "the response this turn's
   instructions call for", so an n-back buffer period (instructed response: *no
   response*) and a `variable_mapping` encoding turn (no instructed response) are both
   covered without naming either;
3. the response obligation is made **explicitly independent of the store operation
   succeeding** — "if the store cannot be changed, or needs no change, respond
   anyway". This is the sentence aimed at finding (c).

**Nothing is removed.** Everything v2 said, v3 also says. The claim is that the
block's *content* was incomplete, not that its *form* was excessive — and (b) is why:
there was no verbosity to economise on. The brief invited the opposite argument; the
data does not support it.

### Why the added line needs no new exemption

Oberauer (2002)'s concentric model exempts task set and activated long-term memory
from the ~4-chunk limit; task set is procedural and retrieved rather than held
(Monsell 2003; Logan & Gordon 2001). A stimulus–response mapping is the textbook
content of a task set, so the added line is the *most* clearly exempt thing in the
block — more clearly than the ordinals, which needed v2's harder argument.

It is also what a human participant has continuously and for free. The n-back display
asks for a keypress on **every** letter, and the fact that it does is on the
block-onset screen, not in the participant's memory for the letters. A harness that
deletes its transcript deletes that, and the deletion is an implementation artifact,
not a capacity limit. `keys_held` 3.82 with `answered` 2.12 is what the artifact looks
like: the agent maintaining memory perfectly and not knowing it owes a response.

### What the ordinals' exemption rests on (unchanged from v2, restated)

`t` and `p` are functions of the method-call count alone — computed without reading
`user_message`, the store, or any stimulus — so they carry **zero bits** about which
stimuli appeared, cannot substitute for the store and cannot be a back door for the
leak. Every serial-order model of working memory posits exactly such an
always-available signal outside the item store: drifting temporal context (Howard &
Kahana 2002), an oscillator-based positional code (Brown, Preece & Hulme 2000), a
loop-position timing signal (Burgess & Hitch 1999), temporal distance as the
discriminative dimension (Brown, Neath & Chater 2007).

### What is deliberately NOT touched, and why the obvious lever is left alone

`_tool_call_cap()` is **not** overridden, although §1(i) names it as the proximate
cause of the `tool_choice="none"` turns. Two reasons, both stated before the run:

* **More budget lets the agent churn the store faster but not hold more.**
  `variable_mapping`'s humanlike error rate comes from capacity, not budget: 20
  assignments against 4 slots, with **674 of 1500 questions asking about a name the
  store had already evicted**. Raising the cap cannot buy those errors back and risks
  the headline gain to purchase a quantity it cannot purchase.
* It would make this candidate two mechanisms and destroy attribution.

Instead the budget is **measured**: P9 reports per-level `tool_call_cap_hit` and
zero-budget shares from the newly persisted n-back `step_log`, against the stub
figures above, with no threshold. That is iteration 5's decision input, and this is
the first run in the project able to produce it.

Also untouched: `WorkingMemory` (not subclassed — `write_key`, overflow policy and
eviction order are literally the baseline's), `recall()` including the
`word_recognition` trial-list presentation, key naming, value formatting, the prompts,
the tools. No RNG, no swept constant: the block is fully determined by the two call
counts and the store.

---

## 3. Why it is general, and what it does to `variable_mapping`

No task is named anywhere in `harness.py` and no branch tests for one. The block is
the same text on every `step()` of every task. Verified offline on the real loops:

* **`variable_mapping` question turns** (`allow_tools=False`; the task's own
  `QUESTION_PROMPT` already renders the store, so the block does not — asserted, store
  header appears exactly once): three lines become four, 422–424 chars. The 38
  unparsed answers should go to ≈0, which **raises** the model's score on ≈38 of 1500
  questions and therefore **lowers** `variable_mapping` humanlikeness slightly,
  because the model is already far better than the humans there (baseline 0.3554 with
  a near point mass at ceiling). **A fall from 0.6854 is predicted and accepted; a
  collapse is not.** The bulk of the 550 errors is capacity-bound, so the error
  structure should survive: P4 and P5 hold it to ≥150 errors and
  `rc_ratio_normalized` in [0.15, 0.95].
  One structural note: on these turns the block cannot be adjacent to the stimulus —
  the task's own ~600 characters follow it, ending in "Output ONLY one line … No extra
  text." That is the *last* thing the model reads there, which is the right ordering
  for this task and is why (b)'s zero-verbosity result is expected to persist.
* **`variable_mapping` encoding turns** (484–486 chars): the agent is told a response
  is due when the task asks for none. The task discards that text, so the cost is
  tokens, not score.
* **The six `encode()` → `recall()` tasks: untouched, structurally.** `encode()` calls
  `step()` exactly once, so the episode turn index is 1 and **no block is prepended**;
  `recall()` is not overridden at all. Asserted offline. This is the same property
  that made v2 move `craft_task`, `narrative_qa` and both digit spans by exactly
  0.0000.

---

## 4. Prompt delta, and why forced

### The block, verbatim (n-back, first stimulus turn, store empty)

```
[ongoing episode]
Turn 2 of this episode. Stimulus presentations so far: 1.
You have no transcript of earlier turns: the key-value store is your only record of them.
Your working memory currently contains:
(memory is empty)
Two things are due on every turn of an episode. Update the store to track what you will need later, and give the response this turn's instructions call for. The response is due either way: if the store cannot be changed, or needs no change, respond anyway.

Next letter: R
```

### Lengths, measured by driving the real loops (not estimated)

| | v2 | v3 |
|---|---|---|
| first stimulus turn, n=1 / n=2 / n=3 | 278 / 278 / 278 | **480 / 480 / 480** |
| all n-back turns, mean (max), n=1 | 272.3 (278) | 474.3 (480) |
| all n-back turns, mean (max), n=2 | 282.7 (284) | 484.7 (486) |
| all n-back turns, mean (max), n=3 | 293.1 (295) | 495.1 (497) |
| `variable_mapping` encode turns | — | 484–486 |
| `variable_mapping` question turns | — | 422–424 |
| n-back step 1 (instruction turn) | no block, 286 chars | no block, 286 chars |

So the block is **byte-identical across levels at the first stimulus turn** in both
harnesses, and spans only 4% across levels overall (474 → 495). v3's block is ~202
characters longer than v2's.

**Why the length is forced, and why it is not a risk I am hiding.** The response
obligation must be unambiguous about the case that produced the failure — the turn on
which the store cannot be written — and that case needs a clause of its own. The
length hypothesis was the briefed one and I refuted it with (a) and (b); the honest
thing is therefore to *stake a prediction on my own refutation*. P7 does exactly that:
if v3's block induces verbosity, `variable_mapping` mean reply length over
non-`<tool_call>` rows rises above 13.5 from v2's 13.09, and P7 fails. Nothing in the
design protects me from that.

---

## 5. Why the leak stays closed and n=1 stays fixed

**The leak, by construction.** `reset_messages()` is called at the top of every
`step()`, exactly as in v2. Everything the agent sees on turn *k* is: the system
prompt, the tool schemas, two integers derived from call counters, fixed English
identical on every task and every turn, the store, and the current `user_message`.
`step()` retains no reference to any previous `user_message`, so no earlier stimulus
can reach turn *k* except through the four slots. Asserted offline at all three
n-back levels: every first LLM request of a step is exactly `[system, user]`, no
request ever carries more than one user message, and no earlier presentation string
appears in any block. Registered as P3 (n=3 letter-identity share ≥ 0.90; baseline
0.0202, v2 0.9843).

**n=1, by retention.** `episodic_reset`'s n=1 failure was an absorbing state created
by its `if store` guard: it prepended nothing when the store was empty, so turn *k*
was a pure function of the current letter, and escape was carried by one letter (C, 11
of 15 empty-store presentations). v2 fixed it with two things, both kept verbatim
here: the store is rendered **unconditionally**, including as `(memory is empty)`, and
the ordinals **strictly increase**, so no two turns of an episode are ever the same
request. The added response obligation can only push n=1 in the direction it already
succeeds in (13.88 answered, 0.9964 accurate, 0 silent, `keys_held` 1.00).

---

## 6. Pre-registered predictions

Machine-readable in `meta_harness/logs/pending_episodic_reset_v3.json`, with a
`capable_of_varying` field and its evidence on every row. Baseline run:
`meta_harness/runs/iter0/baseline/Qwen_Qwen3-30B-A3B-Instruct-2507`.

### The precondition is pitched at restoration, not at ambition — deliberately

`episodic_reset` lost a real +0.32 because its precondition was pitched where it
hoped to land. The lesson is not "use no precondition"; it is "a precondition asserts
the candidate is not broken, and the ambition goes in a separate scored row".
So P1 is set at *"restores the predecessor that did not have this defect"* — v1's n=2
value of 10.16, the baseline's n=3 value of 6.82 — with the strong claims registered
separately as P1b, which cannot void anything.

| id | quantity | source | baseline | v2 | threshold |
|---|---|---|---|---|---|
| **P1** | **PRECONDITION** — n-back at every level | `nback_levels.report()['diagnostics']` | | | |
| P1·n1 | answered / acc_over_answered / n_no_answers / keys_held | " | 13.98 / 0.9943 / 0 / 1.00 | 13.88 / 0.9964 / 0 / 1.00 | ≥13.0 / ≥0.90 / **==0** / ≥0.8 |
| P1·n2 | " | " | 13.24 / 0.8206 / 0 / 1.54 | 6.80 / 0.6691 / **3** / 3.58 | ≥9.5 / ≥0.60 / **==0** / ≥1.0 |
| P1·n3 | " | " | 6.82 / 0.7372 / 0 / 3.96 | 2.12 / 0.6225 / **12** / 3.82 | ≥6.0 / ≥0.50 / **==0** / ≥2.0 |
| P1b | strong claim, not a precondition | " | — | — | n=2 answered ≥12.0, n=3 ≥10.0 |
| **P2** | `nback` humanlikeness clears its floor | `humanlikeness_by_task` | 0.7909 | 0.5601 | **≥0.7309** |
| **P3** | leak stays closed: n=3 share of `final_kv` values carrying a bare capital letter | `check_predictions._nback_letter_share(run_dir, 3)` | 0.0202 | 0.9843 | ≥0.90 |
| **P4** | `variable_mapping` raw / matched | `humanlikeness_by_task`, `axes.variable_mapping_matched` | 0.3554 / 0.3587 | 0.6854 / 0.7568 | ≥0.55 / ≥0.60 |
| **P5** | A4 `n_errors` / `rc_ratio_normalized` / `trustworthy` | `axes.A4` | 12 / 0.6661 / true | 550 / 0.7362 / true | ≥150 / in [0.15,0.95] / true |
| **P6** | parse integrity on `variable_mapping`: unparsed answers, and replies containing `<tool_call>` | `step_logs[].parsed` / `answer_raw` | 0 / 0 | **38 / 38** | ≤5 / ≤5 |
| **P7** | verbosity: mean reply length over ALL rows, and over non-`<tool_call>` rows; max over non-`<tool_call>` rows | `step_logs[].answer_raw` | 13.10 / 13.10 / 14 | 15.91 / **13.09** / 14 | ≤14.5 / ≤13.5 / ≤20 |
| **P8** | no-change control, six `encode()`→`recall()` tasks | `delta_vs_baseline` | 0 | 0.0000 ×4, −0.0005, +0.0071 | \|Δ\| ≤ **0.008** for `digit_span_reverse`, `word_recognition`, `semantic_story_recall`, `craft_task`; ≤ **0.016** for `digit_span_forward`; ≤ **0.02** for `narrative_qa` |
| **P9** | budget and reply-form decomposition from the NEW `step_log` | `wm_nback.jsonl[].step_log` | — | field absent | **REPORTED** + two thresholds, below |
| **P10** | `buffer_no_response_frac` rises at n=2 and n=3 | `nback_levels…['buffer_no_response_frac']` | 0.92 / 0.39 / 0.3333 | 0.86 / 0.17 / 0.1933 | n=2 ≥0.30, n=3 ≥0.28 |

**P1 failure invalidates P1b and P2–P10 regardless of their values**, exactly as
`episodic_reset`'s P9 and v2's P1 did. `n_no_answers == 0` is kept strict at all three
levels: it is the cleanest mechanism-specific bite available (v2 read 0/3/12) and the
baseline achieves it everywhere.

### P9 in detail — what `episodic_reset_v3_checks` should assert

`bench/tasks/wm_nback.py` now persists `step_log` per row (`user_message`, `text`,
`tool_calls`, `kv_snapshot`, `tool_call_cap`, `tool_calls_used_total`,
`tool_call_budget_before`, `tool_call_budget_after`, `tool_call_cap_hit`). Iteration
3's runs lack it; this one will have it. Per n-level, over the 15/16/17 stimulus
turns of each row (i.e. `step_log[1:]`):

*Thresholded legs of P9 — these are the mechanism-confirmation rows:*
1. **share of turns with empty `text`** (`text.strip() == ""`): **≤ 0.05** at every
   level. This is the "spent the turn on the memory update" surface form.
2. **share of turns whose `text` contains `"<tool_call>"`**: **≤ 0.05** at every level.
   This is the surface form measured directly on `variable_mapping` in §1(c).

*Reported, not thresholded:*
3. `tool_call_cap_hit` share per level, and share of turns with
   `tool_call_budget_before == 0`. Compare against the offline stub figures 0/16,
   14/17, 16/18. **This is the decision input for iteration 5 on whether
   `_tool_call_cap()` is the remaining binding constraint.**
4. `tool_calls` per turn per level, mean and max, and how many are `write_memory`
   whose `result` starts with `"Error: memory is full"` — the refusal-loop rate, which
   §1(i) identifies as the surface this candidate is forbidden to repair.

*A contamination check, which must be asserted or the precondition can be satisfied by
garbage:* `_parse_classification` regexes the whole reply, so a `<tool_call>` string
containing e.g. `"value": "Same as previous"` would parse as a classification. Assert
that **the count of `step_log` entries where `text` contains `"<tool_call>"` AND
`bench.tasks.wm_nback._parse_classification(text)` is not `None` is ≤ 2 per level.**
If it is larger, P1's `answered` is contaminated and P1 must be read as INCONCLUSIVE
rather than PASS.

### How P2's threshold was sanity-checked, and what decides it

`NOISE_FLOOR["nback"] = 0.060`, `FLOOR = 0.03`, so the enforced floor is 0.060 and
P2's threshold is 0.7909 − 0.060 = **0.7309**. Measured run-to-run variation on
`nback` is **0.0023** (`logs/run_to_run_floor.json`), so the floor is ~26× the noise.

Fitting the recorded arms, `nback` per-row humanlikeness is well approximated by
`HL ≈ 1 − mean_level|acc_over_14 − 0.866| + c`, with residual `c` measured at
+0.031 (baseline), +0.040 (v2), +0.039 (`full_context`), −0.005 (`episodic_reset`).
Applying it:

| scenario | n=1 / n=2 / n=3 `acc_over_14` | predicted HL |
|---|---|---|
| recovers to v1's n≥2 with n=1 kept | 0.99 / 0.55 / 0.43 | ≈0.738 — clears by 0.007 |
| recovers to baseline | 0.99 / 0.78 / 0.36 | 0.7909 by construction |
| response obligation lands fully | 0.99 / 0.70 / 0.50 | ≈0.81 |
| obligation only partly lands | 0.99 / 0.45 / 0.30 | ≈0.66 — **fails** |

**n=2 is what decides P2**, because n=2 is where leak closure newly saturates the
store (1.54 → 3.58 keys, 5 → 29 of 50 rows at capacity). The approximation is stated
here so the row can be read against it after the run rather than reconstructed, and it
is *not* used to justify any threshold: P2 is the floor, which is fixed by the
contract.

### Accepted costs, stated in advance

* `variable_mapping` is expected to **fall** from 0.6854 — recovering the 38 unparsed
  answers raises the model's score on a task where it is already better than humans.
  P4's floor is 0.55 raw / 0.60 matched; the enforced per-task floor against the
  baseline is 0.3554 − 0.030, which a fall to 0.55 does not approach.
* `nback` is **not** expected to reach `primacy`'s 0.9454. The eviction surface caps
  it (§1(i)) and that surface is out of scope this iteration.
* Token cost per n-back turn rises by ~200 characters of prompt.

### What is deliberately not predicted

* Which way `narrative_qa` moves. Its measured run-to-run spread reaches **0.0160**
  against a 0.030 floor, so no narrative effect below ~0.02 is claimable by anything.
  P8's band is **per-task, not uniform**, for exactly this reason: 0.008 is a
  *structural* no-change claim (the block is never prepended on those tasks), but
  `digit_span_forward` moved +0.0152 and `narrative_qa` up to 0.0160 between two runs
  of one harness, so holding either to 0.008 would make the checker emit a mechanism
  FAIL on measured noise. They get 0.016 and 0.02.
* Bit-identity of any generated text. The serving stack is not row-deterministic (15
  bit-identical `"Next letter: C"` requests split 11/4), so every row here is a scored
  statistic or a rate, never a text comparison.
* A2, A1, A3. Nothing in this candidate touches `recall()` or `encode()`, so they are
  expected unmoved; they are covered by P8's scored deltas and the standing guards.

### Corrections to the brief this candidate makes

1. **The prompt-competition hypothesis is refuted** — §1(a)–(d). Specifically, P7's
   "verbosity" is 38 tool-call strings; excluding them, v2's mean reply length is
   **13.09 against the baseline's 13.10** with the same maximum. The brief's leading
   hypothesis predicts the opposite of what the data shows.
2. **The failure is not an absorbing state and does not track occupancy over time.**
   It is a flat per-turn hazard whose rate is set by the level; v2's n=3 first trial
   is unanswered in 50 of 50 rows.
3. **"`craft`, `narrative`, BOTH digit spans moved by EXACTLY 0.0000" is true of four
   of the six no-change tasks, not six.** v2's `word_recognition` moved −0.0005 and
   `semantic_story_recall` +0.0071. P8's band is set at ±0.008 accordingly.
4. **The `write_key` surface the brief rules out is the one that caps this
   candidate.** The refusing/evicting split is a perfect separation on n=3 `answered`
   at identical occupancy (6.82/6.56/5.26/5.82/5.22 vs 14.0/14.0/14.0). It is not a
   dead end — it is the live lever — but it is correctly out of scope for a
   single-mechanism iteration, so iteration 5 should return to it with
   `primacy_v2`'s batch condition dropped.
5. **`tool_call_cap_hit` fires on 14 of 17 and 16 of 18 n-back steps at n=2 and n=3**,
   meaning the harness silently truncates part of the agent's store update on most
   turns at those levels. This was not previously recorded anywhere.

### Offline verification actually performed

* `python meta_harness/verify_interface.py meta_harness/candidates/episodic_reset_v3/harness.py`
  → **PASS** (overrides `WorkingMemoryAgent`, `SummarizerAgent`, `MAX_KEYS`,
  `CONDITION_PROMPTS`, `TOOLS`; rebound in 8–10 modules; round trip stored 2 keys;
  capacity enforced at 4).
* Stub-LLM drive of the **real** `run_nback_block` at n=1, n=2 **and** n=3, and of the
  **real** `variable_mapping` encode/question loop: **0 assertions failed.**
  Asserted per turn: step 1 carries no block and is the task's verbatim instruction
  message; every later turn carries a block; the turn and presentation ordinals are
  exactly `i` and `i−1`; the response obligation is the **last** line of the block; the
  store is rendered exactly once (block on n-back and on `variable_mapping` encode
  turns, task's own `QUESTION_PROMPT` on question turns, never both); the stimulus is
  the last thing in the message; no earlier presentation string appears in any block;
  every first LLM request of a step is exactly `[system, user]`; no request ever
  carries more than one user message. The same script run against
  `episodic_reset_v2` fails **only** the response-obligation assertions (67 of them)
  and passes every other invariant — which confirms the test discriminates and that v3
  changes exactly one thing.
* **Not tested offline, and stated as such:** whether the real model actually answers
  on a `tool_choice="none"` turn when told to. A stub cannot settle that. It is the
  candidate's central behavioural bet and P6, P7 and P9's two thresholded legs are
  written to catch it failing.
