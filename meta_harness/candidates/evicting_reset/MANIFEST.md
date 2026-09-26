# `evicting_reset` — iteration 5

Parent: `episodic_reset_v3` (mean **0.8163**, one floor violation: `nback` 0.6978
against 0.7309, and P1 FAILED on merit so all eleven rows below it are VOID).

**One change, and it is one method body.** `WorkingMemory.write_key`. `step()` is
copied from v3 character for character, so the **prompt delta is zero**.
`_tool_call_cap()`, `encode()`, `recall()`, `reset_messages()`, `snapshot()`,
`to_recall_text()`, `TOOLS`, `CONDITION_PROMPTS` and `WM_SYSTEM_PROMPTS` are
untouched. `MAX_KEYS` stays 4 and is read live.

---

## 1. What the second cause is, from the v3 `step_log`

I was asked where n=2's missing ~6 answered turns go. They are fully accounted
for, and the shape of the loss is a **period-2 oscillation** that the WORKLOG's
pooled cross-tabulation averages away.

Per-stimulus-turn answer rate, v3 run, 50 blocks per level (`b0` = share of turns
ending with zero tool-call budget):

```
n=1  1.00 1.00 1.00 0.82 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
b0   0.00 0.00 0.18 0.58 0.66 0.66 0.68 0.60 0.68 0.56 0.62 0.38 0.48 0.36 0.50

n=2  1.00 1.00 1.00 0.92 0.86 0.24 0.72 0.16 0.68 0.20 0.74 0.32 0.76 0.26 0.80 0.30
n=3  1.00 1.00 1.00 1.00 1.00 0.00 0.66 0.00 0.90 0.00 0.96 0.00 0.96 0.00 1.00 0.00 1.00
```

At n=3 the rate is **exactly 0.00 on every even turn from turn 6 on**, over 50
blocks at temperature 0, and 0.66–1.00 on every odd turn. Summing the n=2 row over
the 14 scored turns (turns 3–16) gives **7.96**, against the reported `answered` of
**7.98**. So the missing turns are one alternating half of the block and nothing
else is missing. The accounting is closed.

The cycle, read off one n=2 block's `step_log` directly:

```
turn 4  budget_before 2   write last_two_letters (overwrite) -> "Key written."
                          write current_letter   (NEW key)   -> "memory is full"
                          budget_after 0, cap_hit
                          step() switches to tool_choice="none"
        reply = '<tool_call>{"name":"delete_key","arguments":{"key":"second_letter"}}</tool_call>'
        -> no classification parsed, trial unanswered
        -> AND THE DELETE NEVER EXECUTES, because it is text, not a tool call,
           so the store is still full next time
turn 5  budget_before 1   write last_two_letters (overwrite) -> OK
        reply = 'different'                                  -> answered
turn 6  budget_before 2   identical to turn 4.  Hence period 2.
```

**Correction to the brief's conditioning variable: the discriminator is the
refusal, not the budget.** Splitting the same turns by whether the turn contained a
memory-full result:

| level | b=0 & refusal | b=0 & no refusal | b>0 & refusal | b>0 & none |
|---|---|---|---|---|
| 1 | n=0 — | n=347 **P=0.974** | n=0 — | n=403 P=1.000 |
| 2 | n=363 **P=0.174** | n=195 **P=0.990** | n=5 P=1.000 | n=237 P=1.000 |
| 3 | n=374 **P=0.134** | n=276 **P=0.993** | n=0 — | n=200 P=1.000 |

An exhausted budget with no refusal answers 0.990 / 0.993 — indistinguishable from
n=1's 0.974. An exhausted budget *with* a refusal answers 0.174 / 0.134. The budget
is a necessary accessory (it is what removes the tool and forces the model to speak
its repair) but it is not the binding constraint. The demand is 1.48 / 1.47 calls
per turn against a supply of 1.5, and the refused write itself consumes one of them
— `_tool_calls_used += 1` fires in `wm_agent.step()`'s dispatch loop regardless of
what `_dispatch_tool` returned. Remove the refusal and demand drops below supply on
the exact turn it mattered.

**This makes the `_tool_call_cap()` prohibition a measured result rather than a
judgement call**, which is worth recording because iteration 4 argued it on grounds
of risk. On the 471 turns where the budget was exhausted and nothing was refused,
the agent answered 99.1% of the time. More budget buys nothing on the failing cell.

So the two causes separate, and neither is the other:

| | cause | fixed by | evidence |
|---|---|---|---|
| **A** | v2's block carried the maintenance obligation across the turn boundary and dropped the **response** obligation | v3's third sentence | n=3 answered 2.12 → 7.44; vm unparsed 38 → 0 |
| **B** | a refused write costs the last tool call, the agent then spends the turn saying its repair out loud instead of answering, and the repair never lands | **not reachable by any instruction** — the model is denied the tool at the moment it wants it | the table above |

---

## 2. The mechanism — which rule, and why it is episode-conditional

### The obvious construction is measured to fail

"v3's `step()` plus one of the two validated evicting stores" trades the n-back
violation for a different one. Effective floors are `max(0.03, NOISE_FLOOR[task])`:

| rule | `semantic_story_recall` | `craft_task` | `narrative_qa` | `nback` |
|---|---|---|---|---|
| baseline (refuse) | 0.9473 | 0.8907 | 0.9572 | 0.7909 |
| `displacement` (LRU) | 0.8964 **−0.0509 VIOLATION** | 0.8627 −0.0280 | 0.9483 −0.0089 | 0.9421 |
| `primacy` (ACT-R base level) | 0.9477 +0.0004 | 0.8456 **−0.0451 VIOLATION** | 0.9263 **−0.0309 VIOLATION** | 0.9454 |

Both clear n-back; each breaks a different batch task. And on `craft_task` the
cause is **not** the victim-selection order:

```
mean keys held on craft_task:  3.33 refusing  ->  3.67 under EITHER evicting rule
```

A refusing store makes the agent burn tool calls on delete-and-rewrite repairs that
`tool_calls[:remaining]` truncates, so it ends up holding *fewer* chunks. Admitting
those writes hands a batch task more retained material, and since the objective
inverts, more retention is less humanlike. **Every unconditional evicting store pays
that**, so "remove the refusal" as stated cannot be the whole candidate. This is the
reason the rule is conditional, and it is a measured dead end rather than a
preference.

### The rule

A write to a **new** key on a full store:

* **displaces** the least recently refreshed resident that was laid down in an
  **earlier presentation**, and is admitted;
* is **refused**, with the baseline's exact message, when every resident belongs to
  the **current presentation**.

A presentation is one `step()` call, identified by `len(owner._step_log)`. An
overwrite counts as a re-presentation and refreshes. One rule, no task named, no
branch on task, no fitted constant, nothing swept.

### Which account this implements

**Across presentations — displacement.** A stimulus arrives whether or not there is
room. The incoming chunk enters the focus of attention and the least active resident
leaves; declining it would mean the stimulus was not perceived, and no account of
human working memory contains that state. Cowan (2001), the ~4-chunk focus of
attention; overflow-as-overwriting in the interference accounts (Oberauer & Kliegl
2006; Oberauer, Farrell, Jarrold & Lewandowsky 2016); the removal operation in
updating (Ecker, Lewandowsky & Oberauer 2014). The victim is the least recently
refreshed because **running-memory span and n-back show recency without primacy**
(Pollack, Johnson & Knaff 1959; Bunting, Cowan & Saults 2006) — which is why no
activation machinery and no decay constant are needed here, and `primacy`'s own
docstring already argues exactly this asymmetry.

**Within one presentation — a selection cap.** The material is still in front of the
participant and the order of encoding is theirs. What binds is not displacement but
**selection**: how many chunks may be carried forward (Miller 1956 on recoding;
Cowan 2001 on the chunk limit as a limit on what is *held*). "Memory is full" is
then not a refusal to perceive a stimulus — the stimulus is present regardless — it
is the report that the four-label index of that material is spent, and the agent
answers it by overwriting or deleting, which is what it actually does (craft's
observed pattern: `w1..w4` written, `w5` refused, `delete_key` issued, rewrite).
**This is the only reading under which the baseline's error string is
psychologically admissible, and this candidate keeps it exactly there and nowhere
else.**

### Its mapping onto the task set is a consequence, not an input

```
all writes inside ONE step()      -> selection cap, i.e. THE BASELINE
    semantic_story_recall, craft_task, narrative_qa, word_recognition,
    digit_span_forward, digit_span_reverse
    (all six are encode() -> recall(); encode() calls step() exactly once and
     recall() bypasses step() entirely)

writes across MANY step() calls   -> displacement
    nback, variable_mapping
```

and the rule therefore has **its own falsifier**, which an unconditional store does
not: if any of the six batch tasks moves beyond its identical-path band, the rule
has leaked out of the across-presentation path and the composition is wrong.

---

## 3. How `episodic_primacy`'s 3.90 is accounted for

Two things, and the second is a defect in the parent line.

**(a) Cause A is a ceiling, not an addend.** v2 did not ask for a response, so the
answer rate is capped by the task-set failure however the store behaves.
`episodic_primacy` fixed B and left A (3.90); v3 fixed A and left B (7.44). There is
no additivity to assume and none is assumed — this is simply the first arm in which
both are fixed. The claim being made is not "the two gains add"; it is "A caps and B
is periodic, and with A fixed, removing B restores the alternating half of the
block that the period-2 measurement says is the whole of the loss."

**(b) `episodic_primacy` did not contain `primacy`'s rule.**
`PrimacyMemory._episode_mark()` — and `PrimacyV2Memory`'s, verbatim — counts
`role == "user"` messages in `owner._messages` to detect a presentation boundary.
Under `episodic_reset_v2`/`v3`'s `step()`, which calls `reset_messages()` on entry,
that count is **1 on every turn**. So the mark never changes, `_sync_episode()`
never advances, and every write in an entire n-back block lands in **one** episode:
residents accrue unbounded rehearsal credit and the rule is not the published one.

The comment beside it reads *"Monotonic, so CLEARING the message history advances
the episode rather than colliding with an earlier one. This is what makes the
candidate safe to compose with a `reset_messages()` rewrite."* That is backwards.
Monotonicity prevents a collision with an earlier episode; it does nothing about a
mark that never moves.

Measured offline, replaying the real n-back write sequences from the v3 run through
`PrimacyMemory` with an advancing mark and with a frozen one:

```
n=2   final key set differs on 18 of 50 blocks;  evictions 246 -> 155
n=3   final key set differs on 17 of 50 blocks;  evictions 329 -> 201
```

So `episodic_primacy`'s `nback` 0.5893 is a measurement of an **untested third
store**, and it is weaker evidence against composition than the brief treats it as.
This candidate has no `_messages` dependence anywhere: the presentation mark is
`len(self._step_log)`, the same quantity v3 already uses for
`_episode_turn_index()`, and it is invariant to history clearing.

---

## 4. Prompt delta, and why it is forced

**Zero characters.** `EPISODE_MARKER`, `NO_TRANSCRIPT_NOTICE`,
`STANDING_OBLIGATIONS`, `WM_STATE_HEADER`, `_control_state_block()` and `step()`
are v3's, character for character, and nothing is imported from the sibling
candidate (`inject.load_candidate` + `apply` is not idempotent — a second load
subclasses the already-injected class and stacks the overrides, which was observed
emitting the control-state block twice in one message).

It is forced by the diagnosis in section 1: cause B is not reachable by
instruction. v3's obligation sentence already says *"The response is due either
way: if the store cannot be changed, or needs no change, respond anyway"* — the
strongest available form of the instruction — and the model still emits the repair
as text, because on that turn it has no write tool and the only channel it has is
the text channel. No third sentence can create a tool call.

`snapshot()` and `to_recall_text()` are also deliberately **not** overridden, unlike
`displacement`'s, which reordered the read-out by recency and tagged the newest
entry. Under this `step()` the store is rendered into the user message on every
turn, so reordering it would be a prompt change wearing an overflow change's
clothes. The store's *contents* change; its *rendering* does not.

---

## 5. Why the leak stays closed and n=1 stays fixed

**The leak.** `reset_messages()` is copied verbatim and sits upstream of the store,
so no earlier stimulus, reply or tool result survives the turn boundary and the only
route from letter *k−n* to the answer at *k* is the store. v3 measured n=3
letter-identity share 0.0202 → 1.0000 and nothing here touches that path. If
anything the store now carries *more* letter identity, because a write that was
refused is admitted.

**n=1 is the built-in control, structurally rather than hopefully.** At n=1 the
store never overflows: **0.00 memory-full results per turn over all 750 v3 stimulus
turns**. A rule that changes only what happens on overflow cannot reach a store that
never overflows. So n=1 must stay at 13.98 answered with zero silent, and if it
moves, something other than the overflow rule changed.

**A correction the record needs.** "n=1 stays fixed" is true of v3's `answered`
(13.98) and of its zero silent rows, but **not of its accuracy**:

| | `answered` | `acc_over_answered` | `keys_held` | `memory_full`/turn |
|---|---|---|---|---|
| baseline n=1 | 13.98 | **0.9943** | 1.00 | 0.00 |
| v3 n=1 | 13.98 | **0.8497** | 2.06 | 0.00 |

That −0.1446 was not flagged in iteration 4. It is not an overflow effect — the
store still never fills — it is the rendered store block changing the agent's n=1
strategy from one overwritten key to two. This candidate inherits it and cannot fix
it, so **P1's n=1 accuracy leg is pitched at v3's 0.8497, not the baseline's
0.9943**, and a precondition pitched at the baseline there would fail on an
inherited cost. Same lesson as v3's own P1, in a different cell.

---

## 6. Pre-registered predictions

Registered in full, with sources and thresholds, in
`meta_harness/logs/pending_evicting_reset.json`. Summary:

| id | row | quantity | threshold | v3 | role |
|---|---|---|---|---|---|
| **P1** | n-back mechanics at all three levels — **PRECONDITION** | `answered`, `acc_over_answered`, `n_no_answers`, `keys_held` | answered ≥ **13.0 / 11.0 / 9.0**; `n_no_answers == 0` at all three; `acc_over_answered` ≥ **0.80 / 0.55 / 0.45** (n=1 pitched at v3's 0.8497, not the baseline's 0.9943 — see section 5) | 13.98 / 7.98 / 7.44, 0/0/0 silent, acc 0.8497 / 0.8341 / 0.7513 | voids everything below on failure |
| **P2** | full restoration — **STRONG CLAIM, non-voiding** | `answered` | ≥ 13.5 at all three | — | cannot invalidate |
| **P3** | `nback` clears its floor | humanlikeness | ≥ **0.7309** | 0.6978 | primary |
| **P4** | **the mechanism is confirmed, not assumed** | `memory_full` results per turn; `P(answer \| budget=0)` | memory_full ≤ **0.01** per turn at all three levels **and every event within-turn**; `P(answer\|b=0)` ≥ **0.97** at all three | 0.00 / 0.4775 / 0.4682; 0.9741 / 0.4588 / 0.4985 | primary |
| **P5** | the leak stays closed | n=3 letter-identity share in `final_kv` | ≥ 0.90 | 1.0000 (baseline 0.0202) | primary |
| **P6** | `variable_mapping` holds | raw, matched, A4 `n_errors`, `rc_ratio_normalized` | ≥ 0.55 raw, ≥ 0.60 matched, `n_errors` ≥ 150, `rc_ratio_normalized` ∈ [0.15, 0.95] | 0.6767 / 0.7252 / 505 / 0.7284 | primary |
| **P7** | parse integrity | vm unparsed; `<tool_call>`-in-text | ≤ 5 of 1500 each | 0 / 0 | primary |
| **P8** | batch-task control, **per-task identical-path bands** | six `encode()→recall()` tasks vs **v3** | craft ±0.030, narrative ±0.020, story ±0.020, word_rec ±0.050, dsf ±0.020, dsr ±0.010 | — | primary |
| **P9** | **anti-`full_context`** | `keys_held`, `acc_over_answered`, A2 | n=3 `keys_held` ≥ 3.5 and n=2 ≥ 2.0; n=3 `acc_over_answered` ≤ 0.85; `slot_utilization` ≥ 0.85 at n=3 | — | primary |
| **P10** | buffer-period compliance | `buffer_no_response_frac` | rises at n=2 and n=3 | — | reported |
| **H1** | `craft_task` — **named non-voiding hazard** | delta vs baseline | flagged, not failed, if ∈ [−0.045, −0.030] | v3 −0.0248 | hazard |

### What `evicting_reset_checks` should assert, and from which fields

The coordinator wires the dispatch; this section says what to read. All n-back rows
come from `meta_harness.nback_levels.report(run_dir)`.

* **P1** — `report["diagnostics"][n]` for `n` in 1,2,3: keys `answered`,
  `acc_over_answered`, `n_no_answers`, `keys_held`. Verdict FAIL if any leg misses;
  on FAIL, mark P3–P10 VOID.
* **P2** — same source, `answered` only. Tag `cannot_invalidate`.
* **P3** — `humanlikeness_by_task["nback"]` from `score_candidate.evaluate`.
* **P4** — from `tasks/wm_nback.jsonl`, iterating `step_log[1:]` per row, grouped by
  `n_level`:
  * `memory_full_per_turn` = (count of `tool_calls[*].result` containing
    `"memory is full"`) / turns. Threshold **≤ 0.01 per turn**, at all three levels,
    **plus a qualitative leg: every memory-full event must be one where all four
    residents were written in that same turn** (readable from
    `step_log[*].tool_calls`).

    *Not* an absolute zero, and the reason is worth recording. A refusal **is** still
    reachable under this rule — it needs four residents all laid down in the current
    turn plus a fifth write in that same turn, which turn 1 permits because the
    budget is 6 and the store is empty. v3's own calls-per-turn histogram already
    shows one such turn at n=2: `{1: 464, 2: 291, 3: 44, 4: 1}`. An absolute zero
    would therefore be one legitimate event away from failing with the mechanism
    working — the same mis-scaling the WORKLOG records for iteration 4's "at most 2
    turns" contamination bar. The rate bar loses no instrument-integrity power,
    because a non-advancing presentation mark produces ~0.47 per turn, which is 47×
    the bar rather than one event over it.
  * `p_answer_given_zero_budget` = share of turns with
    `tool_call_budget_after == 0` whose `text`, **after stripping
    `<tool_call>…</tool_call>` blocks**, parses under
    `bench.tasks.wm_nback._parse_classification`. Threshold ≥ 0.97.
  * `zero_budget_share` and `tool_calls_per_turn` — **REPORT ONLY, no threshold.**
    See the note below; predicting a collapse here would fail a working mechanism.
  * `displacement_per_turn` = count of results containing `"was displaced and is
    now lost"` / turns. Report only; expected ≈ 0.00 / 0.45 / 0.45.
* **P5** — from `tasks/wm_nback.jsonl`, `n_level == 3`: share of `final_kv` values
  containing a bare capital consonant from `bench.tasks.nback.CONSONANTS`, the same
  computation v3's P4 used.
* **P6** — `humanlikeness_by_task["variable_mapping"]`,
  `axes()["variable_mapping_matched"]["humanlikeness_matched"]`,
  `axes()["A4"]["n_errors"]`, `axes()["A4"]["rc_ratio_normalized"]`.
* **P7** — from `tasks/wm_variable_mapping.jsonl`: count `step_logs[*].parsed`
  empty, and count `answer_raw` containing `"<tool_call>"`.
* **P8** — `delta` per task computed against the **v3 run**, not the baseline, since
  this candidate's batch path is byte-identical to v3's. Per-task bands as tabled.
* **P9** — `report["diagnostics"][n]["keys_held"]`, `acc_over_answered`, and
  `slot_utilization` from `tasks/wm_nback.jsonl`.
* **P10** — `report["diagnostics"][n]["buffer_no_response_frac"]`.
* **H1** — `delta_vs_baseline["craft_task"]`. Emits a FLAG row, never a FAIL.
* **Contamination check**, as v3's: a reply containing `<tool_call>` must not be
  counted as a classification unless it parses *after* the block is stripped, and
  the residual contaminated share must stay ≤ 5% per level.

### Two deviations from the brief, both deliberate

**(i) The brief asks for `zero_budget_share` to collapse. It should not be
predicted.** n=1 is the control and it says an exhausted budget is harmless: share
0.46, answer rate 0.988, zero refusals. Under this rule the agent still wants ~1.5
writes per turn against 1.5 granted, so the budget may well stay pinned at zero —
and in the offline stub drive it **does**: `budget0` on 12 / 13 / 14 of 15 / 16 / 17
turns at n=1/2/3, with `answered` 14/14 at every level and zero tool-call-as-text.
Predicting a collapse would fail the row while the mechanism worked. The falsifiable
form of the same claim is `memory_full == 0.00` plus `P(answer | b=0) ≥ 0.97`, which
is the diagnosis written as a number.

**(ii) The brief asks for a no-change control on the six batch tasks banded from
`logs/run_to_run_floor.json`. That table understates the noise and the row would
fail for reasons unrelated to any mechanism.** Evidence, measured this session:

* `craft_task` moved **0.0248** between the baseline and v3 on a code path that is
  *provably identical* — v3 overrides only `step()`, `encode()` calls `step()`
  exactly once, `_episode_turn_index()` is 1 so no block is prepended, and
  `recall()` never calls `step()`. Recorded craft noise: **0.0031**. Understated 8×.
* `word_recognition` moved **+0.0460** on the same identical path. Recorded: 0.0000.
* `primacy` and `primacy_v2` were established in iteration 3 to be the same harness
  (zero divergences over 874 rows and 3610 write calls). Their `semantic_story_
  recall` runs end with **different stores on 200 of 200 rows**, and `narrative_qa`
  on 27 of 50. The serving stack is not deterministic at temperature 0 on a batch
  encode.

Largest identical-path |delta| observed per task, which is what P8's bands are built
from: craft **0.0248**, word_recognition **0.0460**, narrative **0.0160**, story
**0.0139**, digit_span_forward **0.0152**, digit_span_reverse **0.0000**.

**H1 follows from (ii) and is registered now rather than post hoc:** `craft_task`'s
effective floor is 0.030 and v3 already sits at −0.0248 with the batch path
untouched, so a nondeterminism draw of the size already measured can put craft under
its floor **with this candidate's mechanism working perfectly**. It is flagged, with
that evidence, rather than allowed to surface as a surprise.

---

## 7. Offline verification — what was checked, and what could not be

`python meta_harness/verify_interface.py meta_harness/candidates/evicting_reset/harness.py`
→ **PASS**; capacity enforced at 4, round trip clean, five names rebound in 8–10
modules.

Then, offline, free, no GPU:

**1. Batch-path byte-identity.** Every write sequence the batch tasks actually
produced, from the baseline, v2 and v3 runs, replayed through
`bench.core.working_memory.WorkingMemory` and through
`EpisodicDisplacementMemory(owner=<one presentation>)`, comparing the returned
string *and* the store contents at every call:

```
2485 write sequences, 10918 calls, 1167 overflow refusals -- ZERO divergences
```

**2. Across-presentation displacement, real n-back sequences,** one presentation per
turn, capacity asserted on every call:

```
n=1: baseline refusals   0 -> candidate   0, displacements   0
n=2: baseline refusals 382 -> candidate   0, displacements 253
n=3: baseline refusals 398 -> candidate   0, displacements 330
```

**3. The real `run_nback_block`** at n = 1, 2, 3, driven by a stub that reproduces
the model's observed strategy (overwrite a running key every turn; also create a new
positional key every other turn) and its observed failure (emit the `delete_key`
repair as text when the write was refused and the tool has been taken away).
Counting tool calls and refusals per turn, which is the quantity the mechanism
claims to change:

```
baseline n=1: calls/turn 1.47 refusals 4 cap_hit 12 budget0 12 tc-as-text 4 answered 10/14  [AAAAAAA.A.A.A.A]
baseline n=2: calls/turn 1.50 refusals 5 cap_hit 13 budget0 13 tc-as-text 5 answered  9/14  [AAAAAAA.A.A.A.A.]
baseline n=3: calls/turn 1.47 refusals 5 cap_hit 14 budget0 14 tc-as-text 5 answered  9/14  [AAAAAAA.A.A.A.A.A]
evicting n=1: calls/turn 1.47 refusals 0 cap_hit 12 budget0 12 tc-as-text 0 answered 14/14  [AAAAAAAAAAAAAAA]
evicting n=2: calls/turn 1.50 refusals 0 cap_hit 13 budget0 13 tc-as-text 0 answered 14/14  [AAAAAAAAAAAAAAAA]
evicting n=3: calls/turn 1.47 refusals 0 cap_hit 14 budget0 14 tc-as-text 0 answered 14/14  [AAAAAAAAAAAAAAAAA]
```

The `A`/`.` string is the per-turn reply form. The baseline arm reproduces the
**period-2 alternation** from turn 8 on, from the store rule alone. The candidate
removes it while `cap_hit` and `budget0` are **unchanged** — the budget is still
pinned at zero on 12–14 turns and the agent answers anyway. That is P4's second leg
demonstrated, and it is why P4's first leg is `memory_full == 0` and not
`zero_budget_share` falling.

**4. The real `variable_mapping` loop,** same injection: 20 steps, **0** refusals,
10 question turns, the store rendered in **10 of 10** of them (so `render_store`
still suppresses the duplicate block on question turns, as v3 intended), 4 final
keys.

**5. `primacy`'s episode mark** under a `reset_messages()` `step()` — the
measurement in section 3.

**6. ONE presentation per batch row — asserted, not assumed.** Section 2's whole
argument and P8's bands rest on each batch task building a fresh agent per scored row
with exactly one `step()`. If any reused an agent across rows, `_step_log` would grow,
presentations would advance, and displacement would fire on a task claimed to be
byte-identical. Verified two ways. (i) All five construction sites —
`wm_mcq_common.py:52` (craft, narrative), `wm_word_recognition.py:94`,
`wm_semantic_story_recall.py:165`, `wm_digit_span_forward.py:209`,
`wm_digit_span_reverse.py:110` — are inside the per-row worker function, and
`encode_digit_span`'s two apparent call sites are one `agent.encode()` plus a bare
`llm.generate()`. (ii) Driving the real code paths with a stub that requests five
writes in one assistant message:

```
run_wm_mcq_trial   (craft, narrative)  steps=1 presentations=1 refusals=1 displacements=0 keys=4
encode_digit_span  (forward, reverse)  steps=1 presentations=1 refusals=1 displacements=0 keys=4
```

The fifth write is **refused** with **zero** displacements — the batch regime is the
selection cap, measured on the real task code rather than on a replay that assumed
one presentation.

### What I could not test

* **Anything requiring the model.** Every counterfactual above replays the write
  sequence the model produced *under a refusing store*. Under eviction it will not
  emit the delete-as-text repair and will spend the freed budget differently, so
  these are **bounds on the store rule given those writes, not forecasts of the
  run**. This applies in particular to the variable_mapping figures below.
* **`variable_mapping`'s exposure to the rule change, quantitatively.** Under v3 the
  store *freezes*: it changes on only **211 of 1350** consecutive-question
  transitions, because the first four names are an absorbing state. Accuracy is
  **0.9833** when the queried name is resident and **0.2561** when it is not — the
  leak closure working, and the whole of the +0.3213. The risk is that the gain is
  bought by the freeze. Replaying the three residency rules over the run's own
  records:

  | residency rule | queried name resident | mean `relation_count` non-resident / resident |
  |---|---|---|
  | v3 (frozen / refuse) | 840 / 1500 | 9.439 / 6.869 = **1.374** |
  | LRU over names | 827 / 1500 | 9.456 / 6.815 = **1.388** |
  | ACT-R base-level top-4 | 821 / 1500 | 9.461 / 6.792 = **1.393** |

  Humans sit at **1.386**. So the interference structure A4 measures does not depend
  on the overflow rule, and the residency rate moves by ~1%. That is the basis for
  P6's thresholds, and it is a bound.
* **n-back at n=2 under eviction with a store that is rendered every turn.** No arm
  has run that cell. `displacement` reached 14.0 at n=2 holding only 1.46 keys;
  v3 holds 3.58. A fuller store can evict the 2-back letter before it is needed, so
  `acc_over_answered` may fall even as `answered` rises. This is why P3 carries no
  narrow point estimate — the band is 0.75–0.95, wide on purpose — and why the
  mechanism rows, not the score, carry the falsification weight.
* **Whether the batch tasks stay inside their bands.** Provably identical *code*
  is not identical *output* on this stack (checks 1 and item (ii) of section 6), so
  P8 is a genuine test and H1 is a genuine hazard.
* **The humanlikeness sensitivity of a vm accuracy shift** (predicted 0.663 →
  ~0.639 if LRU residency carried v3's conditional accuracies). Computing it needs
  the human score vector, and `runs/human/` is off-limits to this proposer by
  instruction. Stated rather than estimated.
