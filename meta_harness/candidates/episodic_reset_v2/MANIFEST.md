# `episodic_reset_v2` — iteration 3a

Parent: `episodic_reset` (iteration 2c). Surface touched: `WorkingMemoryAgent.step()`
only. `MAX_KEYS` = 4, `WorkingMemory` untouched, `recall()` untouched.

---

## 1. What the run actually shows the failure was

### 1.1 The shape in the rows

`meta_harness/runs/iter2/episodic_reset/Qwen_Qwen3-30B-A3B-Instruct-2507/tasks/wm_nback.jsonl`,
`n_level == 1`, 50 rows. `step_log` is *not* persisted into these rows
(`run_nback_block` returns it, the row does not carry it — see correction 6), so the
diagnosis is reconstructed from `model_parsed`, `model_parsed_buffer` and `final_kv`.

| fact | value |
|---|---|
| rows with `model_parsed == {}` **and** `final_kv == {}` | **36 of 50** |
| rows that answered ≥ 1 trial | 14 |
| of those, answered trials form a **contiguous run ending at trial 14** | **14 of 14** |
| of those, `final_kv` is exactly `{previous_letter: <the block's LAST letter>}` | **14 of 14** |
| accuracy over answered trials, those 14 rows | 0.787 |

Cumulative rows that had escaped by trial *t*, t = 1..14:
`2, 3, 4, 5, 5, 7, 7, 7, 8, 9, 10, 10, 12, 14`.

So a block has exactly two regimes: a silent regime in which the store stays empty
and nothing is answered, and — after a single transition — a normal regime in which
every remaining trial is answered. No row ever returns to silence.

### 1.2 The cause is a near-absorbing state the candidate's own code creates

`episodic_reset.step()` prepends the store only `if store` is non-empty. At n=1 turn 1
the store is empty, so the agent's entire context is:

```
[system: the C2 n=1 prompt]
[user:   "Next letter: G"]
```

No store block. Not even `(memory is empty)`. No turn index. Nothing that indicates a
turn has ever happened. `HUMAN_PROMPT_BY_N[1]` instructs `"no response"` to the first
letter, so `"no response"` is the **instruction-compliant** reply to that context. If
the agent also writes nothing, the store is still empty on turn 2 and turn 2's request
is identical in form to turn 1's. The only thing that differs between those turns is
the letter.

### 1.3 Which letter breaks it — measured, and it is nearly deterministic

Trigger = the letter presented on the first turn that produced a scored answer.
Denominator = every empty-store presentation of that letter across all 50 n=1 blocks
(661 presentations, 14 escapes, base rate 0.0212):

| letter | escapes / empty-store presentations | rate |
|---|---|---|
| **C** | 11 / 15 | **0.733** |
| B | 1 / 29 | 0.034 |
| H | 1 / 31 | 0.032 |
| M | 1 / 40 | 0.025 |
| the other 17 | 0 / 601 | 0.000 |

Escape is not a uniform per-turn hazard; it is carried by one letter. A memoryless
hazard fits the *aggregate* curve well (p = 0.0232 from `(36/50) = (1-p)^14`; predicted
cumulative 1.2, 2.3, 3.4, 4.5, 5.5, 6.6, 7.6, 8.6, 9.5, 10.5, 11.4, 12.3, 13.1, 14.0
against observed above, max residual −2.3) — but the letter table shows the
per-turn variation is almost entirely *which letter*, not chance.

Why C: the n=1 system prompt's own worked example is `A → A → B → C → C`, explained as
"The second C matches the previous C → same". `"Next letter: C"` collides with that
example and the model answers instead of declining. B, the example's only other
consonant, is the only other letter above 0.03. (A is not in `CONSONANTS`.)

Locus check, which rules out the alternative reading that the write happens a turn
before the first answer: attributing the trigger to the preceding turn scatters it over
eight letters with a maximum rate of 0.143 and no structure. So the write and the first
answer occur on the same turn.

### 1.4 Discriminating among the four candidate explanations the brief listed

| explanation | verdict | evidence |
|---|---|---|
| lost the task instructions | **no** | `"no response"` is exactly what `HUMAN_PROMPT_BY_N[1]` prescribes for the first letter; the system prompt carries it and is unchanged |
| stopped emitting parseable output | **no** | `model_parsed_buffer` records a parsed `"No response"` at position 1 in **50 of 50** rows, and post-escape tails parse cleanly; a parse failure cannot produce 14/14 contiguous tails |
| stopped calling the write tool | **yes, proximately** | `keys_held` 0.28; 36 rows end with `final_kv == {}` after 15 turns |
| lost track of which trial it is on | **yes, and this is why** | the context contains no ordinal information of any kind; every empty-store turn is the same turn as far as the agent can tell |

Not capacity: one item in four slots, and the baseline uses exactly one key
(`previous_letter`, 50/50 blocks, `keys_held` 1.00).

### 1.5 Why n=1 and not n=2 / n=3 — the fix must be general, and this says why it can be

The same empty-store, no-header turn exists at n=2 and n=3, yet `keys_held` is 3.06 and
3.82 and **0 of 50** participants are silent. The difference is in the instruction
text, not the harness. n=1 says the letter "matches the letter **one turn back**" and
`"no response"` to "the first letter"; n=2/n=3 say "two/three turns back" and "the
first two/three letters", which cannot be satisfied without holding more than the
current item — so the model writes on turn 1 and leaves the state immediately. n=1 is
the one level where the model's own write policy leaves the store empty long enough
for the harness defect to bite. The defect is general; n=1 is merely where it surfaces.

### 1.6 Was the pre-committed diagnosis right?

**Partly. Right about what was lost, wrong about the fix, and wrong about a premise.**

It pre-committed: *"a P9 failure localised to buffer-period turns means 'lost sequence
position', whose fix is pinning the instruction turn."*

- **Right:** the agent has lost sequence position. That is the failure.
- **Wrong about localisation:** the failure is not localised to buffer-period turns. It
  spans the whole block — 36 rows never answer any of the 14 *trial* turns either.
- **Wrong about the premise:** its claim that the instruction turn is "redundant with
  the system prompt, so nothing needs pinning" is false. The system prompt
  (`wm_prompt_parts.wm_system_prompt`) says "Use write_memory and delete_key to
  maintain the key-value store while doing the original task". The instruction turn
  (`wm_nback.py:86`) uniquely adds **"Update your working memory each turn to track the
  recent sequence"** — a standing per-turn directive the system prompt does not carry,
  and exactly the directive whose absence reads as `keys_held` 0.28.
- **Wrong about the fix:** pinning it would not have repaired P9. With the instruction
  turn pinned and the store empty, turn *k* is *still* a pure function of the current
  letter — the absorbing state survives untouched, and whether a run escapes rests on
  the model happening to write on turn 1 for every letter. Given the letter table
  above, that is a bet, not a repair.

---

## 2. The mechanism, and why what it restores is task set

### 2.1 One rule

**What crosses a turn boundary is the agent's control state — the task set, its
position in the episode, and its store — and nothing about earlier stimuli.**

`step()` still rebuilds its message list from scratch. On every step after the first
in an episode it prepends:

```
[ongoing episode]
Turn {t} of this episode. Stimulus presentations so far: {p}.
You have no transcript of earlier turns: the key-value store is your only
record of them. Update it each turn to track what you will need later.
Your working memory currently contains:
{wm.to_recall_text()}

{the task's own user message}
```

- `t` = `len(self._step_log) + 1` — `step()` calls including this one.
- `p` = `self._tool_interactions + (1 if allow_tools else 0)` — turns on which the agent
  was permitted to write, including this one when it is. `_tool_interactions` is
  incremented by the base `step()` *after* `_ensure_messages()`, so reading it before
  `super().step()` gives the pre-increment count. **Read, never assigned**, so
  `_tool_call_cap()` is exactly the baseline's.
- The store is rendered **unconditionally**, including as `(memory is empty)`. The
  rendering (only the last two lines) is skipped when the task's own message already
  carries `WM_STATE_HEADER`, so it is never shown twice.

Both ordinals are reported rather than one, so the off-by-one between them at n-back
(turn 2 = presentation 1, because step 1 is the instruction turn) is visible in the
prompt instead of being something the model has to infer.

Three changes from `episodic_reset`, all consequences of the same rule: the store is
rendered even when empty; the ordinals are stated; the standing directive is stated, in
task-agnostic wording, as task set rather than as a replayed episodic turn.

**The instruction turn is not pinned.** Its unique content — the standing per-turn
update directive — is restored as a general harness-level directive, identical on all
eight tasks. Its stimulus-specific and phase-specific content is already in the system
prompt, and replaying an episodic turn to recover a procedural rule is the wrong
category of fix.

### 2.2 The first step of an episode carries no block. Two reasons, both stated

**Psychological:** at block onset no turn has elapsed, there is no store to report and
no transcript to disclaim. Every line of the block is vacuous.

**Consequential:** the six tasks that go `encode()` → `recall()` call `step()` exactly
once, so their requests stay byte-identical to the baseline's and the no-change control
keeps its meaning.

These coincide. That is convenient, not suspicious, and both are written down because a
reader who spots the second one unstated would be right to discount the first. The gate
is justified by the first and audited by the second.

### 2.3 Why the ordinals are task set and not item memory

The concentric model `episodic_reset` argued from (Oberauer 2002) exempts activated
long-term memory and task set from the ~4-chunk limit, and task set is procedural and
retrieved rather than held (Monsell 2003; Logan & Gordon 2001). The standing directive
is unambiguously in that category: it says how to operate the store, not what was in it.

The ordinals need the harder argument. Two grounds, and the first is the decisive one
because it is a property of the code rather than an appeal:

1. **They are provably non-diagnostic of stimulus content.** `t` and `p` are functions
   of the number of method calls. They are computed without reading `user_message`, the
   store, or any stimulus, and carry zero bits about *which* letters or statements
   appeared. So they cannot substitute for the store, cannot raise n-back accuracy above
   what the store supports, and cannot be a route for the leak. "How many items have I
   seen" and "which items were they" are separable here in the strongest available
   sense: one is derivable from the call count alone.
2. **Every serial-order model of working memory posits exactly such a signal, outside
   the item store**: a slowly drifting temporal context that is always available and is
   not itself an item (Howard & Kahana 2002); an oscillator-based positional code
   (Brown, Preece & Hulme 2000); a timing signal indexing position in the loop
   (Burgess & Hitch 1999); temporal distance as the discriminative dimension
   (Brown, Neath & Chater 2007). None charges it against the chunk limit, because it is
   context, not content.

**The concession the brief asked for, in its honest form: yes, humans in this task have
access to something the reset removed, and it is not item memory.** A participant sees a
block-onset screen (2500 ms), then letters at a fixed 2000 ms ISI. They cannot be in the
state this harness put the agent in — unable to tell the first item from the fifteenth —
because the experimental situation supplies that continuously and for free, whatever
their memory does. Deleting it was a harness artifact, not a capacity limit. Restoring
it restores the situation, not extra memory. `keys_held` 0.28 at n=1 against the
baseline's 1.00 is what that artifact looks like in the data.

The exclusions are unchanged and are the point of the candidate: earlier stimuli, the
agent's own earlier replies and tool calls, and any last-N-turns window (N = 0) all stay
out. A verbatim recent window is a second uncapped store, and `full_context` is the
least humanlike harness measured (0.6387 against 0.7861).

---

## 3. Why it is general, and what it does to the other tasks

No task name, level, or stimulus type appears anywhere in `harness.py`. The mechanism is
a function of two call counters and the store. Verified offline (section 8) that the
identical block appears on `variable_mapping`'s turns.

| task family | what changes | predicted effect |
|---|---|---|
| `nback` n=1 | the absorbing state is structurally impossible: ordinals strictly increase, so no two turns are the same request; the empty store is visible; the per-turn write directive is present | `answered` 2.06 → ≥ 13, `acc_over_answered` 0.787 → ≥ 0.90, silent rows 36 → 0, `keys_held` 0.28 → ≥ 0.8 |
| `nback` n=2 | the agent now knows whether it is still in the buffer period, and is told to write each turn | `answered` 10.16 → ≈ 14; accuracy is *store-limited*, so `acc_over_14` rises from 0.5457 toward ≈ 0.75–0.82 against the human 0.8615. From the baseline's 0.7786 that may land nearer the human or past it: **no direction is predicted on n=2 humanlikeness**, and if it overshoots, that is a loss on n=2 accompanying the n=3 gain |
| `nback` n=3 | same | `answered` 9.08 → ≈ 12–14; `acc_over_14` 0.4257 → ≈ 0.55–0.70 against the human 0.7799, i.e. **toward** human from the baseline's 0.36 |
| `variable_mapping` | the interval index is explicit and the write directive is present, so the agent should write more diligently | `n_errors` **falls** from 509 (but stays ≥ 150); vm humanlikeness **falls** from 0.6764 toward the baseline but stays ≥ 0.55 |
| the six `encode()`→`recall()` tasks | nothing: one `step()` call, block suppressed, requests byte-identical (verified offline) | no systematic change; residual movement is serving nondeterminism at the rates measured in correction 1 |

**Aggregate n-back mechanism, stated so it is checkable rather than hoped for.** The
per-row `nback` humanlikeness is driven by across-level spread: baseline `per_row_sd`
0.3209 against `human_sd` 0.0812, with the 50 n=1 rows sitting at ≈ 1.0 and n=3 at 0.36.
Getting all three levels to answer compresses that spread, which is why `displacement`
(0.9421) and `primacy` (0.9454) both gained on `nback` without changing accuracy much.
The same compression is expected here. It is **not** claimed as the headline; the
binding requirement is only that `nback` clears its floor at 0.7309.

---

## 4. Prompt delta, and why each line is forced

One addition, on `step()` turns after the first. Four lines plus the store.

| line | forced by | provenance |
|---|---|---|
| `[ongoing episode]` | nothing psychological; it is a frame marker so the control state is visibly not part of the stimulus | new, fixed, 17 characters, identical on every turn of every task |
| `Turn {t} of this episode. Stimulus presentations so far: {p}.` | §1.2 — without it the empty-store turn is a fixed point and the agent cannot tell turn 1 from turn 15 | new; two integers from call counters |
| `You have no transcript of earlier turns: the key-value store is your only record of them. Update it each turn to track what you will need later.` | §1.6 — this is the one standing directive the reset removed and the system prompt does not carry | second sentence generalises `wm_nback.py:86`'s "Update your working memory each turn to track the recent sequence" by dropping the n-back-specific object |
| `Your working memory currently contains:` + `to_recall_text()`, **including `(memory is empty)`** | `step()` otherwise gives the agent no read access to its own store, and the *empty* case is precisely where `episodic_reset` fell over | header verbatim from `RECALL_PROMPT` (`wm_agent.py`) and `QUESTION_PROMPT` (`wm_variable_mapping.py`); `(memory is empty)` verbatim from `WorkingMemory.to_recall_text` |

No swept constants. No RNG. The block's content is fully determined by the call counts
and the store.

**Chattiness risk, named because it is real.** `variable_mapping`'s `QUESTION_PROMPT`
ends "Output ONLY one line in this format … No extra text", and this candidate prepends
three lines before it. Measured baseline `answer_raw`: mean 13.10 chars, max 14,
p95 14 over 1500 answers (`episodic_reset`: mean 13.18, **max 131**). P7 registers a
threshold on it.

---

## 5. Why the leak stays closed

By construction, not by measurement. Everything the agent sees on turn *k* is: the
system prompt (unchanged), the tool schemas (unchanged), two integers derived from call
counters, fixed English text identical on every task and turn, the store, and the
current `user_message`. `step()` retains no reference to any previous `user_message`, so
no earlier stimulus can reach turn *k* by any route other than the four slots.

Verified offline against the real n-back task at n=1, n=2 and n=3, and against the real
`variable_mapping` loop: **the first `generate_with_tools` call of every `step()`
receives exactly 2 messages (system + user)**, and no request anywhere — including
inside the tool-dispatch loop — ever contains more than one `"Next letter:"`. On
`variable_mapping` no turn shows more than its own `TURNS_PER_QUESTION = 2` statements.
See section 8 for the verbatim contexts.

P2 registers the run-side consequence: the n=3 letter-identity share must stay ≥ 0.90
(`episodic_reset` 1.0000, baseline 0.0202).

---

## 6. Composition notes

**The `RuntimeError` is gone.** `episodic_reset.step()` raised
`RuntimeError(f"store holds {len(store)} keys, capacity is {_capacity()}")` when the
store exceeded capacity. This candidate does not. It was dead code under the baseline's
`write_key` (which refuses rather than overflows), but under a composed overflow rule a
transient overflow would turn a degraded row into a hard job crash, and the invariant is
`write_key`'s to enforce, not `step()`'s to assert. **`episodic_reset_v2 ∘ primacy` is
safe to compose in that respect.**

What a composition must respect:

1. **Surfaces are disjoint.** This candidate overrides `step()` and nothing else. It does
   not subclass `WorkingMemory`, does not touch `write_key`, the overflow policy or the
   eviction order, does not touch `recall()`, `encode()`, `TOOLS`,
   `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS`, `_tool_call_cap()` or `MAX_KEYS`. Composing
   = a class with this `step()` and `primacy`'s memory object as `self.wm`.
2. **`to_recall_text()` is now on the prompt path, so `primacy` becomes visible to the
   agent.** Under the baseline `step()` the store was never rendered into a turn, so
   eviction order was unobservable during a `step()`-driven task. Under this candidate
   the store is rendered every turn, so `primacy`'s eviction — both *which* key dies and
   the *order* the survivors are listed in — is now part of the agent's context on
   `nback` and `variable_mapping`. **The composition is therefore not additive**: some of
   `primacy ∘ this` may come from the agent reacting to a visibly changed store rather
   than from the eviction rule alone. Worth a row in the composed arm's predictions.
3. **`_capacity()` reads `MAX_KEYS` live** via `getattr(_wm_mod, "MAX_KEYS", MAX_KEYS)`,
   so an injected capacity is honoured. It is used only to fill `MANIFEST["capacity"]`
   at import; no control flow depends on it.
4. **Attribute reads to preserve.** `self._step_log` and `self._tool_interactions` are
   read (never written) before `super().step()`. A composed candidate that also
   overrides `step()` must call `super().step()` exactly once per turn and must not
   pre-increment `_tool_interactions`, or the reported presentation ordinal will be
   wrong (harmless to scoring, but the prompt would lie).
5. **`reset_messages()` must stay at the turn boundary**, never inside the
   tool-dispatch loop: clearing mid-loop leaves a `tool` message with no preceding
   assistant `tool_calls` and the request is rejected by the server.
6. **`load_candidate` + `apply` is not idempotent** — see correction 5. A composed arm
   must be a single candidate file, loaded once.

---

## 7. Corrections to the brief and to the record

1. **The serving stack is NOT bit-deterministic, and the brief's licence to "demand
   bit-identity" holds only for a coarse scored statistic, never for rows.**

   *In-run proof, self-contained:* the 15 empty-store presentations of `"Next letter: C"`
   at n=1 under `episodic_reset` are **bit-identical requests** — the n=1 system prompt
   does not vary across participants, the store is empty so no block is prepended, and
   the user message is the same 16 characters. **Eleven escaped and four did not.**
   Divergence rate 4/15 on one identical short-generation request.

   *Cross-arm confirmation.* `episodic_reset`'s `step()` override is provably a no-op on
   a single-`step()`, empty-store episode (`reset_messages()` on an empty list; `if store`
   false). So every `encode()` → `recall()` task issues byte-identical requests under it.
   Comparing rows by `id`, with server-assigned `tool_call_id` stripped:

   | task | rows differing vs baseline | scored HL delta |
   |---|---|---|
   | `semantic_story_recall` | **199 / 200** | −0.0027 |
   | `wm_narrative_qa` | 40 / 50 | −0.0100 |
   | `wm_word_recognition` | 39 / 50 | +0.0379 |
   | `wm_craft_task` | 24 / 150 | −0.0280 |
   | `wm_digit_span_reverse` | 10 / 190 | 0.0000 |
   | `wm_digit_span_forward` | 4 / 190 | 0.0000 |

   Same measurement under arms whose mechanisms also cannot touch these tasks:
   `serial_recognition` (overrides `recall()` only, and only on a ≥ 5-`trial N:`-line
   prompt, which digit span and craft do not have) differs on `digit_span_reverse`
   10/190 and `craft_task` 12/150 — both with scored delta **0.0000**.

   Craft is the decisive pair: **12 differing rows → 0.0000, 24 differing rows →
   −0.0280, same task, same zero mechanism.** The "bit-exact reproduction" recorded in
   the WORKLOG was reproduction of the *scored humanlikeness*, which for
   `digit_span_reverse` is a W₁ over 10 model pseudo-participants and is coarse enough to
   be robust. It was never row-level determinism.

2. **Therefore the WORKLOG's attribution is wrong: "`episodic_reset`'s craft (−0.0280)
   and narrative (−0.0100) moves are genuine effects of its `step()` rewrite" is false,**
   and so is "its claim that the six `encode()`→`recall()` tasks would be untouched is
   empirically wrong". Its claim was structurally correct — the requests *are*
   byte-identical, and I reproduced that offline against a stub. What failed was its
   **P10 threshold** (±0.005 on every task), which the serving stack cannot meet. It
   failed on a wrong threshold, not a wrong mechanism. `semantic_story_recall` at 199/200
   differing rows and −0.0027 is the proof: near-total row divergence, no scored effect.

   Practical consequence, and a second-order correction: `metric_noise.py`'s bootstrapped
   floor for `craft_task` is 0.025, and a provably no-op change moved it −0.0280. The
   bootstrap resamples the model side of a *fixed* set of rows and therefore does not
   capture generation-level run-to-run variation. For `craft_task` at least, the floor
   understates it. P8 registers 0.030 there rather than 0.025 and says why.

3. **The pre-committed fix would not have worked, and its premise was false.** See §1.6.
   Pinning the instruction turn leaves the empty-store turn a pure function of the
   letter; and the instruction turn is *not* redundant with the system prompt — it
   uniquely carries "Update your working memory each turn to track the recent sequence".

4. **The failure was not "localised to buffer-period turns".** 36 of 50 rows answered
   none of the 14 *trial* turns either. And the per-turn escape is not a hazard over
   turns — it is letter-conditioned, 0.733 on C against 0.000–0.034 on everything else,
   traceable to the worked example in the system prompt. A worked example whose letters
   overlap the stimulus alphabet is a stimulus-contamination hazard for any candidate
   that reduces context; it is worth recording independently of this candidate.

5. **`inject.load_candidate` + `apply` is not idempotent, and stacking is silent.**
   Loading a candidate file twice in one process produces a second class whose
   `_BaseAgent` is resolved at import time from `bench.core.wm_agent.WorkingMemoryAgent`
   — which `apply()` has *already rebound to the first candidate class*. The second class
   therefore subclasses the first and the overrides compose: I observed the control-state
   block emitted twice in one user message. `run_candidate.py` loads once
   (`run_candidate.py:79-80`), so the real pipeline is safe. Any test script, any
   future in-process multi-arm runner, and any composed-arm harness must load once.

6. **`wm_nback.jsonl` does not persist `step_log`.** `run_nback_block` returns it
   (`wm_nback.py:129`) but the row assembled in `evaluate` omits it, so the agent's
   actual text output on n-back is not recoverable from a run. That is why "did it emit
   unparseable output?" had to be answered indirectly (via `model_parsed_buffer` and the
   contiguity structure) rather than by reading a reply. `variable_mapping` does persist
   `answer_raw`. Adding `step_log` to the n-back row — or even just the per-turn text —
   would have made this diagnosis a one-liner. Not changed here: the brief forbids
   editing `bench/`.

---

## 8. Offline verification (no GPU, no network, no paid API)

`python meta_harness/verify_interface.py meta_harness/candidates/episodic_reset_v2/harness.py`
→ **PASS**. `WorkingMemoryAgent` rebound in 8 modules, `SummarizerAgent` 8,
`CONDITION_PROMPTS`/`TOOLS`/`MAX_KEYS` in their defining modules; encode/recall round
trip returns the expected shapes; capacity enforced at 4.

Stub-LLM drive of the **real** `wm_nback.run_nback_block` at n = 1, 2, 3 and of the real
`variable_mapping` loop, recording every `messages` list handed to the LLM. **0 failures
over all assertions:**

- one first-call per `step()` at every level (16 / 17 / 18 calls for 15 / 16 / 17 steps
  plus the instruction step);
- **the context on the first call of every `step()` is exactly 2 messages (system +
  user)** — at every level, including all buffer-period turns;
- no request anywhere, including inside the tool-dispatch loop, contains more than one
  `"Next letter:"`;
- the instruction step (step 1) carries **no** control-state block and still carries the
  task's own instruction text;
- every later step carries the block, renders the store, and the turn ordinals are
  strictly increasing and unique;
- `variable_mapping`: 2 messages per step, store header never duplicated on the question
  turn, and no turn shows more than its own 2 statements;
- `encode()` → `recall()` request stream **byte-identical** to `baseline`'s (tool-call
  request list and the `recall()` prompt both).

Observed context, n=1, the two turns where the failure lived (stub writes
`previous_letter` each turn, hence the placeholder value):

```
--- step 1 (the task's instruction turn) ---
This is a 1-back task. You will see letters one at a time. For the first 1 letter(s),
respond 'no response'. After that, respond 'same' if the current letter matches the
letter 1 position(s) back, otherwise 'different'. Update your working memory each turn
to track the recent sequence.

--- step 2 (letter 1) ---
[ongoing episode]
Turn 2 of this episode. Stimulus presentations so far: 1.
You have no transcript of earlier turns: the key-value store is your only record of
them. Update it each turn to track what you will need later.
Your working memory currently contains:
(memory is empty)

Next letter: R

--- step 3 (letter 2) ---
[ongoing episode]
Turn 3 of this episode. Stimulus presentations so far: 2.
You have no transcript of earlier turns: the key-value store is your only record of
them. Update it each turn to track what you will need later.
Your working memory currently contains:
previous_letter: <stub value>

Next letter: R
```

n=3, buffer-period turns (letters 1–3) and the first trial turn (letter 4), read
`Turn 2/3/4/5 … presentations 1/2/3/4`, each with the store rendered and exactly one
`Next letter:`.

`variable_mapping`, question turn (store rendered once, by the task's own template):

```
[ongoing episode]
Turn 4 of this episode. Stimulus presentations so far: 2.
You have no transcript of earlier turns: the key-value store is your only record of
them. Update it each turn to track what you will need later.

Your working memory currently contains:
Mary: Houston

Original task instructions:
…
```

**What is verified vs predicted, stated explicitly.** Verified: the *structure* of the
context on every turn — how many messages, what they contain, what they cannot contain,
and that the six `encode()` tasks are byte-identical. **Not verified, and predicted
only:** that the model will now write on turn 1 and answer from the store. No behaviour
of the frozen LLM is tested here; that is what P1 is for. I did not run the model.

---

## 9. Pre-registered predictions

Machine-readable in `meta_harness/logs/pending_episodic_reset_v2.json`. Thresholds
sanity-checked against `NOISE_FLOOR` (`nback` 0.060, `variable_mapping` 0.017,
`digit_span_forward` 0.140, `digit_span_reverse` 0.059, `word_recognition` 0.121,
`narrative_qa` 0.030, `semantic_story_recall` 0.011, `craft_task` 0.025) and against
`FLOOR` 0.03; the effective per-task floor is `max(FLOOR, NOISE_FLOOR)`.

| id | prediction | threshold | baseline | `episodic_reset` |
|---|---|---|---|---|
| **P1** | **n=1 repaired — PRECONDITION** | `answered` ≥ 13.0 **and** `acc_over_answered` ≥ 0.90 **and** `n_no_answers` == 0 **and** `keys_held` ≥ 0.8 | 13.98 / 0.9943 / 0 / 1.00 | 2.06 / 0.787 / **36** / **0.28** |
| **P2** | leak stays closed on n-back | n=3 letter-share ≥ 0.90; n=3 `keys_held` **reported only** | 0.0202 / 3.96 | 1.0000 / 3.82 |
| **P3** | `nback` humanlikeness clears its floor | ≥ 0.7309 | 0.7909 | 0.4970 (−0.2939, **violated**) |
| **P4** | A4 trustworthy at scale, and the transcript did not come back | `n_errors` ≥ 150 | 12 | 509 |
| **P5** | A4 errors are load-ordered, not noise and not pinned to the ceiling | `rc_ratio_normalized` ∈ [0.15, 0.95] | 0.6661 (untrustworthy, 12 errors) | 0.7302 |
| **P6** | `variable_mapping` keeps most of the gain, on both formulas | HL ≥ 0.55 **and** HL-matched ≥ 0.55 | 0.3554 / 0.3587 | 0.6764 / 0.7295 |
| **P7** | parse integrity **and** no chattiness | unparsed answers < 30 **and** mean `answer_raw` ≤ 15.0 chars | 0 / 13.10 | 0 / 13.18 (max 131) |
| **P8** | no-change control | `digit_span_reverse` HL within 0.059 (point estimate: **exactly 0.9666**, expect FROZEN); five others within their own floors | — | 0.9666 ✓ |
| **P9** | n-back per-level decomposition | reported, no threshold | — | — |

**P1 is a precondition in the same sense its predecessor's P9 was: if P1 fails, P2–P7
are VOID regardless of their values.** The reason is unchanged — at n=1 a single
overwritten key suffices and the baseline uses exactly that, so a drop there cannot be a
capacity result and nothing measured under it can be attributed to the memory module.

**Expected losses, stated in advance.**
- `variable_mapping` humanlikeness is expected to **fall** from 0.6764 (and 0.7295
  matched), and `n_errors` to fall from 509, because the restored write directive should
  make the agent more accurate. That is accepted: this candidate's claim is the P1
  repair, not vm maximisation. A fall below 0.55, or `n_errors` below 150, is a failure.
- `nback` is **not** claimed as a gain. The binding requirement is the 0.7309 floor. The
  mechanism by which it is expected to clear it (spread compression, §3) is registered in
  P9 so the decomposition is checkable rather than narrated afterwards.
- The five non-frozen no-change tasks are registered at their own measured floors rather
  than at a uniform tolerance, because correction 1 shows a uniform tolerance is not
  achievable on this stack. `craft_task` is registered at 0.030 rather than its
  bootstrapped 0.025, with the reason stated.

### What `episodic_reset_v2_checks` should assert (the coordinator wires the dispatch)

| row | fields |
|---|---|
| P1 | `NL.report(run)["diagnostics"][1]` → `answered`, `acc_over_answered`, `n_no_answers`, `keys_held`. All **four** legs. On failure call `_void_rows(out, ("P2","P3","P4","P5","P6","P7"), …)`. The `keys_held ≥ 0.8` leg closes the "answered correctly without a store" hole `full_context` shows is reachable (0.80 keys at 0.9957 accuracy, because its context carried the letter). |
| P2 | `_nback_letter_share(run, level=3)` ≥ 0.90 (already in `check_predictions.py`) is the **only verdict-bearing leg**. Report `NL.report(run)["diagnostics"][3]["keys_held"]` beside it with **no threshold**. |
| P3 | `rec["humanlikeness_by_task"]["nback"]` ≥ 0.7309, equivalently `rec["delta_vs_baseline"]["nback"]` ≥ −0.060. |
| P4 | `IF.a4(run)["n_errors"]` ≥ 150. |
| P5 | `IF.a4(run)["rc_ratio_normalized"]` ∈ [0.15, 0.95]; also report `rc_ratio`, `rc_ratio_ceiling`, and `rc_ratio_normalized_distance` from the human 0.3728. Gate on P4 (`conditional_on`), since the ratio is untrustworthy below 30 errors. |
| P6 | `rec["humanlikeness_by_task"]["variable_mapping"]` and `rec["axes"]["variable_mapping_matched"]["humanlikeness_matched"]`, both ≥ 0.55. |
| P7 | `_vm_row_stats(run)["unparsed_answers"]` < 30, **and** mean `len(step_logs[*].answer_raw)` over `tasks/wm_variable_mapping.jsonl` ≤ 15.0 chars. The second leg needs a new helper; there is no existing field for it. |
| P8 | `rec["humanlikeness_by_task"]["digit_span_reverse"]`: verdict on `abs_delta ≤ 0.059` (its own noise floor), with exact 0.9666 as the point estimate. `predicts_change=False`, so it takes `_frozen_tag` — which will fire on the exact hit — and **not** `_void_if_identical`. Exact equality is deliberately *not* the gate: correction 1 shows it is an empirical regularity over four arms, not a structural guarantee (10 of 190 rows differed under a provably no-op change), and gating on it would repeat the defect this row corrects in `episodic_reset`'s P10. The other five compared against their per-task thresholds in the JSON. |
| P9 | `NL.report(run)["per_level"]` and `["diagnostics"]` for 1/2/3, plus `per_row_sd` and `pooled_sd`. `comparator: "report"` → INCONCLUSIVE by construction; tag `REPORTED`. |

Every row in the JSON carries `capable_of_varying` with the run IDs and values that
demonstrate the quantity has moved on real data, so a change-prediction on an inert
quantity is VOIDed rather than credited.

---

## References

- Baddeley, A. D. (2000). The episodic buffer. *TiCS* 4, 417–423. *(Considered and
  rejected in `episodic_reset`; unchanged here.)*
- Brown, G. D. A., Neath, I., & Chater, N. (2007). A temporal ratio model of memory.
  *Psych. Review* 114, 539–576.
- Brown, G. D. A., Preece, T., & Hulme, C. (2000). Oscillator-based memory for serial
  order. *Psych. Review* 107, 127–181.
- Burgess, N., & Hitch, G. J. (1999). Memory for serial order: a network model of the
  phonological loop. *Psych. Review* 106, 551–581.
- Cowan, N. (2001). The magical number 4 in short-term memory. *BBS* 24, 87–114.
- Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal
  context. *J. Math. Psych.* 46, 269–299.
- Logan, G. D., & Gordon, R. D. (2001). Executive control of visual attention in dual-task
  situations. *Psych. Review* 108, 393–434.
- Monsell, S. (2003). Task switching. *TiCS* 7, 134–140.
- Oberauer, K. (2002). Access to information in working memory. *JEP:LMC* 28, 411–421.
