# `primacy_v2` — iteration 3b

**Parent:** `primacy` (iteration 2a). **Grandparent:** `displacement`. **Capacity:** `MAX_KEYS = 4`, unchanged.

One method body differs from the baseline — `WorkingMemory.write_key` — and exactly one
thing differs from `primacy`: **how far eviction goes when the chunks arrived
simultaneously.** The activation equation, the eviction order, the episode bookkeeping,
the read-out format, `step()`, `reset_messages()`, `recall()`, `_tool_call_cap()`, the
prompts and the tool schemas are all `primacy`'s, untouched.

---

## 1. What the run data shows the craft and narrative regressions actually are

### The headline: they point in opposite directions, so one fix cannot repair both

The brief asks me to find out why "a uniform eviction policy is producing non-uniform
effects" on three tasks that all route through `recall()`. The answer is not that the
effect is non-uniform in size. It is that **the two regressions have opposite capability
signs**, and only one of them is a degradation at all.

| task | baseline model score | primacy model score | Δ score | Δ humanlikeness |
|---|---|---|---|---|
| `craft_task` (accuracy) | 0.9333 | **0.9720** | **+0.0387** | −0.0451 |
| `narrative_qa` (accuracy) | 0.8000 | **0.7340** | **−0.0660** | −0.0309 |
| `semantic_story_recall` (embed. sim.) | 0.5728 | 0.5807 | +0.0079 | +0.0004 |

`craft_task` got **better** and lost humanlikeness for it. `narrative_qa` got **worse**
and lost humanlikeness for that. No single change to a retention rule moves both toward
the baseline, and the brief's suggested framing — that the rehearsal term is misspecified
for burst writes and correcting it repairs both — is refuted by these two numbers before
any mechanism is proposed. **This is the first thing the caller should take from this
report.**

### craft_task is a capability overshoot, and the objective inverts

Craft humanlikeness is a monotone decreasing function of craft accuracy above the
baseline, over six independently measured runs:

| run | craft score distribution | mean accuracy | craft humanlikeness |
|---|---|---|---|
| `full_context` | {1.0: 150} | 1.0000 | 0.8130 |
| `primacy` | {0.8: 21, 1.0: 129} | 0.9720 | 0.8456 |
| `displacement` | {0.8: 32, 1.0: 118} | 0.9573 | 0.8627 |
| `episodic_reset` | {0.8: 32, 1.0: 118} | 0.9573 | 0.8627 |
| `baseline` | {0.8: 50, 1.0: 100} | 0.9333 | 0.8907 |
| `chunk_limit` | {0.8: 50, 1.0: 100} | 0.9333 | 0.8907 |
| `random_decay_v2` | {0.6: 50, 0.8: 79, 1.0: 21} | 0.7613 | **0.9275** |

Two arms with identical distributions score identical humanlikeness to four decimal
places (`displacement`/`episodic_reset`, `baseline`/`chunk_limit`), which is what makes
this a function and not a scatter. The human craft distribution therefore sits **below**
the model's, and `random_decay_v2` — which held only 2.47 keys — is the most humanlike
craft arm measured. On this cell more degradation is rewarded and `primacy` supplied
less.

**Where the extra capability came from.** Craft has three trials per participant, writing
3, 4 and 5 rules. The 3- and 4-rule trials score exactly **1.0000 in all six runs** — they
cannot overflow. Everything happens in the 50 five-rule trials:

| arm | retained write positions | keys held | 5-rule accuracy |
|---|---|---|---|
| `baseline` | 2, 3, 4 | 3 | 0.800 |
| `displacement` | 2, 3, 4, 5 | 4 | 0.872 |
| `primacy` | **1, 2, 4, 5** | 4 | **0.916** |

The baseline loses **two** rules, not one: the fifth write is refused, the agent answers
the refusal with `delete_key('rule1')`, and the replacement write never lands (position-1
survival 0.000, deletes 0.33/row). `displacement` and `primacy` both remove that thrash,
which is worth +0.072 on its own; `primacy` adds a further +0.044 by evicting the *middle*
rule, which is the least costly of the five to lose. So `primacy`'s craft cost is two
effects stacked: the shared admit-on-full repair, and the U-shape picking the best
possible survivor set for an arbitrary list of independent propositions.

**Consequence for a defensible target.** No bug-fixed harness can return craft to
Δ = 0.000 while keeping admit-on-full, because part of the baseline's 0.8907 is produced
by the refusal bug. `displacement`'s −0.028 is the realistic floor-clearing level, and the
enforced floor is `max(FLOOR = 0.03, NOISE_FLOOR = 0.025) = **0.030**`, **not** the 0.025
the brief quotes. Craft needs humanlikeness ≥ 0.8607.

### narrative_qa is a contiguity loss, and it is opposed to the confirmed mechanism

Fixing the batch size at B = 6 removes the confound that the number of chunks the agent
writes is itself endogenous:

| arm | retained positions (B = 6) | n rows | accuracy | retained adjacent pairs |
|---|---|---|---|---|
| `baseline` | 1, 2, 3, 4 | 16 | **0.7875** | 3 |
| `displacement` | 3, 4, 5, 6 | 34 | 0.7647 | 3 |
| `primacy` | 1, 2, 5, 6 | 37 | **0.7000** | 2 |

All three hold four keys. The ordering tracks **retained adjacency** — an unbroken run of
the plot — not retained count. Narrative QA asks ten ordered questions about a causal
chain, and a set spanning both ends with a hole in the middle leaves questions about the
middle unanswerable and un-inferable. That is the opposite of what craft wants, where the
items are independent propositions and a hole in the middle is the cheapest hole
available.

**So narrative_qa and `primacy`'s confirmed both-ends retention cannot be co-satisfied by
any retention rule.** Keeping the U-shape keeps the broken chain. I am not going to
pretend otherwise, and §7 pre-registers narrative as unresolved rather than predicting a
recovery I cannot mechanise.

Two supporting observations the caller should weigh:

- Four iteration-2 candidates on four unrelated surfaces all landed narrative_qa between
  −0.0089 and −0.0310 against an enforced floor of 0.030. `primacy`'s −0.0309 exceeds it
  by **0.0009**, which is 2.06 bootstrap SE.
- narrative_qa's humanlikeness is not a function of its mean. `chunk_limit` moved the mean
  by **+0.0040** and lost **0.0310** of humanlikeness, by collapsing the distribution onto
  a spike (30 of 50 rows at 0.8, sd 0.1233 → 0.1058). The cell is sensitive to
  distributional shape in both directions, which is why it is the least safe cell in the
  search set to predict on.

### The measured account of how writes arrive, which the mechanism turns on

The encode-phase writes are **not** a serial presentation. They arrive as one parallel
assistant message. Evidence, from the baseline's own tool-dispatch logs:

    wm_semantic_story_recall   67 of 200 rows: w w w w R R   (writes 5 AND 6 refused,
                               69 of 200 rows: w w w w R D    no delete_key between them)
    wm_narrative_qa            16 of 50  rows: w w w w R R
    wm_word_recognition        24 of 50  rows: w w w w R R

`step()` appends the assistant message carrying *every* requested tool call, then
dispatches them one at a time. A row with consecutive refusals is a row where the agent
issued write 6 without ever having seen write 5's refusal — so it had already committed to
all six chunks. `craft_task`'s pattern is `w w w w R D`: five writes in one message, then a
second message carrying the repair.

This matters because **`primacy`'s Atkinson–Shiffrin justification assumed serial
arrival**, and it is a correction to that candidate's own argument, not just to mine.

---

## 2. The corrected mechanism

Keep everything about `primacy`'s activation and eviction order:

    A_i = ln( Σ_k w_k · (T − t_k)^−d ),   d = 0.5
    presentations: own write w = 1.0
                   same-episode later write, while resident: w = 1/|store|
    overflow evicts argmin A_i

Change exactly one thing — the **stopping point** of eviction:

| arrival | evict until | occupancy after the write |
|---|---|---|
| simultaneous batch (this assistant message issues > 1 write) | `\|store\| ≤ (MAX_KEYS − 1) − 1 = 2` | **3** |
| serial (one write per assistant message) | `\|store\| ≤ MAX_KEYS − 1 = 3` | 4 — `primacy` exactly |

The maintenance limit is **`MAX_KEYS − 1`**. One integer, tied to capacity, moving with it,
introducing no numeric constant of its own.

> **An earlier draft of this file derived 3 from the decay exponent, as
> `floor(1/(1 − 2^−d)) = 3` at `d = 0.5`. That derivation is withdrawn and should not be
> cited.** It compared a per-step attention *share* against a strength *loss*, so the units
> did not match, and its answer was unstable in `d` — `d = 0.6` gives 2. It also happened to
> land on the number the craft cell needed, which is exactly how a fit disguises itself as a
> derivation. The commitment below is weaker as an argument and honest as a claim.

### Psychological justification

The store is not a data structure with a slot count; it is a set of representations kept
alive by a single attentional resource. Three commitments, each already in the lineage:

1. **Cowan (2001)** — the focus of attention holds ~4 chunks. `MAX_KEYS = 4` stands and is
   never exceeded, not even transiently.
2. **Barrouillet & Camos (2007), time-based resource sharing** — there is one attentional
   resource, time-shared between *processing* and *refreshing*. It cannot do both at once.
   This is what makes simultaneous arrival different in kind from serial arrival, and it is
   the part §1's measurement earned.
3. **Oberauer (2002), three embedded components** — the focus of attention holds the chunk
   currently being operated on, and the focus is *one of* the limited-capacity
   representations, not a free extra. So while a chunk is being encoded, the set that can
   be maintained alongside it is `K − 1`.
4. **Atkinson & Shiffrin (1968) / Murdock (1962)** — cumulative rehearsal in the buffer
   gives the earliest items extra presentations and produces the serial-position curve.
   This is `primacy`'s mechanism and it is kept verbatim.

The correction is the interaction of (1)–(3) that `primacy` missed. Cowan's 4 is what the
store can *hold* when nothing is being encoded; `K − 1 = 3` is what it can *maintain while
attention is committed to encoding*. When the chunks arrive one at a time, each is encoded
and then a whole turn of free time follows in which the residents can be refreshed with the
full resource, so 4 is attainable — that is the running-memory regime, and it is why n-back
must be, and is, left alone. When the chunks arrive **simultaneously**, as §1 shows they do
on every encode task, there is no free time anywhere in the batch: attention is encoding
from the first chunk to the last, the focus is never released, and 3 is the ceiling.

`primacy`'s specific error follows: it evicted exactly one entry per overflowing write, so
the store was always *exactly* full. Occupancy rose from the baseline's 3.33 / 3.60 / 3.60
to 3.67 / 4.00 / 3.92 on craft / narrative / story. A store that can be held permanently
at capacity by an agent that keeps overrunning it is not modelling a capacity limit — it is
modelling a cache.

### What I checked and rejected, so the choice is on the record

- **Reweighting the rehearsal term cannot fix craft.** Sharing one unit of attention
  equally over the store (`1/n` to the incoming item too) makes activation monotone
  decreasing in write position, so eviction takes the *newest* item — recency is destroyed
  and the retention minimum moves to the end, contradicting `primacy`'s confirmed result.
  Sharing over the whole concurrent batch (`1/B`) leaves the U-shape intact at B = 5 but
  moves the argmin to position 2 at B = 6, so the retention minimum is not stably in the
  middle. Neither changes the *count*, which is what craft needs. **`primacy`'s weighting
  is not misspecified, and I am not claiming it is** — that part of the brief's hypothesis
  does not survive the arithmetic.
- **An absolute ACT-R retrieval threshold (`A_i ≥ 0`) is unusable here.** It bites without
  any overflow — it would drop the third of four digit-span keys — and its verdict depends
  entirely on an arbitrary read-out clock: evaluate one tick later and nothing is
  retrievable at all. A victim-relative threshold is vacuous by construction, because the
  victim was argmin.
- **Triggering the maintenance limit on `B > MAX_KEYS` rather than on `|store| ≥ MAX_KEYS`.**
  This was the alternative, it fires on every task, and I rejected it for a principled
  reason and a disclosed one. Principled: it requires the store to act on demand it has
  not yet received — evicting at the fourth write of a five-write batch because a fifth is
  coming. The rule as written uses only what has already happened, and reads the batch
  only to classify the *mode of arrival* of the message in hand. Disclosed: it would also
  take story-recall occupancy to 3.0 and put `primacy`'s one confirmed recovery at risk,
  and I am aware that is a reason I wanted to avoid it. Both reasons are on the record; the
  first is why I think the choice is right.

---

## 3. Why this is a principled correction, not a per-task carve-out

The rule reads exactly two things: the number of `write_memory` calls in the assistant
message currently being dispatched, and the activations of its own entries. It cannot see
a task name, a prompt, a stimulus, a key string or a value. Swap the tasks and the rule
behaves the same way.

**On this benchmark the retained keyset changes on `craft_task` and almost nowhere else** —
offline replay of `primacy`'s own write sequences gives 50/150 craft rows changed, 9/200
story, 0/50 narrative, 0/50 word-recognition, 0/190 and 0/190 digit span. That is the
candidate's **strongest attribution claim**, not an embarrassment: the mechanism's entire
footprint lands on the one cell that regressed, and it does so for a reason stated in
advance rather than by detecting the task. Combined with P12's confinement result — every
craft score change in every arm is inside the 50 overflowing rows, 0 outside, with the
other 100 rows at exactly 1.0000 in all six runs — this is a cleaner single-mechanism
attribution than any iteration-2 candidate achieved.

The reason is batch size relative to capacity:

- **B = MAX_KEYS + 1 (craft's five-rule trials).** The fifth write is the last; it evicts
  two and admits one, and nothing refills. Occupancy 3, retained {1, 2, 5}.
- **B = MAX_KEYS + 2 (story, narrative, word-recognition, the overflowing digit-span
  rows).** The fifth write evicts two and admits one, leaving 3; the sixth write lands on a
  store that is *not* full, so it is admitted with no eviction at all. Occupancy 4,
  retained {1, 2, 5, 6}. **Over the whole batch v2 loses exactly the same keys as
  `primacy`** — it loses both at the fifth write instead of one at the fifth and one at the
  sixth. The two rules provably converge at B = MAX_KEYS + 2.

So the footprint is determined by the **parity of the surplus** `B − MAX_KEYS`, which is a
property of the material, not of the task identity. Craft's five-rule trial is the only cell
in the search set with an odd surplus, and it is the cell that regressed. Swap the stimuli so
that story wrote five chunks and craft six, and the footprint would swap with them.

The channel by which the other tasks can still move is the tool-result text (§5), and that
is declared rather than assumed away.

---

## 4. Why the confirmed U-shaped retention survives

`primacy`'s confirmed result — both-ends retention 0.000 → 1.000 with the retention minimum
at a middle write position — is preserved, and it is measured rather than argued.

Offline replay, `primacy`'s own logged write sequences through the real `PrimacyV2Memory`:

| task | overflow rows | both-ends (primacy) | both-ends (v2) | v2 survival by write position |
|---|---|---|---|---|
| `semantic_story_recall` | 143 | 1.000 | **1.000** | 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 |
| `narrative_qa` | 37 | 1.000 | **1.000** | 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 |
| `craft_task` | 50 | 1.000 | **1.000** | 1.0, 1.0, 0.0, 0.0, 1.0 |
| `word_recognition` | 32 | 1.000 | **1.000** | 1.0, 1.0, 0.0, 0.03, 1.0, 1.0 |

The minimum is still at positions 3–4 and the ends still survive, because eviction still
takes argmin activation — only twice instead of once, and the second-weakest entry is the
one `primacy` would have evicted on the *next* write anyway.

**The quantity can still vary, so a prediction on it will not be VOIDed.** 143 of 200 story
rows still overflow under v2; the retained set is still chosen by activation; a rule that
stopped overflowing, or that kept a contiguous head or tail, would read 0.000 — as the
baseline and `displacement` both did on real data.

### Replay fidelity, reported before any v2 figure is used

`PrimacyMemory` replayed against `primacy`'s own logs reproduces the observed `final_kv`
on **830 of 830 rows, keys and values**, across craft, narrative, story, word-recognition
and both digit spans. The replay is therefore validated, not assumed.

Assistant-message boundaries are reconstructed from the flat dispatch log as maximal runs
of consecutive `write_memory` calls, with a `delete_key` starting a new message — licensed
by the consecutive-refusal measurement in §1.

**What the replay is not.** Replaying `primacy`'s sequences through v2 is counterfactual
for the agent's own behaviour. The number of chunks written is endogenous (story B = 6 in
67/200 baseline rows against 134/200 primacy rows) and v2's tool result names two lost
keys instead of one. These are mechanism demonstrations, not score predictions.

### In-process verification of the arrival classifier

`verify_interface.py` passes: injection rebinds `WorkingMemoryAgent` in 8 modules and
`MAX_KEYS` in 10, the round trip completes, and capacity is enforced at 4. Beyond that,
`_batch_writes()` was exercised against the real `step()` contract with a scripted stub LLM,
because it is the one piece of the mechanism that reads the agent's internals:

| case | batch count seen at each write | final store | verdict |
|---|---|---|---|
| 5 writes in **one** message | 5, 5, 5, 5, 5 | `{k1, k2, k5}` | simultaneous; occupancy 3 ✓ |
| 6 writes in **one** message | 6 ×6 | `{k1, k2, k5, k6}` | converges with `primacy`; occupancy 4 ✓ |
| 5 writes, **one per message** | 1, 1, 1, 1, 1 | `{k1, k2, k4, k5}` | serial ⇒ `primacy` exactly ✓ (this is the n-back case) |
| 4 writes, then a message with `delete_key` + `write_memory` | 4, 4, 4, 4, then **1**, 1 | `{k2, k3, k4, k9}` | a mixed message counts its *writes*, so it is correctly serial ✓ |
| single write on a **fresh** agent | 1 | `{only}` | the assistant message is present before dispatch ✓ |
| bare store, `owner = None`, 8 writes | `None` | 4 keys | fails safe to serial; capacity still enforced ✓ |

The two fallbacks — `None` (no message list) and `1` (one write in the message) — both mean
"serial", i.e. `primacy`. There is no input on which the classifier silently becomes *more*
aggressive than intended.

### Digit span cannot move, and that is now provable

- Replay: 0 of 190 forward and 0 of 190 reverse rows change their retained keyset, because
  the overflowing rows write six distinct keys in one batch (B = MAX_KEYS + 2, the
  converging case).
- Independently: all 6 overflowing forward rows and all 7 overflowing reverse rows are
  span-18 and span-20 trials that already score `exact = 0.0`. An extra eviction cannot
  change a score that is already zero. `best_span` 18.4 comes from span-18 trials that
  never overflow.

---

## 5. Prompt delta, and why it is forced

`TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS` and `RECALL_PROMPT` are **byte-identical
to the baseline**. `snapshot()` and `to_recall_text()` are **not** overridden, keeping
`primacy`'s deliberate reversion of `displacement`'s recency-reordered read-out, so a
retention change is never confounded with a representation change.

One string changes, and only one. When a write displaces **two** entries the tool result
must name two:

    one entry  (byte-identical to displacement's and primacy's):
      "Key 'X' written. Memory was full, so the least recently used entry 'Y'
       was displaced and is now lost."
    two entries (new):
      "Key 'X' written. Memory was full, so the least recently used entries 'Y'
       and 'Z' were displaced and are now lost."

Forced, because the alternative is to lose an entry without telling the agent — the agent
would later recall against a store it believes contains something it does not. The
singular form is preserved verbatim so a single-eviction write remains textually
indistinguishable from `primacy`'s.

**This is the candidate's main uncontrolled risk and I am not going to bury it.** The
two-entry form fires on every overflowing row of every encode task — 143 story, 37
narrative, 50 craft, 31 word-recognition, 6+6 digit span in the replay — including all the
rows whose retained keyset is provably unchanged. The number of chunks the agent writes is
endogenous, so "story and narrative are mechanically unchanged" is a claim about the
*store*, not a guarantee about the *scores*. P4, P8 and C1 are written as bands around
`primacy`'s values for exactly this reason, and a move outside those bands localises the
cause to this string.

Nothing else changed: no change to `_tool_call_cap()`, so a gain cannot be read as extra
compute, and v2 lowers tool-call demand no further than `primacy` already did.

---

## 6. Composition notes for `episodic_reset_v2 ∘ primacy_v2`

Answering the three things asked, plus one hazard I found.

1. **`write_key` never transiently exceeds capacity.** Eviction strictly precedes
   insertion: occupancy goes `cap → target → target + 1`, where `target ≤ cap − 1`. There
   is no intermediate state in which `len(store) > cap`. `episodic_reset`'s
   `RuntimeError` on `len(store) > _capacity()` therefore cannot fire from this side, and
   that holds whether one entry or two are evicted.
2. **v2 can hold *fewer* than `MAX_KEYS`, never more.** After a simultaneous overflowing
   write the store holds 3. A `> _capacity()` assertion is safe; an `== _capacity()`
   assertion or anything that assumes the store is full when it has overflowed would now
   be wrong. The maintenance limit is computed live from `_capacity()` in `_maintained()`
   and floored at 1, so an injected `MAX_KEYS` of 1 or 2 degrades gracefully rather than
   going negative, and `MAX_KEYS = 10000` gives 9999 (verified).
3. **The episode counter is monotonic, which is the specific property your composition
   needs.** `_sync_episode` increments on *any* change to the user-message count, so
   `reset_messages()` clearing `_messages` drives the count to 0, which differs from the
   previous mark and therefore **advances** the episode rather than colliding with an
   earlier one. Residents carry their old episode id, so after a reset they receive no
   further rehearsal credit — which is the correct semantics: an episodic boundary ends
   the rehearsal set. Nothing in v2 writes through `self._owner`.
4. **Hazard, new in v2 and specific to a `step()` rewrite.** `_batch_writes()` classifies
   arrival by walking `_messages` backwards to the last message carrying `tool_calls` and
   counting its `write_memory` entries. It assumes the baseline's `step()` contract: the
   assistant message is appended *before* its calls are dispatched, and `role: "tool"`
   results are appended after each. **If `episodic_reset_v2` changes that order, or clears
   `_messages` mid-dispatch, or dispatches tool calls without first appending the assistant
   message, `_batch_writes()` will read a stale message or `None`.** It fails safe — `None`
   and 1 both mean "serial", i.e. `primacy` — so the composition degrades to `primacy`
   rather than crashing or violating capacity. But the composed arm would then silently be
   measuring `primacy`, not `primacy_v2`. **Please assert, in the composed arm, that
   `craft_task`'s five-rule rows hold 3 keys and not 4**; that single check distinguishes
   "v2 is active" from "v2 silently degraded to primacy", and it is the cheapest possible
   diagnostic for this failure mode.
5. Everything else is inherited from `primacy` and stays off your surface: `step()`,
   `reset_messages()`, `encode()`, `recall()`, segmentation, key naming, the tool-call cap
   and every prompt are the baseline's.

---

## 7. Pre-registered predictions

Machine-readable in `meta_harness/logs/pending_primacy_v2.json`, with a
`capable_of_varying` field and its evidence on every row. Thresholds are sanity-checked
against `NOISE_FLOOR` and against the **enforced** floor `max(0.03, NOISE_FLOOR[task])`.
`check_predictions.py` is not edited; `primacy_v2_checks` should assert the rows below
from the sources named.

| id | role | quantity and source | threshold |
|---|---|---|---|
| **P1** | **primary** | `humanlikeness_by_task['craft_task']` | **≥ 0.8607** (baseline 0.8907 − enforced floor 0.030). Expect 0.87–0.89. Disconfirmed below. |
| **P2** | mechanism P1 rests on | `wm_craft_task.jsonl`: mean `len(final_kv)` over rows writing > `MAX_KEYS` distinct keys, and their mean accuracy | kv **≤ 3.20** (replay: 3.00; primacy 4.00) — the sharp clause — **and** accuracy **< 0.916** (primacy's, with one *more* key: three keys must not outscore four). The accuracy clause is deliberately *not* set at displacement's 0.872, because `{1,2,5}` retains both ends and both-ends sets outscore contiguous ones on craft, so 0.872 would make the row fail for a reason unrelated to the mechanism. |
| **P3** | must not lose the confirmed mechanism | both-ends retention over overflowing story rows, derived from `turn_logs` write order and `final_kv` | **≥ 0.80** (baseline 0.000, displacement 0.000, primacy 1.000, replay 1.000). Disconfirmed below 0.60. Secondary: positions 3–4 remain the least retained. |
| **P4** | must not lose the recovered task | `humanlikeness_by_task['semantic_story_recall']` | **[0.9173, 0.9773]**, expect ≈ 0.9477. Two-sided: a large gain means the store got more capable. |
| **P5** | must not lose displacement's response fix | `axes()['nback_levels']['diagnostics']['3']` | `answered ≥ 13`, `keys_held ≥ 3.5`, `acc_over_answered ∈ [0.687, 0.787]` |
| **P6** | no-change control | `axes()['A1']` `sub_span_leak`, `best_span`; both digit-span deltas | `\|leak − 0.136\| ≤ 0.02`, `best_span ≥ 18.0`, `\|Δ_fwd\| < 0.140`, `\|Δ_rev\| < 0.059` |
| **P7** | guard | `axes()['A3']` `precision_distance`, `word_distance` | `≤ 0.0422` and `≤ 55.8`. BLEU deliberately not thresholded. |
| **P8** | **pre-registered UNRESOLVED** | `humanlikeness_by_task['narrative_qa']` | **[0.9222, 0.9322]** — a band around primacy's 0.9263, which is *below* the 0.9272 floor. Passing this row means the candidate **still violates narrative's floor.** |
| **P9** | fails if the fix evicts *less* | story mean `len(final_kv)` and mean `embeddingSimilarity`; craft mean `len(final_kv)` | story kv ≤ 3.93 **and** similarity ≤ 0.60 **and** craft kv ≤ 3.67 (full_context: 6.25 / 0.9960 / 4.853) |
| **P10** | no-change, A4 owes nothing | `delta_vs_baseline['variable_mapping']`, `axes()['A4']`, `variable_mapping_matched` | `\|Δ\| < 0.017`; if it exceeds, A4 needs ≥ 30 errors and `rc_ratio_normalized ≥ 0.15` |
| **P11** | headline, explicitly not the claim | `mean_humanlikeness_search` | ≥ 0.7945; only ≥ 0.8121 would be a credible gain over the baseline |
| **P12** | mechanism confinement | `wm_craft_task.jsonl` partitioned by whether the row writes > `MAX_KEYS` keys | non-overflow partition mean accuracy **== 1.0000** and **0** non-overflow rows with a changed score |
| **C1** | covariate, no direction | `humanlikeness_by_task['word_recognition']`, `axes()['A2']` | report only. Disconfirming: moves > 0.121 from 0.4706 **and** `fa_rate > 0.211` → the liberal-bias artifact is back, do not credit. |

**P8 is the honest cost of this candidate.** It keeps `primacy`'s confirmed mechanism and
therefore keeps narrative_qa's broken causal chain, and it says so in advance rather than
predicting a recovery it cannot mechanise. If narrative_qa's floor must be cleared, that
is a different candidate on a different axis — one that chooses contiguity over both-ends
retention — and it would have to give up P3 to do it.

### On the withdrawn determinism premise

The caller withdrew "the serving stack is deterministic" mid-task. No row here predicts
bit-identity. Re-measured from scratch (see `generation_noise_band` in the JSON):

| task | provable no-op arm | rows whose writes differ | rows whose **score** differs | Δ humanlikeness |
|---|---|---|---|---|
| craft | `chunk_limit` | 7 | **0** | **0.0000** |
| craft | `serial_recognition` | 12 | **0** | **0.0000** |
| narrative | `serial_recognition` | 47 | 29 | −0.0034 |
| story | `serial_recognition` | 200 | 188 | −0.0026 |
| ds fwd | `serial_recognition` / `episodic_reset` | 2 / 3 | 0 / 0 | 0.0000 / 0.0000 |

Generation-level variation is real and large in *phrasing* and strongly task-dependent in
its effect on the scored statistic. On craft it is exactly zero twice: 19 rows of phrasing
variation, 0 score changes, Δ = 0.0000 in both arms — craft's five-question 2AFC score is
too coarse to register a rewording. On narrative it is not zero. So craft's scored delta is
the most robust of the three gist cells and narrative's is the least characterised, which
is precisely why this candidate predicts on craft and declines to predict on narrative.

**Correction to the correction.** The caller cites `episodic_reset` as "a no-op arm [that]
moved craft −0.0280". It is not a no-op on craft: craft routes `encode()` through `step()`,
which is the surface `episodic_reset` rewrites, and the WORKLOG already records that its
claim of leaving the six `encode()`→`recall()` tasks untouched was "empirically wrong". Its
24 differing craft rows include 18 score changes, all inside the overflow group, its craft
accuracy delta is **+0.0240** in the same direction as `displacement`'s, and its craft score
distribution is byte-identical to `displacement`'s. So craft's empirical no-op band is
**0.0000 and 0.0000**, not −0.0280, and a craft threshold at the 0.030 enforced floor is
defensible.

**What replaces bit-identity as the attribution argument.** Every craft score change, in
every one of five candidate arms, falls inside the 50 rows that write more than `MAX_KEYS`
keys — `chunk_limit` 0 in / 0 out, `serial_recognition` 0 / 0, `displacement` 18 / 0,
`episodic_reset` 18 / 0, `primacy` 29 / 0 — and the 100 craft rows that cannot overflow
score exactly 1.0000 in all six runs. Craft's movement is localised to the rows where the
overflow rule acts, with no determinism premise required. P12 pre-registers that same
partition as a no-change row: it allows phrasing to vary freely and asks only that the
score not move where the mechanism cannot reach.

**Requested from a repeat baseline run, if one is made:** narrative_qa's per-row score
agreement and the resulting humanlikeness delta between two identical configurations. It is
the only cell in the search set whose scored statistic this analysis cannot bound, and it is
the cell on which three of four iteration-2 candidates were charged a floor violation.
Second priority: the craft five-rule-trial accuracy over its 50 rows, to confirm the
0.0000 / 0.0000 no-op band is not a two-observation accident.

---

## 8. References

Anderson, J. R. & Lebiere, C. (1998). *The Atomic Components of Thought.* — base-level
activation. · Anderson, J. R. & Schooler, L. J. (1991). Reflections of the environment in
memory. *Psych. Science* — `d ≈ 0.5` from environmental statistics. · Atkinson, R. C. &
Shiffrin, R. M. (1968). Human memory: a proposed system. — rehearsal buffer, primacy. ·
Barrouillet, P. & Camos, V. (2007). The time-based resource-sharing model. — attention
shared between processing and refreshing; the source of `n*`. · Cowan, N. (2001). The magical
number 4. *BBS* — `MAX_KEYS = 4`. · Glanzer, M. & Cunitz, A. R. (1966). Two storage
mechanisms in free recall. — recency from a separate store. · Murdock, B. B. (1962). The
serial position effect of free recall. — the U. · Nairne, J. S. (1990). A feature model of
immediate memory. — feature overwriting. · Oberauer, K. (2002). Access to information in
working memory. — focus of attention inside the region of direct access. · Oberauer, K. &
Kliegl, R. (2006). A formal model of capacity limits in working memory. *JML* — interference
by overwriting.
