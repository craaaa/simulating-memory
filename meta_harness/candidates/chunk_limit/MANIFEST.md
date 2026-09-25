# `chunk_limit` — iteration 2d

Parent **`baseline`**. Not `displacement`, not `primacy`.

One method body differs from the baseline: `WorkingMemory.write_key`. It bounds
the **information inside a slot** before delegating to the baseline's own
`write_key`, which is left intact — including its refuse-on-full overflow.

Explicitly untouched, so the surfaces stay disjoint:

| surface | owner | state here |
|---|---|---|
| overflow policy, eviction order, primacy protection | iteration 2a (`primacy`) | baseline's refuse-on-full, unchanged |
| `recall()` context construction, word-recognition trial list | iteration 2b | untouched |
| `step()`, `reset_messages()`, conversation-history leak | iteration 2c | untouched |
| `MAX_KEYS` | — | **4**, unchanged, read live |
| `TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS` | — | **zero bytes differ** |
| `_tool_call_cap()`, `snapshot()`, `to_recall_text()` | — | untouched (verified byte-identical read-out) |

Building on `baseline` rather than on `primacy` is deliberate: every delta, floor
and guard in `score_candidate.py` is computed against the baseline, and iteration
2a differs from the baseline in the overflow rule. Stacking would make my
attribution depend on theirs. Exactly one method body separates this candidate
from `baseline`.

---

## 1. The defect, measured per task — and the brief's premise is wrong

`MAX_KEYS = 4` bounds the number of **slots**. Nothing bounds the information
inside a slot. The single worst value in the iteration-0 baseline run is the
studied word list itself, stored under one key:

    "recent_items": "camera, corner, forest, artist, corner, forest, artist,
                     artist, artist, message, thunder, artist, shadow, violin,
                     violin, singer, police, message, tunnel, hobby, ..."   (99 items)

A 4-chunk store that admits a 99-item chunk is not Cowan's (2001) limit in any
meaningful sense. The slot count is cosmetic while a slot can hold an unbounded
enumeration.

**My own measurements**, over `final_kv` in every task of
`meta_harness/runs/iter0/baseline/…`. An *element* is a non-empty field after
splitting on comma, semicolon, newline or a spaced slash — the model's own
enumeration markers:

| task | rows | slots/store | elements/value mean (p90 / max) | elements/store mean (max) | words/value mean (max) |
|---|---|---|---|---|---|
| **word_recognition** | 50 | 3.40 | **13.21** (38 / **99**) | **44.90** (**180**) | 22.62 (111) |
| semantic_story_recall | 200 | 3.60 | 3.89 (6 / 9) | 13.98 (23) | 30.25 (74) |
| narrative_qa | 50 | 3.60 | 3.76 (6 / 9) | 13.54 (25) | 23.48 (107) |
| digit_span_forward | 190 | 3.48 | 1.94 (4 / 13) | 6.75 (25) | 6.51 (31) |
| digit_span_reverse | 190 | 3.48 | 1.94 (4 / 13) | 6.75 (25) | 6.49 (26) |
| nback | 150 | 2.17 | 1.13 (2 / **2**) | 2.45 (6) | 2.98 (7) |
| variable_mapping | 150 | 3.51 | 1.00 (1 / **1**) | 3.51 (4) | 3.46 (4) |
| craft_task | 150 | 3.33 | 1.00 (1 / **1**) | 3.33 (4) | 5.60 (7) |

The word-recognition figures reproduce the brief's (3.40 slots, 12.8 mean
comma-items, max 99, 166 chars/value, ~43.5 elements/store — I get 44.9 with
semicolons and newlines also counted as enumeration markers). The rest do not
support the brief's reading of them.

> **Correction to the brief.** The brief calls this defect "task-general rather
> than specific to one leak", and says digit-span values are "exactly the kind of
> list your bound would cut". Measured, neither holds.
>
> * It is **one extreme task** (word_recognition, 13.21 elements/value, max 99,
>   44.9 per store), **two moderate ones** (story 3.89, narrative 3.76 — both
>   already sitting *at* the bound this candidate imposes, so the bound trims
>   their tail rather than transforming them), **one weak one** (digit span, 1.94
>   mean; the bound fires on 28 of 190 rows and costs about **one digit** at
>   spans ≥ 15 — see §7 P11), and **three tasks where it cannot bite at all**:
>   craft_task and variable_mapping hold exactly 1.00 elements per value (max 1
>   over 500 and 527 values), and nback holds 1.13 (max **2** over 325 values).
> * The consequence matters for what can be claimed. The task where the
>   mechanism does most of its work is the confirmed word-recognition leak, whose
>   score cannot be credited and whose recall surface iteration 2b owns. Where
>   the score *is* creditable the effect is: story −8.8% stored words, narrative
>   −11.1%, craft **0%**, digit span ~1 digit inside a 0.140 noise floor. **So the
>   8-task mean is not expected to move credibly, and that is not the claim.**
>   See P14.
> * The upside of the same fact: craft_task, variable_mapping and nback are
>   *derived* no-change controls rather than asserted ones, and digit span — which
>   the brief suggested as the natural control — is disqualified by measurement.

---

## 2. What a chunk is, and the mechanism

### The bound has no new constant in it

Miller (1956) defines a chunk as a **recoded** unit: you beat the span limit by
recoding, and what is recalled is the recoded unit, not an enumeration.
Cowan (2001) fixes the number of chunks at ~4 **with the explicit caveat** that
the estimate holds for chunks whose internal structure is already consolidated
in long-term memory; where grouping is possible, apparent capacity in *items*
rises precisely because the chunk is not an item. Simon (1974), *How big is a
chunk?*, makes the complementary point empirically: span measured in chunks
falls as the material's units get larger and less familiar, so chunk size is a
property of the material's consolidation, not a free parameter. Chase & Simon
(1973) measured it directly in chess recall and found chunks of ~2–4 pieces
outside expert-familiar configurations; Gobet & Simon's later templates hold more
only because they are LTM structures, which is the opposite of a novel word list.

The step this candidate takes: **binding *n* novel elements into one retrievable
unit is itself an operation in the focus of attention, and is therefore subject
to the same capacity limit.** That is Halford, Wilson & Phillips's (1998)
relational-complexity limit of 4 — the number of elements that can be bound into
a single relation — and it is Cowan's own reason the 4-chunk figure is stable:
the bottleneck is on simultaneous binding, and there is no reason it should be
one size when it binds chunks into a store and another size when it binds
elements into a chunk.

So the within-slot bound **is `MAX_KEYS`**:

    a slot may hold at most MAX_KEYS enumerated elements
    the whole store holds at most MAX_KEYS² = 16 novel elements

**There is no numeric literal in `harness.py`.** `MAX_KEYS` is read live via
`getattr(_wm_mod, "MAX_KEYS", MAX_KEYS)`, verified by injecting `MAX_KEYS = 2`
and observing the within-slot cap follow to 2. Nothing was swept and no threshold
was moved after seeing a number. This also satisfies the no-fitting rule
mechanically: 16 is not within 10% of any human statistic in the contract, and I
deliberately did **not** choose a bound that would land digit span near the human
best span of 6.88 — that would be fitting to a held-out constant.

Stated plainly so it cannot be misread: **16 elements is not a claim that human
span is 16.** It is an upper bound on what may be *encoded*. Human span is lower
because retrieval and output interference cost more, none of which is modelled
here.

### What counts as an element

An enumeration delimiter the model itself wrote: `,` `;` newline, or a spaced
slash. Verified behaviour:

    "1945"                            -> 1 element, untouched
    "1-9-4-5 as a date"               -> 1 element, untouched
    "a; b; c; d"                      -> 4 elements, untouched
    "a; b; c; d; e"                   -> 5 -> "a; b; c; d"
    "one / two / three / four / five" -> 5 -> "one / two / three / four"
    a 74-word single clause            -> 1 element, untouched

`" and "` is deliberately **not** a delimiter: it is ubiquitous in prose, and
splitting on it would cut grammatical sentences and turn the rule into a length
cap by the back door.

Prose of any length passes. That is the point and not a loophole: the observable
difference between a chunk and a list is whether the content is *abstracted* or
*enumerated*, and the delimiter is the model's own declaration of which it wrote.
It also means the rule cannot be satisfied by writing shorter prose — it only
binds enumeration, so it does not push the agent toward telegraphic style.

**Known limitation, stated rather than hidden.** A single-clause value of
unbounded length still passes (max observed 74 words on story recall, 111 on word
recognition). Bounding *length* would need a new constant — why 15 words and not
20? — and no such constant is derivable from Miller, Cowan, Simon, Chase & Simon
or Halford. Inventing one is precisely the fitted-constant failure this search
forbids, so it is left as a named target for a later iteration rather than
smuggled in here.

### The rule

```
write_key(key, value):
    bounded = first MAX_KEYS enumerated elements of value   # cut at an element
                                                           # boundary, original
                                                           # punctuation kept
    result  = baseline.write_key(key, bounded)             # overflow unchanged
    if result was a refusal:  return it verbatim           # nothing encoded,
                                                           # nothing dropped
    if elements were dropped: append "Only the first N of M items fit in one
                              memory slot; the rest were not encoded."
```

The retained text is exactly what the model wrote, truncated at an element
boundary. Nothing is rewritten, reordered or paraphrased.

---

## 3. Cap, not consolidation — and why consolidation is worse here

The brief offers (A) cap the value and (B) force consolidation, and asks for a
deliberate choice. **This is (A).** Five reasons (B) is worse, in descending
order of how much they would have cost:

1. **It is not reachable from my surface.** `step()` dispatches writes through
   the module-level `_dispatch_tool`, which is *not* on `inject.py`'s
   `OVERRIDABLE` list. The only injectable place a value arrives is
   `WorkingMemory.write_key`, which holds no reference to the LLM. Making a
   consolidation call from there needs either a back-reference into the agent's
   `llm` or an override of `step()` — and `step()` belongs to iteration 2c. I
   would have had to take another iteration's surface to implement (B), which the
   brief forbids and which would have made both candidates unattributable.
2. **It adds a second frozen-model behaviour to the thing under search.** The
   search attributes a delta to one mechanism. With a summariser call inside the
   store, a delta could come from the bound or from the quality of the
   summariser's output on that particular material, and nothing in the record
   would separate them. `displacement`'s and `primacy`'s whole value came from
   having exactly one method body differ.
3. **The evidence says asking this model to abstract does not work.** It is
   *already* told to, twice: the `write_memory` tool description says "The value
   should be an abstractive summary of the relevant information", and the C2
   system prompt says "Compress realistically — humans retain gist, not verbatim
   detail." It wrote a 99-item list anyway. A second ask, in the same voice, to
   the same frozen model, is not a mechanism — it is a retry.
4. **It cannot be verified offline, so the mechanism-confirmation prediction
   would have been a guess.** Iteration 2a's strongest move was replaying real
   write sequences through its real class. (A) can be replayed for free (§6);
   (B) cannot be replayed at all without GPU time and an LLM call per over-long
   value.
5. **On the material where the bound actually bites, an LLM gist destroys
   *more* than the cap does, and in the wrong way.** The overlong values are word
   lists. A gist of 99 random nouns is something like "nature and city nouns,
   several repeated" — from which no individual word is recognisable, so
   recognition collapses to pure guessing. Keeping four words keeps four
   recognisable items. Humans in this task do not lose all item information; they
   are *conservative* (miss 0.272 against false alarm 0.045), which requires some
   retained items plus a cautious criterion, not zero items.

**The honest cost of choosing (A), which the brief names correctly:** a truncated
99-item list is still a list, just shorter. My answer is that the bound is on
*enumerated elements*, and a **4**-element list is a legitimate chunk under the
sizes Chase & Simon actually measured. What the rule forbids is the 99-item list,
not lists as such. The second cost the brief names — "truncation may destroy
exactly the recency information the model needs" — is real and is why the
*direction* of the cut matters; §7 P11 and P10 are the tests for whether the cut
destroyed needed information rather than surplus.

Two smaller design choices, both following iteration 1's findings:

* **The write is never refused.** Iteration 1 established that refuse-to-encode
  is the defect (response omission, which humans never show). An "Error: too many
  items, rewrite it" result would have re-created exactly that pathology at the
  value level, and would have burned the tool-call budget doing it (§5).
* **The affordance lives in the tool *result*, not the tool *description*.**
  Exactly `displacement`'s argument: the description is in context from turn 1 on
  every task, so announcing a per-slot bound there would change encoding strategy
  *before* any bound is hit — a strategy change wearing a chunk bound's clothes.
  The result is seen only after the bound has actually bitten.

---

## 4. Prompt delta: **none. Zero bytes.**

`TOOLS`, `CONDITION_PROMPTS` and `WM_SYSTEM_PROMPTS` are re-exported from `bench`
unchanged. `snapshot()` and `to_recall_text()` are not overridden and were
verified byte-identical to the baseline's on the same store contents. The only
new text anywhere is one sentence appended to the `write_memory` **tool result**,
and only on a write where elements were actually dropped:

> `Key 'recent_items' written. Only the first 4 of 99 items fit in one memory slot; the rest were not encoded.`

It is **purely descriptive**. An earlier draft ended it with "A slot holds one
chunk, so combine or summarise instead of listing"; that was cut, because it is a
strategy directive rather than a report of what happened, and it would have given
the model a reason to behave differently on tasks where the bound barely bites —
which is exactly what the controls in §7 are there to rule out. `displacement`'s
result was descriptive in the same way ("'X' was displaced and is now lost").

Consequence, which is why this matters: on **craft_task** no value is ever
bounded (0 of 500), so no such sentence is ever emitted and the tool results are
byte-identical to the baseline's. P7 is therefore a real control, not an
approximate one.

**Is silent loss realistic?** Loss is not silent here — the agent is told, after
the fact. A human is not told what they failed to encode, so the strictly
realistic version would say nothing. I chose the message anyway, for the same
reason iteration 1 did: it makes the mechanism visible in the transcript and
attributable in the record, and the cost is bounded because the message cannot
arrive before the bound has bitten. Where it does arrive, the agent has almost no
budget to act on it (§5), which is a limitation of the test rather than a virtue.

---

## 5. Call-count cost: zero extra LLM calls

No LLM call is added; `write_key` does string work only. `_tool_call_cap()` —
`max(6, int(interactions * 1.5))` — is untouched, so no gain can be read as extra
compute.

The relevant risk is the opposite one: could the agent *spend* more calls
reacting to the message? Measured tool calls per row in the baseline run:

| task | tool calls / row | cap |
|---|---|---|
| word_recognition | 5.56 | 6 |
| narrative_qa | 5.44 | 6 |
| semantic_story_recall | 5.40 | 6 |
| craft_task | 4.33 | 6 |
| digit span fwd/rev | 3.59 | 6 |

On the three tasks where the bound bites, the agent is already at 5.4–5.6 of its
6-call budget. **The cap binds**, so there is essentially no room to re-consolidate
or to refill the store toward the 16-element ceiling. That is the honest reason
the replayed store lands at 11.10 elements rather than 16 — not a virtue of the
rule, a consequence of a budget the rule does not change. It also means the run
cost is unchanged to within the agent's own variation.

---

## 6. Not noise, and not a capacity change

**Not noise.** No random number is drawn anywhere; `random` is not imported
(asserted in the offline check). Given the LLM's outputs the store is a
deterministic function of the write sequence, verified by replaying the same
sequence twice. The rule is also structurally unlike `random_decay_v2`, the
candidate that bought the aggregate (`best_span` 18.4 → 8.2 against human 6.88)
while breaking the error structure (A1 sub-span leak 0.130 → **0.249** against
human 0.087): stochastic dropping removes information *uniformly*, including
below a participant's own span, whereas this rule removes only the 5th-and-later
element of an enumeration. At short spans there is no 5th element, so there is
nothing to remove. P11 is that prediction, and it is the test that separates a
chunk bound from information vandalism at the aggregate level; P10 is the
row-level version.

**Not capacity.** `MAX_KEYS = 4`; `verify_interface.py` confirms writes beyond 4
keys are refused, and a 400-write fuzz asserts `len(store) ≤ 4` *and*
`elements(value) ≤ 4` after every single write. Slots held is unchanged on every
replayed task (e.g. word_recognition 3.40 → 3.40, story 3.60 → 3.60). The
direction is also the safe one for this search: capacity was already settled by
`full_context` (10 000 slots, 0.6387 against the baseline's 0.7861), and this
candidate *reduces* stored information rather than increasing it.

---

## 7. Pre-registered predictions

Machine-readable in `meta_harness/logs/pending_chunk_limit.json`. Every threshold
is checked against `NOISE_FLOOR` in `score_candidate.py`, and every quantity
carries a `capable_of_varying` field recording the evidence that it *can* move —
iteration 1's buffer-compliance prediction was VOID rather than failed because
nobody checked that first.

| # | quantity | threshold | baseline | offline replay | role |
|---|---|---|---|---|---|
| **P1** | word_recognition `final_kv`: **leg A** elements/value, elements/store, max; **leg B** distinct **studied words** in store | A: ≤ 5.0 / ≤ 16 / **== 4**. B: ≤ 8.9 | 13.21 / 44.90 / 99 / 17.78 | **3.26 / 11.10 / 4 / 4.62** | **mechanism confirmation** |
| **P2** | word_recognition words/value | ≤ 14 | 22.62 | 8.10 | evasion check |
| **P3** | max elements/value over **all** tasks | **== 4** | 99 | 4 | plumbing invariant |
| **P4** | `A3.words` (story recall **length**) | 104 – 120 | 121.4 | ~111 (scaled) | story, length leg |
| **P5** | `A3.precision` (story **verbatimness**) | 0.030 – 0.055 | 0.0414 | 0.0405 | story, verbatimness leg |
| **P6** | `semantic_story_recall` HL | Δ > −0.03, direction **not** derivable | 0.9473 | — | story, score leg |
| **P7** | `craft_task` HL | \|Δ\| < 0.025 | 0.8907 | 0 of 500 values bounded | **no-change (primary control)** |
| **P8** | nback n=3 `keys_held` / `answered` / HL | ≈ baseline; \|Δ HL\| < 0.060 | 3.96 / 6.82 / 0.7910 | not replayable | no-change |
| **P9** | `variable_mapping` HL | \|Δ\| < 0.017 | 0.3554 | not replayable | no-change |
| **P10** | narrative_qa accuracy on rows the bound never touched | \|Δ\| < 0.05 on the 14 unbounded rows | 0.7500 (14 rows) | — | **anti-vandalism** |
| **P11** | `A1.sub_span_leak`, `A1.best_span` | ≤ 0.1450, 18.4 ± 2.0 | 0.1298 / 18.4 | 28/190 rows, ~1 digit | **anti-vandalism / guard** |
| **P12** | `A2` miss_rate, fa_rate, ratio (+ trials covariate) | miss > 0.043, fa ≤ 0.127, **ratio > 0.71** (baseline CI upper) | 0.0432 / 0.1269 / 0.340, CI [0.007, 0.71] | studied words 17.78 → 4.62 | **directional axis claim** |
| **P13** | `word_recognition` HL | \|Δ\| < 0.121 | 0.4948 | — | leak test, **risky** |
| **P14** | 8-task mean HL | reported, ≥ 0.7861 − 0.026 | 0.7861 | — | headline, **not the claim** |

### What each one is for

**P1 — the mechanism-confirmation test, and it is measured, not guessed.** The
analogue of `displacement`'s n=3 `answered` 6.82 → 14.00. Replaying all 49 real
word-recognition write sequences through the real `ChunkBoundedMemory` class
bounds 113 values, drops 2 309 elements, and takes elements/store from 44.90
(max 180) to **11.10** (max 16), with **distinct studied words in the store
falling 17.78 → 4.62** (max 53 → 11).

The two legs are scored **separately**, because they do not test the same thing:

* **Leg A** (elements/value ≤ 5.0, elements/store ≤ 16, `max == 4`) is an
  *invariant* of the rule, in the same class as P3. If elements/store stays above
  16, the bound is not being applied and the run is void rather than failed.
* **Leg B** (studied words ≤ 8.9, expected ~4.6) is the only *behavioural*
  measurement, and it is a bet on the agent's strategy rather than a consequence
  of the rule. Check the ceiling: the bound permits 4 slots × 4 elements = **16**
  elements, so an agent that *reallocates* — four slots each holding four studied
  words instead of one slot holding 99 — lands at ~16 against a baseline of 17.78.
  The rule alone therefore guarantees only ~10% on this leg. The replay's 4.62
  arises because the observed sequences dump the list into a single `recent_items`
  slot, leaving the other three holding non-list content. So the bar is "at least
  halves" (≤ 8.9), and there is an explicit **third outcome**: studied words in
  8.9–16 *with* leg A satisfied means the bound held and the agent reallocated to
  fill the ceiling — informative about adaptation, not a mechanism failure.

Precision about what this leg proves: it is proof against **punctuation gaming**
(a space-separated list satisfies an element count trivially while keeping every
word), which is what it was built for. It is **not** proof against slot
reallocation. Those are different channels, and the third outcome above is how
the second one gets recorded rather than misread as a failure.

**P2 — did the bound bound information, or only punctuation?** Words per value on
word recognition must fall from 22.62. Replay gives 8.10; the threshold is 14.
Read together with P1's studied-word count: low elements + high words + high
studied-word count is evasion; low elements + high words + low studied-word count
would be genuine prose consolidation and a *good* outcome, which is why P2 alone
is not treated as disconfirming.

**P3 — plumbing.** Injection rebinds `WorkingMemoryAgent` in 8 modules. If any
`final_kv` value anywhere carries more than 4 elements, some task is still
running the original store and the run is void, not merely a failure. This is the
same class of check that caught the "injection cannot patch one module" bug.

**P4 / P5 — story recall, with length and verbatimness predicted separately**, per
the A3 defect. They point *different ways* and that is the whole reason the brief
asked for them to be split.

* **Length falls, and this is a cost I am pre-registering as a cost.** The
  reconstructed KV concatenation goes 108.73 → 99.18 words on replay; the logged
  `recall_text`/concatenation ratio in the baseline run is 121.39 / 108.73 =
  1.116, so the expected `A3.words` is ≈ **111**. Human is 137.2, so
  `word_distance` moves 15.8 → ≈ 26 — **away** from human, on the one A3 quantity
  with a live human reference, and in the opposite direction from displacement,
  which improved it to 9.0. The `A3_WORD_TOLERANCE` of 40 absorbs it so the guard
  clears, but the guard clearing is not the same as the move being good, and I am
  not going to present it as one. Disconfirmed above 125 or below 95; and "above
  125" has two readings that **P1 disambiguates** — bound-never-fired shows `max
  elements/value > 4`, whereas bound-held-plus-adaptation (spreading a 9-element
  value over two slots, or switching to prose, which passes the bound entirely)
  shows compliant element counts with high word counts.
* **Verbatimness stays flat, which is the good news and it is measured.** Median
  clipped 4-gram precision on the reconstructed recall goes 0.0450 → **0.0405**
  while length drops 8.8% — i.e. the values that survive are no more surface-form
  than before. *Proxy error stated up front:* my reconstruction gives 0.0450 on
  the baseline against the contract's `recall_text` figure of 0.0414, so read the
  threshold with ±0.004 of proxy error. Two-sided on purpose: truncation keeps
  the opening of a value and drops trailing clauses, so if the dropped clauses
  were the *paraphrased* ones, precision could rise. **Disconfirmed above
  0.0613**, where `precision_distance` exceeds 0.0221 + 0.02 and the guard fires.
  A drop in BLEU, per the brief, will not be treated as evidence of anything.

**P6 — story recall's score, with no direction asserted, and the reason is a base
rate rather than a hunch.** Humanlikeness is a distributional distance, and *every*
prior perturbation of this store lost here: `displacement` −0.0509 and
`random_decay` −0.1548 both reduced what was retained, while `full_context`, which
*added* content instead, lost 0.339 (0.9473 → 0.608). Both directions lose, which
is the signature of a task sitting near a local optimum where distribution *shape*
dominates rather than one with headroom in a particular direction. So a structured
reduction can move it either way, and guessing would produce an uninformative FAIL
— exactly the reasoning iteration 2a used to decline a direction on word
recognition. Disconfirmed below −0.03 (the floor `displacement` violated at
−0.0509): the bound would then have cut chunk *content* rather than enumerated
surplus, and this task would be the wrong place to have done it.

**P7 — the primary no-change control, and it is derived rather than asserted.**
craft_task holds 1.00 elements per value, max 1, over 500 values; **0 of 500 are
bounded on replay**, the replayed store is byte-identical to the baseline's, and
no tool-result sentence is ever emitted (§4). So craft_task movement past its
0.025 noise floor cannot come from the bound — it would have to come from
something I did not intend to change. *Capable of varying:* yes — it moved −0.028
under `displacement` and +0.0368 under `random_decay_v2`. This is the analogue of
iteration 1's clean digit-span no-change, which is what made its attribution
unambiguous. Note that digit span **cannot** serve that role here, by measurement
(P11).

**P8 — nback, with its limitation stated.** nback values hold 1.13 elements each,
max **2**, over 325 values in the iteration-0 run and 322 in iteration 1, so the
bound cannot bite: three letters is 3 elements, four is 4, both inside the cap.
Because I build on `baseline` and not on `displacement`, the expected values are
the **baseline's** (`answered` 6.82, `keys_held` 3.96), not displacement's 14.00.
*Honest limitation:* `wm_nback.jsonl` records no tool calls, so this is the one
prediction that could **not** be replayed; it rests on 647 observed `final_kv`
values across two runs whose maximum is 2 elements. *Capable of varying:* yes —
`answered` moved 6.82 → 14.00 between baseline and displacement, and `keys_held`
is 1.06 under `full_context`.

**P9 — variable_mapping.** 1.00 elements per value, max 1, over 527 values; the
bound cannot bite, and `step_logs` record no writes so it is not replayable
either. *Capable of varying:* barely — the store is off the causal path (674 of
1 500 questions are answered at 0.985 with the key evicted), which is exactly why
this is recorded as a no-change and not offered as a result. If it moves past
0.017, A4 becomes binding and must show ≥ 30 errors with `rc_ratio ≥ 1.15`.

**P10 — the anti-vandalism test, at row level and where the bound actually
bites.** The bound is deterministic, so a checker can recompute *which rows it
touched* from the pre-bound values in `encoding_log.tool_calls[*].arguments`. On
narrative_qa that splits the baseline run into **36 bounded rows** (accuracy
0.8194) and **14 unbounded rows** (accuracy 0.7500). Prediction: the accuracy
change is confined to the bounded rows, and on the unbounded rows |Δ| < 0.05. If
accuracy falls on rows the bound never touched, something other than the bound
changed the agent's behaviour — most likely the tool-result sentence leaking a
strategy change across rows, which is precisely the risk §4 is guarding. This is
also the row-level version of P7. Same split on story recall for reference: 131
bounded rows (`embeddingSimilarity` 0.5770) against 69 unbounded (0.5650).

**P11 — the anti-vandalism test at aggregate level, and the reason digit span is
not my control.** Measured: the bound fires on 28 of 190 digit-span rows, caps
elements/value at 13 → 4 and elements/store at 25 → 16, and costs about **one
digit** of retention at the long spans (span 15: 18.8 → 17.5; span 16: 19.4 →
18.0; span 18: 17.9 → 16.7). So digit span is *weakly* affected and I will not
claim it is untouched — the brief's suggestion that it would be heavily cut is
wrong because most digit-span values store digits without delimiters (1.94
elements/value mean). What the prediction asserts is the *shape* of the effect:
`sub_span_leak ≤ 0.1450` (baseline 0.1298, guard fires at 0.1598,
`random_decay_v2` hit **0.249**) with `best_span` at 18.4 ± 2.0 and both HL deltas
inside their noise floors (0.140, 0.059). **Disconfirmed if the leak rises past
0.1450 while best_span barely moves** — that is the signature of removing
information *below* the participant's own span, i.e. vandalism rather than a
bound. *Capable of varying:* yes — `best_span` went to 2.0 under `random_decay`,
8.2 under v2, 20.0 under `full_context`, and the leak to 0.500 / 0.249 / 0.032.

**P12 — the one directional claim on an axis with real headroom, and the
covariate that decides whether it counts.** Unlike iteration 2a I *can* derive a
direction here. The store loses 74% of its content and the distinct studied words
it holds fall 17.78 → 4.62, so a participant who actually consults the store now
finds far fewer studied words and should answer "new" more often: **miss_rate up
from 0.043, false-alarm rate flat or down from 0.127, ratio up from 0.340 toward
the human 6.09.** That is the humanlike direction and the exact opposite of
`displacement`, whose humanlikeness rose because false alarms nearly doubled
(0.127 → 0.211) — away from the human conservative bias.

**The bar is the baseline's own bootstrap CI, not bare movement.** A2's denominator
is small and dominated by the ~7 participants who actually consult the store, which
is why `a2_ratio_ci` exists at all; the baseline's `ratio_ci` is **[0.007, 0.71]**,
so the threshold is `ratio > 0.71`, and a ratio in (0.340, 0.71] is recorded as
movement in the predicted direction that is **not creditable on its own**. Without
that bar a directional axis claim could pass on noise, which is the one thing it
must not do.

Two further conditions, without which the move must not be credited:
  * **`trials_attempted` is a covariate, not a bonus.** PROPOSER.md is explicit
    that an A2 ratio gain bought by surviving longer or shorter is not creditable.
    If misses rise, strikes accumulate faster and trials_attempted falls
    (`displacement` 82.9 → 68.9; human 34.5; baseline 82.92). A ratio move
    accompanied by a large trials_attempted swing is **uninterpretable**, not a
    win, and must be reported as such.
  * **Disconfirming condition:** `fa_rate > 0.211` means the liberal-bias
    artifact has been reproduced and the gain must not be credited regardless of
    what the ratio did.

**P13 — the leak test, and it is genuinely risky.** If word_recognition HL moves
less than its 0.121 noise floor *despite* the store losing 74% of its content and
75% of its distinct studied words, that is the strongest confirmation yet that
the score is determined by the studied list re-presented in the recall prompt
rather than by the store — no prior candidate removed anything like this much.
Stated at risk: **iteration 1 pre-registered this same threshold and it FAILED at
0.1388**, because on a distribution where 36 of 50 participants sit above 0.98 and
7 below 0.04, six participants moving is worth a lot of `W_1`. I am predicting it
again anyway because my manipulation is different in kind and because both
outcomes are informative: movement would mean the leak reading needs revisiting
*or* that the 7 store-consulting participants moved a long way, and P12 tells
those two apart.

**P14 — the headline, and it is explicitly not the claim.** Stated before the run
so it cannot be claimed afterwards. A mean delta is only credible past 0.026, and
§1 shows why this candidate should not produce one: the task it transforms is the
uncreditable one, and the creditable tasks move by 0% (craft), ~1 digit (digit
span, inside a 0.140 floor) and −9 to −11% of stored words (story, narrative). The
claims are **P1** (the mechanism), **P12** (the A2 direction), **P13** (the leak),
**P10/P11** (that it is a bound and not vandalism), and the §1 correction to the
brief's premise.

### Deliberately not predicted

`narrative_qa`'s own HL delta has no derivable direction: 36 of 50 rows lose a
mean of 4.2 elements each, and on an inverted objective with the baseline at
0.9572 a capability loss can move HL either way. It is reported, with the floor
(Δ > −0.03) as the only assertion, and P10 carries the interpretable content.

### For the coordinator: what the check function should assert

`chunk_limit_checks(run_dir, baseline)`, mirroring `displacement_checks`' row
shape (`{prediction, verdict, observed, threshold, note}`) plus the
`baseline_value` / `capable_of_varying` fields `pending_primacy.json` introduced.
Recompute recipes, all deterministic:

* **elements(value)** — `sum(1 for p in re.split(r"\s*(?:,|;|\n|\r|\s/\s)\s*", v) if p.strip())`,
  which is `count_elements` in `harness.py`; import it rather than re-deriving it.
* **P1 / P2 / P3** — over `final_kv` in each `tasks/wm_*.jsonl`. The
  *studied-words* leg: distinct lowercase `[a-z]+` tokens shared between the
  concatenated `final_kv` values and `encoding_log.content` on
  `wm_word_recognition.jsonl`.
* **P4 / P5** — `axes()["A3"]["words"]`, and `axes()["A3"]["precision"]` /
  `["precision_distance"]` — the length-free median clipped 4-gram precision that
  replaced BLEU as the enforced quantity this wave (`A3_ENFORCED_FIELDS =
  ("precision_distance", "word_distance")`, computed by
  `error_structure.a3_precision_model`). Do not score BLEU. Note the field is
  `precision` in the `axes()` dict even though `error_structure.py`'s docstring
  calls the quantity `verbatim_precision`.
* **P6 – P9, P13, P14** — `humanlikeness_by_task` / `delta_vs_baseline`;
  `nback_levels.report(run_dir)["diagnostics"][3]` for P8.
* **P10** — split each row of `wm_narrative_qa.jsonl` by
  `any(count_elements(json.loads(tc["arguments"])["value"]) > MAX_KEYS
  for tc in encoding_log["tool_calls"] if tc["name"] == "write_memory")`,
  computed **on the candidate run's own pre-bound arguments**, and compare
  `metrics.accuracy` against the baseline's same-split means (bounded 0.8194 over
  36 rows, unbounded 0.7500 over 14).
* **P11** — `axes()["A1"]` → `sub_span_leak`, `best_span`.
* **P12** — `axes()["A2"]` → `miss_rate`, `fa_rate`, `ratio`, and
  `trials_attempted` reported as a covariate with the uninterpretability note.

Per the brief I have **not** touched `check_predictions.py`. Two things only the
coordinator can do: wire `chunk_limit_checks` into the `CHECKS` dict, and **add
`chunk_limit` to `meta_harness/logs/pending_eval.json`** — PROPOSER.md step 5
requires the listing and my brief forbade me from editing that file, so without it
this candidate will never be scheduled.

---

## 8. Verification performed (no GPU, no network, no spend, no paid API call)

* `python meta_harness/verify_interface.py meta_harness/candidates/chunk_limit/harness.py`
  → **PASS**. Overrides `WorkingMemoryAgent` (8 modules), `MAX_KEYS` (10),
  `TOOLS` (1), `CONDITION_PROMPTS` (1), `SummarizerAgent` (8); round trip
  completes; capacity enforced at 4.
* In-process injection + unit checks: the real 99-item value → 4 elements; a
  4-comma prose value stored unchanged; `"1945"` and `"1-9-4-5 as a date"`
  untouched; refusal path byte-identical with `n_bounded == 0`; overwrite-when-full
  still admitted and bounded; `MAX_KEYS = 2` injected → within-slot cap follows to
  2 (the bound is not frozen at import); 400-write fuzz holds `≤ 4` slots **and**
  `≤ 4` elements/value after every write; repeating a sequence gives an identical
  store; `to_recall_text()` and `snapshot()` byte-identical to the baseline's; no
  `random` import.
* **Offline replay of the real write sequences** through the real
  `ChunkBoundedMemory`, on both the iteration-0 and iteration-1 runs. Replay
  fidelity against the observed `final_kv` is **1.000 on all five replayable
  baseline tasks**, so the replay is faithful before anything is claimed from it:

| task (baseline sequences) | rows touched | values bounded | elements dropped | elements/store | words/store |
|---|---|---|---|---|---|
| word_recognition | 34 / 50 | 113 | 2 309 | 44.90 → **11.10** | 76.92 → 27.54 |
| semantic_story_recall | 112 / 200 | 214 | 334 | 13.98 → 12.40 | 108.73 → 99.18 |
| narrative_qa | 36 / 50 | 66 | 150 | 13.54 → 11.46 | 84.54 → 75.12 |
| digit_span fwd / rev | 28 / 190 | 34 | 73 | 6.75 → 6.37 | 22.67 → 21.59 |
| craft_task | **0 / 150** | **0** | **0** | 3.33 → 3.33 | 18.67 → 18.67 |

* **Not possible offline, and not claimed:** any end-to-end behavioural number.
  Once the tool results differ the agent writes different values, so the replays
  bound the *mechanism*, not the outcome. `nback` and `variable_mapping` log no
  tool calls, so P8 and P9 rest on observed `final_kv` element counts rather than
  on a replay, and this is said rather than glossed.
* No file under `runs/human/` was read. Every human constant quoted here comes
  from `PROPOSER.md`, `score_candidate.py` or `error_structure.py`.
* Nothing outside `meta_harness/candidates/chunk_limit/` and
  `meta_harness/logs/pending_chunk_limit.json` was written. `bench/`, `src/`,
  `data/`, `runs/`, `score_candidate.py`, `check_predictions.py` and every other
  candidate's directory are untouched. No commit was made.
