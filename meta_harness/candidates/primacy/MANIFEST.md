# `primacy` — iteration 2a

Parent `displacement` (iteration 1). Grandparent `baseline`.

One method body differs from the baseline: `WorkingMemory.write_key`. One thing
differs from `displacement`: **which** entry overflow removes. `MAX_KEYS` is 4,
`TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS` and `_tool_call_cap()` are
untouched, `recall()`'s context construction is untouched (iteration 2b owns it),
and `step()` / `reset_messages()` are untouched (iteration 2c owns them).

---

## 1. Diagnosis

Iteration 1 confirmed its mechanism and was rejected for a reason it also
predicted: pure least-recently-refreshed displacement has no primacy, so the
earliest-encoded gist is the first thing evicted.

Measured, from the run records:

| | story-recall surviving keys (most common) | story HL | A3 BLEU @ words |
|---|---|---|---|
| baseline (refuse on full) | `setting`, `beginning`, `first encounter`, `Pie Man`, `main_character` | 0.9473 | 0.0031 @ 121.4 |
| displacement (evict LRU) | `Pie Man`, `second appearance`, `key_moment`, `Mabel and Reggie`, `emotional_impact` | 0.8964 | 0.0413 @ 146.2 |
| *human* | — | — | 0.002 @ 137.2 |

I quantified the shift rather than reading it off the key names. Replaying every
observed write sequence from both runs through all three overflow rules, and
scoring **serial-position survival** on the rows that actually overflow:

    semantic_story_recall, 139 overflowing rows of 200 (displacement's sequences)
      write position          1      2      3      4      5      6
      refuse (baseline)     .235   .251   .257   .257     0      0     pure primacy
      evict LRU (iter 1)      0    .022   .251   .251   .251   .224   pure recency
      THIS RULE             .235   .251     0    .028   .257   .229   U-shaped

Humans show both ends (Murdock 1962). Neither existing rule shows both: the
baseline retains only the head, displacement only the tail. The diagnostic
summary statistic is **P(the first-written AND the last-written chunk both
survive)**, over overflowing rows:

| | baseline | displacement | this rule (offline replay) |
|---|---|---|---|
| semantic_story_recall | **0.000** | **0.000** | **0.914** / 0.993 |
| narrative_qa | 0.000 | 0.000 | 1.000 |
| word_recognition | 0.028 | 0.054 | 0.811 / 0.917 |

(Two figures where the write sequences from both prior runs were replayed. The
0.000s are the *observed* `final_kv` of the real runs, not a model of them.)

Overflow is not rare, so this is not a corner case: 68–70% of story-recall rows,
68–74% of narrative-QA rows, 72–74% of word-recognition rows and 33% of
craft-task rows overflow. Digit span overflows on 6 of 190 rows, and
`variable_mapping` never does.

---

## 2. Mechanism, and the psychological account

**The rule.** Each entry carries a list of *presentations* `(tick, weight)`.
Overflow evicts the entry with the lowest ACT-R base-level activation,

    A_i = ln( SUM_k  w_k * (T - t_k)^(-d) ),        d = 0.5

and admits the new key, naming the loss in the tool result exactly as
`displacement` did. Presentations are:

* an item's **own write** — full attention, `w = 1`;
* **cumulative rehearsal**: when another item is written *in the same
  presentation episode* while this item is resident, one unit of attention is
  shared over what is held, `w = 1/|store|`.

A presentation episode is one `step()` presentation. It is read off the agent's
message list (the number of user turns), not by overriding `step()`, so this
candidate stays off iteration 2c's surface.

**Why this produces primacy.** This is the Atkinson & Shiffrin (1968) account
verbatim: the first items of a list get extra rehearsal *because the buffer is
not yet full*, so rehearsal effort is concentrated on few items. With one unit of
attention shared over the items held, the first item is rehearsed at 1/2, the
second at 1/3, the third at 1/4, and the fourth not at all before the store
saturates — attention sharing in the sense of Barrouillet & Camos's (2007)
time-based resource sharing, where refreshing is a limited resource divided among
maintained traces. Scored by the base-level equation, the resulting activations
at the fifth write are 1.274 / 1.063 / 0.957 / 1.000 for positions 1–4: the
**middle** is weakest. That is the prediction that distinguishes this from
"protect the first N", and the replay confirms it — position 3 is what dies, then
4, then 5, outward from the middle.

**Why it degenerates correctly in continuous tasks.** Rehearsal credit accrues
only within an episode. In a running-update task — one write per turn, which is
what n-back does — every resident has a single presentation, `A_i` reduces to
`(T − t_i)^(−0.5)`, argmin is the oldest entry, and the rule *is* `displacement`.
Verified in-process: eight writes over eight `step()` calls leaves
`p5,p6,p7,p8`, contiguous, exactly as displacement did. This asymmetry is the
empirically right one, not a convenience: running-memory span shows recency
**without** primacy (Pollack, Johnson & Knaff 1959; Hockey 1973; Bunting, Cowan &
Saults 2006), while free recall of a presented list shows both. A mechanism that
imposed primacy on n-back would be wrong about humans *and* would pay back
iteration 1's only confirmed gain, since 3-back needs four contiguous slots.

**Why nothing becomes immortal.** Credit accrues only inside an episode, and an
episode here is at most six writes long — `_tool_call_cap()` is
`max(6, 1.5 × interactions)` and a batch encode is one interaction, and the
observed write-count distribution on story recall is `{4: 61, 5: 15, 6: 124}`,
i.e. 62% of rows already sit at that ceiling. Once the episode ends every
presentation decays as `T^(−0.5)`, so every entry becomes evictable again. The
cap also bounds the obvious counter-move: the eviction message names a *middle*
key, and if the agent rewrites it, it re-enters with a single age-1 presentation
(sum 1.0) against primacy items at ~1.3 and dies at the next write — but the
budget of 6 stops that loop from running away, so the agent reacting to the
message cannot undo the U-shape.

### Accounts considered and rejected

| account | what it would do here | why rejected |
|---|---|---|
| **Oberauer & Kliegl (2006) overflow-as-overwriting** | overflow overwrites rather than refuses | *already implemented* — this is exactly `displacement`, and it is the part being kept, not the part being added. It says nothing about *which* item is overwritten. |
| **Pure A&S rehearsal count** (strength = number of rehearsals, no decay) | early items accumulate unbounded strength | immortality. Simulated: positions 1–2 freeze permanently and never become evictable, which re-creates the baseline's jam by another route and breaks 3-back. The power-law decay term is what makes the mechanism recoverable. |
| **TBRS with refreshing suspended during processing** (Barrouillet & Camos's strict cognitive-load reading: attention cannot refresh while it is encoding) | no covert rehearsal during a write burst | gives pure recency and therefore *no primacy at all* — i.e. it reproduces `displacement` and does nothing. TBRS's attention-*sharing* claim is the usable part; its strict serial-bottleneck claim is not, because a batch encode is one processing episode with nothing else in it. |
| **Page & Norris (1998) primacy gradient at encoding**, `w_i = φ^(i−1)` or `1/i` | encoding strength declines with write index | two problems. (a) Indexed globally it makes *late* items permanently weak, so a continuous task freezes on its early keys — simulated, and it breaks n-back. (b) The gradient slope is a free fitted parameter, which is exactly the "tuned constant" this search is supposed to avoid. |
| **Activation decay with retrieval-induced strengthening** | retrieval boosts what is retrieved | no retrieval events exist inside the encode phase to hang it on; `recall()` happens once, after the store is final. It would be an unexercised mechanism. |

**The parameter claim, stated plainly.** There is exactly **one** numeric
parameter: `d = 0.5`, the canonical ACT-R base-level decay (Anderson & Lebiere
1998; Anderson & Schooler 1991 derive `d ≈ 0.5` from the statistics of when
information is actually needed again). It is not fitted here and was not varied.
The primacy gradient's slope is not chosen at all — it falls out of
`MAX_KEYS = 4` through 1/n attention sharing. No sweep was run and no threshold
in this file was moved after seeing a result.

### Read-out is deliberately the baseline's

`snapshot()` and `to_recall_text()` are **not** overridden. `displacement`
reordered the read-out by recency and tagged the last entry `(most recent)`; that
is a change to the recall-time representation rather than to the store's
dynamics, and keeping it would confound the comparison against the baseline —
which is the comparison every delta, floor and guard in `score_candidate.py` is
computed against. It is also the wrong read-out for this store: the surviving
keys sit in `_store` in write order, so the baseline's plain iteration already
presents them in **serial order with a gap where the middle was lost**, which is
what a primacy-plus-recency store should hand to recall. Consequence to note
honestly: this candidate differs from `baseline` in one thing (the overflow rule)
and from `displacement` in two (the victim choice and the read-out format), so
the baseline comparison is the clean one.

---

## 3. Not a capacity increase, and not noise

**Not capacity.** `MAX_KEYS = 4`. At most four entries exist at any instant;
`verify_interface.py` writes `MAX_KEYS + 3` keys and confirms it, and a
200-write fuzz asserts `len(store) ≤ 4` after every single write. Mean keys held
in the offline replay is 3.87 against the baseline's 3.87 and displacement's
3.92 — this rule holds *no more* than either. `_tool_call_cap()` is untouched, so
no gain can be read as extra compute. `full_context` already settled that
capacity is the wrong direction (0.6387 against 0.7861).

**Not noise.** No random number is drawn anywhere; `random` is not imported. The
run is deterministic given the LLM's outputs, and was verified so (the same write
sequence twice gives the same surviving set). More to the point, the failure mode
that killed `random_decay_v2` is directly tested against:

* v2 bought the aggregate and broke the error structure — `best_span` 18.4 → 8.2
  (human 6.88) while A1 sub-span leak went 0.130 → **0.249** (human 0.087). This
  candidate predicts A1 and `best_span` **unchanged**, because digit span
  overflows on 6 of 190 rows and 184 of 190 replayed key sets are identical to
  displacement's, which were byte-identical to the baseline's.
* The signature in §1 is one a noise generator cannot produce. Uniform random
  retention of 4 of 6 chunks gives P(first and last both survive) ≈ 0.4; a
  recency rule gives 0.000; a refusal rule gives 0.000; this rule gives 0.914.
  The prediction is ≥ 0.80.

---

## 4. The A3 BLEU rise — measured, and it is not what the worklog guessed

The worklog carried this forward as unexplained, with "chunks kept by recency are
later and less abstracted, so what survives is closer to surface form" as the
likeliest reading. **That reading is wrong.** Two measurements:

**(a) Length-free verbatimness is flat across serial position.** For every
written chunk, the fraction of its n-grams that occur in the story transcript —
BLEU's precision term with no brevity penalty:

    write position        1      2      3      4      5      6
    p1 (unigram)        .808   .764   .793   .745   .720   .704
    p4 (4-gram)         .066   .068   .045   .073   .029   .113

Position 5 is the *least* verbatim chunk and position 1 is as verbatim as
position 6. Later chunks are not closer to surface form.

**(b) It is a brevity-penalty effect, entirely attributable to recall length.**
`recall_text` on this task is essentially the concatenation of the surviving KV
values (reconstructing it from `final_kv` reproduces the logged BLEU to 0.0335 vs
0.0413). Holding the agent's **written value text fixed** and changing only which
subset is retained:

| retained subset | recall words | BLEU |
|---|---|---|
| refuse (head) | 122.1 | **0.0048** |
| evict LRU (tail) | 140.5 | **0.0335** |
| this rule (head + tail) | 124.4 | **0.0087** |

A 7× BLEU swing from identical sentences. The cause is `sacrebleu`'s brevity
penalty against a ~980-token reference. Within a single row, adding chunks:
39 words → 3.5e-9, 58 → 5.7e-6, 95 → 0.0034, 119 → 0.0282. Across runs the
relation is monotone in length and nothing else: 65 w → 0.0001
(`random_decay`), 121 w → 0.0031 (baseline), 146 w → 0.0413 (displacement),
411 w → 0.3047 (`full_context`). Binned within the displacement run:
`[100,125) → 0.0039`, `[125,150) → 0.0117`, `[150,175) → 0.0892`.

So displacement's guard failure is attributable to its recall length moving
121.4 → 146.2, i.e. **toward** the human 137.2 (word_distance 15.8 → 9.0, an
improvement), not to any increase in verbatimness. Displacement's manifest
claimed displacement "cannot cause regurgitation"; that claim was wrong, but the
mechanism is composition-of-retained-chunks driving length, not regurgitation.

**Note for the evaluation contract, not a request to change it.** In this length
regime the A3 BLEU tolerance of 0.02 absolute acts as a length cap near ~140
words, which is tighter than, and points the opposite way from, the ±40-word band
the A3 word guard permits. The two are not strictly incompatible — the
`[125,150)` bin averages 0.0117, inside the 0.0231 cap — but the BLEU axis is
currently measuring length far more than it is measuring verbatimness. A
length-matched BLEU, or reporting the n-gram precisions without the brevity
penalty (table (a) above), would separate them. I have not touched
`score_candidate.py`.

**What this predicts for this candidate:** BLEU falls back to roughly 0.009–0.02
with recall length near the baseline's, so the guard should not fire. That is
prediction P5 and it is the riskiest one here.

---

## 5. Pre-registered predictions

Written before the run. Machine-readable in
`meta_harness/logs/pending_primacy.json`. Every threshold is checked against the
measured noise floors in `score_candidate.py`, and every quantity was checked to
be *capable of varying* — the `capable_of_varying` field in the JSON records the
evidence, because iteration 1's buffer-compliance prediction was VOID rather than
failed.

| # | quantity | threshold | baseline | displacement | role |
|---|---|---|---|---|---|
| **P1** | `semantic_story_recall` HL | `abs(Δ vs baseline) < 0.03` **and** `Δ vs displacement > +0.025` | 0.9473 | 0.8964 | **primary** |
| **P2** | overflowing story rows where the first-written **and** last-written key both survive | **≥ 0.80** | 0.000 | 0.000 | **anti-noise** |
| **P3** | `nback` n=3 `keys_held` / `answered` / `acc_over_answered` | `≥ 3.5` / `≥ 13` / `0.687–0.787` | 3.96 / 6.82 / 0.737 | 3.98 / 14.00 / 0.744 | **no-change** |
| **P4** | `A1.sub_span_leak`, `A1.best_span` | `0.1298 ± 0.01`, `18.4 ± 1.0` | 0.1298 / 18.4 | 0.1298 / 18.4 | **no-change** |
| **P5** | `A3.bleu`, `A3.words` | `< 0.023` and `115 ≤ words ≤ 145` | 0.0031 / 121.4 | 0.0413 / 146.2 | riskiest |
| P6 | 3-task gist subgroup mean (story, craft, narr) | `≥ 0.9181` **and** `> 0.9161` | 0.9317 | 0.9025 | subgroup |
| P7 | 5-task store subgroup mean | `≥ 0.9139` | 0.9296 | 0.9148 | subgroup, **under-powered** |
| P8 | `craft_task`, `narrative_qa` HL | `Δ > −0.03` each | 0.8907 / 0.9572 | 0.8627 / 0.9483 | floor |
| P9 | 8-task mean HL | `≥ 0.80` | 0.7861 | 0.8122 | headline, not the claim |
| P10 | `variable_mapping` HL | `abs(Δ) < 0.017`, A4 owes nothing | 0.3554 | 0.3484 | no-change |

### What each one is for

**P1 — the task this candidate exists to recover.** Two-sided on purpose. A
large *gain* would not confirm the mechanism: the retained set now covers the
beginning *and* the end of the narrative arc, so judged coverage could rise, and
on an inverted objective a capability gain costs humanlikeness. The claim is
recovery to the baseline, not improvement past it. Noise floor 0.011, so both
legs are above it. **Disconfirmed** if story recall ends at or below 0.9214 —
primacy protection then did not recover what displacement lost, and the
primacy/recency reading of the loss is wrong.

**P2 — the one a noise generator cannot pass.** Both prior candidates score
exactly 0.000; uniform random retention would give ≈0.4; this rule gives 0.914 on
replay. **Disconfirmed** below 0.60. Secondary form: mid-list positions (3 and 4
of 6) should be the least-retained positions, which "protect the first N" does
not predict.

**P3 — the no-change that protects iteration 1's only confirmed win.** Honest
limitation: `wm_nback.jsonl` logs no tool calls, so *this one could not be
replayed offline*. It rests on the argument that n-back writes one key per turn
(supported by iteration 1's record: the baseline's non-contiguous
`position_2/3/8/10` sets requiring delete+write, and displacement ending on four
contiguous positional keys) plus the in-process demonstration that one write per
`step()` reduces this rule to displacement exactly. `variable_mapping`'s
`step_logs` record zero writes per episode, so they could not ground it either.
**Disconfirmed** if n=3 `keys_held < 3.5` or `answered < 13`: the episode-scoping
of rehearsal is then not doing what is claimed, and the likeliest cause is turns
that emit two writes (in which case the first of the pair gains protection — an
in-process test of that mixed case leaves `q1` resident).

**P4 — the no-change that makes attribution clean**, exactly as iteration 1's
was. Also the direct contrast with `random_decay_v2`, which moved both.

**P5 — the riskiest.** Grounded in §4's replay (0.0087 @ 124.4 words on identical
value text), but the agent's own writing behaviour will differ from the sequences
replayed, so the number will move. **Disconfirmed** if BLEU ≥ 0.023, which is
where the guard fires.

**P6 / P7 — subgroup means, and a correction to the brief.** The brief asks for a
prediction on "the 5 tasks that measure the store" (baseline 0.9296, displacement
0.9148, Δ −0.0148). From `metric_noise.json`, that subgroup's mean delta has
`SE = sqrt(Σ SE²)/5 = 0.0157`, i.e. **min credible delta 0.0314** — so
displacement's −0.0148 is *inside* its own noise band and is not a credible
regression. 87% of that SE comes from the two digit-span cells (SE 0.0698 and
0.0294), which displacement moved by 0.0000 and +0.0138. Dropping them gives the
three gist tasks, `SE = 0.0068`, **min credible delta 0.0136**, on which
displacement's mean of −0.0293 *is* credible. P6 is the informative version and
P7 is reported for comparability. This is not subgroup shopping: the subgroup is
defined by dropping the two cells that carry almost all the noise and that the
parent candidate did not move, and the definition is fixed here before the run.
The solid grounds for rejecting `displacement` remain the `semantic_story_recall`
floor violation (−0.0509 against −0.03) and the A3 guard, not the 5-task mean.

**P9 — not the claim.** A mean delta is only credible above 0.026. This candidate
is expected to hold displacement's n-back gain, recover story recall, and make no
claim on word recognition, which lands the mean somewhere around 0.80–0.81 —
i.e. plausibly *inside* the credible band relative to displacement. The primary
claims are P1–P5.

### Deliberately NOT predicted: `word_recognition` and A2

Reported as covariates with no directional prediction. Three reasons: the task is
a confirmed leak (36 of 50 participants read the studied list off the recall
prompt), its distribution is bimodal so six participants moving is worth a lot of
`W_1` — which is exactly how iteration 1's threshold here failed — and iteration
2b owns that surface. This rule *will* change what the store holds there (74% of
rows overflow, and replay puts P(first and last survive) at 0.811 against
displacement's 0.054), but I cannot derive a direction for the miss/false-alarm
asymmetry from that, and iteration 1's experience is that guessing produces an
uninformative FAIL.

A **disconfirming condition** instead, which does have teeth: if
`word_recognition` moves by more than its 0.121 floor *and* `A2.fa_rate` exceeds
displacement's 0.211, then this rule is producing the same liberal-bias artifact
displacement did — humanlikeness rising because the model says "old" more often,
away from the human conservative bias — rather than a primacy effect, and the
gain should not be credited. For context, displacement's A2 went the wrong way
(distance 5.754 → 5.920) while its trials-attempted covariate fell 82.9 → 68.9.

### For the coordinator: what the check function should assert

`primacy_checks(run_dir, baseline)` should mirror `displacement_checks`' row shape
and assert, in order: P1 (two-sided, needs both baseline and displacement runs,
or baseline plus the recorded 0.8964), P2 (computed from `final_kv` and the write
order recovered from `turn_logs[*].tool_calls` on `wm_semantic_story_recall.jsonl`,
restricted to rows with more than `MAX_KEYS` distinct written keys), P3 from
`nback_levels.report(run_dir)["diagnostics"][3]`, P4 from `axes()["A1"]`, P5 from
`axes()["A3"]`, P6–P10 from `humanlikeness_by_task`. The word-recognition entry
should be reported as INCONCLUSIVE by construction unless the disconfirming
condition above fires, in which case FAIL with that note.

---

## 6. Verification performed (no GPU, no network, no spend)

* `python meta_harness/verify_interface.py meta_harness/candidates/primacy/harness.py`
  → **PASS**; overrides `WorkingMemoryAgent`, `MAX_KEYS`, `TOOLS`,
  `CONDITION_PROMPTS`, `SummarizerAgent`; rebound in 10 / 8 / 1 / 1 / 8 modules;
  capacity enforced at 4.
* In-process injection + scripted-stub round trips: 6 writes in one `step()` →
  `{c1, c2, c5, c6}` (U-shape); 8 writes over 8 `step()` calls → `{p5, p6, p7, p8}`
  (pure recency, identical to displacement); read-out is the baseline's plain
  `key: value` with no recency tags; 200-write fuzz keeps `≤ 4` keys and the
  `_pres`/`_ep` bookkeeping exactly tracks `store`; repeating a sequence gives an
  identical result.
* Replaying all 139 / 136 overflowing story-recall write sequences from the real
  iteration-1 and iteration-0 runs through the **real** `PrimacyMemory` class
  reproduces P(first and last survive) = 0.914 / 0.993.
* No `runs/human/` file was read. All human constants used here are the ones
  already written into `PROPOSER.md` and `score_candidate.py`.

Not possible offline, and not claimed: any end-to-end behavioural number. The LLM
will write different chunks once the tool results differ, so the replays bound the
*mechanism*, not the outcome.
