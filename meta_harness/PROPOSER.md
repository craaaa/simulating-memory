# Proposer instructions

You are proposing candidate harnesses for a Meta-Harness search over the
compactor: the 4-slot key-value working-memory agent that wraps a frozen LLM in
this repo. Read this before proposing anything.

> ## What iteration 2 superseded — read this first
>
> Five defects in the evaluation contract and two claims in this file were
> corrected after iteration 2. The full record is at the end of
> `meta_harness/WORKLOG.md`; these are the ones that change what you should do.
>
> **1. `variable_mapping` IS reachable, and it is the largest gain the project has
> produced.** This file says below that its headroom "is not reachable by anything
> you are allowed to change" and tells you not to target it. That was wrong.
> `episodic_reset` rebuilt `step()`'s message list each turn, closing the
> conversation-history leak, and moved it 0.3554 → 0.6764 (+0.3210), broke the
> ceiling point mass from 0.9867 to 0.2733, and took A4 from 12 errors to 509. The
> store was off the causal path *because the history leak kept it off*, not
> inherently. The analysis under "Two of the eight tasks do not test the memory
> module" is otherwise sound; its conclusion was not.
>
> **2. A3 no longer enforces BLEU.** BLEU's brevity penalty made it a length proxy:
> pooled Spearman(recall length, BLEU) = 0.822 over 1000 rows, and all 121 rows
> under 60 words score *exactly* 0.0000, so it cannot distinguish a verbatim short
> recall from an abstracted one. It also contradicted the A3 word guard, which
> rewards moving length toward the human mean. The enforced field is now
> `precision_distance` — clipped 4-gram precision, no brevity penalty, compared on
> **medians** because the human distribution is skewed (37 of 53 below 0.05, 5 above
> 0.5). Human median 0.0192, baseline 0.0414. BLEU is still reported, with
> `bleu_enforced: false`. **Do not threshold BLEU.**
>
> **3. A4 no longer enforces the raw `rc_ratio`.** Its ceiling moves with the error
> count — 1.2525 at 12 errors, 1.2575 at 35, 1.3750 at 400 — so the old fixed 1.15
> bar could never have rejected anything, and `displacement`'s 1.2575 at exactly 35
> errors was its arithmetic maximum. The guard now uses `rc_ratio_normalized` (0.0
> noise, 1.0 the run's own ceiling). **Humans sit at 0.3728, not near 1.0**, so
> "approach the human 1.386" is withdrawn as a target — it is unreachable below
> several hundred errors. Also: A4 had already fired before iteration 2 despite an
> earlier brief claiming otherwise, and its assignment window was under-counted
> about twofold until it was fixed.
>
> **4. `variable_mapping`'s two sides were scored by different formulas.** Human is
> `sum(q.correct)/10`, a correct count; the model's is `relation_count` of the last
> consecutively correct question, which saturates by question 5, so **every error
> after question 5 was invisible to the model's score** — `displacement` erred on 25
> runs and 24 still scored 1.0. The matched figure is reported as
> `axes()["variable_mapping_matched"]`. The "99% at 1.0, two unique values" point
> mass was substantially a scoring artifact.
>
> **5. The chunk defect is NOT task-general.** Unbounded value length concentrates
> almost entirely in `word_recognition` (13.21 elements per value, max 99). Story and
> narrative sit at ~3.8, already at a 4-element bound; `nback` maxes at 2;
> `variable_mapping` and `craft_task` at exactly 1 and cannot move at all.
>
> **6. The serving stack is NOT bit-deterministic — do not demand bit-identity.**
> An earlier version of this preface said the opposite, on the grounds that several
> candidates reproduced the baseline's scored humanlikeness exactly. That was my
> error: humanlikeness is a Wasserstein distance over a score *distribution*, so it is
> invariant to which participant got which score and to any generation change that
> does not move a score. Measured on `craft_task` with server-assigned tool-call ids
> stripped, against the baseline:
>
>     chunk_limit         provable no-op    7 of 150 rows differ   scored delta  0.0000
>     serial_recognition  provable no-op   12 of 150 rows differ   scored delta  0.0000
>     episodic_reset                      24 of 150 rows differ   scored delta -0.0280
>     primacy                             57 of 150 rows differ   scored delta -0.0451
>
> Real differences — the model writes `"rule1": "A and B make D"` in one run and
> `"A + B -> E"` in another. vLLM with continuous batching at 50-way concurrency is
> not reproducible at temperature 0, because batch composition changes reduction
> order.
>
> Two things follow for your predictions. **A differing-row count is a far more
> sensitive mechanism-confirmation quantity than a scored delta** — use it where you
> can compute it, and compare it against the 7–12 band two provable no-ops produced
> rather than against zero. And **`NOISE_FLOOR` understates the truth**, because
> `metric_noise.py` resamples the model side of a fixed set of rows and so cannot see
> generation-level variation between runs; a no-op moved `craft_task` past its 0.025
> floor. Do not threshold anything within about 0.03 of zero on a gist task without
> saying why it is safe.
>
> **7. Predictions are checked mechanically with four verdicts**, not two: PASS,
> FAIL, INCONCLUSIVE, and VOID. VOID fires when a quantity your row documents as
> capable of varying turns out identical to the baseline's to full precision. Every
> prediction row must therefore carry a `capable_of_varying` field with evidence.
> Three rows were VOIDed on real data in iteration 2, and two whole candidates were
> voided by their own pre-registered preconditions — which is the system working.

## What you are optimizing, and why it is backwards

The target is **humanlikeness**, `1 - W_1` between the model's and humans'
per-participant score distributions. The current compactor is *better than
humans* on 8 of 10 tasks. So you are searching for **calibrated degradation**,
not capability.

The direction is not uniform. N-Back is the one task where the model is *worse*
than humans (0.710 vs 0.866 locally), so it opposes every other task's gradient. A
single "forget harder" knob improves the others and breaks N-Back. Whether one
mechanism can satisfy both is an open question and is arguably the most interesting
result available here — do not paper over it.

Wave 0 confirmed the inversion empirically rather than by assertion:
`full_context` (capacity 10 000, stimulus stored verbatim) is the most capable
harness measured and the *least* humanlike, 0.6387 against the baseline's 0.7861.

## The thing that will tempt you, and why it fails

You can match a score distribution trivially: draw a random per-participant
"ability" and drop that fraction of what the harness stores. The spread widens
until it overlaps the human spread and humanlikeness goes up. **This is worth
nothing** — it is noise, not a memory model.

`candidates/random_decay/` already does exactly that, deliberately, as a control.
If your proposal amounts to the same trick with extra steps, it will be caught by
the axes below and rejected. Propose mechanisms, not noise.

For what it is worth, the trick also did not work: `random_decay` at uniform(0,0.9)
*lost* 0.088 of humanlikeness and collapsed digit-span best span from 18.4 to 2.0,
overshooting the human distribution rather than matching it. Matching a human score
distribution with undirected noise is harder than it looks, because the noise has
to be the right size on every task at once.

## Hard constraints

**Do not read `runs/human/`.** The human statistics you need are quoted in this
file and in `NOTES.md`. Reading the raw human data invites fitting constants to
it, and any numeric literal in a candidate within 10% of a human statistic is
rejected on review unless it is a named psychological constant with a citation.

**`MAX_KEYS` is searchable but expensive to change.** It encodes Cowan (2001)'s
~4-chunk limit, which is the theoretical claim the compactor exists to
instantiate. A candidate that moves it must argue for that explicitly in its
manifest — "it scored better" is not an argument. Same for any decay function: it
must correspond to a named mechanism (interference, trace decay, rehearsal
failure), not a fitted noise term.

**Every task must keep working.** No search task may regress below the baseline by
more than `max(0.03, min_credible_delta[task])` — see the noise section, because
for digit span that threshold is 0.140, not 0.03. The mean alone is not the
objective; a per-task vector is reported so a headline cannot be bought from one
cell.

## Two of the eight tasks do not test the memory module at all

Read this before you pick a target, because it invalidates the most attractive
cell in the table.

`WorkingMemoryAgent` keeps conversation history across `step()` calls, and
`reset_messages()` is never called by any task. `recall()`, by contrast, builds a
fresh prompt from the KV store alone. So tasks split by how they answer:

| regime | route | tasks | mean HL |
|---|---|---|---|
| **bottlenecked** | `encode()` → `recall()`, KV only | ds_fwd .886, ds_rev .967, story .947, craft .891, narr .957 | 0.930 |
| **leaky** | answers via `step()`, study history still in context | nback .791, variable_mapping .355 | 0.573 |
| **leaky** | `recall()`, but the prompt embeds the studied list verbatim | word_rec .495 | 0.495 |

Iteration 1 found the third one: `wm_word_recognition.py` formats `trials_text` —
every trial in order — into its recall prompt, so although it routes through
`recall()`, the stimulus is visible anyway. See the H2 withdrawal below.

On `variable_mapping`, conditioning each question on what the store actually held:

| store state at question time | n | accuracy |
|---|---|---|
| queried name present, correct value | 795 | 1.000 |
| queried name present but **stale** value | 31 | 0.935 |
| queried name **absent**, store had evicted it | 674 | 0.985 |

45% of questions ask about a name the store no longer holds and the model answers
them at 0.985; where the store actively contradicts the truth it still wins 93.5%
of the time. The store is not on the causal path. That is why capacity 4, capacity
10 000 and random decay all produce exactly 0.992.

**Consequence: `variable_mapping`'s 0.645 of apparent headroom is not reachable by
anything you are allowed to change.** Do not target it. If you move it, you did so
by manufacturing failures, and axis A4 will ask you to justify their shape.

> **WRONG, corrected by iteration 2 — and this was the most consequential error in
> this file.** The table above is accurate and its inference was not. The store is
> off `variable_mapping`'s causal path *because the conversation-history leak keeps
> it off*: `step()` retains every stimulus verbatim, so the 674 questions asking
> about an evicted name are answered from history at 0.985. Close that leak and the
> store becomes the only route. `episodic_reset` did exactly that and moved the task
> 0.3554 → 0.6764, the largest single-task gain the project has produced, with the
> ceiling mass broken 0.9867 → 0.2733 and A4 errors 12 → 509 — trustworthy at scale
> for the first time, at `rc_ratio_normalized` 0.7302 against the human 0.3728.
>
> A4 did ask the candidate to justify the shape of those errors, exactly as this
> paragraph says it would, and the errors passed: they were load-ordered, not noise,
> and parse integrity held at 1 unparsed answer in 1500. What sank that candidate
> was an unrelated mechanics failure at n-back level 1, not the legitimacy of its
> variable_mapping gain.
>
> So: **do target it, by closing the leak rather than by manufacturing failures.**
> The distinction this paragraph was reaching for is real; the pessimism was not.

## Axes and guards, with the numbers

All baseline figures below are measured **locally** on this vLLM stack. The
released per-model numbers came from OpenRouter and do not reproduce at bf16;
ignore them.

| | human | baseline (qwen3-30b, local) | role |
|---|---|---|---|
| **A2** word-recognition miss/false-alarm ratio | **6.09** | 0.340 | **the axis** |
| A1 digit-span sub-span leak (protocol-matched) | 0.087 | 0.130 | guard |
| A1b digit-span **best span** | **6.88** | **18.4** | guard, and see below |
| A3 story-recall BLEU @ recall words | 0.002 @ 137 | 0.003 @ 121 | guard |
| **A4** variable-mapping error interference ratio | **1.386** | 1.17 (only 12 errors) | conditional guard |

Guards are **relative**: `|candidate − human|` must not exceed
`|baseline − human| + tolerance`. Absolute bands were tried and failed, because
they are serving-stack-dependent — the first A1 band was calibrated on OpenRouter's
0.105 and the local baseline came in at 0.130, i.e. the baseline failed its own
guard.

**A1 goes blind at ceiling.** `sub_span_leak` is only defined where the staircase
fails. `full_context` reached best_span 20.0 on a 19-span schedule, never
terminated, had no failures to leak, and therefore scored *closer* to human on A1
(0.032) than the baseline (0.130) purely by being uninformative. `at_ceiling` and
`best_span_distance` are now reported for this reason. Note also what best_span
says on its own: **the baseline reaches span 18.4 where humans stop at 6.88**, so
digit span is less closed than its 0.886 humanlikeness suggests — the marginal
score distributions line up while the staircase behaviour does not. Nothing
currently rewards fixing that, and it is a legitimate thing to go after.

**A4 catches unstructured degradation on variable_mapping.** Human errors are
interference errors: `relationCount` averages 6.18 on errors against 4.46 on
correct answers (ratio 1.386), and 69.1% of errors are intrusions — a city already
assigned to that name or another — against a 57.7% chance rate from option
composition. Random key-dropping produces errors independent of interference load,
so its ratio tends to 1.0. The guard is conditional: improve `variable_mapping`
past its noise floor and you must show ≥30 errors with `rc_ratio ≥ 1.15`, or the
gain is rejected as unstructured.

**A2 was billed as "where the headroom is". Treat that with suspicion now.** Humans
are conservative on recognition — they answer "new" when unsure, giving miss 0.272
against false-alarm 0.045. The baseline does the opposite. But the word-recognition
leak means 36 of 50 participants score ≥ 0.98 by reading the studied list off the
recall prompt, and A2's 0.340 is dominated by the 7 who actually consult the store.
So A2 measures something real about those 7 and almost nothing about the rest. A
candidate that closes the leak will change A2's *meaning*, not just its value —
expect the axis to need recalibration against a fresh baseline at that point, and
say so in your manifest rather than claiming the delta.

A2 has a trap: word recognition stops after 3 strikes, so trials-attempted
varies (34.5 for humans, 82.9 for the local baseline). You can move the ratio by
surviving longer rather than by fixing the asymmetry. Trials-attempted is
recorded as a covariate and an A2 gain that came from it will not be credited.

A3 is already near-human on this substrate, so it offers no gradient and exists to
catch regressions — which it does: it is what caught `full_context` regurgitating
the stimulus at BLEU 0.3047 and 411 words against a human 0.002 @ 137.

## Noise floor — do not chase what you cannot measure

Smallest per-task delta that is not sampling noise — 2× the bootstrap SE of a
*difference* between two runs, from `metric_noise.py`. These supersede the earlier
human split-half numbers, which ignored model-side sampling and understated the
real imprecision by up to 4×:

| task | n_model | min credible \|Δ\| |
|---|---|---|
| digit_span_forward | **10** | **0.140** |
| word_recognition | 50 | **0.121** |
| nback | 150 | 0.060 |
| digit_span_reverse | 10 | 0.059 |
| narrative_qa | 50 | 0.030 |
| craft_task | 150 | 0.025 |
| variable_mapping | 150 | 0.017 |
| semantic_story_recall | 200 | 0.011 |

The digit-span figures are that bad because `search_set.yaml` cuts those tasks to
190 rows for a 4× throughput win and `score.py` resolves 190 rows into just **10**
model participants. So a 0.10 swing on digit span is *nothing*. Watch `best_span`
there instead — it is a scalar over all 190 trials.

The mean is much better conditioned than any single task: `sqrt(Σ SE²)/8 = 0.013`,
so **a mean delta of 0.026 or more is credible.** Mean attainable ceiling is 0.955,
not 1.0. `history.py diff` marks per-task deltas that fall inside the noise. Do not
build a story on one.

## Where the headroom actually is, already located

Five of eight tasks sit at 0.89–0.97 against a 0.955 attainable ceiling and are
closed. `variable_mapping` is unreachable (see the leak). That leaves two cells,
and wave 0 already localised both — you are not expected to rediscover them, you
are expected to explain and fix one.

**H1 — N-Back's whole deficit is at n=3, and it is response omission, not bad
judgement.** Iteration 1 corrected two errors in the earlier version of this
section; both corrections were independently verified and are recorded here so
nobody re-derives them.

*There is no bimodality.* `score.py` scores the model as one observation per
`(participant, n_level)` — 150 points — while a human is one pooled observation
over all three of that participant's levels. The model's distribution therefore
spans the level effect and the human's averages it away:

    model, per-row (what humanlikeness uses)   mean 0.710  sd 0.321   HL 0.791
    model, pooled to human granularity         mean 0.710  sd 0.102   HL 0.844
    human                                      mean 0.866  sd 0.081

The "35% of participants at exactly 1.0" is exactly the 50 n=1 rows. Pooled, the
model's *shape* matches the human shape and the deficit is a clean mean shift.

Human records support the better fix: `payload.trials` carries a `level` field with
14 scored trials per level, so both sides can be compared per level
(`meta_harness/nback_levels.py`). Doing so localises the entire deficit:

| n | model | human | HL | answered/14 | acc over answered | keys held |
|---|---|---|---|---|---|---|
| 1 | 0.993 | 0.946 | 0.942 | 13.98 | 0.994 | 1.00 |
| 2 | 0.779 | 0.862 | 0.919 | 13.24 | 0.821 | 1.54 |
| 3 | **0.360** | **0.780** | **0.583** | **6.82** | **0.737** | **3.96 — jammed** |

n=1 and n=2 are effectively solved. At n=3 the store saturates and the model
**stops answering**, while staying 0.737 accurate on the trials it does answer.
Compare `full_context`: 14.00/14 answered, acc-over-answered only 0.770, **and it
holds just 1.06 keys** — with capacity 10 000 it holds *fewer* keys than the
baseline, because never refusing lets it overwrite one rolling key instead of
accumulating a jammed set of four. So its entire +0.157 is response production, and
the mechanism is the overflow rule, not capacity. `answered` can rise for dull
reasons; buffer-phase compliance cannot, so check `model_parsed_buffer` too (the
baseline emits only 50 `No response` of 150 buffer slots at n=3; `full_context` is
150/150).

**H2 is withdrawn — word recognition is a THIRD leak.** `wm_word_recognition.py`
builds its recall prompt with `trials_text` = every trial in order, byte-for-byte
the studied list, despite the prompt saying "Based ONLY on the above contents". So
Old/New is fully determined by visible text. Verified: **36 of 50 participants score
≥ 0.98 and 7 score ≤ 0.04** (humans: 1 of 53 above 0.98, mean 0.315). The 43 at
ceiling are reading the list off the prompt; the 7 are the only ones consulting the
store. A2's 0.340 is therefore produced by a handful of participants while most are
not doing the task at all, and the "A2 headroom" is not what it appeared to be.

So **three of eight tasks bypass the memory module** — nback, variable_mapping and
word_recognition. Closing the word-recognition leak is the highest-value target
remaining, and `recall()`'s context construction is inside the override surface.

## What you may change

`inject.py` defines the surface. A candidate is a `harness.py` overriding any of:

    WorkingMemoryAgent   the harness class: encode/recall/step, segmentation,
                         write/delete policy, key naming, value formatting,
                         recall-time context construction
    MAX_KEYS             capacity (see the constraint above)
    TOOLS                the tool schemas and their docstrings
    CONDITION_PROMPTS    the per-condition system prompts
    WM_SYSTEM_PROMPTS    the per-task prompt overrides

Out of scope: the base model, the task stimuli, the scorers, the human data.

## Workflow

1. Read prior candidates: `python meta_harness/history.py list`, then `show`,
   `diff`, `regressions`, and `trace` for the episodes behind a number.
   Read the actual traces — `encoding_log` shows what the harness chose to store
   and `recall_raw` how it reconstructed. That is where the mechanism is visible.
2. Form a hypothesis about a *specific* failure you can see in the traces.
3. Write `meta_harness/candidates/<id>/harness.py` with a `MANIFEST` stating the
   parent, what you changed, and why — including the psychological argument if
   you touched capacity or decay.
4. Check it offline, free: `python meta_harness/verify_interface.py <path>`.
5. List it in `meta_harness/logs/pending_eval.json` for evaluation.

## Known failure modes in this setup

Recorded so you do not rediscover them:

- Runs happen on one H200 via SLURM; `bench` must run **in-process** with the
  injection applied, which is what `run_candidate.py` does. Shelling out to
  `python -m bench.cli` would silently run the original harness.
- `--repeat` must be passed on the command line; YAML `run.repeat` is inert.
- vLLM needs `--enable-auto-tool-choice --tool-call-parser hermes`, because the
  compactor is a tool-calling agent.
