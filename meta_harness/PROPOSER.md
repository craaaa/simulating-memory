# Proposer instructions

You are proposing candidate harnesses for a Meta-Harness search over the
compactor: the 4-slot key-value working-memory agent that wraps a frozen LLM in
this repo. Read this before proposing anything.

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
| **bottlenecked** | `encode()` → `recall()`, KV only | ds_fwd .886, ds_rev .967, word_rec .495, story .947, craft .891, narr .957 | 0.854 |
| **leaky** | answers via `step()`, study history still in context | nback .791, variable_mapping .355 | 0.573 |

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

**A2 is where the headroom is.** Humans are conservative on recognition — they
answer "new" when unsure, giving miss 0.272 against false-alarm 0.045. Every one
of the eight baselined models does the opposite, false-alarming heavily; the
closest is still 4.6x off. A harness that makes the model appropriately reluctant
to claim recognition is the single most valuable thing you can find.

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

**H1 — N-Back fails episodically, not gradually. This is the anti-correlated task,
so a real fix here is a structural result rather than a knob.**

    model  mean 0.710  sd 0.321   35% of participants at exactly 1.0, min 0.071
    human  mean 0.866  sd 0.081   tightly clustered, 0.69 to 1.00

The model is not uniformly worse than humans; it is bimodal. It either aces N-Back
or collapses completely, while humans are consistently good. `full_context` narrows
sd to 0.125 and mean to 0.863 — essentially human — which says the collapses are a
resource failure, not a strategy failure. N-Back is also **leaky** (it answers via
`step()`), so the mechanism is not simply store capacity. Go find the collapsed
episodes in the traces and read what the harness did on those specific trials.
Note the tension: N-Back wants *more* retained, digit span and story recall want
the bottleneck. One global `MAX_KEYS` provably cannot serve both — `full_context`
takes N-Back +0.157 while destroying five other tasks. A demand-sensitive mechanism
(interference, displacement, retrieval competition) rather than a hard slot count
is the obvious candidate and has a literature behind it.

**H2 — Word recognition has the wrong error asymmetry, and it is the safest place
for a legitimate gain.** Humans are conservative: they answer "new" when unsure,
giving miss 0.272 against false-alarm 0.045, a ratio of 6.09. The baseline is at
0.340 — it barely misses anything and false-alarms instead. It is also the *only*
axis with real headroom, the task is bottlenecked so the store is genuinely on the
causal path, and humanlikeness there is 0.495 with plenty of room. Beware the
covariate: trials-attempted is 82.9 for the baseline against 34.5 for humans, and
you can move the ratio by surviving longer instead of by fixing the asymmetry.

Prefer H1 if you want the interesting result, H2 if you want the reliable one. Say
which you are doing and why.

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
