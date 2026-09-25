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
than humans (0.699 vs 0.866), so it opposes every other task's gradient. A single
"forget harder" knob improves 8 tasks and breaks N-Back. Whether one mechanism
can satisfy both is an open question and is arguably the most interesting result
available here — do not paper over it.

## The thing that will tempt you, and why it fails

You can match a score distribution trivially: draw a random per-participant
"ability" and drop that fraction of what the harness stores. The spread widens
until it overlaps the human spread and humanlikeness goes up. **This is worth
nothing** — it is noise, not a memory model.

`candidates/random_decay/` already does exactly that, deliberately, as a control.
If your proposal amounts to the same trick with extra steps, it will be caught by
the axes below and rejected. Propose mechanisms, not noise.

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

**Every task must keep working.** The per-task floor is 0.03: no search task may
regress more than that below the baseline. The mean alone is not the objective,
because Variable Mapping holds 0.569 of the 2.25 total headroom and would
otherwise let you buy the entire headline from one task.

## Axes and guards, with the numbers

| | human | baseline (qwen3-30b) | role |
|---|---|---|---|
| **A2** word-recognition miss/false-alarm ratio | **6.09** | 0.018 | **the axis** |
| A1 protocol-matched digit-span sub-span leak | 0.087 | 0.105 | guard, band [0.05, 0.12] |
| A3 story-recall BLEU @ recall words | 0.002 @ 137 | 0.003 @ 128 | guard, BLEU < 0.02, words in [100, 175] |

**A2 is where the headroom is.** Humans are conservative on recognition — they
answer "new" when unsure, giving miss 0.272 against false-alarm 0.045. Every one
of the eight baselined models does the opposite, false-alarming heavily; the
closest is still 4.6x off. A harness that makes the model appropriately reluctant
to claim recognition is the single most valuable thing you can find.

A2 has a trap: word recognition stops after 3 strikes, so trials-attempted
varies (34.5 for humans, 87.5 for the baseline). You can move the ratio by
surviving longer rather than by fixing the asymmetry. Trials-attempted is
recorded as a covariate and an A2 gain that came from it will not be credited.

A1 and A3 are already near-human on this substrate, so they offer no gradient
and exist to catch regressions. Note what A1 catches: qwen3-8b reaches a
near-human ceiling (8.28 vs 6.88) while leaking twice as much below it (0.165 vs
0.087) — the signature of stochastic dropping.

## Noise floor — do not chase what you cannot measure

Split-half W_1 within the human participants, per task:

    digit_span_forward 0.036   digit_span_reverse 0.026   nback 0.025
    word_recognition   0.075   variable_mapping   0.041   factual_qa 0.061
    narrative_qa       0.053   semantic_story_recall 0.042
    map_task           0.054   craft_task         0.041

Mean attainable ceiling is 0.955, not 1.0. **A single-task delta under ~0.05 is
noise.** `history.py diff` marks these. Do not build a story on them.

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
