"""Adversary v3: decay over TIME, seeded from the participant. NOT a contender.

Why a v3 at all. An audit of v1 (`random_decay`) and v2 (`random_decay_v2`) found the
adversary broken in two independent ways, both verified by instrumented runs under a
stub LLM:

  DEFECT 1 -- hook coverage. v1 and v2 override `encode()` (v1 also `recall()`).
  Three of the eight search tasks never call either: they drive `step()` repeatedly.
  Measured step / encode / recall counts per unit: nback 96 / 0 / 0,
  word_recognition 12 / 0 / 0, variable_mapping 40 / 0 / 0. The decay was a complete
  NO-OP on those three tasks, so an "adversary" that the axes were validated against
  was in fact the baseline on 3 of 8 tasks.

  DEFECT 2 -- seeding. `_seed_from()` hashes the STIMULUS CONTENT, with a comment
  asserting "participants get different stimuli". That is false for three tasks.
  Measured distinct decay rates against rows: digit_span_forward 190 of 190,
  narrative_qa 10 of 50, semantic_story_recall 4 of 200, craft_task 3 of 150 -- and
  all 150 craft rows ended with an empty store, i.e. 3 high rates rather than a
  population. A per-participant "ability" spread was the whole mechanism, and on those
  tasks there was no spread.

WHAT CHANGED IN MEANING, not just in plumbing
---------------------------------------------
1. `step()` only, rolled once per TOOL-ENABLED turn. This is decay over *time*: a key
   is exposed to loss on every turn it survives. v1/v2 decay at *encode* -- a single
   lossy write. These are different mechanisms, not the same mechanism relocated.
   Consequences, all deliberate:
     * The 5 batch tasks call `encode()`, which calls `step()` exactly once with
       `allow_tools=True`. So they get exactly ONE roll after the writes land --
       identical in effect to v2. The batch tasks are the comparable part of v3.
     * The 3 turn-based tasks get one roll per tool-enabled turn: 15-17 for nback
       (14+n letters), 20 for variable_mapping (20 questions x 1 encode turn), and up
       to 100 for word_recognition (1 encode turn per word, cut short by the human
       3-error stop). These were the no-ops. There is no v2 behaviour here to hold
       comparable, because v2 had none.
2. No `recall()` roll. v1 rolled at BOTH encode and recall, so on a batch task a key
   that survived encoding was rolled a second time. Dropping the recall roll halves
   batch-task severity and aligns v1's plumbing with the move v2 already made.
3. Rolls happen only where `allow_tools=True`. An answer turn is a turn on which the
   model cannot touch the store at all, and the answer prompt's `wm_contents` is
   formatted by the CALLER before `step()` is entered -- verified in
   `bench/tasks/wm_nback.py` (ANSWER_PROMPT.format(wm_contents=agent.wm.to_recall_text())
   passed with allow_tools=False) and `bench/tasks/wm_word_recognition.py` (same shape).
   So a roll inside `step()` could not affect that turn's own prompt even if it fired.
   Rolling there would only double the per-item rate for no mechanism.
4. The rate is drawn from `participant_id`, the scored-unit key threaded through
   `bench/tasks/` (S1), not from content. One rate per scored unit, by construction.

The roll runs AFTER `super().step()` returns, never before. On a batch task the store
starts EMPTY and is filled inside that turn's tool loop, so a roll at the top of
`step()` would see `{}` and reintroduce Defect 1 in a new disguise.

DECAY_MAX -- a calibration HYPOTHESIS, not a calibrated value
------------------------------------------------------------
Hypothesis: 0.35, v2's value, is the right per-turn rate because it holds the batch
tasks numerically identical to the one arm that was ever calibrated against the human
score distribution, and the three turn-based tasks have no prior value to preserve.

Two survival numbers, which must not be conflated. p is the per-key per-turn drop
probability (a proportion); R is the tool-enabled roll count (an integer count of turns):

  (1-p)^R   survival of a key that is NEVER refreshed -- the worst case
  1/p turns expected retention horizon under per-turn refresh -- the regime the
            encode prompts actually produce, since each turn asks the model to store
            the current item

At p = DECAY_MAX = 0.35, horizon 1/p = 2.9 turns, per-turn survival 0.65:

  task                    R (tool-enabled rolls)   (1-0.35)^R
  digit_span_forward       1                        0.650
  digit_span_reverse       1                        0.650
  narrative_qa             1                        0.650
  craft_task               1                        0.650
  semantic_story_recall    1                        0.650
  nback                    15 / 16 / 17 (n=1/2/3)   0.0013 / 0.0009 / 0.0006
  variable_mapping         20                       0.0002
  word_recognition         up to 100 (human mean
                           survival 34.49 trials)   ~0 at either end

The turn-based column reads as annihilation and would be misleading alone: the store is
rewritten every turn, so the operative quantity is the 2.9-turn horizon -- the store
becomes a short recency window rather than emptying out. What the run measures is
whether that window is too short.

REJECTION CRITERION, stated before the run. Two of the three newly-live tasks (nback,
variable_mapping) are LEAKY per `score_candidate.py`: the model answers from the
current stimulus and no store change moves the score -- random decay, capacity 4 and
capacity 10 000 all produced 0.992 on variable_mapping. So overshoot there should be
invisible. word_recognition is BOTTLENECKED and carries A2. If its survival length
collapses below the human minimum (4 of 100 trials) for most scored units, 0.35 is too
high for the per-turn regime and the hypothesis is rejected: a v4 would lower
DECAY_MAX or make the total expected loss, rather than the per-turn rate, the constant.
Passing the adversary's own pass condition (mean humanlikeness >= baseline + 0.05 while
A2 distance stays >= 2x the baseline's) cannot be checked offline; it needs the GPU run.

Calibrating an attack downward is legitimate, as v2 argued. Nothing here is a
psychological claim and none of it may inform a real candidate.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    TOOLS,
    SummarizerAgent,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS

# Per-key drop probability per tool-enabled turn, drawn once per scored unit from
# uniform(DECAY_MIN, DECAY_MAX). A proportion, not a count.
DECAY_MIN = 0.0
DECAY_MAX = 0.35

# Seed material used when no participant_id was supplied. Every construction site in
# `bench/tasks/` passes one; the only caller that does not is
# `meta_harness/verify_interface.py`, whose single synthetic round trip must still run.
# Such an agent draws a rate normally, from this fixed sentinel, so it is deterministic
# and non-zero -- but every such agent shares one rate, which is why
# `seeded_from_default` is recorded rather than left implicit.
_DEFAULT_SEED_KEY = "<no participant_id>"


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness plus per-turn random retention failure, per scored unit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        key = self.participant_id
        self.seeded_from_default = key is None
        digest = hashlib.sha256(
            (_DEFAULT_SEED_KEY if key is None else str(key)).encode("utf-8")
        ).hexdigest()
        self._rng = random.Random(int(digest[:16], 16))
        self._decay_rate = self._rng.uniform(DECAY_MIN, DECAY_MAX)
        # Observational only; nothing reads these back to make a decision. They exist
        # so `meta_harness/verify_decay_coverage.py` can prove the decay fired per step
        # on every task rather than asserting it.
        self.decay_rolls = 0
        self.decay_rolls_that_dropped = 0
        self.decay_keys_dropped = 0

    # ------------------------------------------------------------------
    def _apply_decay(self) -> int:
        """Drop each stored key independently with probability _decay_rate."""
        dropped = 0
        for key in list(self.wm.store):
            if self._rng.random() < self._decay_rate:
                self.wm.clear_key(key)
                dropped += 1
        self.decay_rolls += 1
        if dropped:
            self.decay_rolls_that_dropped += 1
            self.decay_keys_dropped += dropped
        return dropped

    def step(
        self, user_message: str, *, allow_tools: bool = True, max_tokens: int = 1024
    ) -> str:
        out = super().step(user_message, allow_tools=allow_tools, max_tokens=max_tokens)
        # AFTER the tool loop, so the writes this turn made are exposed to loss. Before
        # it, a batch task's store is still empty and the roll would be a no-op.
        if allow_tools:
            self._apply_decay()
        return out


MANIFEST = {
    "id": "random_decay_v3",
    "parent": "random_decay_v2",
    "capacity": MAX_KEYS,
    "decay": (
        f"uniform({DECAY_MIN}, {DECAY_MAX}) per-key drop probability, drawn once per "
        f"scored unit from participant_id, rolled once per tool-enabled step()"
    ),
    "role": "adversary / validity control -- not a contender",
    "mechanism_change_vs_v2": (
        "decay over TIME rather than at encode: one roll per tool-enabled turn instead "
        "of one roll per episode. Identical in effect to v2 on the 5 batch tasks (which "
        "call encode() once, hence one tool-enabled step()); newly live on nback (15-17 "
        "rolls), variable_mapping (20) and word_recognition (up to 100), where v1 and v2 "
        "were measured no-ops. v1's recall() roll is gone, halving batch severity and "
        "matching the move v2 already made."
    ),
    "seeding_change_vs_v2": (
        "rate drawn from the scored-unit participant_id threaded through bench/tasks/, "
        "not from a stimulus-content hash. Content is shared across participants on "
        "three tasks: 4 distinct contents over 200 semantic_story_recall rows, 3 over "
        "150 craft_task rows, 10 over 50 narrative_qa rows."
    ),
    "decay_max_rationale": (
        "0.35 is v2's value, kept so the batch tasks stay numerically identical to the "
        "only arm ever calibrated against the human score distribution. Stated as a "
        "hypothesis, not a calibration: at p=0.35 the per-turn refresh horizon is 1/p = "
        "2.9 turns, and the never-refreshed survival (1-p)^R is 0.650 at R=1 but 0.0006 "
        "at R=17 and ~0 at R=100. Rejected if word_recognition survival length falls "
        "below the human minimum of 4 of 100 trials for most scored units."
    ),
    "summary": (
        "Third attempt at the validity control. v1 overshot the human distribution and "
        "v2 was calibrated for it, but an audit found both were no-ops on 3 of the 8 "
        "search tasks and drew as few as 3 distinct rates across 150 rows on another. "
        "v3 fixes coverage (step() hook) and seeding (participant_id). Pass condition "
        "unchanged: mean humanlikeness >= baseline + 0.05 while A2 distance stays >= 2x "
        "the baseline's. Passing both would invalidate the axes."
    ),
}
