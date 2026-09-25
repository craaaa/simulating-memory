"""Adversary candidate: per-participant random decay. NOT a contender.

This exists to test whether the error-structure axes have teeth. It is the
cheapest way to match a human score *distribution* while learning nothing about
memory: give each simulated participant a random "ability" by dropping a random
fraction of what the harness stores, and the spread of scores widens until it
overlaps the human spread.

Pass condition, from domain_spec.md -- the control must show BOTH of:

  * mean humanlikeness >= baseline + 0.05   (it can match the distribution)
  * A2 distance >= 2x baseline's            (while its error structure is wrong)

If it fails the first, it is too weak an adversary and the axes are untested
rather than validated. If it passes the first AND matches the axes, the axes do
not discriminate and the whole Pareto setup is invalid -- in that case stop and
redesign the axes before spending candidates.

Deliberately transparent about what it is: the decay is pure noise with no
psychological content, which is exactly the property the axes must catch.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict

from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    TOOLS,
    SummarizerAgent,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS

# Drop probability is drawn per participant from this range, so the population
# shows a spread of "abilities" rather than one uniform degradation.
DECAY_MIN = 0.0
DECAY_MAX = 0.9


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness plus a per-instance random retention failure.

    The rate is seeded from the instance identity so a single participant is
    internally consistent within a trial while the population still varies.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seeded lazily from the stimulus content, NOT from id(self): an
        # address-derived seed would make this control irreproducible across
        # runs, which is unacceptable for something whose whole job is to be a
        # trustworthy negative result.  Content hashing is deterministic and
        # still varies per participant, because participants get different
        # stimuli.
        self._rng: random.Random | None = None
        self._decay_rate = 0.0

    def _seed_from(self, content: Any) -> None:
        if self._rng is not None:
            return
        digest = hashlib.sha256(repr(content).encode("utf-8")).hexdigest()
        self._rng = random.Random(int(digest[:16], 16))
        self._decay_rate = self._rng.uniform(DECAY_MIN, DECAY_MAX)

    def _apply_decay(self) -> None:
        """Drop each stored key independently with probability _decay_rate."""
        if self._rng is None:  # recall before encode: nothing was seeded
            return
        for key in list(self.wm.store):
            if self._rng.random() < self._decay_rate:
                self.wm.clear_key(key)

    def encode(self, content: Any) -> Dict[str, Any]:
        self._seed_from(content)
        log = super().encode(content)
        self._apply_decay()
        # Keep the log honest: report post-decay state, since that is what
        # recall will actually see.
        log["final_kv"] = self.wm.store
        log["decay_rate"] = self._decay_rate
        return log

    def recall(self, recall_prompt: str | None = None, max_tokens: int = 512) -> str:
        self._apply_decay()
        return super().recall(recall_prompt, max_tokens=max_tokens)


MANIFEST = {
    "id": "random_decay",
    "parent": "baseline",
    "capacity": MAX_KEYS,
    "decay": f"per-participant uniform({DECAY_MIN}, {DECAY_MAX}) key drop",
    "role": "adversary / validity control -- not a contender",
    "summary": (
        "Matches the human score distribution by injecting per-participant "
        "noise with no psychological content. Must score well on humanlikeness "
        "and badly on the error-structure axes, or the axes are not doing their "
        "job."
    ),
}
