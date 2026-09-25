"""Adversary v2: gently calibrated random decay. NOT a contender.

v1 (`random_decay`, uniform(0, 0.9)) FAILED as an adversary: it dropped
humanlikeness from the baseline's 0.7861 to 0.6977 and collapsed digit-span best
span from 18.4 to 2.0 against a human 6.88. It did not match the human
distribution, it overshot it, so it tested nothing. Per the spec's own rule, a
weak adversary leaves the axes unvalidated rather than validated.

This is the fair version of the attack. The adversary's job is to be the
*strongest* trivial threat -- if even a well-calibrated noise injector cannot
match the human score distribution while leaving the error-structure axes wrong,
that is a real result in the metric's favour. If it *can*, the Pareto setup is
invalid and must be redesigned before candidates are spent on it.

Calibrating the rate downward is legitimate here precisely because this is an
attack, not a proposal. It must never inform a real candidate, and nothing in it
is a psychological claim.

Range chosen from v1's overshoot: v1 at up to 0.9 per-key drop per call, applied
at both encode and recall, destroyed retention. v2 drops at most 0.35 and only
once per episode, so scores should land between the baseline (too good) and v1
(far too bad) -- which is the region the human distribution occupies.
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

DECAY_MIN = 0.0
DECAY_MAX = 0.35


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness plus one gentle, per-participant retention failure.

    Differences from v1, both deliberate:
      * max drop probability 0.35 rather than 0.9
      * decay applied ONCE after encoding, not again at recall, so a key that
        survived encoding is not rolled for a second time
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Content-hash seeded, as in v1: deterministic across runs (an
        # address-derived seed would make the control irreproducible) while still
        # varying per participant, because participants get different stimuli.
        self._rng: random.Random | None = None
        self._decay_rate = 0.0
        self._decayed = False

    def _seed_from(self, content: Any) -> None:
        if self._rng is not None:
            return
        digest = hashlib.sha256(repr(content).encode("utf-8")).hexdigest()
        self._rng = random.Random(int(digest[:16], 16))
        self._decay_rate = self._rng.uniform(DECAY_MIN, DECAY_MAX)

    def encode(self, content: Any) -> Dict[str, Any]:
        self._seed_from(content)
        log = super().encode(content)
        if not self._decayed and self._rng is not None:
            for key in list(self.wm.store):
                if self._rng.random() < self._decay_rate:
                    self.wm.clear_key(key)
            self._decayed = True
        log["final_kv"] = self.wm.store
        log["decay_rate"] = self._decay_rate
        return log


MANIFEST = {
    "id": "random_decay_v2",
    "parent": "random_decay",
    "capacity": MAX_KEYS,
    "decay": f"per-participant uniform({DECAY_MIN}, {DECAY_MAX}) key drop, once after encode",
    "role": "adversary / validity control -- not a contender",
    "summary": (
        "Calibrated retry of the v1 adversary, which overshot the human "
        "distribution so badly it tested nothing. Pass condition: mean "
        "humanlikeness >= baseline + 0.05 while A2 distance stays >= 2x the "
        "baseline's. Passing both would invalidate the axes; failing the first "
        "again would be evidence that matching the human distribution with pure "
        "noise is harder than assumed."
    ),
}
