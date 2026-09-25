"""Control: unlimited, verbatim memory. NOT a contender.

The spec's third seed baseline. Where `random_decay` bounds the degraded end,
this bounds the other: the stimulus is kept in full, so nothing is ever
compressed or forgotten. It answers a question the search otherwise cannot —
**how much of the current humanlikeness is the 4-slot memory module responsible
for, versus the base model's own limits?**

Expected to be the *least* humanlike candidate, because the compactor's
over-performance relative to humans should get worse when the capacity limit is
removed. If it is not much worse, that is a genuinely important negative result:
it would mean the 4-slot bottleneck is not doing the work the paper attributes to
it, and the search should target the prompts and recall construction instead.

Implementation note: capacity is raised rather than the class being rewritten, so
this stays a minimal, honest diff against the baseline. `MAX_KEYS` is exported at
a high value and `WorkingMemory` enforces the limit at write time, so the agent
simply never hits it.
"""
from __future__ import annotations

from typing import Any, Dict

from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    TOOLS,
    SummarizerAgent,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent

# High enough that the agent never hits it, which is the point of the control.
# This is NOT a psychological claim -- it is the "no bottleneck" anchor, and it
# is why this candidate is a control rather than a proposal.
MAX_KEYS = 10_000
NUM_SLOTS = MAX_KEYS


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with the capacity bottleneck removed.

    Also stores the raw stimulus verbatim, so recall is not limited by whatever
    the agent chose to write during encoding. Between the raised capacity and the
    verbatim copy, this is as close to perfect retention as the harness allows.
    """

    VERBATIM_KEY = "__full_stimulus__"

    def encode(self, content: Any) -> Dict[str, Any]:
        log = super().encode(content)
        text = content if isinstance(content, str) else "\n".join(map(str, content))
        self.wm.write_key(self.VERBATIM_KEY, text)
        log["final_kv"] = self.wm.store
        return log


MANIFEST = {
    "id": "full_context",
    "parent": "baseline",
    "capacity": MAX_KEYS,
    "decay": None,
    "role": "control / upper anchor -- not a contender",
    "summary": (
        "Removes the 4-slot bottleneck and keeps the stimulus verbatim. Isolates "
        "how much of current humanlikeness comes from the memory module rather "
        "than the base model. Expected to be the least humanlike candidate; if it "
        "is not, the bottleneck is not doing the work it is credited with."
    ),
    "not_a_proposal": (
        "MAX_KEYS=10000 is an anchor, not a psychological claim, and must never be "
        "read as a search result."
    ),
}
