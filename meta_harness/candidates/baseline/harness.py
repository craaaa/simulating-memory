"""Candidate 0: the released compactor, unmodified.

This is the number every other candidate has to beat, and it doubles as an
end-to-end test of the injection path: if the baseline scores differently from
the released run, the harness plumbing is wrong, not the harness.

It overrides the surface names with the originals, so `inject.apply()` has
something to rebind and the report shows the same shape as a real candidate's.
"""
from __future__ import annotations

from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    TOOLS,
    SummarizerAgent,
    WorkingMemoryAgent,
)
from bench.core.working_memory import MAX_KEYS  # noqa: F401

MANIFEST = {
    "id": "baseline",
    "parent": None,
    "capacity": MAX_KEYS,
    "decay": None,
    "summary": "Released compactor, unmodified. Control and plumbing test.",
}
