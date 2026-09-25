"""Candidate `chunk_limit`: the same 4 slots, but a slot may only hold a chunk.

Parent: `baseline` (NOT `displacement`, NOT `primacy`) -- see MANIFEST.md §0.

THE DEFECT
----------
`MAX_KEYS = 4` bounds the number of slots. Nothing bounds the information inside
a slot. Measured on the iteration-0 baseline run's `final_kv`, the single worst
value in `wm_word_recognition.jsonl` is the studied word list itself, stored
under one key:

    "recent_items": "camera, corner, forest, artist, corner, forest, artist,
                     artist, artist, message, thunder, artist, shadow, violin,
                     ... "                                        (99 items)

A 4-chunk store that admits a 99-item chunk is not Cowan's (2001) limit. The
slot count is cosmetic while a slot can hold an unbounded enumeration.

WHAT THIS CANDIDATE CHANGES
---------------------------
One thing, in one method body: `WorkingMemory.write_key` bounds the number of
ENUMERATED ELEMENTS a value may carry before delegating to the baseline's own
`write_key`. Everything else is the baseline's:

  * overflow policy, eviction order, primacy protection -- UNTOUCHED. Overflow
    is still the baseline's refuse-on-full. Iteration 2a (`primacy`) owns which
    entry leaves; this candidate owns what a slot may contain.
  * `recall()`'s context construction -- untouched (iteration 2b owns it).
  * `step()` / `reset_messages()` -- untouched (iteration 2c owns it).
  * `TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS`, `_tool_call_cap()`,
    `snapshot()`, `to_recall_text()` -- untouched. Zero bytes of prompt or tool
    text differ from the baseline.

THE BOUND, AND WHY IT HAS NO NEW CONSTANT
-----------------------------------------
A chunk in the Miller (1956) sense is a *recoded* unit; in Cowan's (2001) sense
the ~4-chunk limit holds for chunks whose internal structure is already
consolidated in long-term memory. Binding n NOVEL elements into one retrievable
unit is itself an operation in the focus of attention, and that binding is
subject to the same capacity limit -- ~4 elements (Halford, Wilson & Phillips
1998, relational complexity; Cowan 2001; Chase & Simon 1973 measure chess chunks
of 2-4 pieces in non-experts).

So the within-slot bound *is* `MAX_KEYS`. There is no new number in this file:

    a slot may hold at most MAX_KEYS enumerated elements
    => the whole store holds at most MAX_KEYS**2 = 16 novel elements

against the 44.9 elements (worst case 180) the baseline's word-recognition store
actually holds. `MAX_KEYS` is read live via `getattr(_wm_mod, "MAX_KEYS", ...)`
and is never frozen at import, so the identity survives injection.

WHAT COUNTS AS AN ELEMENT
-------------------------
An enumeration delimiter written by the model itself: comma, semicolon, newline,
or a spaced slash. `"1, 9, 4, 5"` is four elements; `"1945"` and `"1-9-4-5 as a
date"` are one. Prose of ANY length passes, because a consolidated description is
one unit however long it is -- what the rule forbids is the 99-item list, not
length. That is deliberate: the bound is on *enumeration*, which is the
observable difference between a chunk and a list, and it is the only bound
available without inventing a constant. Known limitation, stated rather than
hidden: an unbounded single-clause value still slips through (max observed 74
words on story recall, 111 on word recognition). See MANIFEST §2.

CAP, NOT LLM CONSOLIDATION
--------------------------
The surplus is dropped, not summarised by a second LLM call. Cost: zero extra
calls. The write is ALWAYS admitted -- no refusal is added, because iteration 1
established that refuse-to-encode is the defect, not the mechanism -- and the
tool result names what did not fit, exactly as `displacement` carried its
affordance in the result rather than in the tool description. The result is
descriptive only: it reports the loss and issues no strategy directive, so on a
task where the bound never bites the tool results are byte-identical to the
baseline's. MANIFEST §3 gives the five reasons LLM consolidation is worse here,
including that it is unreachable without overriding `step()`, which another
iteration owns.

WHERE IT CAN AND CANNOT BITE (correction to the brief -- see MANIFEST §1)
------------------------------------------------------------------------
The brief called this defect "task-general". Measured, it is not: it is one
extreme task, two moderate ones, and three that cannot move at all.

    task                elements/value (max)   elements/store (max)   can move?
    word_recognition       13.21  (99)            44.90  (180)        extreme
    semantic_story_recall   3.89   (9)            13.98   (23)        moderate
    narrative_qa            3.76   (9)            13.54   (25)        moderate
    digit_span_fwd/rev      1.94  (13)             6.75   (25)        weak
    nback                   1.13   (2)             2.45    (6)        NO
    variable_mapping        1.00   (1)             3.51    (4)        NO
    craft_task              1.00   (1)             3.33    (4)        NO
"""
from __future__ import annotations

import re
from typing import Any, Tuple

from bench.core import working_memory as _wm_mod
from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    SummarizerAgent,
    TOOLS,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS, WorkingMemory as _BaseMemory

# Enumeration delimiters, and nothing else. Deliberately NOT " and ", which is
# ubiquitous in ordinary prose -- splitting on it would cut grammatical
# sentences and would make the rule a length cap by the back door.
ENUM_DELIM = re.compile(r"\s*(?:,|;|\n|\r|\s/\s)\s*")


def count_elements(value: str) -> int:
    """Number of non-empty enumerated elements in *value*."""
    return sum(1 for p in ENUM_DELIM.split(str(value)) if p.strip())


def bound_value(value: str, cap: int) -> Tuple[str, int, int]:
    """Cut *value* to at most *cap* enumerated elements.

    Returns ``(bounded_value, n_before, n_after)``. The cut happens at an element
    boundary and preserves the original punctuation of what it keeps, so the
    retained text is exactly what the model wrote, truncated -- nothing is
    rewritten, reordered or paraphrased.
    """
    value = str(value)
    n = count_elements(value)
    if cap <= 0 or n <= cap:
        return value, n, n
    pos, seen = 0, 0
    for m in ENUM_DELIM.finditer(value):
        if value[pos:m.start()].strip():
            seen += 1
            if seen == cap:
                return value[:m.start()].rstrip().rstrip(",;/"), n, cap
        pos = m.end()
    # No delimiter follows the cap-th element (cannot happen when n > cap, but
    # the invariant must hold even so): leave the value alone rather than guess.
    return value, n, n


class ChunkBoundedMemory(_BaseMemory):
    """Baseline store with a bound on the information inside a slot.

    The slot count, the overflow rule and the read-out format are the
    baseline's, byte for byte. The only difference is that a value carrying more
    than ``MAX_KEYS`` enumerated elements is admitted with the surplus
    elements not encoded.
    """

    def __init__(self) -> None:
        super().__init__()
        # Bookkeeping for offline measurement and for the run record. It is
        # observational only -- nothing reads it back to make a decision.
        self.n_bounded = 0
        self.elements_dropped = 0

    def _capacity(self) -> int:
        # Read live so an injected MAX_KEYS is honoured; never frozen at import.
        return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))

    def write_key(self, key: str, value: str) -> str:
        cap = self._capacity()
        bounded, before, after = bound_value(value, cap)
        result = super().write_key(key, bounded)
        if result.startswith("Error"):
            # The baseline refused the write (store full, new key). Nothing was
            # encoded, so nothing was dropped and the refusal text is unchanged.
            return result
        if before > after:
            self.n_bounded += 1
            self.elements_dropped += before - after
            # Purely descriptive, exactly as `displacement`'s result was. It
            # names what happened and gives no strategy directive, so on a task
            # where the bound never bites (craft_task: 0 of 500 values) the tool
            # results are byte-identical to the baseline's.
            return (
                f"{result} Only the first {after} of {before} items fit in one "
                f"memory slot; the rest were not encoded."
            )
        return result


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with the chunk-bounded store installed.

    Nothing else is overridden: encode/recall/step, segmentation, key naming,
    overflow policy, eviction order, read-out format, the tool-call cap and
    every prompt are the baseline's, so exactly one method body separates this
    candidate from `baseline`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wm = ChunkBoundedMemory()


MANIFEST = {
    "id": "chunk_limit",
    "parent": "baseline",
    "capacity": MAX_KEYS,
    "decay": (
        "none. No stochastic loss of any kind, no random number drawn anywhere, "
        "`random` not imported. No decay, no eviction change: overflow is still "
        "the baseline's refuse-on-full. The only loss is deterministic: "
        "enumerated elements beyond MAX_KEYS in a single value are not encoded."
    ),
    "role": "contender -- iteration 2d, the within-slot information bound",
    "summary": (
        "MAX_KEYS = 4 bounds SLOTS; nothing bounds the information inside a slot. "
        "MEASURED on the iteration-0 baseline's final_kv: word_recognition holds "
        "13.2 enumerated elements per value (max 99, the studied word list stored "
        "under one key 'recent_items') and 44.9 elements per 4-slot store (max "
        "180); story recall 3.9/value and 14.0/store; narrative_qa 3.8 and 13.5; "
        "digit span 1.9/value (max 13) and 6.8/store (max 25); craft_task, "
        "variable_mapping and nback are already at 1.0-1.1 elements per value and "
        "CANNOT be moved by this rule. PSYCHOLOGICAL ARGUMENT: a chunk is a "
        "recoded unit (Miller 1956) whose internal structure is already "
        "consolidated (Cowan 2001); binding n NOVEL elements into one retrievable "
        "unit is itself an act in the focus of attention and is bounded by the "
        "same ~4 limit (Halford, Wilson & Phillips 1998; Chase & Simon 1973 "
        "measure chess chunks of 2-4 pieces in non-experts). So the within-slot "
        "bound IS MAX_KEYS -- there is no new numeric constant in the candidate, "
        "and the whole store is bounded at MAX_KEYS**2 = 16 novel elements. "
        "MECHANISM: cap, not LLM consolidation. A value carrying more than "
        "MAX_KEYS comma/semicolon/newline-delimited elements is admitted with the "
        "surplus not encoded; the write is never refused (iteration 1 established "
        "refuse-to-encode is the defect) and the tool result names the loss, so "
        "the affordance arrives only after the bound has actually bitten -- zero "
        "bytes of prompt or tool text differ from the baseline. Zero extra LLM "
        "calls. Prose of any length passes: the bound is on ENUMERATION, which is "
        "the observable difference between a chunk and a list. CONSOLIDATION "
        "REJECTED because it needs a second frozen-model behaviour on the "
        "injectable surface (unattributable delta), is unreachable without "
        "overriding step() which iteration 2c owns, cannot be verified offline, "
        "and the model was ALREADY instructed to write 'an abstractive summary' "
        "in both the tool description and the system prompt and wrote a 99-item "
        "list anyway. FALSIFIABLE, and measured by replaying the real write "
        "sequences through the real class: word_recognition elements/store 44.9 "
        "-> 8.1, elements/value 13.2 -> 2.0, max/value 99 -> 4. Digit span IS "
        "affected (best_span should fall from 18.4) so it is NOT the control; "
        "craft_task, variable_mapping and nback are. The test that separates a "
        "chunk bound from information vandalism is A1: destroying information at "
        "random raises sub_span_leak (random_decay_v2 went 0.130 -> 0.249) "
        "whereas enforcing a bound should lower best_span with the leak FLAT."
    ),
}
