"""Candidate `displacement`: the same 4 slots, a different overflow rule.

WHAT IS WRONG WITH THE BASELINE
-------------------------------
`WorkingMemory.write_key` implements the capacity limit as a **refusal**:

    if key not in self._store and len(self._store) >= MAX_KEYS:
        return "Error: memory is full (4 keys). Delete an existing key first
                or overwrite one."

No account of human working memory contains a refuse-to-encode state. In Cowan
(2001) the focus of attention holds ~4 chunks and a new chunk *enters* it,
pushing the least active one out; in interference accounts (Oberauer & Kliegl
2006; Oberauer et al. 2016) overflow is overwriting, not rejection. Refusal is a
data-structure reading of a capacity limit, and it produces a failure mode humans
do not show: under load humans **respond and are wrong**, they do not fall silent.
The baseline falls silent.

The cost is measurable in the wave-0 traces, and it is arithmetic rather than
interpretive.

1. n-back, per n level (`meta_harness/runs/iter0/baseline/.../wm_nback.jsonl`):

       n   slot_utilization   answered / 14   acc_over_answered   acc_over_14
       1        0.25              13.98             0.994            0.993
       2        0.385             13.24             0.821            0.779
       3        0.99               6.82             0.737            0.360

   Accuracy *on the trials it answers* barely moves with n. What collapses is
   whether it answers at all, and it collapses exactly where the store saturates.
   `full_context` (capacity 10 000) answers **14/14 at every level** and its
   acc_over_answered at n=3 is 0.770 against the baseline's 0.737 -- i.e. its
   entire +0.157 n-back gain is response production, not better judgement.

2. The n=3 key sets are the fingerprint of manual eviction. 31 of 50 blocks end
   holding exactly `{position_2, position_3, position_8, position_10}`: four keys,
   **non-contiguous, high-index**. With four slots occupied, creating
   `position_8` is impossible without a preceding `delete_key`, so the agent is
   paying two tool calls per letter. `_tool_call_cap()` is
   `max(6, int(interactions * 1.5))` -- one and a half per letter-step. Two
   exceeds one and a half, so the budget goes to zero and stays there.

3. Demonstrated offline, not inferred. `test_offline.py` beside this file runs a
   scripted stub LLM -- one positional write per letter, delete+write repair on
   refusal -- through 16 letter-turns of both harnesses, free and with no GPU:

       baseline       budget pinned at 0 from turn 5, cap_hit on 12/16 turns,
                      repairs truncated by `tool_calls[:remaining]` so the delete
                      lands and the replacement write is dropped,
                      final_kv = {position_1: C, position_4: C, position_15: A}
       displacement   cap_hit on 0/16 turns, budget grows monotonically,
                      final_kv = {position_13: W, position_14: Z,
                                  position_15: A, position_16: B}

   The baseline's store decays into a stale, holey set -- which is what the real
   run shows. Only **4 of 198** n=3 KV values still contain a bare capital
   letter; the rest have degenerated into contentless position labels
   ("position_3": "seventeenth letter in sequence"). At n=3 the store holds no
   letters at all. Under displacement it holds the last four, contiguous, which
   is exactly what a 3-back window needs.

   Contrast n=1, where all 50 blocks converge on one key, `previous_letter`,
   overwritten each turn -- never full, never refused, 0.993 accuracy. The
   information for 3-back *fits* in four slots. The overflow rule is what makes
   it unreachable.

WHAT THIS CANDIDATE CHANGES
---------------------------
One thing: overflow. A write to a new key on a full store **displaces the least
recently refreshed entry** and is admitted, and the tool result names what was
lost. `MAX_KEYS` stays at 4 -- this is the same capacity with psychologically
correct overflow semantics, not more capacity. At most four keys are ever held,
which `verify_interface.py` checks.

Two supporting details, both consequences of the same mechanism rather than
separate knobs:

  * **Rehearsal protects.** Displacement is least-recently-*refreshed*, and an
    overwrite counts as a refresh. An item the agent keeps touching resists
    displacement, which is what rehearsal does (Baddeley's articulatory loop;
    time-based resource sharing, Barrouillet & Camos 2007). Pure FIFO would make
    the oldest slot unconditionally doomed and would leave the agent no policy.
  * **Read-out is recency-ordered.** The store renders least-recent to
    most-recent with the most recent marked. Insertion order in a Python dict is
    an implementation accident; an activation gradient is the thing being
    modelled, and serial-order information in working memory is recency-coded.

**Not one word of prompt or tool text differs from the baseline.** `TOOLS`,
`CONDITION_PROMPTS` and `WM_SYSTEM_PROMPTS` are all left alone, so exactly one
method body differs between this candidate and `baseline` and any change is
attributable to the overflow rule rather than to a rewritten affordance. An
earlier draft rewrote the `write_memory` description to announce displacement;
that was dropped, because the description is in context from turn 1 on every task
and "writing when full always succeeds" would remove the compression pressure on
the bottlenecked tasks *before any overflow occurs* -- a strategy change wearing
an overflow change's clothes. The affordance is carried instead by the tool
*result*, which the agent only sees once it has actually overflowed: "Memory was
full, so the least recently used entry 'X' was displaced and is now lost."

Reachability is not in doubt. The system prompt already says "Once your slots are
filled, accept that the rest will be lost", and the agent writes past capacity
anyway -- 203 refusals across 200 story-recall rows, 58 across 50
word-recognition rows, 52 across 50 narrative-QA rows. It will meet displacement
whether or not it is told about it.

`_tool_call_cap()` is also left untouched, and that is the point: displacement
lowers tool-call *demand* from two calls per letter to one. It does not raise the
supply. A gain here cannot be explained by "you gave it more compute".

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
Primary, and the falsifiable one. On n-back at n=3:

  * `answered` rises from 6.82 to >= 12 of 14, while `acc_over_answered` stays
    within +-0.05 of 0.737. nback humanlikeness 0.791 -> 0.88-0.95 (floor 0.060).
  * `slot_utilization` at n=3 stays ~1.0. Displacement is not extra capacity.
  * The share of n=3 KV values containing a bare letter rises from 4/198 to a
    majority, and key sets become contiguous.
  * **Phase compliance.** `model_parsed_buffer` at n=3 is the sharper test,
    because `answered` can rise for dull reasons and this cannot. The baseline
    answers *during the buffer period*, when it is instructed to say "no
    response": 50 `No response` against 43 `Same` and 57 `Different` out of 150
    buffer slots. `full_context` is 150/150 `No response`. The baseline has lost
    track of where it is in the sequence -- exactly what a holey, stale store
    ({position_1, position_4, position_15}) predicts and what a contiguous one
    ({position_13..16}) repairs. Prediction: n=3 buffer `No response` rises from
    50/150 well past 100/150.

  DISCONFIRMED IF: `answered` does not rise, or it rises while
  `acc_over_answered` falls below ~0.65. The first kills the refusal diagnosis
  outright; the second says the agent was silent because it could not do the task,
  and the store was never the issue.

Secondary:

  * `word_recognition` and A2 move by less than the 0.121 noise floor. This is a
    second, independent test -- of a leak diagnosis rather than of displacement.

    A THIRD LEAK, not in PROPOSER.md, which lists word_recognition as
    bottlenecked with "the store genuinely on the causal path". It is not.
    `bench/tasks/wm_word_recognition.py` builds its recall prompt as
    `RECALL_PROMPT.format(trials_text=..., ...)` where `trials_text` is every
    trial in order -- byte-for-byte the studied list. Old/New is therefore fully
    determined by text visible in the recall prompt, despite that prompt saying
    "Based ONLY on the above contents". Evidence: the baseline gets 2094 of 2099
    Old trials right (miss rate 0.002) while holding four keys; that is not
    possible from a 4-slot store. The score distribution is the giveaway --
    34 of 50 participants score >= 98/100 (they read the list off the prompt) and
    7 score <= 4. The 7 are the only ones that actually consult the store, and
    they collapse the same way: one packs 30+ words into a single key called
    "repeated" and then answers "old" on every first occurrence -- familiarity
    without recollection, the right mechanism pointed the wrong way, since humans
    are conservative (miss 0.272 vs FA 0.045). So A2's 0.340 is produced by 7 of
    50 participants and the other 43 are not doing the task at all.

    If word_recognition moves substantially under a change that only touches
    overflow semantics, this reading is wrong. If it does not move, closing that
    leak is the highest-value target left and should be iteration 2.
  * `digit_span_forward/reverse` essentially unchanged: only 6 of 190 rows ever
    hit the refusal, and the agent writes at most 4 keys. `best_span` stays near
    18.4. This candidate does not address the 18.4-vs-6.88 staircase problem.
  * `semantic_story_recall` is the exposed task -- 136 of 200 rows hit the
    refusal, so displacement will bite. Under refusal the kept chunks are the
    *first* four ("beginning", "first encounter"); under displacement they shift
    toward the last four. Human free recall has both primacy and recency, so a
    regression here is expected to be small but real, and is informative:
    **if story recall regresses, the missing ingredient is primacy protection
    (rehearsal of the earliest chunks), and that is iteration 2.**
  * `craft_task` (50/150 rows) and `narrative_qa` (36/50 rows) shift for the same
    reason; both are MCQ over gist, where recency-kept is not obviously worse
    than primacy-kept, so expect within-floor movement in either direction.
  * `variable_mapping` unchanged (0.992 regardless of the store -- it is leaky),
    so A4 owes nothing.
  * Mean humanlikeness: +0.02 is the honest expectation, which is **inside** the
    0.026 min credible mean delta. The primary claim is therefore the per-task
    n-back delta and the mechanism predictions above, not the headline mean.
"""
from __future__ import annotations

from typing import Any, List

from bench.core import working_memory as _wm_mod
from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    SummarizerAgent,
    TOOLS,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS, WorkingMemory as _BaseMemory


class DisplacementMemory(_BaseMemory):
    """Capacity-limited store whose overflow rule is displacement, not refusal.

    Invariant: at most ``MAX_KEYS`` entries at any time -- identical to the
    baseline. What differs is which write fails. In the baseline the *new* item
    fails to enter; here the *least recently refreshed* item leaves.
    """

    def __init__(self) -> None:
        super().__init__()
        self._order: List[str] = []   # least recently refreshed first

    # -- activation bookkeeping ------------------------------------------
    def _refresh(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _capacity(self) -> int:
        # Read live so an injected MAX_KEYS is honoured.
        return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))

    # -- the one behavioural change --------------------------------------
    def write_key(self, key: str, value: str) -> str:
        cap = self._capacity()
        displaced: str | None = None
        if key not in self._store and len(self._store) >= cap:
            # Displace the least recently refreshed entry to make room.
            while self._order and len(self._store) >= cap:
                victim = self._order.pop(0)
                if victim in self._store:
                    del self._store[victim]
                    displaced = victim
                    break
            if displaced is None:
                # _order desynchronised (shouldn't happen); fall back to
                # insertion order so the invariant still holds.
                victim = next(iter(self._store))
                del self._store[victim]
                displaced = victim
        self._store[key] = str(value)
        self._refresh(key)
        if displaced is not None:
            return (
                f"Key '{key}' written. Memory was full, so the least recently "
                f"used entry '{displaced}' was displaced and is now lost."
            )
        return f"Key '{key}' written."

    def clear_key(self, key: str) -> str:
        out = super().clear_key(key)
        if key in self._order:
            self._order.remove(key)
        return out

    # -- read-out is recency ordered -------------------------------------
    def _ordered_items(self) -> List[tuple[str, str]]:
        seen = [k for k in self._order if k in self._store]
        seen += [k for k in self._store if k not in seen]
        return [(k, self._store[k]) for k in seen]

    def snapshot(self) -> str:
        items = self._ordered_items()
        if not items:
            return "  (empty)"
        lines = []
        for i, (k, v) in enumerate(items):
            tag = "  (most recent)" if i == len(items) - 1 else ""
            lines.append(f'  "{k}": {v!r}{tag}')
        return "\n".join(lines)

    def to_recall_text(self) -> str:
        items = self._ordered_items()
        if not items:
            return "(memory is empty)"
        lines = []
        for i, (k, v) in enumerate(items):
            tag = "  (most recent)" if i == len(items) - 1 else ""
            lines.append(f"{k}: {v}{tag}")
        return "\n".join(lines)


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with the displacement store installed.

    Nothing else is overridden: encode/recall/step, segmentation, key naming,
    the tool-call cap and the prompts are the baseline's, so the only difference
    between this candidate and `baseline` is what happens on overflow.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wm = DisplacementMemory()


MANIFEST = {
    "id": "displacement",
    "parent": "baseline",
    "capacity": MAX_KEYS,
    "decay": (
        "none. No stochastic loss of any kind. Overflow-driven displacement only: "
        "a write to a new key on a full store evicts the least recently refreshed "
        "entry. Deterministic, agent-controlled, and reproducible."
    ),
    "role": "contender -- first mechanism proposal of the search",
    "summary": (
        "Same capacity (MAX_KEYS = 4, Cowan 2001 untouched), different overflow "
        "semantics: displacement of the least recently refreshed entry instead of "
        "refusing the write. PSYCHOLOGICAL ARGUMENT: no account of human working "
        "memory contains a refuse-to-encode state. Cowan's focus of attention "
        "admits a new chunk and pushes the least active one out; interference "
        "accounts (Oberauer & Kliegl 2006) make overflow overwriting, not "
        "rejection; rehearsal refreshes activation and protects an item "
        "(Baddeley; Barrouillet & Camos 2007), which is why displacement is "
        "least-recently-*refreshed* rather than FIFO. Refuse-on-full is a "
        "data-structure reading of a capacity limit and it produces a failure "
        "humans never show -- response omission rather than error. EVIDENCE: on "
        "n-back the baseline answers only 6.82 of 14 trials at n=3, where the "
        "store saturates (slot_utilization 0.99), while acc_over_answered stays "
        "0.737; full_context answers 14/14 at 0.770, so its whole +0.157 n-back "
        "gain is response production. The n=3 key sets (31/50 end as "
        "position_2/3/8/10 -- non-contiguous, high-index) require delete+write per "
        "letter, i.e. 2 tool calls against a budget of max(6, 1.5*steps); "
        "reproduced offline, the budget pins at 0 from turn 5 and truncated "
        "repairs leave a holey, stale store in which only 4 of 198 n=3 values "
        "still contain a letter. Displacement lowers tool-call demand to one "
        "write per letter; the cap itself is left untouched so a gain cannot be "
        "read as extra compute, and no prompt or tool-description text differs "
        "from the baseline so exactly one method body separates the two. "
        "FALSIFIABLE: n=3 `answered` >= 12 with "
        "acc_over_answered within +-0.05 of 0.737 and slot_utilization still ~1.0. "
        "If `answered` does not rise the diagnosis is dead; if it rises while "
        "acc_over_answered drops below 0.65 the silence was incapacity, not the "
        "store. Exposed task is semantic_story_recall (136/200 rows hit the "
        "refusal): kept chunks shift from the first four to the last four, so a "
        "small regression is expected and tells us primacy protection is the "
        "iteration-2 ingredient."
    ),
}
