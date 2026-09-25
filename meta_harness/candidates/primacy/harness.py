"""Candidate `primacy`: displacement's overflow fix, plus the rehearsal that
makes early chunks resist it.

Parent: `displacement` (iteration 1). Grandparent: `baseline`.

ONE method body differs from the baseline -- `WorkingMemory.write_key` -- exactly
as in `displacement`. What differs from `displacement` is *which* entry leaves:
the least *activated* entry rather than the least recently refreshed one, where
activation is the ACT-R base-level equation over an item's presentations
(Anderson & Lebiere 1998), and presentations include the cumulative rehearsal an
item receives from later writes in the same presentation episode.

`MAX_KEYS` stays 4. `TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS`,
`_tool_call_cap()`, `recall()`'s context construction, `step()` and
`reset_messages()` are all untouched.

See MANIFEST.md beside this file for the full argument, the offline measurements
and the pre-registered predictions. The short version:

  activation   A_i = ln( SUM_k  w_k * (T - t_k)^-d ),  d = 0.5
  presentation own write          w = 1.0
               same-episode later write, while resident:  w = 1 / |store|
  overflow     evict argmin A_i, admit the new key, name the loss in the result

One numeric parameter, d = 0.5, the canonical ACT-R base-level decay. The
primacy gradient's slope is not chosen at all: it falls out of MAX_KEYS = 4 via
1/n attention sharing (the first item is rehearsed at weight 1/2, the second at
1/3, the third at 1/4, the fourth not at all before the store saturates).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from bench.core import working_memory as _wm_mod
from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    SummarizerAgent,
    TOOLS,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS, WorkingMemory as _BaseMemory

# The ACT-R base-level decay. Anderson & Lebiere (1998), The Atomic Components
# of Thought; Anderson & Schooler (1991) derive d ~ 0.5 from the statistics of
# how often information is actually needed again in the environment. It is used
# at 0.5 across the ACT-R modelling literature and is NOT fitted here -- it is
# the one numeric constant in this candidate and it is a named one.
ACTR_DECAY = 0.5

_UNSET = object()


class PrimacyMemory(_BaseMemory):
    """Capacity-limited store whose overflow rule is activation-based eviction.

    Invariant: at most ``MAX_KEYS`` entries at any time -- identical to the
    baseline and to `displacement`. What differs is which entry leaves.

    Activation is the ACT-R base-level equation. An item's presentations are its
    own writes (full attention, weight 1) plus the cumulative rehearsal it
    receives while another item is being written *in the same presentation
    episode*, with the single unit of attention shared over the items currently
    held (weight 1/|store|).

    Consequences, all of them forced rather than chosen:

      * Within a presentation episode the earliest items are rehearsed most
        often (1/2, then 1/3, then 1/4 as the store fills), so they accumulate
        strength and resist displacement -- the Atkinson & Shiffrin (1968)
        account of primacy. The newest item has recency. The *middle* of the
        list is the weakest and is evicted first, which is the Murdock (1962)
        serial-position curve rather than a "protect the first N" rule.
      * Across presentation episodes no rehearsal credit accrues, because
        attention is consumed by the incoming item. So in a continuous
        updating task -- one write per turn, as in n-back -- every resident has
        a single presentation, activation reduces to (T - t)^-d, argmin is the
        oldest entry, and the rule is exactly `displacement`. This is the
        empirically correct asymmetry: running-memory span shows recency
        without primacy (Pollack, Johnson & Knaff 1959; Bunting, Cowan &
        Saults 2006), unlike free recall of a presented list.
      * Nothing becomes immortal. Rehearsal credit only accrues inside an
        episode, and episodes here are at most six writes long (the agent's own
        `_tool_call_cap()` is max(6, 1.5*interactions) and a batch encode is one
        interaction). After the episode ends, every presentation decays as
        T^-0.5, so every entry becomes evictable again. This is why the store
        cannot re-freeze the way the baseline's refuse-on-full state did.
    """

    def __init__(self, owner: Any = None) -> None:
        super().__init__()
        # Read-only back-reference, used ONLY to observe presentation-episode
        # boundaries. It is deliberately NOT an override of `step()`: iteration
        # 2c owns `step()` / `reset_messages()`, so this candidate stays off
        # that surface entirely and infers the boundary from the message list
        # instead. Nothing is written through this reference.
        self._owner = owner
        self._pres: Dict[str, List[Tuple[int, float]]] = {}
        self._ep: Dict[str, int] = {}
        self._tick = 0
        self._episode = 0
        self._mark: Any = _UNSET

    # -- capacity ---------------------------------------------------------
    def _capacity(self) -> int:
        # Read live so an injected MAX_KEYS is honoured; never freeze at import.
        return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))

    # -- presentation-episode bookkeeping ---------------------------------
    def _episode_mark(self) -> Any:
        """A value that changes exactly when a new presentation begins.

        The number of user turns in the agent's message list. Every write inside
        one `step()` sees the same count; the next `step()` appends a user
        message and the count moves. Returns None when there is no owner (e.g.
        the offline capacity check instantiates the store bare), in which case
        all writes fall in one episode and the invariant still holds.
        """
        msgs = getattr(self._owner, "_messages", None)
        if not isinstance(msgs, list):
            return None
        return sum(1 for m in msgs
                   if isinstance(m, dict) and m.get("role") == "user")

    def _sync_episode(self) -> None:
        mark = self._episode_mark()
        if mark != self._mark:
            self._mark = mark
            # Monotonic, so clearing the message history advances the episode
            # rather than colliding with an earlier one.
            self._episode += 1

    # -- activation -------------------------------------------------------
    def _activation(self, key: str, now: int) -> float:
        """ACT-R base level: ln( SUM_k w_k * age_k^-d ). -inf if unknown."""
        pres = self._pres.get(key)
        if not pres:
            # No recorded history (cannot happen through write_key, but a
            # desynchronised entry must be evictable rather than raise).
            return float("-inf")
        total = 0.0
        for t, w in pres:
            total += w * (max(now - t, 1) ** (-ACTR_DECAY))
        return math.log(total) if total > 0.0 else float("-inf")

    def _last_tick(self, key: str) -> int:
        pres = self._pres.get(key)
        return pres[-1][0] if pres else -1

    def _victim(self, now: int) -> str | None:
        """Least activated resident; ties go to the older last presentation."""
        if not self._store:
            return None
        return min(self._store,
                   key=lambda k: (self._activation(k, now), self._last_tick(k)))

    # -- the one behavioural change --------------------------------------
    def write_key(self, key: str, value: str) -> str:
        cap = self._capacity()
        self._sync_episode()
        self._tick += 1
        now = self._tick

        displaced: str | None = None
        if key not in self._store and len(self._store) >= cap:
            while self._store and len(self._store) >= cap:
                victim = self._victim(now)
                if victim is None:
                    break
                del self._store[victim]
                self._pres.pop(victim, None)
                self._ep.pop(victim, None)
                displaced = victim
                break
            if displaced is None and len(self._store) >= cap:
                # Defensive: the invariant must hold unconditionally.
                victim = next(iter(self._store))
                del self._store[victim]
                self._pres.pop(victim, None)
                self._ep.pop(victim, None)
                displaced = victim

        self._store[key] = str(value)
        self._pres.setdefault(key, []).append((now, 1.0))
        self._ep[key] = self._episode

        # Cumulative rehearsal: one unit of attention, shared over what is held.
        n = len(self._store)
        if n:
            share = 1.0 / n
            for k in self._store:
                if k != key and self._ep.get(k) == self._episode:
                    self._pres.setdefault(k, []).append((now, share))

        if displaced is not None:
            # Byte-identical to `displacement`'s message, so the two candidates
            # differ only in which key it names.
            return (
                f"Key '{key}' written. Memory was full, so the least recently "
                f"used entry '{displaced}' was displaced and is now lost."
            )
        return f"Key '{key}' written."

    def clear_key(self, key: str) -> str:
        out = super().clear_key(key)
        self._pres.pop(key, None)
        self._ep.pop(key, None)
        return out

    # -- read-out: deliberately the baseline's -----------------------------
    #
    # `snapshot()` and `to_recall_text()` are NOT overridden. `displacement`
    # reordered the read-out by recency and tagged the last entry "(most
    # recent)"; that is a change to the recall-time representation rather than
    # to the store's dynamics, and it would confound the comparison that
    # matters here. It is also wrong for this store: the surviving keys sit in
    # `_store` in write order, so the baseline's plain iteration already
    # presents them in serial order with a gap where the middle was lost, which
    # is exactly what a primacy-plus-recency store should hand to recall.


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with the activation-based store installed.

    Nothing else is overridden. `step()`, `encode()`, `recall()`, segmentation,
    key naming, `_tool_call_cap()` and every prompt are the baseline's, so the
    only difference from `baseline` is what happens on overflow, and the only
    difference from `displacement` is which entry overflow removes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wm = PrimacyMemory(owner=self)


MANIFEST = {
    "id": "primacy",
    "parent": "displacement",
    "capacity": MAX_KEYS,
    "decay": (
        "ACT-R base-level activation, A_i = ln(SUM_k w_k * (T - t_k)^-0.5), with "
        "d = 0.5 the canonical ACT-R decay (Anderson & Lebiere 1998; Anderson & "
        "Schooler 1991). Not stochastic: no random number is drawn anywhere. "
        "Presentations are an item's own writes (w = 1) plus cumulative rehearsal "
        "from later writes in the same presentation episode, with one unit of "
        "attention shared over the items held (w = 1/|store|). Overflow evicts "
        "argmin A_i."
    ),
    "role": "contender -- iteration 2a, primacy protection on top of displacement",
    "summary": (
        "Keeps displacement's overflow fix (a write to a full store is admitted, "
        "not refused) and changes only which entry leaves: least ACTIVATED rather "
        "than least recently refreshed. PSYCHOLOGICAL ARGUMENT: displacement by "
        "recency alone has no primacy mechanism, and iteration 1 measured the "
        "consequence -- the surviving story-recall chunks moved from setting / "
        "beginning / first encounter to pie man / second appearance / key_moment, "
        "and the task lost 0.0509. Atkinson & Shiffrin (1968) explain primacy by "
        "the extra rehearsal early items receive before the buffer saturates; "
        "with one unit of attention shared over what is held (Barrouillet & Camos "
        "2007), the first item is rehearsed at 1/2, the second at 1/3, the third "
        "at 1/4 and the fourth not at all. Scored by the ACT-R base-level "
        "equation with d = 0.5, that yields a U-shaped survival function: the "
        "MIDDLE of the list is evicted first, then outward. Rehearsal credit "
        "accrues only within a presentation episode, so a continuous updating "
        "task (one write per turn, n-back) reduces exactly to displacement -- "
        "which is the empirically right asymmetry, since running-memory span "
        "shows recency without primacy. NOT CAPACITY: MAX_KEYS = 4, at most four "
        "entries ever held, tool-call cap untouched. NOT NOISE: no random draw; "
        "replaying the iteration-1 write sequences offline, the fraction of "
        "overflowing story-recall rows in which BOTH the first-written and the "
        "last-written chunk survive is 0.000 for the baseline, 0.000 for "
        "displacement and 0.914 here, while uniform random retention of 4 of 6 "
        "would give ~0.4. FALSIFIABLE: semantic_story_recall recovers to within "
        "0.03 of the baseline's 0.9473 and beats displacement's 0.8964 by more "
        "than 0.025; n-back n=3 keys_held stays >= 3.5 and answered >= 13, i.e. "
        "the iteration-1 gain is not paid back; digit span byte-unchanged "
        "(A1 0.1298, best_span 18.4); A3 BLEU falls back under 0.023 with recall "
        "length 115-145 words. ALSO CARRIED: a measured account of iteration 1's "
        "unexplained A3 BLEU rise -- it is a brevity-penalty artifact of recall "
        "LENGTH, not verbatimness. Length-free n-gram precision against the story "
        "is flat across write position (p4 = 0.066 / 0.068 / 0.045 / 0.073 / "
        "0.029 / 0.113 for positions 1..6), and replaying displacement's own "
        "written values through the three overflow rules gives BLEU 0.0048 "
        "(refuse, 122 words) / 0.0335 (LRU, 141 words) / 0.0087 (this rule, "
        "124 words) from the identical value text."
    ),
}
