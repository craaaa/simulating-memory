"""Candidate `primacy_v2`: primacy's confirmed U-shaped retention, with the
maintenance limit that simultaneous arrival forces on it.

Parent: `primacy` (iteration 2a). Grandparent: `displacement`.

ONE method body differs from the baseline -- `WorkingMemory.write_key` -- as in
`displacement` and `primacy`. What differs from `primacy` is exactly one thing:
**how far the store is reduced when a write overruns it.** The activation
formula, the eviction ORDER, the episode bookkeeping, the read-out format, the
capacity, the prompts, the tools and `_tool_call_cap()` are `primacy`'s
untouched, so the U-shaped survival function it confirmed is preserved verbatim.

WHY, IN ONE PARAGRAPH
---------------------
`primacy` evicts exactly one entry per overflowing write, so the store is always
exactly full -- occupancy rose from the baseline's 3.33/3.60 to 3.67/3.92/4.00 on
the three gist tasks. On `craft_task` that is a *capability gain* (accuracy
0.9333 -> 0.9720) and on an inverted objective a capability gain is a
humanlikeness loss (craft humanlikeness is monotone decreasing in accuracy above
the baseline across six measured runs). The correction is not to the eviction
order -- which is confirmed -- but to the assumption that a full focus of
attention can absorb a surplus chunk one-for-one. Measured here: the encode-phase
writes arrive as ONE parallel assistant message (67 of 200 baseline story rows
refuse writes 5 AND 6 with no intervening `delete_key`, i.e. the agent issued
write 6 without having seen write 5's refusal). Under simultaneous arrival there
is no free time in which to refresh what is held: attention must encode and
maintain at once (Barrouillet & Camos 2007, a single time-shared resource), and
the focus is committed to the chunk being encoded rather than being a free extra
slot (Oberauer 2002). The set maintainable in that regime is therefore
MAX_KEYS - 1 = 3, not 4. So an overflowing write in a simultaneous batch settles
the store at 3; a serially arriving write (one per turn, as in n-back and
variable_mapping) has its free time back and settles at 4, which is `primacy`
exactly.

Net: no numeric constant of its own -- the maintenance limit is capacity - 1 and
moves with MAX_KEYS -- and the store can no longer be held full by an agent that
overruns it.

See MANIFEST.md beside this file for the run-data evidence, the replay
measurements, the composition notes and the pre-registered predictions.
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

# The ACT-R base-level decay. Anderson & Lebiere (1998), The Atomic Components of
# Thought; Anderson & Schooler (1991) derive d ~ 0.5 from the statistics of how
# often information is actually needed again. Used at 0.5 throughout the ACT-R
# literature and NOT fitted here. Inherited unchanged from `primacy`.
ACTR_DECAY = 0.5


def _maintenance_limit(capacity: int) -> int:
    """How many entries can be MAINTAINED while attention is also encoding.

    ONE integer, and it is tied to capacity rather than fitted: capacity - 1.

    Oberauer (2002): the focus of attention holds the chunk currently being
    operated on, and the focus is one of the limited-capacity representations,
    not a free extra. Barrouillet & Camos (2007): there is a single attentional
    resource, time-shared between processing and refreshing. Put together, while
    a chunk is being encoded under contention the focus is committed to it, so
    the set that can be held alongside is capacity - 1.

    Cowan's (2001) 4 is what the store can hold when nothing is being encoded --
    that is MAX_KEYS and it is unchanged. capacity - 1 is what it can hold while
    encoding is in progress, which is the regime a simultaneous supra-capacity
    batch never leaves.

    This is a COMMITMENT, not a derivation: it introduces no numeric constant of
    its own, it moves with MAX_KEYS, and an earlier draft that derived 3 from the
    decay exponent is withdrawn -- that derivation compared an attention share
    against a strength loss, so its units did not match, and its answer was
    unstable in d (d = 0.6 gives 2). It is also bounded below by 1 and above by
    capacity - 1, so it can only ever make the store hold FEWER entries than
    MAX_KEYS, never more.
    """
    return max(1, int(capacity) - 1)

_UNSET = object()


class PrimacyV2Memory(_BaseMemory):
    """`primacy`'s activation-ordered store, with a maintenance limit.

    Invariant: at most ``MAX_KEYS`` entries at any time, and never more than that
    even transiently -- eviction strictly precedes insertion. Identical to the
    baseline, to `displacement` and to `primacy` in that respect.

    Activation, eviction order and episode bookkeeping are `primacy`'s, verbatim:

        A_i = ln( SUM_k  w_k * (T - t_k)^-d ),  d = 0.5
        presentations: own write w = 1.0;
                       same-episode later write, while resident: w = 1/|store|
        overflow evicts argmin A_i

    The single change is the STOPPING POINT of eviction:

        simultaneous batch (this assistant message issues >1 write)
            evict until |store| <= (MAX_KEYS - 1) - 1, then admit      -> 3 held
        serial arrival (one write per assistant message)
            evict until |store| <= MAX_KEYS - 1, then admit            -> 4 held
              == `primacy` == `displacement` for this case

    Consequences, all forced rather than chosen:

      * The U-shape is untouched. Eviction still takes argmin activation, which
        is still a middle write position, and the first- and last-written entries
        still survive: for a 5-write batch the store settles at {1, 2, 5} and for
        a 6-write batch at {1, 2, 5, 6} (the sixth write lands on a store of 3,
        which is not full, so it is admitted without any eviction). Both retain
        both ends. `primacy`'s confirmed P2 quantity therefore still reads 1.000
        and still CAN vary, because a rule that kept only a contiguous head or
        tail would read 0.000, as the baseline and `displacement` both did.
      * Occupancy oscillates rather than pinning at capacity. A displacing write
        leaves 3; a following non-displacing write refills to 4. So the store's
        occupancy at recall depends on whether the last write displaced --
        the store cannot be kept full by an agent that overruns it, which is the
        specific thing `primacy` got wrong.
      * Serial updating is untouched by construction. One write per turn is a
        batch of one, so n-back and variable_mapping see `primacy` exactly.
      * Digit span cannot move. Only 6 of 190 baseline rows write more than four
        distinct keys, and every one of those 6 already scores exact = 0, so the
        extra loss cannot change a score that is already zero.
    """

    def __init__(self, owner: Any = None) -> None:
        super().__init__()
        # Read-only back-reference, used ONLY to observe (a) presentation-episode
        # boundaries and (b) how many writes the current assistant message
        # issues. Deliberately NOT an override of `step()`: iteration 3a owns
        # `step()` / `reset_messages()`, so this candidate stays off that surface
        # and infers both facts from the message list. Nothing is ever written
        # through this reference.
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

    def _maintained(self) -> int:
        """Entries maintainable while attention is also encoding. < capacity."""
        cap = self._capacity()
        return max(1, min(cap, _maintenance_limit(cap)))

    # -- presentation-episode bookkeeping ---------------------------------
    def _episode_mark(self) -> Any:
        """A value that changes exactly when a new presentation begins.

        The number of user turns in the agent's message list. Every write inside
        one `step()` sees the same count; the next `step()` appends a user
        message and the count moves. Returns None when there is no owner (the
        offline capacity check instantiates the store bare), in which case all
        writes fall in one episode and the invariant still holds.
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
            # Monotonic, so CLEARING the message history advances the episode
            # rather than colliding with an earlier one. This is what makes the
            # candidate safe to compose with a `reset_messages()` rewrite.
            self._episode += 1

    # -- simultaneity -----------------------------------------------------
    def _batch_writes(self) -> int | None:
        """How many write_memory calls the assistant message being dispatched has.

        `step()` appends the assistant message (carrying every tool call it
        requested) BEFORE dispatching them one at a time, appending a `role:
        tool` result after each. So the last message in the list that carries
        `tool_calls` is the one currently being dispatched, and its length is the
        number of chunks the agent offered simultaneously.

        Read-only, and returns None when there is no message list to read, in
        which case the store falls back to serial semantics -- i.e. to `primacy`.
        """
        msgs = getattr(self._owner, "_messages", None)
        if not isinstance(msgs, list):
            return None
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            calls = m.get("tool_calls")
            if not calls:
                continue
            n = 0
            for tc in calls:
                try:
                    if tc["function"]["name"] == "write_memory":
                        n += 1
                except (KeyError, TypeError):
                    continue
            return n
        return None

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

    def _drop(self, key: str) -> None:
        self._store.pop(key, None)
        self._pres.pop(key, None)
        self._ep.pop(key, None)

    # -- the one behavioural change --------------------------------------
    def write_key(self, key: str, value: str) -> str:
        cap = self._capacity()
        self._sync_episode()
        self._tick += 1
        now = self._tick

        displaced: List[str] = []
        if key not in self._store and len(self._store) >= cap:
            batch = self._batch_writes()
            simultaneous = bool(batch is not None and batch > 1)
            # The only difference from `primacy`: where eviction stops.
            target = (self._maintained() - 1) if simultaneous else (cap - 1)
            target = max(0, min(target, cap - 1))
            # Eviction strictly precedes insertion, so occupancy goes
            # cap -> target -> target + 1 and NEVER transiently exceeds cap.
            guard = 0
            while self._store and len(self._store) > target and guard <= cap:
                guard += 1
                victim = self._victim(now)
                if victim is None:
                    break
                self._drop(victim)
                displaced.append(victim)
            if len(self._store) >= cap:
                # Defensive: the invariant must hold unconditionally.
                victim = next(iter(self._store))
                self._drop(victim)
                displaced.append(victim)

        self._store[key] = str(value)
        self._pres.setdefault(key, []).append((now, 1.0))
        self._ep[key] = self._episode

        # Cumulative rehearsal: one unit of attention, shared over what is held.
        # Unchanged from `primacy`.
        n = len(self._store)
        if n:
            share = 1.0 / n
            for k in self._store:
                if k != key and self._ep.get(k) == self._episode:
                    self._pres.setdefault(k, []).append((now, share))

        if len(displaced) == 1:
            # Byte-identical to `displacement`'s and `primacy`'s message, so a
            # single-eviction write is textually indistinguishable from them.
            return (
                f"Key '{key}' written. Memory was full, so the least recently "
                f"used entry '{displaced[0]}' was displaced and is now lost."
            )
        if displaced:
            names = ", ".join(f"'{d}'" for d in displaced[:-1])
            names = f"{names} and '{displaced[-1]}'"
            return (
                f"Key '{key}' written. Memory was full, so the least recently "
                f"used entries {names} were displaced and are now lost."
            )
        return f"Key '{key}' written."

    def clear_key(self, key: str) -> str:
        out = super().clear_key(key)
        self._pres.pop(key, None)
        self._ep.pop(key, None)
        return out

    # -- read-out: deliberately the baseline's -----------------------------
    #
    # `snapshot()` and `to_recall_text()` are NOT overridden, exactly as in
    # `primacy`. `displacement` reordered the read-out by recency and tagged the
    # last entry "(most recent)"; that is a change to the recall-time
    # representation rather than to the store's dynamics and it would confound
    # the comparison that matters here. Surviving keys sit in `_store` in write
    # order, so the baseline's plain iteration already presents them in serial
    # order with a gap where the middle was lost.


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with the maintenance-limited activation store installed.

    Nothing else is overridden. `step()`, `reset_messages()`, `encode()`,
    `recall()`, segmentation, key naming, `_tool_call_cap()` and every prompt are
    the baseline's, so the only difference from `baseline` is what happens on
    overflow, and the only difference from `primacy` is how far eviction goes
    when the chunks arrived simultaneously.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wm = PrimacyV2Memory(owner=self)


MANIFEST = {
    "id": "primacy_v2",
    "parent": "primacy",
    "capacity": MAX_KEYS,
    "decay": (
        "ACT-R base-level activation, A_i = ln(SUM_k w_k * (T - t_k)^-0.5), with "
        "d = 0.5 the canonical ACT-R decay (Anderson & Lebiere 1998; Anderson & "
        "Schooler 1991), identical to `primacy`. Not stochastic: no random number "
        "is drawn anywhere. Presentations are an item's own writes (w = 1) plus "
        "cumulative rehearsal from later writes in the same presentation episode, "
        "with one unit of attention shared over the items held (w = 1/|store|). "
        "Overflow evicts argmin A_i, as in `primacy`. NEW: eviction stops at "
        "(MAX_KEYS - 1) - 1 = 2 rather than MAX_KEYS - 1 = 3 when the chunks "
        "arrived in one simultaneous assistant message, so such a write settles "
        "the store at 3. The maintenance limit is MAX_KEYS - 1: the focus of "
        "attention is committed to the chunk being encoded and is not a free "
        "extra slot (Oberauer 2002), and there is one time-shared resource for "
        "processing and refreshing (Barrouillet & Camos 2007). It introduces no "
        "numeric constant of its own and moves with MAX_KEYS."
    ),
    "role": "contender -- iteration 3b, repairs primacy's craft_task overshoot",
    "summary": (
        "Keeps `primacy`'s confirmed U-shaped retention (both-ends survival "
        "0.000 -> 1.000, retention minimum at a middle write position) and "
        "changes ONE thing: how far eviction goes when the chunks arrived "
        "simultaneously. DIAGNOSIS FROM THE RUN DATA: primacy's craft_task "
        "'regression' is a CAPABILITY GAIN. Craft accuracy went 0.9333 -> 0.9720 "
        "while its humanlikeness went 0.8907 -> 0.8456, and craft humanlikeness "
        "is monotone decreasing in accuracy across six measured runs "
        "(random_decay_v2 0.7613 -> 0.9275, baseline 0.9333 -> 0.8907, "
        "displacement 0.9573 -> 0.8627, primacy 0.9720 -> 0.8456, full_context "
        "1.0000 -> 0.8130). narrative_qa moved the OPPOSITE way (0.800 -> 0.734), "
        "so the two iteration-2 regressions have opposite capability signs and no "
        "single rehearsal correction can move both toward the baseline -- that "
        "premise of the brief is refuted by the data. What craft needs is lower "
        "occupancy: primacy holds the store exactly full (3.67/3.92/4.00 on the "
        "three gist tasks, against the baseline's 3.33/3.60/3.60). MEASURED "
        "MECHANISM: the encode-phase writes arrive as ONE parallel assistant "
        "message -- 67 of 200 baseline story rows refuse writes 5 AND 6 with no "
        "intervening delete_key, so the agent never saw the first refusal. Under "
        "simultaneous arrival there is no free time in which to refresh what is "
        "held, and with one time-shared resource for processing and refreshing "
        "(Barrouillet & Camos 2007) and the focus committed to the chunk being "
        "encoded (Oberauer 2002), the maintainable set is MAX_KEYS - 1 = 3 rather "
        "than Cowan's 4. So a simultaneous overflowing write settles the store "
        "at 3; a serially arriving write keeps 4 and is `primacy` exactly, which "
        "is why n-back and variable_mapping are untouched by construction. NOT "
        "CAPACITY: MAX_KEYS = 4, never exceeded, not even transiently (eviction "
        "strictly precedes insertion). NOT NOISE: no random draw; the retained "
        "sets are deterministic functions of write order. NOT EVICTING LESS: "
        "occupancy falls on every gist task, and a pre-registered row FAILS if "
        "retained-key count or story similarity rises above primacy's. NOT A "
        "CARVE-OUT: the rule reads only the number of writes in the current "
        "assistant message and the store's own activations; it cannot name a task. "
        "FALSIFIABLE: craft_task recovers to >= 0.8657 (primacy 0.8456, floor "
        "-0.030 of the baseline's 0.8907) with craft accuracy <= 0.9333; "
        "semantic_story_recall stays >= 0.9173; both-ends retention on "
        "overflowing story rows stays >= 0.80; n=3 n-back keeps answered >= 13 and "
        "keys_held >= 3.5; digit span unmoved, which is now provable rather than "
        "argued -- all 6 overflowing digit-span rows already score exact = 0, so "
        "the extra eviction cannot change a score. CARRIED CORRECTION: the serving "
        "stack is NOT bit-deterministic. chunk_limit reproduced only 143 of 150 "
        "craft rows at temperature 0.0, and primacy differs from the baseline on 7 "
        "craft rows that never overflow. Score DISTRIBUTIONS reproduce exactly; "
        "rows do not. Every no-change prediction here is written on a distribution "
        "or a scalar, never on bit-identity."
    ),
}
