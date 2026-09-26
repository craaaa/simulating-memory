"""Candidate `evicting_reset`: `episodic_reset_v3`, with overflow made an
across-presentation displacement and left a within-presentation selection cap.

Parent: `episodic_reset_v3` (iteration 4, mean 0.8163, one floor violation).

ONE change from the parent, and it is confined to `WorkingMemory.write_key`.
`step()` is copied from v3 verbatim, character for character, so the PROMPT DELTA
IS ZERO. `TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS`, `_tool_call_cap()`,
`encode()`, `recall()`, `snapshot()`, `to_recall_text()` and `reset_messages()`
are all untouched.

    write to a NEW key on a full store
      -> if every resident was laid down in the CURRENT presentation, refuse
         (byte-identical to the baseline, including the error string)
      -> otherwise displace the least recently refreshed resident laid down in an
         EARLIER presentation, and admit the write

A "presentation" is one `step()` call, read from `len(self._step_log)`.

===============================================================================
SECTION 1 -- WHAT THE SECOND CAUSE IS, AND WHY THE BRIEF'S FRAMING IS ONE
             CONDITIONING VARIABLE OFF
===============================================================================
I was asked to find, from v3's `step_log`, where n=2's missing ~6 answered turns
go. They are fully accounted for, and the shape of the loss is a **period-2
oscillation** that the pooled cross-tabulation in the WORKLOG averages away.

Answer rate per stimulus turn, v3 run, 50 blocks per level (turn 1 is the first
letter; `b0` is the share of turns ending with zero tool-call budget):

    n=1  1.00 1.00 1.00 0.82 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    b0   0.00 0.00 0.18 0.58 0.66 0.66 0.68 0.60 0.68 0.56 0.62 0.38 0.48 0.36 0.50

    n=2  1.00 1.00 1.00 0.92 0.86 0.24 0.72 0.16 0.68 0.20 0.74 0.32 0.76 0.26 0.80 0.30
    n=3  1.00 1.00 1.00 1.00 1.00 0.00 0.66 0.00 0.90 0.00 0.96 0.00 0.96 0.00 1.00 0.00 1.00

At n=3 the rate is **exactly 0.00 on every even turn from turn 6 on**, over 50
blocks, and 0.66-1.00 on every odd turn. Summing the n=2 row over the 14 scored
turns (turns 3..16) gives **7.96**, against the reported `answered` of **7.98**.
So the missing turns are not diffuse: they are one alternating half of the block,
and nothing else is missing.

The cycle, read directly off one n=2 block's `step_log`:

    turn 4  budget_before 2: write `last_two_letters` (overwrite) -> OK
                             write `current_letter`   (new key)   -> "memory is full"
                             budget_after 0, cap_hit
                             tools switched to tool_choice="none"
            reply = '<tool_call>{"name":"delete_key","arguments":{"key":"second_letter"}}</tool_call>'
            -> no classification, trial unanswered
            -> AND THE DELETE NEVER EXECUTES, because it is text and not a tool
               call, so the store is still full on the next turn
    turn 5  budget_before 1: write `last_two_letters` (overwrite) -> OK
            reply = 'different'                       -> answered
    turn 6  budget_before 2: same as turn 4.  And so on.

**The conditioning variable that matters is the refusal, not the budget.** The
WORKLOG reports P(answer | budget=0) as 0.459 / 0.498 and reads the budget as the
cause. Splitting those same turns by whether the turn contained a memory-full
result:

    level   b=0 & refusal        b=0 & no refusal      b>0 & refusal    b>0 & none
      1        n=0   --            n=347  P=0.974         n=0   --       n=403  P=1.000
      2        n=363 P=0.174       n=195  P=0.990         n=5   P=1.000  n=237  P=1.000
      3        n=374 P=0.134       n=276  P=0.993         n=0   --       n=200  P=1.000

An exhausted budget with no refusal answers **0.990 / 0.993** -- indistinguishable
from n=1's 0.974 and from the unexhausted cells. An exhausted budget with a
refusal answers **0.174 / 0.134**. The budget is a *necessary accessory* (it is
what removes the tool and forces the model to speak its repair), but it is not the
binding constraint, and the arithmetic says so: the demand is ~1.48-1.47 tool
calls per turn against a supply of 1.5, and the refused write itself consumes one
of them (`_tool_calls_used += 1` fires on the error return in `wm_agent._dispatch_
tool`'s caller regardless of the result). Remove the refusal and the demand drops
below supply on the very turn it mattered.

This is also, and independently, why **`_tool_call_cap()` must stay untouched**,
and it is now a measured result rather than a judgement call: on the 471 turns
where the budget was exhausted and nothing was refused, the agent answered 99.1%
of the time. More budget buys nothing on the cell that is failing.

So the two causes separate cleanly, and neither is the other:

    CAUSE A (task set)   v2's control-state block carried the maintenance
                         obligation across the turn boundary and dropped the
                         response obligation. Fixed by v3's third sentence:
                         n=3 answered 2.12 -> 7.44, vm unparsed 38 -> 0.
    CAUSE B (overflow)   a refused write costs a tool call, the last one, and the
                         agent then spends its turn saying the repair out loud
                         instead of answering -- and the repair never lands, so
                         the state is self-sustaining. Not reachable by any
                         instruction: the model is denied the tool at the moment
                         it wants it. Fixed only by not refusing.

===============================================================================
SECTION 2 -- HOW I ACCOUNT FOR `episodic_primacy`'s 3.90, AND A BUG IN THE
             PARENT LINE THAT MAKES THAT NUMBER MEAN SOMETHING ELSE
===============================================================================
The cautionary datum is `episodic_primacy` = v2's `step()` + an evicting store,
which took n=3 from 2.12 only to 3.90 while eviction under the *baseline*
`step()` took 6.82 to 14.0. Two things explain it, and the second is a defect.

(a) **Cause A dominates and eviction cannot touch it.** v2 did not ask for a
    response, so the answer rate is capped by the task-set failure however the
    store behaves. `episodic_primacy` fixed B and left A; v3 fixed A and left B.
    Neither alone can clear, and there is no additivity to assume -- A is a
    ceiling, not an addend. This candidate is the first arm in which both are
    fixed.

(b) **`episodic_primacy` did not actually contain `primacy`'s rule.**
    `PrimacyMemory._episode_mark()` (and `PrimacyV2Memory`'s, verbatim) counts
    `role == "user"` messages in `owner._messages` to detect a presentation
    boundary. Under `episodic_reset_v2`/`v3`'s `step()`, which calls
    `reset_messages()` on entry, that count is **1 on every turn**, so the mark
    never changes and `_sync_episode()` never advances: every write in an entire
    n-back block lands in ONE episode, residents accrue unbounded rehearsal
    credit, and the rule is not the published one.

    The comment beside it says the monotonic counter is "what makes the candidate
    safe to compose with a `reset_messages()` rewrite." That is exactly backwards:
    monotonicity prevents a *collision* with an earlier episode; it does nothing
    about a mark that never moves.

    Measured, not argued. Replaying the real n-back write sequences from the v3
    run through `PrimacyMemory` with an advancing mark and with a frozen mark
    (offline, see MANIFEST.md section 7 for the full output):

        n=2   final key set differs on 18 of 50 blocks; evictions 246 -> 155
        n=3   final key set differs on 17 of 50 blocks; evictions 329 -> 201

    So `episodic_primacy`'s nback 0.5893 is a measurement of an untested third
    store, and it is weaker evidence against composition than it looks. This
    candidate has no `_messages` dependence anywhere: the presentation mark is
    `len(self._step_log)`, the same quantity v3 already uses for
    `_episode_turn_index()`, and it advances per `step()` whatever happens to the
    history.

===============================================================================
SECTION 3 -- WHY THE RULE IS EPISODE-CONDITIONAL, WHICH IS A FORCED CHOICE
===============================================================================
The obvious construction -- v3's `step()` plus one of the two validated evicting
stores -- **is measured to trade the n-back violation for a different one.**
Per-task humanlikeness against the baseline, floors being
`max(0.03, NOISE_FLOOR[task])`:

    rule                          story            craft            narrative
    baseline (refuse)             0.9473           0.8907           0.9572
    displacement (LRU)            0.8964  -0.0509  0.8627  -0.0280  0.9483
                                          VIOLATION
    primacy (ACT-R base level)    0.9477  +0.0004  0.8456  -0.0451  0.9263  -0.0309
                                                           VIOLATION      VIOLATION

Both rules evict; both clear n-back (0.9421, 0.9454); each breaks a different
batch task. LRU has no primacy mechanism, so free recall loses its early chunks.
The ACT-R rule protects them and costs `craft_task`, and the mechanism there is
not the eviction *order* at all: mean keys held on craft goes 3.33 (refusing) ->
3.67 (either evicting rule), because a refusing store makes the agent spend its
tool calls on delete-and-rewrite repairs that `tool_calls[:remaining]` truncates,
so it ends up holding *fewer* chunks. Admitting the writes hands the model more
retained material on a batch task, and since the objective inverts, more
retention is less humanlike. **Any** unconditional evicting store pays that,
which is why "remove the refusal" as stated cannot be the whole candidate.

The way out is not a better victim-selection rule. It is to notice that the
failure in section 1 is specifically an ACROSS-PRESENTATION failure, and that
overflow within a single presentation is a different operation:

  * **Across presentations** a stimulus arrives whether or not there is room. The
    incoming chunk enters the focus of attention and the least active resident is
    displaced; there is no option to decline it, because declining would mean the
    stimulus was not perceived. This is Cowan's (2001) focus of attention with a
    ~4-chunk capacity, overflow-as-overwriting in the interference accounts
    (Oberauer & Kliegl 2006; Oberauer et al. 2016), and the removal/updating
    operation of Ecker, Lewandowsky & Oberauer (2014). It is also where the
    empirical asymmetry lies: running-memory span and n-back show **recency
    without primacy** (Pollack, Johnson & Knaff 1959; Bunting, Cowan & Saults
    2006), so the victim is the least recently refreshed resident and no
    activation machinery, and no fitted decay constant, is needed.

  * **Within one presentation** the material is still in front of the participant
    and the order of encoding is theirs. What binds is not displacement but
    **selection**: how many chunks may be carried forward (Miller 1956 on
    recoding; Cowan 2001 on the chunk limit as a limit on what is *held*). The
    store's "memory is full" is then not a refusal to perceive a stimulus -- the
    stimulus is present regardless -- it is the report that the four-label index
    of that material is spent, and the agent may re-organise by overwriting or
    deleting, which is what it does. **This is the one reading under which the
    baseline's error string is psychologically admissible, and it is why this
    candidate keeps it exactly there and nowhere else.**

That distinction is task-agnostic: it is a property of when a write occurs
relative to a `step()` boundary, no task is named and no branch on task exists.
Its mapping onto the task set is a consequence, not a design input:

    all writes inside one step()     -> selection cap, i.e. THE BASELINE
        semantic_story_recall, craft_task, narrative_qa, word_recognition,
        digit_span_forward, digit_span_reverse  (all six use encode() -> recall(),
        and encode() calls step() exactly once)
    writes across many step() calls  -> displacement
        nback, variable_mapping

and the project's own record already treats the presentation episode as
psychologically real: `primacy`'s docstring argues the recency-without-primacy
asymmetry from the same boundary. What is new here is reading the boundary from
the turn counter instead of from a message list this candidate erases.

It also gives the rule its own falsifier, which an unconditional store does not
have: if any of the six batch tasks moves beyond its identical-path band, the rule
has leaked out of the across-presentation path and the composition is wrong.

===============================================================================
SECTION 4 -- WHY THE LEAK STAYS CLOSED AND n=1 STAYS FIXED
===============================================================================
The leak is closed by `step()`'s `reset_messages()`, which is copied verbatim and
is upstream of the store: no earlier stimulus, reply or tool result survives the
turn boundary, so the only route from letter k-n to the answer at k is the store.
v3 measured the consequence as n=3 letter-identity share 0.0202 -> 1.0000, and
nothing here touches it. If anything the store now carries *more* letter identity,
because a write that was refused is now admitted.

n=1 stays fixed for a reason that is structural rather than hopeful: at n=1 the
store never overflows. Over all 750 v3 stimulus turns the memory-full result count
is **0.00 per turn**, and a rule that changes only what happens on overflow cannot
reach a store that never overflows. So n=1 is the built-in control: it must stay at
13.98 answered with zero silent, and if it moves, something other than the overflow
rule changed.

**One qualification, because the record overstates this.** "n=1 stays fixed" is
true of v3's `answered` (13.98) and of its zero silent rows, but NOT of its
accuracy: v3's n=1 `acc_over_answered` is **0.8497** against the baseline's
**0.9943**, with `keys_held` 1.00 -> **2.06**. The store still never fills, so that
is not an overflow effect -- rendering the store into the prompt every turn changed
the agent's n=1 strategy from one overwritten key to two. It is a cost of the parent
that iteration 4 did not flag, and this candidate inherits it and cannot fix it, so
P1's n=1 accuracy leg is pitched at v3's 0.8497 and not at the baseline's 0.9943.

===============================================================================
SECTION 5 -- WHAT I COULD NOT TEST, AND WHAT IS A BOUND RATHER THAN A FORECAST
===============================================================================
Every counterfactual below replays the write sequence the model produced *under a
refusing store*. Under eviction it will not emit the delete-as-text repair and
will spend the freed budget differently, so these are bounds on the store rule's
behaviour given those writes, not forecasts of the run.

  * variable_mapping. Under v3 the store FREEZES: it changes on only 211 of 1350
    consecutive-question transitions, because the first four names are an
    absorbing state. Accuracy is 0.9833 when the queried name is resident and
    0.2561 when it is not, which is the leak closure working. The worry is that
    the +0.3213 gain is bought by the freeze. Replaying, the queried name is
    resident on 840/1500 questions under v3, 827/1500 under LRU-over-names and
    821/1500 under an ACT-R top-4, and mean `relation_count` among non-residents
    over residents is 1.374 / 1.388 / 1.393 (humans 1.386). So the interference
    structure A4 measures does not depend on the overflow rule. Bound, not
    forecast.
  * n-back at n=2 under eviction with a store that is *rendered every turn*. No
    arm has run that cell: `displacement` reached 14.0 at n=2 holding only 1.46
    keys, while v3 holds 3.58. A fuller store can evict the 2-back letter before
    it is needed, so `acc_over_answered` may fall even as `answered` rises. This
    is why P3's point estimate is a wide band and the mechanism rows, not the
    score, carry the falsification weight.
  * The serving stack is not deterministic and the recorded run-to-run floor
    understates it on the batch tasks. See MANIFEST.md section 6 and the
    `brief_corrections` block in `logs/pending_evicting_reset.json`:
    `craft_task` moved 0.0248 between the baseline and v3 on a code path that is
    provably identical, and `primacy` vs `primacy_v2` -- established as the same
    harness -- end with different story-recall stores on **200 of 200** rows.
"""
from __future__ import annotations

from typing import Any, Dict, List

from bench.core import working_memory as _wm_mod
from bench.core.wm_agent import (  # noqa: F401  -- re-exported unchanged
    CONDITION_PROMPTS,
    SummarizerAgent,
    TOOLS,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS, WorkingMemory as _BaseMemory

# ---------------------------------------------------------------------------
# The control-state block. COPIED FROM `episodic_reset_v3` VERBATIM.
#
# Every string below is character-for-character the parent's. The prompt delta of
# this candidate is ZERO by construction, and the one-line diff against v3 is the
# store. Nothing is imported from the sibling candidate: `inject.load_candidate` +
# `apply` is not idempotent, and a second load would subclass the already-injected
# class and stack the overrides.
# ---------------------------------------------------------------------------
WM_STATE_HEADER = "Your working memory currently contains:"

EPISODE_MARKER = "[ongoing episode]"

NO_TRANSCRIPT_NOTICE = (
    "You have no transcript of earlier turns: the key-value store is your only "
    "record of them."
)

STANDING_OBLIGATIONS = (
    "Two things are due on every turn of an episode. Update the store to track what "
    "you will need later, and give the response this turn's instructions call for. "
    "The response is due either way: if the store cannot be changed, or needs no "
    "change, respond anyway."
)

# The baseline's overflow message, reproduced here so that the within-presentation
# path is byte-identical to `bench.core.working_memory.WorkingMemory.write_key`
# rather than merely similar. The offline check asserts that equality over every
# write sequence the batch tasks actually produced: 2485 sequences, 10918 calls,
# 1167 overflow refusals, zero divergences in the returned string or the store.
REFUSAL_TEMPLATE = (
    "Error: memory is full ({cap} keys). "
    "Delete an existing key first or overwrite one."
)

# `displacement`'s result string, also byte-identical, so the affordance the agent
# is shown on overflow is not a new variable relative to the two validated
# evicting arms.
DISPLACED_TEMPLATE = (
    "Key '{key}' written. Memory was full, so the least recently "
    "used entry '{victim}' was displaced and is now lost."
)

_UNSET = object()


def _capacity() -> int:
    """Read capacity live, so an injected `MAX_KEYS` is honoured."""
    return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))


class EpisodicDisplacementMemory(_BaseMemory):
    """Capacity-limited store with two overflow regimes and one rule.

    Invariant, identical to the baseline and to both validated evicting arms: at
    most ``MAX_KEYS`` entries at any time. `verify_interface.py` checks it by
    instantiating this class bare and writing past capacity, which is why
    ``owner`` is optional.

    A write to a NEW key on a full store:

      * displaces the least recently refreshed resident that was laid down in an
        EARLIER presentation, and is admitted;
      * is refused, with the baseline's exact message, when every resident was
        laid down in the CURRENT presentation -- there is nothing that a new
        presentation has superseded, so the limit is a selection cap rather than
        a displacement.

    A presentation is one ``step()`` call, identified by ``len(owner._step_log)``.
    That counter is appended to by the base ``step()`` *after* the turn completes,
    so it is constant for every write within a turn and advances between turns --
    and, unlike a count of ``role == "user"`` messages, it is invariant to
    ``reset_messages()``. See the module docstring, section 2: reading the
    boundary from the message list is what silently disabled `primacy`'s rule
    inside `episodic_primacy`.

    With no owner (a bare instantiation) the mark is ``None`` forever, so every
    write falls in one presentation and the store is exactly the baseline. That is
    the conservative direction: the capacity check still passes and no eviction
    can happen by accident off the live path.
    """

    def __init__(self, owner: Any = None) -> None:
        super().__init__()
        # Read-only back-reference. Used ONLY to observe presentation boundaries;
        # nothing is ever written through it.
        self._owner = owner
        self._order: List[str] = []        # least recently refreshed first
        self._laid: Dict[str, int] = {}    # key -> presentation ordinal
        self._mark: Any = _UNSET
        self._presentation = 0

    # -- capacity ---------------------------------------------------------
    def _capacity(self) -> int:
        return _capacity()

    # -- presentation bookkeeping -----------------------------------------
    def _presentation_mark(self) -> Any:
        """A value that changes exactly when a new `step()` begins.

        ``None`` when there is no readable owner, in which case every write falls
        in one presentation and the store degenerates to the baseline.
        """
        log = getattr(self._owner, "_step_log", None)
        if not isinstance(log, list):
            return None
        return len(log)

    def _sync_presentation(self) -> None:
        mark = self._presentation_mark()
        if mark != self._mark:
            self._mark = mark
            # Monotonic, so a mark that goes backwards (it cannot here, but a
            # future harness could truncate the log) still advances rather than
            # colliding with an earlier presentation.
            self._presentation += 1

    # -- activation bookkeeping -------------------------------------------
    def _refresh(self, key: str) -> None:
        """An overwrite is a re-presentation: it refreshes recency AND re-stamps
        the presentation, so a chunk the agent has just written in this turn is
        not a candidate for displacement by its own next write in the same turn."""
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self._laid[key] = self._presentation

    def _victim(self) -> str | None:
        """Least recently refreshed resident from an EARLIER presentation."""
        for key in self._order:
            if key in self._store and self._laid.get(key) != self._presentation:
                return key
        # `_order` desynchronised (should not happen): fall back to insertion
        # order, still restricted to earlier presentations, so the regime
        # distinction cannot be lost to a bookkeeping slip.
        for key in self._store:
            if self._laid.get(key) != self._presentation:
                return key
        return None

    # -- the one behavioural change ---------------------------------------
    def write_key(self, key: str, value: str) -> str:
        cap = self._capacity()
        self._sync_presentation()

        if key in self._store or len(self._store) < cap:
            self._store[key] = str(value)
            self._refresh(key)
            return f"Key '{key}' written."

        victim = self._victim()
        if victim is None:
            # Within-presentation overflow: the selection cap. Byte-identical to
            # the baseline, including the message.
            return REFUSAL_TEMPLATE.format(cap=cap)

        del self._store[victim]
        if victim in self._order:
            self._order.remove(victim)
        self._laid.pop(victim, None)
        self._store[key] = str(value)
        self._refresh(key)
        return DISPLACED_TEMPLATE.format(key=key, victim=victim)

    def clear_key(self, key: str) -> str:
        out = super().clear_key(key)
        if key in self._order:
            self._order.remove(key)
        self._laid.pop(key, None)
        return out

    # -- read-out: deliberately NOT overridden -----------------------------
    #
    # `displacement` reordered `snapshot()` / `to_recall_text()` by recency and
    # tagged the last entry "(most recent)". Under this candidate's `step()` the
    # store is rendered into the user message on every turn, so that would be a
    # PROMPT change wearing an overflow change's clothes, and the prompt delta
    # here is zero. The baseline's plain insertion-order rendering is kept.


class WorkingMemoryAgent(_BaseAgent):
    """`episodic_reset_v3`'s agent with the episodic-displacement store installed.

    Two members differ from the live baseline: `__init__` (installs the store) and
    `step()` (copied verbatim from `episodic_reset_v3`). Everything else --
    `encode()`, `recall()`, `reset_messages()`, segmentation, key naming, value
    formatting, `_tool_call_cap()`, the prompts and the tools -- is the baseline's.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wm = EpisodicDisplacementMemory(owner=self)

    # ------------------------------------------------------------------
    # Everything from here to the end of the class is `episodic_reset_v3`.
    # ------------------------------------------------------------------
    def _episode_turn_index(self) -> int:
        """Ordinal of the current `step()` call within this episode, 1-based.

        Read-only: `_step_log` is appended by the base `step()` after the turn
        completes, so its length before the call is the number of completed turns.
        """
        return len(getattr(self, "_step_log", ())) + 1

    def _presentation_index(self, allow_tools: bool) -> int:
        """How many turns the agent has been permitted to encode on, including this
        one when it may.

        `_tool_interactions` is incremented by the base `step()` *after*
        `_ensure_messages()`, so reading it here gives the pre-increment count. It is
        read and never assigned, so `_tool_call_cap()` is unchanged.
        """
        return int(getattr(self, "_tool_interactions", 0)) + (1 if allow_tools else 0)

    def _control_state_block(self, allow_tools: bool, render_store: bool) -> str:
        lines = [
            EPISODE_MARKER,
            f"Turn {self._episode_turn_index()} of this episode. "
            f"Stimulus presentations so far: {self._presentation_index(allow_tools)}.",
            NO_TRANSCRIPT_NOTICE,
        ]
        if render_store:
            lines.append(WM_STATE_HEADER)
            lines.append(self.wm.to_recall_text())
        # Last, so the response obligation is the line adjacent to the stimulus.
        lines.append(STANDING_OBLIGATIONS)
        return "\n".join(lines)

    def step(self, user_message: str, *, allow_tools: bool = True,
             max_tokens: int = 1024) -> str:
        # Turn boundary. Doing this here and not inside the tool loop is load
        # bearing: clearing mid-loop would leave a `tool` message with no preceding
        # assistant `tool_calls` and the request would be rejected.
        self.reset_messages()

        # First step of an episode: no turn has elapsed, there is no store to report
        # and no transcript to disclaim, and the task's own first message carries its
        # instructions, so nothing is prepended. This is also what keeps the six
        # encode()->recall() tasks on the baseline's prompts, since encode() calls
        # step() exactly once.
        if self._episode_turn_index() > 1:
            block = self._control_state_block(
                allow_tools,
                render_store=WM_STATE_HEADER not in user_message,
            )
            user_message = f"{block}\n\n{user_message}"

        return super().step(user_message, allow_tools=allow_tools,
                            max_tokens=max_tokens)


MANIFEST: dict[str, Any] = {
    "id": "evicting_reset",
    "parent": "episodic_reset_v3",
    "capacity": _capacity(),
    "decay": (
        "none. No stochastic loss of any kind and no RNG in the file. Overflow only, "
        "and overflow is regime-dependent on ONE boundary: a write to a new key on a "
        "full store displaces the least recently refreshed resident laid down in an "
        "EARLIER presentation (a presentation being one step() call, read from "
        "len(_step_log)), and is refused with the baseline's exact message when every "
        "resident belongs to the CURRENT presentation. Deterministic, agent-controlled "
        "and reproducible; an overwrite counts as a re-presentation and refreshes."
    ),
    "role": (
        "contender -- iteration 5, clears episodic_reset_v3's single remaining floor "
        "violation (nback) by removing the across-presentation refusal loop, with a "
        "zero-character prompt delta"
    ),
    "summary": (
        "episodic_reset_v3 reached mean 0.8163 with exactly one floor violation, nback "
        "0.6978 against 0.7309, and its P1 precondition failed on merit (9.5 wanted at "
        "n=2, 7.98 got) so its variable_mapping +0.3213 and A4 505 errors are VOID from "
        "that run. THE SECOND CAUSE, from the v3 step_log: the loss is a PERIOD-2 "
        "OSCILLATION, not a diffuse rate. At n=3 the per-turn answer rate is EXACTLY "
        "0.00 on every even turn from turn 6 (50 blocks, temperature 0) and 0.66-1.00 on "
        "every odd turn; summing the n=2 per-turn rates over the 14 scored turns gives "
        "7.96 against the reported answered of 7.98, so the missing turns are one "
        "alternating half of the block and nothing else. The cycle: the agent overwrites "
        "one key (OK) and creates a second (REFUSED, and the refused call consumes its "
        "last budget unit), tools switch to tool_choice='none', and it emits the "
        "delete_key repair AS TEXT instead of the answer -- so the trial is unanswered "
        "AND THE DELETE NEVER EXECUTES, leaving the store full for the same thing two "
        "turns later. Self-sustaining, which is why it is periodic. A CORRECTION TO THE "
        "BRIEF'S CONDITIONING: the discriminator is the refusal, not the budget. "
        "Splitting the same turns by whether the turn contained a memory-full result, "
        "P(answer | budget=0, no refusal) = 0.990 (n=195) and 0.993 (n=276) against "
        "P(answer | budget=0, refusal) = 0.174 (n=363) and 0.134 (n=374). That also "
        "makes the _tool_call_cap() prohibition a measured result rather than a "
        "judgement: on the 471 exhausted-budget turns with no refusal the agent answered "
        "99.1% of the time, so more budget buys nothing on the failing cell. "
        "WHY NOT SIMPLY EVICT: measured, both validated evicting rules trade the nback "
        "violation for another one -- displacement (LRU) semantic_story_recall -0.0509, "
        "primacy (ACT-R) craft_task -0.0451 and narrative_qa -0.0309, against floors of "
        "0.030 -- and the craft mechanism is not the victim order but the occupancy: "
        "mean keys held on craft goes 3.33 refusing -> 3.67 under EITHER evicting rule, "
        "because a refusing store makes the agent burn tool calls on repairs that "
        "tool_calls[:remaining] truncates, so it ends holding fewer chunks. Admitting "
        "those writes hands a batch task more retained material, and the objective "
        "inverts, so every unconditional evicting store pays it. MECHANISM: one rule, "
        "no task named, no branch on task -- overflow is displacement ACROSS "
        "presentations and a selection cap WITHIN one. Across presentations a stimulus "
        "arrives whether or not there is room, the incoming chunk enters the focus of "
        "attention and the least active resident is displaced (Cowan 2001; "
        "overflow-as-overwriting, Oberauer & Kliegl 2006, Oberauer et al. 2016; removal "
        "in updating, Ecker, Lewandowsky & Oberauer 2014), and the victim is "
        "least-recently-refreshed because running-memory span and n-back show recency "
        "WITHOUT primacy (Pollack, Johnson & Knaff 1959; Bunting, Cowan & Saults 2006) "
        "-- so no activation machinery and no fitted constant. Within one presentation "
        "the material is still present and the order of encoding is the participant's; "
        "what binds is SELECTION, how many chunks may be carried forward (Miller 1956 "
        "recoding; Cowan 2001 chunk limit), and 'memory is full' is then not a refusal "
        "to perceive a stimulus but the report that the four-label index is spent, which "
        "the agent answers by overwriting or deleting. That is the only reading under "
        "which the baseline's string is admissible and it is kept exactly there and "
        "nowhere else. Consequence, not input: all six encode()->recall() tasks do every "
        "write inside ONE step(), so they are the baseline; nback and variable_mapping "
        "write across steps, so they displace. HOW episodic_primacy's 3.90 IS "
        "ACCOUNTED FOR, and a defect found doing it: (a) v2 dropped the RESPONSE "
        "obligation, so cause A caps the answer rate however the store behaves -- A is a "
        "ceiling, not an addend, and this is the first arm with both causes fixed; (b) "
        "episodic_primacy DID NOT CONTAIN primacy's rule. PrimacyMemory._episode_mark() "
        "counts role=='user' messages in owner._messages, which is 1 on EVERY turn under "
        "a step() that calls reset_messages(), so the mark never changes, "
        "_sync_episode() never advances, and every write in a block lands in one "
        "episode. Replaying the real n-back write sequences through PrimacyMemory with "
        "an advancing vs a frozen mark: final key sets differ on 18 of 50 blocks at n=2 "
        "and 17 of 50 at n=3, evictions 330 -> 201. The comment claiming monotonicity "
        "makes it 'safe to compose with a reset_messages() rewrite' is backwards -- "
        "monotonicity prevents collision, not a mark that never moves. This candidate "
        "reads the boundary from len(_step_log), which v3 already uses and which is "
        "invariant to history clearing. PROMPT DELTA: ZERO. Every string is v3's, "
        "character for character, and step() is copied verbatim -- the docstring argues "
        "no instruction CAN fix cause B, because the model is denied the tool at the "
        "moment it wants it. snapshot()/to_recall_text() deliberately NOT overridden, "
        "unlike displacement's, because the store is rendered into the prompt every turn "
        "and reordering it would be a prompt change. CAPACITY UNTOUCHED: MAX_KEYS 4 read "
        "live, at most 4 entries ever, _tool_call_cap() not overridden, recall() "
        "untouched. LEAK STAYS CLOSED by reset_messages(), upstream of the store (v3: "
        "n=3 letter identity 0.0202 -> 1.0000). n=1 IS THE BUILT-IN CONTROL: its store "
        "never fills (memory-full 0.00 per turn over 750 turns, keys_held 1.00), so a "
        "rule that changes only overflow cannot reach it. FALSIFIABLE: PRECONDITION "
        "pitched at restoration, answered >= 13.0/11.0/9.0 with n_no_answers == 0 at all "
        "three -- n=3 deliberately above the BASELINE's 6.82, since merely matching it "
        "would mean the loop is alive -- with 14/14/14 registered separately as a "
        "non-voiding strong claim; nback >= 0.7309; THE MECHANISM ROW, which separates "
        "'the fix engaged' from 'the score moved': memory_full results per turn EXACTLY "
        "0.00 at all three levels (v3 0.00/0.48/0.47) and P(answer | budget=0) >= 0.97 "
        "at all three, matching n=1's own 0.974 -- and zero_budget_share is REPORTED, "
        "NOT PREDICTED, because n=1 proves an exhausted budget is harmless and demand "
        "will still be ~1.5/turn; leak n=3 letter-identity share >= 0.90; vm >= 0.55 raw "
        "and >= 0.60 matched with A4 n_errors >= 150 and rc_ratio_normalized in "
        "[0.15,0.95]; parse integrity vm unparsed <= 5 and tool-call-in-text <= 5; "
        "ANTI-full_context, since that arm reaches nback 0.9484 by never filling the "
        "store (keys_held 1.06) -- n=3 keys_held >= 3.5 and n=2 >= 2.0, and "
        "acc_over_answered at n=3 <= 0.85 against full_context's 0.770 and the human "
        "0.866-at-pooled; and a batch-task control BANDED FROM IDENTICAL-PATH DELTAS, "
        "NOT from the recorded noise floor, which is the correction this candidate "
        "insists on: craft_task moved 0.0248 between the baseline and v3 on a code path "
        "that is provably identical (encode() calls step() once, no block is prepended "
        "on turn 1), and primacy vs primacy_v2 -- the same harness -- end with different "
        "story-recall stores on 200 of 200 rows, so the recorded craft noise of 0.0031 "
        "understates the real spread by 8x and a uniform +-0.008 band would void eleven "
        "rows for nondeterminism. Craft is pre-registered as a NAMED NON-VOIDING HAZARD "
        "for exactly that reason: its effective floor is 0.030 and v3 already sits at "
        "-0.0248 with the batch path untouched."
    ),
}
