"""Candidate `respond_first`: `evicting_reset`, with the response ordered BEFORE
the store update on every turn the harness grants a maintenance slot.

Parent: `evicting_reset` (iteration 5, Qwen mean 0.8460, held-out FAIL on nback
and on the A4 guard).

ONE change, and it is confined to `step()`. The store
(`EpisodicDisplacementMemory`) is copied from the parent character for character.
`MAX_KEYS`, `_tool_call_cap()`, `encode()`, `recall()`, `snapshot()`,
`to_recall_text()`, `reset_messages()`, `TOOLS` and `CONDITION_PROMPTS` are all
untouched.

    a turn that the harness permits tools on becomes TWO ACTS IN A FIXED ORDER
      ACT 1  respond.   tool_choice="none". The reply this act produces is the
                        turn's reply -- it is what `step()` returns and the only
                        thing `step_log["text"]` carries.
      ACT 2  maintain.  tools offered, drawing on the SAME cumulative budget the
                        parent had. Its text goes to `step_log["maintenance_text"]`
                        and never into `text`.

    a turn the harness does NOT permit tools on is left exactly as the parent has
    it: one act, tools withheld, the parent's own obligation wording. That is
    deliberate -- it is this candidate's internal control (see section 6).

Why ordering and not the tool-call cap: measured, in section 2. The brief's
account of route 3 -- "spends the floor of 6 by turn 4, loses that turn, then
recovers permanently" -- is contradicted in the two runs it cites. The failures
happen at caps of 7 through 22, throughout the block, so there is no early-turn
shape to reshape; only raising the per-turn RATE above the agent's ~1.43/turn
demand would reach them, and that is the capacity change the prohibition is
actually about.

===============================================================================
SECTION 1 -- WHAT THE RUN DATA SHOWS ROUTE 3 TO BE
===============================================================================
Verified from `step_log` in
`runs/iter5/evicting_reset/Qwen_Qwen3-30B-A3B-Instruct-2507` and
`runs/heldout/{baseline,episodic_reset_v3,evicting_reset}/NousResearch_Hermes-4-70B`.
Four corrections to the brief, in descending order of how much they change the
design.

(1) `budget=0` IS `tool_call_budget_AFTER`, AND IT IS NOT THE FAILING CELL.
    Conditioning on the budget ENTERING the turn, no turn that loses a trial on
    Hermes enters with 0. `tool_call_budget_before == 0` occurs on 0.000 of
    Hermes turns at every level. The failing cell is `b_in == 1`:

        run          lvl   P(unparsed | b_in=1)   P(unparsed | b_in>=2)   n at b_in=1
        herm evict    2           0.812                  0.000                112
        herm evict    3           0.766                  0.000                 47
        qwen evict    2           0.449                 <=0.010                69

    The mechanism is one step off what the brief describes. The agent enters with
    one call left, spends it, the loop re-enters with `_remaining_tool_calls()==0`,
    `step()` switches to `tool_choice="none"`, and the model -- which wanted a
    SECOND call -- emits it as plain text in place of the answer. Every one of the
    24 unparsed replies I sampled is a bare `<tool_call>{"name": "write_memory",
    ...}</tool_call>` on a turn with `b_in=1`, `calls=1`, `cap_hit=True`.

(2) IT IS NOT A TRANSIENT. Mean budget-entering, by turn index, Hermes
    `evicting_reset` n=2 (50 blocks):

        b_in  6.00 5.00 4.00 2.78 2.20 2.46 1.62 2.32 1.82 2.48 1.92 2.64 2.18 2.82 2.08 2.66
        calls 1.00 1.00 1.22 1.58 1.74 1.84 1.30 1.50 1.34 1.56 1.28 1.46 1.36 1.74 1.42 1.58

    It oscillates with period 2 around ~2 for the whole block and never recovers;
    the unparsed turns sit at trial indices 3,5,7,9,11,13 with caps 7,10,13,16,19,22.
    Consumption tracks the cap: at turn 15, cap 22 and used 21. On Qwen at n=2 the
    demand is lower (1.19/turn) so `b_in` does grow, 2.9 -> 6.3, and the loss is
    small; at n=3 the demand is 1.03 and the loss is exactly zero. So route 3's
    size is set by the MARGIN between demand and the 1.5/turn allowance, not by a
    start-up transient:

        per-block demand    blocks   unparsed/block   corr(demand, unparsed)
        herm n=2  [1.3,1.6)    47          1.91                0.497
        herm n=3  [1.05,1.3)   27          0.04                0.714
        herm n=3  [1.3,1.6)    23          1.52
        qwen n=2  [0,1.05)     27          0.00                0.604
        qwen n=2  [1.3,1.6)    19          1.68

(3) ROUTE 3 IS NOT THE LARGEST N-BACK LOSS CHANNEL ON HERMES, AND FIXING IT
    ALONE CANNOT CLEAR THE FLOOR. `answered` counts trial-period replies that
    parse to Same/Different. A turn is lost two ways, scored identically and
    caused differently:

        UNPARSED  `_parse_classification` returns None  -- route 3
        NORESP    it returns "No response" during the TRIAL period -- call it
                  route 4; a position/task-set failure, not a tool denial

        run          lvl  answered  lost  UNPARSED  NORESP
        herm base     2     12.78   1.22    0.00     1.22
        herm base     3     11.42   2.58    0.00     2.58
        herm evict    2      8.88   5.12    1.82     3.30
        herm evict    3      9.72   4.28    0.72     3.56
        qwen evict    2     13.34   0.66    0.66     0.00
        qwen evict    3     14.00   0.00    0.00     0.00

    Crediting recovered turns at each block's own `acc_over_answered` (an upper
    bound -- the recovered turns are the high-maintenance ones), and recomputing
    nback humanlikeness:

        Hermes (baseline 0.8724, floor 0.8124)      Qwen (baseline 0.7909)
          as-run              0.7366  -0.1358         as-run       0.9587  +0.1678
          + route 3           0.7875  -0.0849  FAIL   + route 3    0.9634  +0.1725
          + route 3 + 4       0.8895  +0.0171  PASS   (route 4 is 0 on Qwen)

    Discounting the credited accuracy: route 3 + 4 clears the floor at 1.00x
    (+0.0171), 0.75x (-0.0181) and 0.50x (-0.0535) and fails at 0.25x (-0.0904);
    route 3 alone fails at every discount. So the brief's "route 3 is the only
    thing standing between `evicting_reset` and a candidate that passes on both
    substrates" is false by about 0.10 of humanlikeness, and a candidate that
    fixes route 3 only is a predictable held-out failure.

    The two channels are one pathology, which is what makes a single mechanism
    available: both rise with per-block maintenance demand. corr(demand, noresp)
    = 0.334 (herm n=2) and 0.550 (herm n=3) beside corr(demand, unparsed) = 0.497
    and 0.714. Route 3 is the turn where maintenance is cut off and the repair is
    spoken; route 4 is the turn where maintenance completes and the response is an
    afterthought -- 5 of 6 sampled mid-block NORESP turns made exactly two calls
    (a delete and a write) and then said "no response", one of them adding
    "because it is the first two letters of the sequence" on trial 11. And route 4
    is not a per-participant quirk that eviction inherited: Hermes' baseline
    NORESP histogram is bimodal, {0: 41, 1: 5, 14: 4} at n=2 -- four blocks refuse
    the whole way and forty-one are clean -- while under `evicting_reset` NO block
    is clean and every block loses 1 to 9 turns. The reset harness converted a
    rare catastrophic failure into a pervasive per-turn one.

(4) A DEFECT THAT MAKES EVERY EPISODIC RUN'S PROMPT DELTA TWICE WHAT IT WAS
    THOUGHT TO BE. `run_candidate.py` calls `verify_interface.check()`, which
    itself does `load_candidate` + `apply`, and then calls `load_candidate` +
    `apply` again. The second `load_candidate` re-executes the candidate module,
    whose `from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent`
    now resolves to the ALREADY-INJECTED first copy. Reproduced:

        after verify:            mh_candidate.WorkingMemoryAgent      (MRO 3)
        after the second apply:  mh_candidate.WorkingMemoryAgent
                                 mh_candidate.WorkingMemoryAgent
                                 bench.core.wm_agent.WorkingMemoryAgent

    So `step()` ran twice per turn and the control-state block was prepended
    twice. Counted over the recorded runs: 2400 of 2400 n-back turns carry
    `[ongoing episode]` TWICE in `episodic_reset_v3` and `evicting_reset`, on both
    substrates; the baseline carries it zero times. The outer copy's
    `render_store=WM_STATE_HEADER not in user_message` test then suppressed the
    store in one copy and not the other, so what the agent actually read was the
    block once without the store and once with, and two counters -- "Turn 5 of
    this episode" and "Stimulus presentations so far: 4" -- stated twice each.

    This candidate is immune by construction rather than by a guard: no code path
    calls the candidate's own `step()` recursively. Turn 1 goes to the genuine
    baseline `step()`, located by walking `type(self).__mro__` for the first class
    whose `__module__` is `bench.core.wm_agent`, which skips every stacked copy;
    later turns run `_ordered_step`/`_probe_step`, which build the block once and
    never delegate upward. A stacked duplicate's `step()` is simply never called.

    Consequence for attribution, stated rather than buried: this candidate's
    prompt differs from the parent's REALISED prompt in two ways, the ordering
    (the mechanism) and the de-duplication (a defect repair that any correct
    implementation performs). `logs/pending_respond_first.json` therefore names a
    required 14-minute companion arm -- `evicting_reset` run with the injection
    applied once -- without which the mechanism rows carry the attribution and
    the score does not.

===============================================================================
SECTION 2 -- THE MECHANISM, AND WHY NOT THE CAP
===============================================================================
WHY NOT CAP-RESHAPING, decided on the measurement and not on the prohibition.
The brief offers a reshape that does not raise the total: front-load the early
allowance. Section 1(2) rules it out on this data. The failing turns are at trial
indices 3 to 13 with caps of 7 to 22 and the residual budget oscillating
persistently around 2 -- there is no early-turn deficit to move, because
consumption tracks whatever the cap supplies. To reach the `b_in == 1` cell one
would have to lift the per-turn RATE above the ~1.43/turn demand Hermes exhibits,
i.e. from 1.5 to >=2.0. That is exactly the capacity change the prohibition
protects `variable_mapping` from, and it would also merely relocate the margin:
the same blocks that delete-then-write would consume 2.0/turn against a 2.0/turn
supply and sit back on the barrier.

Two further reasons, one empirical and one about what a budget IS.

  * Cap-reshaping cannot touch route 4 at all, and route 4 is the larger half of
    the Hermes deficit (section 1(3)). A mechanism that addresses one of two
    channels is a mechanism that fails held-out on arithmetic already in hand.
  * A fixed per-turn action budget has no counterpart in human memory. Nothing
    stops a person rehearsing twice between two stimuli; what limits them is
    capacity and time, both of which this harness models elsewhere (MAX_KEYS 4;
    one presentation per turn). If the cap is a harness artifact -- and it is: it
    exists to bound API spend -- then the right response to an artifact is to
    stop routing the RESPONSE through it, not to re-tune it.

WHAT ORDERING DOES, mechanically. `step()` accumulates text across the tool loop
but the parent's turn produces its classification in the LAST assistant message,
after the maintenance is done. So the classification is downstream of every way
maintenance can go wrong. Under ACT 1 / ACT 2 the classification is produced in
its own model call, before any tool exists to be denied, and is the only text the
turn returns. Route 3 then cannot cost a trial even when it happens: a spoken
tool call in ACT 2 lands in `maintenance_text` and the trial is already scored.
Route 4 is addressed by the same move if the unified account in section 1(3) is
right -- the response is no longer an afterthought to a delete/write narrative --
and that is the candidate's real bet, stated as such in the predictions.

WHAT IS DELIBERATELY NOT DONE. Probe turns (`allow_tools=False`) are left exactly
as the parent has them: one withheld-tools act, the parent's obligation wording,
no maintenance slot. Granting them a slot would fix route 2 as well (section 6),
and the measurements say it would help, but it is a SECOND behavioural change --
on tool-permitted turns an existing act moves, on probe turns an act appears --
and with both in one arm neither would be attributable. Kept as the next
iteration's mechanism, with the full measurement in section 6 so it can be
proposed without re-deriving it.

===============================================================================
SECTION 3 -- THE PSYCHOLOGICAL ARGUMENT
===============================================================================
The claim is about ORDER, not about capacity or about how much may be stored.

A participant in an n-back block does not withhold the response until the
rehearsal is finished. The response is a speeded decision on the current
stimulus, made against the contents of the focus of attention; updating -- the
removal of the item that has fallen out of the window and the encoding of the
new one -- is what happens next, in the inter-stimulus interval.

  * Response selection and memory updating are separable operations with
    separable costs. Oberauer (2002, 2009) and Ecker, Lewandowsky & Oberauer
    (2014) isolate REMOVAL as a distinct, time-consuming component of updating
    that follows retrieval-and-response rather than preceding it; Kessler &
    Meiran (2008) show the content-updating cost is paid after the current
    comparison, not before.
  * The focus of attention holds the item the response is about (Cowan 2001;
    Oberauer 2002), so the comparison is available BEFORE any store operation.
    Requiring the store to be reorganised first inverts the dependency.
  * Rehearsal and maintenance occupy the inter-stimulus interval, not the
    response window (Baddeley 1986 on the articulatory loop; Barrouillet,
    Bernardin & Camos 2004's time-based resource sharing, in which the cognitive
    load of a task is the proportion of the free time between stimuli that
    processing consumes -- free time being, by construction, after the response).
  * Running-memory and n-back tasks show recency without primacy (Pollack,
    Johnson & Knaff 1959; Bunting, Cowan & Saults 2006), which is why the
    parent's displacement rule is left untouched: this candidate changes when the
    response is emitted, not what is forgotten.

The converse artifact is worth naming. A harness that requires maintenance first
and takes the answer afterwards models a participant who will not speak until the
filing is done -- which is not a memory limitation but a turn-taking one, and it
is the single largest source of lost trials in every episodic run this project
has produced.

===============================================================================
SECTION 4 -- PROMPT DELTA, AND WHY EACH PIECE IS FORCED
===============================================================================
Against the parent's INTENDED prompt (one control-state block), the delta is two
strings and nothing else.

  1. `ORDERED_OBLIGATIONS` replaces `STANDING_OBLIGATIONS` on tool-permitted
     turns only. The parent's sentence is "Update the store to track what you
     will need later, and give the response this turn's instructions call for" --
     which names the store update FIRST. A structural reordering that left the
     instruction saying the opposite would be testing two things against each
     other. The new text states the order, and states that the maintenance slot
     is coming, because ACT 1 is a `tool_choice="none"` call and the whole bet is
     that a withheld-tools call which PROMISES a slot behaves differently from one
     that denies tools outright. Without the promise the candidate is asking the
     model to do the thing it demonstrably does wrong.
  2. `MAINTENANCE_CUE` opens ACT 2. It is required: ACT 2 is a new user turn and
     something has to occupy it. It deliberately contains none of "same",
     "different" or "no response", so that no echo of it could reach a
     classification parser even by accident.

On probe turns the wording is the parent's, character for character, and the
first turn of every episode is not prepended at all -- so the six
`encode()` -> `recall()` tasks see the baseline's prompt exactly, as under the
parent.

Against the parent's REALISED prompt there is a third difference, the
de-duplication, which is a defect repair and is handled in section 1(4).

===============================================================================
SECTION 5 -- WHY THE LEAK STAYS CLOSED AND `variable_mapping` IS NOT TRADED AWAY
===============================================================================
The leak. `reset_messages()` is called at the top of every turn of every path --
`_ordered_step`, `_probe_step`, and the baseline `step()` on turn 1, which calls
it never because there is no history to clear on the first turn of an episode.
It is upstream of everything this candidate changes: ACT 1 and ACT 2 share the
message list that was just cleared, and neither can see an earlier turn.
Concretely, the n=3 store carries letter identity at 1.0000 against the
baseline's 0.0202 because the store is the only channel, and nothing here adds a
channel.

`variable_mapping` is untouched by construction on the side that produces its
score. Its probe turns -- the only turns whose text is parsed into an answer --
take `_probe_step`, which is the parent's path with the parent's wording. Its
encode turns become two-act, so the agent emits an acknowledgement before
writing; the task discards the return value of those steps
(`wm_variable_mapping.py:148`), so the acknowledgement cannot enter any score.
The store it builds can differ, which is why `variable_mapping` raw, matched, A4
`n_errors` and `rc_ratio_normalized` are all pre-registered as HOLD rows rather
than assumed.

Capacity is not increased anywhere. `MAX_KEYS` is 4, read live; the store class is
the parent's; `_tool_call_cap()` is not overridden; `_tool_interactions` is
incremented exactly once per tool-permitted turn, as the baseline does, so the
cap sequence over a block is identical to the parent's. Probe turns do not
increment it and do not consume from it. What changes is only WHEN within a turn
the budget is spent relative to the response.

===============================================================================
SECTION 6 -- WHAT I PREDICT DIFFERS BETWEEN QWEN AND HERMES
===============================================================================
Three things differ, and all three are measured rather than guessed.

(a) ROUTE 3's SIZE. Qwen's demand is 1.03-1.19 calls/turn and Hermes' is
    1.28-1.47 against the same 1.5 supply, so Qwen touches `b_in == 1` on 69
    turns of 800 and Hermes on 112 of 800 (n=2). The fix therefore recovers
    0.66 turns/block on Qwen and 1.82 on Hermes at n=2.

    The demand difference is a strategy difference, and my first hypothesis about
    it was wrong in direction, so the corrected version: blocks that use FEW
    distinct keys carry MORE deletes, not fewer (9.7 vs 7.5 per block at Hermes
    n=2) and higher demand (1.47 vs 1.41), because they delete a rolling key and
    rewrite it. Qwen issues 0.0-0.1 deletes per block and lets the displacing
    store free the slot; Hermes issues 5-10. Under a displacing store a
    delete-then-write is REDUNDANT -- the write alone would have evicted -- so
    Hermes is paying two calls for what the store gives for one, and that is what
    puts it on the budget margin. The corollary is a clean future candidate:
    `DISPLACED_TEMPLATE` tells the agent "the least recently used entry X was
    displaced and is now lost", which may be teaching it to delete first in order
    to CHOOSE the victim. That is a one-string mechanism and a better next move
    than any cap change.

(b) ROUTE 4 EXISTS ONLY ON HERMES. Zero NORESP turns on Qwen at either level;
    3.30 and 3.56 per block on Hermes. So the ordering's effect on route 4 -- the
    part of this candidate that is a bet rather than a construction -- is
    testable only on the held-out substrate. On Qwen the candidate can do no
    better than +0.005 on nback over the parent, and that is the honest ceiling
    there.

(c) THE A4 GUARD WILL FIRE ON HERMES AND NOT ON QWEN, AND I CAN SAY WHY.
    The brief asks whether A4 is measuring something different on Hermes. It is
    not. It is measuring a failure the brief records as fixed. Route 2 --
    `allow_tools=False` on a question turn -- is WIDE OPEN on Hermes: 549 of 1500
    `variable_mapping` answers do not parse, and the raw replies are
    `<tool_call>{"name": "write_memory", ...}</tool_call>`. On Qwen the same count
    is 1. Those 549 sit at `relation_count` 6.22 against 8.43 on correct answers
    -- they are concentrated on EARLY questions -- so they drag
    `rc_mean_error` below `rc_mean_correct` and the normalized ratio below zero.
    Recomputing A4 on Hermes with only the parsed trials:

        as-run              n_err 1136  rc_ratio 0.9320  ceiling 2.6742  norm -0.0406
        unparsed dropped    n_err  587  rc_ratio 1.1142  ceiling 1.3402  norm  0.3356
        humans                                                           norm  0.3728

    Hermes' parsed error structure is human-shaped to within 0.04 of the human
    value. The axis is fine; the parse failure is not. And the vm score is not
    what protects it: simulating recovery of those 549 at recovery probability p,
    raw humanlikeness goes 0.6968 (p=0) -> 0.7862 (p=0.383, the observed parsed
    accuracy) -> 0.9362 (p=0.7) -> 0.7280 (p=1.0), i.e. UP at every rate, while
    Qwen is unmoved at 0.6850 throughout. So fixing route 2 is predicted to
    improve `variable_mapping` on Hermes and leave it alone on Qwen. This
    candidate does not do it, for the attribution reason in section 2, and its
    Hermes A4 row is pre-registered as an expected-and-explained violation.

    The internal control that tests this account in THIS run: Hermes' nback ACT 1
    is a withheld-tools call that promises a maintenance slot; Hermes' vm probe
    turn is a withheld-tools call that does not. If the promise is what matters,
    nback ACT-1 unparsed collapses while vm unparsed stays near 549. If vm
    unparsed falls WITHOUT a probe slot, my account of route 2 is wrong.
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
from bench.core.wm_agent import _dispatch_tool
from bench.core.working_memory import MAX_KEYS, WorkingMemory as _BaseMemory

# The module the GENUINE baseline agent is defined in. Used to find the real
# `step()` through `type(self).__mro__`, so a stacked duplicate of this class
# (section 1(4)) cannot wrap the untouched path twice.
_BASE_MODULE = "bench.core.wm_agent"

# ---------------------------------------------------------------------------
# The control-state block. Every string below except ORDERED_OBLIGATIONS and
# MAINTENANCE_CUE is `evicting_reset`'s, character for character.
# ---------------------------------------------------------------------------
WM_STATE_HEADER = "Your working memory currently contains:"

EPISODE_MARKER = "[ongoing episode]"

NO_TRANSCRIPT_NOTICE = (
    "You have no transcript of earlier turns: the key-value store is your only "
    "record of them."
)

# The parent's wording, kept for the turns where the parent's structure is kept:
# probe turns, which get no maintenance slot, so promising one would be false.
STANDING_OBLIGATIONS = (
    "Two things are due on every turn of an episode. Update the store to track what "
    "you will need later, and give the response this turn's instructions call for. "
    "The response is due either way: if the store cannot be changed, or needs no "
    "change, respond anyway."
)

# The one new obligation string, used only on turns that DO get a maintenance
# slot. It states the order the structure enforces, and it promises the slot --
# which is the hypothesis under test (section 4, item 1).
ORDERED_OBLIGATIONS = (
    "Two things are due on every turn of an episode, and they come in this order. "
    "First give the response this turn's instructions call for. Then, once you have "
    "responded, you will be asked to update the store to track what you will need "
    "later; the store cannot be changed until then, so nothing is lost by responding "
    "first. The response comes first and is due either way."
)

# ACT 2's opening message. Contains none of "same", "different" or "no response",
# so that an echo of it could not reach a classification parser.
MAINTENANCE_CUE = (
    "You have responded. Now update the store for the turns ahead: write, overwrite "
    "or delete keys as needed, or leave it as it is. Do not restate your response."
)

# The baseline's overflow message, byte-identical, so the within-presentation path
# is the baseline's and not merely similar.
REFUSAL_TEMPLATE = (
    "Error: memory is full ({cap} keys). "
    "Delete an existing key first or overwrite one."
)

# `displacement`'s result string, also byte-identical.
DISPLACED_TEMPLATE = (
    "Key '{key}' written. Memory was full, so the least recently "
    "used entry '{victim}' was displaced and is now lost."
)


def _capacity() -> int:
    """Read capacity live, so an injected `MAX_KEYS` is honoured."""
    return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))


class EpisodicDisplacementMemory(_BaseMemory):
    """`evicting_reset`'s store, copied verbatim. Nothing here is changed.

    A write to a NEW key on a full store displaces the least recently refreshed
    resident laid down in an EARLIER presentation, and is refused with the
    baseline's exact message when every resident belongs to the CURRENT
    presentation. A presentation is one `step()` call, read from
    ``len(owner._step_log)``, which is invariant to `reset_messages()`.

    With no owner (a bare instantiation, which `verify_interface.py` performs) the
    mark is ``None`` forever, every write falls in one presentation, and the store
    is exactly the baseline -- the conservative direction.
    """

    def __init__(self, owner: Any = None) -> None:
        super().__init__()
        # Read-only back-reference, used ONLY to observe presentation boundaries.
        self._owner = owner
        self._order: List[str] = []        # least recently refreshed first
        self._laid: Dict[str, int] = {}    # key -> presentation ordinal
        self._mark: Any = object()
        self._presentation = 0

    def _capacity(self) -> int:
        return _capacity()

    def _presentation_mark(self) -> Any:
        log = getattr(self._owner, "_step_log", None)
        if not isinstance(log, list):
            return None
        return len(log)

    def _sync_presentation(self) -> None:
        mark = self._presentation_mark()
        if mark != self._mark:
            self._mark = mark
            self._presentation += 1

    def _refresh(self, key: str) -> None:
        """An overwrite is a re-presentation: it refreshes recency AND re-stamps
        the presentation, so a chunk written in this turn is not a candidate for
        displacement by its own next write in the same turn."""
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self._laid[key] = self._presentation

    def _victim(self) -> str | None:
        """Least recently refreshed resident from an EARLIER presentation."""
        for key in self._order:
            if key in self._store and self._laid.get(key) != self._presentation:
                return key
        for key in self._store:
            if self._laid.get(key) != self._presentation:
                return key
        return None

    def write_key(self, key: str, value: str) -> str:
        cap = self._capacity()
        self._sync_presentation()

        if key in self._store or len(self._store) < cap:
            self._store[key] = str(value)
            self._refresh(key)
            return f"Key '{key}' written."

        victim = self._victim()
        if victim is None:
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

    # -- read-out: deliberately NOT overridden, as in the parent. The store is
    # rendered into the user message every turn, so reordering it would be a
    # prompt change wearing an overflow change's clothes.


class WorkingMemoryAgent(_BaseAgent):
    """`evicting_reset`'s agent with the response ordered before the store update.

    Three members differ from the parent: `__init__` (unchanged in effect --
    installs the same store), `step()` (the dispatch between three paths) and the
    two new path implementations. `encode()`, `recall()`, `reset_messages()`,
    `_tool_call_cap()`, segmentation, key naming, value formatting, the prompts
    and the tools are all the baseline's.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wm = EpisodicDisplacementMemory(owner=self)

    # ------------------------------------------------------------------
    # Control-state block. `_episode_turn_index` and `_presentation_index` are
    # `episodic_reset_v3`'s, verbatim.
    # ------------------------------------------------------------------
    def _episode_turn_index(self) -> int:
        """Ordinal of the current `step()` call within this episode, 1-based."""
        return len(getattr(self, "_step_log", ())) + 1

    def _presentation_index(self, allow_tools: bool) -> int:
        """How many turns the agent has been permitted to encode on, including
        this one when it may. `_tool_interactions` is read and never assigned
        here, so `_tool_call_cap()` is unchanged."""
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
        # Last, so the obligation is the line adjacent to the stimulus. The
        # ordered wording is used only where the maintenance slot it promises
        # actually exists.
        lines.append(ORDERED_OBLIGATIONS if allow_tools else STANDING_OBLIGATIONS)
        return "\n".join(lines)

    def _prepend_block(self, user_message: str, allow_tools: bool) -> str:
        block = self._control_state_block(
            allow_tools,
            render_store=WM_STATE_HEADER not in user_message,
        )
        return f"{block}\n\n{user_message}"

    # ------------------------------------------------------------------
    # The genuine baseline `step()`, reached past any stacked duplicate.
    # ------------------------------------------------------------------
    def _base_step(self):
        """`bench.core.wm_agent.WorkingMemoryAgent.step`, found by MRO.

        `_BaseAgent` is NOT safe to use for this: when `inject.apply()` runs twice
        (see the module docstring, section 1(4)) the second module execution binds
        `_BaseAgent` to the first injected copy, so `_BaseAgent.step` would be this
        candidate's own `step` and the turn would be built twice. Walking
        `type(self).__mro__` for the defining module is invariant to how many
        copies have been stacked.
        """
        for cls in type(self).__mro__:
            if cls.__module__ == _BASE_MODULE and "step" in cls.__dict__:
                return cls.__dict__["step"]
        return _BaseAgent.step  # pragma: no cover -- bench moved; fail loudly later

    # ------------------------------------------------------------------
    # step(): three paths, none of which re-enters this class's step().
    # ------------------------------------------------------------------
    def step(self, user_message: str, *, allow_tools: bool = True,
             max_tokens: int = 1024) -> str:
        base_step = self._base_step()

        # Turn 1 of an episode. No turn has elapsed, there is no store to report
        # and no transcript to disclaim, and the task's own first message carries
        # its instructions, so nothing is prepended and the baseline's own step()
        # runs. This is what keeps the six encode()->recall() tasks on the
        # baseline's prompts and its exact code path: encode() calls step() once.
        if self._episode_turn_index() <= 1:
            return base_step(self, user_message, allow_tools=allow_tools,
                             max_tokens=max_tokens)

        if not allow_tools:
            # PROBE TURN. The parent's path, with the parent's wording: one act,
            # tools withheld, no maintenance slot. Deliberately unchanged -- it is
            # this candidate's internal control (docstring, section 6(c)).
            self.reset_messages()
            return base_step(self, self._prepend_block(user_message, False),
                             allow_tools=False, max_tokens=max_tokens)

        return self._ordered_step(user_message, max_tokens)

    # ------------------------------------------------------------------
    # The mechanism.
    # ------------------------------------------------------------------
    def _ordered_step(self, user_message: str, max_tokens: int) -> str:
        """ACT 1 respond (tools withheld), then ACT 2 maintain (tools offered).

        Bookkeeping mirrors the baseline `step()` exactly: `_tool_interactions` is
        incremented once, `_tool_calls_used` once per dispatched call, and
        `_tool_call_cap()` is not touched -- so the cumulative budget over a block
        is identical to the parent's.
        """
        # Turn boundary, as the parent. Done here and not inside the tool loop:
        # clearing mid-loop would leave a `tool` message with no preceding
        # assistant `tool_calls` and the request would be rejected.
        self.reset_messages()
        self._ensure_messages()
        self._tool_interactions += 1

        prompt = self._prepend_block(user_message, True)
        self._messages.append({"role": "user", "content": prompt})

        cap_before = self._tool_call_cap()
        budget_before = self._remaining_tool_calls()

        # -- ACT 1: the response -------------------------------------------
        r1 = self.llm.generate_with_tools(
            messages=self._messages,
            tools=TOOLS,
            tool_choice="none",
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        response_text = r1.content or ""
        self._messages.append({"role": "assistant", "content": response_text})

        # -- ACT 2: maintenance, on the parent's budget ---------------------
        self._messages.append({"role": "user", "content": MAINTENANCE_CUE})
        tool_calls_log: List[Dict[str, Any]] = []
        maintenance_text = ""
        cap_hit = False

        while True:
            tools_available = self._remaining_tool_calls() > 0
            if not tools_available:
                cap_hit = True
            r2 = self.llm.generate_with_tools(
                messages=self._messages,
                tools=TOOLS,
                tool_choice="auto" if tools_available else "none",
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            tool_calls = r2.tool_calls or []
            if tool_calls:
                remaining = self._remaining_tool_calls()
                if len(tool_calls) > remaining:
                    cap_hit = True
                    tool_calls = tool_calls[:remaining]

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if r2.content:
                assistant_msg["content"] = r2.content
                maintenance_text += r2.content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg.setdefault("content", "")
            self._messages.append(assistant_msg)

            if not tool_calls:
                break

            for tc in tool_calls:
                result = _dispatch_tool(self.wm, tc["function"]["name"],
                                        tc["function"]["arguments"])
                self._tool_calls_used += 1
                tool_calls_log.append({
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "result": result,
                })
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        # `text` carries ACT 1 ONLY. Keeping ACT 2's text out of it is load
        # bearing, not tidiness: `_parse_classification` tests "no response"
        # before "different", so a leaked maintenance string could invert a
        # correct answer.
        self._step_log.append({
            "user_message": prompt,
            "tool_calls": tool_calls_log,
            "text": response_text,
            "maintenance_text": maintenance_text,
            "kv_snapshot": self.wm.store,
            "tool_call_cap": cap_before,
            "tool_calls_used_total": self._tool_calls_used,
            "tool_call_budget_before": budget_before,
            "tool_call_budget_after": self._remaining_tool_calls(),
            "tool_call_cap_hit": cap_hit,
            "ordered_turn": True,
        })

        if self.debug:
            print(f"  ACT1 -> {response_text[:80]!r}")
            for tc in tool_calls_log:
                print(f"  ACT2 {tc['name']}({tc['arguments']}) -> {tc['result'][:60]}")
            print(f"  Memory:\n{self.wm.snapshot()}")

        return response_text


MANIFEST: dict[str, Any] = {
    "id": "respond_first",
    "parent": "evicting_reset",
    "capacity": _capacity(),
    "decay": (
        "none. The parent's store, copied verbatim: no stochastic loss, no RNG. "
        "Overflow only, regime-dependent on ONE boundary -- a write to a new key on "
        "a full store displaces the least recently refreshed resident laid down in "
        "an EARLIER presentation (a presentation being one step() call, read from "
        "len(_step_log)), and is refused with the baseline's exact message when "
        "every resident belongs to the CURRENT presentation. Deterministic, "
        "agent-controlled, reproducible; an overwrite counts as a re-presentation."
    ),
    "role": (
        "contender -- iteration 6, orders the response BEFORE the store update on "
        "every turn the harness grants a maintenance slot, so no lost tool call can "
        "cost a classification; targets the n-back floor violation that fails "
        "evicting_reset on the held-out substrate"
    ),
    "summary": (
        "ONE change, confined to step(): a tool-permitted turn after the first of an "
        "episode becomes ACT 1 respond (tool_choice='none', and this act's reply is "
        "the turn's reply and the only thing step_log['text'] carries) then ACT 2 "
        "maintain (tools offered, same cumulative budget). Probe turns "
        "(allow_tools=False) and the first turn of every episode are the parent's "
        "paths untouched, the latter reached through the GENUINE baseline step() "
        "found by MRO walk. FOUR CORRECTIONS TO THE BRIEF, all from step_log. (1) "
        "'budget=0' is tool_call_budget_AFTER and is not the failing cell: "
        "budget_before==0 occurs on 0.000 of Hermes turns, and the failing cell is "
        "b_in==1, where P(unparsed)=0.812/0.766 on Hermes n=2/n=3 and 0.449 on Qwen "
        "n=2 against <=0.010 at b_in>=2. The agent enters with one call left, spends "
        "it, the loop re-enters with zero remaining, tools switch to "
        "tool_choice='none', and the model speaks the SECOND call it wanted. (2) NOT "
        "A TRANSIENT: mean b_in by turn on Hermes n=2 is 6.00 5.00 4.00 2.78 2.20 "
        "2.46 1.62 2.32 1.82 2.48 1.92 2.64 2.18 2.82 2.08 2.66 -- a persistent "
        "period-2 oscillation around 2, with the unparsed turns at caps 7 through "
        "22, and consumption tracking the cap (turn 15: cap 22, used 21). So there "
        "is no early-turn shape to reshape; only lifting the RATE above the "
        "~1.43/turn demand reaches it, which is the forbidden capacity change and "
        "would merely relocate the margin. Route 3's size is set by the demand-supply "
        "margin: corr(per-block demand, unparsed) 0.497/0.714/0.604. (3) ROUTE 3 IS "
        "NOT THE LARGEST N-BACK CHANNEL ON HERMES AND CANNOT CLEAR THE FLOOR ALONE. "
        "Decomposing the answered deficit into UNPARSED (route 3) and 'No response' "
        "in the trial period (call it route 4): herm evict n=2 lost 5.12 = 1.82 + "
        "3.30, n=3 lost 4.28 = 0.72 + 3.56, against a Hermes BASELINE 1.22 and 2.58 "
        "all of which is route 4. Crediting recovered turns at each block's own "
        "acc_over_answered, nback humanlikeness goes 0.7366 -> 0.7875 (-0.0849, "
        "STILL FAILS the -0.060 floor) with route 3 alone and -> 0.8895 (+0.0171, "
        "passes) with both; discounted to 0.75x and 0.50x the credited accuracy the "
        "both-channels figure holds (-0.0181, -0.0535) and fails only at 0.25x, "
        "while route 3 alone fails at every discount. The two channels are ONE "
        "pathology -- corr(demand, noresp) 0.334/0.550 beside corr(demand, unparsed) "
        "0.497/0.714, and 5 of 6 sampled mid-block noresp turns made a delete and a "
        "write and then said 'no response' -- which is what makes a single mechanism "
        "available. Hermes' baseline noresp histogram is bimodal {0:41, 1:5, 14:4}; "
        "under evicting_reset no block is clean, so the reset harness converted a "
        "rare catastrophic failure into a pervasive per-turn one. (4) A DEFECT: "
        "run_candidate.py calls verify_interface.check(), which applies the "
        "candidate, then applies it again; the second load_candidate re-executes the "
        "module and binds _BaseAgent to the ALREADY-INJECTED copy, so the class "
        "stacks (MRO: mh_candidate, mh_candidate, bench.core.wm_agent) and step() "
        "ran TWICE per turn. 2400 of 2400 n-back turns in episodic_reset_v3 and "
        "evicting_reset carry '[ongoing episode]' twice, on both substrates, the "
        "baseline zero times -- so every episodic run's prompt delta was twice what "
        "it was thought to be, with the store rendered in one copy and not the "
        "other. This candidate is immune by construction: no path re-enters its own "
        "step(). MECHANISM: step() accumulates text across the tool loop but the "
        "parent produces its classification in the LAST assistant message, "
        "downstream of everything maintenance can do wrong. Ordered, the "
        "classification exists before any tool can be denied, so a spoken tool call "
        "in ACT 2 lands in maintenance_text and the trial is already scored. WHY NOT "
        "THE CAP: ruled out on the data above, plus it cannot touch route 4 at all, "
        "plus a fixed per-turn action budget has no human counterpart -- it bounds "
        "API spend -- and the response is what should stop being routed through it. "
        "PSYCHOLOGY: response selection and memory updating are separable, and "
        "removal follows the response rather than preceding it (Oberauer 2002, 2009; "
        "Ecker, Lewandowsky & Oberauer 2014; Kessler & Meiran 2008); the focus of "
        "attention already holds the item the comparison is about (Cowan 2001); "
        "rehearsal occupies the inter-stimulus interval, not the response window "
        "(Baddeley 1986; Barrouillet et al. 2004); n-back shows recency without "
        "primacy, so the displacement rule is left alone (Pollack et al. 1959; "
        "Bunting et al. 2006). PROMPT DELTA: two strings. ORDERED_OBLIGATIONS "
        "replaces the parent's obligation sentence on tool-permitted turns ONLY -- "
        "the parent's own wording names the store update first, so leaving it would "
        "test the instruction against the structure -- and MAINTENANCE_CUE opens ACT "
        "2, containing none of 'same', 'different' or 'no response' so no echo can "
        "reach a parser. Probe turns keep the parent's wording character for "
        "character. LEAK STAYS CLOSED: reset_messages() at the top of every path, "
        "upstream of everything changed; ACT 1 and ACT 2 share the just-cleared "
        "message list. vm's SCORED side is untouched -- its probe turns are the "
        "parent's path, and the task discards encode-step return values. CAPACITY "
        "AND BUDGET UNCHANGED: MAX_KEYS 4 read live, _tool_call_cap() not "
        "overridden, _tool_interactions incremented exactly once per tool-permitted "
        "turn so the cap sequence is identical, probe turns neither increment nor "
        "consume. A4 ON HERMES, answering the brief's open question: the axis is NOT "
        "measuring something different -- it is measuring route 2, which the brief "
        "records as fixed and which is wide open there. 549 of 1500 vm answers do "
        "not parse and the raw replies are <tool_call> write_memory strings (Qwen: "
        "1). They sit at relation_count 6.22 against 8.43 on correct answers, so "
        "they drag rc_mean_error below rc_mean_correct; drop them and Hermes reads "
        "rc_ratio 1.1142, ceiling 1.3402, normalized 0.3356 against the human "
        "0.3728. Simulating recovery of those 549, raw vm humanlikeness goes 0.6968 "
        "-> 0.7862 (p=0.383) -> 0.9362 (p=0.7) -> 0.7280 (p=1.0), up at every rate, "
        "Qwen unmoved at 0.6850. This candidate does NOT fix route 2 -- granting "
        "probe turns a maintenance slot would be a second behavioural change and "
        "neither would be attributable -- so its Hermes A4 row is pre-registered as "
        "an EXPECTED AND EXPLAINED violation, and route 2 is handed to iteration 7 "
        "measured. FALSIFIABLE: PRECONDITION pitched at the MECHANISM, not the "
        "score, because route 3 alone provably cannot restore the Hermes answered "
        "baseline -- unparsed replies per block <= 0.2 at all three levels on both "
        "substrates (now 0.66 Qwen n=2, 1.82 and 0.72 Hermes) with ACT-1 unparsed "
        "share <= 0.02 per level; the nback floor and humanlikeness go in separate "
        "NON-VOIDING ambition rows, Hermes banded [-0.055, +0.020] from the "
        "discount sweep. THE MECHANISM ROW also pre-registers noresp per block, "
        "which is the bet: if unparsed falls and noresp does not, the unified "
        "account is wrong and the candidate fails honestly. vm HOLDS: raw >= 0.66, "
        "matched >= 0.62 on Qwen, A4 n_errors >= 150 and rc_ratio_normalized >= 0.15 "
        "on Qwen. Leak: n=3 letter-identity share >= 0.90. ANTI-full_context: n=3 "
        "keys_held >= 3.5, n=2 >= 2.0, acc_over_answered at n=3 <= 0.85. BATCH "
        "CONTROL differenced against evicting_reset on the six encode()->recall() "
        "tasks, where the code path is now provably the baseline's, banded from the "
        "parent's identical-path spread (craft 0.0248) and not from the recorded "
        "noise floor. REQUIRED COMPANION ARM: evicting_reset with the injection "
        "applied ONCE (14 min, Qwen), without which the de-duplication and the "
        "ordering are not separable."
    ),
}
