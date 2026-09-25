"""Candidate `episodic_reset_v3`: the history reset, with the WHOLE task set restored.

WHAT IS BEING REPAIRED
----------------------
`episodic_reset_v2` closed the conversation-history leak and produced the project's
first creditable large gain -- `variable_mapping` 0.3554 -> 0.6854 raw, 0.3587 ->
0.7568 matched, A4 `n_errors` 12 -> 550 with `rc_ratio_normalized` 0.7362, and the
n=3 store carrying letter identity 0.0202 -> 0.9843 -- with its n=1 precondition
passed (answered 13.88, zero silent, `keys_held` 1.00) and with `craft_task`,
`narrative_qa` and both digit spans moving by exactly 0.0000.

It was rejected because it silences the agent at n>=2:

    level   baseline          episodic_reset      episodic_reset_v2
    n=1     13.98,  0 silent    2.06, 36 silent    13.88,  0 silent
    n=2     13.24,  0 silent   10.16,  0 silent     6.80,  3 silent
    n=3      6.82,  0 silent    9.08,  0 silent     2.12, 12 silent

`nback` fell 0.2308 against a 0.060 floor. This candidate keeps every line of v2's
mechanism and adds the one thing its control-state block left out.

SECTION 1 -- WHAT THE FAILURE ACTUALLY IS, AND THE BRIEFED HYPOTHESIS IS REFUTED
--------------------------------------------------------------------------------
The hypothesis I was asked to test first was prompt competition: the prepended block
competes with `variable_mapping`'s "output ONLY one line ... No extra text", and the
competition scales with how much the block contains, which is why the failure tracks
store occupancy. **It does not survive contact with the run.** Four measurements,
each of which kills it independently.

(a) **THE BLOCK IS BYTE-IDENTICAL ACROSS LEVELS ON THE TURN WHERE THE FAILURE IS
    ALREADY PRESENT.** On the first stimulus turn of an n-back block the agent is at
    episode turn 2, presentations 1, store empty, so v2's block is the same **278
    characters** at n=1, n=2 and n=3 (the whole `user_message` is 294) -- verified by
    driving the real `run_nback_block` offline at all three levels. The share of rows
    whose reply at that turn parses at all is

        n=1  50/50 = 1.00     n=2  17/50 = 0.34     n=3  7/50 = 0.14

    A quantity that is constant cannot explain a threefold-to-sevenfold difference.

(b) **THERE IS NO VERBOSITY.** v2's reported `variable_mapping` reply length -- mean
    15.91 against the baseline's 13.10, maximum 147 -- decomposes as follows:

        arm                 n      all replies       non-<tool_call> replies
        baseline          1500   mean 13.10 max 14   1500 rows, mean 13.10 max 14
        episodic_reset    1500   mean 13.18 max 131  1499 rows, mean 13.10 max 14
        episodic_reset_v2 1500   mean 15.91 max 147  1462 rows, mean 13.09 max 14

    Excluding 38 rows, v2's mean reply length is **13.09 against the baseline's
    13.10 and the same maximum of 14**. The block induced exactly zero verbosity;
    the format instruction was obeyed on every turn on which the model answered at
    all. P7 failed on a quantity that, decomposed, shows the opposite of what it was
    written to catch.

(c) **ALL 38 OF THOSE ROWS ARE THE SAME THING, AND IT IS NOT PROSE.** Every one is a
    literal tool call emitted as plain text:

        '<tool_call>\\n{"name": "write_memory", "arguments": {"key": "Linda",
         "value": "Currently lives in Philadelphia"}}\\n</tool_call>'

    38 in v2, 1 in `episodic_reset`, 0 in the baseline. They occur on
    `variable_mapping`'s QUESTION turns, which are `allow_tools=False`, i.e. the
    harness called the model with `tool_choice="none"`. The model wanted to write to
    the store, was denied the tool, and emitted the call as text **instead of the
    answer**. On `variable_mapping` that costs one question. On n-back the same
    string reaches `_parse_classification`, which returns `None`, and the trial is
    recorded as unanswered.

(d) **THE FAILURE IS FLAT OVER TURNS, SO IT IS NOT AN ABSORBING STATE AND NOT
    CUMULATIVE OCCUPANCY.** Rows parsing at each successive presentation (buffer
    positions first, then the 14 trials), out of 50:

        baseline  n=3   50 50 50 |  7 14 36 49 37 44 21 15 21 20 20 15 21 21
        v2        n=3    7 37  9 |  0  8 11  2  9  3 10  4 11  8 11 11  8 10
        v2        n=2   17 48 23 |  9 22 23 22 25 20 27 22 25 34 27 32 29

    v2's n=3 rate is ~0.18 from the first presentation to the last with no trend,
    and the answered-trial index sets are `scattered` (17/50) or
    `contiguous-not-from-1` (21/50) with the first trial unanswered in 50 of 50
    rows. `episodic_reset`'s n=1 failure was a genuine absorbing state -- 14 of 14
    non-silent rows answered a contiguous run ENDING at trial 14. This is the
    opposite shape: an independent per-turn hazard whose rate is set by the level.

SO WHAT DOES SET THE RATE? Two things, one of them the project's own strongest
uncontrolled variable.

**(i) The refusal loop, which is the baseline's own n=3 defect and is NOT mine to
fix.** Sorting every scored arm by whether `WorkingMemory.write_key` refuses when
full gives a perfect separation on n=3 `answered`, at essentially identical store
occupancy:

    write_key REFUSES (baseline semantics)        write_key EVICTS
      baseline            6.82   keys 3.96          displacement  14.0  keys 3.98
      random_decay        6.56   keys 3.94          primacy       14.0  keys 4.00
      random_decay_v2     5.26   keys 3.92          primacy_v2    14.0  keys 3.98
      chunk_limit         5.82   keys 3.86
      serial_recognition  5.22   keys 3.86

    (`full_context` answers 14.0 at `keys_held` 1.06 -- it never fills the store.)

A refusing 4-slot store costs up to three tool calls to change one slot: the refused
write, a `delete_key`, then the write. `_tool_call_cap()` is `max(6, int(1.5 *
_tool_interactions))`, i.e. 1.5 calls per turn. Driving the real `run_nback_block`
offline with a stub that maintains an n-slot buffer, `tool_call_cap_hit` -- the base
`step()`'s own flag for "the budget ran out, or the agent's tool calls were
TRUNCATED" -- fires on

    n=1, 1 write/turn    0 of 16 steps   (budget never reached zero)
    n=2, 2 writes/turn  14 of 17 steps   (zero budget on step 5)
    n=3, 3 writes/turn  16 of 18 steps   (zero budget on steps 4 and 5)

-- perfect level-grading, the same ordering as the failure, and the state in which
finding (c) shows the model emits its tool call as text in place of the answer. Note
the second reading: at n>=2 the harness is silently DISCARDING part of the agent's
store update on most turns. This mechanism is present
in the baseline, which is why the BASELINE ITSELF only answers 6.82 of 14 at n=3.
The brief forbids the eviction surface, so **the ceiling on this candidate's n=3
recovery is set by a defect it is not allowed to touch.** That is stated here, before
the run, as the honest limit on iteration 4.

**(ii) What closing the leak does to (i), and what v2 added on top.** Closing the
leak forces the store to be genuinely maintained, so occupancy at n=2 goes from the
baseline's 1.54 keys (5 of 50 rows at capacity) to v2's 3.58 (29 of 50 at capacity)
-- and v2's n=2 `answered` of **6.80** is, to two decimals, the baseline's n=3 value
of **6.82**. n=2 was pushed into the regime n=3 was already in. That much
`episodic_reset` also does (n=2 10.16 at 3.06 keys).

But v1 and v2 hold the SAME 3.82 keys at n=3 and answer 9.08 and 2.12. The
difference between them is three lines of text, and only one of those lines is an
instruction: `TASK_SET_REMINDER`, "Update it each turn to track what you will need
later." It is the last thing the agent reads before the stimulus, it names the store
and nothing else, and the reset has deleted the fourteen prior turns of its own
replies that were the only other place a per-turn response obligation was visible.
The agent complies with the one standing directive it can see.

**DIAGNOSIS: the n>=2 failure is a task-set failure, not a memory failure.** At n=3
v2 is 0.6225 accurate on the trials it does answer -- it can do the task. It omits
the response. And what it was told to do instead, it did: `keys_held` 3.82, letter
identity 0.9843.

SECTION 2 -- THE MECHANISM
--------------------------
One rule, unchanged in scope from v2 and completed in content: **what crosses a turn
boundary is the agent's control state -- its position in the episode, its store, and
the COMPLETE task set -- and nothing about earlier stimuli.**

v2 stated half the task set. A task set is not a maintenance policy; it is the
control configuration that binds the current stimulus to a response (Logan & Gordon
2001's executive control settings; Monsell 2003's task-set reconfiguration), and in a
task with a concurrent memory requirement it comprises both the response mapping and
the maintenance requirement. v2 carried the maintenance requirement across the turn
boundary and dropped the response requirement, then deleted the transcript that was
the response requirement's only other carrier. The repair is to state both, in the
order in which they are discharged, with the response obligation **last** -- adjacent
to the stimulus, where v2 had put the store directive.

Three deltas from v2, all in the same block, none anywhere else:

  * the maintenance imperative and the no-transcript disclaimer are separated. v2
    welded them into one sentence, which made the only imperative in the block a
    store imperative;
  * a response obligation is stated, task-agnostically ("the response this turn's
    instructions call for" -- so during an n-back buffer period the instructed
    response is "no response", and on a `variable_mapping` encoding turn there is
    none to give);
  * the response obligation is made explicitly independent of the store operation
    succeeding, which is the sentence aimed at finding (c): on a turn where the
    model cannot write, it must answer rather than emit its tool call as text.

Nothing is removed. Everything v2 said, v3 also says. The claim is that the block's
content was incomplete, not that its form was excessive -- and finding (b) is why:
there was no verbosity to economise on.

WHY THE ORDINALS AND THE DISCLAIMER KEEP THEIR EXEMPTION
--------------------------------------------------------
Unchanged from v2, and its argument stands. Oberauer (2002) exempts task set and
activated long-term memory from the ~4-chunk limit; task set is procedural and
retrieved rather than held (Monsell 2003; Logan & Gordon 2001). The two ordinals are
functions of the method-call count alone -- computed without reading `user_message`,
the store, or any stimulus -- so they carry zero bits about which stimuli appeared and
cannot substitute for the store. Every serial-order model of working memory posits
exactly such an always-available signal outside the item store: drifting temporal
context (Howard & Kahana 2002), an oscillator positional code (Brown, Preece & Hulme
2000), a loop-position timing signal (Burgess & Hitch 1999), temporal distance as the
discriminative dimension (Brown, Neath & Chater 2007).

The added response obligation needs no new exemption and is the easiest line in the
block to defend. It is a stimulus-response mapping, the textbook content of a task
set, and it is what a human participant has continuously and for free: the n-back
screen asks for a keypress on every letter, and the instruction that it does is on
the block-onset screen, not in the participant's memory for the letters. A harness
that deletes the transcript deletes that, and that deletion is an artifact of the
implementation, not a capacity limit.

SECTION 3 -- GENERAL, NOT AN N-BACK SPECIAL CASE
-------------------------------------------------
No task is named anywhere in this file and no branch tests for one. The block is the
same text on every `step()` of every task. What it does elsewhere:

  * `variable_mapping`, question turns (`allow_tools=False`, store already rendered
    by the task's own `QUESTION_PROMPT`): three lines become four. The 38 unparsed
    answers should go to ~0, which RAISES the model's score on ~38 of 1500 questions
    and therefore LOWERS `variable_mapping` humanlikeness slightly, because the model
    is already better than the humans there. A fall from 0.6854 is predicted and
    accepted; a collapse is not. The bulk of the 550 errors is capacity-bound, not
    obligation-bound -- 20 assignments against 4 slots, with 674 of 1500 questions
    asking about an already-evicted name -- so the error structure should survive.
  * `variable_mapping`, encoding turns: the agent is told a response is due when the
    task asks for none. Its text on those turns is discarded by the task, so the cost
    is tokens, not score.
  * the six `encode()` -> `recall()` tasks: **untouched, for the same structural
    reason as v2.** `encode()` calls `step()` exactly once, so the episode turn index
    is 1 and no block is prepended. `recall()` is not overridden at all.

SECTION 4 -- NOT A CAPACITY CHANGE, AND WHAT IS DELIBERATELY NOT TOUCHED
------------------------------------------------------------------------
`MAX_KEYS` stays 4, read live. `WorkingMemory` is neither subclassed nor imported for
modification: `write_key`, the overflow policy and the eviction order are literally
the baseline's. `recall()` is untouched, including the `word_recognition` trial-list
presentation. `_tool_call_cap()` is **not** overridden, although section 1 identifies
it as the proximate cause of the `tool_choice="none"` turns. Two reasons, both
stated before the run:

  * raising it would let the agent churn the store faster but not hold more, so it
    cannot fix the 674-of-1500 already-evicted questions that produce
    `variable_mapping`'s humanlike error rate -- it risks the headline gain to buy a
    quantity it cannot buy;
  * it would make this candidate two mechanisms and destroy attribution.

Instead the budget is **measured**: P9 reports per-level `tool_call_cap_hit` and
zero-budget-turn shares from the newly persisted n-back `step_log`, against the stub
figures above, with no threshold. That is the decision input for iteration 5 and it
is the first run in the project able to produce it.

There is no RNG in this file and no swept constant: the block is fully determined by
the two call counts and the store.

SECTION 5 -- WHY THE LEAK STAYS CLOSED AND n=1 STAYS FIXED
----------------------------------------------------------
The leak, by construction: everything the agent sees on turn k is the system prompt,
the tool schemas, two integers derived from call counters, fixed English that is
identical on every task and every turn, the store, and the current `user_message`.
`step()` retains no reference to any previous `user_message` -- `reset_messages()` is
called at the top of every turn, exactly as in v2 -- so no earlier stimulus can reach
turn k except through the four slots.

n=1, by retention: `episodic_reset`'s n=1 failure was an absorbing state created by
its `if store` guard, which prepended nothing when the store was empty and left turn
k a pure function of the current letter. v3 renders the store unconditionally,
including as "(memory is empty)", and states strictly increasing ordinals, so no two
turns of an episode are ever the same request. Both of those are v2's, both are kept
verbatim, and the added response obligation can only push n=1 in the direction it
already succeeds in. v2 read 13.88 answered, 0.9964 accurate, 0 silent, `keys_held`
1.00 at n=1.

SECTION 6 -- PREDICTIONS
------------------------
`MANIFEST.md` beside this file, machine-readable in
`meta_harness/logs/pending_episodic_reset_v3.json`. The precondition is n-back at
every level, deliberately set at "restores the predecessor that did not have this
defect" rather than at the candidate's ambition, because a precondition pitched at the
ambition can VOID a successful candidate -- which is how `episodic_reset` lost a
+0.32.
"""
from __future__ import annotations

from typing import Any

from bench.core import working_memory as _wm_mod
from bench.core.wm_agent import (  # noqa: F401  -- re-exported unchanged
    CONDITION_PROMPTS,
    SummarizerAgent,
    TOOLS,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS

# Verbatim from `RECALL_PROMPT` (bench/core/wm_agent.py) and `QUESTION_PROMPT`
# (bench/tasks/wm_variable_mapping.py). Used both to render the store and to detect
# that the task's own message already renders it, so it is never shown twice.
WM_STATE_HEADER = "Your working memory currently contains:"

# The block marker. One line, fixed, so the control state is visibly a frame around
# the turn rather than part of the stimulus.
EPISODE_MARKER = "[ongoing episode]"

# The epistemic state, separated from any imperative. In `episodic_reset_v2` this
# sentence and the maintenance directive were welded together, which made the only
# imperative in the block a store imperative -- see the module docstring, section 1.
NO_TRANSCRIPT_NOTICE = (
    "You have no transcript of earlier turns: the key-value store is your only "
    "record of them."
)

# The complete task set: both standing obligations, in the order in which they are
# discharged, with the response obligation last -- adjacent to the stimulus. The
# third sentence is aimed at the measured failure surface: on a turn where the model
# is denied the write tool it emitted the tool call as text INSTEAD of the answer
# (38 of 1500 variable_mapping answers under v2, 0 under the baseline).
#
# Task-agnostic by construction: it refers to "this turn's instructions" rather than
# to any task's response set, so an n-back buffer period (instructed response: "no
# response") and a variable_mapping encoding turn (no instructed response) are both
# covered without naming either.
STANDING_OBLIGATIONS = (
    "Two things are due on every turn of an episode. Update the store to track what "
    "you will need later, and give the response this turn's instructions call for. "
    "The response is due either way: if the store cannot be changed, or needs no "
    "change, respond anyway."
)


def _capacity() -> int:
    """Read capacity live, so an injected `MAX_KEYS` is honoured."""
    return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with no episodic record of earlier turns, but with its full
    control state carried across the turn boundary.

    What the agent sees on turn k of an episode: the system prompt (task set as the
    task wrote it), its position in the episode, the notice that it has no
    transcript, the current contents of the 4-slot store, both standing obligations
    of a turn, and the current stimulus. Nothing else -- no earlier stimuli, no
    earlier replies of its own, no earlier tool calls.

    Only `step()` is overridden. `encode()`, `recall()`, segmentation, key naming,
    value formatting, the tool-call cap, the prompts, the tools and `WorkingMemory`
    itself are the baseline's.
    """

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
        # encode()->recall() tasks byte-identical to the baseline, since encode()
        # calls step() exactly once.
        if self._episode_turn_index() > 1:
            block = self._control_state_block(
                allow_tools,
                render_store=WM_STATE_HEADER not in user_message,
            )
            user_message = f"{block}\n\n{user_message}"

        return super().step(user_message, allow_tools=allow_tools,
                            max_tokens=max_tokens)


MANIFEST: dict[str, Any] = {
    "id": "episodic_reset_v3",
    "parent": "episodic_reset_v2",
    "capacity": _capacity(),
    "decay": (
        "none. No stochastic loss of any kind and no RNG in the file. The "
        "conversation history is not carried across turns, so the 4-slot store is "
        "the sole route from an earlier stimulus to an answer; what does cross the "
        "turn boundary is control state only -- two ordinal counters derived from "
        "call counts, three fixed sentences, and the store itself. Deterministic "
        "and per-turn identical given the same call sequence."
    ),
    "role": (
        "contender -- iteration 4, repairs episodic_reset_v2's n>=2 response "
        "omission while keeping the history leak closed and n=1 fixed"
    ),
    "summary": (
        "episodic_reset_v2 closed the step() history leak and produced the project's "
        "first creditable large gain (variable_mapping 0.3554 -> 0.6854 raw, 0.3587 "
        "-> 0.7568 matched; A4 n_errors 12 -> 550 at rc_ratio_normalized 0.7362; n=3 "
        "store letter identity 0.0202 -> 0.9843; craft/narrative/both digit spans "
        "exactly 0.0000) with its n=1 precondition passed, and was rejected for "
        "silencing the agent at n>=2 (answered 13.88/6.80/2.12 with 0/3/12 silent "
        "against the baseline's 13.98/13.24/6.82 with 0/0/0; nback -0.2308 against a "
        "0.060 floor). THE BRIEFED HYPOTHESIS IS REFUTED, four ways. (a) On the "
        "first stimulus turn the block is BYTE-IDENTICAL across levels -- 294 chars, "
        "empty store, verified by driving the real run_nback_block offline at n=1, 2 "
        "and 3 -- yet the share of rows parsing at that turn is 1.00 / 0.34 / 0.14. "
        "(b) THERE IS NO VERBOSITY: excluding 38 rows, v2's variable_mapping mean "
        "reply length is 13.09 against the baseline's 13.10, with the same maximum "
        "of 14. The reported 15.91 / max 147 is those 38 rows and nothing else. (c) "
        "All 38 are one thing -- a literal <tool_call>{\"name\": \"write_memory\", "
        "...}</tool_call> string emitted as plain text on a QUESTION turn, which is "
        "allow_tools=False, i.e. tool_choice='none'. 38 in v2, 1 in episodic_reset, "
        "0 in the baseline. The model wanted to write, was denied the tool, and "
        "emitted the call INSTEAD of the answer; on n-back that string parses to "
        "None and the trial is unanswered. (d) The hazard is FLAT over turns (v2 n=3 "
        "parses ~0.18 from the first presentation to the last, first trial "
        "unanswered in 50/50 rows, answered-index sets scattered) so it is neither "
        "an absorbing state nor cumulative occupancy. WHAT DOES SET THE RATE: (i) "
        "the write_key refusal loop, which separates the arms PERFECTLY on n=3 "
        "answered at identical occupancy -- refusing semantics give baseline 6.82 / "
        "random_decay 6.56 / random_decay_v2 5.26 / chunk_limit 5.82 / "
        "serial_recognition 5.22, evicting semantics give displacement 14.0 / "
        "primacy 14.0 / primacy_v2 14.0 -- because a refused write plus delete_key "
        "plus rewrite costs 3 tool calls against _tool_call_cap()'s 1.5 per turn; "
        "driving the real run_nback_block with a stub, tool_call_cap_hit (budget "
        "exhausted OR the agent's tool calls truncated) fires on 0 of 16 steps at 1 "
        "write/turn, 14 of 17 at 2 and 16 of 18 at 3, with the budget actually "
        "reaching zero on 0, 1 and 2 steps -- so at n>=2 the harness is silently "
        "DISCARDING part of the store update on most turns. THE BRIEF FORBIDS THAT "
        "SURFACE, so it caps this candidate's n=3 "
        "recovery and that is stated before the run. (ii) Closing the leak pushes "
        "n=2 into the regime n=3 was already in -- keys 1.54 (5/50 at capacity) -> "
        "3.58 (29/50), and v2's n=2 answered of 6.80 IS the baseline's n=3 value of "
        "6.82 -- and v2's own TASK_SET_REMINDER adds the rest: v1 and v2 hold the "
        "SAME 3.82 keys at n=3 and answer 9.08 vs 2.12, and the only instruction "
        "separating them is 'Update it each turn', the last line before the "
        "stimulus, naming the store and nothing else, with the 14 prior answering "
        "turns deleted. DIAGNOSIS: a TASK-SET failure, not a memory failure -- at "
        "n=3 v2 is 0.6225 accurate on what it does answer and holds 3.82 keys at "
        "0.9843 letter identity. It did what it was told and was told half of it. "
        "MECHANISM: one rule, every task -- what crosses the turn boundary is "
        "control state (position in the episode, the store, and the COMPLETE task "
        "set) and nothing about earlier stimuli. Nothing v2 said is removed; the "
        "no-transcript notice is separated from the maintenance imperative, and both "
        "standing obligations are stated with the RESPONSE obligation LAST, adjacent "
        "to the stimulus, and explicitly independent of the store operation "
        "succeeding ('if the store cannot be changed, or needs no change, respond "
        "anyway') -- the sentence aimed at (c). JUSTIFICATION: a task set is a "
        "control configuration binding stimulus to response (Logan & Gordon 2001; "
        "Monsell 2003), exempt from the chunk limit along with activated LTM "
        "(Oberauer 2002); v2 carried the maintenance half across the boundary and "
        "dropped the response half, then deleted the transcript that was its only "
        "other carrier. A human has the response mapping continuously and for free "
        "from the block-onset screen, so its absence is an implementation artifact. "
        "The ordinals keep v2's exemption: functions of the call count alone, zero "
        "bits about which stimuli appeared, and exactly the always-available "
        "temporal-context signal every serial-order model posits outside the item "
        "store (Howard & Kahana 2002; Brown, Preece & Hulme 2000; Burgess & Hitch "
        "1999; Brown, Neath & Chater 2007). GENERAL: no task named, no branch on "
        "task; on variable_mapping the 38 unparsed should go to ~0, which RAISES the "
        "model's score on ~38 of 1500 questions and so LOWERS humanlikeness -- a "
        "fall from 0.6854 is predicted and accepted, a collapse is not, and the bulk "
        "of the 550 errors is capacity-bound (674 of 1500 questions ask about an "
        "already-evicted name) rather than obligation-bound. CAPACITY UNTOUCHED: "
        "MAX_KEYS 4 read live, WorkingMemory not subclassed, write_key / overflow / "
        "eviction order literally the baseline's, recall() untouched. "
        "_tool_call_cap() DELIBERATELY NOT OVERRIDDEN although section 1 names it as "
        "the proximate cause -- more budget lets the agent churn faster but not hold "
        "more, so it cannot buy the evicted-name errors that make vm humanlike, and "
        "it would make this two mechanisms; instead P9 REPORTS per-level "
        "tool_call_cap_hit and zero-budget shares from the newly persisted n-back "
        "step_log, against the stub figures, as iteration 5's decision input. "
        "FALSIFIABLE: PRECONDITION at every level, pitched at 'restores the "
        "predecessor without this defect' rather than at the ambition, because a "
        "precondition pitched at the ambition is how episodic_reset lost a +0.32 -- "
        "n=1 answered >= 13.0 / acc >= 0.90, n=2 >= 9.5 / >= 0.60, n=3 >= 6.0 / >= "
        "0.50, and n_no_answers == 0 at ALL THREE (v2 read 0/3/12), a failure of "
        "which invalidates everything after it; strong claims registered separately "
        "at n=2 >= 12.0 and n=3 >= 10.0; nback humanlikeness >= 0.7309, its floor; "
        "n=3 letter-identity share >= 0.90; vm >= 0.55 raw and >= 0.60 matched with "
        "A4 n_errors >= 150 and rc_ratio_normalized in [0.15, 0.95]; PARSE INTEGRITY "
        "-- vm unparsed <= 5 of 1500 and <tool_call>-in-text <= 5, mean reply length "
        "<= 14.5 over all rows AND <= 13.5 over non-<tool_call> rows, since the "
        "latter is where v2 actually scored 13.09; buffer_no_response_frac must RISE "
        "at n=2 and n=3, which answering more eagerly cannot buy; a contamination "
        "check that a reply containing <tool_call> is not being counted as a "
        "classification; and a no-change control on the six encode()->recall() tasks "
        "at |delta| <= 0.008, which is v2's own worst value (semantic_story_recall "
        "+0.0071; four of the six moved by exactly 0.0000 and word_recognition by "
        "-0.0005, so the brief's 'all six exactly 0.0000' is true of four)."
    ),
}
