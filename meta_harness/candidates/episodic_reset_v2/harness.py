"""Candidate `episodic_reset_v2`: the history reset, with the control state restored.

WHAT `episodic_reset` GOT RIGHT, AND WHAT IT BROKE
-------------------------------------------------
Iteration 2c closed the conversation-history leak on the two tasks that answer
through `step()`, and the mechanism worked: the n=3 store went from carrying
letter identity in 0.0202 of its values to 1.0000, `variable_mapping` A4 produced
509 errors instead of 12, and `variable_mapping` humanlikeness rose 0.3554 ->
0.6764 (0.3587 -> 0.7295 on the matched formula). All of it was scored VOID,
because its own n=1 control failed: 36 of 50 participants answered nothing, mean
`answered` 2.06 of 14, `keys_held` 0.28.

This candidate keeps the reset and repairs the n=1 failure. The failure is not
what its predecessor pre-committed to, and the diagnosis below is measured from
the run rather than assumed.

THE FAILURE, FROM THE RAW ROWS
------------------------------
`meta_harness/runs/iter2/episodic_reset/.../tasks/wm_nback.jsonl`, n_level == 1,
50 rows:

  * 36 rows have `model_parsed == {}` and `final_kv == {}`. They answered nothing
    and wrote nothing, for fifteen consecutive turns.
  * All 14 rows that answered anything answered a **contiguous run ending at
    trial 14** -- 14 of 14, no exceptions -- and every one of them ends holding
    exactly one key, `previous_letter`, whose value is the block's LAST letter.
  * So the block has exactly two regimes: a silent regime in which the store stays
    empty, and, after a single transition, a normal regime in which the agent
    answers every remaining trial (accuracy over those answers 0.787).

That shape is produced by a **near-absorbing state that the candidate's own code
creates**. `episodic_reset.step()` prepends the store only `if store` is
non-empty. At n=1 turn 1 the store is empty, so the agent sees exactly

    [system: the C2 n=1 prompt]
    [user: "Next letter: G"]

and nothing else -- no store block, not even "(memory is empty)", no turn index,
no trace that a turn has ever happened. The instruction-compliant reply to that
context is "no response", because `HUMAN_PROMPT_BY_N[1]` says to answer "no
response" to the first letter and nothing in the context says this is not the
first letter. If the agent also does not write, the store is still empty on turn
2, and turn 2's request is identical in form to turn 1's. The state is
self-sustaining: the only thing that varies between those turns is the letter.

WHICH LETTER BREAKS IT (measured, and it is nearly deterministic)
----------------------------------------------------------------
Taking the trigger to be the letter presented on the first turn that produced a
scored answer, and counting every empty-store presentation of each letter across
all 50 n=1 blocks (661 presentations, 14 escapes, base rate 0.0212):

    letter   escapes / empty-store presentations
      C          11 / 15   = 0.733
      B           1 / 29   = 0.034
      H           1 / 31   = 0.032
      M           1 / 40   = 0.025
      the other 17 letters  0 / 601 = 0.000

Escape is not a uniform hazard over turns; it is carried by one letter. The
system prompt's own worked example for n=1 is `A -> A -> B -> C -> C`, whose
explanation is "The second C matches the previous C -> same". "Next letter: C"
collides with that example, and the model answers rather than declining -- which
also accounts for B, the example's other consonant, being the only other letter
above 0.03. (Locus check: attributing the trigger instead to the turn BEFORE the
first answer scatters it across eight letters with no letter above 0.143, i.e. no
structure. So the write and the first answer happen on the same turn.)

This also settles a question the project had answered the other way. Those 15
empty-store presentations of "Next letter: C" at n=1 are **bit-identical
requests**: the n=1 system prompt does not vary across participants, the store is
empty so no block is prepended, and the user message is the same six characters.
Eleven escaped and four did not. The serving stack is therefore not
bit-deterministic, and the divergence rate on a short generation from one
identical request is 4/15 here. See MANIFEST.md section 1 for the cross-arm
confirmation and for what that costs the old P10 threshold.

WHAT THE PRE-COMMITTED DIAGNOSIS GOT RIGHT AND WRONG
----------------------------------------------------
`episodic_reset` pre-committed: "a P9 failure localised to buffer-period turns
means 'lost sequence position', whose fix is pinning the instruction turn."

Right: the agent has lost sequence position. It cannot tell turn 1 from turn 15,
which is the whole failure.

Wrong in two places, both checkable.

  1. Its claim that the task's instruction turn is "redundant with the system
     prompt, so nothing needs pinning" is false. The system prompt
     (`wm_prompt_parts.wm_system_prompt`) says "Use write_memory and delete_key to
     maintain the key-value store while doing the original task". The instruction
     turn (`wm_nback.py:86`) uniquely adds **"Update your working memory each turn
     to track the recent sequence"** -- a standing per-turn directive that the
     system prompt does not contain, and precisely the directive whose absence
     shows up as `keys_held` 0.28.
  2. But pinning that turn would not have fixed it. With the instruction turn
     pinned and the store empty, turn k is still a pure function of the current
     letter: the absorbing state survives, and whether the run escapes it would
     rest entirely on the model happening to write on turn 1 for every letter.
     That is a behavioural bet, not a repair. Nor is the failure "lost
     instructions" (the "no response" reply is instruction-compliant), "broken
     parsing" (`model_parsed_buffer` records a parsed "No response" at position 1
     in 50 of 50 rows, and the post-escape tails parse cleanly), or a capacity
     limit (one item, four slots).

     The proximate cause is that the agent stopped writing; the reason it stopped
     is that it had no way to know a turn had elapsed.

WHY n=1 AND NOT n=2 / n=3
-------------------------
The same empty-store, no-header turn exists at n=2 and n=3, yet `keys_held` is
3.06 and 3.82 there and no participant is silent. The difference is in the
instruction text, not in the harness. `HUMAN_PROMPT_BY_N[1]` says the letter
"matches the letter **one turn back**" and to answer "no response" to "the first
letter"; n=2 and n=3 say "two/three turns back" and "the first two/three
letters", which cannot be satisfied without holding more than the current item,
so the model writes on turn 1 and leaves the absorbing state immediately. At n=1
"one turn back" admits a reading under which nothing needs storing. So n=1 is not
a special task; it is the one level at which the model's own write policy leaves
the store empty long enough for the harness defect to bite. The defect is
general, which is why the repair is general.

THE MECHANISM
-------------
One rule, applied to every `step()` call on every task: **what crosses a turn
boundary is the agent's control state -- the task set, its position in the
episode, and its store -- and nothing about earlier stimuli.**

Concretely, `step()` still rebuilds its message list from scratch. On every step
after the first in an episode it prepends a control-state block:

    [ongoing episode]
    Turn {t} of this episode. Stimulus presentations so far: {p}.
    You have no transcript of earlier turns: the key-value store is your only
    record of them. Update it each turn to track what you will need later.
    Your working memory currently contains:
    {wm.to_recall_text()}

    {the task's own user message}

`t` is `len(self._step_log) + 1`, the number of `step()` calls including this one.
`p` is `self._tool_interactions + (1 if allow_tools else 0)`, the number of turns
on which the agent was permitted to write, including this one when it is. Both are
read, never assigned, so `_tool_call_cap()` is exactly the baseline's. The store
is rendered unconditionally -- including as "(memory is empty)" -- and the
rendering is skipped only when the task's own message already carries
`WM_STATE_HEADER`, so it is never shown twice.

Three changes from `episodic_reset`, all consequences of the same rule:

  * the store is rendered even when empty (it was silently omitted);
  * the two ordinal counters are stated;
  * the standing per-turn directive is stated, in task-agnostic wording, once, as
    part of the task set rather than as a replayed episodic turn.

On the first step of an episode the block is omitted entirely. That has a reason
and a consequence, and both are stated because they coincide. The reason: at block
onset no turn has elapsed, there is no store to report and no transcript to
disclaim, so every line of the block is vacuous. The consequence: the six tasks
that go `encode()` -> `recall()` call `step()` exactly once, so their requests stay
byte-identical to the baseline's and the no-change control keeps its meaning. The
gate is justified by the first and audited by the second; it is not chosen for the
second.

WHY THIS IS TASK SET AND NOT ITEM MEMORY
----------------------------------------
The concentric model `episodic_reset` argued from (Oberauer 2002) exempts
activated long-term memory and task set from the ~4-chunk limit, and task set is
procedural and retrieved rather than held (Monsell 2003; Logan & Gordon 2001).
The standing directive is unambiguously in that category: it says how to operate
the store, not what was in it.

The ordinal counters need the harder argument, and it is available rather than
assumed. Two grounds:

  * **They are provably non-diagnostic of stimulus content.** `t` and `p` are
    functions of the number of method calls. They are computed without reading
    `user_message`, the store, or any stimulus, and they carry zero bits about
    which letters or statements appeared. So they cannot substitute for the store,
    cannot raise n-back accuracy above what the store supports, and cannot be a
    back door for the leak. "How many items have I seen" and "which items were
    they" are separable here in the strongest sense: one is derivable from the
    call count alone.
  * **Every serial-order model of working memory posits exactly such a signal,
    outside the item store.** A slowly drifting temporal context that is always
    available and is not itself an item (Howard & Kahana 2002); an oscillator-based
    positional code (Brown, Preece & Hulme 2000); a timing signal indexing
    position in the loop (Burgess & Hitch 1999); temporal distance as the
    discriminative dimension (Brown, Neath & Chater 2007). None of them charges
    this signal against the chunk limit, because it is context, not content.

And the honest form of the concession the brief asked for: **yes, humans in this
task have access to something the reset removed, and it is not item memory.** A
participant sees a block-onset screen, then letters at a fixed 2000 ms ISI. They
cannot be in the state this harness put the agent in -- unable to tell the first
item from the fifteenth -- because the situation supplies that, continuously and
for free, whatever their memory does. Deleting it was a harness artifact, not a
capacity limit, and restoring it is restoring the experimental situation rather
than granting extra memory. `keys_held` 0.28 at n=1 against 1.00 at baseline is
what that artifact looks like in the data.

The exclusions from `episodic_reset` are unchanged and are the point of the
candidate: earlier stimuli, the agent's own earlier replies and tool calls, and
any last-N-turns window (N = 0) all stay out. A verbatim recent window is a second
uncapped store, and `full_context` is the least humanlike harness measured.

WHY THE LEAK STAYS CLOSED
-------------------------
By construction. Everything the agent sees on turn k is: the system prompt
(unchanged), the tool schemas (unchanged), two integers derived from call
counters, fixed English text that is the same on every task and every turn, the
store, and the current `user_message`. `step()` retains no reference to any
previous `user_message`, so no earlier stimulus can reach turn k by any route
other than the four slots. The n=3 letter-identity result (0.0202 -> 1.0000) is
therefore expected to survive; it is registered as P2.

NOT A CAPACITY CHANGE, NOT NOISE, AND NOT A CRASH
-------------------------------------------------
`MAX_KEYS` stays 4 and `WorkingMemory` is neither subclassed nor touched, so
`write_key`, the overflow policy and the eviction order are literally the
baseline's -- iteration 2's `primacy` owns that surface. `recall()` is untouched,
including the word_recognition trial-list presentation. There is no RNG in this
file and no swept constant: the block's content is fully determined by the call
counts and the store.

`episodic_reset` raised `RuntimeError` when the store exceeded capacity. That is
**removed**, not merely relaxed. It was dead code under the baseline's
`write_key`, which refuses rather than overflows, but under a composed overflow
rule a transient overflow would have turned a degraded row into a hard crash, and
the invariant is `write_key`'s to enforce. See MANIFEST.md section 6.

PREDICTIONS
-----------
`MANIFEST.md` beside this file, and machine-readable in
`meta_harness/logs/pending_episodic_reset_v2.json`.
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
# (bench/tasks/wm_variable_mapping.py). Used both to render the store and to
# detect that the task's own message already renders it, so it is never shown
# twice.
WM_STATE_HEADER = "Your working memory currently contains:"

# The block marker. One line, fixed, so the control state is visibly a frame
# around the turn rather than part of the stimulus.
EPISODE_MARKER = "[ongoing episode]"

# The standing procedural directive. Task-agnostic: it describes how to operate
# the store, not what any task's material is. The second sentence generalises
# `wm_nback.py:86`'s "Update your working memory each turn to track the recent
# sequence" by dropping the n-back-specific object, which is the one piece of
# standing guidance the history reset removed and the system prompt does not
# carry.
TASK_SET_REMINDER = (
    "You have no transcript of earlier turns: the key-value store is your only "
    "record of them. Update it each turn to track what you will need later."
)


def _capacity() -> int:
    """Read capacity live, so an injected `MAX_KEYS` is honoured."""
    return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with no episodic record of earlier turns, but with its
    control state carried across the turn boundary.

    What the agent sees on turn k of an episode: the system prompt (task set),
    its position in the episode, the standing directive to maintain the store,
    the current contents of the 4-slot store, and the current stimulus. Nothing
    else -- no earlier stimuli, no earlier replies of its own, no earlier tool
    calls.

    Only `step()` is overridden. `encode()`, `recall()`, segmentation, key
    naming, value formatting, the tool-call cap, the prompts, the tools and
    `WorkingMemory` itself are the baseline's.
    """

    def _episode_turn_index(self) -> int:
        """Ordinal of the current `step()` call within this episode, 1-based.

        Read-only: `_step_log` is appended by the base `step()` after the turn
        completes, so its length before the call is the number of completed
        turns.
        """
        return len(getattr(self, "_step_log", ())) + 1

    def _presentation_index(self, allow_tools: bool) -> int:
        """How many turns the agent has been permitted to encode on, including
        this one when it may.

        `_tool_interactions` is incremented by the base `step()` *after*
        `_ensure_messages()`, so reading it here gives the pre-increment count.
        It is read and never assigned, so `_tool_call_cap()` is unchanged.
        """
        return int(getattr(self, "_tool_interactions", 0)) + (1 if allow_tools else 0)

    def _control_state_block(self, allow_tools: bool, render_store: bool) -> str:
        lines = [
            EPISODE_MARKER,
            f"Turn {self._episode_turn_index()} of this episode. "
            f"Stimulus presentations so far: {self._presentation_index(allow_tools)}.",
            TASK_SET_REMINDER,
        ]
        if render_store:
            lines.append(WM_STATE_HEADER)
            lines.append(self.wm.to_recall_text())
        return "\n".join(lines)

    def step(self, user_message: str, *, allow_tools: bool = True,
             max_tokens: int = 1024) -> str:
        # Turn boundary. Doing this here and not inside the tool loop is load
        # bearing: clearing mid-loop would leave a `tool` message with no
        # preceding assistant `tool_calls` and the request would be rejected.
        self.reset_messages()

        # First step of an episode: every line of the block would be vacuous (no
        # turn has elapsed, no store to report, no transcript to disclaim), so
        # nothing is prepended. This is also what keeps the six encode()->recall()
        # tasks byte-identical to the baseline, since encode() calls step() once.
        if self._episode_turn_index() > 1:
            block = self._control_state_block(
                allow_tools,
                render_store=WM_STATE_HEADER not in user_message,
            )
            user_message = f"{block}\n\n{user_message}"

        return super().step(user_message, allow_tools=allow_tools,
                            max_tokens=max_tokens)


MANIFEST: dict[str, Any] = {
    "id": "episodic_reset_v2",
    "parent": "episodic_reset",
    "capacity": _capacity(),
    "decay": (
        "none. No stochastic loss of any kind and no RNG in the file. The "
        "conversation history is not carried across turns, so the 4-slot store is "
        "the sole route from an earlier stimulus to an answer; what does cross the "
        "turn boundary is control state only -- two ordinal counters derived from "
        "call counts, one fixed procedural sentence, and the store itself. "
        "Deterministic and per-turn identical given the same call sequence."
    ),
    "role": (
        "contender -- iteration 3a, repairs episodic_reset's n=1 mechanics while "
        "keeping the history leak closed"
    ),
    "summary": (
        "episodic_reset closed the step() history leak (n=3 store carrying letter "
        "identity 0.0202 -> 1.0000; variable_mapping A4 n_errors 12 -> 509; vm "
        "humanlikeness 0.3554 -> 0.6764, matched 0.3587 -> 0.7295) and voided all "
        "of it by failing its own n=1 control (36/50 silent, answered 2.06, "
        "keys_held 0.28). DIAGNOSIS FROM THE ROWS, not from the pre-commitment: "
        "all 14 non-silent n=1 rows answered a CONTIGUOUS RUN ENDING AT TRIAL 14 "
        "(14/14) and all 14 end holding exactly {previous_letter: <last letter>}, "
        "while 36 rows have both model_parsed and final_kv empty. That is a "
        "near-absorbing state created by episodic_reset's own `if store` guard: "
        "with an empty store it prepends nothing, so the turn is [system] + "
        "['Next letter: X'] with no store block, no '(memory is empty)' and no "
        "turn index, 'no response' is the instruction-compliant reply, and not "
        "writing keeps the next turn identical in form. Escape is carried by ONE "
        "LETTER: C escapes on 11 of 15 empty-store presentations (0.733) against "
        "0.034/0.032/0.025 for B/H/M and 0.000 for the other 17 letters over 601 "
        "presentations -- C collides with the n=1 system prompt's own worked "
        "example A->A->B->C->C ('the second C matches the previous C -> same'). "
        "THE PRE-COMMITTED FIX WAS WRONG: pinning the instruction turn leaves the "
        "turn a pure function of the letter while the store is empty, so the "
        "absorbing state survives. It was also wrong that the instruction turn is "
        "redundant -- it uniquely carries 'Update your working memory each turn to "
        "track the recent sequence', which the system prompt does not. Not lost "
        "instructions ('no response' is compliant), not broken parsing "
        "(model_parsed_buffer records a parsed 'No response' at position 1 in "
        "50/50), not capacity (one item, four slots). MECHANISM: one rule, every "
        "task -- what crosses a turn boundary is control state (task set, position "
        "in the episode, the store) and nothing about earlier stimuli. On every "
        "step after the first, prepend the turn ordinal, the presentation ordinal, "
        "one fixed procedural sentence, and the store rendered UNCONDITIONALLY "
        "including '(memory is empty)'. First step of an episode: nothing "
        "prepended, because every line would be vacuous at block onset -- which "
        "also keeps the six encode()->recall() tasks byte-identical. JUSTIFICATION: "
        "Oberauer (2002) exempts task set / activated LTM from the chunk limit "
        "(Monsell 2003; Logan & Gordon 2001), and the ordinal counters are (a) "
        "provably non-diagnostic of stimulus content, being functions of the call "
        "count alone, and (b) exactly the always-available temporal-context signal "
        "every serial-order model posits outside the item store (Howard & Kahana "
        "2002; Brown, Preece & Hulme 2000; Burgess & Hitch 1999; Brown, Neath & "
        "Chater 2007). Humans in this task genuinely do have what the reset "
        "removed -- the block-onset event and a 2000 ms ISI tell them a turn has "
        "elapsed -- so its absence was a harness artifact, not a capacity limit. "
        "GENERAL, not an n-back special case: no task is named anywhere in the "
        "file, the same block goes on variable_mapping's turns, and n=1 is merely "
        "the one level where the model's own write policy leaves the store empty "
        "long enough for the defect to bite (n=2/n=3 instructions say 'two/three "
        "turns back', which cannot be met without storing, so keys_held is "
        "3.06/3.82 there). LEAK STAYS CLOSED BY CONSTRUCTION: the block adds only "
        "fixed text, two call-count integers, and the store; step() retains no "
        "reference to any earlier user_message. CAPACITY UNTOUCHED: MAX_KEYS 4, "
        "WorkingMemory not subclassed, write_key/overflow/eviction untouched "
        "(primacy owns them), recall() untouched. COMPOSITION: episodic_reset's "
        "RuntimeError on store-over-capacity is REMOVED, so composing a candidate "
        "overflow rule degrades a row instead of crashing a job. FALSIFIABLE: n=1 "
        "answered >= 13.0, acc_over_answered >= 0.90, 0 of 50 silent and keys_held "
        ">= 0.8 (so the answers come from the store, closing the hole full_context "
        "shows is reachable), a failure of which invalidates the rest; n=3 "
        "letter-identity share >= 0.90 with keys_held reported not thresholded, "
        "since a leak-free store may legitimately be a single rolling key; the "
        "anti-restoration test is instead A4 n_errors >= 150, because a restored "
        "transcript returns variable_mapping to the baseline's 12 errors; "
        "rc_ratio_normalized "
        "in [0.15, 0.95]; vm humanlikeness >= 0.55 raw and matched, with a FALL "
        "from 0.6764 expected and accepted; nback humanlikeness >= 0.7309, its "
        "floor, which episodic_reset violated at -0.2939; digit_span_reverse "
        "humanlikeness as the no-change control, point estimate exactly 0.9666 "
        "(held in 4/4 iteration-2 arms) but gated at its own 0.059 noise floor, "
        "because the serving stack is NOT row-deterministic -- 15 bit-identical "
        "'Next letter: C' requests at n=1 split 11/4."
    ),
}
