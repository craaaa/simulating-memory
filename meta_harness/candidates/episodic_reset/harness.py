"""Candidate `episodic_reset`: the store becomes the only thing that carries a trial.

THE DEFECT
----------
`WorkingMemoryAgent.step()` maintains the full conversation history across calls
(`bench/core/wm_agent.py:176`, "Conversation history is maintained across calls").
`reset_messages()` exists at line 161 and is called by nothing, anywhere in
`bench/` or `src/`. So on the two tasks that answer through `step()` -- `nback`
and `variable_mapping` -- every stimulus the agent has ever been shown is still
in its context verbatim when it answers. The 4-slot store is decorative there.
`recall()` (line ~373) is the contrast case: it builds a single fresh prompt from
`wm.to_recall_text()` alone, which is why the six tasks that answer through
`recall()` are the six where the bottleneck actually binds.

The decisive number is in the n=3 n-back stores, and it is stronger evidence than
anything previously recorded. Counting `final_kv` values that contain a bare
capital letter (i.e. that carry letter identity at all), in
`meta_harness/runs/iter0/baseline/.../wm_nback.jsonl`:

    n   final_kv values   carry a letter   acc_over_answered
    1         50            50  (1.000)         0.994
    2         77            77  (1.000)         0.821
    3        198             4  (0.020)         0.737

At n=3 the store contains **no letters at all** -- the keys are `position_2`,
`position_3`, `position_8`, `position_10` holding contentless labels like
"seventeenth letter in sequence" -- and the agent is nevertheless 0.737 accurate
on the trials it answers. A store with no letter information cannot support
0.737 on a 3-back judgement. The answers are coming from the dialogue history.

The same holds under `displacement` (iteration 1): its n=3 store is a tidy
contiguous `{position_16..19}`, still only 4 of 199 values carry a letter, and it
answers 14/14 at 0.744 for an n=3 humanlikeness of 0.9387. **That number is
leak-derived.** Displacement's confirmed result is about response *production* --
the refusal was suppressing answers -- not about the store becoming load-bearing.
Worth recording, because "n-back per level is closed" is not the same claim as
"the memory module closed n-back".

On `variable_mapping` the leak was already measured (WORKLOG): 674 of 1500
questions ask about a name the store has already evicted and are answered at
0.985; where the store holds a *stale* value contradicting the truth the model
overrides its own memory and is still right 93.5% of the time.

WHAT SURVIVES A RESET, AND WHY
------------------------------
"Call `reset_messages()` after every step" is the naive reading and it is wrong,
for a reason that is visible in the code rather than merely psychological:
`step()` never shows the agent its own store. The only read access `step()` ever
had to `wm` was the conversation history of its own past tool calls. Wipe that
and the agent is not memory-limited, it is blind to its own current mental
contents -- a state no account of working memory posits, and one that would
degrade n-back by preventing any strategy at all rather than by imposing a
capacity limit. So a total wipe is the wrong intervention twice over.

The account used here is Oberauer's (2002) concentric model of working memory --
Cowan (2001) with the access structure made explicit -- because it is the account
that licenses exactly three things and no fourth:

  * **focus of attention**, one item, currently being processed
        -> the incoming `user_message` (this letter, these two statements, this
           question). Kept.
  * **region of direct access**, ~4 chunks, directly retrievable
        -> the 4-slot key-value store, `wm.to_recall_text()`. Kept, and now
           *rendered into the turn*, because a region of direct access that the
           processing system cannot read is not a region of direct access.
  * **activated long-term memory / task set** -- procedural knowledge of what the
    task is and how to act in it, which is not subject to the chunk limit
        -> the system prompt (which for these tasks already carries the complete
           original human instructions, see below) and the `TOOLS` schemas. Kept.

There is **no component in this model corresponding to a verbatim record of
previously presented, no-longer-attended stimuli.** That is precisely what the
conversation history is, and that is why all of it goes.

Justifying each inclusion and exclusion:

  * *System prompt: kept.* Task set is procedural and is retrieved from LTM, not
    held in the capacity-limited store (Monsell 2003 on task set; Logan & Gordon
    2001). Humans in these tasks do not forget what the task is. Verified to be
    sufficient rather than assumed: for n-back the C2 system prompt is
    `wm_system_prompt(..., human_task_prompt=HUMAN_PROMPT_BY_N[n])`, and
    `HUMAN_PROMPT_BY_N[n]` states the full rule -- "respond with 'no response' to
    the first n letter(s)", "respond to each new letter as 'same' or
    'different'" -- plus a worked example with its explanation. The task's own
    instruction turn (`wm_nback.py:86`) is therefore redundant with the system
    prompt, so **nothing needs pinning** and the rule "system prompt + store +
    current stimulus, nothing else" is implemented with no exceptions.
  * *Tool schemas: kept.* Affordances, not episodic content. They are how the
    agent acts, and their text is unchanged from the baseline.
  * *The store's contents: kept, and now visible.* See above.
  * *The last N turns: excluded, N = 0.* This is the inclusion that would look
    most humane and is the one that must not be made. At n=1 keeping one turn
    hands over the answer outright; at n=2 and n=3 it hands over part of it. The
    size of the leak would then scale with n, which would confound the per-level
    comparison (`nback_levels.py`) that is the measurement this candidate exists
    to make honest. More fundamentally, a verbatim window of the last N turns is
    a second memory store with unbounded fidelity and no declared capacity --
    `full_context` at small scale, and `full_context` is the least humanlike
    harness measured (0.6387 against 0.7861). The "sense of recent context" a
    human retains is exactly what the agent chose to write into its four slots.
    That is the whole claim of the compactor.
  * *The agent's own prior assistant turns and tool calls: excluded.* These are
    an episodic record of its own encoding events, and on n-back they are
    stimulus-correlated: a "same" emitted at position k reveals that letter k
    equalled letter k-n. Keeping them would reintroduce the leak through the
    agent's own mouth. Within a single step the tool results stay, because the
    agent must see whether its write succeeded -- the reset happens only at turn
    boundaries, never inside the tool-dispatch loop (which would also emit an
    orphaned `tool` message and break the OpenAI message protocol).

The alternative account considered and rejected: Baddeley's (2000) episodic
buffer with its LTM interface. It would license a gist-level residue of earlier
stimuli living *outside* the four slots. The harness has no representation for
that, so implementing it would mean adding a second store -- capacity growth by
the back door, on a search where capacity has already been shown to be the wrong
direction.

WHAT CHANGED IN THE PROMPT, AND WHY IT WAS FORCED
-------------------------------------------------
One addition, on turns delivered through `step()` when the store is non-empty:
the current store is rendered at the top of the turn as

    Your working memory currently contains:
    <wm.to_recall_text()>

    <the task's own user message>

Forced, per the argument above: without it `step()` gives the agent no read
access to its own store and the candidate would degrade by blindness rather than
by capacity limit. The wording is not new text -- "Your working memory currently
contains:" is taken verbatim from `RECALL_PROMPT` in `wm_agent.py` and from
`QUESTION_PROMPT` in `wm_variable_mapping.py`, and the rendering method is the
same `to_recall_text()` those prompts use. Where the task's own message already
contains that header (variable_mapping's question turn does) the block is not
duplicated, so on those turns the prompt is byte-identical to the baseline's
apart from the missing history.

Nothing else changes. `TOOLS`, `CONDITION_PROMPTS`, `WM_SYSTEM_PROMPTS`,
`MAX_KEYS`, `WorkingMemory` (including `write_key` and the overflow policy, which
proposer 2a owns) and `recall()` (including the word_recognition trial list,
which proposer 2b owns) are all untouched. Exactly one method body differs from
`baseline`: `step()`.

NOT A CAPACITY CHANGE, AND NOT NOISE
------------------------------------
`MAX_KEYS` stays 4 and `WorkingMemory` is not subclassed, so the capacity
invariant is literally the baseline's. The change is strictly *subtractive* in
what the model can see: it removes an information channel and adds no
stochasticity of any kind. There is no random number generator in this file. The
mechanism is deterministic and per-turn identical across runs, which is why the
no-change control below is byte-identical rather than merely within-noise.

It also cannot be `random_decay_v2` with extra steps. That candidate bought the
aggregate statistic (best_span 18.4 -> 8.2 against a human 6.88) while making the
error structure twice as wrong (A1 sub-span leak 0.249 against a human 0.087),
because it dropped content the agent had chosen to keep. This candidate drops
nothing the agent chose to keep -- the four slots are exactly what the agent put
there -- and on the six tasks that route through `recall()` it changes literally
nothing, so it cannot buy an aggregate anywhere except the two tasks whose defect
it is addressing.

The tool-call cap is untouched. Note the direction: rendering the store makes
*overwriting* an existing key discoverable, which is one tool call where the
baseline's blind delete-then-write repair costs two against a budget of
`max(6, 1.5 * steps)`. So this candidate can reduce tool-call demand without
raising supply, and a gain cannot be read as "you gave it more compute" -- it
gets strictly less context.

PREDICTIONS
-----------
See `MANIFEST.md` in this directory and the machine-readable form in
`meta_harness/logs/pending_episodic_reset.json`.
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


def _capacity() -> int:
    """Read capacity live, so an injected `MAX_KEYS` is honoured."""
    return int(getattr(_wm_mod, "MAX_KEYS", MAX_KEYS))


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness with no episodic record of earlier turns.

    Every `step()` starts from a fresh message list. What the agent sees on turn
    k is: the system prompt (task set), the current contents of the 4-slot store,
    and the current stimulus. Nothing else -- no earlier stimuli, no earlier
    replies of its own, no earlier tool calls.

    Only `step()` is overridden. `encode()`, `recall()`, segmentation, key
    naming, value formatting, the tool-call cap, the prompts, the tools and
    `WorkingMemory` itself are the baseline's.

    Structural consequence worth stating, because it is this candidate's
    attribution guarantee: `encode()` calls `step()` exactly once, and the six
    tasks that go `encode()` -> `recall()` construct a fresh agent per trial
    (verified at wm_digit_span_forward.py:209, wm_digit_span_reverse.py:110,
    wm_semantic_story_recall.py:165, wm_mcq_common.py:52,
    wm_word_recognition.py:94). On that single step there is no history to drop
    and the store is empty, so no state block is prepended. Those six tasks are
    therefore byte-identical to the baseline, and only `nback` and
    `variable_mapping` -- the two tasks that call `step()` repeatedly, and the
    two that leak -- can move at all.
    """

    def step(self, user_message: str, *, allow_tools: bool = True,
             max_tokens: int = 1024) -> str:
        # Turn boundary. Doing this here and not inside the tool loop is load
        # bearing: clearing mid-loop would leave a `tool` message with no
        # preceding assistant `tool_calls` and the request would be rejected.
        self.reset_messages()

        store = self.wm.store
        if len(store) > _capacity():
            raise RuntimeError(
                f"store holds {len(store)} keys, capacity is {_capacity()}"
            )

        if store and WM_STATE_HEADER not in user_message:
            # The region of direct access, made readable. See the module
            # docstring: without this, wiping the history leaves the agent
            # unable to inspect its own store at all.
            user_message = (
                f"{WM_STATE_HEADER}\n{self.wm.to_recall_text()}\n\n{user_message}"
            )

        return super().step(user_message, allow_tools=allow_tools,
                            max_tokens=max_tokens)


MANIFEST: dict[str, Any] = {
    "id": "episodic_reset",
    "parent": "baseline",
    "capacity": MAX_KEYS,
    "decay": (
        "none. No stochastic loss of any kind and no RNG in the file. The only "
        "change is that the conversation history is not carried across turns, so "
        "the 4-slot store is the sole route from an earlier stimulus to an "
        "answer. Deterministic and per-turn identical across runs."
    ),
    "role": "contender -- iteration 2c, closes the conversation-history leak",
    "summary": (
        "Closes the context leak on the two tasks that answer through step(). "
        "DEFECT: WorkingMemoryAgent.step() maintains conversation history across "
        "calls and reset_messages() (wm_agent.py:161) is called by nothing in "
        "bench/ or src/, so on nback and variable_mapping every stimulus stays in "
        "context verbatim and the 4-slot store is decorative. DECISIVE EVIDENCE, "
        "new here: at n=3 only 4 of 198 baseline final_kv values contain a bare "
        "capital letter (n=1: 50/50, n=2: 77/77) while acc_over_answered is 0.737 "
        "-- a store with no letter information cannot support a 3-back judgement, "
        "so the answers come from history. Same under displacement (4/199, 0.744, "
        "n=3 HL 0.9387), so displacement's n=3 result is response production, not "
        "a load-bearing store. MECHANISM: step() rebuilds its message list each "
        "turn. Survives, on Oberauer's (2002) concentric model: the system prompt "
        "and TOOLS (task set / activated LTM, not chunk-limited -- Monsell 2003), "
        "the 4-slot store rendered into the turn (region of direct access), and "
        "the current stimulus (focus of attention). Excluded: all earlier "
        "stimuli, the agent's own earlier replies and tool calls, and any "
        "last-N-turns window -- N=0, because a verbatim recent window is a second "
        "uncapped store (full_context at small scale) and its size would scale "
        "with n, confounding the per-level nback comparison. PROMPT DELTA: one "
        "forced addition, the store rendered as 'Your working memory currently "
        "contains:' + to_recall_text(), wording taken verbatim from RECALL_PROMPT "
        "and variable_mapping's QUESTION_PROMPT, skipped when the task's own "
        "message already renders it. Without it step() gives the agent no read "
        "access to its own store and the candidate would degrade by blindness "
        "rather than by capacity. NOT CAPACITY, NOT NOISE: MAX_KEYS 4, "
        "WorkingMemory not subclassed, write_key and overflow untouched (2a), "
        "recall() untouched (2b), strictly subtractive in what the model sees, no "
        "RNG. ATTRIBUTION: encode() calls step() once and the other six tasks "
        "build a fresh agent per trial, so on that single step there is no "
        "history to drop and the store is empty -- those six tasks are "
        "byte-identical, verified offline against a stub LLM. FALSIFIABLE: "
        "variable_mapping A4 n_errors >= 150 (error_rate >= 0.10 against "
        "baseline 0.008, displacement 0.023) with rc_ratio > 1.15; share of runs "
        "with first_error_at <= 5 rises from 0.013 to > 0.25 (the only errors the "
        "score can see, since score = relation_count of the last correct question "
        "and relation_count = min(2q,10) saturates at q5); n=3 share of final_kv "
        "values carrying a letter rises from 0.020 to > 0.40; n=1 "
        "acc_over_answered >= 0.90 and answered >= 13/14, which fails if the "
        "reset merely confuses the model, since one key suffices at n=1. Expects "
        "to pay for it on nback, possibly through the -0.060 floor, because "
        "refuse-on-full is still in place and 2a owns that fix."
    ),
}
