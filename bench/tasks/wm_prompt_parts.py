"""Prompt composition helpers for working-memory tasks.

The normal task modules own the participant-facing task language.  WM tasks add
only the key-value memory mechanics and answer-from-memory framing here.
"""

from __future__ import annotations

import os
from typing import Dict

from ..core.working_memory import MAX_KEYS

CONDITIONS: Dict[str, Dict[str, str]] = {
    "C2": {"name": "Human cognitive limits framing"},
}


def wm_system_prompt(
    condition_id: str,
    *,
    task_prompt: str,
    human_task_prompt: str,
) -> str:
    """Compose a WM-agent system prompt from original task language plus WM mechanics."""
    if condition_id == "C2":
        return f"""\
You are simulating a human participant in a psychology experiment on working memory.
You have a key-value memory store with exactly {MAX_KEYS} slots, reflecting the ~4-chunk
limit of human short-term memory (Cowan, 2001). There is no "add" and no "delete" — the
only action is replace_key(old_key, new_key, value), which replaces whatever currently
occupies one slot with a new label and value. All {MAX_KEYS} slots exist from the start;
empty ones start out as placeholder keys — but their exact names are NOT fixed and change as
slots fill up. NEVER assume a placeholder's name from memory or from an earlier turn; always
read the CURRENT memory contents shown below this turn and use the exact key text shown
there. A slot's placeholder name from an earlier segment may already be gone.

Original human-task instructions:
{human_task_prompt.strip()}

Each slot should hold ONE chunk — a small bundle of information a person would bind
together because it feels meaningfully connected (a name with its role, a group of related
items or numbers, one gist). When the task asks for verbatim retrieval of a sequence, a
human will form meaningful chunks of 1–3 items, starting from the beginning.

A chunk is ONE atomic fact, not several facts stitched together even if they're about the
same subject. A single sentence can pack in several distinct attributes of one thing —
what it is, where it's from, how it was made or found, some measurement. Each attribute is
its own chunk. Don't merge them into a single slot just because they share a subject.
Keep new_key under 4 words and value under 10 words — short enough that packing two facts
into either one becomes obviously impossible.

Three ways to use replace_key:
  - FILL AN EMPTY SLOT: old_key = the exact key of an empty slot AS CURRENTLY SHOWN below
    (read it, don't guess it), new_key = your label
    for the new chunk, value = the summary.
  - AMEND something you already stored: old_key AND new_key = the SAME existing key, value
    = the complete updated chunk. This replaces the old wording entirely (it does not
    merge), so carry forward whatever from the old value is still worth keeping, plus the
    new detail, in one value — don't drop the old content, and don't skip the new detail.
    Amend when new material adds to, corrects, or refines something you already have a
    chunk for (same person, place, event, or concept).
  - EVICT to make room: old_key = the entry you're evicting, new_key = the new label. Only
    do this when all slots are full of real content and the new fact is genuinely separate
    from anything you're keeping (not just related to the same topic). Choose what to evict
    in this priority: (a) an entry that duplicates or is subsumed by another you're keeping,
    (b) whichever entry seems least useful or important on its own, judged independently of
    the others — hints for "least useful": background or scene-setting detail rather than a
    specific name/number/relationship, something not likely to matter for questions about
    the material, or the entry you'd be least confident recalling correctly anyway.

You may issue several replace_key calls in the same turn, but each must target a
DIFFERENT old_key — two calls can't replace the same slot at once.

NEVER pack a long run of items into one slot. Once your real slots are all full, accept
that the rest will be lost. Compress realistically, and behave as a real human would:
imperfect and sensitive to what seems important at the time you encounter it."""

    if condition_id == "C2-stream":
        return f"""\
You are simulating a human participant in a psychology experiment on working memory.
You have a key-value memory store with exactly {MAX_KEYS} slots, reflecting the ~4-chunk
limit of human short-term memory (Cowan, 2001). There is no "add" and no "delete" — the
only action is replace_key(old_key, new_key, value), which replaces whatever currently
occupies one slot with a new label and value. All {MAX_KEYS} slots exist from the start;
empty ones start out as placeholder keys — but their exact names are NOT fixed and change as
slots fill up. NEVER assume a placeholder's name from memory or from an earlier turn; always
read the CURRENT memory contents shown below this turn and use the exact key text shown
there. A slot's placeholder name from an earlier segment may already be gone.

Original human-task instructions:
{human_task_prompt.strip()}

The material arrives as a sequence of ordered segments, one at a time — like listening
once, with no replay. You will NOT see a segment again once you move past it, so decide
now what's worth keeping. Each turn you'll see your current {MAX_KEYS}-slot memory contents
and the new segment.

Each slot should hold ONE chunk — a small bundle of information a person would bind
together because it feels meaningfully connected (a name with its role, a group of related
items or numbers, one gist).

A chunk is ONE atomic fact, not several facts stitched together even if they're about the
same subject. A single segment can pack in several distinct attributes of one thing —
what it is, where it's from, how it was made or found, some measurement. Each attribute is
its own chunk. Replace separate empty slots for each, rather than merging them into a
single slot just because they share a subject. Keep new_key under 4 words and value under
10 words — short enough that packing two facts into either one becomes obviously
impossible.

Three ways to use replace_key:
  - FILL AN EMPTY SLOT: old_key = the exact key of an empty slot AS CURRENTLY SHOWN below
    (read it, don't guess it), new_key = your label
    for the new chunk, value = the summary.
  - AMEND something you already stored: old_key AND new_key = the SAME existing key, value
    = the complete updated chunk. This replaces the old wording entirely (it does not
    merge), so carry forward whatever from the old value is still worth keeping, plus the
    new detail, in one value — don't drop the old content, and don't skip the new detail.
    Amend when a new segment adds to, corrects, or refines something you already have a
    chunk for (same person, place, event, or concept). Use this often, not only when every
    slot is full — it's how you update a chunk as you learn more over the course of the
    material.
  - EVICT to make room: old_key = the entry you're evicting, new_key = the new label. Only
    do this when every slot already holds real content and the new fact is genuinely
    separate from anything you're keeping (not just related to the same topic). Choose what
    to evict in this priority: (a) an entry that duplicates or is subsumed by another you're
    keeping, (b) whichever entry seems least useful or important on its own, judged
    independently of the others — hints for "least useful": background or scene-setting
    detail rather than a specific name/number/relationship, something not likely to matter
    for questions about the material, or the entry you'd be least confident recalling
    correctly anyway.

If nothing in this segment is worth keeping or amending, make no tool calls this turn. You
may issue several replace_key calls in the same turn, but each must target a DIFFERENT
old_key — two calls can't replace the same slot at once. You'll see the result (success or
error) of each call before your next action — use that feedback rather than repeating an
identical call that just failed.

NEVER pack a long run of items into one slot. Once your real slots are all full, accept
that the rest will be lost. Compress realistically, and behave as a real human would:
imperfect and sensitive to what seems important at the time you encounter it."""

    raise KeyError(f"Unknown condition_id: {condition_id}")


def wm_system_prompts(
    *,
    task_prompt: str,
    human_task_prompt: str,
) -> Dict[str, str]:
    """Build WM system prompts."""
    return {
        cond_id: wm_system_prompt(
            cond_id,
            task_prompt=task_prompt,
            human_task_prompt=human_task_prompt,
        )
        for cond_id in CONDITIONS
    }


def wm_recall_prompt(
    *,
    task_prompt: str,
    wm_recall_instructions: str,
    format_rules: str,
) -> str:
    """Compose a recall prompt. The returned string keeps ``{wm_contents}`` for recall()."""
    return f"""\
The study phase is now over.
Your working memory currently contains:
{{wm_contents}}

Original task instructions:
{task_prompt.strip()}

Based ONLY on the above contents, {wm_recall_instructions.strip()}

{format_rules.strip()}"""


def wm_mcq_recall_preamble(*, task_prompt: str, answer_focus: str) -> str:
    """Build the task-specific preamble used by shared WM MCQ recall prompts."""
    return (
        "Original task instructions:\n"
        f"{task_prompt.strip()}\n\n"
        f"Based ONLY on the above contents, answer the following questions {answer_focus}."
    )


# ---------------------------------------------------------------------------
# Summarizer ablation: simpler baseline than the WM key-value agent.
# Reads material once, produces an abstractive summary, then answers from it.
# ---------------------------------------------------------------------------

ALL_SUMMARIZER_CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {"name": "Summarizer baseline"},
    "C2": {"name": "Summarizer baseline + human participant framing"},
}


def _selected_summarizer_conditions() -> Dict[str, Dict[str, str]]:
    raw = os.getenv("BENCH_SUMMARIZER_CONDITIONS", "").strip()
    if not raw:
        return {"C1": ALL_SUMMARIZER_CONDITIONS["C1"]}
    selected: Dict[str, Dict[str, str]] = {}
    for cond_id in [p.strip() for p in raw.split(",") if p.strip()]:
        if cond_id not in ALL_SUMMARIZER_CONDITIONS:
            raise KeyError(f"Unknown summarizer condition_id: {cond_id}")
        selected[cond_id] = ALL_SUMMARIZER_CONDITIONS[cond_id]
    return selected


SUMMARIZER_CONDITIONS: Dict[str, Dict[str, str]] = _selected_summarizer_conditions()


def summarizer_system_prompt(task_prompt: str, condition_id: str = "C1") -> str:
    """Compose a summarizer system prompt, parametrized by task and condition."""
    if condition_id == "C1":
        prefix = "You are a summarizer agent."
    elif condition_id == "C2":
        prefix = (
            "You are a summarizer agent.\n"
            "You are simulating a human participant in a psychology experiment."
        )
    else:
        raise KeyError(f"Unknown summarizer condition_id: {condition_id}")

    return f"""\
{prefix}

Original task instructions:
{task_prompt.strip()}

You will first be shown material to remember. Produce a concise abstractive
summary of it — keep the summary short (prefer brief, dense prose; aim for
roughly a paragraph, not a transcript). You will later have to answer questions
using ONLY your summary, so make sure the summary captures what you'll need
for the task above."""


def summarizer_system_prompts(task_prompt: str) -> Dict[str, str]:
    """Build condition-specific summarizer system prompts."""
    return {
        cond_id: summarizer_system_prompt(task_prompt, condition_id=cond_id)
        for cond_id in SUMMARIZER_CONDITIONS
    }


def summarizer_recall_prompt(
    task_prompt: str,
    recall_instructions: str,
    format_rules: str,
) -> str:
    """Compose a recall prompt for the summarizer. Keeps ``{summary}`` placeholder."""
    return f"""\
The study phase is now over.
Your summary of the material is:
{{summary}}

Original task instructions:
{task_prompt.strip()}

Based ONLY on the above summary, {recall_instructions.strip()}

{format_rules.strip()}"""


def summarizer_mcq_recall_preamble(task_prompt: str, answer_focus: str) -> str:
    """Build the task-specific preamble used by the summarizer MCQ recall prompt."""
    return (
        "Original task instructions:\n"
        f"{task_prompt.strip()}\n\n"
        f"Based ONLY on the above summary, answer the following questions {answer_focus}."
    )
