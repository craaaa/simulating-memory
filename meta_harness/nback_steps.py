"""Which step-log entry belongs to which n-back letter, across both row generations.

WHY THIS EXISTS. Every n-back analysis in this project hand-rolled the same positional rule:
`step_log[0]` is the instruction turn, `step_log[i]` is letter *i*, and scored trial *k* is
`step_log[n + k]`. That rule described rows written before `exp/compactor-prefix-v1` and is
WRONG for rows written after it: `wm_nback` now issues TWO turns per letter -- an encode turn
(store + letter, tools ON) and an answer turn (store + restated instructions, tools OFF, no
stimulus) -- and there is no instruction turn at all.

Reading a post-fix row with the old rule silently mixes the two turn kinds, which inflates
any per-turn count by about 2x and looks for a classification in bookkeeping text. The
project has had to retract three numbers already; this module exists so the fourth is not a
row-format confusion.

WHICH KIND TO ASK FOR.
  * tool activity -- `memory is full`, budget, displacement, cap hits -> ENCODE turns. The
    answer turn runs with `allow_tools=False`, so its budget fields describe nothing.
  * the reply -- classification, unparsed replies, spoken tool calls -> ANSWER turns.
For pre-fix rows both kinds return the same entries, because one turn did both jobs.

Post-fix rows carry `answer_step_by_position`, so the mapping is read rather than inferred.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def is_split(row: Dict[str, Any]) -> bool:
    """True if this row has one encode turn and one answer turn per letter."""
    return bool(row.get("answer_step_by_position") or row.get("answer_steps"))


def _by_position(row: Dict[str, Any]) -> Dict[int, int]:
    """position (1-based letter index) -> index into step_log, for split rows."""
    raw = row.get("answer_step_by_position") or {}
    if raw:
        return {int(k): int(v) for k, v in raw.items()}
    # Fall back to the parallel lists if the dict is absent.
    return {i + 1: int(v) for i, v in enumerate(row.get("answer_steps") or [])}


def answer_turns(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """position -> the step-log entry whose reply was scored for that letter."""
    log = row.get("step_log") or []
    if not log:
        return {}
    if is_split(row):
        return {pos: log[i] for pos, i in _by_position(row).items() if i < len(log)}
    # Pre-fix: step 0 is the instruction turn, step i is letter i.
    return {i: log[i] for i in range(1, len(log))}


def encode_turns(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """position -> the step-log entry where that letter's tool calls were made."""
    log = row.get("step_log") or []
    if not log:
        return {}
    if is_split(row):
        enc = row.get("encode_steps") or []
        if enc:
            return {i + 1: log[v] for i, v in enumerate(enc) if v < len(log)}
        # No `encode_steps` recorded. The encode turn is adjacent to its answer turn, but
        # which side depends on the generation: the first split ran encode-then-answer, and
        # since 2026-09-29 n-back runs answer-then-encode (encoding first overwrote the
        # comparison target at n=1). So pick the neighbour that is not itself an answer step
        # rather than assuming a side.
        ans_idx = set(_by_position(row).values())
        out: Dict[int, Dict[str, Any]] = {}
        for pos, i in _by_position(row).items():
            for j in (i + 1, i - 1):
                if 0 <= j < len(log) and j not in ans_idx:
                    out[pos] = log[j]
                    break
        return out
    return {i: log[i] for i in range(1, len(log))}


def scored_answer_turns(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """scored trial k (1-based) -> its answer turn.

    The first `n_level` letters are lead-in: "no response" is the correct reply there and
    they are recorded in `buffer_map`, never in the 14 scored trials.
    """
    n = int(row.get("n_level") or 1)
    ans = answer_turns(row)
    n_trials = len(row.get("per_trial") or []) or 14
    out: Dict[int, Dict[str, Any]] = {}
    for k in range(1, n_trials + 1):
        st = ans.get(n + k)
        if st is not None:
            out[k] = st
    return out


def counts(row: Dict[str, Any]) -> Tuple[int, int]:
    """(n encode turns, n answer turns) -- useful for asserting a row's generation."""
    return len(encode_turns(row)), len(answer_turns(row))
