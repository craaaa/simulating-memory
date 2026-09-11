"""Build the STaR dataset D: 1:1 success/fail human trials, split train/eval.

STaR takes D as given; balancing it is a deliberate deviation. Without it the
match-to-human objective is satisfiable by always recalling correctly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .data import HumanTrial, by_length


@dataclass(frozen=True)
class Selection:
    train: List[HumanTrial]
    eval: List[HumanTrial]
    report: Dict[str, object]


def _hist(trials: Sequence[HumanTrial]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for (direction, length), cell in by_length(trials).items():
        out[f"{direction}:{length}"] = {
            "n": cell["n"],
            "n_correct": cell["n_correct"],
            "n_fail": cell["n"] - cell["n_correct"],
        }
    return out


def select(
    trials: Sequence[HumanTrial],
    *,
    seed: int,
    eval_frac: float,
) -> Selection:
    """All fail trials + an equal-sized random sample of success trials (global 1:1),
    then a random trial-level split stratified on (direction, correct).

    Known limitation: under global 1:1, fails concentrate at long spans and successes
    at short ones, so span length is a shortcut the model can exploit. The per-length
    eval curve is what detects it; per-cell balancing is a change to this function
    alone.
    """
    rng = random.Random(seed)

    fails = [t for t in trials if not t.correct]
    successes = [t for t in trials if t.correct]
    if len(successes) < len(fails):
        raise ValueError(
            f"cannot build 1:1 set: {len(fails)} fails but only {len(successes)} successes"
        )
    kept_successes = rng.sample(successes, len(fails))
    pool = fails + kept_successes

    # Stratified split on (direction, correct) so both strata appear in eval.
    strata: Dict[tuple, List[HumanTrial]] = {}
    for t in pool:
        strata.setdefault((t.direction, t.correct), []).append(t)

    train: List[HumanTrial] = []
    held: List[HumanTrial] = []
    for key in sorted(strata, key=lambda k: (k[0], k[1])):
        group = sorted(strata[key], key=lambda t: t.trial_id)
        rng.shuffle(group)
        n_eval = max(1, int(round(len(group) * eval_frac)))
        held.extend(group[:n_eval])
        train.extend(group[n_eval:])

    train.sort(key=lambda t: t.trial_id)
    held.sort(key=lambda t: t.trial_id)

    # Short sequences recur across participants; flag any presented sequence that
    # appears in both splits so eval isn't scored on a memorized stimulus.
    train_seqs = {(t.direction, tuple(t.digits)) for t in train}
    overlapping = [t for t in held if (t.direction, tuple(t.digits)) in train_seqs]
    overlap = sorted({f"{t.direction}:{''.join(map(str, t.digits))}" for t in overlapping})
    dropped_ids = [t.trial_id for t in overlapping]
    held = [t for t in held if t.trial_id not in set(dropped_ids)]

    report = {
        "seed": seed,
        "eval_frac": eval_frac,
        "pool_total": len(pool),
        "pool_fail": len(fails),
        "pool_success": len(kept_successes),
        "n_train": len(train),
        "n_eval": len(held),
        "train_fail": sum(1 for t in train if not t.correct),
        "eval_fail": sum(1 for t in held if not t.correct),
        "train_by_length": _hist(train),
        "eval_by_length": _hist(held),
        # Short sequences recur across participants; an eval trial whose presented
        # sequence also appears in train is dropped rather than scored.
        "stimulus_overlap_train_eval": overlap,
        "eval_dropped_for_stimulus_overlap": dropped_ids,
        "train_ids": [t.trial_id for t in train],
        "eval_ids": [t.trial_id for t in held],
    }
    return Selection(train=train, eval=held, report=report)


def restore(trials: Sequence[HumanTrial], report: Dict[str, object]) -> Selection:
    """Rebuild a Selection from a saved selection.json so every round uses the
    identical D (STaR regenerates rationales for the same dataset each round)."""
    index = {t.trial_id: t for t in trials}
    missing = [i for i in list(report["train_ids"]) + list(report["eval_ids"]) if i not in index]
    if missing:
        raise ValueError(f"selection.json references {len(missing)} unknown trial ids, e.g. {missing[:3]}")
    return Selection(
        train=[index[i] for i in report["train_ids"]],
        eval=[index[i] for i in report["eval_ids"]],
        report=report,
    )
