"""Build D for listening QA and split it BY PARTICIPANT.

Deliberately not `rationales.select.select`. Two things that selector does are wrong
here, and both fail loudly rather than subtly if reused:

  * It drops any held-out trial whose stimulus also appears in train. Digit span has
    hundreds of distinct sequences; listening QA has sixteen stimuli (4 topics x 4
    levels) and every one of them is in train, so that rule would empty the eval split.
  * It subsamples successes down to a 1:1 match with failures. The listening pool is
    already 46.7% exact-correct, so that would throw away half the data to fix an
    imbalance that isn't there.

What replaces them is a participant-level split, which is required rather than
cosmetic: an item's prompt carries that same participant's answers to the other four
questions of the passage, so splitting per item would put a held-out item's own
sibling answers into the training set.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence

from ..select import Selection
from .data import ListeningItem, by_cell

# Cells outside this band are flagged in the report. A (topic, level, question) cell
# pinned near 0 or 1 is learnable from the question's shape alone, which is not the
# thing the fine-tune is supposed to pick up.
DEGENERATE_LOW = 0.15
DEGENERATE_HIGH = 0.85


def _cell_hist(items: Sequence[ListeningItem]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for (topic, level, q_id), cell in by_cell(items).items():
        n = cell["n"]
        out[f"{topic}:{level}:{q_id}"] = {
            "n": n,
            "n_correct": cell["n_correct"],
            "n_fail": n - cell["n_correct"],
            "correct_share": (cell["n_correct"] / n) if n else None,
        }
    return out


def _degenerate(hist: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for key, cell in hist.items():
        share = cell["correct_share"]
        if share is None:
            continue
        if share < DEGENERATE_LOW or share > DEGENERATE_HIGH:
            out.append({"cell": key, "n": cell["n"], "correct_share": round(share, 3)})
    return sorted(out, key=lambda d: d["correct_share"])


def select(
    items: Sequence[ListeningItem],
    *,
    seed: int,
    eval_frac: float,
) -> Selection:
    """Hold out whole participants, stratified on the counterbalancing group.

    No subsampling: every item of every participant goes to exactly one side.
    """
    rng = random.Random(seed)

    by_group: Dict[str, List[str]] = defaultdict(list)
    for it in items:
        key = it.group_id if it.group_id is not None else "ungrouped"
        if it.respondent_id not in by_group[key]:
            by_group[key].append(it.respondent_id)

    held_ids: set = set()
    for group in sorted(by_group):
        respondents = sorted(by_group[group])
        rng.shuffle(respondents)
        # 24 groups at a 15% holdout is ~1.25 respondents each, so round up rather than
        # down: a group contributing zero held-out participants would silently drop a
        # counterbalancing condition out of eval entirely.
        n_eval = max(1, int(round(len(respondents) * eval_frac)))
        held_ids.update(respondents[:n_eval])

    train = [it for it in items if it.respondent_id not in held_ids]
    held = [it for it in items if it.respondent_id in held_ids]

    train.sort(key=lambda it: it.item_id)
    held.sort(key=lambda it: it.item_id)

    train_cells = _cell_hist(train)
    eval_cells = _cell_hist(held)
    eval_pairs = sorted({f"{it.topic}:{it.level}" for it in held})
    all_pairs = sorted({f"{it.topic}:{it.level}" for it in items})

    report = {
        "split": "by_respondent",
        "seed": seed,
        "eval_frac": eval_frac,
        "pool_total": len(items),
        "pool_fail": sum(1 for it in items if not it.correct),
        "pool_success": sum(1 for it in items if it.correct),
        "subsampled": False,
        "n_train": len(train),
        "n_eval": len(held),
        "n_train_respondents": len({it.respondent_id for it in train}),
        "n_eval_respondents": len(held_ids),
        "train_fail": sum(1 for it in train if not it.correct),
        "eval_fail": sum(1 for it in held if not it.correct),
        "groups": len(by_group),
        "eval_respondents_by_group": {
            g: sum(1 for r in by_group[g] if r in held_ids) for g in sorted(by_group)
        },
        "train_by_cell": train_cells,
        "eval_by_cell": eval_cells,
        # A cell here is not a reason to stop, but it is a reason to decide something
        # before training rather than to discover it in the eval curve afterwards.
        "degenerate_cells_train": _degenerate(train_cells),
        # The whole point of stratifying: every (topic, level) must be scoreable.
        "topic_level_pairs_total": len(all_pairs),
        "topic_level_pairs_in_eval": len(eval_pairs),
        "topic_level_pairs_missing_from_eval": sorted(set(all_pairs) - set(eval_pairs)),
        "train_ids": [it.item_id for it in train],
        "eval_ids": [it.item_id for it in held],
    }

    if not held:
        raise ValueError(
            f"empty eval split from {len(items)} items at eval_frac={eval_frac}"
        )
    overlap = {it.respondent_id for it in train} & {it.respondent_id for it in held}
    if overlap:
        raise AssertionError(
            f"{len(overlap)} respondents appear in both splits, e.g. {sorted(overlap)[:3]}. "
            "Their sibling-answer context would leak across the split."
        )

    return Selection(train=train, eval=held, report=report)


def restore(items: Sequence[ListeningItem], report: Dict[str, Any]) -> Selection:
    """Rebuild a Selection from a saved selection.json, so every round trains on the
    identical D (STaR regenerates rationales for the same dataset each round)."""
    index = {it.item_id: it for it in items}
    wanted = list(report["train_ids"]) + list(report["eval_ids"])
    missing = [i for i in wanted if i not in index]
    if missing:
        raise ValueError(
            f"selection.json references {len(missing)} unknown item ids, e.g. {missing[:3]}"
        )
    return Selection(
        train=[index[i] for i in report["train_ids"]],
        eval=[index[i] for i in report["eval_ids"]],
        report=report,
    )
