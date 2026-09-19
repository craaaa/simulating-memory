"""Held-out evaluation for listening QA.

Read this with the same ceiling in mind as the digit-span evaluator: on items the human
got wrong, the model has to predict WHICH options that particular person endorsed, and
that is genuinely stochastic. Exact set match is the right STaR *filter* but a poor
headline metric. The distributional readings -- the by-level curve and the endorsement
profile over option types -- are what to read.

The profile deliberately reuses labels that already exist rather than inventing a
taxonomy. Every option in the bank is labelled blind in the analysis pipeline with an
``option_type`` (true / false_interference / false_plain) and a ``cue_match`` (exact
lift from the passage / paraphrase), and those labels ride along on every row of
responses.csv. "Which kind of option did they endorse" is therefore free, and it is the
listening analogue of digit span's truncation/transposition profile: interference foils
are what a decayed memory reaches for, and exact-cue options are what survives.

The metric key names here are not free choices. ``rationales.evaluate.compare`` reads
four paths by literal name -- overall.human_match_rate,
human_fail_trials.human_match_rate, overall.ground_truth_accuracy and
overall.error_profile_tv_distance. Rename a block and every headline delta in eval.json
comes out null while the tests still pass.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import errors as err
from .data import ListeningItem

OPTION_TYPES = ("true", "false_interference", "false_plain")
CUE_MATCHES = ("exact", "paraphrase")

# Distribution support for the endorsement profile. "none" covers an item where nothing
# was endorsed at all, which is a real human behavior and must not vanish from a
# normalized distribution.
CATEGORIES: Tuple[str, ...] = tuple(
    f"{t}:{c}" for t in OPTION_TYPES for c in CUE_MATCHES
) + ("none",)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def option_category(labels: Dict[Any, Any], option_n: int) -> str:
    """The (option_type, cue_match) cell an option falls in."""
    pair = labels.get(option_n) or labels.get(str(option_n))
    if not pair:
        return "unlabelled"
    option_type, cue_match = pair[0], pair[1]
    return f"{option_type}:{cue_match}"


def endorsement_profile(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Normalized distribution over option cells of everything ``key`` endorsed.

    Not an endorsement *rate* per cell: a rate vector does not sum to 1 and total
    variation over it means nothing. This is "of all the options selected, what share
    were interference foils", which is a distribution and is what TV distance compares.
    """
    counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
    for r in rows:
        selected = r.get(key) or []
        if not selected:
            counts["none"] += 1
            continue
        for n in selected:
            cat = option_category(r.get("option_labels") or {}, n)
            counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values()) or 1
    return {c: counts.get(c, 0) / total for c in sorted(counts)}


def eval_row(
    item: ListeningItem,
    reasoning: Optional[str],
    answer: List[int],
    raw: str,
    parse_errors: List[str],
) -> Dict[str, Any]:
    pred = set(answer or [])
    human = set(item.endorsed)
    gold = set(item.gold)
    options = sorted(item.options)
    return {
        "trial_id": item.item_id,
        "respondent_id": item.respondent_id,
        "topic": item.topic,
        "level": item.level,
        "question_id": item.question_id,
        "human_correct": item.correct,
        "gold_options": sorted(gold),
        "human_options": sorted(human),
        "pred_options": sorted(pred),
        "option_labels": {str(n): list(item.option_labels.get(n, ())) for n in options},
        "reasoning": reasoning,
        "raw": raw,
        "parse_errors": parse_errors,
        "human_match": pred == human,
        "ground_truth_correct": pred == gold,
        # Graded companion to exact match: agreement across the five binary judgments,
        # which stays informative when the exact set is one option off.
        "per_option_agreement": (
            sum(1 for n in options if (n in pred) == (n in human)) / len(options)
            if options
            else None
        ),
        "n_endorsed_model": len(pred),
        "n_endorsed_human": len(human),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def subset(pred) -> List[Dict[str, Any]]:
        return [r for r in rows if pred(r)]

    def block(sel: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not sel:
            return {"n": 0}
        model_profile = endorsement_profile(sel, "pred_options")
        human_profile = endorsement_profile(sel, "human_options")
        return {
            "n": len(sel),
            # Primary STaR-filter metric. Low on the human-fail half by construction.
            "human_match_rate": _mean([float(r["human_match"]) for r in sel]),
            # Must FALL toward the human level; staying high means nothing was learned
            # about forgetting.
            "ground_truth_accuracy": _mean([float(r["ground_truth_correct"]) for r in sel]),
            "human_ground_truth_accuracy": _mean(
                [float(set(r["human_options"]) == set(r["gold_options"])) for r in sel]
            ),
            "per_option_agreement": _mean(
                [r["per_option_agreement"] for r in sel if r["per_option_agreement"] is not None]
            ),
            "mean_n_endorsed_model": _mean([float(r["n_endorsed_model"]) for r in sel]),
            "mean_n_endorsed_human": _mean([float(r["n_endorsed_human"]) for r in sel]),
            # Named error_profile_* on purpose -- see the module docstring.
            "error_profile_model": model_profile,
            "error_profile_human": human_profile,
            "error_profile_tv_distance": err.total_variation(model_profile, human_profile),
            "parse_failure_rate": _mean([float(bool(r["parse_errors"])) for r in sel]),
        }

    def cells(key) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            cell = out.setdefault(
                key(r), {"n": 0, "model_exact": 0, "human_exact": 0, "human_match": 0}
            )
            cell["n"] += 1
            cell["model_exact"] += int(r["ground_truth_correct"])
            cell["human_exact"] += int(set(r["human_options"]) == set(r["gold_options"]))
            cell["human_match"] += int(r["human_match"])
        for cell in out.values():
            n = cell["n"]
            cell["model_exact_rate"] = cell["model_exact"] / n
            cell["human_exact_rate"] = cell["human_exact"] / n
            cell["human_match_rate"] = cell["human_match"] / n
        return dict(sorted(out.items()))

    return {
        "overall": block(rows),
        "by_level": {
            lv: block(subset(lambda r, lv=lv: r["level"] == lv))
            for lv in sorted({r["level"] for r in rows})
        },
        "by_topic": {
            t: block(subset(lambda r, t=t: r["topic"] == t))
            for t in sorted({r["topic"] for r in rows})
        },
        # The interesting split. Items the human got right are close to a reading
        # comprehension task; the ones they got wrong are where human-likeness lives.
        "human_success_trials": block(subset(lambda r: r["human_correct"])),
        "human_fail_trials": block(subset(lambda r: not r["human_correct"])),
        "by_level_cell": cells(lambda r: r["level"]),
        "by_question_cell": cells(lambda r: f"{r['topic']}:{r['level']}:{r['question_id']}"),
    }
