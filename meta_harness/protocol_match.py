"""Make digit-span error structure (axis A1) comparable between humans and model.

The two protocols differ, and the difference invalidates the naive comparison:

* Humans get an **adaptive, terminating** staircase.  Inferred from all 52
  released participants: 2 trials per span, ascending from span 2, stop as soon
  as both trials at a span fail (every participant's terminal span is exactly
  2 trials / 0 correct).  Best span = highest span with at least one correct.
* The compactor runs **all 19 span lengths** (2..20), one trial per
  (span, sequence_index), 100 sequences.

Consequence: the model gets many trials far above its ceiling that a human
would never have been administered, which is what produced the apparent
"supra-span hit rate 0.574 vs 0.000" gap in the first pass.  That number was an
artifact of the schedule, not a psychological difference.

Under a matched protocol `supra_span_hit` is 0 on BOTH sides by construction --
the staircase stops at the first double-failure, so the only administered
supra-ceiling trials are the two failed ones.  So it cannot be an axis.  What
survives as a genuine discriminator is **sub-span leakage**: how often a
participant fails a span at or below their own ceiling.  Humans are not
perfectly reliable below threshold (0.087) but they are not random either; a
harness that drops items stochastically will show a much higher rate.

This module emulates the human staircase on model trials by pairing adjacent
sequence_index values into 2-trial-per-span blocks (100 sequences -> 50
emulated participants), then applying the real stop rule.  No resampling and no
synthetic trials: every trial used is one the model actually produced.
"""
from __future__ import annotations

import collections
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "runs/human/working-memory-digit-span"

TRIALS_PER_SPAN = 2


def _administer(by_span: dict[int, list], is_correct) -> tuple[list[tuple[int, object]], int]:
    """The human stop rule, over arbitrary per-trial items.

    Extracted from `_staircase` so the error typology and serial-position curves
    below run on EXACTLY the trials A1 runs on -- one stop rule, one place. With
    `is_correct=bool` and boolean items this is byte-for-byte the old behaviour;
    `test_error_shape.py` asserts A1's human summary is unchanged.

    Returns (administered [(span, item)], best span).
    """
    administered: list[tuple[int, object]] = []
    best = 0
    for span in sorted(by_span):
        block = by_span[span][:TRIALS_PER_SPAN]
        if len(block) < TRIALS_PER_SPAN:
            break
        administered.extend((span, it) for it in block)
        if any(is_correct(it) for it in block):
            best = span
        else:
            break  # both failed -> staircase terminates
    return administered, best


def _staircase(by_span: dict[int, list[bool]]) -> dict | None:
    """Apply the human stop rule to a span -> [outcomes] mapping.

    Returns the administered trials, the ceiling, and the sub-span leak rate.
    """
    administered, best = _administer(by_span, bool)
    if not administered or best == 0:
        return None
    sub = [ok for span, ok in administered if span <= best]
    sup = [ok for span, ok in administered if span > best]
    return {
        "best_span": best,
        "n_administered": len(administered),
        "sub_span_fail": 1.0 - float(np.mean(sub)) if sub else np.nan,
        "supra_span_hit": float(np.mean(sup)) if sup else np.nan,
    }


def human_participants() -> list[dict]:
    out = []
    for f in sorted(HUMAN_DIR.glob("run-*.json")):
        trials = json.load(open(f)).get("payload", {}).get("trials", [])
        by_span: dict[int, list[bool]] = defaultdict(list)
        for t in trials:
            if t.get("length") is None:
                continue
            by_span[int(t["length"])].append(bool(t["correct"]))
        rec = _staircase(by_span)
        if rec:
            out.append(rec)
    return out


def model_participants(jsonl_path: Path, condition: str = "C2") -> list[dict]:
    """Pair adjacent sequence_index values into 2-trial-per-span blocks.

    The released rows carry a constant participant_id, so sequence_index is the
    participant unit (the same grouping src/score.py uses).  Pairing sequences
    2k-1 and 2k gives the two trials per span the human staircase requires.
    """
    by_seq: dict[int, dict[int, bool]] = defaultdict(dict)
    for line in open(jsonl_path):
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != condition:
            continue
        seq = int(r["sequence_index"])
        by_seq[seq][int(r["span_length"])] = bool(r.get("metrics", {}).get("exact", 0) >= 1.0)

    seqs = sorted(by_seq)
    out = []
    for a, b in zip(seqs[0::2], seqs[1::2]):
        merged: dict[int, list[bool]] = defaultdict(list)
        for span in sorted(set(by_seq[a]) | set(by_seq[b])):
            for src in (a, b):
                if span in by_seq[src]:
                    merged[span].append(by_seq[src][span])
        rec = _staircase(merged)
        if rec:
            out.append(rec)
    return out


LETTERS = "ABCDEFGH"


def variable_mapping_scores(jsonl_path: Path, condition: str = "C2") -> list[float]:
    """Model-side variable_mapping scores recomputed with the HUMAN formula.

    The two sides do not compute the same quantity. They are not merely on
    different protocols, as digit span was -- they are different formulas sharing
    a denominator of 10:

      human  (src/score.py:180)  sum(1 for q in questions if q["correct"]) / 10
                                 -- a correct COUNT.
      model  (bench/tasks/variable_mapping.py:265-280)  iterate the questions,
                                 break on the first error, and score
                                 q["relation_count"] of the last consecutively
                                 correct question, / 10.

    `relation_count` is `len(mapping)`, the number of distinct people introduced so
    far, and it saturates at TARGET_RELATIONS = 10 by question 5. So once a run
    answers the first five questions correctly it has banked the maximum, and every
    later error is INVISIBLE to its score. Measured on the released runs:

      baseline      first_error_at {3: 2, 8: 4, 9: 2, 10: 4}; 10 of the 12 runs
                    with an error still score 1.0.
                    scored {1.0: 148, 0.4: 2}  ->  human formula {1.0: 138, 0.9: 12}
      displacement  25 runs erred and 24 of them still score 1.0.
                    scored {1.0: 149, 0.8: 1}  ->  human formula
                    {1.0: 125, 0.9: 18, 0.8: 4, 0.7: 3}

    So the "99% at 1.0, two unique values" point mass that made this task look like
    a ceiling with no headroom is substantially a SCORING artifact. The errors are
    there -- displacement made 25 of them -- and the formula discards them.

    This is an analysis-side correction and deliberately not a harness one: a
    harness must not know the scoring protocol. It also does not touch the human
    side, which already uses this formula.
    """
    out: list[float] = []
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("condition_id") or row.get("condition")) != condition:
            continue
        answers = row.get("parsed_answers") or {}
        questions = row.get("questions") or []
        if not questions:
            continue
        correct = 0
        for q in questions:
            idx = q.get("question_index")
            letter = answers.get(str(idx), answers.get(idx))
            options = q.get("options") or []
            if isinstance(letter, str) and letter in LETTERS:
                j = LETTERS.index(letter)
                # A malformed or out-of-range answer is simply not correct, which
                # matches the human formula: it counts correct answers and says
                # nothing about how a wrong one was expressed.
                if j < len(options) and options[j] == q.get("correct_city"):
                    correct += 1
        out.append(min(correct, 10) / 10.0)
    return out


# =========================== M2: digit-span error typology + serial position ==
#
# REPORT ONLY. Nothing below gates, floors, or enters mean_humanlikeness_search.
#
# Why here and not in a new module: these run on the *administered* trials only,
# i.e. the ones `_administer` above hands A1. Both digit-span tasks have come out
# at exactly +0.0000 humanlikeness delta for every candidate measured so far, so
# the score metric is blind to them; if anything about digit span discriminates,
# it has to be the shape of the wrong answers and where in the list they go wrong.
#
# UNITS, each with its own reference point:
#   * typology  -- share of that side's ADMINISTERED ERRORS falling in a class,
#                  in [0,1], the six classes summing to 1.000. Reference point is
#                  the error count, not the trial count.
#   * serial position -- proportion of PRESENTED positions reported with the right
#                  digit in the right place, in [0,1], over administered trials.
#                  Reference point is the trial's own length, so the relative
#                  thirds are comparable across spans.
#   * reversal (reverse span only) -- share of administered errors on
#                  NON-PALINDROMIC sequences, because when presented == reversed
#                  (e.g. span-2 "33") a forward-order report is indistinguishable
#                  from a correct one and belongs in neither numerator nor
#                  denominator.
#
# MINIMUM n. Measured on the released runs, the matched staircase yields roughly
# 30 administered errors per model run on forward span and 20 on reverse (against
# 160 and 169 on the human side), and the typology splits those six ways. The bar
# below is deliberately the same 30 A4 uses; reverse span will usually print
# `insufficient` rather than a six-way split of 20 errors, which is the honest
# outcome and not a reason to widen the filter. The unmatched all-19-spans variant
# is available for a bigger n and is labelled NOT human-comparable, because the
# extra trials are ones no human was ever administered.
HUMAN_REVERSE_DIR = ROOT / "runs/human/working-memory-reverse-digit-span"

MIN_ERRORS_TYPOLOGY = 30
SPAN_ERROR_CLASSES = ("reversal", "transposition", "truncation", "omission",
                      "substitution", "other")
_DIGITS = re.compile(r"\d")


def _digits(v) -> tuple[int, ...]:
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return tuple(int(c) for c in _DIGITS.findall(str(v)))


def classify_span_error(gold: tuple[int, ...], response: tuple[int, ...],
                        presented: tuple[int, ...] | None = None,
                        reverse: bool = False) -> str:
    """Name the shape of one wrong digit-span response.

    Precedence is fixed and ordered from most specific to least, so a response can
    only land in one class:

      reversal      reverse span only: the presented order was reported verbatim.
                    Undefined (and excluded by the caller) when the sequence is a
                    palindrome, where it coincides with the correct answer.
      transposition the right digits in the wrong order -- same multiset, same
                    length. This is the classic working-memory order error.
      truncation    a proper prefix of the target: the participant stopped early
                    rather than losing an item from the middle.
      omission      a subset of the target's digits, shorter, not a prefix.
      substitution  the right length, but at least one digit is not in the target.
      other         a MIXED error: the length differs from the target AND the content
                    is not a subset of it -- in the human forward records this is
                    41 of 160 errors, off by one digit in length (len-diff -3:2,
                    -2:5, -1:15, +1:19) with a substitution as well. The six classes
                    are FIXED: splitting `other` changes every share's denominator
                    and invalidates the cached human reference, so it must be done
                    by rebuilding the cache, not by reinterpreting the number.
    """
    if reverse and presented and response == presented and gold != presented:
        return "reversal"
    if len(response) == len(gold) and sorted(response) == sorted(gold):
        return "transposition"
    if len(response) < len(gold) and response == gold[:len(response)]:
        return "truncation"
    if len(response) < len(gold) and not (
            collections.Counter(response) - collections.Counter(gold)):
        return "omission"
    if len(response) == len(gold):
        return "substitution"
    return "other"


def _span_trial(span: int, presented, gold, response, correct: bool) -> dict:
    return {"span": span, "presented": _digits(presented), "gold": _digits(gold),
            "response": _digits(response), "correct": bool(correct)}


def human_span_trials(reverse: bool = False) -> list[list[dict]]:
    """Administered trials per human participant, under the real staircase."""
    out: list[list[dict]] = []
    folder = HUMAN_REVERSE_DIR if reverse else HUMAN_DIR
    for f in sorted(folder.glob("run-*.json")):
        trials = json.load(open(f)).get("payload", {}).get("trials", [])
        by_span: dict[int, list[dict]] = defaultdict(list)
        for t in trials:
            if t.get("length") is None:
                continue
            by_span[int(t["length"])].append(_span_trial(
                int(t["length"]), t.get("presentedSequence"),
                t.get("expectedResponse"), t.get("userResponse"),
                bool(t.get("correct"))))
        adm, best = _administer(by_span, lambda it: it["correct"])
        if adm and best:
            out.append([it for _, it in adm])
    return out


def model_span_trials(jsonl_path: Path, condition: str = "C2") -> list[list[dict]]:
    """Administered trials per model pseudo-participant, same stop rule.

    Pseudo-participants are adjacent `sequence_index` pairs, exactly as
    `model_participants` builds them, so M2 and A1 see the same trials. NOTE the
    released meta_harness runs carry 10 sequence_index values, not the 100 the
    module docstring describes (that count came from `runs/compactor/`), so this
    yields 5 pseudo-participants per run and the per-participant n is small on the
    model side -- reported, not hidden.
    """
    by_seq: dict[int, dict[int, dict]] = defaultdict(dict)
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != condition:
            continue
        by_seq[int(r["sequence_index"])][int(r["span_length"])] = _span_trial(
            int(r["span_length"]), r.get("digits"), r.get("gold"), r.get("pred"),
            bool((r.get("metrics") or {}).get("exact", 0) >= 1.0))

    seqs = sorted(by_seq)
    out: list[list[dict]] = []
    for a, b in zip(seqs[0::2], seqs[1::2]):
        merged: dict[int, list[dict]] = defaultdict(list)
        for span in sorted(set(by_seq[a]) | set(by_seq[b])):
            for src in (a, b):
                if span in by_seq[src]:
                    merged[span].append(by_seq[src][span])
        adm, best = _administer(merged, lambda it: it["correct"])
        if adm and best:
            out.append([it for _, it in adm])
    return out


def all_span_trials(jsonl_path: Path, condition: str = "C2") -> list[list[dict]]:
    """Every model trial, staircase NOT applied. NOT human-comparable.

    Exists only so a typology that prints `insufficient` under the matched
    protocol can still be inspected at a usable n. Any number from here is about
    the model's own behaviour on a schedule no human was administered.
    """
    rows: list[dict] = []
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != condition:
            continue
        rows.append(_span_trial(
            int(r["span_length"]), r.get("digits"), r.get("gold"), r.get("pred"),
            bool((r.get("metrics") or {}).get("exact", 0) >= 1.0)))
    return [rows]


def span_typology(participants: list[list[dict]], reverse: bool = False,
                  min_errors: int = MIN_ERRORS_TYPOLOGY) -> dict:
    """Pooled error-class shares over administered trials."""
    counts = {k: 0 for k in SPAN_ERROR_CLASSES}
    n_err = n_tr = 0
    n_err_nonpalin = 0
    for trials in participants:
        for t in trials:
            n_tr += 1
            if t["correct"]:
                continue
            n_err += 1
            palin = bool(t["presented"]) and t["presented"] == t["presented"][::-1]
            if not palin:
                n_err_nonpalin += 1
            counts[classify_span_error(t["gold"], t["response"], t["presented"],
                                       reverse=reverse and not palin)] += 1
    res: dict = {
        "unit": "share of administered errors (6 classes sum to 1.000)",
        "n_participants": len(participants),
        "n_trials": n_tr,
        "n_errors": n_err,
        "n_errors_nonpalindromic": n_err_nonpalin,
        "min_errors": min_errors,
        "sufficient": bool(n_err >= min_errors),
        "counts": counts,
    }
    if n_err >= min_errors:
        res["shares"] = {k: round(v / n_err, 4) for k, v in counts.items()}
        # The reversal class has its own denominator: palindromic sequences cannot
        # express it, so they are excluded from both sides of that one share.
        res["reversal_share_nonpalindromic"] = (
            round(counts["reversal"] / n_err_nonpalin, 4) if n_err_nonpalin else None)
    else:
        res["shares"] = None
        res["reversal_share_nonpalindromic"] = None
        res["note"] = f"insufficient (n={n_err}); minimum {min_errors} errors"
    return res


def _positional_hits(t: dict) -> list[bool]:
    """Position i of the target reported with the right digit in place i."""
    gold, resp = t["gold"], t["response"]
    return [i < len(resp) and resp[i] == gold[i] for i in range(len(gold))]


def serial_position(participants: list[list[dict]], n_bins: int = 3) -> dict:
    """Recall probability by RELATIVE serial position -- the primacy/recency curve.

    Relative rather than absolute, because the staircase administers different span
    lengths to different participants: pooling absolute position 5 mixes "the last
    item of a 5-span" with "the middle of a 12-span". Position i of a length-L
    target falls in bin `min(n_bins-1, n_bins*i // L)`, so bin 0 is the primacy
    end and bin n_bins-1 the recency end for every length.

    Per-participant values are returned so a 1 - W_1 over them is possible; the
    absolute-position curve is returned pooled, as a descriptive companion.
    """
    per_p: list[list[float]] = []
    abs_hits: dict[int, list[bool]] = defaultdict(list)
    for trials in participants:
        bins: list[list[bool]] = [[] for _ in range(n_bins)]
        for t in trials:
            hits = _positional_hits(t)
            L = len(hits)
            for i, ok in enumerate(hits):
                bins[min(n_bins - 1, n_bins * i // L)].append(ok)
                abs_hits[i + 1].append(ok)
        if all(bins):
            per_p.append([float(np.mean(b)) for b in bins])
    arr = np.asarray(per_p, float)
    return {
        "unit": ("proportion of presented positions reported with the right digit "
                 "in the right place"),
        "n_bins": n_bins,
        "n_participants": int(arr.shape[0]),
        "bin_means": [round(float(v), 4) for v in arr.mean(axis=0)] if arr.size else None,
        "per_participant": per_p,
        "absolute_curve": {str(k): round(float(np.mean(v)), 4)
                           for k, v in sorted(abs_hits.items())},
        "absolute_n": {str(k): len(v) for k, v in sorted(abs_hits.items())},
    }


def summarize(label: str, recs: Iterable[dict]) -> dict:
    recs = list(recs)
    return {
        "label": label,
        "n": len(recs),
        "best_span": float(np.mean([r["best_span"] for r in recs])),
        "n_administered": float(np.mean([r["n_administered"] for r in recs])),
        "sub_span_fail": float(np.nanmean([r["sub_span_fail"] for r in recs])),
        "supra_span_hit": float(np.nanmean([r["supra_span_hit"] for r in recs])),
    }


def main() -> None:
    import sys

    models = sys.argv[1:] or [
        "runs/compactor/claude-opus-4-6",
        "runs/compactor/qwen_qwen3-30b-a3b-instruct-2507",
        "runs/compactor/qwen_qwen3-8b_false",
    ]

    rows = [summarize("HUMANS", human_participants())]
    for md in models:
        p = ROOT / md / "tasks/wm_digit_span_forward.jsonl"
        if not p.exists():
            print(f"  (missing {p})")
            continue
        rows.append(summarize(Path(md).name, model_participants(p)))

    print("A1  DIGIT SPAN, protocol-matched staircase "
          f"({TRIALS_PER_SPAN} trials/span, stop on double failure)")
    print(f"    {'source':<44}{'n':>5}{'best_span':>11}{'trials':>8}"
          f"{'sub_span_fail':>15}{'supra_hit':>11}")
    for r in rows:
        sup = "0.000*" if np.isnan(r["supra_span_hit"]) else f"{r['supra_span_hit']:.3f}"
        print(f"    {r['label']:<44}{r['n']:>5}{r['best_span']:>11.2f}"
              f"{r['n_administered']:>8.1f}{r['sub_span_fail']:>15.3f}{sup:>11}")
    print()
    print("    * supra-ceiling trials are 0 by construction under the matched")
    print("      staircase, on both sides -- so sub_span_fail is the axis, and")
    print("      the earlier 0.574-vs-0.000 gap was a schedule artifact.")


if __name__ == "__main__":
    main()
