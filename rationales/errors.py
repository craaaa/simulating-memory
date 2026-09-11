"""Error taxonomy for digit-span responses.

Applied identically to human responses and model outputs so the two profiles are
directly comparable. Exact match to a human's specific error has a low ceiling (which
transposition a given human makes is stochastic), so these distributional labels are
the metric to read, not the exact-match rate alone.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

CATEGORIES = (
    "exact",
    "truncation",
    "omission",
    "insertion",
    "substitution",
    "transposition",
    "mixed",
    "empty",
)


def classify_error(expected: Sequence[int], response: Sequence[int]) -> str:
    """Label one response relative to the correct answer.

    Order matters: the more specific labels are tested first, and anything that does
    not fall cleanly into one bucket is "mixed" rather than being forced.
    """
    exp = list(expected)
    got = list(response)

    if not got:
        return "empty"
    if got == exp:
        return "exact"

    # Dropped a suffix (the classic span failure) vs. dropped somewhere inside.
    if len(got) < len(exp):
        if exp[: len(got)] == got:
            return "truncation"
        if _is_subsequence(got, exp):
            return "omission"

    if len(got) > len(exp) and _is_subsequence(exp, got):
        return "insertion"

    if len(got) == len(exp):
        diff = [i for i, (a, b) in enumerate(zip(exp, got)) if a != b]
        if sorted(exp) == sorted(got):
            # Same multiset, different order.
            if len(diff) == 2:
                i, j = diff
                if exp[i] == got[j] and exp[j] == got[i]:
                    return "transposition"
            return "transposition"
        if len(diff) == 1:
            return "substitution"

    return "mixed"


def error_features(expected: Sequence[int], response: Sequence[int]) -> Dict[str, object]:
    """Graded description of an error, for when the category is "mixed".

    ~44% of real human errors combine a length change with a reordering, so the
    single-label taxonomy above saturates on "mixed". These features stay informative
    for those cases and are averaged in the eval report alongside the categories.
    """
    exp, got = list(expected), list(response)
    return {
        "len_delta": len(got) - len(exp),
        "n_positions_wrong": len(error_positions(exp, got)),
        "same_multiset": sorted(exp) == sorted(got),
        "prefix_correct": prefix_correct(exp, got),
        "prefix_frac": (prefix_correct(exp, got) / len(exp)) if exp else 0.0,
    }


def _is_subsequence(small: Sequence[int], big: Sequence[int]) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def profile(pairs: Sequence[tuple]) -> Dict[str, float]:
    """(expected, response) pairs -> normalized error-type distribution."""
    counts = Counter(classify_error(e, r) for e, r in pairs)
    total = sum(counts.values()) or 1
    return {c: counts.get(c, 0) / total for c in CATEGORIES}


def counts(pairs: Sequence[tuple]) -> Dict[str, int]:
    c = Counter(classify_error(e, r) for e, r in pairs)
    return {k: int(c.get(k, 0)) for k in CATEGORIES}


def total_variation(a: Dict[str, float], b: Dict[str, float]) -> float:
    """TV distance between two error profiles; 0 = identical."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def prefix_correct(expected: Sequence[int], response: Sequence[int]) -> int:
    """How many leading digits are right -- a graded companion to exact match."""
    n = 0
    for e, g in zip(expected, response):
        if e != g:
            break
        n += 1
    return n


def error_positions(expected: Sequence[int], response: Sequence[int]) -> List[int]:
    """0-indexed positions where the response departs from the correct answer."""
    out: List[int] = []
    for i in range(max(len(expected), len(response))):
        e = expected[i] if i < len(expected) else None
        g = response[i] if i < len(response) else None
        if e != g:
            out.append(i)
    return out
