"""Check a candidate's PRE-REGISTERED predictions against its run. Mechanically.

Why this exists. A candidate's manifest states what it expects to happen, including
what it expects to lose. After the run, it is very easy to read the numbers and
write a story in which the candidate did roughly what it meant to — the predictions
were qualitative, the evidence is a 40-field record, and nobody is checking. That is
how a search stops being able to tell a mechanism from a lucky number.

So the predictions are encoded here as executable assertions, written from the
candidate's docstring, and evaluated as PASS / FAIL / INCONCLUSIVE with the observed
value printed next to the threshold. A FAIL is not an argument for rejecting the
candidate — the per-task floor and the guards do that — it is a record that the
stated mechanism did not do what it claimed, which is the thing worth knowing for the
next iteration.

`displacement` (iteration 1) pre-registered, in its own words:

  primary      n=3 `answered` rises from 6.82 to >= 12 of 14, while
               `acc_over_answered` stays within +-0.05 of 0.737
  disconfirm   `answered` does not rise  -> the refusal diagnosis is dead
               `answered` rises but acc_over_answered < 0.65 -> the silence was
               incapacity and the store was never the issue
  capacity     n=3 slot utilisation stays ~1.0; displacement is not extra capacity
  compliance   n=3 buffer `No response` rises from 50/150 past 100/150
  leak test    word_recognition moves by LESS than its 0.121 noise floor; if it
               moves a lot under a change that only touches overflow semantics, the
               third-leak diagnosis is wrong
  exposed      semantic_story_recall regresses -- expected, names primacy
               protection as the iteration-2 ingredient

Iteration 2 adds four candidates (`primacy`, `serial_recognition`,
`episodic_reset`, `chunk_limit`), whose pre-registered rows live in
`meta_harness/logs/pending_<candidate>.json`. Their thresholds are transcribed
here as literals with the pre-registered prose quoted in a comment beside the
code, so the two can be compared by eye without resolving anything at runtime.

Two things iteration 1 got wrong, and what was added to stop them recurring:

  VOID     displacement's n=3 buffer-compliance prediction was scored FAIL, but
           the n=3 buffer response is a fixed positional pattern that is
           identical in every run -- the quantity could not move, so neither PASS
           nor FAIL meant anything. A row that registers `capable_of_varying`
           and then lands on a value bit-identical to the baseline's is now VOID.
           VOID also covers a failed PRECONDITION: episodic_reset's P9 states
           that its own failure invalidates P1-P7 whatever they read, and
           serial_recognition's P12 states that if the open ablation arm also
           collapses then the masked arm's collapse means nothing.
  ADAPTED  one row-local verdict, chunk_limit P1 leg B: studied words in
           8.9-16 with leg A satisfied is pre-registered as an explicit THIRD
           OUTCOME ("the bound held and the agent reallocated"), which is neither
           a pass nor a failure and must not be filed as either.

Rows also carry `tags`, which mark a verdict that must not be cited as evidence:
ENTAILED (an arithmetic consequence of another row -- serial_recognition P13/P14),
REPORTED (registered with no direction, so INCONCLUSIVE by construction),
NEAR-INERT (the row itself says the quantity can barely vary, so a PASS is not a
passed test), UNDER-POWERED, GATED and NEEDS-REVIEW.

A checker fed the BASELINE run must not report all-pass: every candidate predicts
some change from it. That is the negative control each function below was
validated against. Note that a self-comparison suppresses the identity-VOID test
(a run compared against itself is trivially identical, and all-VOID is the same
disease as all-PASS), so the VOID path is exercised against iter1/displacement.

Usage:
    python meta_harness/check_predictions.py <candidate_id> <run_dir> \
        [--baseline <run_dir>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from meta_harness import interference as IF  # noqa: E402
from meta_harness import nback_levels as NL  # noqa: E402
from meta_harness import score_candidate as SC  # noqa: E402

PASS, FAIL, INCONCL, VOID = "PASS", "FAIL", "INCONCLUSIVE", "VOID"
# Row-local to chunk_limit P1 leg B, which pre-registers a third outcome that is
# neither pass nor fail. It is a separate label rather than INCONCLUSIVE because
# INCONCLUSIVE means "could not be measured", and this outcome was measured.
ADAPTED = "ADAPTED"

VERDICTS = (PASS, FAIL, INCONCL, VOID, ADAPTED)
# Visually distinct markers, so VOID and ADAPTED cannot be skimmed as a PASS.
MARK = {PASS: "   ", FAIL: " x ", INCONCL: " ? ", VOID: "!!!", ADAPTED: " ~ "}

# Distinguishes "the record has no such path" from "the path resolved to None".
# Both happen -- `delta_vs_baseline` is absent entirely when no baseline is given,
# while `axes.A2.ratio` can be present and NaN -- and an INCONCLUSIVE reason has
# to say which field was missing, by name.
MISSING = object()

ALL_TASK_FILES = [
    "wm_digit_span_forward", "wm_digit_span_reverse", "wm_nback",
    "wm_word_recognition", "wm_variable_mapping", "wm_narrative_qa",
    "wm_semantic_story_recall", "wm_craft_task",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def _adder(out: list[dict[str, Any]]) -> Callable[..., None]:
    """Row factory in displacement_checks' shape, plus `tags`."""
    def add(name: str, verdict: str, observed: Any, threshold: str,
            note: str = "", tags: tuple[str, ...] = ()) -> None:
        out.append({"prediction": name, "verdict": verdict, "observed": observed,
                    "threshold": threshold, "note": note, "tags": list(tags)})
    return add


def _jsonl(run_dir: Path, task: str) -> list[dict[str, Any]] | None:
    """Rows of one task file, or None if the run or the task file is absent.

    None and [] are different answers: a task that was not part of an arm has no
    file at all, and that must surface as INCONCLUSIVE naming the file rather than
    as a zero.
    """
    path = Path(run_dir) / "tasks" / f"{task}.jsonl"
    if not path.exists():
        return None
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


_REC_CACHE: dict[tuple[str, str | None], dict[str, Any]] = {}


def _record(run_dir: Path, baseline: Path | None) -> dict[str, Any]:
    """`score_candidate.evaluate()`'s record, memoised, and never raising.

    The iteration-2 runs are still in flight, so a run dir may be missing or hold
    only some of the eight tasks. A traceback there would read as a checker bug
    rather than as missing data, so the failure is captured in `_error` and every
    row that needed the record reports INCONCLUSIVE with the field it wanted.
    """
    key = (str(run_dir), str(baseline) if baseline else None)
    if key not in _REC_CACHE:
        if not Path(run_dir).exists():
            _REC_CACHE[key] = {"_error": f"run dir does not exist: {run_dir}"}
        else:
            try:
                _REC_CACHE[key] = SC.evaluate(Path(run_dir), baseline)
            except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
                _REC_CACHE[key] = {"_error": f"{type(exc).__name__}: {exc}"}
    return _REC_CACHE[key]


_AXES_CACHE: dict[str, dict[str, Any]] = {}


def _axes(run_dir: Path) -> dict[str, Any]:
    """A1/A2/A3/A4 and the nback levels for one run. Memoised: A3's precision and
    A4 are not cheap, and several rows want the baseline's axes."""
    key = str(run_dir)
    if key not in _AXES_CACHE:
        if not Path(run_dir).exists():
            _AXES_CACHE[key] = {}
        else:
            try:
                _AXES_CACHE[key] = SC.axes(Path(run_dir))
            except Exception:  # noqa: BLE001
                _AXES_CACHE[key] = {}
    return _AXES_CACHE[key]


def _dotted(rec: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path, returning MISSING if any step is absent.

    Integer-keyed levels (`axes.nback_levels.diagnostics.3.answered`) are looked
    up as int as well as str, because nback_levels keys its diagnostics by int.
    """
    cur: Any = rec
    for part in path.split("."):
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                continue
            if part.lstrip("-").isdigit() and int(part) in cur:
                cur = cur[int(part)]
                continue
            return MISSING
        if isinstance(cur, list) and part.lstrip("-").isdigit():
            i = int(part)
            if -len(cur) <= i < len(cur):
                cur = cur[i]
                continue
        return MISSING
    return cur


def _get(rec: dict[str, Any], path: str) -> tuple[Any, str | None]:
    """(value, reason-if-unusable). Reason always names the field."""
    v = _dotted(rec, path)
    if v is MISSING:
        extra = f" ({rec['_error']})" if rec.get("_error") else ""
        return None, f"field absent: {path}{extra}"
    if v is None:
        return None, f"field present but null: {path}"
    if isinstance(v, float) and v != v:
        return None, f"field is NaN: {path}"
    return v, None


def _same_run(run_dir: Path, baseline: Path | None) -> bool:
    if baseline is None:
        return False
    try:
        return Path(run_dir).resolve() == Path(baseline).resolve()
    except OSError:
        return str(run_dir) == str(baseline)


def _exact_eq(a: Any, b: Any) -> bool:
    """Bit-identical to full precision. Not a tolerance -- the point of the VOID
    test is that the quantity did not move AT ALL, which is what distinguishes a
    prediction that could not move from one that did not happen to."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_exact_eq(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _void_if_identical(verdict: str, observed: Any, base_observed: Any,
                       evidence: str, same_run: bool,
                       *, predicts_change: bool = True) -> tuple[str, str]:
    """VOID when a row that predicted a CHANGE did not move off the baseline at all.

    Two restrictions, both of which matter:

    `predicts_change=False` -- identity-VOID is applied ONLY to rows that predict a
    change. On a NO-CHANGE row, landing bit-identically on the baseline is the
    predicted outcome and is the attribution claim itself, so voiding it would
    destroy a confirmation rather than withhold a meaningless one. episodic_reset's
    P10 makes the point explicitly: "Expected delta is EXACTLY 0". Those rows keep
    their verdict and are tagged FROZEN by `_frozen_tag` instead.

    Suppressed on a self-comparison: identity to the baseline is evidence only when
    the two are different runs, and an all-VOID report on a run scored against
    itself would be as uninformative as an all-PASS one.

    On a compound row, every leg must be identical -- a subset of frozen legs beside
    one that moved is not a frozen quantity.
    """
    if same_run or not predicts_change or base_observed is None or observed is None:
        return verdict, ""
    if _exact_eq(observed, base_observed):
        return VOID, (
            f"VOID, not {verdict}: identical to the baseline's {_fmt(base_observed)} "
            f"to full precision, although the row registers the quantity as capable "
            f"of varying ({evidence}). A change-prediction on a quantity the run did "
            f"not move at all cannot be scored in either direction -- this is "
            f"iteration 1's n=3 buffer-compliance error."
        )
    return verdict, ""


def _frozen_tag(observed: Any, base_observed: Any,
                same_run: bool) -> tuple[tuple[str, ...], str]:
    """For a NO-CHANGE row: record that the value is bit-identical to the baseline.

    Not a verdict change. The row asserted that this quantity would not move, so
    identity is what it predicted; the tag exists so a reader can see that the PASS
    rests on an exactly unmoved number rather than on a tolerance being wide.
    """
    if same_run or base_observed is None or observed is None:
        return (), ""
    if _exact_eq(observed, base_observed):
        return ("FROZEN",), ("bit-identical to the baseline to full precision, "
                             "which is what this no-change row predicted")
    return (), ""


def _void_rows(rows: list[dict[str, Any]], ids: tuple[str, ...], reason: str) -> None:
    """Rewrite named rows to VOID because a precondition row failed.

    Pre-registered mandates, not annotations: episodic_reset P9 says its own
    failure invalidates P1-P7 REGARDLESS of their values, and serial_recognition
    P12 says an open arm that also collapses voids the masked arm's claim. Leaving
    those rows at PASS/FAIL with a footnote would let a later reader cite them.
    """
    for r in rows:
        if r["prediction"].split()[0] in ids:
            # A row that could not be measured stays INCONCLUSIVE: "the field was
            # absent" is a more useful reason than "voided by another row", and
            # overwriting it would hide missing data behind a precondition.
            if r["verdict"] == INCONCL:
                continue
            r["verdict"] = VOID
            r["note"] = (reason + (" || previous note: " + r["note"]) if r["note"]
                         else reason)
            if "VOIDED" not in r["tags"]:
                r["tags"].append("VOIDED")


def displacement_checks(run_dir: Path, baseline: Path | None) -> list[dict[str, Any]]:
    rep = NL.report(run_dir)
    d3 = (rep.get("diagnostics") or {}).get(3) or {}
    lv3 = (rep.get("per_level") or {}).get(3) or {}

    answered = d3.get("answered")
    acc_ans = d3.get("acc_over_answered")
    keys = d3.get("keys_held")
    buf = d3.get("buffer_no_response_frac")

    out: list[dict[str, Any]] = []

    def add(name: str, verdict: str, observed: Any, threshold: str,
            note: str = "") -> None:
        out.append({"prediction": name, "verdict": verdict,
                    "observed": observed, "threshold": threshold, "note": note})

    # --- primary -------------------------------------------------------------
    if answered is None:
        add("n=3 answered >= 12/14", INCONCL, None, ">= 12",
            "no n=3 rows found")
    elif answered >= 12.0:
        add("n=3 answered >= 12/14", PASS, answered, ">= 12 (baseline 6.82)")
    elif answered > 6.82 + 1.0:
        add("n=3 answered >= 12/14", FAIL, answered, ">= 12 (baseline 6.82)",
            "rose but fell short of the stated threshold -- mechanism partially "
            "works; do not round this up to a confirmation")
    else:
        add("n=3 answered >= 12/14", FAIL, answered, ">= 12 (baseline 6.82)",
            "DISCONFIRMS the refusal diagnosis outright: the store jam was not "
            "what was suppressing responses")

    if acc_ans is None:
        add("n=3 acc_over_answered within +-0.05 of 0.737", INCONCL, None,
            "0.687 - 0.787")
    elif 0.687 <= acc_ans <= 0.787:
        add("n=3 acc_over_answered within +-0.05 of 0.737", PASS, acc_ans,
            "0.687 - 0.787")
    elif acc_ans < 0.65:
        add("n=3 acc_over_answered within +-0.05 of 0.737", FAIL, acc_ans,
            "0.687 - 0.787",
            "below the 0.65 disconfirmation line: extra responses are guesses, so "
            "any score gain here is production without judgement")
    else:
        add("n=3 acc_over_answered within +-0.05 of 0.737", FAIL, acc_ans,
            "0.687 - 0.787",
            "outside the band but above 0.65 -- judgement changed too, so the "
            "change is not purely to response production as claimed")

    # --- capacity: displacement must not smuggle in extra room ---------------
    if keys is None:
        add("n=3 slot utilisation ~1.0 (4 keys held)", INCONCL, None, ">= 3.5")
    elif keys >= 3.5:
        add("n=3 slot utilisation ~1.0 (4 keys held)", PASS, keys,
            ">= 3.5 of 4 (baseline 3.96)")
    else:
        add("n=3 slot utilisation ~1.0 (4 keys held)", FAIL, keys,
            ">= 3.5 of 4 (baseline 3.96)",
            "the store is no longer saturating, so something other than the "
            "overflow rule changed -- compare against full_context's 1.06")

    # --- compliance: the check that cannot be bought by answering eagerly ----
    # Compare against the baseline's MEASURED value rather than a literal, so a run
    # scored against itself is not described as having "improved" on itself.
    base_buf = None
    if baseline is not None:
        base_buf = ((NL.report(baseline).get("diagnostics") or {}).get(3) or {}
                    ).get("buffer_no_response_frac")
    ref = "baseline 0.333" if base_buf is None else f"baseline {base_buf:.3f}"

    if buf is None:
        add("n=3 buffer 'No response' > 100/150", INCONCL, None, "> 0.667")
    elif buf > 0.667:
        add("n=3 buffer 'No response' > 100/150", PASS, buf, f"> 0.667 ({ref})")
    elif base_buf is not None and buf > base_buf + 1e-9:
        add("n=3 buffer 'No response' > 100/150", FAIL, buf, f"> 0.667 ({ref})",
            "improved but under the stated bar")
    else:
        add("n=3 buffer 'No response' > 100/150", FAIL, buf, f"> 0.667 ({ref})",
            "phase tracking did not improve; a rise in `answered` without this is "
            "consistent with answering more eagerly rather than knowing where it is")

    # --- the leak test, and the expected loss -------------------------------
    if baseline is not None:
        cand = SC.humanlikeness_by_task(run_dir, SC.SEARCH_TASKS)
        base = SC.humanlikeness_by_task(baseline, SC.SEARCH_TASKS)

        wr_c, wr_b = cand.get("word_recognition"), base.get("word_recognition")
        if wr_c is None or wr_b is None:
            add("word_recognition moves < 0.121 (leak test)", INCONCL, None,
                "< 0.121")
        else:
            delta = abs(wr_c - wr_b)
            add("word_recognition moves < 0.121 (leak test)",
                PASS if delta < 0.121 else FAIL, round(delta, 4), "< 0.121",
                "" if delta < 0.121 else
                "word recognition moved a lot under a change that only touches "
                "overflow semantics -- the third-leak reading needs revisiting")

        sr_c, sr_b = cand.get("semantic_story_recall"), base.get("semantic_story_recall")
        if sr_c is None or sr_b is None:
            add("semantic_story_recall regresses (expected loss)", INCONCL, None,
                "delta < 0")
        else:
            delta = round(sr_c - sr_b, 4)
            # Either direction is informative here, so this is reported rather than
            # scored: the candidate predicted a loss and said what a loss would mean.
            add("semantic_story_recall regresses (expected loss)",
                PASS if delta < 0 else INCONCL, delta, "delta < 0",
                "as predicted; primacy protection is the named iteration-2 "
                "ingredient" if delta < 0 else
                "did NOT regress -- the primacy-vs-recency story is wrong or the "
                "refusal was not shaping which chunks were kept")

    return out


_HL_CACHE: dict[str, dict[str, float | None]] = {}


def _hl(run_dir: Path | None) -> dict[str, float | None]:
    """Per-task humanlikeness for one run, memoised, never raising."""
    if run_dir is None:
        return {}
    key = str(run_dir)
    if key not in _HL_CACHE:
        if not Path(run_dir).exists():
            _HL_CACHE[key] = {}
        else:
            try:
                _HL_CACHE[key] = SC.humanlikeness_by_task(Path(run_dir),
                                                          SC.SEARCH_TASKS)
            except Exception:  # noqa: BLE001
                _HL_CACHE[key] = {}
    return _HL_CACHE[key]


def _max_keys() -> int:
    """MAX_KEYS live from the store module, never a literal 4 here."""
    try:
        from bench.core.working_memory import MAX_KEYS
        return int(MAX_KEYS)
    except Exception:  # noqa: BLE001
        return 4


def _mean(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and float(x) == float(x)]
    return sum(vals) / len(vals) if vals else None


def _subgroup_mean(hl: dict[str, float | None], tasks: list[str]) -> float | None:
    vals = [hl.get(t) for t in tasks]
    return None if any(v is None for v in vals) else _mean(vals)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# primacy (iteration 2a) -- meta_harness/logs/pending_primacy.json
# ---------------------------------------------------------------------------
# Literals transcribed from that file's `baseline_value` / `displacement_value`
# fields. Where the baseline run is supplied they are only a fallback: comparing
# against a measured baseline is what stops a threshold drifting away from the run
# it was calibrated on.
PRIM_STORY_BASE, PRIM_STORY_DISPL = 0.9473, 0.8964
PRIM_A1_LEAK, PRIM_A1_SPAN = 0.1298, 18.4
PRIM_GIST = ["semantic_story_recall", "craft_task", "narrative_qa"]
PRIM_STORE5 = ["digit_span_forward", "digit_span_reverse", "semantic_story_recall",
               "craft_task", "narrative_qa"]


def _primacy_u_shape(run_dir: Path, max_keys: int) -> dict[str, Any] | None:
    """P2's quantity: in overflowing story-recall rows, is BOTH ends retained?

    Write order is the first occurrence of each key across `turn_logs[*].tool_calls`
    (the baseline run writes every key inside a single turn, so the order within a
    turn is load-bearing and rows cannot be summarised per turn). Rows are
    restricted to those with more than MAX_KEYS distinct written keys, because a
    row that never overflowed has nothing to protect and would dilute the fraction
    toward 1.0 for free.
    """
    rows = _jsonl(run_dir, "wm_semantic_story_recall")
    if rows is None:
        return None
    both = n = 0
    hits: dict[int, int] = defaultdict(int)
    tot: dict[int, int] = defaultdict(int)
    for r in rows:
        order: list[str] = []
        for turn in (r.get("turn_logs") or []):
            for tc in (turn.get("tool_calls") or []):
                if tc.get("name") != "write_memory":
                    continue
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                except json.JSONDecodeError:
                    continue
                key = args.get("key")
                if isinstance(key, str) and key not in order:
                    order.append(key)
        if len(order) <= max_keys:
            continue
        kv = r.get("final_kv") or {}
        n += 1
        if order[0] in kv and order[-1] in kv:
            both += 1
        for i, key in enumerate(order, 1):
            tot[i] += 1
            hits[i] += 1 if key in kv else 0
    if not n:
        return {"n_overflow_rows": 0, "both_ends_frac": None, "retention": {}}
    return {
        "n_overflow_rows": n,
        "both_ends_frac": round(both / n, 4),
        "retention": {i: round(hits[i] / tot[i], 3) for i in sorted(tot)},
    }


def primacy_checks(run_dir: Path, baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    same = _same_run(run_dir, baseline)
    rec = _record(run_dir, baseline)
    hl = rec.get("humanlikeness_by_task") or {}
    delta = rec.get("delta_vs_baseline") or {}
    ax = rec.get("axes") or {}
    bax = _axes(baseline) if baseline is not None else {}
    bhl = _hl(baseline)
    mk = _max_keys()

    # --- P1 primary: "abs(cand - 0.9473) < 0.03 and (cand - 0.8964) > 0.025" ---
    # Two-sided ON PURPOSE: a large GAIN does not confirm the mechanism, because
    # on an inverted objective a capability gain costs humanlikeness.
    b_story = bhl.get("semantic_story_recall")
    b_story = PRIM_STORY_BASE if b_story is None else b_story
    lo, hi = PRIM_STORY_DISPL + 0.025, b_story + 0.03
    band = f"{max(lo, b_story - 0.03):.4f} < x < {hi:.4f} (two-sided)"
    cand = hl.get("semantic_story_recall")
    if cand is None:
        add("P1 semantic_story_recall recovers (two-sided)", INCONCL, None, band,
            "field absent: humanlikeness_by_task.semantic_story_recall")
    else:
        lo_eff = max(lo, b_story - 0.03)
        if lo_eff < cand < hi:
            v, note = PASS, ""
        elif cand <= lo_eff:
            v, note = FAIL, ("DISCONFIRMED: primacy protection did not recover "
                             "displacement's loss, so the primacy/recency reading "
                             "of that loss is wrong")
        else:
            v, note = FAIL, ("above the band -- an over-recovery is NOT a "
                             "confirmation: the retained set now spans both ends "
                             "of the narrative, so judged coverage can rise, and "
                             "on an inverted objective that costs humanlikeness")
        v, vn = _void_if_identical(v, cand, bhl.get("semantic_story_recall"),
                                   "moved -0.0509 baseline->displacement, "
                                   "-0.1548 under random_decay", same)
        add("P1 semantic_story_recall recovers (two-sided)", v, cand, band,
            vn or note)

    # --- P2 anti-noise: both ends retained in >= 0.80 of overflowing rows ------
    # "DISCONFIRMED below 0.60". A uniform random keep-4-of-6 gives ~0.40, pure
    # recency 0.000, refuse-on-full 0.000 -- which is why this is the row a noise
    # generator cannot fake, and why it is the mechanism test rather than P1.
    u = _primacy_u_shape(run_dir, mk)
    if u is None:
        add("P2 U-shaped survival: both ends retained", INCONCL, None, ">= 0.80",
            "task file absent: tasks/wm_semantic_story_recall.jsonl")
    elif u["both_ends_frac"] is None:
        add("P2 U-shaped survival: both ends retained", INCONCL, None, ">= 0.80",
            "no story-recall row wrote more than MAX_KEYS distinct keys, so no "
            "row overflowed and the quantity is undefined on this run")
    else:
        frac = u["both_ends_frac"]
        if frac >= 0.80:
            v, note = PASS, f"over {u['n_overflow_rows']} overflowing rows"
        elif frac < 0.60:
            v, note = FAIL, (f"DISCONFIRMED (< 0.60) over {u['n_overflow_rows']} "
                             f"overflowing rows: retention is not U-shaped, so the "
                             f"rule is not protecting the first-written chunk")
        else:
            v, note = FAIL, (f"0.60-0.80 over {u['n_overflow_rows']} rows: above "
                             f"the disconfirmation line but short of the stated "
                             f"bar -- partial, do not round up")
        base_u = _primacy_u_shape(baseline, mk) if baseline is not None else None
        v, vn = _void_if_identical(
            v, frac, (base_u or {}).get("both_ends_frac"),
            "0.000 in both prior runs, 0.914 replaying displacement's own write "
            "sequences through the real PrimacyMemory class", same)
        add("P2 U-shaped survival: both ends retained", v, frac, ">= 0.80",
            vn or note)
        # Secondary form of the same prediction, reported: mid-list write
        # positions (3 and 4 of 6) should be the LEAST retained. A "protect the
        # first N" rule does not predict this, so it separates the two rules.
        ret = u["retention"]
        mid_lowest = None
        if len(ret) >= 4:
            worst = min(ret, key=lambda k: ret[k])
            mid_lowest = worst in (3, 4)
        add("P2b mid-list positions least retained (secondary, reported)",
            INCONCL, ret, "argmin of retention at write position 3 or 4",
            f"argmin at mid position: {mid_lowest}; registered as the secondary "
            f"form of P2 and reported rather than scored", ("REPORTED",))

    # --- P3 no-change: protects iteration 1's confirmed n=3 gain --------------
    # "keys_held >= 3.5 AND answered >= 13.0 AND 0.687 <= acc_over_answered <= 0.787"
    d3 = (NL.report(Path(run_dir)).get("diagnostics") or {}).get(3) or {} \
        if Path(run_dir).exists() else {}
    b3 = ((NL.report(Path(baseline)).get("diagnostics") or {}).get(3) or {}) \
        if baseline is not None and Path(baseline).exists() else {}
    obs3 = {k: d3.get(k) for k in ("keys_held", "answered", "acc_over_answered")}
    thr3 = "keys_held >= 3.5, answered >= 13.0, acc_over_answered in [0.687,0.787]"
    missing3 = [k for k, v in obs3.items() if v is None]
    if missing3:
        add("P3 n=3 unchanged from displacement", INCONCL, obs3, thr3,
            f"absent from nback_levels diagnostics[3]: {', '.join(missing3)}")
    else:
        ok = (obs3["keys_held"] >= 3.5 and obs3["answered"] >= 13.0
              and 0.687 <= obs3["acc_over_answered"] <= 0.787)
        note = "" if ok else (
            "DISCONFIRMED: episode-scoping of rehearsal is not doing what is "
            "claimed -- most likely some turns emit two writes, in which case the "
            "first of the pair gains protection")
        v, vn = _void_if_identical(
            PASS if ok else FAIL, obs3,
            {k: b3.get(k) for k in obs3} if b3 else None,
            "answered moved 6.82 -> 14.00 baseline->displacement, keys_held is "
            "1.06 under full_context", same)
        add("P3 n=3 unchanged from displacement", v, obs3, thr3, vn or note)

    # --- P4 no-change (attribution): digit span untouched ---------------------
    # "abs(sub_span_leak - 0.1298) <= 0.01 and abs(best_span - 18.4) <= 1.0 and
    #  abs(delta_ds_fwd) < 0.140 and abs(delta_ds_rev) < 0.059"
    a1 = ax.get("A1") or {}
    b_a1 = bax.get("A1") or {}
    ref_leak = b_a1.get("sub_span_leak", PRIM_A1_LEAK)
    ref_span = b_a1.get("best_span", PRIM_A1_SPAN)
    obs4 = {"sub_span_leak": a1.get("sub_span_leak"), "best_span": a1.get("best_span"),
            "d_fwd": delta.get("digit_span_forward"),
            "d_rev": delta.get("digit_span_reverse")}
    thr4 = (f"|leak - {ref_leak}| <= 0.01, |best_span - {ref_span}| <= 1.0, "
            f"|d_fwd| < 0.140, |d_rev| < 0.059")
    missing4 = [k for k, v in obs4.items() if v is None]
    if missing4:
        add("P4 digit span untouched", INCONCL, obs4, thr4,
            f"absent: {', '.join(missing4)} (A1 needs "
            f"tasks/wm_digit_span_forward.jsonl; the deltas need --baseline)")
    else:
        ok = (abs(obs4["sub_span_leak"] - ref_leak) <= 0.01
              and abs(obs4["best_span"] - ref_span) <= 1.0
              and abs(obs4["d_fwd"]) < 0.140 and abs(obs4["d_rev"]) < 0.059)
        # NO-CHANGE row: identity to the baseline IS the prediction, so it is
        # tagged FROZEN rather than voided.
        tags, fn_note = _frozen_tag(
            {"sub_span_leak": obs4["sub_span_leak"], "best_span": obs4["best_span"],
             "d_fwd": obs4["d_fwd"], "d_rev": obs4["d_rev"]},
            ({"sub_span_leak": b_a1.get("sub_span_leak"),
              "best_span": b_a1.get("best_span"), "d_fwd": 0.0, "d_rev": 0.0}
             if b_a1 else None), same)
        add("P4 digit span untouched", PASS if ok else FAIL, obs4, thr4,
            fn_note or ("" if ok else
                        "the rule is firing on digit-span rows it should pass "
                        "through; this is the direct contrast with random_decay_v2, "
                        "which bought the aggregate and broke exactly this "
                        "structure"), tags)

    # --- P5 riskiest: A3 guard does not fire ---------------------------------
    # "bleu < 0.023 ... and 115 <= words <= 145". Transcribed as written, but note
    # the staleness: score_candidate no longer ENFORCES bleu (`bleu_enforced:
    # false`, A3_ENFORCED_FIELDS = ('precision_distance','word_distance')), so the
    # row's claim that 0.023 "is where the guard fires" is no longer true. The
    # enforced quantity is reported as a companion row rather than substituted.
    a3 = ax.get("A3") or {}
    obs5 = {"bleu": a3.get("bleu"), "words": a3.get("words")}
    thr5 = "bleu < 0.023 AND 115 <= words <= 145"
    if obs5["bleu"] is None or obs5["words"] is None:
        add("P5 A3 guard does not fire (BLEU, as pre-registered)", INCONCL, obs5,
            thr5, "absent: axes.A3 (needs tasks/wm_semantic_story_recall.jsonl)")
    else:
        ok = obs5["bleu"] < 0.023 and 115 <= obs5["words"] <= 145
        v, vn = _void_if_identical(
            PASS if ok else FAIL, obs5,
            ({"bleu": (bax.get("A3") or {}).get("bleu"),
              "words": (bax.get("A3") or {}).get("words")} if bax.get("A3") else None),
            "0.0001 (random_decay) to 0.3047 (full_context) across prior runs",
            same)
        add("P5 A3 guard does not fire (BLEU, as pre-registered)", v, obs5, thr5,
            vn or ("" if ok else "DISCONFIRMED at bleu >= 0.023 or length outside "
                   "115-145; BLEU in this length regime is brevity-penalty "
                   "dominated, so this is largely a prediction about length"),
            ("NEEDS-REVIEW",))
        add("P5b A3 enforced quantity (precision_distance) -- companion, reported",
            INCONCL,
            {"precision": a3.get("precision"),
             "precision_distance": a3.get("precision_distance")},
            f"guard fires above baseline precision_distance + "
            f"{SC.A3_PRECISION_TOLERANCE}",
            "P5 thresholds BLEU, which score_candidate no longer enforces "
            "(bleu_enforced: false). The enforced pair is "
            "(precision_distance, word_distance); reported so the row's "
            "'where the guard fires' claim can be audited.", ("REPORTED",))

    # --- P6 subgroup: ">= 0.9181 AND > 0.9161" -------------------------------
    gist = _subgroup_mean(hl, PRIM_GIST)
    b_gist = _subgroup_mean(bhl, PRIM_GIST)
    thr6 = ">= 0.9181 (baseline - 0.0136) AND > 0.9161 (displacement + 0.0136)"
    if gist is None:
        add("P6 3-task gist subgroup recovers", INCONCL, None, thr6,
            f"one of {PRIM_GIST} did not score")
    else:
        ok = gist >= 0.9181 and gist > 0.9161
        v, vn = _void_if_identical(
            PASS if ok else FAIL, round(gist, 4),
            None if b_gist is None else round(b_gist, 4),
            "displacement's -0.0293 on this subgroup exceeds its 0.0136 min "
            "credible delta", same)
        add("P6 3-task gist subgroup recovers", v, round(gist, 4), thr6, vn)

    # --- P7 subgroup, flagged UNDER-POWERED by its own author ----------------
    # ">= 0.9139 (baseline - 0.0157)", but min credible delta on this quantity is
    # 0.0314, so a movement inside that band cannot be scored in either direction.
    store5 = _subgroup_mean(hl, PRIM_STORE5)
    b_store5 = _subgroup_mean(bhl, PRIM_STORE5)
    thr7 = ">= 0.9139 (baseline - one SE); min credible delta 0.0314"
    if store5 is None:
        add("P7 5-task store subgroup: no credible regression", INCONCL, None,
            thr7, f"one of {PRIM_STORE5} did not score", ("UNDER-POWERED",))
    else:
        ref5 = 0.9296 if b_store5 is None else b_store5
        if abs(store5 - ref5) < 0.0314:
            v = INCONCL
            note = (f"movement of {store5 - ref5:+.4f} is inside this quantity's "
                    f"own 0.0314 min credible delta -- UNDER-POWERED, and this is "
                    f"the row that corrects the brief's reading of displacement's "
                    f"-0.0148 as a regression")
        else:
            v = PASS if store5 >= 0.9139 else FAIL
            note = ""
        v, vn = _void_if_identical(v, round(store5, 4),
                                   None if b_store5 is None else round(b_store5, 4),
                                   "only weakly -- see the row's own note", same)
        add("P7 5-task store subgroup: no credible regression", v,
            round(store5, 4), thr7, vn or note, ("UNDER-POWERED",))

    # --- P8 floor: craft_task and narrative_qa each stay inside the floor -----
    obs8 = {"craft_task": delta.get("craft_task"),
            "narrative_qa": delta.get("narrative_qa")}
    if any(v is None for v in obs8.values()):
        add("P8 craft_task and narrative_qa inside the floor", INCONCL, obs8,
            "delta > -0.03 each",
            "absent: delta_vs_baseline for craft_task / narrative_qa "
            "(needs --baseline)")
    else:
        ok = all(v > -0.03 for v in obs8.values())
        tags, fn_note = _frozen_tag(obs8, {"craft_task": 0.0, "narrative_qa": 0.0},
                                    same)
        add("P8 craft_task and narrative_qa inside the floor", PASS if ok else FAIL,
            obs8, "delta > -0.03 each", fn_note, tags)

    # --- P9 headline, explicitly NOT the claim -------------------------------
    mean = rec.get("mean_humanlikeness_search")
    if mean is None:
        add("P9 8-task mean >= 0.80 (headline, not the claim)", INCONCL, None,
            ">= 0.80", "field absent: mean_humanlikeness_search", ("REPORTED",))
    else:
        add("P9 8-task mean >= 0.80 (headline, not the claim)",
            PASS if mean >= 0.80 else FAIL, mean,
            ">= 0.80 (only >= 0.8121 would be a credible gain over the baseline)",
            "stated before the run so it cannot be claimed after it; the primary "
            "claims are P1-P5", ("REPORTED",))

    # --- P10 no-change, NEAR-INERT by the row's own admission ----------------
    d_vm = delta.get("variable_mapping")
    a4 = ax.get("A4") or {}
    if d_vm is None:
        add("P10 variable_mapping unchanged; A4 owes nothing", INCONCL, None,
            "abs(delta) < 0.017",
            "field absent: delta_vs_baseline.variable_mapping", ("NEAR-INERT",))
    elif abs(d_vm) < 0.017:
        # No identity-VOID here: the row registers the quantity as "barely"
        # capable of varying, so a value equal to the baseline's is what the row
        # itself expects rather than evidence that the test could not run.
        add("P10 variable_mapping unchanged; A4 owes nothing", PASS, d_vm,
            "abs(delta) < 0.017",
            "the row declares this quantity near-inert (the store never overflows "
            "on this task and is off the causal path), so a PASS here is not a "
            "passed test", ("NEAR-INERT",))
    else:
        # ">= 30 errors with rc_ratio >= 1.15", as the row words it. STALE, and
        # reported as such: score_candidate's A4 guard now tests
        # rc_ratio_normalized < 0.15 rather than the raw ratio, because the raw
        # ratio's ceiling moves with the error count.
        ok = (a4.get("n_errors") or 0) >= 30 and (a4.get("rc_ratio") or 0) >= 1.15
        add("P10 variable_mapping unchanged; A4 owes nothing",
            PASS if ok else FAIL, {"delta": d_vm, "n_errors": a4.get("n_errors"),
                                   "rc_ratio": a4.get("rc_ratio"),
                                   "rc_ratio_normalized": a4.get("rc_ratio_normalized")},
            "abs(delta) < 0.017, else A4 must show >= 30 errors and rc_ratio >= 1.15",
            "variable_mapping moved past its noise floor, so A4 became binding" +
            ("" if ok else " and was not met -- the gain is unstructured"),
            ("NEAR-INERT",))

    # --- C1 covariate: NO directional prediction, but a disconfirming condition
    d_wr = delta.get("word_recognition")
    a2 = ax.get("A2") or {}
    obs_c1 = {"delta_hl": d_wr, "miss_rate": a2.get("miss_rate"),
              "fa_rate": a2.get("fa_rate"), "ratio": a2.get("ratio"),
              "distance": a2.get("distance"),
              "trials_attempted": a2.get("trials_attempted")}
    if d_wr is None or a2.get("fa_rate") is None:
        add("C1 word_recognition / A2 (reported; disconfirms if fa_rate > 0.211)",
            INCONCL, obs_c1, "none -- report only",
            "absent: delta_vs_baseline.word_recognition or axes.A2.fa_rate",
            ("REPORTED",))
    elif abs(d_wr) > 0.121 and a2["fa_rate"] > 0.211:
        add("C1 word_recognition / A2 (reported; disconfirms if fa_rate > 0.211)",
            FAIL, obs_c1, "fa_rate <= 0.211 if hl moves past its 0.121 floor",
            "DISCONFIRMING CONDITION FIRED: the rule is reproducing "
            "displacement's liberal-bias artifact (humanlikeness rising because "
            "the model says 'old' more often, away from the human conservative "
            "bias of miss 0.272 vs fa 0.045). The gain must NOT be credited.")
    else:
        add("C1 word_recognition / A2 (reported; disconfirms if fa_rate > 0.211)",
            INCONCL, obs_c1, "none -- report only",
            "no direction is derivable: the task is a confirmed leak and this "
            "rule's keyset differs from both prior candidates, so there is no "
            "base rate", ("REPORTED",))

    return out


# ---------------------------------------------------------------------------
# serial_recognition (iteration 2b) -- pending_serial_recognition.json
# ---------------------------------------------------------------------------
# Every row there carries `verdict_if_missing: INCONCLUSIVE`, which is honoured
# by construction below: no branch reaches FAIL without the quantity in hand.
_WR_LINE = re.compile(r"^trial\s+(\d+):\s*(Old|New)\s*$", re.I)
_WR_HDR_COV = re.compile(r"coverage=([0-9.]+)")


def _wr_summary(run_dir: Path | None) -> dict[str, Any] | None:
    """The four word-recognition quantities no existing tooling reports.

    `compute` strings from the pending file, implemented literally:
      ceiling_group  sum(1 for s in wr_scores if s >= 0.98)
      mean_score     mean(wr_scores)            [metrics.score / 100]
      coverage       parseable Old/New judgements / 100, per row then averaged
      old_rate       parsed judgements equal to 'old' / 100, per row then averaged
    The old-rate denominator is 100 rather than the number parsed, per the literal
    "over all 100 trials". The two readings agree on any run with full coverage
    and diverge only on a collapsed one, which is the case the row is guarding.
    """
    if run_dir is None:
        return None
    rows = _jsonl(run_dir, "wm_word_recognition")
    if rows is None:
        return None
    scores, covs, olds = [], [], []
    for r in rows:
        sc = (r.get("metrics") or {}).get("score")
        if sc is not None:
            scores.append(float(sc) / 100.0)
        raw = str(r.get("recall_raw") or "")
        judged: dict[int, str] = {}
        for line in raw.splitlines():
            m = _WR_LINE.match(line.strip())
            if m:
                judged[int(m.group(1))] = m.group(2).lower()
        n_gold = len(r.get("gold_trials") or []) or 100
        # The harness writes a '# serial_recognition ... coverage=' header; prefer
        # it, because it counts what the harness itself accepted rather than what
        # this regex can re-parse.
        hdr = next((ln for ln in raw.splitlines()
                    if ln.lstrip().startswith("# serial_recognition")), None)
        m_cov = _WR_HDR_COV.search(hdr) if hdr else None
        covs.append(float(m_cov.group(1)) if m_cov else len(judged) / n_gold)
        olds.append(sum(1 for v in judged.values() if v == "old") / n_gold)
    if not scores:
        return {"n_rows": len(rows), "ceiling_group": None, "mean_score": None,
                "coverage": None, "old_rate": None}
    return {
        "n_rows": len(rows),
        "ceiling_group": sum(1 for s in scores if s >= 0.98),
        "mean_score": round(_mean(scores) or 0.0, 4),
        "coverage": round(_mean(covs) or 0.0, 4) if covs else None,
        "old_rate": round(_mean(olds) or 0.0, 4) if olds else None,
    }


def _open_arm_dir(run_dir: Path) -> tuple[Path | None, str]:
    """The ablation arm's run dir: the sibling `<arm>_open/<model>/`.

    Returns (dir or None, the path that was tried) so a missing arm reports the
    path it looked for rather than a bare INCONCLUSIVE.
    """
    run_dir = Path(run_dir)
    root = run_dir.parent.parent / f"{run_dir.parent.name}_open"
    sib = root / run_dir.name
    if (sib / "tasks/wm_word_recognition.jsonl").exists():
        return sib, str(sib)
    hits = sorted(root.glob("*/tasks/wm_word_recognition.jsonl")) if root.exists() else []
    if len(hits) == 1:
        return hits[0].parent.parent, str(hits[0].parent.parent)
    return None, str(sib) + (f" (and {len(hits)} glob matches under {root})"
                             if hits else "")


def serial_recognition_checks(run_dir: Path,
                              baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    delta = rec.get("delta_vs_baseline") or {}
    wr = _wr_summary(run_dir)

    def a2(field: str) -> tuple[Any, str | None]:
        return _get(rec, f"axes.A2.{field}")

    # --- P1 CENTRAL: the ceiling group collapses ------------------------------
    if wr is None or wr.get("ceiling_group") is None:
        add("P1 ceiling group collapses (<= 3 of 50)", INCONCL, None, "<= 3",
            "absent: tasks/wm_word_recognition.jsonl (or no metrics.score in it)")
    else:
        n = wr["ceiling_group"]
        if n <= 3:
            v, note = PASS, ""
        elif n >= 15:
            v, note = FAIL, ("DISCONFIRMED (>= 15 of 50 still at ceiling): the "
                             "studied list was not what produced the ceiling "
                             "group and the leak diagnosis is wrong")
        else:
            v, note = FAIL, ("fell but not to the stated bar; under displacement "
                             "the group fell only 36 -> 30, so a partial fall is "
                             "not enough")
        add("P1 ceiling group collapses (<= 3 of 50)", v, n,
            "<= 3 of 50 (baseline 36; disconfirms at >= 15)", note)

    # --- P2 mean word-recognition score falls --------------------------------
    if wr is None or wr.get("mean_score") is None:
        add("P2 mean word_recognition score < 0.35", INCONCL, None, "< 0.35",
            "absent: tasks/wm_word_recognition.jsonl metrics.score")
    else:
        s = wr["mean_score"]
        add("P2 mean word_recognition score < 0.35",
            PASS if s < 0.35 else FAIL, s,
            "< 0.35 (baseline 0.816, human 0.315; disconfirms above 0.60)",
            "" if s < 0.35 else (
                "above 0.60: the model is still answering at near-leak accuracy "
                "from four chunks, which the store-decidability audit says is "
                "impossible -- something other than memory is supplying the answer"
                if s > 0.60 else "fell short of the stated bar"))

    # --- P3 / P4 A2 rates rise ------------------------------------------------
    # The disconfirming line in both rows is "the baseline's own rate", and the
    # literals in the pending file are rounded (0.043, 0.127) while the measured
    # values are 0.0432 and 0.1269. Comparing against the rounded literal makes a
    # run that did not move at all read as having "risen" or "fallen", so the
    # measured baseline is preferred wherever it is available.
    b_a2 = (_axes(baseline) or {}).get("A2") or {} if baseline is not None else {}
    for name, field, thr, lit, disnote in (
        ("P3 A2 miss_rate > 0.10", "miss_rate", 0.10, 0.043,
         "misses FELL: the model got better at detecting old words after losing "
         "the list, which is incoherent and means the serialisation is not doing "
         "what it claims"),
        ("P4 A2 fa_rate > 0.20", "fa_rate", 0.20, 0.127,
         "false alarms FELL: the model became conservative, which would refute "
         "the familiarity-without-recollection reading"),
    ):
        val, why = a2(field)
        dis = b_a2.get(field, lit)
        if why:
            add(name, INCONCL, None, f"> {thr}", why)
        else:
            add(name, PASS if val > thr else FAIL, val,
                f"> {thr} (baseline {dis}; disconfirms at <= {dis})",
                "" if val > thr else (disnote if val < dis else
                                      "did not move off the baseline" if val == dis
                                      else "rose but short of the stated bar"))

    # --- P5 trials_attempted falls, and the comparability GATE ---------------
    trials, why = a2("trials_attempted")
    gate_fired = trials is not None and trials < 15
    if why:
        add("P5 A2 trials_attempted < 40", INCONCL, None, "< 40.0", why)
    else:
        add("P5 A2 trials_attempted < 40", PASS if trials < 40.0 else FAIL, trials,
            "< 40.0 (baseline 82.9, human 34.5; disconfirms above 60)",
            ("GATE FIRED (< 15): A2's per-participant rates rest on ~5 trials, so "
             "its distance from the human 6.09 is NOT interpretable in either "
             "direction. " if gate_fired else "") +
            ("" if trials < 40.0 else
             "the model is surviving nearly as long as it did with the list "
             "visible" if trials > 60 else "fell short of the stated bar"),
            ("GATED",) if gate_fired else ())

    # --- P6 A2 distance does not approach human ------------------------------
    # The gate names A2's DISTANCE explicitly as uninterpretable, so when it fires
    # this row is INCONCLUSIVE rather than scored. P3/P4 keep their verdicts,
    # tagged GATED: their disconfirming conditions still carry signal.
    dist, why = a2("distance")
    if why:
        add("P6 A2 distance > 4.0 (does not approach human)", INCONCL, None,
            "> 4.0", why)
    elif gate_fired:
        add("P6 A2 distance > 4.0 (does not approach human)", INCONCL, dist,
            "> 4.0", "P5's gate fired (trials_attempted < 15), and the gate names "
            "A2's distance as not interpretable in either direction", ("GATED",))
    else:
        add("P6 A2 distance > 4.0 (does not approach human)",
            PASS if dist > 4.0 else FAIL, dist,
            "> 4.0 (baseline 5.754; disconfirms below 2.0)",
            "closing the leak removes a mixture artifact; it does not install a "
            "conservative response criterion, so a value near 5.754 is a "
            "coincidence of the mixture and must not be read as 'A2 unchanged'"
            if dist > 4.0 else
            "below 2.0: closing the leak alone fixed the miss/false-alarm "
            "asymmetry, which would make a separate response-criterion candidate "
            "unnecessary" if dist < 2.0 else "moved toward human")
    if gate_fired:
        for r in out:
            if r["prediction"].split()[0] in ("P3", "P4") and "GATED" not in r["tags"]:
                r["tags"].append("GATED")

    # --- P7 / P8 / P9 NO-CHANGE rows, all-within bands -----------------------
    for name, bands, extra in (
        ("P7 digit span untouched",
         {"axes.A1.sub_span_leak": (0.1198, 0.1398),
          "axes.A1.best_span": (17.9, 18.9),
          "delta_vs_baseline.digit_span_forward": (-0.140, 0.140),
          "delta_vs_baseline.digit_span_reverse": (-0.059, 0.059)},
         "any movement outside the bands means the recall() override is firing on "
         "prompts it should pass through, and nothing else in the run can be "
         "attributed"),
        ("P8 nback untouched",
         {"delta_vs_baseline.nback": (-0.060, 0.060),
          "axes.nback_levels.diagnostics.3.answered": (5.82, 7.82)},
         "n-back answers through step(), which this candidate never calls. If "
         "n-back moves here, this candidate and iteration 2c are entangled and "
         "neither is attributable"),
        ("P9 other tasks within floor",
         {"delta_vs_baseline.variable_mapping": (-0.017, 0.017),
          "delta_vs_baseline.craft_task": (-0.025, 0.025),
          "delta_vs_baseline.narrative_qa": (-0.030, 0.030),
          "delta_vs_baseline.semantic_story_recall": (-0.030, 0.030)},
         "semantic_story_recall's band is deliberately the 0.030 enforced floor, "
         "not its 0.011 noise floor -- treat that leg as soft and the other three "
         "as hard"),
    ):
        obs: dict[str, Any] = {}
        absent: list[str] = []
        bad: list[str] = []
        for path, (lo, hi) in bands.items():
            val, why = _get(rec, path)
            obs[path.split(".")[-1]] = val
            if why:
                absent.append(why)
            elif not (lo <= float(val) <= hi):
                bad.append(f"{path.split('.')[-1]}={_fmt(val)} outside [{lo}, {hi}]")
        thr = "; ".join(f"{p.split('.')[-1]} in [{lo}, {hi}]"
                        for p, (lo, hi) in bands.items())
        if absent:
            add(name, INCONCL, obs, thr, "; ".join(absent))
        else:
            add(name, PASS if not bad else FAIL, obs, thr,
                extra if not bad else "; ".join(bad) + " -- " + extra)

    # P9 also asserts `floor_violations` is empty. Kept as its own row because a
    # large POSITIVE word_recognition delta cannot violate a floor (the check is
    # `d < -eff`), so this is not in tension with P13.
    fv = _dotted(rec, "floor_violations")
    if fv is MISSING:
        add("P9b floor_violations empty", INCONCL, None, "empty list",
            "field absent: floor_violations (needs --baseline)")
    else:
        add("P9b floor_violations empty", PASS if not fv else FAIL,
            [f"{v['task']} {v['delta']}" for v in fv] or "none", "empty list",
            "only regressions can violate a floor, so P13's large positive "
            "word_recognition delta cannot appear here")

    # --- P10 / P11 ANTI-GARBAGE ---------------------------------------------
    if wr is None or wr.get("coverage") is None:
        add("P10 response coverage >= 0.95", INCONCL, None, ">= 0.95",
            "absent: tasks/wm_word_recognition.jsonl recall_raw")
    else:
        cov = wr["coverage"]
        add("P10 response coverage >= 0.95", PASS if cov >= 0.95 else FAIL, cov,
            ">= 0.95 (baseline 1.0; disconfirms below 0.90)",
            "" if cov >= 0.95 else (
                "below 0.90: the result is a compliance or parsing failure and NO "
                "claim about memory can be made from it -- score_game treats an "
                "unparsed trial as neither correct nor an error, so garbage posts "
                "a low score and a FLATTERED humanlikeness"
                if cov < 0.90 else "short of the stated bar"))
    if wr is None or wr.get("old_rate") is None:
        add("P11 old-rate within [0.15, 0.95]", INCONCL, None, "[0.15, 0.95]",
            "absent: tasks/wm_word_recognition.jsonl recall_raw")
    else:
        orate = wr["old_rate"]
        add("P11 old-rate within [0.15, 0.95]",
            PASS if 0.15 <= orate <= 0.95 else FAIL, orate,
            "[0.15, 0.95] (baseline 0.528, per-participant 0.23-0.79)",
            "band is wide on the high side on purpose: the expected failure mode "
            "is an OLD bias, which is a real bias and must not be scored as "
            "degeneracy" if 0.15 <= orate <= 0.95 else
            "a constant answer or a collapse into a single token sits at 0 or 1")

    # --- P12 CENTRAL / ATTRIBUTION: the open ablation arm --------------------
    open_dir, tried = _open_arm_dir(Path(run_dir))
    owr = _wr_summary(open_dir)
    if owr is None or owr.get("ceiling_group") is None:
        add("P12 open ablation reproduces the baseline", INCONCL, None,
            "ceiling >= 25 of 50 AND mean score >= 0.70",
            f"open arm not found: {tried}. The void condition therefore could not "
            f"be EVALUATED, so P1/P2's collapse is unattributed -- not "
            f"attributed, and not voided either.")
        for r in out:
            if r["prediction"].split()[0] in ("P1", "P2"):
                r["tags"].append("UNATTRIBUTED")
    else:
        oc, om = owr["ceiling_group"], owr["mean_score"]
        obs12 = {"ceiling_group": oc, "mean_score": om}
        if oc < 15 or (om is not None and om < 0.50):
            add("P12 open ablation reproduces the baseline", VOID, obs12,
                "ceiling >= 25 of 50 AND mean score >= 0.70",
                "VOID CONDITION FIRED: the OPEN arm also collapsed (ceiling < 15 "
                "or mean < 0.50), so the masked arm's collapse is caused by the "
                "serial framing, the call pattern or the stitching rather than by "
                "removing information. This candidate's claim is VOID regardless "
                "of how good its humanlikeness looks.")
            _void_rows(out, ("P1", "P2"),
                       "VOID via P12: the open arm collapsed too, so this "
                       "collapse is not attributable to the leak.")
        elif oc >= 25 and om is not None and om >= 0.70:
            add("P12 open ablation reproduces the baseline", PASS, obs12,
                "ceiling >= 25 of 50 AND mean score >= 0.70",
                "the open arm runs identical machinery and differs only in that "
                "the other 99 trial lines stay visible, so a masked collapse "
                "beside an open non-collapse is caused by removing information")
        else:
            add("P12 open ablation reproduces the baseline", FAIL, obs12,
                "ceiling >= 25 of 50 AND mean score >= 0.70",
                "the open arm neither reproduced the baseline nor collapsed; "
                "attribution is weakened but the explicit void condition "
                "(ceiling < 15 or mean < 0.50) did not fire")

    # --- P13 / P14 ENTAILED: arithmetic consequences, not evidence -----------
    d_wr, why = _get(rec, "delta_vs_baseline.word_recognition")
    add("P13 word_recognition humanlikeness rises (ENTAILED)",
        INCONCL if why else (PASS if d_wr >= 0.121 else FAIL), d_wr, ">= 0.121",
        why or ("ENTAILED by any large accuracy drop, NOT evidence of a "
                "mechanism: the baseline sits at 0.816 against a human mean of "
                "0.315, so even a point mass at zero would give humanlikeness "
                "~0.685. A fall would instead mean P1 and P2 also failed."),
        ("ENTAILED",))
    mean = rec.get("mean_humanlikeness_search")
    add("P14 8-task search mean rises (ENTAILED)",
        INCONCL if mean is None else (PASS if mean >= 0.8121 else FAIL), mean,
        ">= 0.8121 (baseline 0.7861 + min credible mean delta 0.026)",
        "field absent: mean_humanlikeness_search" if mean is None else
        "ENTAILED and carried almost entirely by word_recognition. This is an "
        "instrument fix, so the mean is NOT the claim and must not be quoted as a "
        "capability gain; the valid word_recognition contrast is masked vs open.",
        ("ENTAILED",))

    # Reported, not scored: A2's discrimination index. A sum >= 1.0 is at-or-below
    # chance and is EXPECTED here (the 7 store-consulting baseline participants sum
    # to 1.167); P10 and P12 are the garbage tests, not this.
    miss, why_m = a2("miss_rate")
    fa, why_f = a2("fa_rate")
    add("R1 discrimination index miss+fa (reported, not scored)",
        INCONCL, None if (why_m or why_f) else round(miss + fa, 4),
        "expected band [0.9, 1.4] (baseline 0.170, human 0.317)",
        (why_m or why_f) or "at-or-below-chance discrimination is expected: it is "
        "familiarity without recollection from a store built with foreknowledge "
        "of the whole list", ("REPORTED",))

    # Reported, not scored: the '# serial_recognition' header's `mode` must read
    # `masked` in every row of the candidate arm and `open` in every row of the
    # ablation. A mixed or absent mode means the arms are not what they claim, so
    # this is an integrity check on the arm labelling rather than a prediction.
    rows_wr = _jsonl(Path(run_dir), "wm_word_recognition")
    if rows_wr is None:
        add("R2 arm mode header (reported integrity check)", INCONCL, None,
            "mode reads `masked` (candidate) or `open` (ablation) in every row",
            "absent: tasks/wm_word_recognition.jsonl", ("REPORTED",))
    else:
        modes: dict[str, int] = defaultdict(int)
        for r in rows_wr:
            hdr = next((ln for ln in str(r.get("recall_raw") or "").splitlines()
                        if ln.lstrip().startswith("# serial_recognition")), "")
            m = re.search(r"mode=(\w+)", hdr)
            modes[m.group(1) if m else "no-header"] += 1
        add("R2 arm mode header (reported integrity check)", INCONCL, dict(modes),
            "mode reads `masked` (candidate) or `open` (ablation) in every row",
            "a `no-header` count equal to the row count means this run predates "
            "the serialisation harness (the iter0 baseline does), not that the arm "
            "is mislabelled; a MIXED count would mean the arms are entangled",
            ("REPORTED",))
    return out


# ---------------------------------------------------------------------------
# episodic_reset (iteration 2c) -- pending_episodic_reset.json
# ---------------------------------------------------------------------------
_BARE_CAP = re.compile(r"(?<![A-Za-z])[A-Z](?![A-Za-z])")
EPI_NOCHANGE_TASKS = ["word_recognition", "digit_span_forward", "digit_span_reverse",
                      "semantic_story_recall", "craft_task", "narrative_qa"]


def _vm_row_stats(run_dir: Path) -> dict[str, Any] | None:
    """P3/P4/P8 quantities, straight off tasks/wm_variable_mapping.jsonl."""
    rows = _jsonl(run_dir, "wm_variable_mapping")
    if rows is None:
        return None
    n = len(rows)
    early = sum(1 for r in rows
                if (r.get("metrics") or {}).get("first_error_at") is not None
                and (r["metrics"]["first_error_at"]) <= 5)
    scores = [(r.get("metrics") or {}).get("score") for r in rows]
    at_ceiling = sum(1 for s in scores if s == 10)
    # Parse integrity: exactly interference.py's `selected is None` case, and
    # restricted to C2 rows for the same reason -- that is the only condition
    # interference scores, so a count over all rows would not be the same number.
    letters = "ABCDEFGH"
    unparsed = considered = 0
    for r in rows:
        if (r.get("condition_id") or r.get("condition")) != "C2":
            continue
        answers = r.get("parsed_answers") or {}
        for q in (r.get("questions") or []):
            considered += 1
            idx = q.get("question_index")
            letter = answers.get(str(idx), answers.get(idx))
            opts = q.get("options") or []
            ok = (isinstance(letter, str) and letter in letters
                  and letters.index(letter) < len(opts))
            unparsed += 0 if ok else 1
    return {
        "n_rows": n,
        "early_error_frac": round(early / n, 4) if n else None,
        "ceiling_frac": round(at_ceiling / n, 4) if n else None,
        "n_distinct_scores": len({s for s in scores if s is not None}),
        "unparsed_answers": unparsed,
        "questions_considered": considered,
    }


def _nback_letter_share(run_dir: Path, level: int = 3) -> tuple[float | None, int]:
    """P6: share of n=3 final_kv values containing a bare capital letter."""
    rows = _jsonl(run_dir, "wm_nback")
    if rows is None:
        return None, 0
    vals = [str(v) for r in rows if r.get("n_level") == level
            for v in (r.get("final_kv") or {}).values()]
    if not vals:
        return None, 0
    return round(sum(bool(_BARE_CAP.search(v)) for v in vals) / len(vals), 4), len(vals)


def episodic_reset_checks(run_dir: Path,
                          baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    delta = rec.get("delta_vs_baseline") or {}
    bax = _axes(baseline) if baseline is not None else {}
    a4 = ax.get("A4") or {}
    if not a4 and Path(run_dir).exists():
        try:
            a4 = IF.a4(Path(run_dir))
        except Exception:  # noqa: BLE001
            a4 = {}
    vm = _vm_row_stats(Path(run_dir))

    # --- P1 primary: the store becomes load-bearing on variable_mapping -------
    # source: meta_harness.interference.a4(run_dir)['n_errors'], >= 150.
    # The bar is 150 and not 30 because displacement already produced 35 errors,
    # so 30 no longer discriminates.
    n_err = a4.get("n_errors")
    if n_err is None:
        add("P1 A4 n_errors >= 150", INCONCL, None, ">= 150",
            "absent: interference.a4(run_dir)['n_errors'] (needs "
            "tasks/wm_variable_mapping.jsonl)")
        p1_pass = False
    else:
        p1_pass = n_err >= 150
        add("P1 A4 n_errors >= 150", PASS if p1_pass else FAIL, n_err,
            ">= 150 (baseline 12, displacement 35; point estimate 225-600)",
            "" if p1_pass else (
                "DISCONFIRMED (< 75): the store was never needed and the leak "
                "reading is wrong" if n_err < 75 else
                "rose but short of 150 -- do not round up to a confirmation"))

    # --- P2 rc_ratio > 1.15, CONDITIONAL ON P1 --------------------------------
    rc = a4.get("rc_ratio")
    obs2 = {"rc_ratio": rc, "rc_ratio_normalized": a4.get("rc_ratio_normalized"),
            "rc_ratio_ceiling": a4.get("rc_ratio_ceiling"),
            "n_errors": n_err, "trustworthy": a4.get("trustworthy")}
    if rc is None:
        add("P2 A4 rc_ratio > 1.15 (conditional on P1)", INCONCL, obs2, "> 1.15",
            "absent: interference.a4(run_dir)['rc_ratio']")
    elif not p1_pass:
        # Not scored when P1 fails: at 12 errors the BASELINE already reads 1.1682
        # with trustworthy=false, so scoring this unconditionally would manufacture
        # a PASS out of a handful of errors.
        add("P2 A4 rc_ratio > 1.15 (conditional on P1)", INCONCL, obs2, "> 1.15",
            "conditional_on P1, which did not pass: rc_ratio at a small error "
            "count is not interpretable (the baseline reads 1.1682 at 12 errors "
            "with trustworthy=false, and displacement's 1.2575 at 35 errors was "
            "exactly its arithmetic ceiling)")
    else:
        add("P2 A4 rc_ratio > 1.15 (conditional on P1)",
            PASS if rc > 1.15 else FAIL, obs2,
            "> 1.15 (human 1.386, pure noise 1.0; ceiling at 150 errors is 1.2879)",
            "necessary but not sufficient for an interference account: for the "
            "model relation_count is collinear with question index, so this says "
            "only that errors are concentrated late"
            if rc > 1.15 else
            "ACCEPTED AS DISCONFIRMING: 150+ errors with rc_ratio <= 1.15 means "
            "errors are uniform over question index -- the model was made noisy, "
            "not memory-limited")

    # --- P3 primary: early errors, the quantity the score actually depends on --
    if vm is None or vm.get("early_error_frac") is None:
        add("P3 share of rows with first_error_at <= 5 > 0.25", INCONCL, None,
            "> 0.25", "absent: tasks/wm_variable_mapping.jsonl metrics.first_error_at")
    else:
        f = vm["early_error_frac"]
        add("P3 share of rows with first_error_at <= 5 > 0.25",
            PASS if f > 0.25 else FAIL, f,
            "> 0.25 (baseline 0.0133; point estimate 0.87; disconfirms at <= 0.10)",
            "0.25 is a deliberate lower bound and will not discriminate a strong "
            "result from a marginal one -- read the observed value against the "
            "0.87 point estimate" if f > 0.25 else
            ("DISCONFIRMED (<= 0.10)" if f <= 0.10 else "rose but short of the bar")
            + ". This cannot be bought by late errors, which is what distinguishes "
            "this candidate from displacement.")

    # --- P4 primary: the ceiling point mass breaks up -------------------------
    if vm is None or vm.get("ceiling_frac") is None:
        add("P4 variable_mapping ceiling point mass breaks up", INCONCL, None,
            "ceiling_frac < 0.8 AND distinct scores >= 5",
            "absent: tasks/wm_variable_mapping.jsonl metrics.score")
    else:
        cf, nd = vm["ceiling_frac"], vm["n_distinct_scores"]
        ok = cf < 0.8 and nd >= 5
        add("P4 variable_mapping ceiling point mass breaks up",
            PASS if ok else FAIL, {"frac_at_10": cf, "distinct_scores": nd},
            "frac_at_score_10 < 0.8 AND distinct scores >= 5 (baseline 0.9867, 2)",
            "weaker proxy for P3, retained because a point-mass prediction was "
            "asked for explicitly" if ok else
            ("point mass INTACT (>= 0.95 at ceiling), which contradicts P1 and P3"
             if cf >= 0.95 else "moved but not to both bars"))

    # --- P5 primary: variable_mapping humanlikeness rises by >= 0.05 ----------
    d_vm, why = _get(rec, "delta_vs_baseline.variable_mapping")
    if why:
        add("P5 variable_mapping humanlikeness rises >= 0.05", INCONCL, None,
            ">= 0.05", why)
    elif d_vm > 0.6:
        add("P5 variable_mapping humanlikeness rises >= 0.05", FAIL, d_vm,
            ">= 0.05 and <= 0.6 (point estimate 0.45-0.55)",
            "OVERSHOOT, not calibration: past +0.6 the model has gone beyond the "
            "human mean of 0.394 and is now WORSE than humans on this task")
    else:
        add("P5 variable_mapping humanlikeness rises >= 0.05",
            PASS if d_vm >= 0.05 else FAIL, d_vm,
            ">= 0.05 (noise floor 0.017; point estimate 0.45-0.55)",
            "" if d_vm >= 0.05 else (
                # The row's own `reads_as` applies only when P1 passed; without
                # that condition the same number means something else entirely.
                "with P1 passing this reads as the score formula absorbing the "
                "errors -- apply protocol_match_rule before interpreting"
                if p1_pass else
                "P1 also failed, so this is consistent with the store never "
                "becoming load-bearing rather than with the score formula "
                "absorbing errors"))

    # --- P6 primary: the n=3 store starts carrying letter identity -----------
    share, n_vals = _nback_letter_share(Path(run_dir))
    if share is None:
        add("P6 n=3 store carries letter identity > 0.4", INCONCL, None, "> 0.4",
            "absent: tasks/wm_nback.jsonl rows with n_level == 3")
        p6_pass = False
    else:
        p6_pass = share > 0.4
        add("P6 n=3 store carries letter identity > 0.4",
            PASS if p6_pass else FAIL, share,
            f"> 0.4 over {n_vals} values (baseline 0.0202, full_context 1.0; "
            f"disconfirms at <= 0.2)",
            "the decisive mechanism test on nback: it cannot be bought by "
            "guessing or by breaking output parsing" if p6_pass else
            "DISCONFIRMED (<= 0.2): the n=3 store still holds no letters, so its "
            "accuracy is still coming from history rather than from the store"
            if share <= 0.2 else "rose but short of the bar")

    # --- P7 branch discriminator: REPORTED, not thresholded ------------------
    d3 = (NL.report(Path(run_dir)).get("diagnostics") or {}).get(3) or {} \
        if Path(run_dir).exists() else {}
    obs7 = {k: d3.get(k) for k in ("keys_held", "answered", "acc_over_answered")}
    branch = None
    if p6_pass and obs7["keys_held"] is not None:
        if obs7["keys_held"] < 2.5:
            branch = "a"        # rolling single-key strategy, never overflows
        elif obs7["keys_held"] >= 3.5 and (obs7["answered"] or 0) < 6.82:
            branch = "b"        # refuse-on-full jam dominates
    add("P7 n=3 keys_held direction (branch discriminator, REPORTED)", INCONCL,
        dict(obs7, branch=branch or "indeterminate"), "none -- reported jointly "
        "with P6 and answered",
        "branch (a) = agent adopted full_context's rolling single-key strategy "
        "(nback may improve, for the right reason); branch (b) = the refuse-on-full "
        "jam dominates and closing the leak needs proposer 2a's overflow fix. "
        "Informative either way, not a refutation." if obs7["keys_held"] is not None
        else "absent: nback_levels diagnostics[3].keys_held", ("REPORTED",))

    # --- P8 guard: variable_mapping output format intact ---------------------
    if vm is None:
        add("P8 variable_mapping parse integrity < 30 bad answers", INCONCL, None,
            "< 30", "absent: tasks/wm_variable_mapping.jsonl")
    else:
        bad = vm["unparsed_answers"]
        add("P8 variable_mapping parse integrity < 30 bad answers",
            PASS if bad < 30 else FAIL,
            {"unparsed": bad, "of_questions": vm["questions_considered"]},
            "< 30 (baseline 0 of 1500)",
            "interference.py counts an unparsed answer as a novel_guess ERROR, so "
            "a candidate that reached P1 by emitting garbage inflates n_errors "
            "while failing here")

    # --- P9 guard: n=1 unaffected. ITS FAILURE VOIDS P1-P7. ------------------
    d1 = (NL.report(Path(run_dir)).get("diagnostics") or {}).get(1) or {} \
        if Path(run_dir).exists() else {}
    obs9 = {"acc_over_answered": d1.get("acc_over_answered"),
            "answered": d1.get("answered")}
    if any(v is None for v in obs9.values()):
        add("P9 n=1 nback unaffected (KEY TEST)", INCONCL, obs9,
            "acc_over_answered >= 0.9 AND answered >= 13.0",
            "absent: nback_levels diagnostics[1] (needs tasks/wm_nback.jsonl "
            "rows with n_level == 1)")
    else:
        ok9 = obs9["acc_over_answered"] >= 0.9 and obs9["answered"] >= 13.0
        add("P9 n=1 nback unaffected (KEY TEST)", PASS if ok9 else FAIL, obs9,
            "acc_over_answered >= 0.9 AND answered >= 13.0 (baseline 0.9943, 13.98)",
            "one item in four slots cannot be a capacity failure, so this holding "
            "is what licenses reading P1-P7 at all" if ok9 else
            "A P9 FAILURE INVALIDATES P1-P7 REGARDLESS OF THEIR VALUES: at n=1 a "
            "single overwritten key suffices, so a drop here can only mean the "
            "reset broke instruction-following, phase tracking or output parsing")
        if not ok9:
            _void_rows(out, ("P1", "P2", "P3", "P4", "P5", "P6", "P7"),
                       "VOID via P9: the n=1 control failed, which the candidate "
                       "pre-registered as invalidating P1-P7 regardless of their "
                       "values.")

    # --- P10 no-change control: six tasks, byte-identical by construction ----
    # "Expected delta is EXACTLY 0; any nonzero movement is serving
    # nondeterminism and must not be narrated as a mechanism effect."
    obs10: dict[str, Any] = {}
    bad10: list[str] = []
    absent10: list[str] = []
    for task in EPI_NOCHANGE_TASKS:
        val, why = _get(rec, f"delta_vs_baseline.{task}")
        obs10[task] = val
        if why:
            absent10.append(why)
        elif abs(float(val)) > 0.005:
            bad10.append(f"{task} delta {val:+.4f}")
    for path, tol in (("axes.A1.sub_span_leak", 0.01), ("axes.A1.best_span", 0.5),
                      ("axes.A2.ratio", 0.02), ("axes.A3.bleu", 0.005),
                      ("axes.A3.words", 5.0)):
        val, why = _get(rec, path)
        bval, bwhy = (_get(bax, path[len("axes."):]) if bax else (None, "no baseline"))
        short = path[len("axes."):]
        obs10[short] = val
        if why or bwhy:
            absent10.append(why or f"baseline {short}: {bwhy}")
        elif abs(float(val) - float(bval)) > tol:
            bad10.append(f"{short} {_fmt(val)} vs baseline {_fmt(bval)} (tol {tol})")
    thr10 = ("each of six humanlikeness deltas <= 0.005; A1 leak 0.01, best_span "
             "0.5, A2 ratio 0.02, A3 bleu 0.005, A3 words 5.0")
    if absent10:
        add("P10 six tasks byte-identical (no-change control)", INCONCL, obs10,
            thr10, "; ".join(dict.fromkeys(absent10)))
    else:
        add("P10 six tasks byte-identical (no-change control)",
            PASS if not bad10 else FAIL, obs10, thr10,
            "structural, not statistical: encode() calls step() once and all six "
            "tasks build a fresh agent per trial, so there is no history to drop"
            if not bad10 else "; ".join(bad10))

    # --- P11 mean, CONDITIONAL on P7's branch --------------------------------
    mean = rec.get("mean_humanlikeness_search")
    d_mean = None if mean is None else round(
        mean - (0.7861 if baseline is None else
                (_mean([v for t, v in (_hl(baseline) or {}).items()
                        if t in SC.SEARCH_TASKS and v is not None]) or 0.7861)), 4)
    ranges = {"a": (0.05, 0.10), "b": (-0.04, 0.05)}
    if mean is None:
        add("P11 mean humanlikeness delta (conditional on P7's branch)", INCONCL,
            None, "branch a: [0.05,0.10]; branch b: [-0.04,0.05]",
            "field absent: mean_humanlikeness_search", ("REPORTED",))
    elif branch is None:
        add("P11 mean humanlikeness delta (conditional on P7's branch)", INCONCL,
            {"delta": d_mean, "branch": "indeterminate"},
            "branch a: [0.05,0.10]; branch b: [-0.04,0.05]",
            "P7 did not select a branch (it requires P6 to pass), so neither "
            "pre-registered range applies -- the delta is reported against both. "
            "An unconditional range would have been miscalibrated the way "
            "iteration 1's word_recognition threshold was.", ("REPORTED",))
    else:
        lo, hi = ranges[branch]
        add("P11 mean humanlikeness delta (conditional on P7's branch)",
            PASS if lo <= d_mean <= hi else FAIL,
            {"delta": d_mean, "branch": branch}, f"branch {branch}: [{lo}, {hi}]",
            "NOT a capability result: it is what happens when two tasks stop "
            "being scored on a leak, and its variable_mapping component is not "
            "interpretable until protocol_match_rule is applied",
            ("REPORTED",))
    return out


# ---------------------------------------------------------------------------
# chunk_limit (iteration 2d) -- pending_chunk_limit.json
# ---------------------------------------------------------------------------
_LOWER_TOKEN = re.compile(r"[a-z]+")


def _count_elements() -> Callable[[str], int] | None:
    """The candidate's own element-splitting rule, imported rather than re-derived.

    A re-derivation that disagreed with the mechanism by one delimiter would
    silently mis-score every one of P1/P2/P3/P10, and the disagreement would look
    like a result. Import failure is reported, not papered over with a local regex.
    """
    try:
        from meta_harness.candidates.chunk_limit.harness import count_elements
        return count_elements
    except Exception:  # noqa: BLE001
        return None


def _element_stats(run_dir: Path, task: str,
                   count_elements: Callable[[str], int]) -> dict[str, Any] | None:
    rows = _jsonl(run_dir, task)
    if rows is None:
        return None
    per_value, per_store, words = [], [], []
    for r in rows:
        vals = [str(v) for v in (r.get("final_kv") or {}).values()]
        counts = [count_elements(v) for v in vals]
        per_value.extend(counts)
        per_store.append(sum(counts))
        words.extend(len(v.split()) for v in vals)
    if not per_value:
        return {"n_rows": len(rows), "n_values": 0}
    return {
        "n_rows": len(rows), "n_values": len(per_value),
        "elements_per_value_mean": round(_mean(per_value) or 0.0, 3),
        "elements_per_value_max": max(per_value),
        "elements_per_store_mean": round(_mean(per_store) or 0.0, 3),
        "elements_per_store_max": max(per_store),
        "words_per_value_mean": round(_mean(words) or 0.0, 3),
    }


def _studied_words_in_store(run_dir: Path) -> dict[str, Any] | None:
    """P1 leg B: distinct lowercase [a-z]+ tokens shared between the concatenated
    final_kv values and encoding_log.content, on word_recognition.

    This leg is what makes the element count proof against PUNCTUATION gaming (a
    space-separated list satisfies an element count while keeping every word). It
    is NOT proof against slot reallocation, which is what the third outcome
    records.
    """
    rows = _jsonl(run_dir, "wm_word_recognition")
    if rows is None:
        return None
    counts = []
    for r in rows:
        kv = " ".join(str(v) for v in (r.get("final_kv") or {}).values()).lower()
        content = str((r.get("encoding_log") or {}).get("content") or "").lower()
        counts.append(len(set(_LOWER_TOKEN.findall(kv))
                          & set(_LOWER_TOKEN.findall(content))))
    if not counts:
        return None
    return {"mean": round(_mean(counts) or 0.0, 3), "max": max(counts),
            "n_rows": len(counts)}


def _narrative_bound_split(run_dir: Path, count_elements: Callable[[str], int],
                           cap: int) -> dict[str, Any] | None:
    """P10: split narrative_qa rows by whether the bound touched them, using the
    run's OWN pre-bound tool-call arguments (the bound is deterministic, so which
    rows it touched is recomputable from the log rather than guessed)."""
    rows = _jsonl(run_dir, "wm_narrative_qa")
    if rows is None:
        return None
    bounded, unbounded = [], []
    for r in rows:
        touched = False
        for tc in ((r.get("encoding_log") or {}).get("tool_calls") or []):
            if tc.get("name") != "write_memory":
                continue
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if count_elements(str(args.get("value", ""))) > cap:
                touched = True
                break
        acc = (r.get("metrics") or {}).get("accuracy")
        if acc is None:
            continue
        (bounded if touched else unbounded).append(float(acc))
    return {
        "n_bounded": len(bounded), "n_unbounded": len(unbounded),
        "acc_bounded": None if not bounded else round(_mean(bounded) or 0.0, 4),
        "acc_unbounded": None if not unbounded else round(_mean(unbounded) or 0.0, 4),
    }


def chunk_limit_checks(run_dir: Path, baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    same = _same_run(run_dir, baseline)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    delta = rec.get("delta_vs_baseline") or {}
    bax = _axes(baseline) if baseline is not None else {}
    ce = _count_elements()
    cap = _max_keys()
    p3_void = False

    if ce is None:
        add("P1/P2/P3/P10 element-based rows", INCONCL, None,
            "count_elements from candidates/chunk_limit/harness.py",
            "could not import count_elements from "
            "meta_harness/candidates/chunk_limit/harness.py, and the MANIFEST "
            "forbids re-deriving the element rule, so every element-based row is "
            "unscoreable on this invocation")
    else:
        wr_stats = _element_stats(Path(run_dir), "wm_word_recognition", ce)
        b_wr_stats = (_element_stats(Path(baseline), "wm_word_recognition", ce)
                      if baseline is not None else None)
        studied = _studied_words_in_store(Path(run_dir))
        b_studied = (_studied_words_in_store(Path(baseline))
                     if baseline is not None else None)

        # --- P1 LEG A (invariant, guaranteed if injection worked) ------------
        # "elements_per_value_mean <= 5.0 AND elements_per_store_mean <= 16.0 AND
        #  elements_per_value_max == 4"
        # `== 4` is read as the invariant it states in P3 ("never > 4"): a maximum
        # BELOW 4 satisfies the bound, and is flagged rather than failed because it
        # means the bound never actually bit.
        leg_a_ok = False
        if wr_stats is None or not wr_stats.get("n_values"):
            add("P1a LEG A word_recognition element invariant", INCONCL, wr_stats,
                "elements/value mean <= 5.0, elements/store mean <= 16.0, "
                "max elements/value <= 4",
                "absent: tasks/wm_word_recognition.jsonl (or no final_kv values)")
        else:
            leg_a_ok = (wr_stats["elements_per_value_mean"] <= 5.0
                        and wr_stats["elements_per_store_mean"] <= 16.0
                        and wr_stats["elements_per_value_max"] <= 4)
            note = ""
            if leg_a_ok and wr_stats["elements_per_value_max"] < 4:
                note = ("max elements/value is BELOW 4, so the bound never bit at "
                        "its own limit -- the invariant holds but leg A is not "
                        "evidence that the cap engaged")
            elif not leg_a_ok:
                note = ("if elements_per_store_mean stays above 16 the bound is "
                        "not being applied at all and the run is void rather "
                        "than failed -- see P3, which is the invariant's own test")
            v, vn = _void_if_identical(
                PASS if leg_a_ok else FAIL, wr_stats, b_wr_stats,
                "13.21 elements/value in iteration 0 and 16.29 under "
                "displacement, max 99 and 93", same)
            add("P1a LEG A word_recognition element invariant", v, wr_stats,
                "elements/value mean <= 5.0, elements/store mean <= 16.0, "
                "max elements/value <= 4 (baseline 13.21 / 44.9 / 99)",
                vn or note, ("NEEDS-REVIEW",) if leg_a_ok and
                wr_stats["elements_per_value_max"] < 4 else ())

        # --- P1 LEG B (the only behavioural leg) + the explicit THIRD OUTCOME -
        # "<= 8.9 (half the baseline's 17.78)"; studied words in 8.9-16 WITH leg A
        # satisfied means the bound held and the agent REALLOCATED to fill the
        # 16-element ceiling -- recorded as adaptation, not as a mechanism failure.
        if studied is None:
            add("P1b LEG B studied words in store <= 8.9", INCONCL, None,
                "<= 8.9",
                "absent: tasks/wm_word_recognition.jsonl final_kv / "
                "encoding_log.content")
        else:
            m = studied["mean"]
            if m <= 8.9:
                v, note = PASS, (f"expected ~4.6 on replay; observed over "
                                 f"{studied['n_rows']} rows, max {studied['max']}")
            elif m <= 16.0 and leg_a_ok:
                v, note = ADAPTED, (
                    "THIRD OUTCOME, neither pass nor fail: the bound HELD (leg A "
                    "satisfied) and the agent REALLOCATED to fill the 16-element "
                    "ceiling -- four slots of four studied words instead of one "
                    "slot of 99. The rule alone guarantees only a ~10% reduction "
                    "on this leg, so this is not a mechanism failure.")
            else:
                v, note = FAIL, (
                    "above the bound's own 16-element ceiling"
                    + ("" if leg_a_ok else " AND leg A failed, so the bound is "
                       "not being applied at all"))
            v2, vn = _void_if_identical(v, m, (b_studied or {}).get("mean"),
                                        "17.78 in the baseline run, 4.62 "
                                        "replaying the real write sequences",
                                        same)
            add("P1b LEG B studied words in store <= 8.9", v2, m,
                "<= 8.9 (baseline 17.78); 8.9-16 with leg A satisfied is the "
                "registered THIRD OUTCOME", vn or note)

        # --- P2 evasion check: words per value <= 14 -------------------------
        if wr_stats is None or not wr_stats.get("n_values"):
            add("P2 word_recognition words per value <= 14", INCONCL, None,
                "<= 14.0", "absent: tasks/wm_word_recognition.jsonl final_kv")
        else:
            w = wr_stats["words_per_value_mean"]
            v, vn = _void_if_identical(
                PASS if w <= 14.0 else FAIL, w,
                (b_wr_stats or {}).get("words_per_value_mean"),
                "22.62 baseline, 27.86 under displacement, max 111 and 425 words "
                "in a single value", same)
            add("P2 word_recognition words per value <= 14", v, w,
                "<= 14.0 (baseline 22.62, replay 8.1)", vn or
                "read together with P1's studied-word leg: low elements + high "
                "words + high studied-word count is evasion; low elements + high "
                "words + LOW studied-word count is genuine prose consolidation")

        # --- P3 plumbing invariant: max elements per value across ALL 8 tasks -
        # "== 4 (never > 4 anywhere)". A value above 4 anywhere means some task
        # module is still bound to the original store, so the run is VOID rather
        # than failed -- the same class of defect as the injection bug.
        maxima: dict[str, Any] = {}
        absent3 = []
        for task in ALL_TASK_FILES:
            st = _element_stats(Path(run_dir), task, ce)
            if st is None:
                absent3.append(task)
            elif st.get("n_values"):
                maxima[task.replace("wm_", "")] = st["elements_per_value_max"]
        if not maxima:
            add("P3 max elements per value <= 4 across all 8 tasks", INCONCL, None,
                "<= 4 everywhere",
                f"no task file present: {', '.join(absent3)}")
        else:
            worst = max(maxima.values())
            if worst > cap:
                over = {k: v for k, v in maxima.items() if v > cap}
                p3_void = True
                add("P3 max elements per value <= 4 across all 8 tasks", VOID,
                    maxima, f"<= {cap} everywhere (baseline max 99)",
                    f"VOID, not FAIL: {over} exceed the bound, so some task module "
                    f"is still bound to the ORIGINAL store and the run does not "
                    f"test the mechanism at all. Every mechanism-dependent row "
                    f"below is tagged UNATTRIBUTED for that reason -- the rows are "
                    f"NOT rewritten to VOID, so a reader can still see what the "
                    f"numbers were"
                    + (f". Tasks absent from this run: {', '.join(absent3)}"
                       if absent3 else ""))
            else:
                add("P3 max elements per value <= 4 across all 8 tasks", PASS,
                    maxima, f"<= {cap} everywhere (baseline max 99)",
                    f"tasks absent from this run: {', '.join(absent3)}"
                    if absent3 else "")

        # --- P10 ANTI-VANDALISM, row level -----------------------------------
        split = _narrative_bound_split(Path(run_dir), ce, cap)
        b_split = (_narrative_bound_split(Path(baseline), ce, cap)
                   if baseline is not None else None)
        ref_unb = ((b_split or {}).get("acc_unbounded")
                   if b_split and b_split.get("acc_unbounded") is not None
                   else 0.7500)
        if split is None or split.get("acc_unbounded") is None:
            add("P10 narrative_qa accuracy unchanged on rows the bound never "
                "touched", INCONCL, split, f"|acc_unbounded - {ref_unb}| < 0.05",
                "absent: tasks/wm_narrative_qa.jsonl, or no row escaped the bound")
        else:
            d = abs(split["acc_unbounded"] - ref_unb)
            # NO-CHANGE row (anti-vandalism): identity is the prediction.
            tags, fn_note = _frozen_tag(split["acc_unbounded"],
                                        (b_split or {}).get("acc_unbounded"), same)
            add("P10 narrative_qa accuracy unchanged on rows the bound never "
                "touched", PASS if d < 0.05 else FAIL, split,
                f"|acc_unbounded - {ref_unb}| < 0.05",
                fn_note or ("" if d < 0.05 else
                       "accuracy fell on rows the bound never touched, so "
                       "something other than the bound changed the agent's "
                       "behaviour -- most likely the tool-result sentence leaking "
                       "a strategy change across rows"), tags)

    # --- P4 story recall LENGTH leg, pre-registered as a COST ----------------
    a3 = ax.get("A3") or {}
    b_a3 = bax.get("A3") or {}
    words = a3.get("words")
    if words is None:
        add("P4 A3 recall length falls to 104-120 words", INCONCL, None,
            "104 <= words <= 120", "absent: axes.A3.words")
    else:
        ok = 104 <= words <= 120
        v, vn = _void_if_identical(PASS if ok else FAIL, words, b_a3.get("words"),
                                   "65 (random_decay), 121.4 (baseline), 146.2 "
                                   "(displacement), 411 (full_context)", same)
        add("P4 A3 recall length falls to 104-120 words", v, words,
            "104 <= words <= 120 (baseline 121.4; disconfirms above 125 or "
            "below 95)", vn or (
                "PRE-REGISTERED AS A COST: human is 137.2, so word_distance moves "
                "15.8 -> ~26, AWAY from human. The 40-word A3 tolerance absorbs "
                "it, but the guard clearing is not the move being good."
                if ok else
                "cut far more than the enumerated surplus" if words < 95 else
                "outside the band but inside the row's own disconfirmation lines "
                "(95-125), so this is short of the prediction rather than a "
                "disconfirmation" if words <= 125 else
                "above 125 has two readings and P1 disambiguates them: "
                "bound-never-fired shows max elements/value > 4, whereas "
                "bound-held-plus-adaptation shows compliant element counts with "
                "high word counts"))

    # --- P5 story recall VERBATIMNESS leg (the ENFORCED quantity, not BLEU) --
    prec = a3.get("precision")
    if prec is None:
        add("P5 A3 median 4-gram precision in 0.030-0.055", INCONCL, None,
            "0.030 <= precision <= 0.055",
            "absent: axes.A3.precision (error_structure.a3_precision_model)")
    else:
        ok = 0.030 <= prec <= 0.055
        v, vn = _void_if_identical(
            PASS if ok else FAIL, prec, b_a3.get("precision"),
            "median precision 1.0000 under full_context, 0.0414 baseline, 0.0345 "
            "displacement", same)
        add("P5 A3 median 4-gram precision in 0.030-0.055", v,
            {"precision": prec, "precision_distance": a3.get("precision_distance")},
            "0.030 <= precision <= 0.055 (disconfirms at >= 0.0613, where "
            "precision_distance exceeds baseline 0.0221 + tolerance 0.02)",
            vn or ("two-sided on purpose: truncation keeps the opening of a value "
                   "and drops trailing clauses, so if the dropped clauses were the "
                   "paraphrased ones precision could RISE. Read with +-0.004 of "
                   "the proxy error the row states." if ok else
                   "DISCONFIRMED at >= 0.0613: the A3 guard fires" if prec >= 0.0613
                   else "outside the band but below the guard"))

    # --- P6 story recall SCORE leg: floor only, DIRECTION NOT DERIVABLE ------
    d_story, why = _get(rec, "delta_vs_baseline.semantic_story_recall")
    if why:
        add("P6 semantic_story_recall inside the floor (no direction asserted)",
            INCONCL, None, "delta > -0.03", why)
    else:
        # Floor-only row with no direction asserted, so identity to the baseline is
        # inside what it predicts: FROZEN, not VOID.
        tags, fn_note = _frozen_tag(d_story, 0.0, same)
        add("P6 semantic_story_recall inside the floor (no direction asserted)",
            PASS if d_story > -0.03 else FAIL, d_story,
            "delta > -0.03 (the floor displacement violated at "
            "-0.0509)", fn_note or ("NO DIRECTION IS ASSERTED: every prior "
            "perturbation of this store lost here in both directions, which is "
            "the signature of a task near a local optimum where distribution "
            "SHAPE dominates" if d_story > -0.03 else
            "DISCONFIRMED: the bound cut chunk CONTENT rather than enumerated "
            "surplus"), tags)

    # --- P7 no-change, PRIMARY CONTROL --------------------------------------
    d_craft, why = _get(rec, "delta_vs_baseline.craft_task")
    if why:
        add("P7 craft_task unchanged (primary control)", INCONCL, None,
            "abs(delta) < 0.025", why)
    else:
        tags, fn_note = _frozen_tag(d_craft, 0.0, same)
        add("P7 craft_task unchanged (primary control)",
            PASS if abs(d_craft) < 0.025 else FAIL, d_craft,
            "abs(delta) < 0.025 (baseline 0.8907)", fn_note or (
                "craft_task holds 1.00 elements per value, so the bound cannot "
                "bite and the tool results are byte-identical to the baseline's"
                if abs(d_craft) < 0.025 else
                "movement past the noise floor cannot come from the bound; it "
                "would have to come from something unintended -- most plausibly "
                "the tool-result sentence changing strategy on other tasks"), tags)

    # --- P8 no-change: nback n=3 is the BASELINE's, not displacement's -------
    d3 = (NL.report(Path(run_dir)).get("diagnostics") or {}).get(3) or {} \
        if Path(run_dir).exists() else {}
    d_nb = delta.get("nback")
    obs8 = {"keys_held": d3.get("keys_held"), "answered": d3.get("answered"),
            "acc_over_answered": d3.get("acc_over_answered"), "delta_nback": d_nb}
    thr8 = "keys_held >= 3.5 AND 5.3 <= answered <= 8.3 AND abs(delta_nback) < 0.060"
    missing8 = [k for k, v in obs8.items() if v is None]
    if missing8:
        add("P8 nback n=3 is the baseline's", INCONCL, obs8, thr8,
            f"absent: {', '.join(missing8)} (delta_nback needs --baseline; the "
            f"rest need tasks/wm_nback.jsonl)")
    else:
        ok = (obs8["keys_held"] >= 3.5 and 5.3 <= obs8["answered"] <= 8.3
              and abs(d_nb) < 0.060)
        b3: dict[str, Any] = {}
        if baseline is not None and Path(baseline).exists():
            b3 = (NL.report(Path(baseline)).get("diagnostics") or {}).get(3) or {}
        tags, fn_note = _frozen_tag(
            {k: obs8[k] for k in ("keys_held", "answered", "acc_over_answered")},
            ({"keys_held": b3.get("keys_held"), "answered": b3.get("answered"),
              "acc_over_answered": b3.get("acc_over_answered")} if b3 else None),
            same)
        add("P8 nback n=3 is the baseline's", PASS if ok else FAIL, obs8, thr8,
            fn_note or (
            "expected values are the BASELINE's, not displacement's: this "
            "candidate does not change overflow policy. HONEST LIMITATION: "
            "wm_nback.jsonl records no tool calls, so this is the one prediction "
            "that could not be replayed offline." if ok else
            "nback moved although nback values hold 1.13 elements each (max 2), "
            "so three or four letters in one value are still inside the cap"), tags)

    # --- P9 no-change, NEAR-INERT by the row's own admission ----------------
    d_vm = delta.get("variable_mapping")
    a4 = ax.get("A4") or {}
    if d_vm is None:
        add("P9 variable_mapping unchanged; A4 owes nothing", INCONCL, None,
            "abs(delta) < 0.017",
            "field absent: delta_vs_baseline.variable_mapping", ("NEAR-INERT",))
    elif abs(d_vm) < 0.017:
        add("P9 variable_mapping unchanged; A4 owes nothing", PASS, d_vm,
            "abs(delta) < 0.017",
            "the row declares this quantity near-inert (values hold exactly 1.00 "
            "element each, so the bound cannot bite), so a PASS is not a passed "
            "test", ("NEAR-INERT",))
    else:
        ok = (a4.get("n_errors") or 0) >= 30 and (a4.get("rc_ratio") or 0) >= 1.15
        add("P9 variable_mapping unchanged; A4 owes nothing",
            PASS if ok else FAIL,
            {"delta": d_vm, "n_errors": a4.get("n_errors"),
             "rc_ratio": a4.get("rc_ratio")},
            "abs(delta) < 0.017, else A4 must show >= 30 errors and rc_ratio >= 1.15",
            "variable_mapping moved past its noise floor, so A4 became binding"
            + ("" if ok else " and was not met"), ("NEAR-INERT",))

    # --- P11 ANTI-VANDALISM, aggregate level / A1 guard ---------------------
    a1 = ax.get("A1") or {}
    obs11 = {"sub_span_leak": a1.get("sub_span_leak"),
             "best_span": a1.get("best_span"),
             "d_fwd": delta.get("digit_span_forward"),
             "d_rev": delta.get("digit_span_reverse")}
    thr11 = ("sub_span_leak <= 0.1450 AND abs(best_span - 18.4) <= 2.0 AND "
             "abs(d_fwd) < 0.140 AND abs(d_rev) < 0.059")
    missing11 = [k for k, v in obs11.items() if v is None]
    if missing11:
        add("P11 digit span: a bound, not vandalism", INCONCL, obs11, thr11,
            f"absent: {', '.join(missing11)}")
    else:
        ok = (obs11["sub_span_leak"] <= 0.1450
              and abs(obs11["best_span"] - 18.4) <= 2.0
              and abs(obs11["d_fwd"]) < 0.140 and abs(obs11["d_rev"]) < 0.059)
        note = ""
        if not ok and obs11["sub_span_leak"] > 0.1450 and \
                abs(obs11["best_span"] - 18.4) <= 2.0:
            note = ("DISCONFIRMED in the registered shape: the leak rose past "
                    "0.1450 while best_span barely moved -- that is vandalism "
                    "below the participant's own span, not a bound (the A1 guard "
                    "itself fires at 0.1598)")
        elif not ok:
            note = "outside the bands, but not in the vandalism shape"
        tags, fn_note = _frozen_tag(
            {"sub_span_leak": obs11["sub_span_leak"],
             "best_span": obs11["best_span"], "d_fwd": obs11["d_fwd"],
             "d_rev": obs11["d_rev"]},
            ({"sub_span_leak": (bax.get("A1") or {}).get("sub_span_leak"),
              "best_span": (bax.get("A1") or {}).get("best_span"),
              "d_fwd": 0.0, "d_rev": 0.0} if bax.get("A1") else None), same)
        add("P11 digit span: a bound, not vandalism", PASS if ok else FAIL, obs11,
            thr11, fn_note or note or (
                "the SHAPE of the effect is what is asserted: random information "
                "destruction raises the sub-span leak, whereas enforcing a bound "
                "leaves it flat"), tags)

    # --- P12 the one axis with real headroom --------------------------------
    a2 = ax.get("A2") or {}
    obs12 = {"miss_rate": a2.get("miss_rate"), "fa_rate": a2.get("fa_rate"),
             "ratio": a2.get("ratio"), "distance": a2.get("distance"),
             "trials_attempted": a2.get("trials_attempted")}
    thr12 = ("miss_rate > 0.0432 AND fa_rate <= 0.1269 AND ratio > 0.71 (the "
             "UPPER bound of the baseline's own bootstrap ratio_ci [0.007, 0.71])")
    if any(obs12[k] is None for k in ("miss_rate", "fa_rate", "ratio")):
        add("P12 A2 moves toward the human conservative bias", INCONCL, obs12,
            thr12, "absent: axes.A2 (needs tasks/wm_word_recognition.jsonl)")
    else:
        base_trials = ((bax.get("A2") or {}).get("trials_attempted")
                       if bax.get("A2") else 82.92)
        swing = (None if obs12["trials_attempted"] is None or base_trials is None
                 else round(obs12["trials_attempted"] - base_trials, 2))
        if obs12["fa_rate"] > 0.211:
            v, note = FAIL, ("DISCONFIRMING CONDITION FIRED: fa_rate above "
                             "displacement's 0.211 means the liberal-bias "
                             "artifact has been reproduced and the gain must NOT "
                             "be credited whatever the ratio did")
        elif (obs12["miss_rate"] > 0.0432 and obs12["fa_rate"] <= 0.1269
              and obs12["ratio"] > 0.71):
            v, note = PASS, ""
        elif (obs12["miss_rate"] > 0.0432 and obs12["fa_rate"] <= 0.1269
              and obs12["ratio"] > 0.340):
            v, note = FAIL, ("moved in the predicted direction but the ratio is "
                             "inside the baseline's own bootstrap CI [0.007, "
                             "0.71], so it is NOT creditable on its own")
        else:
            v, note = FAIL, "did not move in the predicted direction"
        if swing is not None and abs(swing) > 10:
            note += (f" || COVARIATE: trials_attempted moved {swing:+.2f} vs the "
                     f"baseline. PROPOSER.md is explicit that a ratio gain bought "
                     f"by surviving a different number of trials is not credited, "
                     f"so this ratio move is UNINTERPRETABLE, not a win.")
        v2, vn = _void_if_identical(
            v, {k: obs12[k] for k in ("miss_rate", "fa_rate", "ratio")},
            ({"miss_rate": (bax.get("A2") or {}).get("miss_rate"),
              "fa_rate": (bax.get("A2") or {}).get("fa_rate"),
              "ratio": (bax.get("A2") or {}).get("ratio")}
             if bax.get("A2") else None),
            "fa_rate moved 0.127 -> 0.211 between baseline and displacement",
            same)
        add("P12 A2 moves toward the human conservative bias", v2, obs12, thr12,
            vn or note)

    # --- P13 leak test, AT RISK BY DESIGN -----------------------------------
    d_wr, why = _get(rec, "delta_vs_baseline.word_recognition")
    if why:
        add("P13 word_recognition humanlikeness moves < its noise floor",
            INCONCL, None, "abs(delta) < 0.121", why)
    else:
        # A leak test asserting NO movement, so an exactly unmoved delta is its
        # strongest possible pass and must not be voided.
        tags, fn_note = _frozen_tag(d_wr, 0.0, same)
        add("P13 word_recognition humanlikeness moves < its noise floor",
            PASS if abs(d_wr) < 0.121 else FAIL, d_wr,
            "abs(delta) < 0.121", fn_note or (
                "strongest confirmation yet that the score is determined by the "
                "studied list re-presented in the recall prompt rather than by "
                "the store" if abs(d_wr) < 0.121 else
                "moved past the floor: either the leak reading needs revisiting "
                "or the 7 store-consulting participants moved a long way -- P12 "
                "tells those two apart"), tags)

    # --- P14 headline, explicitly NOT the claim -----------------------------
    mean = rec.get("mean_humanlikeness_search")
    if mean is None:
        add("P14 8-task mean shows no credible regression", INCONCL, None,
            ">= 0.7601", "field absent: mean_humanlikeness_search", ("REPORTED",))
    else:
        add("P14 8-task mean shows no credible regression",
            PASS if mean >= 0.7601 else FAIL, mean,
            ">= 0.7601 (baseline 0.7861 - 0.026); only >= 0.8121 would be a "
            "credible GAIN",
            "stated before the run so it cannot be claimed afterwards; the claims "
            "are P1, P12, P13, P10/P11", ("REPORTED",))

    # P3 is the plumbing invariant. If it went VOID the bound was not installed
    # everywhere, so every row that depends on the bound having bitten is
    # unattributed -- tagged rather than rewritten, because rewriting them to VOID
    # would also erase the negative control's "change-predictions fail" signal.
    if p3_void:
        dependent = ("P1a", "P1b", "P2", "P4", "P5", "P6", "P10", "P12", "P13")
        for r in out:
            if r["prediction"].split()[0] in dependent:
                r["tags"].append("UNATTRIBUTED")
    return out


# ===========================================================================
# Iteration 3
# ===========================================================================


def _nback_levels(run_dir: Path | None) -> dict[int, dict[str, Any]]:
    """Per-level n-back diagnostics, or {} when unavailable.

    Never raises: a candidate that suppresses responses entirely produces
    present-but-null metrics, which is exactly the case this has to describe.
    """
    if run_dir is None or not Path(run_dir).exists():
        return {}
    try:
        rep = NL.report(Path(run_dir))
    except Exception:  # noqa: BLE001
        return {}
    diag = rep.get("diagnostics", rep) if isinstance(rep, dict) else {}
    out: dict[int, dict[str, Any]] = {}
    if isinstance(diag, dict):
        for k, v in diag.items():
            if isinstance(v, dict):
                try:
                    out[int(k)] = v
                except (TypeError, ValueError):
                    continue
    return out


def _kv_occupancy(run_dir: Path, task: str,
                  only_overflowing: bool = False,
                  max_keys: int | None = None) -> dict[str, Any] | None:
    """Mean len(final_kv) for a task, optionally restricted to rows whose agent
    tried to write more distinct keys than capacity.

    `only_overflowing` is what separates "the bound fired" from "the task never
    filled the store", which is the distinction primacy_v2's P2 turns on.
    """
    rows = _jsonl(Path(run_dir), task)
    if not rows:
        return None
    cap = max_keys if max_keys is not None else _max_keys()
    occ, acc, n_over = [], [], 0
    for r in rows:
        kv = r.get("final_kv")
        if not isinstance(kv, dict):
            continue
        written = set()
        log = r.get("encoding_log") or {}
        for tc in (log.get("tool_calls") or []):
            if tc.get("name") != "write_memory":
                continue
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(args, dict) and args.get("key"):
                written.add(str(args["key"]))
        overflowed = len(written) > cap
        if overflowed:
            n_over += 1
        if only_overflowing and not overflowed:
            continue
        occ.append(len(kv))
        m = r.get("metrics") or {}
        a = m.get("accuracy", m.get("score"))
        if a is not None:
            acc.append(float(a))
    if not occ:
        return {"n": 0, "mean_kv": None, "mean_accuracy": None,
                "n_overflowing": n_over}
    return {
        "n": len(occ),
        "mean_kv": round(sum(occ) / len(occ), 4),
        "mean_accuracy": round(sum(acc) / len(acc), 4) if acc else None,
        "n_overflowing": n_over,
    }


def _answer_raw_len(run_dir: Path) -> dict[str, Any] | None:
    """Mean and max length of variable_mapping's raw answers.

    The anti-verbosity test: a candidate that prepends context to a prompt whose
    instruction is "output ONLY one line" can pass a content test while flooding
    the reply, and that would be a different mechanism than the one claimed.
    """
    rows = _jsonl(Path(run_dir), "wm_variable_mapping")
    if not rows:
        return None
    lens, unparsed, total = [], 0, 0
    for r in rows:
        for sl in (r.get("step_logs") or []):
            raw = sl.get("answer_raw")
            if raw is None:
                continue
            total += 1
            lens.append(len(str(raw)))
            if not sl.get("parsed"):
                unparsed += 1
    if not lens:
        return None
    return {"mean_chars": round(sum(lens) / len(lens), 2), "max_chars": max(lens),
            "unparsed": unparsed, "of_answers": total}


def episodic_reset_v2_checks(run_dir: Path,
                            baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    hl = rec.get("humanlikeness_by_task") or {}
    bhl = _hl(baseline)
    a4 = ax.get("A4") or {}
    if not a4 and Path(run_dir).exists():
        try:
            a4 = IF.a4(Path(run_dir))
        except Exception:  # noqa: BLE001
            a4 = {}
    lv = _nback_levels(Path(run_dir))

    # --- P1 PRECONDITION: the n=1 mechanics are repaired ---------------------
    # episodic_reset read 2.06 answered with 36 of 50 silent and keys_held 0.28.
    # At n=1 one overwritten key suffices, so a drop there can only be broken
    # mechanics. A failure here invalidates P2-P7, as its predecessor's P9 did.
    l1 = lv.get(1) or {}
    obs1 = {"answered": l1.get("answered"),
            "acc_over_answered": l1.get("acc_over_answered"),
            "n_no_answers": l1.get("n_no_answers"),
            "keys_held": l1.get("keys_held")}
    thr1 = ("answered >= 13.0 AND acc_over_answered >= 0.90 AND n_no_answers == 0 "
            "AND keys_held >= 0.8 (baseline 13.98/0.9943/0/1.00; "
            "episodic_reset 2.06/0.787/36/0.28)")
    if any(v is None for v in obs1.values()):
        add("P1 n=1 mechanics repaired -- PRECONDITION", INCONCL, obs1, thr1,
            "absent: nback_levels.report(run_dir)['diagnostics'][1]")
        p1 = False
    else:
        p1 = (obs1["answered"] >= 13.0 and obs1["acc_over_answered"] >= 0.90
              and obs1["n_no_answers"] == 0 and obs1["keys_held"] >= 0.8)
        add("P1 n=1 mechanics repaired -- PRECONDITION", PASS if p1 else FAIL,
            obs1, thr1,
            "" if p1 else "A P1 FAILURE INVALIDATES P2-P7 REGARDLESS OF VALUE")

    # --- P2 the leak stays closed, and not by restoring context ---------------
    share, n_vals = _nback_letter_share(Path(run_dir), 3)
    l3 = lv.get(3) or {}
    obs2 = {"n3_letter_share": share, "n_values": n_vals,
            "keys_held": l3.get("keys_held")}
    if share is None:
        add("P2 leak stays closed (n=3 letter share)", INCONCL, obs2, ">= 0.90",
            "absent: n=3 final_kv values")
    else:
        ok = share >= 0.90
        add("P2 leak stays closed (n=3 letter share)", PASS if ok else FAIL,
            obs2, ">= 0.90 over n=3 final_kv values "
                  "(baseline 0.0202, episodic_reset 1.0000)",
            "keys_held is REPORTED not gated: a leak-free store may legitimately "
            "be one rolling key, so a keys_held bar can fail while the mechanism "
            "works")

    # --- P3 nback clears its floor -------------------------------------------
    v, why = _get(rec, "humanlikeness_by_task.nback")
    if v is MISSING or v is None:
        add("P3 nback clears its floor", INCONCL, None, ">= 0.7309", why or "absent")
    else:
        ok = v >= 0.7309
        d = None if bhl.get("nback") is None else v - bhl["nback"]
        add("P3 nback clears its floor", PASS if ok else FAIL,
            {"nback": v, "delta": None if d is None else round(d, 4)},
            ">= 0.7309 (baseline 0.7909 minus the 0.060 floor); "
            "episodic_reset read 0.4970",
            "" if ok else "still violates the nback floor")

    # --- P4 A4 at scale ------------------------------------------------------
    n_err = a4.get("n_errors")
    if n_err is None:
        add("P4 A4 n_errors >= 150", INCONCL, None, ">= 150", "absent: A4.n_errors")
    else:
        add("P4 A4 n_errors >= 150", PASS if n_err >= 150 else FAIL, n_err,
            ">= 150 (baseline 12, displacement 35, episodic_reset 509)")

    # --- P5 A4 structure, normalized -----------------------------------------
    norm = a4.get("rc_ratio_normalized")
    obs5 = {"rc_ratio_normalized": norm, "rc_ratio": a4.get("rc_ratio"),
            "ceiling": a4.get("rc_ratio_ceiling"),
            "trustworthy": a4.get("trustworthy")}
    if norm is None:
        add("P5 A4 structure (normalized)", INCONCL, obs5, "in [0.15, 0.95]",
            "absent: A4.rc_ratio_normalized")
    else:
        ok = 0.15 <= norm <= 0.95
        add("P5 A4 structure (normalized)", PASS if ok else FAIL, obs5,
            "in [0.15, 0.95]; no direction asserted (humans 0.3728, noise 0.0, "
            "run's own ceiling 1.0; episodic_reset 0.7302)",
            "the raw ratio is NOT scale-free -- its ceiling moves with the error "
            "count -- which is why this row thresholds the normalized value")

    # --- P6 variable_mapping, both formulas ----------------------------------
    vmm = (ax.get("variable_mapping_matched") or {}).get("humanlikeness_matched")
    raw = hl.get("variable_mapping")
    obs6 = {"raw": raw, "matched": vmm}
    if raw is None or vmm is None:
        add("P6 variable_mapping, both formulas", INCONCL, obs6,
            ">= 0.55 on both", "absent: variable_mapping humanlikeness or the "
                               "matched-formula figure")
    else:
        ok = raw >= 0.55 and vmm >= 0.55
        add("P6 variable_mapping, both formulas", PASS if ok else FAIL, obs6,
            ">= 0.55 raw AND >= 0.55 matched (baseline 0.3554/0.3587, "
            "episodic_reset 0.6764/0.7295)",
            "a FALL from episodic_reset's figures is expected and is not a "
            "failure: this candidate restores task set, which should let the "
            "agent use the store better, and being better is a cost here")

    # --- P7 parse integrity and anti-verbosity -------------------------------
    ar = _answer_raw_len(Path(run_dir))
    if ar is None:
        add("P7 parse integrity + anti-verbosity", INCONCL, None,
            "unparsed < 30 AND mean answer_raw <= 15.0 chars",
            "absent: variable_mapping step_logs[*].answer_raw")
    else:
        ok = ar["unparsed"] < 30 and ar["mean_chars"] <= 15.0
        add("P7 parse integrity + anti-verbosity", PASS if ok else FAIL, ar,
            "unparsed < 30 AND mean answer_raw <= 15.0 chars "
            "(baseline 0 / 13.10, episodic_reset 0 / 13.18)",
            "" if ok else "prepending context to a prompt that says 'output ONLY "
                          "one line' can flood the reply; that would be a "
                          "different mechanism than the one claimed")

    # --- P8 no-change control ------------------------------------------------
    # digit_span_reverse is the verdict-bearing leg. NOT thresholded at
    # bit-identity: run-to-run measurement shows generated content varies on every
    # task, though this task's SCORE was reproduced exactly twice.
    obs8: dict[str, Any] = {}
    legs_ok = True
    for t, tol in (("digit_span_reverse", 0.059), ("digit_span_forward", 0.140),
                   ("craft_task", 0.030), ("narrative_qa", 0.030),
                   ("semantic_story_recall", 0.030), ("word_recognition", 0.121)):
        a, b = hl.get(t), bhl.get(t)
        if a is None or b is None:
            obs8[t] = None
            continue
        obs8[t] = round(a - b, 4)
        if abs(a - b) > tol:
            legs_ok = False
    add("P8 six tasks unchanged (control)", PASS if legs_ok else FAIL, obs8,
        "each within its enforced floor; digit_span_reverse is the verdict-bearing "
        "leg at 0.059",
        "measured run-to-run variation on these tasks is 0.0000-0.0152, so a "
        "movement inside the floors is not evidence of a mechanism")

    # --- P9 per-level, reported ----------------------------------------------
    obs9 = {n: {"answered": d.get("answered"),
                "acc": d.get("acc_over_answered"),
                "keys": d.get("keys_held"),
                "silent": f"{d.get('n_no_answers')}/{d.get('n_rows')}"}
            for n, d in sorted(lv.items())}
    add("P9 per-level n-back (reported)", INCONCL, obs9, "reported, not scored",
        "directions were stated in advance per level; read against the baseline "
        "13.98/13.24/6.82 answered and episodic_reset's 2.06/10.16/9.08",
        tags=("REPORTED",))

    if not p1:
        _void_rows(out, ("P2", "P3", "P4", "P5", "P6", "P7"),
                   "VOID via P1: the n=1 precondition failed, which this candidate "
                   "pre-registered as invalidating P2-P7 regardless of their values.")
    return out


def primacy_v2_checks(run_dir: Path,
                      baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    hl = rec.get("humanlikeness_by_task") or {}
    bhl = _hl(baseline)
    cap = _max_keys()
    lv = _nback_levels(Path(run_dir))

    # --- P1 primary: craft recovers past its ENFORCED floor ------------------
    # 0.8607 = baseline 0.8907 minus max(FLOOR 0.03, NOISE_FLOOR 0.025) = 0.030.
    # The enforced floor is 0.030, not the 0.025 bootstrap figure.
    v = hl.get("craft_task")
    if v is None:
        add("P1 craft_task recovers", INCONCL, None, ">= 0.8607", "absent")
        p1 = False
    else:
        p1 = v >= 0.8607
        add("P1 craft_task recovers", PASS if p1 else FAIL,
            {"craft_task": v,
             "delta": None if bhl.get("craft_task") is None
                      else round(v - bhl["craft_task"], 4)},
            ">= 0.8607 (baseline 0.8907 minus the enforced floor 0.030); "
            "expected 0.87-0.89; primacy read 0.8456",
            "craft run-to-run variation is exactly 0.0000 over 150 rows, so any "
            "movement here is mechanism")

    # --- P2 the occupancy change P1 rests on ---------------------------------
    occ = _kv_occupancy(Path(run_dir), "wm_craft_task", only_overflowing=True,
                        max_keys=cap)
    if not occ or occ.get("mean_kv") is None:
        add("P2 craft overflow rows settle below capacity", INCONCL, occ,
            "mean_kv <= 3.20 AND accuracy < 0.916",
            "absent: craft rows whose encoding_log writes more than "
            f"{cap} distinct keys")
    else:
        ok = occ["mean_kv"] <= 3.20 and (occ["mean_accuracy"] is None
                                         or occ["mean_accuracy"] < 0.916)
        add("P2 craft overflow rows settle below capacity",
            PASS if ok else FAIL, occ,
            "mean_kv <= 3.20 (sharp) AND overflow-row accuracy < 0.916 "
            "(primacy read 4.0 and 0.916); replay predicted 3.333",
            "this is the mechanism-confirmation row: a mean_kv of 4.0 means the "
            "batch rule never fired and the arm is measuring `primacy`")

    # --- P3 the confirmed U-shape must survive -------------------------------
    u = _primacy_u_shape(Path(run_dir), cap)
    if not u or u.get("both_ends_fraction") is None:
        add("P3 U-shaped survival survives", INCONCL, u, ">= 0.80",
            "absent: story-recall write order from turn_logs")
    else:
        ok = u["both_ends_fraction"] >= 0.80
        add("P3 U-shaped survival survives", PASS if ok else FAIL, u, ">= 0.80",
            "baseline 0.000, displacement 0.000, primacy 1.000, replay 1.000. If "
            "the store stopped overflowing on story recall this becomes inert -- "
            f"n_overflowing is reported ({u.get('n_overflowing')})")

    # --- P4 story stays recovered -------------------------------------------
    v = hl.get("semantic_story_recall")
    if v is None:
        add("P4 story stays recovered", INCONCL, None, "in [0.9173, 0.9773]",
            "absent")
    else:
        ok = 0.9173 <= v <= 0.9773
        add("P4 story stays recovered", PASS if ok else FAIL, v,
            "in [0.9173, 0.9773], expected ~0.9477 (baseline 0.9473, "
            "displacement 0.8964, primacy 0.9477)")

    # --- P5 n=3 response behaviour retained ---------------------------------
    l3 = lv.get(3) or {}
    obs5 = {"answered": l3.get("answered"), "keys_held": l3.get("keys_held"),
            "acc_over_answered": l3.get("acc_over_answered")}
    if any(x is None for x in obs5.values()):
        add("P5 n=3 response behaviour retained", INCONCL, obs5,
            "answered >= 13 AND keys_held >= 3.5 AND acc in [0.687, 0.787]",
            "absent: nback diagnostics at n=3")
    else:
        ok = (obs5["answered"] >= 13 and obs5["keys_held"] >= 3.5
              and 0.687 <= obs5["acc_over_answered"] <= 0.787)
        add("P5 n=3 response behaviour retained", PASS if ok else FAIL, obs5,
            "answered >= 13 AND keys_held >= 3.5 AND acc_over_answered in "
            "[0.687, 0.787] (primacy 14.0/4.0/0.76)")

    # --- P6 A1 digit span, no-change ----------------------------------------
    a1 = ax.get("A1") or {}
    obs6 = {"sub_span_leak": a1.get("sub_span_leak"),
            "best_span": a1.get("best_span"),
            "d_fwd": None if hl.get("digit_span_forward") is None
                     or bhl.get("digit_span_forward") is None
                     else round(hl["digit_span_forward"] - bhl["digit_span_forward"], 4),
            "d_rev": None if hl.get("digit_span_reverse") is None
                     or bhl.get("digit_span_reverse") is None
                     else round(hl["digit_span_reverse"] - bhl["digit_span_reverse"], 4)}
    if obs6["sub_span_leak"] is None or obs6["best_span"] is None:
        add("P6 A1 digit span unchanged", INCONCL, obs6,
            "|leak - 0.136| <= 0.02 AND best_span >= 18.0", "absent: axes.A1")
    else:
        ok = (abs(obs6["sub_span_leak"] - 0.136) <= 0.02
              and obs6["best_span"] >= 18.0
              and (obs6["d_fwd"] is None or abs(obs6["d_fwd"]) < 0.140)
              and (obs6["d_rev"] is None or abs(obs6["d_rev"]) < 0.059))
        add("P6 A1 digit span unchanged", PASS if ok else FAIL, obs6,
            "|leak - 0.136| <= 0.02 AND best_span >= 18.0 AND both deltas inside "
            "their floors (primacy 0.136 / 18.4)",
            "digit_span_forward's measured run-to-run variation is +0.0152, which "
            "is exactly what primacy reported there, so a move of that size is "
            "noise rather than mechanism")

    # --- P7 A3, on the ENFORCED quantity ------------------------------------
    a3 = ax.get("A3") or {}
    obs7 = {"precision_distance": a3.get("precision_distance"),
            "word_distance": a3.get("word_distance"),
            "bleu": a3.get("bleu")}
    if obs7["precision_distance"] is None or obs7["word_distance"] is None:
        add("P7 A3 within tolerance", INCONCL, obs7,
            "precision_distance <= 0.0422 AND word_distance <= 55.8",
            "absent: axes.A3")
    else:
        ok = (obs7["precision_distance"] <= 0.0422
              and obs7["word_distance"] <= 55.8)
        add("P7 A3 within tolerance", PASS if ok else FAIL, obs7,
            "precision_distance <= 0.0422 AND word_distance <= 55.8",
            "BLEU is deliberately NOT thresholded: it is a length proxy, reported "
            "with bleu_enforced=false")

    # --- P8 narrative: a pre-registered COST, band below the floor -----------
    v = hl.get("narrative_qa")
    if v is None:
        add("P8 narrative_qa (pre-registered cost)", INCONCL, None,
            "in [0.9222, 0.9322]", "absent")
    else:
        ok = 0.9222 <= v <= 0.9322
        d = None if bhl.get("narrative_qa") is None else round(v - bhl["narrative_qa"], 4)
        add("P8 narrative_qa (pre-registered cost)", PASS if ok else FAIL,
            {"narrative_qa": v, "delta": d},
            "in [0.9222, 0.9322] -- a band BELOW the 0.030 floor, registered as a "
            "cost rather than a recovery the candidate cannot mechanise",
            "narrative wants contiguity and P3 requires both ends; the two are "
            "opposed. Measured run-to-run noise here is 0.0065, so a delta of "
            "~0.03 is a real effect and not an instrument artifact",
            tags=("REPORTED",))

    # --- P9 anti-full_context ------------------------------------------------
    so = _kv_occupancy(Path(run_dir), "wm_semantic_story_recall", max_keys=cap)
    co = _kv_occupancy(Path(run_dir), "wm_craft_task", max_keys=cap)
    rows = _jsonl(Path(run_dir), "wm_semantic_story_recall") or []
    sims = [float((r.get("metrics") or {}).get("embeddingSimilarity"))
            for r in rows
            if (r.get("metrics") or {}).get("embeddingSimilarity") is not None]
    obs9 = {"story_kv": None if not so else so.get("mean_kv"),
            "story_similarity": round(sum(sims) / len(sims), 4) if sims else None,
            "craft_kv": None if not co else co.get("mean_kv")}
    if any(x is None for x in obs9.values()):
        add("P9 not drifting toward full_context", INCONCL, obs9,
            "story_kv <= 3.93 AND similarity <= 0.60 AND craft_kv <= 3.67",
            "absent: store occupancy or story similarity")
    else:
        ok = (obs9["story_kv"] <= 3.93 and obs9["story_similarity"] <= 0.60
              and obs9["craft_kv"] <= 3.67)
        add("P9 not drifting toward full_context", PASS if ok else FAIL, obs9,
            "story_kv <= 3.93 AND similarity <= 0.60 AND craft_kv <= 3.67",
            "an eviction rule that fixes a task by keeping MORE is drifting "
            "toward the most capable and least humanlike harness measured")

    # --- P10 variable_mapping untouched -------------------------------------
    a, b = hl.get("variable_mapping"), bhl.get("variable_mapping")
    if a is None or b is None:
        add("P10 variable_mapping untouched", INCONCL, None, "|delta| < 0.017",
            "absent")
    else:
        d = a - b
        add("P10 variable_mapping untouched", PASS if abs(d) < 0.017 else FAIL,
            {"variable_mapping": a, "delta": round(d, 4)},
            "|delta| < 0.017",
            "values hold exactly 1.00 element each on this task, so the store "
            "never overflows and the rule cannot bite -- a PASS is not a passed "
            "test", tags=("NEAR-INERT",))

    # --- P11 the mean, reported ----------------------------------------------
    m = rec.get("mean_humanlikeness_search")
    add("P11 8-task mean", INCONCL if m is None else
        (PASS if m >= 0.7945 else FAIL), m,
        ">= 0.7945 (primacy's value); only >= 0.8121 would be a credible gain "
        "over the baseline's 0.7861",
        "stated before the run so it cannot be claimed afterwards; the claims are "
        "P1, P2, P3 and P12", tags=("REPORTED",))

    # --- P12 craft's non-overflow partition must be untouched ----------------
    allc = _kv_occupancy(Path(run_dir), "wm_craft_task", max_keys=cap)
    rows = _jsonl(Path(run_dir), "wm_craft_task") or []
    non_over_acc = []
    for r in rows:
        written = set()
        for tc in ((r.get("encoding_log") or {}).get("tool_calls") or []):
            if tc.get("name") != "write_memory":
                continue
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(args, dict) and args.get("key"):
                written.add(str(args["key"]))
        if len(written) <= cap:
            a = (r.get("metrics") or {}).get("accuracy",
                                             (r.get("metrics") or {}).get("score"))
            if a is not None:
                non_over_acc.append(float(a))
    obs12 = {"n_non_overflowing": len(non_over_acc),
             "mean_accuracy": round(sum(non_over_acc) / len(non_over_acc), 4)
                              if non_over_acc else None,
             "all_rows_mean_kv": None if not allc else allc.get("mean_kv")}
    if obs12["mean_accuracy"] is None:
        add("P12 craft non-overflow partition untouched", INCONCL, obs12,
            "mean accuracy == 1.0000 on rows the bound never touched",
            "absent: craft rows with <= capacity distinct writes")
    else:
        ok = abs(obs12["mean_accuracy"] - 1.0) < 1e-9
        add("P12 craft non-overflow partition untouched",
            PASS if ok else FAIL, obs12,
            "mean accuracy == 1.0000 on the non-overflowing rows (every arm "
            "measured so far reads exactly 1.0000 there)",
            "confines the whole craft effect to the overflow group; a move here "
            "would mean the rule is firing where it should not")

    # --- C1 word_recognition, reported --------------------------------------
    a2 = ax.get("A2") or {}
    a, b = hl.get("word_recognition"), bhl.get("word_recognition")
    obsc = {"word_recognition": a,
            "delta": None if a is None or b is None else round(a - b, 4),
            "miss_rate": a2.get("miss_rate"), "fa_rate": a2.get("fa_rate"),
            "ratio": a2.get("ratio"), "distance": a2.get("distance"),
            "trials_attempted": a2.get("trials_attempted")}
    bad = (obsc["delta"] is not None and abs(obsc["delta"]) > 0.121
           and obsc["fa_rate"] is not None and obsc["fa_rate"] > 0.211)
    add("C1 word_recognition / A2 (reported)", FAIL if bad else INCONCL, obsc,
        "report only; disconfirming if |delta| > 0.121 AND fa_rate > 0.211",
        "the task is a confirmed leak, so no direction is derivable; the "
        "disconfirming pair would mean this rule reproduced displacement's "
        "liberal-bias artifact", tags=("REPORTED",))
    return out


def episodic_primacy_checks(run_dir: Path,
                            baseline: Path | None) -> list[dict[str, Any]]:
    """The composed arm. Coordinator-written; no new mechanism, so it scores only
    what neither component can answer alone."""
    out: list[dict[str, Any]] = []
    add = _adder(out)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    hl = rec.get("humanlikeness_by_task") or {}
    bhl = _hl(baseline)
    cap = _max_keys()
    lv = _nback_levels(Path(run_dir))

    # Sibling single-mechanism arms, for the additivity comparison. Without them
    # C2 cannot be scored -- comparing the composition to the BASELINE would not
    # distinguish additive from redundant.
    root = Path(run_dir).parent.parent
    sib = {name: (root / name / Path(run_dir).name) for name in
           ("episodic_reset_v2", "primacy_v2")}
    sib_hl = {k: _hl(v if v.exists() else None) for k, v in sib.items()}

    # --- C1 PRECONDITION: both mechanisms demonstrably active ----------------
    occ = _kv_occupancy(Path(run_dir), "wm_craft_task", only_overflowing=True,
                        max_keys=cap)
    share, n_vals = _nback_letter_share(Path(run_dir), 3)
    obs1 = {"craft_overflow_kv": None if not occ else occ.get("mean_kv"),
            "n3_letter_share": share}
    if obs1["craft_overflow_kv"] is None or share is None:
        add("C1 both mechanisms active -- PRECONDITION", INCONCL, obs1,
            "craft overflow kv <= 3.20 AND n=3 letter share >= 0.90",
            "absent: craft overflow rows or n=3 store values")
        c1 = False
    else:
        c1 = obs1["craft_overflow_kv"] <= 3.20 and share >= 0.90
        add("C1 both mechanisms active -- PRECONDITION", PASS if c1 else FAIL,
            obs1,
            "craft overflow kv <= 3.20 AND n=3 letter share >= 0.90 "
            "(baseline 4.0 / 0.0202)",
            "" if c1 else "primacy_v2's _batch_writes() fails SAFE to serial, so "
                          "a craft occupancy of 4.0 means it silently degraded to "
                          "`primacy` rather than erroring. C2-C4 are then VOID.")

    # --- C2 additivity, against the SIBLING arms -----------------------------
    obs2: dict[str, Any] = {}
    for t, sibling, tol in (("variable_mapping", "episodic_reset_v2", 0.05),
                            ("craft_task", "primacy_v2", 0.030)):
        mine, theirs = hl.get(t), sib_hl.get(sibling, {}).get(t)
        obs2[t] = {"composed": mine, sibling: theirs,
                   "delta": None if mine is None or theirs is None
                            else round(mine - theirs, 4)}
    legs = [v["delta"] for v in obs2.values() if v["delta"] is not None]
    if len(legs) < 2:
        add("C2 additive with the single-mechanism arms", INCONCL, obs2,
            "variable_mapping >= its arm - 0.05 AND craft >= its arm - 0.030",
            "absent: one or both sibling arms not present beside this run")
    else:
        ok = (obs2["variable_mapping"]["delta"] >= -0.05
              and obs2["craft_task"]["delta"] >= -0.030)
        add("C2 additive with the single-mechanism arms", PASS if ok else FAIL,
            obs2,
            "variable_mapping >= episodic_reset_v2 - 0.05 AND craft_task >= "
            "primacy_v2 - 0.030",
            "compared against the SIBLING arms, not the baseline: that is the "
            "only comparison that distinguishes additive from redundant. "
            "One-sided -- the interesting failure is the composition being WORSE "
            "than its parts.")

    # --- C3 the composition's own effect, reported ---------------------------
    obs3: dict[str, Any] = {}
    for t in ("nback", "variable_mapping"):
        obs3[t] = {"composed": hl.get(t),
                   "episodic_reset_v2": sib_hl.get("episodic_reset_v2", {}).get(t),
                   "primacy_v2": sib_hl.get("primacy_v2", {}).get(t),
                   "baseline": bhl.get(t)}
    obs3["nback_levels"] = {n: {"answered": d.get("answered"),
                                "silent": f"{d.get('n_no_answers')}/{d.get('n_rows')}"}
                            for n, d in sorted(lv.items())}
    add("C3 composition-specific effect on nback / variable_mapping", INCONCL,
        obs3, "reported, no direction derivable",
        "episodic_reset_v2 puts to_recall_text() on the prompt path every turn, so "
        "primacy_v2's eviction becomes VISIBLE to the agent on exactly the two "
        "tasks where it was previously unobservable. Any effect here is a property "
        "of the composition, not of either mechanism.", tags=("REPORTED",))

    # --- C4 neither component's known failure returns ------------------------
    l1 = lv.get(1) or {}
    ca = _kv_occupancy(Path(run_dir), "wm_craft_task", max_keys=cap)
    obs4 = {"n1_answered": l1.get("answered"),
            "n1_no_answers": l1.get("n_no_answers"),
            "craft_accuracy": None if not ca else ca.get("mean_accuracy")}
    if any(v is None for v in obs4.values()):
        add("C4 neither known failure returns", INCONCL, obs4,
            "n=1 answered >= 13.0 AND n_no_answers == 0 AND craft accuracy < 0.916",
            "absent: n=1 diagnostics or craft accuracy")
    else:
        ok = (obs4["n1_answered"] >= 13.0 and obs4["n1_no_answers"] == 0
              and obs4["craft_accuracy"] < 0.916)
        add("C4 neither known failure returns", PASS if ok else FAIL, obs4,
            "n=1 answered >= 13.0 AND n_no_answers == 0 AND craft accuracy < 0.916 "
            "(episodic_reset read 2.06 with 36 silent; primacy read 0.972)",
            "craft accuracy is thresholded BELOW primacy's 0.916 because "
            "humanlikeness decreases in accuracy above the baseline here -- the "
            "model is already better than humans, so a capability gain is a cost")

    # --- C5 no-change control -----------------------------------------------
    a1 = ax.get("A1") or {}
    obs5 = {"d_fwd": None if hl.get("digit_span_forward") is None
                     or bhl.get("digit_span_forward") is None
                     else round(hl["digit_span_forward"] - bhl["digit_span_forward"], 4),
            "d_rev": None if hl.get("digit_span_reverse") is None
                     or bhl.get("digit_span_reverse") is None
                     else round(hl["digit_span_reverse"] - bhl["digit_span_reverse"], 4),
            "best_span": a1.get("best_span")}
    if obs5["d_fwd"] is None or obs5["d_rev"] is None:
        add("C5 digit span untouched (control)", INCONCL, obs5,
            "|d_fwd| < 0.140 AND |d_rev| < 0.059 AND best_span >= 18.0", "absent")
    else:
        ok = (abs(obs5["d_fwd"]) < 0.140 and abs(obs5["d_rev"]) < 0.059
              and (obs5["best_span"] is None or obs5["best_span"] >= 18.0))
        add("C5 digit span untouched (control)", PASS if ok else FAIL, obs5,
            "|d_fwd| < 0.140 AND |d_rev| < 0.059 AND best_span >= 18.0",
            "NOT thresholded at bit-identity: generated content varies run to run "
            "on every task. digit_span_forward's own run-to-run variation is "
            "+0.0152.")

    if not c1:
        _void_rows(out, ("C2", "C3", "C4"),
                   "VOID via C1: one of the two mechanisms was not active, so the "
                   "composition is measuring a single mechanism and the "
                   "additivity question is unanswerable from this run.")
    return out


def _vm_answer_forms(run_dir: Path) -> dict[str, Any] | None:
    """Decompose variable_mapping replies into the two populations that matter.

    v2's P7 "verbosity" failure was 38 literal <tool_call> strings emitted as plain
    text on turns where tools were denied; excluding them its mean reply length was
    13.09 against the baseline's 13.10. So the ONLY informative split is
    tool-call-shaped versus everything else, and a mean over all rows is an
    arithmetic restatement of the unparsed count rather than a measure of verbosity.
    """
    rows = _jsonl(Path(run_dir), "wm_variable_mapping")
    if not rows:
        return None
    all_len, clean_len, unparsed, toolcall = [], [], 0, 0
    for r in rows:
        for sl in (r.get("step_logs") or []):
            raw = sl.get("answer_raw")
            if raw is None:
                continue
            raw = str(raw)
            all_len.append(len(raw))
            is_tc = "<tool_call>" in raw
            if is_tc:
                toolcall += 1
            else:
                clean_len.append(len(raw))
            if not sl.get("parsed"):
                unparsed += 1
    if not all_len:
        return None
    return {
        "n_answers": len(all_len),
        "unparsed": unparsed,
        "tool_call_in_text": toolcall,
        "mean_all": round(sum(all_len) / len(all_len), 2),
        "max_all": max(all_len),
        "mean_clean": round(sum(clean_len) / len(clean_len), 2) if clean_len else None,
        "max_clean": max(clean_len) if clean_len else None,
    }


def _nback_step_forms(run_dir: Path) -> dict[int, dict[str, Any]] | None:
    """Reply-form and tool-budget decomposition from n-back's per-turn step_log.

    The field is NEW (bench persists it as of the step_log commit) and the
    iteration-3 runs predate it, so absence means "this run is older", not "broken".
    Excludes the first step of each block, which carries no control-state block.
    """
    rows = _jsonl(Path(run_dir), "wm_nback")
    if not rows:
        return None
    if not any(r.get("step_log") for r in rows):
        return None
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        n = r.get("n_level")
        log = r.get("step_log") or []
        if n is None or not log:
            continue
        d = out.setdefault(int(n), {"turns": 0, "empty_text": 0, "tool_call_in_text": 0,
                                    "contaminated": 0, "cap_hit": 0, "zero_budget": 0,
                                    "tool_calls": 0, "memory_full": 0,
                                    "answer_plus_stray_call": 0})
        for st in log[1:]:
            d["turns"] += 1
            text = str(st.get("text") or "")
            if not text.strip():
                d["empty_text"] += 1
            has_tc_text = "<tool_call>" in text
            if has_tc_text:
                d["tool_call_in_text"] += 1
                # Contamination means the classification was parsed OUT OF the
                # tool-call JSON, not that a genuine answer happened to be followed
                # by a stray call. Those are opposite situations and an earlier
                # version of this check conflated them, voiding a whole run:
                # inspecting the replies showed the model reasoning, stating "the
                # response is 'different'", and only THEN emitting a call as text
                # because tools were denied that turn. That answer is real.
                #
                # So strip the tool-call block and re-parse. Still parses => the
                # answer stands on its own. Parses only with the block present =>
                # the parse is reading the JSON and `answered` really is inflated.
                try:
                    from bench.tasks.wm_nback import _parse_classification as _pc
                    if _pc(text) is not None:
                        stripped = re.sub(r"<tool_call>.*?</tool_call>", " ", text,
                                          flags=re.S)
                        if _pc(stripped) is None:
                            d["contaminated"] += 1
                        else:
                            d["answer_plus_stray_call"] += 1
                except Exception:  # noqa: BLE001
                    pass
            if st.get("tool_call_cap_hit"):
                d["cap_hit"] += 1
            if st.get("tool_call_budget_after") == 0:
                d["zero_budget"] += 1
            d["tool_calls"] += len(st.get("tool_calls") or [])
            for tc in (st.get("tool_calls") or []):
                if "memory is full" in str(tc.get("result") or ""):
                    d["memory_full"] += 1
    for n, d in out.items():
        t = max(1, d["turns"])
        d["empty_text_share"] = round(d["empty_text"] / t, 4)
        d["tool_call_in_text_share"] = round(d["tool_call_in_text"] / t, 4)
        d["cap_hit_share"] = round(d["cap_hit"] / t, 4)
        d["zero_budget_share"] = round(d["zero_budget"] / t, 4)
        d["tool_calls_per_turn"] = round(d["tool_calls"] / t, 2)
        # A share, not a raw count. The absolute bar this replaced was mis-scaled:
        # 8 contaminated turns out of 750 at n=1 would move `answered` from 13.98 to
        # about 13.82, which cannot flip a threshold set at 13.0, yet an absolute
        # "<= 2" bar voided the entire run over it.
        d["contaminated_share"] = round(d["contaminated"] / t, 4)
    return out


def episodic_reset_v3_checks(run_dir: Path,
                            baseline: Path | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    add = _adder(out)
    rec = _record(run_dir, baseline)
    ax = rec.get("axes") or {}
    hl = rec.get("humanlikeness_by_task") or {}
    bhl = _hl(baseline)
    lv = _nback_levels(Path(run_dir))
    a4 = ax.get("A4") or {}
    if not a4 and Path(run_dir).exists():
        try:
            a4 = IF.a4(Path(run_dir))
        except Exception:  # noqa: BLE001
            a4 = {}

    LEVELS = (1, 2, 3)
    THR = {"answered": (13.0, 9.5, 6.0), "acc_over_answered": (0.90, 0.60, 0.50),
           "keys_held": (0.8, 1.0, 2.0)}

    # --- P9 first: it can contaminate P1, so it has to be computed before it ----
    forms = _nback_step_forms(Path(run_dir))

    # --- P1 PRECONDITION: n-back mechanics at EVERY level ----------------------
    # Pitched at RESTORATION, not ambition: 9.5 sits just under episodic_reset's
    # 10.16 at n=2 and 6.0 just under the baseline's 6.82 at n=3. A precondition
    # pitched at the ambition is how episodic_reset lost a creditable +0.32.
    obs1: dict[str, Any] = {}
    legs_ok, missing = True, False
    for i, n in enumerate(LEVELS):
        d = lv.get(n) or {}
        row = {"answered": d.get("answered"),
               "acc": d.get("acc_over_answered"),
               "silent": d.get("n_no_answers"),
               "keys": d.get("keys_held")}
        obs1[f"n={n}"] = row
        if any(v is None for v in row.values()):
            missing = True
            continue
        if (row["answered"] < THR["answered"][i]
                or row["acc"] < THR["acc_over_answered"][i]
                or row["silent"] != 0
                or row["keys"] < THR["keys_held"][i]):
            legs_ok = False
    thr1 = ("per level n=1/2/3: answered >= 13.0/9.5/6.0, acc_over_answered >= "
            "0.90/0.60/0.50, n_no_answers == 0 at all three, keys_held >= "
            "0.8/1.0/2.0 (baseline 13.98/13.24/6.82 answered, 0 silent everywhere; "
            "v2 13.88/6.80/2.12 with 0/3/12 silent)")
    # Gate on the SHARE of turns, on the same 5% scale as P9's other legs, and only
    # where it could plausibly move a verdict. Contamination here means the
    # classification was parsed out of tool-call JSON rather than stated by the model.
    contaminated = bool(forms and any(
        (forms.get(n) or {}).get("contaminated_share", 0.0) > 0.05 for n in LEVELS))
    contam_note = ""
    if forms:
        contam_note = "; contamination share by level " + ", ".join(
            f"n={n}: {(forms.get(n) or {}).get('contaminated_share')}"
            for n in LEVELS if forms.get(n))
    if missing:
        add("P1 n-back at every level -- PRECONDITION", INCONCL, obs1, thr1,
            "absent: nback_levels diagnostics for at least one level")
        p1 = False
    elif contaminated:
        # A reply that both looks like a tool call and parses as a classification
        # would inflate `answered`, so the quantity is not trustworthy here.
        add("P1 n-back at every level -- PRECONDITION", INCONCL, obs1, thr1,
            "CONTAMINATED per P9: more than 2 turns at some level have a reply that "
            "both contains <tool_call> and parses as a classification, so `answered` "
            "is inflated and cannot bear a verdict")
        p1 = False
    else:
        p1 = legs_ok
        add("P1 n-back at every level -- PRECONDITION", PASS if p1 else FAIL,
            obs1, thr1,
            ("" if p1 else "A P1 FAILURE INVALIDATES EVERY ROW BELOW, as both "
                           "predecessors' preconditions did") + contam_note)

    # --- P1b the strong claim; cannot void anything --------------------------
    d2, d3 = lv.get(2) or {}, lv.get(3) or {}
    obs1b = {"n=2 answered": d2.get("answered"), "n=3 answered": d3.get("answered")}
    if obs1b["n=2 answered"] is None or obs1b["n=3 answered"] is None:
        add("P1b response obligation lands fully (strong claim)", INCONCL, obs1b,
            "n=2 >= 12.0 AND n=3 >= 10.0", "absent")
    else:
        ok = obs1b["n=2 answered"] >= 12.0 and obs1b["n=3 answered"] >= 10.0
        add("P1b response obligation lands fully (strong claim)",
            PASS if ok else FAIL, obs1b,
            "n=2 >= 12.0 AND n=3 >= 10.0 (baseline 13.24/6.82, v2 6.80/2.12)",
            "NOT a precondition: failing here while P1 passes means the obligation "
            "landed partially and the refusal loop is the remaining cap")

    # --- P2 nback clears its floor -------------------------------------------
    v = hl.get("nback")
    if v is None:
        add("P2 nback clears its floor", INCONCL, None, ">= 0.7309", "absent")
    else:
        ok = v >= 0.7309
        add("P2 nback clears its floor", PASS if ok else FAIL,
            {"nback": v, "delta": None if bhl.get("nback") is None
                                  else round(v - bhl["nback"], 4)},
            ">= 0.7309 (baseline 0.7909 minus the 0.060 floor; measured run-to-run "
            "noise 0.0023)",
            "the candidate stated in advance that if P1 passes, P1b fails and this "
            "lands near the floor, the reading is that the mechanism worked and the "
            "eviction cap dominated -- not that the diagnosis failed")

    # --- P3 the leak stays closed -------------------------------------------
    share, n_vals = _nback_letter_share(Path(run_dir), 3)
    obs3 = {"n3_letter_share": share, "n_values": n_vals,
            "keys_held": d3.get("keys_held")}
    if share is None:
        add("P3 history leak stays closed", INCONCL, obs3, ">= 0.90", "absent")
    else:
        add("P3 history leak stays closed", PASS if share >= 0.90 else FAIL, obs3,
            ">= 0.90 (baseline 0.0202, v2 0.9843, displacement 0.0201)",
            "keys_held reported not thresholded: P1 already covers the one "
            "direction that matters")

    # --- P4 variable_mapping holds up ---------------------------------------
    vmm = ax.get("variable_mapping_matched") or {}
    obs4 = {"raw": hl.get("variable_mapping"),
            "matched": vmm.get("humanlikeness_matched"),
            "n_unique_values": vmm.get("n_unique_values"),
            "share_at_ceiling": vmm.get("share_at_ceiling")}
    if obs4["raw"] is None or obs4["matched"] is None:
        add("P4 variable_mapping holds up", INCONCL, obs4,
            "raw >= 0.55 AND matched >= 0.60 AND n_unique_values >= 5", "absent")
    else:
        ok = (obs4["raw"] >= 0.55 and obs4["matched"] >= 0.60
              and (obs4["n_unique_values"] or 0) >= 5)
        add("P4 variable_mapping holds up", PASS if ok else FAIL, obs4,
            "raw >= 0.55 AND matched >= 0.60 AND n_unique_values >= 5 "
            "(baseline 0.3554/0.3587/2, v2 0.6854/0.7568/9)",
            "a FALL from v2's figures is predicted and accepted: recovering 38 of "
            "1500 answers raises the score on a task where the model already beats "
            "humans, and being better is a cost here")

    # --- P5 A4 still at scale and still interference-shaped ------------------
    obs5 = {"n_errors": a4.get("n_errors"),
            "rc_ratio_normalized": a4.get("rc_ratio_normalized"),
            "trustworthy": a4.get("trustworthy"),
            "rc_ratio": a4.get("rc_ratio"), "ceiling": a4.get("rc_ratio_ceiling")}
    if obs5["n_errors"] is None or obs5["rc_ratio_normalized"] is None:
        add("P5 A4 at scale and interference-shaped", INCONCL, obs5,
            "n_errors >= 150 AND normalized in [0.15, 0.95] AND trustworthy",
            "absent: A4")
    else:
        ok = (obs5["n_errors"] >= 150
              and 0.15 <= obs5["rc_ratio_normalized"] <= 0.95
              and bool(obs5["trustworthy"]))
        add("P5 A4 at scale and interference-shaped", PASS if ok else FAIL, obs5,
            "n_errors >= 150 AND rc_ratio_normalized in [0.15, 0.95] AND "
            "trustworthy (baseline 12/0.6661, v2 550/0.7362; humans 0.3728)",
            "this is the anti-restoration test: if the leak reopened, the errors "
            "vanish and n_errors collapses toward the baseline's 12")

    # --- P6 parse integrity, the surface v2 failed on ------------------------
    forms_vm = _vm_answer_forms(Path(run_dir))
    if forms_vm is None:
        add("P6 parse integrity on variable_mapping", INCONCL, None,
            "unparsed <= 5 AND tool_call_in_text <= 5", "absent: step_logs")
    else:
        ok = forms_vm["unparsed"] <= 5 and forms_vm["tool_call_in_text"] <= 5
        add("P6 parse integrity on variable_mapping", PASS if ok else FAIL,
            {k: forms_vm[k] for k in ("unparsed", "tool_call_in_text", "n_answers")},
            "unparsed <= 5 AND <tool_call>-in-text <= 5 (baseline 0/0, "
            "episodic_reset 1/1, v2 38/38)",
            "this is the candidate's central behavioural bet: that the model will "
            "ANSWER on a tools-denied turn when told to, instead of emitting the "
            "tool call as text. No offline test could settle it.")

    # --- P7 no verbosity -- staked against the proposer's own refutation -----
    if forms_vm is None:
        add("P7 no verbosity", INCONCL, None,
            "mean_all <= 14.5 AND mean_clean <= 13.5 AND max_clean <= 20", "absent")
    else:
        legs = {"mean_all": forms_vm["mean_all"], "mean_clean": forms_vm["mean_clean"],
                "max_clean": forms_vm["max_clean"]}
        ok = (legs["mean_all"] <= 14.5
              and (legs["mean_clean"] is None or legs["mean_clean"] <= 13.5)
              and (legs["max_clean"] is None or legs["max_clean"] <= 20))
        add("P7 no verbosity", PASS if ok else FAIL, legs,
            "mean over all rows <= 14.5 AND mean over non-<tool_call> rows <= 13.5 "
            "AND max non-<tool_call> <= 20 (baseline 13.10/13.10/14, "
            "v2 15.91/13.09/14)",
            "the second leg is staked AGAINST the prompt-competition hypothesis "
            "this candidate refuted: v3's block is ~202 characters longer than "
            "v2's, so if length drove the failure this must rise above 13.5. The "
            "first leg is an arithmetic restatement of P6 and is not independent "
            "evidence.", tags=("ENTAILED-FIRST-LEG",))

    # --- P8 no-change control, PER-TASK bands -------------------------------
    # A uniform band would make the checker report a mechanism failure on measured
    # run-to-run noise: digit_span_forward moved 0.0152 and narrative_qa 0.0160
    # between two runs of one harness.
    BANDS = {"digit_span_reverse": 0.008, "word_recognition": 0.008,
             "semantic_story_recall": 0.008, "craft_task": 0.008,
             "digit_span_forward": 0.016, "narrative_qa": 0.02}
    obs8, ok8 = {}, True
    for t, band in BANDS.items():
        a, b = hl.get(t), bhl.get(t)
        if a is None or b is None:
            obs8[t] = None
            continue
        d = round(a - b, 4)
        obs8[t] = d
        if abs(d) > band:
            ok8 = False
    add("P8 six encode-to-recall tasks unchanged", PASS if ok8 else FAIL, obs8,
        "per-task bands: 0.008 except digit_span_forward 0.016 and narrative_qa "
        "0.02, each set to that task's own MEASURED run-to-run spread",
        "v2 held four of these at exactly 0.0000 but moved word_recognition "
        "-0.0005 and semantic_story_recall +0.0071, so exact identity is not the "
        "prediction", tags=("ENTAILED-IF-STRUCTURAL",))

    # --- P9 reply-form and budget decomposition, from the NEW step_log -------
    if forms is None:
        add("P9 reply-form and tool-budget decomposition", INCONCL, None,
            "empty_text_share <= 0.05, tool_call_in_text_share <= 0.05, "
            "contamination <= 2, per level",
            "absent: wm_nback.jsonl has no `step_log` -- this run predates the "
            "bench change that persists it, which is a fact about the run's age, "
            "not a defect", tags=("REPORTED",))
    else:
        obs9 = {n: {k: (forms.get(n) or {}).get(k) for k in
                    ("empty_text_share", "tool_call_in_text_share", "contaminated",
                     "cap_hit_share", "zero_budget_share", "tool_calls_per_turn",
                     "memory_full")}
                for n in sorted(forms)}
        bad = any((forms.get(n) or {}).get("empty_text_share", 0) > 0.05
                  or (forms.get(n) or {}).get("tool_call_in_text_share", 0) > 0.05
                  or (forms.get(n) or {}).get("contaminated_share", 0.0) > 0.05
                  for n in forms)
        add("P9 reply-form and tool-budget decomposition",
            FAIL if bad else PASS, obs9,
            "per level: empty_text_share <= 0.05 AND tool_call_in_text_share <= "
            "0.05 AND contaminated_share <= 0.05; cap_hit_share, zero_budget_share, "
            "tool_calls_per_turn and memory_full REPORTED",
            "the reported legs are iteration 5's decision input: cap_hit_share is "
            "the refusal loop's fingerprint, measured offline at 14 of 17 n-back "
            "steps at n=2 and 16 of 18 at n=3")

    # --- P10 buffer-period compliance ---------------------------------------
    obs10 = {n: (lv.get(n) or {}).get("buffer_no_response_frac") for n in LEVELS}
    if obs10.get(2) is None or obs10.get(3) is None:
        add("P10 buffer-period compliance", INCONCL, obs10,
            "n=2 >= 0.30 AND n=3 >= 0.28", "absent: buffer_no_response_frac")
    else:
        ok = obs10[2] >= 0.30 and obs10[3] >= 0.28
        add("P10 buffer-period compliance", PASS if ok else FAIL, obs10,
            "n=2 >= 0.30 AND n=3 >= 0.28; n=1 reported only (baseline 0.92/0.39/"
            "0.3333, v2 0.86/0.17/0.1933)",
            "a rise here cannot be bought by answering more eagerly, which is why "
            "it is the anti-degeneracy row; n=1 is reported only because "
            "episodic_reset scored 1.0 there while failing catastrophically")

    if not p1:
        _void_rows(out, ("P1b", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"),
                   "VOID via P1: the n-back precondition failed or was contaminated, "
                   "which this candidate pre-registered as invalidating every row "
                   "below it.")
    return out


CHECKS: dict[str, Callable[[Path, Path | None], list[dict[str, Any]]]] = {
    "displacement": displacement_checks,
    "primacy": primacy_checks,
    "serial_recognition": serial_recognition_checks,
    # The open ablation arm is scored with the same function: P12 reads the open
    # arm's own dir, and pointing the checker at the ablation would otherwise have
    # no encoded predictions at all.
    "serial_recognition_open": serial_recognition_checks,
    "episodic_reset": episodic_reset_checks,
    "chunk_limit": chunk_limit_checks,
    "episodic_reset_v2": episodic_reset_v2_checks,
    "primacy_v2": primacy_v2_checks,
    "episodic_primacy": episodic_primacy_checks,
    "episodic_reset_v3": episodic_reset_v3_checks,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    fn = CHECKS.get(args.candidate)
    if fn is None:
        print(f"no pre-registered predictions encoded for '{args.candidate}'. "
              f"Known: {sorted(CHECKS)}")
        return 2

    rows = fn(Path(args.run_dir), Path(args.baseline) if args.baseline else None)

    if not rows:
        # Should not happen -- every function returns a row per prediction even
        # when the run is missing -- but an empty list must not become a traceback.
        print("no rows produced (the checker returned nothing at all)")
        return 1

    width = max(len(r["prediction"]) for r in rows)
    for r in rows:
        tags = r.get("tags") or []
        tag_s = ("  [" + " ".join(tags) + "]") if tags else ""
        print(f"{MARK.get(r['verdict'], '   ')} {r['verdict']:13s} "
              f"{r['prediction']:{width}s}  observed {_fmt(r['observed'])}  "
              f"(needs {r['threshold']}){tag_s}")
        if r["note"]:
            print(f"{'':17s} -> {r['note']}")

    counts = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
    print(f"\n{counts[PASS]} passed, {counts[FAIL]} failed, "
          f"{counts[INCONCL]} inconclusive, {counts[VOID]} VOID, "
          f"{counts[ADAPTED]} adapted (third outcome), of {len(rows)} rows")
    n_entailed = sum("ENTAILED" in (r.get("tags") or []) for r in rows)
    if n_entailed:
        print(f"{n_entailed} row(s) are ENTAILED -- arithmetic consequences of "
              f"another row, not independent evidence, and must not be cited as "
              f"support")
    other = [r["verdict"] for r in rows if r["verdict"] not in VERDICTS]
    if other:
        print(f"WARNING: unrecognised verdicts {sorted(set(other))}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
