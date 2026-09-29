"""Do the six error-shape measures compute what they claim?

Each measure is exercised on hand-built input where the right answer is known BY
CONSTRUCTION -- a known transposition, a known stale-binding intrusion, a known
n+-1 lure, a known distractor concentration -- so a passing test means the
classification rule is the rule the documentation describes, not merely that the
code runs.

Then three drift guards, because every previous error-structure mistake in this
project was a silent change of denominator rather than a crash:

  1. Every human reference recomputed from the released records must equal the
     value cached in `logs/human_error_shape.json`. A reference can therefore
     never be recomputed inline and differently without this failing.
  2. A1's human summary must be unchanged by the `_administer` refactor that M2
     needed (sub_span_fail 0.0866, best_span 6.885).
  3. The model's per-question MCQ choice is not persisted, so M5 recovers it by
     re-parsing `recall_raw`. The recovered choices must reproduce the stored
     `metrics.correct` exactly, on every row of a real run.

Run:
    python meta_harness/test_error_shape.py            # full, loads MiniLM
    python meta_harness/test_error_shape.py --fast     # skip the M6 recomputation
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_harness import error_structure as ES      # noqa: E402
from meta_harness import interference as IF         # noqa: E402
from meta_harness import protocol_match as PM       # noqa: E402
from meta_harness import report_error_shape as RS   # noqa: E402

# A real pre-fix run, used only for the re-parse validation -- no model number
# from it is asserted as a result.
REPARSE_RUN = (ROOT / "meta_harness/runs/iter8repA/baseline"
               / "Qwen_Qwen3-30B-A3B-Instruct-2507")

FAILURES: list[str] = []


def check(label: str, got, want, tol: float | None = None) -> None:
    ok = (abs(float(got) - float(want)) <= tol) if tol is not None else (got == want)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}"
          + (f" (tol {tol})" if tol else ""))
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


# --------------------------------------------------------------------- M1
def test_m1_intrusion_classes() -> None:
    """Four hand-built variable-mapping errors, one of each class.

    Timeline of assignments (turn: name -> city)
        1: Ann  -> Paris        2: Bob  -> Lima
        3: Ann  -> Oslo         4: Bob  -> Kiev
    At a question asked after turn 4, asking Ann's city (Oslo):
        picking Paris -> own_stale        (Ann's superseded city)
        picking Kiev  -> other_current    (Bob's city right now)
        picking Lima  -> other_stale      (Bob's superseded city)
        picking Cairo -> novel            (never assigned to anyone)
    """
    print("M1 variable_mapping intrusion type")
    before = {"Ann": ["Paris", "Oslo"], "Bob": ["Lima", "Kiev"]}
    for sel, want4, want3 in (("Paris", "own_stale", "stale_same_name"),
                              ("Kiev", "other_current", "intrusion_other_name"),
                              ("Lima", "other_stale", "intrusion_other_name"),
                              ("Cairo", "novel", "novel_guess")):
        check(f"kind4({sel})", IF._classify4(sel, "Oslo", "Ann", before), want4)
        check(f"kind3({sel})", IF._classify(sel, "Oslo", "Ann", before), want3)

    # And the pooled profile, where the denominator is the error count.
    trials = [{"correct": False, "kind": "stale_same_name", "kind4": "own_stale",
               "participant": "p1"}] * 10
    trials += [{"correct": False, "kind": "novel_guess", "kind4": "novel",
                "participant": "p2"}] * 30
    trials += [{"correct": True, "kind": "novel_guess", "kind4": "novel",
                "participant": "p3"}] * 60
    prof = IF.intrusion_profile(trials, min_errors=40)
    check("pooled n_errors", prof["n_errors"], 40)
    check("pooled stale share", prof["shares"]["stale_same_name"], 0.25, 1e-9)
    check("pooled novel share", prof["shares"]["novel_guess"], 0.75, 1e-9)
    # Below the minimum it must say so rather than print a number.
    thin = IF.intrusion_profile(trials, min_errors=41)
    check("below minimum -> shares None", thin["shares"], None)
    check("below minimum -> note", thin["note"].startswith("insufficient (n=40)"), True)


# --------------------------------------------------------------------- M2
def test_m2_span_typology() -> None:
    """One constructed error of every class, forward and reverse."""
    print("M2 digit-span error typology")
    cases = [
        # (gold, response, presented, reverse, expected class)
        ((1, 2, 3, 4), (1, 3, 2, 4), (1, 2, 3, 4), False, "transposition"),
        ((1, 2, 3, 4), (1, 2), (1, 2, 3, 4), False, "truncation"),
        ((1, 2, 3, 4), (1, 3, 4), (1, 2, 3, 4), False, "omission"),
        ((1, 2, 3, 4), (1, 2, 9, 4), (1, 2, 3, 4), False, "substitution"),
        ((1, 2, 3, 4), (1, 2, 3, 4, 5), (1, 2, 3, 4), False, "other"),
        # reverse span: gold is the reversal of presented, and reporting the
        # PRESENTED order is its own class.
        ((4, 3, 2, 1), (1, 2, 3, 4), (1, 2, 3, 4), True, "reversal"),
        # a palindrome cannot express a reversal; caller passes reverse=False for
        # those, and then it is simply correct-order (here a transposition case).
        ((1, 2, 2, 1), (1, 2, 1, 2), (1, 2, 2, 1), False, "transposition"),
    ]
    for gold, resp, pres, rev, want in cases:
        check(f"{gold}->{resp}{' rev' if rev else ''}",
              PM.classify_span_error(gold, resp, pres, reverse=rev), want)

    # The palindrome guard, at the level that matters: a palindromic sequence must
    # not be counted in the reversal denominator.
    parts = [[
        {"span": 2, "presented": (3, 3), "gold": (3, 3), "response": (3, 4),
         "correct": False},                                   # palindrome, an error
        {"span": 4, "presented": (1, 2, 3, 4), "gold": (4, 3, 2, 1),
         "response": (1, 2, 3, 4), "correct": False},          # a real reversal
    ]]
    typ = PM.span_typology(parts, reverse=True, min_errors=2)
    check("reversal counted once", typ["counts"]["reversal"], 1)
    check("n_errors", typ["n_errors"], 2)
    check("n_errors_nonpalindromic", typ["n_errors_nonpalindromic"], 1)
    check("reversal share over non-palindromic",
          typ["reversal_share_nonpalindromic"], 1.0, 1e-9)


def test_m2_serial_position() -> None:
    """A constructed recall where the primacy/recency split is known exactly."""
    print("M2 digit-span serial position")
    # A 6-digit target; positions 1,2 right, 3,4 wrong, 5,6 right.
    # Thirds of length 6 are {1,2} {3,4} {5,6}, so the bin means must be 1, 0, 1.
    parts = [[{"span": 6, "presented": (1, 2, 3, 4, 5, 6), "gold": (1, 2, 3, 4, 5, 6),
               "response": (1, 2, 9, 9, 5, 6), "correct": False}]]
    sp = PM.serial_position(parts)
    check("bin means", sp["bin_means"], [1.0, 0.0, 1.0])
    check("absolute curve pos 3", sp["absolute_curve"]["3"], 0.0, 1e-9)
    # A shorter response must count the missing tail as not recalled, not crash.
    parts = [[{"span": 3, "presented": (1, 2, 3), "gold": (1, 2, 3),
               "response": (1,), "correct": False}]]
    sp = PM.serial_position(parts)
    check("truncated response bin means", sp["bin_means"], [1.0, 0.0, 0.0])


# --------------------------------------------------------------------- M3
def test_m3_nback() -> None:
    """A hand-built 2-back stream with one known target, one known n+-1 lure."""
    print("M3 n-back miss / false alarm / lure")
    # index:      0    1    2    3    4    5
    # letters:    A    B    A    C    C    D
    # 2-back defined trials start at index 2:
    #   i=2 'A' vs i=0 'A'  -> TARGET
    #   i=3 'C' vs i=1 'B'  -> not a target; lag-1 is 'A', lag-3 is 'A' -> no lure
    #   i=4 'C' vs i=2 'A'  -> not a target; lag-1 (i=3) is 'C'         -> LURE
    #   i=5 'D' vs i=3 'C'  -> not a target, no match at any lag        -> no lure
    letters = ["A", "B", "A", "C", "C", "D"]
    resp = [None, None, "target", "nontarget", "target", None]
    recs = ES._nback_trial_records(letters, resp, 2, "p1")
    check("n defined trials", len(recs), 4)
    check("positions", [r["position"] for r in recs], [1, 2, 3, 4])
    check("targets", [r["target"] for r in recs], [True, False, False, False])
    check("lures", [r["lure"] for r in recs], [False, False, True, False])

    prof = ES.nback_error_profile(recs, min_lure=1)
    o = prof["overall"]
    check("miss_rate (target answered 'same' -> 0 misses)", o["miss_rate"], 0.0, 1e-9)
    # Two answered non-targets: i=3 'nontarget', i=4 'target' -> 1 false alarm of 2.
    check("fa_rate", o["fa_rate"], 0.5, 1e-9)
    check("unanswered_rate (1 of 4 defined trials)", o["unanswered_rate"], 0.25, 1e-9)
    check("lure_fa_rate (the one lure was called 'same')", o["lure_fa_rate"], 1.0, 1e-9)
    check("nonlure_fa_rate", o["nonlure_fa_rate"], 0.0, 1e-9)
    # Silence must never be booked as a miss.
    recs2 = ES._nback_trial_records(letters, [None] * 6, 2, "p1")
    p2 = ES.nback_error_profile(recs2)["overall"]
    check("all-silent miss_rate is undefined, not 1.0", p2["miss_rate"], None)
    check("all-silent unanswered_rate", p2["unanswered_rate"], 1.0, 1e-9)


# --------------------------------------------------------------------- M4
def test_m4_lag() -> None:
    """A constructed session where every lag is known."""
    print("M4 word-recognition lag curve")
    # trial:   1  2  3  4  5
    # word:    a  b  a  c  b     -> 'a' repeats at lag 2 (bin 1-2)
    #                               'b' repeats at lag 3 (bin 3-5)
    words = ["a", "b", "a", "c", "b"]
    correct = [True, True, True, True, False]
    bins = ES._lag_one_session(words, correct)
    check("bin 1-2 accuracy (the lag-2 repeat was right)", bins[0], 1.0, 1e-9)
    check("bin 3-5 accuracy (the lag-3 repeat was wrong)", bins[1], 0.0, 1e-9)
    check("bin 6-10 has no trials", bins[2], None)
    # Unequal exposure must be averaged per participant, not pooled: a session that
    # never reaches a bin must not lower that bin's n.
    summ = ES.wr_lag_summary([bins, [None, 0.0, None, None, None]],
                             min_participants=1)
    b = {x["lag"]: x for x in summ["bins"]}
    check("bin 1-2 n_participants", b["1-2"]["n_participants"], 1)
    check("bin 3-5 n_participants", b["3-5"]["n_participants"], 2)
    check("bin 3-5 mean of 0.0 and 0.0", b["3-5"]["accuracy"], 0.0, 1e-9)
    check("bin 6-10 below minimum -> None", b["6-10"]["accuracy"], None)
    check("broken-task label present", summ["task_is_broken"], True)


# --------------------------------------------------------------------- M5
def test_m5_distractor() -> None:
    """Known distractor concentration and a known cross-side agreement."""
    print("M5 distractor choice")
    # One question, 4 errors all on distractor 'B': every ordered pair agrees.
    counts = {f"q{i}": collections.Counter({"B": 4}) for i in range(10)}
    prof = ES.distractor_profile(counts)
    check("perfect concentration", prof["within_side_agreement"], 1.0, 1e-9)
    check("ordered pairs (10 questions x 4*3)", prof["n_ordered_pairs"], 120)
    # Two errors split across two distractors: no ordered pair agrees.
    counts = {f"q{i}": collections.Counter({"B": 1, "C": 1}) for i in range(20)}
    prof = ES.distractor_profile(counts)
    check("split concentration", prof["within_side_agreement"], 0.0, 1e-9)
    # Exactly at chance: 3 errors, one per distractor, over many questions.
    counts = {f"q{i}": collections.Counter({"B": 1, "C": 1, "D": 1}) for i in range(20)}
    prof = ES.distractor_profile(counts)
    check("uniform over 3 distractors -> 0.0 (no repeat within a question)",
          prof["within_side_agreement"], 0.0, 1e-9)
    # Cross-side: the two sides pick the same distractor on every shared question.
    h = {f"q{i}": collections.Counter({"B": 2}) for i in range(12)}
    m = {f"q{i}": collections.Counter({"B": 3}) for i in range(12)}
    cross = ES.cross_side_distractor_agreement(h, m)
    check("cross-side perfect agreement", cross["cross_side_agreement"], 1.0, 1e-9)
    m = {f"q{i}": collections.Counter({"C": 3}) for i in range(12)}
    cross = ES.cross_side_distractor_agreement(h, m)
    check("cross-side no agreement", cross["cross_side_agreement"], 0.0, 1e-9)
    # Below the minimum, no number.
    cross = ES.cross_side_distractor_agreement(
        {"q0": collections.Counter({"B": 1})}, {"q0": collections.Counter({"B": 1})})
    check("cross-side below minimum -> None", cross["cross_side_agreement"], None)


def test_m5b_error_index() -> None:
    print("M5b error-index profile (binary-choice tasks)")
    # Participant A: q1 wrong, q2 right. Participant B: both right.
    sess = [[1.0, 0.0], [0.0, 0.0]]
    summ = ES.error_index_summary(sess)
    b = {x["index"]: x for x in summ["by_index"]}
    check("q1 error rate", b[1]["error_rate"], 0.5, 1e-9)
    check("q2 error rate", b[2]["error_rate"], 0.0, 1e-9)
    check("labelled not-a-distractor-measure", summ["not_a_distractor_measure"], True)


# --------------------------------------------------------------------- M6
def test_m6_gist() -> None:
    print("M6 story-recall gist similarity")
    recs = [{"participant": "a", "story": "S", "stored": 0.6, "recomputed": 0.5},
            {"participant": "b", "story": "S", "stored": 0.8, "recomputed": 0.7},
            {"participant": "c", "story": "T", "stored": None, "recomputed": None}]
    s = ES.gist_summary(recs, "recomputed")
    check("n excludes the null", s["n"], 2)
    check("mean", s["mean"], 0.6, 1e-9)
    check("by_story keeps its own n", s["by_story"]["S"]["n"], 2)
    check("stored column is separate", ES.gist_summary(recs, "stored")["mean"],
          0.7, 1e-9)


# ------------------------------------------------------- drift guard 1: the cache
def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(obj, list):
        out[prefix] = obj
    else:
        out[prefix] = obj


def test_cached_human_reference_matches_recomputation(fast: bool) -> None:
    """A cached reference that no longer matches a fresh recomputation is a silent
    change of filter, which is exactly how A3 and A4 went wrong the first time."""
    print("drift guard: cached human references == recomputed")
    cached = RS.load_cache()
    fresh = RS.human_reference()
    a, b = {}, {}
    _flatten("", cached["measures"], a)
    _flatten("", fresh["measures"], b)
    skip = ".M6_story_recall_gist." if fast else "\0"
    keys = sorted(set(a) | set(b))
    bad = []
    for k in keys:
        if skip in k:
            continue
        if a.get(k) != b.get(k):
            bad.append((k, a.get(k), b.get(k)))
    check(f"all {len(keys)} cached reference fields match"
          + (" (M6 skipped in --fast)" if fast else ""), bad[:5], [])


# --------------------------------------------- drift guard 2: A1 is untouched
def test_a1_unchanged_by_refactor() -> None:
    """M2 refactored `_staircase` onto a shared `_administer`. A1's human numbers
    must be bit-identical, or the typology and the guard are no longer looking at
    the same trials."""
    print("drift guard: A1 human summary unchanged by the _administer refactor")
    summ = PM.summarize("HUMANS", PM.human_participants())
    check("n participants", summ["n"], 52)
    check("best_span", summ["best_span"], 6.884615384615385, 1e-12)
    check("n_administered", summ["n_administered"], 13.76923076923077, 1e-12)
    check("sub_span_fail (score_candidate.HUMAN_A1_LEAK = 0.087)",
          summ["sub_span_fail"], 0.0866414835164835, 1e-12)
    check("supra_span_hit is 0 by construction", summ["supra_span_hit"], 0.0, 1e-12)


# -------------------------------- drift guard 3: the MCQ re-parse is faithful
def test_mcq_reparse_reproduces_stored_metrics() -> None:
    """The model's chosen option is not persisted, so M5 recovers it from
    `recall_raw`. If the recovery is not exact, every distractor number is wrong."""
    print("drift guard: re-parsed MCQ choices reproduce metrics.correct")
    if not REPARSE_RUN.exists():
        print(f"  skip (missing {REPARSE_RUN})")
        return
    for task, want_rows in (("narrative_qa", 50), ("craft_task", 150)):
        parse = ES._mcq_parser(task)
        path = REPARSE_RUN / f"tasks/wm_{task}.jsonl"
        rows = agree = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            ans = parse(r.get("recall_raw") or "")
            got = sum(1 for i, q in enumerate(r.get("questions") or [], start=1)
                      if ans.get(i) == str(q.get("answer") or "").strip().upper())
            agree += (got == int((r.get("metrics") or {})["correct"]))
        check(f"{task}: rows", rows, want_rows)
        check(f"{task}: re-parse reproduces metrics.correct on every row",
              agree, rows)


def main() -> int:
    fast = "--fast" in sys.argv
    test_m1_intrusion_classes()
    test_m2_span_typology()
    test_m2_serial_position()
    test_m3_nback()
    test_m4_lag()
    test_m5_distractor()
    test_m5b_error_index()
    test_m6_gist()
    test_a1_unchanged_by_refactor()
    test_mcq_reparse_reproduces_stored_metrics()
    test_cached_human_reference_matches_recomputation(fast)

    print()
    if FAILURES:
        for f in FAILURES:
            print(f"FAIL: {f}")
        return 1
    print("all error-shape tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
