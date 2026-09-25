"""Exercise the matched variable_mapping scoring path.

`protocol_match.variable_mapping_scores` recomputes the model's score with the human
formula, because the two sides were computing different quantities that shared a
denominator: the human score counts correct answers out of 10, while the model score
is `relation_count` of the last consecutively correct question, which saturates at
10 by question 5 and therefore discards every later error.

Two things are checked, because the interesting failure mode is a recomputation that
runs cleanly while measuring the wrong thing:

  1. Behaviour on synthetic rows where the right answer is known by construction --
     including the case the defect is about, an error AFTER question 5, which the
     released formula scores as a perfect run.
  2. Agreement with the real runs' recorded distributions, so the function cannot
     drift away from what the WORKLOG reports.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "meta_harness"))
import protocol_match as PM  # noqa: E402

LETTERS = "ABCDEFGH"


def make_row(correct_flags, *, condition="C2"):
    """A C2 row whose answers are correct exactly where correct_flags[i] is True.

    Options are built so the correct city is always at a known index and a wrong
    answer picks a genuinely different city, matching the real rows' shape.
    """
    questions, answers = [], {}
    for i, ok in enumerate(correct_flags, start=1):
        correct_city = f"city{i}"
        options = [correct_city, "other1", "other2", "other3"]
        questions.append({
            "question_index": i,
            "name": f"name{i}",
            "correct_city": correct_city,
            "options": options,
            # min(2q, 10), as the generator produces
            "relation_count": min(2 * i, 10),
        })
        answers[str(i)] = LETTERS[0] if ok else LETTERS[1]
    return {
        "condition_id": condition,
        "questions": questions,
        "parsed_answers": answers,
        "assignments": [{"turn": t, "name": f"name{t}", "city": f"city{t}",
                         "statement": ""} for t in range(1, 21)],
    }


def score_rows(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        path = Path(fh.name)
    try:
        return PM.variable_mapping_scores(path)
    finally:
        path.unlink(missing_ok=True)


checks: dict[str, bool] = {}

# --- synthetic behaviour -----------------------------------------------------
all_right = [True] * 10
checks["all 10 correct -> 1.0"] = score_rows([make_row(all_right)]) == [1.0]

all_wrong = [False] * 10
checks["all 10 wrong -> 0.0"] = score_rows([make_row(all_wrong)]) == [0.0]

# The defect itself: one error at question 8. The released model formula banks
# relation_count 10 at question 5 and scores this a perfect 1.0; the human formula
# must score 9 of 10.
late_error = [True] * 7 + [False] + [True] * 2
got = score_rows([make_row(late_error)])
checks["one error at q8 -> 0.9 (released formula gives 1.0)"] = got == [0.9]

# An error at question 2 is visible to BOTH formulas, so this is the control that
# the function is not simply always disagreeing.
early_error = [True, False] + [True] * 8
checks["one error at q2 -> 0.9"] = score_rows([make_row(early_error)]) == [0.9]

# A malformed answer is not correct, matching the human formula, which counts
# correct answers and says nothing about how a wrong one was expressed.
row = make_row(all_right)
row["parsed_answers"]["4"] = "not a letter"
checks["malformed answer counts as incorrect"] = score_rows([row]) == [0.9]

row = make_row(all_right)
del row["parsed_answers"]["4"]
checks["missing answer counts as incorrect"] = score_rows([row]) == [0.9]

# Out-of-range letter must not index past the options list.
row = make_row(all_right)
row["parsed_answers"]["4"] = "H"
checks["out-of-range letter counts as incorrect"] = score_rows([row]) == [0.9]

# Non-C2 rows are excluded, as everywhere else in the analysis.
checks["non-C2 rows excluded"] = score_rows(
    [make_row(all_right, condition="C1")]) == []

# Score is capped at 1.0 even if a row somehow carries more than 10 questions.
checks["score capped at 1.0"] = score_rows([make_row([True] * 12)]) == [1.0]

# --- agreement with the real runs -------------------------------------------
M = "Qwen_Qwen3-30B-A3B-Instruct-2507"
EXPECTED = {
    "iter0/baseline": {1.0: 138, 0.9: 12},
    "iter1/displacement": {1.0: 125, 0.9: 18, 0.8: 4, 0.7: 3},
}
for rel, expected in EXPECTED.items():
    p = ROOT / f"meta_harness/runs/{rel}/{M}/tasks/wm_variable_mapping.jsonl"
    if not p.exists():
        print(f"  (skipped {rel}: run not present)")
        continue
    scores = PM.variable_mapping_scores(p)
    dist: dict[float, int] = {}
    for s in scores:
        dist[round(s, 2)] = dist.get(round(s, 2), 0) + 1
    checks[f"{rel} distribution matches the recorded one"] = dist == expected
    if dist != expected:
        print(f"  {rel}: got {dist}, expected {expected}")

print("=== matched variable_mapping scoring")
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{sum(checks.values())} of {len(checks)} passed")
sys.exit(0 if all(checks.values()) else 1)
