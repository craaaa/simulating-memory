"""Error taxonomy and held-out evaluation, including the degenerate cases.

Evaluation has to survive a model that emits nothing, emits an unparseable blob, or
emits the right digits for the wrong reason -- and it must keep human-match and
ground-truth accuracy strictly separate, since on fail trials they are opposites.
"""
from __future__ import annotations

import pytest

from rationales import errors as err
from rationales import evaluate as ev
from rationales import prompting as P


# --- taxonomy --------------------------------------------------------------
@pytest.mark.parametrize(
    "expected,response,label",
    [
        ([1, 2, 3], [1, 2, 3], "exact"),
        ([1, 2, 3, 4], [1, 2], "truncation"),
        ([1, 2, 3, 4], [1, 3, 4], "omission"),
        ([1, 2, 3], [1, 2, 9, 3], "insertion"),
        ([1, 2, 3], [1, 5, 3], "substitution"),
        ([1, 2, 3], [1, 3, 2], "transposition"),
        ([1, 2, 3], [], "empty"),
        ([1, 2, 3, 4], [9, 8], "mixed"),
    ],
)
def test_classify_error_labels(expected, response, label):
    assert err.classify_error(expected, response) == label


def test_exact_label_agrees_with_the_human_correct_flag():
    from rationales.data import load_all

    trials, _ = load_all(["forward", "reverse"])
    for t in trials:
        is_exact = err.classify_error(t.expected_digits, t.user_digits) == "exact"
        assert is_exact == t.correct


def test_profile_sums_to_one_and_tv_distance_bounds():
    a = err.profile([([1, 2], [1, 2]), ([1, 2], [2, 1])])
    assert sum(a.values()) == pytest.approx(1.0)
    assert err.total_variation(a, a) == 0
    b = err.profile([([1, 2], []), ([1, 2], [])])
    assert 0 < err.total_variation(a, b) <= 1


def test_prefix_and_features():
    f = err.error_features([1, 2, 3, 4], [1, 2, 9])
    assert f["prefix_correct"] == 2
    assert f["prefix_frac"] == pytest.approx(0.5)
    assert f["len_delta"] == -1
    assert f["same_multiset"] is False
    assert err.error_positions([1, 2, 3], [1, 9, 3]) == [1]


def test_features_on_empty_expected_do_not_divide_by_zero():
    assert err.error_features([], [])["prefix_frac"] == 0.0


# --- evaluation ------------------------------------------------------------
def _gen_const(text):
    return lambda prompt, n: [text] * n


def test_eval_separates_human_match_from_ground_truth(cfg, fwd_fail):
    """On a fail trial these are opposite: matching the human means being wrong."""
    matching_human = P.build_completion("tail lost", fwd_fail.user_digits)
    res = ev.run_eval(cfg, [fwd_fail], generate=_gen_const(matching_human), label="t")
    row = res["rows"][0]
    assert row["human_match"] is True
    assert row["ground_truth_correct"] is False
    m = res["metrics"]["human_fail_trials"]
    assert m["human_match_rate"] == 1.0
    assert m["ground_truth_accuracy"] == 0.0
    assert m["human_ground_truth_accuracy"] == 0.0


def test_eval_counts_a_perfect_recall_as_a_miss_on_fail_trials(cfg, fwd_fail):
    perfect = P.build_completion("recalled all", fwd_fail.expected_digits)
    res = ev.run_eval(cfg, [fwd_fail], generate=_gen_const(perfect), label="t")
    assert res["metrics"]["overall"]["human_match_rate"] == 0.0
    assert res["metrics"]["overall"]["ground_truth_accuracy"] == 1.0


def test_eval_uses_the_zero_shot_training_prompt(cfg, rev_fail, monkeypatch):
    monkeypatch.setattr(P, "load_fewshot", lambda d: "FEWSHOT-MARKER")
    seen = []

    def generate(prompt, n):
        seen.append(prompt)
        return [""]

    ev.run_eval(cfg, [rev_fail], generate=generate, label="t")
    assert "FEWSHOT-MARKER" not in seen[0]
    assert "actually responded" not in seen[0]


@pytest.mark.parametrize("raw", ["", "no block at all", "<reasoning>only reasoning</reasoning>"])
def test_eval_survives_unusable_output(cfg, fwd_fail, raw):
    res = ev.run_eval(cfg, [fwd_fail], generate=_gen_const(raw), label="t")
    row = res["rows"][0]
    assert row["pred_digits"] == []
    assert row["human_match"] is False
    assert row["error_type_model"] == "empty"
    assert res["metrics"]["overall"]["parse_failure_rate"] == 1.0


def test_eval_handles_a_generator_returning_nothing(cfg, fwd_fail):
    res = ev.run_eval(cfg, [fwd_fail], generate=lambda p, n: [], label="t")
    assert res["rows"][0]["raw"] == ""
    assert res["metrics"]["overall"]["n"] == 1


def test_summarize_on_empty_rows_is_safe():
    m = ev.summarize([])
    assert m["overall"] == {"n": 0}
    assert m["by_direction"] == {}
    assert m["by_length"] == {}


def test_by_length_curve_is_reported_per_direction(cfg, fwd_fail, rev_fail):
    def generate(prompt, n):
        digits = fwd_fail.user_digits if "[3, 9, 2]" in prompt else rev_fail.user_digits
        return [P.build_completion("r", digits)]

    res = ev.run_eval(cfg, [fwd_fail, rev_fail], generate=generate, label="t")
    by_len = res["metrics"]["by_length"]
    assert set(by_len) == {"forward:3", "reverse:4"}
    for cell in by_len.values():
        assert cell["human_match_rate"] == 1.0
        assert cell["model_exact_rate"] == 0.0   # matching the human means being wrong
        assert cell["human_exact_rate"] == 0.0


def test_success_and_fail_blocks_are_split(cfg, fwd_success, fwd_fail):
    def generate(prompt, n):
        return [P.build_completion("r", fwd_success.user_digits)]

    res = ev.run_eval(cfg, [fwd_success, fwd_fail], generate=generate, label="t")
    m = res["metrics"]
    assert m["human_success_trials"]["n"] == 1
    assert m["human_fail_trials"]["n"] == 1
    assert m["human_success_trials"]["human_match_rate"] == 1.0
    assert m["human_fail_trials"]["human_match_rate"] == 0.0


def test_compare_reports_deltas_in_the_right_direction(cfg, fwd_fail):
    base = ev.run_eval(
        cfg, [fwd_fail],
        generate=_gen_const(P.build_completion("r", fwd_fail.expected_digits)),
        label="base",
    )
    tuned = ev.run_eval(
        cfg, [fwd_fail],
        generate=_gen_const(P.build_completion("r", fwd_fail.user_digits)),
        label="tuned",
    )
    cmp = ev.compare(base, tuned)
    assert cmp["human_match_rate_delta"] == 1.0
    assert cmp["human_match_rate_delta_fail_trials"] == 1.0
    # ground truth accuracy must FALL toward the human level
    assert cmp["ground_truth_accuracy_delta"] == -1.0
    assert cmp["error_profile_tv_delta"] < 0


def test_compare_tolerates_missing_blocks():
    empty = {"metrics": ev.summarize([])}
    assert ev.compare(empty, empty)["human_match_rate_delta"] is None
