"""Listening eval metrics, and the contract with evaluate.compare().

compare() reads four metric paths by literal name. A renamed block leaves every
headline delta in eval.json null while the tests still pass, so one test here asserts
those paths exist and resolve rather than trusting that they do.
"""
from __future__ import annotations

import pytest

from rationales.evaluate import compare
from rationales.listening.evaluate import (
    CATEGORIES,
    endorsement_profile,
    eval_row,
    summarize,
)

from listening_fixtures import make_item


def row(endorsed, pred, *, level="control", gold=(1, 2, 3), topic="lighthouses"):
    item = make_item(gold=gold, endorsed=endorsed, level=level, topic=topic)
    return eval_row(item, "because", list(pred), "raw", [])


@pytest.fixture
def rows():
    return [
        row((1, 2, 3), (1, 2, 3)),                      # human right, model matches
        row((1, 3), (1, 3), level="distractor"),        # human wrong, model matches
        row((1, 3), (1, 2, 3), level="distractor"),     # human wrong, model answers well
        row((4,), (5,), level="repeat_short"),          # both wrong, differently
    ]


# ---------------------------------------------------------------------------
# Row scoring
# ---------------------------------------------------------------------------
def test_human_match_is_set_equality_and_ignores_order():
    assert row((1, 3), (3, 1))["human_match"]
    assert not row((1, 3), (1,))["human_match"]


def test_human_match_and_ground_truth_are_scored_separately():
    r = row((1, 3), (1, 3))
    assert r["human_match"] and not r["ground_truth_correct"]
    r = row((1, 3), (1, 2, 3))
    assert not r["human_match"] and r["ground_truth_correct"]


def test_per_option_agreement_is_graded_where_exact_match_is_binary():
    """One option off should not score the same as everything off. Both are a plain
    exact-match miss; this is what separates them."""
    close = row((1, 2, 3), (1, 2))["per_option_agreement"]
    mid = row((1, 2, 3), (4,))["per_option_agreement"]
    far = row((1, 2, 3), (4, 5))["per_option_agreement"]
    assert far < mid < close < 1
    # Disagreeing on all five options is the floor.
    assert far == 0.0


def test_an_unparseable_completion_scores_as_an_empty_selection():
    item = make_item(gold=(1, 2, 3), endorsed=(1, 3))
    r = eval_row(item, None, [], "junk", ["no_answer_line"])
    assert r["pred_options"] == []
    assert not r["human_match"]
    assert r["n_endorsed_model"] == 0


# ---------------------------------------------------------------------------
# The endorsement profile
# ---------------------------------------------------------------------------
def test_profile_is_a_distribution_over_option_kinds():
    prof = endorsement_profile([row((1, 2, 3), (1, 2, 3))], "pred_options")
    assert set(prof) <= set(CATEGORIES)
    assert pytest.approx(sum(prof.values())) == 1.0


def test_profile_separates_true_options_from_interference_foils():
    """The listening analogue of digit span's error taxonomy: a model that answers
    correctly endorses only true options, while humans spend part of their budget on
    foils. That gap is what the fine-tune is supposed to close."""
    perfect = endorsement_profile([row((4,), (1, 2, 3))], "pred_options")
    human = endorsement_profile([row((4,), (1, 2, 3))], "human_options")
    assert perfect["false_interference:paraphrase"] == 0
    assert human["false_interference:paraphrase"] == 1.0


def test_endorsing_nothing_is_kept_in_the_distribution():
    """A human who selects nothing is a real behavior; dropping it would renormalize
    the rest and quietly overstate every other cell."""
    prof = endorsement_profile([row((1,), ())], "pred_options")
    assert prof["none"] == 1.0


def test_tv_distance_is_zero_when_model_and_human_endorse_alike(rows):
    same = [r for r in rows if r["human_match"]]
    metrics = summarize(same)
    assert metrics["overall"]["error_profile_tv_distance"] == pytest.approx(0.0)


def test_tv_distance_is_positive_when_they_diverge():
    metrics = summarize([row((4,), (1, 2, 3))])
    assert 0 < metrics["overall"]["error_profile_tv_distance"] <= 1


# ---------------------------------------------------------------------------
# summarize() blocks
# ---------------------------------------------------------------------------
def test_success_and_fail_blocks_split_on_the_humans_correctness(rows):
    m = summarize(rows)
    assert m["human_success_trials"]["n"] == 1
    assert m["human_fail_trials"]["n"] == 3
    assert m["overall"]["n"] == 4


def test_by_level_replaces_the_digit_span_by_length_curve(rows):
    m = summarize(rows)
    assert set(m["by_level"]) == {"control", "distractor", "repeat_short"}
    assert m["by_level_cell"]["distractor"]["n"] == 2


def test_empty_block_reports_n_zero_rather_than_dividing_by_zero():
    m = summarize([row((1, 2, 3), (1, 2, 3))])
    assert m["human_fail_trials"] == {"n": 0}


def test_summarize_emits_every_key_path_compare_reads(rows):
    """The contract. compare() looks these up by literal name; if summarize renamed a
    block, every delta would silently be None."""
    metrics = summarize(rows)
    deltas = compare({"metrics": metrics}, {"metrics": metrics})
    for key, value in deltas.items():
        if key == "reading":
            continue
        assert value is not None, f"compare() could not resolve {key}"


def test_compare_delta_signs_read_the_right_way(rows):
    """Success is human_match UP and ground-truth accuracy DOWN toward the human."""
    base = summarize([row((1, 3), (1, 2, 3))])     # answers correctly, unlike the human
    tuned = summarize([row((1, 3), (1, 3))])       # reproduces the human's error
    deltas = compare({"metrics": base}, {"metrics": tuned})
    assert deltas["human_match_rate_delta"] > 0
    assert deltas["ground_truth_accuracy_delta"] < 0
