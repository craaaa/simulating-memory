"""The listening loader, and the invariants of the real dataset it produces.

Tests that need responses.csv are marked and skipped when it is absent: it is a
gitignored artifact of the analysis ingest step, so a fresh checkout does not have it.
Everything that can be tested on synthetic items is.
"""
from __future__ import annotations

import pytest

from rationales.listening.data import (
    RESPONSES_CSV,
    SiblingAnswer,
    by_cell,
    load_items,
)

from listening_fixtures import make_item

needs_real_data = pytest.mark.skipif(
    not RESPONSES_CSV.is_file(),
    reason=f"{RESPONSES_CSV} is a gitignored analysis artifact; run the ingest step",
)


# ---------------------------------------------------------------------------
# Item semantics (synthetic)
# ---------------------------------------------------------------------------
def test_correct_is_set_equality_not_order():
    assert make_item(gold=(1, 2, 3), endorsed=(3, 1, 2)).correct
    assert not make_item(gold=(1, 2, 3), endorsed=(1, 2)).correct
    # A superset is wrong too -- endorsing everything must not read as correct.
    assert not make_item(gold=(1, 2, 3), endorsed=(1, 2, 3, 4)).correct


def test_item_id_identifies_participant_topic_and_question():
    a = make_item(respondent="r1", question_id="QV01")
    b = make_item(respondent="r1", question_id="QV02")
    c = make_item(respondent="r2", question_id="QV01")
    assert len({a.item_id, b.item_id, c.item_id}) == 3


def test_sibling_answer_renders_option_text_not_bare_numbers():
    s = SiblingAnswer(
        question_id="QV02",
        question="q",
        options={1: "From small boats", 2: "By drone"},
        endorsed=[1],
    )
    assert s.endorsed_text() == ["From small boats"]


def test_sibling_answer_tolerates_an_endorsement_with_no_option_text():
    s = SiblingAnswer(question_id="QV02", question="q", options={1: "a"}, endorsed=[1, 9])
    assert s.endorsed_text() == ["a"]


# ---------------------------------------------------------------------------
# The real join
# ---------------------------------------------------------------------------
@needs_real_data
def test_loads_every_human_item_once():
    items, report = load_items()
    assert report["n_items"] == len(items) == 4020
    assert report["n_respondents"] == 201
    assert report["n_topics"] == 4
    assert len({it.item_id for it in items}) == len(items)


@needs_real_data
def test_only_human_rows_are_read():
    """responses.csv carries ~106k model rows alongside the human ones. Pulling those
    in would be silent and would poison the corpus."""
    items, report = load_items()
    assert report["human_rows_read"] == 20100
    # 5 options per item, and every row consumed.
    assert report["human_rows_read"] == len(items) * 5


@needs_real_data
def test_attention_checks_are_not_items():
    items, report = load_items()
    assert not any(it.question_id.startswith("AT_") for it in items)
    assert report["skipped_unknown_questions"] == []


@needs_real_data
def test_every_item_has_exactly_five_options_and_a_labelled_set():
    items, _ = load_items()
    assert {it.n_options for it in items} == {5}
    assert all(set(it.option_labels) == set(it.options) for it in items)


@needs_real_data
def test_level_is_read_per_respondent_and_topic_not_assumed():
    """Level is counterbalanced. If it were assumed (say, from topic) every respondent
    would share one level per topic, which is exactly what this rules out."""
    items, report = load_items()
    assert report["n_level_conflicts"] == 0
    by_topic = {}
    for it in items:
        by_topic.setdefault(it.topic, set()).add(it.level)
    assert all(len(levels) == 4 for levels in by_topic.values())


@needs_real_data
def test_all_five_questions_of_a_topic_share_one_level_and_passage():
    items, _ = load_items()
    grouped = {}
    for it in items:
        grouped.setdefault((it.respondent_id, it.topic), []).append(it)
    for group in grouped.values():
        assert len({it.level for it in group}) == 1
        assert len({it.passage for it in group}) == 1


@needs_real_data
def test_siblings_are_the_other_four_questions_and_never_the_target():
    items, _ = load_items()
    for it in items[:200]:
        assert len(it.other_answers) == 4
        assert it.question_id not in {s.question_id for s in it.other_answers}


@needs_real_data
def test_gold_matches_the_question_bank_for_every_item():
    """load_items raises if responses.csv and the bank disagree, so reaching here at
    all is the assertion; this pins the count it checked."""
    items, _ = load_items()
    assert all(it.gold for it in items)
    assert sum(1 for it in items if it.correct) == 1877


@needs_real_data
def test_cell_table_covers_every_topic_level_question():
    items, _ = load_items()
    cells = by_cell(items)
    assert len(cells) == 4 * 4 * 5
    shares = [c["n_correct"] / c["n"] for c in cells.values()]
    # The point of reporting cells at all: the pooled 47% is not what any cell does.
    assert min(shares) < 0.15 and max(shares) > 0.85


@needs_real_data
def test_missing_responses_csv_names_the_command_that_rebuilds_it(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_items(responses_csv=tmp_path / "nope.csv")
    assert "make -C application/listening_qa/analysis" in str(exc.value)
