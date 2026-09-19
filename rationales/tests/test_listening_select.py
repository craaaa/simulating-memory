"""The participant-level split.

The first two tests are regressions against the two rules in rationales/select.py that
would have broken this dataset silently: the stimulus-overlap drop (which would empty
the eval split, since all sixteen stimuli are in train) and the 1:1 subsample (which
would discard half the data).
"""
from __future__ import annotations

import pytest

from rationales.listening.data import RESPONSES_CSV, load_items
from rationales.listening.select import select, restore

from listening_fixtures import make_cohort

needs_real_data = pytest.mark.skipif(
    not RESPONSES_CSV.is_file(),
    reason=f"{RESPONSES_CSV} is a gitignored analysis artifact; run the ingest step",
)


@pytest.fixture
def cohort():
    return make_cohort(n_respondents=12, groups=4)


# ---------------------------------------------------------------------------
# The two rules that had to be replaced
# ---------------------------------------------------------------------------
def test_eval_split_is_not_empty(cohort):
    """Regression: every stimulus appears in train, so a stimulus-overlap drop would
    take the whole eval split with it."""
    sel = select(cohort, seed=42, eval_frac=0.25)
    assert sel.eval


def test_nothing_is_subsampled(cohort):
    sel = select(cohort, seed=42, eval_frac=0.25)
    assert len(sel.train) + len(sel.eval) == len(cohort)
    assert sel.report["subsampled"] is False


# ---------------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------------
def test_no_respondent_appears_on_both_sides(cohort):
    """The reason the split is by participant at all: an item's prompt carries that
    participant's answers to the passage's other questions."""
    sel = select(cohort, seed=42, eval_frac=0.25)
    train_ids = {it.respondent_id for it in sel.train}
    eval_ids = {it.respondent_id for it in sel.eval}
    assert train_ids and eval_ids
    assert not (train_ids & eval_ids)


def test_a_respondents_items_are_never_split(cohort):
    sel = select(cohort, seed=42, eval_frac=0.25)
    eval_ids = {it.respondent_id for it in sel.eval}
    assert all(it.respondent_id not in eval_ids for it in sel.train)


def test_every_counterbalancing_group_contributes_to_eval(cohort):
    sel = select(cohort, seed=42, eval_frac=0.25)
    per_group = sel.report["eval_respondents_by_group"]
    assert per_group
    assert all(n >= 1 for n in per_group.values())


def test_thin_eval_fraction_still_takes_one_respondent_per_group(cohort):
    """24 real groups at 15% is ~1.25 respondents each. Rounding down would drop a
    counterbalancing condition out of eval entirely."""
    sel = select(cohort, seed=42, eval_frac=0.01)
    assert all(n >= 1 for n in sel.report["eval_respondents_by_group"].values())


def test_split_is_deterministic_in_the_seed(cohort):
    a = select(cohort, seed=42, eval_frac=0.25)
    b = select(cohort, seed=42, eval_frac=0.25)
    c = select(cohort, seed=7, eval_frac=0.25)
    assert [i.item_id for i in a.train] == [i.item_id for i in b.train]
    assert [i.item_id for i in a.train] != [i.item_id for i in c.train]


def test_report_flags_degenerate_cells(cohort):
    """Every fixture cell is all-correct or all-wrong by construction, so all of them
    should be flagged; on the real data five are."""
    sel = select(cohort, seed=42, eval_frac=0.25)
    flagged = {d["cell"] for d in sel.report["degenerate_cells_train"]}
    assert flagged == set(sel.report["train_by_cell"])


def test_restore_rebuilds_the_identical_split(cohort):
    sel = select(cohort, seed=42, eval_frac=0.25)
    again = restore(cohort, sel.report)
    assert [i.item_id for i in again.train] == [i.item_id for i in sel.train]
    assert [i.item_id for i in again.eval] == [i.item_id for i in sel.eval]


def test_restore_rejects_an_unknown_item_id(cohort):
    sel = select(cohort, seed=42, eval_frac=0.25)
    report = dict(sel.report, eval_ids=list(sel.report["eval_ids"]) + ["nope:x:Q1"])
    with pytest.raises(ValueError, match="unknown item ids"):
        restore(cohort, report)


# ---------------------------------------------------------------------------
# On the real cohort
# ---------------------------------------------------------------------------
@needs_real_data
def test_real_split_keeps_every_topic_level_pair_in_eval():
    """If a (topic, level) pair fell out of eval, the by-level curve -- the headline
    figure -- would have a hole in it."""
    items, _ = load_items()
    sel = select(items, seed=42, eval_frac=0.15)
    assert sel.report["topic_level_pairs_missing_from_eval"] == []
    assert sel.report["topic_level_pairs_in_eval"] == 16


@needs_real_data
def test_real_split_holds_out_whole_participants():
    items, _ = load_items()
    sel = select(items, seed=42, eval_frac=0.15)
    assert sel.report["n_train_respondents"] + sel.report["n_eval_respondents"] == 201
    assert not (
        {it.respondent_id for it in sel.train} & {it.respondent_id for it in sel.eval}
    )
    assert len(sel.train) + len(sel.eval) == 4020
