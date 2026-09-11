"""Loading the human corpus and building D (1:1, split).

These run against the real files in runs/human/, so they also serve as a check that
the committed data still has the shape the pipeline assumes.
"""
from __future__ import annotations

import pytest

from rationales.data import HUMAN_TASK_DIR, by_length, load_all, load_human_trials
from rationales.select import restore, select


# --- loader ----------------------------------------------------------------
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_loader_returns_int_lists_not_strings(direction):
    trials, _ = load_human_trials(direction)
    t = trials[0]
    assert isinstance(t.user_digits, list) and all(isinstance(d, int) for d in t.user_digits)
    assert isinstance(t.expected_digits, list)
    assert isinstance(t.digits, list)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_correct_flag_is_rederivable_from_the_strings(direction):
    trials, _ = load_human_trials(direction)
    for t in trials:
        assert t.correct == (t.user_digits == t.expected_digits)


def test_reverse_expected_is_the_reversed_presentation():
    trials, _ = load_human_trials("reverse")
    for t in trials:
        assert t.expected_digits == list(reversed(t.digits))


def test_forward_expected_is_the_presentation():
    trials, _ = load_human_trials("forward")
    for t in trials:
        assert t.expected_digits == t.digits


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_incomplete_runs_and_duplicate_participants_are_excluded(direction):
    trials, report = load_human_trials(direction)
    assert report["skipped_incomplete"], "expected at least one status!=completed run"
    # one run per participant after de-duping
    seen = {(t.participant_id, t.run_id) for t in trials}
    assert len({p for p, _ in seen}) == len(seen)
    assert report["participants"] == len({t.participant_id for t in trials})


def test_worktree_copy_is_not_double_counted():
    """.claude/worktrees/ holds a byte-identical copy; a recursive glob would double
    every count."""
    from rationales.data import HUMAN_ROOT

    trials, report = load_human_trials("forward")
    n_files = len(list((HUMAN_ROOT / HUMAN_TASK_DIR["forward"]).glob("run-*.json")))
    assert report["files_seen"] == n_files
    assert not any("worktrees" in t.run_id for t in trials)


def test_stimulus_text_matches_bench_wording():
    trials, _ = load_human_trials("forward")
    assert trials[0].stimulus_text.startswith("The digits are the following: [")


def test_trial_ids_are_unique():
    trials, _ = load_all(["forward", "reverse"])
    assert len({t.trial_id for t in trials}) == len(trials)


def test_by_length_is_dominated_by_span(caplog):
    trials, _ = load_all(["forward"])
    cells = by_length(trials)
    short = cells[("forward", 2)]
    long_keys = [k for k in cells if k[1] >= 7]
    assert short["n_correct"] / short["n"] > 0.9
    long_rate = sum(cells[k]["n_correct"] for k in long_keys) / sum(
        cells[k]["n"] for k in long_keys
    )
    assert long_rate < short["n_correct"] / short["n"]


# --- selection -------------------------------------------------------------
def test_selection_is_one_to_one_before_the_split():
    trials, _ = load_all(["forward", "reverse"])
    sel = select(trials, seed=42, eval_frac=0.15)
    r = sel.report
    assert r["pool_fail"] == r["pool_success"]
    assert r["pool_total"] == 2 * r["pool_fail"]
    assert r["pool_fail"] == sum(1 for t in trials if not t.correct)


def test_split_is_stratified_and_has_both_classes():
    trials, _ = load_all(["forward", "reverse"])
    sel = select(trials, seed=42, eval_frac=0.15)
    for split in (sel.train, sel.eval):
        for direction in ("forward", "reverse"):
            sub = [t for t in split if t.direction == direction]
            assert any(t.correct for t in sub)
            assert any(not t.correct for t in sub)


def test_train_and_eval_are_disjoint():
    trials, _ = load_all(["forward", "reverse"])
    sel = select(trials, seed=42, eval_frac=0.15)
    assert not ({t.trial_id for t in sel.train} & {t.trial_id for t in sel.eval})


def test_eval_stimuli_recurring_in_train_are_dropped():
    trials, _ = load_all(["forward", "reverse"])
    sel = select(trials, seed=42, eval_frac=0.15)
    train_seqs = {(t.direction, tuple(t.digits)) for t in sel.train}
    assert not any((t.direction, tuple(t.digits)) in train_seqs for t in sel.eval)
    assert sel.report["eval_dropped_for_stimulus_overlap"]


def test_selection_is_deterministic_per_seed():
    trials, _ = load_all(["forward", "reverse"])
    a = select(trials, seed=42, eval_frac=0.15)
    b = select(trials, seed=42, eval_frac=0.15)
    c = select(trials, seed=7, eval_frac=0.15)
    assert [t.trial_id for t in a.train] == [t.trial_id for t in b.train]
    assert [t.trial_id for t in a.train] != [t.trial_id for t in c.train]


def test_restore_reproduces_the_same_split():
    """Every STaR round must regenerate rationales for the SAME D."""
    trials, _ = load_all(["forward", "reverse"])
    sel = select(trials, seed=42, eval_frac=0.15)
    again = restore(trials, sel.report)
    assert [t.trial_id for t in again.train] == [t.trial_id for t in sel.train]
    assert [t.trial_id for t in again.eval] == [t.trial_id for t in sel.eval]


def test_restore_rejects_an_unknown_trial_id():
    trials, _ = load_all(["forward"])
    sel = select(trials, seed=42, eval_frac=0.15)
    bad = dict(sel.report, train_ids=list(sel.report["train_ids"]) + ["nope:1"])
    with pytest.raises(ValueError, match="unknown trial ids"):
        restore(trials, bad)


def test_selection_refuses_when_successes_are_too_few(fwd_fail, rev_fail, fwd_success):
    with pytest.raises(ValueError, match="cannot build 1:1"):
        select([fwd_fail, rev_fail, fwd_success], seed=1, eval_frac=0.2)


def test_report_records_the_length_shortcut_risk():
    """Global 1:1 concentrates fails at long spans -- the report must expose that so
    the per-length eval curve can be read against it."""
    trials, _ = load_all(["forward", "reverse"])
    sel = select(trials, seed=42, eval_frac=0.15)
    hist = sel.report["train_by_length"]
    short = [v for k, v in hist.items() if k.endswith(":2")]
    long = [v for k, v in hist.items() if k.split(":")[1].isdigit() and int(k.split(":")[1]) >= 8]
    short_fail = sum(c["n_fail"] for c in short) / max(sum(c["n"] for c in short), 1)
    long_fail = sum(c["n_fail"] for c in long) / max(sum(c["n"] for c in long), 1)
    assert long_fail > short_fail
