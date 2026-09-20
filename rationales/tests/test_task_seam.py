"""The task seam itself.

Two jobs. First, fidelity: the digit-span adapter must be a pure pass-through, so that
pointing STaR at a second task cannot change what the committed digit-span runs would
do. Second, completeness: a task that is missing a method fails at the point the loop
reaches it, which on a live run could be an hour and several dollars in.
"""
from __future__ import annotations

import pytest

from rationales import prompting as dsp
from rationales.config import StarConfig
from rationales.data import load_all
from rationales.sample import GENERATION, SampleRow, training_pairs
from rationales.select import select as ds_select
from rationales.task import RationaleTask, default_task, resolve, task_names
from rationales.tasks.digit_span import TASK as DIGIT_SPAN
from rationales.tasks.listening_qa import TASK as LISTENING

from listening_fixtures import make_item

ALL_TASKS = [DIGIT_SPAN, LISTENING]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_default_task_is_digit_span():
    """Every call site that predates the seam relies on this."""
    assert default_task() is DIGIT_SPAN
    assert StarConfig().task == "digit_span"


def test_registry_resolves_both_tasks_by_name():
    assert resolve("digit_span").name == "digit_span"
    assert resolve("listening_qa").name == "listening_qa"
    assert task_names() == ["digit_span", "listening_qa"]


def test_unknown_task_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="listening_qa"):
        resolve("no_such_task")


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.name)
def test_task_implements_the_whole_protocol(task):
    assert isinstance(task, RationaleTask)
    assert isinstance(getattr(task, "defaults", None), dict)
    for method in (
        "for_config",
        "load", "select", "restore", "item_id", "check_ready",
        "build_sample_prompt", "build_rationalize_prompt", "build_completion",
        "prompt_additions", "parse", "accepts", "answer_from_row", "hint_leak",
        "human_answer", "gold_answer", "probe_items",
        "item_fields", "answer_fields", "pair_fields", "cell_key", "pool_cell_key",
        "eval_row", "summarize", "plot_round",
    ):
        assert callable(getattr(task, method, None)), f"{task.name} lacks {method}"


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.name)
def test_item_fields_carry_human_correct(task):
    """dry_run, filter and both cell keys read this column by name."""
    item = make_item() if task is LISTENING else _a_digit_trial()
    assert "human_correct" in task.item_fields(item)


def _a_digit_trial():
    trials, _ = load_all(["forward"])
    return trials[0]


# ---------------------------------------------------------------------------
# Digit-span fidelity: the adapter must not have changed anything
# ---------------------------------------------------------------------------
@pytest.fixture
def trial(fwd_fail):
    return fwd_fail


def test_adapter_prompts_are_byte_identical_to_the_originals(trial):
    for fewshot in (True, False):
        assert DIGIT_SPAN.build_sample_prompt(
            trial, fewshot=fewshot
        ) == dsp.build_sample_prompt(trial, fewshot=fewshot)
        assert DIGIT_SPAN.build_rationalize_prompt(
            trial, fewshot=fewshot
        ) == dsp.build_rationalize_prompt(trial, fewshot=fewshot)


def test_adapter_parse_and_completion_match_the_originals(trial):
    raw = "<reasoning>\nthey chunked it.\n</reasoning>\npress <<3>>.\npress <<2>>."
    assert DIGIT_SPAN.parse(raw) == dsp.parse_rationale(raw)
    assert DIGIT_SPAN.build_completion("r", [3, 2]) == dsp.build_completion("r", [3, 2])


def test_adapter_accepts_reproduces_the_original_filter(trial):
    """The line this replaced was `digits == trial.user_digits`, comparing against the
    human's response rather than the correct answer."""
    assert DIGIT_SPAN.accepts(trial, trial.user_digits)
    assert not DIGIT_SPAN.accepts(trial, trial.expected_digits)
    assert not DIGIT_SPAN.accepts(trial, [])


def test_adapter_select_matches_the_original_selector():
    trials, _ = load_all(["forward", "reverse"])
    via_task = DIGIT_SPAN.select(trials, seed=42, eval_frac=0.15)
    direct = ds_select(trials, seed=42, eval_frac=0.15)
    assert [t.trial_id for t in via_task.train] == [t.trial_id for t in direct.train]
    assert [t.trial_id for t in via_task.eval] == [t.trial_id for t in direct.eval]


def test_sample_row_json_keeps_the_digit_span_columns(trial):
    row = SampleRow(trial, GENERATION, 0, "p", "raw", "r", [1, 2], True, []).to_json()
    for key in (
        "trial_id", "direction", "participant_id", "length", "sequence_index",
        "human_correct", "digits_presented", "expected_digits", "target_digits",
        "via", "sample_index", "prompt", "raw", "reasoning", "pred_digits",
        "accepted", "parse_errors",
    ):
        assert key in row, f"{key} disappeared from the sample pool schema"
    assert row["pred_digits"] == [1, 2]
    assert row["target_digits"] == trial.user_digits


def test_training_pairs_keep_the_digit_span_columns(cfg, trial):
    rows = [SampleRow(trial, GENERATION, 0, "p", "raw", "r", trial.user_digits, True, [])]
    pair = training_pairs(rows, cfg)[0]
    for key in ("trial_id", "direction", "length", "human_correct", "via",
                "prompt", "completion", "reasoning", "target_digits"):
        assert key in pair


# ---------------------------------------------------------------------------
# The two tasks stay distinguishable
# ---------------------------------------------------------------------------
def test_a_written_pool_row_round_trips_back_to_an_answer():
    """star._reload_pool rebuilds SampleRows from disk through this."""
    assert DIGIT_SPAN.answer_from_row({"pred_digits": [4, 1]}) == [4, 1]
    assert LISTENING.answer_from_row({"pred_options": [1, 3]}) == [1, 3]
    # A row written before the answer was parsed must not crash the reload.
    assert DIGIT_SPAN.answer_from_row({}) == []
    assert LISTENING.answer_from_row({}) == []


def test_cell_keys_split_human_success_from_human_fail():
    """pool_cell_key feeds fail_side_generation_yield, the bootstrap signal. If it did
    not split on human correctness the signal would be meaningless."""
    for task in ALL_TASKS:
        fields = task.item_fields(make_item() if task is LISTENING else _a_digit_trial())
        fail = task.pool_cell_key({**fields, "human_correct": False})
        success = task.pool_cell_key({**fields, "human_correct": True})
        assert fail.endswith(":fail") and success.endswith(":success")


def test_sibling_context_reaches_the_prompt_from_the_config():
    """It is a StarConfig field rather than a constructor-only argument so that it is
    serialized with the run and can actually be turned off. Bound through for_config;
    if that link broke, run_config.json would report a knob that does nothing."""
    from rationales.task import resolve_for

    item = make_item()
    on = resolve_for(StarConfig(task="listening_qa", sibling_context=True))
    off = resolve_for(StarConfig(task="listening_qa", sibling_context=False))
    assert on.sibling_context and not off.sibling_context
    assert "By helicopter" in on.build_sample_prompt(item, fewshot=False)
    assert "By helicopter" not in off.build_sample_prompt(item, fewshot=False)


def test_disabling_sibling_context_is_recorded_as_a_deviation():
    assert "sibling_context" not in StarConfig(task="listening_qa").deviations
    off = StarConfig(task="listening_qa", sibling_context=False)
    assert "sibling_context" in off.deviations


def test_dry_run_picks_up_the_per_task_cost_constants():
    """The override is looked up by f"{task.name.upper()}_*", so a constant named for
    the wrong key silently falls back to the digit-span number and the estimate is
    quietly wrong. That happened once."""
    from rationales import config as cfg_mod
    from rationales.star import dry_run

    for task in ALL_TASKS:
        for suffix in ("COMPLETION_TOKENS", "SUCCESS_MISS_RATE"):
            name = f"{task.name.upper()}_{suffix}"
            if hasattr(cfg_mod, name):
                assert getattr(cfg_mod, name) != getattr(cfg_mod, suffix), (
                    f"{name} duplicates the default; either it is wrong or it is pointless"
                )

    # listening_qa declares both, so its estimate must differ from the digit-span one.
    assert cfg_mod.LISTENING_QA_COMPLETION_TOKENS != cfg_mod.COMPLETION_TOKENS
    plan = dry_run(
        StarConfig(task="listening_qa", steps_1=10), round_n=1, task=LISTENING
    )
    assert f"{cfg_mod.LISTENING_QA_COMPLETION_TOKENS} completion tokens" in plan["estimate_basis"]
    assert f"{cfg_mod.LISTENING_QA_SUCCESS_MISS_RATE:.0%}" in plan["estimate_basis"]


def test_a_task_default_survives_the_cli_path():
    """typer passes a non-None default on every invocation, which is indistinguishable
    from the user typing the flag, so a task default is silently discarded unless the
    flag is declared Optional(None). Round 1 of listening_qa trained 60 steps instead
    of 200 for exactly this reason."""
    from rationales.cli import _cfg

    for field, expected in LISTENING.defaults.items():
        assert getattr(_cfg(task="listening_qa"), field) == expected, (
            f"{field}: the task default did not reach StarConfig through _cfg"
        )
    # An explicit value still wins.
    assert _cfg(task="listening_qa", steps_1=40).steps_1 == 40
    # And digit span keeps StarConfig's own defaults.
    assert _cfg(task="digit_span").steps_1 == StarConfig().steps_1


def test_every_cli_flag_a_task_overrides_is_declared_optional():
    """The structural guard. If a task adds a default for a field whose flag has a
    non-None typer default, the default is dead on arrival."""
    import inspect

    from rationales import cli

    sig = inspect.signature(cli.star_cmd)
    for task in ALL_TASKS:
        for field in getattr(task, "defaults", {}):
            if field in sig.parameters:
                assert sig.parameters[field].default.default is None, (
                    f"--{field.replace('_','-')} has a non-None typer default, so "
                    f"{task.name}'s default for it can never apply"
                )


def test_listening_declares_the_config_defaults_digit_span_would_get_wrong():
    """max_seq_length above all: build_datum truncates from the right, so a too-small
    budget silently removes the answer line the loss covers."""
    assert LISTENING.defaults["max_seq_length"] > StarConfig().max_seq_length
    assert LISTENING.defaults["steps_1"] > StarConfig().steps_1
