"""One listening round through the real loop, against the fake Tinker SDK.

Everything else in the listening tests is unit-level. This exercises the path that
only exists when the pieces are wired together --

    sample_round -> sample_pool.jsonl -> _reload_pool -> training_pairs
                 -> build_corpus -> train_round -> run_eval -> eval.json

-- which is where a key-name mismatch between a task's item_fields, pair_fields and
cell_key would surface. That failure mode costs nothing here and costs a paid
sampling phase on a live run, because it lands at the filter step, after every
rationale has been bought.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from rationales.select import Selection
from rationales.star import run_round
from rationales.tasks.listening_qa import ListeningQATask

from listening_fixtures import make_item

TASK = ListeningQATask()

# The target question is the one under "Question to predict:"; the sibling block quotes
# other questions, so the stub has to read the right one to answer as that human.
TARGET_RE = re.compile(r"Question to predict:\n(.+?)\n", re.DOTALL)


@pytest.fixture
def cfg(cfg):
    """The digit-span cfg fixture, pointed at listening QA.

    steps_1 is left small: this test is about wiring, not about a schedule.
    """
    return replace(cfg, task="listening_qa", max_seq_length=4096)


@pytest.fixture
def items():
    """Twelve items over three participants, spanning levels and both correctness
    strata, each with a distinct question text so the stub can route on it."""
    out = []
    for p in range(3):
        for q, (gold, endorsed, level) in enumerate(
            [
                ((1, 2, 3), (1, 2, 3), "control"),          # human correct
                ((1, 2, 3), (1, 3), "repeat_short"),        # missed a true option
                ((1, 2), (1, 4), "distractor"),             # endorsed a foil
                ((1, 2), (5,), "repeat_long"),              # none of the above
            ]
        ):
            item = make_item(
                respondent=f"r{p:04d}",
                question_id=f"QV{q:02d}",
                gold=gold,
                endorsed=endorsed,
                level=level,
            )
            out.append(
                replace(item, question=f"Question number {q} about the Skerry Light?")
            )
    return out


@pytest.fixture
def selection(items):
    # Split by participant, as the real selector does.
    train = [i for i in items if i.respondent_id != "r0002"]
    held = [i for i in items if i.respondent_id == "r0002"]
    return Selection(train=train, eval=held, report={"train_ids": [], "eval_ids": []})


def _responder(items, tag="base"):
    """Answer every prompt with that human's own selection, so the whole train split
    is accepted and the round has a corpus to train on."""
    by_question = {i.question: i for i in items}

    def responder(prompt: str, model_path):
        target = TARGET_RE.search(prompt).group(1)
        item = next(i for q, i in by_question.items() if prompt and q in target)
        return TASK.build_completion(f"reasoning {tag}", item.endorsed)

    return responder


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture
def round_one(cfg, session, fake_tinker, selection, items, tmp_path):
    fake_tinker.responder = _responder(items)
    return run_round(
        cfg, session, tmp_path, selection, round_n=1, sampler_path=None, task=TASK
    )


# ---------------------------------------------------------------------------
def test_a_listening_round_runs_end_to_end(round_one, tmp_path):
    payload = round_one
    assert payload["round"] == 1
    assert payload["checkpoint"]
    assert (tmp_path / "rounds" / "round_1" / "eval.json").is_file()


def test_the_sample_pool_carries_the_listening_columns(round_one, tmp_path):
    rows = _rows(tmp_path / "rounds" / "round_1" / "sample_pool.jsonl")
    assert rows
    for key in (
        "trial_id", "respondent_id", "topic", "level", "question_id",
        "human_correct", "gold_options", "target_options", "pred_options",
        "via", "prompt", "raw", "reasoning", "accepted", "parse_errors",
    ):
        assert key in rows[0], f"{key} missing from the listening sample pool"


def test_every_train_item_is_accepted_when_the_model_answers_as_the_human(
    round_one, selection, tmp_path
):
    rows = _rows(tmp_path / "rounds" / "round_1" / "sample_pool.jsonl")
    accepted = {r["trial_id"] for r in rows if r["accepted"]}
    assert accepted == {i.item_id for i in selection.train}


def test_the_corpus_trains_on_hint_free_zero_shot_prompts(round_one, tmp_path):
    corpus = _rows(tmp_path / "rounds" / "round_1" / "accepted.jsonl")
    assert corpus
    for row in corpus:
        assert "actually selected" not in row["prompt"]
        assert "Bracklin" not in row["prompt"]      # no few-shot demo text
        assert row["completion"].startswith("<reasoning>")
        assert "Answer:" in row["completion"]


def test_filter_stats_cell_keys_resolve_for_listening(round_one, tmp_path):
    """cell_key reads topic/level/question_id/human_correct off pair_fields. A name
    that pair_fields does not emit would KeyError here, after sampling was paid for."""
    stats = json.loads((tmp_path / "rounds" / "round_1" / "filter_stats.json").read_text())
    assert stats["n_accepted_total"] > 0
    assert stats["by_cell"]
    key = next(iter(stats["by_cell"]))
    assert key.count(":") == 3 and key.endswith(("success", "fail"))


def test_the_bootstrap_signal_is_computed_over_human_fail_items(round_one):
    """pool_cell_key must split on human correctness, or fail_side_generation_yield
    is meaningless. Here every item accepts unhinted, so the yield is 1.0."""
    assert round_one["fail_side_generation_yield"] == pytest.approx(1.0)
    cells = round_one["sample_report"]["stats"]["cells"]
    assert any(k.endswith(":fail") for k in cells)
    assert any(k.endswith(":success") for k in cells)


def test_eval_scores_the_held_out_participant_only(round_one, selection, tmp_path):
    rows = _rows(tmp_path / "rounds" / "round_1" / "eval_rows_tuned.jsonl")
    assert {r["trial_id"] for r in rows} == {i.item_id for i in selection.eval}
    assert {r["respondent_id"] for r in rows} == {"r0002"}


def test_every_headline_delta_survives_the_round(round_one):
    """The end of the contract that test_listening_evaluate checks in isolation: the
    deltas compare() writes into eval.json must not be null."""
    comparison = round_one["comparison"]
    for key, value in comparison.items():
        if key == "reading":
            continue
        assert value is not None, f"{key} came out null in eval.json"


def test_figures_are_written_rather_than_swallowed(round_one):
    """star.py catches plotting errors so a completed round is never lost, which also
    means a broken figure is silent. This is the thing that notices."""
    assert "figures_error" not in round_one, round_one.get("figures_error")
    assert round_one["figures"]
