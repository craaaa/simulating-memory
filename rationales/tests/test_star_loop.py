"""The outer loop: which model is used where, and what is thrown away each round.

Three things STaR's Algorithm 1 requires that are easy to get silently wrong:
  * round N+1 SAMPLES from round N's checkpoint (line 3 uses M_{n-1});
  * every round TRAINS from the original base model (line 7 trains M, not M_{n-1});
  * every round uses its own D_n u D^rat_n -- rationales are regenerated from scratch
    and the previous round's are dropped, never accumulated.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from rationales import prompting as P
from rationales.select import Selection
from rationales.star import _reload_pool, prepare_run, run_round
from rationales.tinker_client import TinkerSession

DIGITS_RE = re.compile(r"The digits are the following: \[([0-9, ]+)\]")


@pytest.fixture
def selection(fwd_success, fwd_fail, rev_success, rev_fail) -> Selection:
    return Selection(
        train=[fwd_success, fwd_fail, rev_success],
        eval=[rev_fail],
        report={"train_ids": [], "eval_ids": []},
    )


def _responder_factory(trials, tag_by_round):
    """Answer each prompt with that trial's human response, tagged by which model
    produced it, so rationales can be traced back to a round."""
    by_digits = {tuple(t.digits): t for t in trials}

    def responder(prompt: str, model_path):
        m = DIGITS_RE.search(prompt)
        digits = tuple(int(x) for x in m.group(1).split(","))
        trial = by_digits[digits]
        tag = tag_by_round.get(model_path, f"from:{model_path}")
        return P.build_completion(f"reasoning {tag}", trial.user_digits)

    return responder


def _round(cfg, session, root, selection, n, sampler_path):
    return run_round(cfg, session, root, selection, round_n=n, sampler_path=sampler_path)


def test_round_one_samples_the_base_model(cfg, session, fake_tinker, selection, tmp_path):
    fake_tinker.responder = _responder_factory(
        selection.train + selection.eval, {None: "base"}
    )
    _round(cfg, session, tmp_path, selection, 1, None)
    # first sampling client built for generation is the base model
    assert fake_tinker.sampling_clients[0].model_path is None
    assert fake_tinker.sampling_clients[0].base_model == cfg.base_model


def test_round_two_samples_from_round_one_checkpoint(cfg, session, fake_tinker, selection, tmp_path):
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder_factory(trials, {None: "base"})
    r1 = _round(cfg, session, tmp_path, selection, 1, None)

    n_clients = len(fake_tinker.sampling_clients)
    fake_tinker.responder = _responder_factory(trials, {r1["checkpoint"]: "round1"})
    r2 = _round(cfg, session, tmp_path, selection, 2, r1["checkpoint"])

    assert fake_tinker.sampling_clients[n_clients].model_path == r1["checkpoint"]
    assert r2["sampled_from"] == r1["checkpoint"]
    assert r1["checkpoint"] != r2["checkpoint"]


def test_every_round_trains_from_the_base_model(cfg, session, fake_tinker, selection, tmp_path):
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder_factory(trials, {None: "base"})
    r1 = _round(cfg, session, tmp_path, selection, 1, None)
    fake_tinker.responder = _responder_factory(trials, {r1["checkpoint"]: "round1"})
    _round(cfg, session, tmp_path, selection, 2, r1["checkpoint"])

    assert len(fake_tinker.training_clients) == 2
    assert {c.base_model for c in fake_tinker.training_clients} == {cfg.base_model}
    # a fresh client each round -- never resumed from the previous adapter
    assert fake_tinker.training_clients[0] is not fake_tinker.training_clients[1]
    logs = [
        json.loads((tmp_path / "rounds" / f"round_{n}" / "train_log.json").read_text())
        for n in (1, 2)
    ]
    assert all(l["trained_from"] == cfg.base_model for l in logs)
    assert all("not accumulated" in l["dataset_scope"] for l in logs)


def test_each_round_samples_fresh_rationales(cfg, session, fake_tinker, selection, tmp_path):
    """Round 2 must regenerate rationales for the whole train split, not reuse round 1."""
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder_factory(trials, {None: "base"})
    r1 = _round(cfg, session, tmp_path, selection, 1, None)
    fake_tinker.responder = _responder_factory(trials, {r1["checkpoint"]: "round1"})
    _round(cfg, session, tmp_path, selection, 2, r1["checkpoint"])

    pool1 = _rows(tmp_path / "rounds" / "round_1" / "sample_pool.jsonl")
    pool2 = _rows(tmp_path / "rounds" / "round_2" / "sample_pool.jsonl")

    # every train trial is re-sampled in round 2
    assert {r["trial_id"] for r in pool2} == {t.trial_id for t in selection.train}
    # and the round-2 rationales are the new model's, not round 1's
    assert all("reasoning base" in r["raw"] for r in pool1)
    assert all("reasoning round1" in r["raw"] for r in pool2)
    assert not any("reasoning base" in r["raw"] for r in pool2)


def test_previous_round_rationales_are_dropped_not_accumulated(
    cfg, session, fake_tinker, selection, tmp_path
):
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder_factory(trials, {None: "base"})
    r1 = _round(cfg, session, tmp_path, selection, 1, None)
    fake_tinker.responder = _responder_factory(trials, {r1["checkpoint"]: "round1"})
    r2 = _round(cfg, session, tmp_path, selection, 2, r1["checkpoint"])

    corpus1 = _rows(tmp_path / "rounds" / "round_1" / "accepted.jsonl")
    corpus2 = _rows(tmp_path / "rounds" / "round_2" / "accepted.jsonl")

    # same size, not double: round 2 trains on its own D_n only
    assert len(corpus2) == len(corpus1) == len(selection.train)
    assert all("reasoning round1" in r["completion"] for r in corpus2)
    assert r2["train_log"]["corpus_size"] == len(corpus2)
    # each round writes into its own directory
    assert (tmp_path / "rounds" / "round_1" / "accepted.jsonl").exists()
    assert (tmp_path / "rounds" / "round_2" / "accepted.jsonl").exists()


def test_rerunning_a_round_truncates_the_old_pool(cfg, session, fake_tinker, selection, tmp_path):
    """JsonlSink truncates on construction, so a re-run replaces rather than appends."""
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder_factory(trials, {None: "base"})
    _round(cfg, session, tmp_path, selection, 1, None)
    n_first = len(_rows(tmp_path / "rounds" / "round_1" / "sample_pool.jsonl"))
    _round(cfg, session, tmp_path, selection, 1, None)
    n_second = len(_rows(tmp_path / "rounds" / "round_1" / "sample_pool.jsonl"))
    assert n_second == n_first


def test_round_evaluates_tuned_against_base(cfg, session, fake_tinker, selection, tmp_path):
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder_factory(trials, {None: "base"})
    report = _round(cfg, session, tmp_path, selection, 1, None)

    assert set(report) >= {"base", "tuned", "comparison", "fail_side_generation_yield", "usage"}
    paths = [c.model_path for c in fake_tinker.sampling_clients]
    assert report["checkpoint"] in paths and None in paths
    assert (tmp_path / "rounds" / "round_1" / "eval.json").exists()
    assert report["usage"]["estimated_cost_usd"] is not None


def test_empty_corpus_fails_loudly(cfg, session, fake_tinker, selection, tmp_path):
    fake_tinker.responder = lambda prompt, model_path: "unparseable"
    with pytest.raises(RuntimeError, match="no rationale was accepted"):
        _round(cfg, session, tmp_path, selection, 1, None)


def test_step_budget_grows_twenty_percent_per_round(cfg):
    assert cfg.steps_for_round(1) == cfg.steps_1
    big = replace(cfg, steps_1=40)
    assert big.steps_for_round(2) == 48
    assert big.steps_for_round(3) == 58


def test_learning_rate_warms_up_then_holds(cfg):
    from rationales.train import _lr_at

    warm = replace(cfg, warmup_steps=4, learning_rate=1e-4)
    lrs = [_lr_at(s, warm) for s in range(6)]
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3]
    assert lrs[3] == pytest.approx(1e-4)
    assert lrs[4] == lrs[5] == pytest.approx(1e-4)


def test_prepare_run_writes_provenance_and_pins_the_split(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("rationales.star.out_root", lambda c, ts: tmp_path / ts)
    root, sel = prepare_run(cfg, timestamp="20260911T000000Z")
    cfg_json = json.loads((root / "run_config.json").read_text())

    assert cfg_json["git_provenance"]["commit"]
    assert cfg_json["sampler_backend"] == "tinker"
    assert "Algorithm 1" in cfg_json["star_reference"]
    assert cfg_json["prompt_additions"]["task_line_reverse"]
    assert cfg_json["config"]["deviations"]["filter_target"] == "human_response_not_ground_truth"

    # re-preparing the same run reuses selection.json rather than resampling
    root2, sel2 = prepare_run(cfg, timestamp="20260911T000000Z")
    assert [t.trial_id for t in sel2.train] == [t.trial_id for t in sel.train]


def test_deviation_record_flags_non_greedy_decoding(cfg):
    assert "decoding" not in cfg.deviations
    noisy = replace(cfg, k=4, temperature=1.0)
    assert "k=4" in noisy.deviations["decoding"]


def test_k_above_one_at_zero_temperature_is_refused(cfg):
    """Greedy decoding is deterministic, so k>1 at temperature 0 is k identical
    samples -- k-fold spend for nothing."""
    with pytest.raises(ValueError, match="no extra coverage"):
        replace(cfg, k=4)
    # with a temperature it is allowed, just recorded as a deviation
    assert replace(cfg, k=4, temperature=1.0).k == 4


def test_k_must_be_at_least_one(cfg):
    with pytest.raises(ValueError, match="k must be >= 1"):
        replace(cfg, k=0)


def _rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
