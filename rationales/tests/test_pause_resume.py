"""Pausing on spend, and resuming without re-paying for finished work."""
from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from rationales import prompting as P
from rationales.resume import AppendSink, RunState, completed_trial_ids, read_jsonl
from rationales.sample import sample_round
from rationales.select import Selection
from rationales.star import run_round, run_star
from rationales.tinker_client import BudgetExceeded, CostTracker, TinkerSession

DIGITS_RE = re.compile(r"The digits are the following: \[([0-9, ]+)\]")


@pytest.fixture
def selection(fwd_success, fwd_fail, rev_success, rev_fail) -> Selection:
    return Selection(
        train=[fwd_success, fwd_fail, rev_success],
        eval=[rev_fail],
        report={"train_ids": [], "eval_ids": []},
    )


def _responder(trials, tag="v1"):
    by_digits = {tuple(t.digits): t for t in trials}

    def responder(prompt, model_path):
        m = DIGITS_RE.search(prompt)
        trial = by_digits[tuple(int(x) for x in m.group(1).split(","))]
        return P.build_completion(f"reasoning {tag}", trial.user_digits)

    return responder


# --- run state ---------------------------------------------------------------
def test_state_round_trips(tmp_path):
    s = RunState(round=2, phase="train", train_step=40, train_state_path="tinker://s")
    s.save(tmp_path / "progress.json")
    assert RunState.load(tmp_path / "progress.json") == s


def test_missing_state_is_a_fresh_run(tmp_path):
    assert RunState.load(tmp_path / "nope.json") == RunState()


def test_phase_ordering(tmp_path):
    s = RunState(phase="train")
    assert s.is_done_with("sample") is True
    assert s.is_done_with("train") is False
    assert s.is_done_with("eval_tuned") is False


def test_finish_round_promotes_the_checkpoint():
    s = RunState(round=1, checkpoint="tinker://ckpt")
    s.finish_round()
    assert s.sampler_path == "tinker://ckpt"      # next round samples from it
    assert s.completed_rounds == [1]


# --- what counts as already paid for ----------------------------------------
def test_completed_trials_are_accepted_or_exhausted():
    rows = [
        {"trial_id": "a", "via": "generation", "accepted": True},
        {"trial_id": "b", "via": "generation", "accepted": False},
        {"trial_id": "b", "via": "rationalization", "accepted": False},
        {"trial_id": "c", "via": "generation", "accepted": False},   # cut off mid-flight
    ]
    done = completed_trial_ids(rows, k=1)
    assert done == {"a", "b"}
    assert "c" not in done, "a trial interrupted before rationalization must be retried"


def test_append_sink_does_not_truncate(tmp_path):
    s = AppendSink(tmp_path / "rows.jsonl")
    s.append({"x": 1})
    AppendSink(tmp_path / "rows.jsonl").append({"x": 2})
    assert [r["x"] for r in read_jsonl(tmp_path / "rows.jsonl")] == [1, 2]


# --- sampling resume ---------------------------------------------------------
def test_resumed_sampling_skips_finished_trials(cfg, selection, tmp_path):
    trials = selection.train
    pool = tmp_path / "pool.jsonl"

    calls = []

    def gen_first(prompt, n):
        calls.append(prompt)
        m = DIGITS_RE.search(prompt)
        digits = tuple(int(x) for x in m.group(1).split(","))
        trial = {tuple(t.digits): t for t in trials}[digits]
        # only the first trial succeeds; the others are left undecided
        if trial is trials[0]:
            return [P.build_completion("r", trial.user_digits)] * n
        raise RuntimeError("simulated pause")

    with pytest.raises(RuntimeError):
        sample_round(cfg, trials, generate=gen_first, pool_path=pool)

    done_before = len(read_jsonl(pool))
    assert done_before >= 1

    calls.clear()
    report = sample_round(
        cfg, trials, generate=_gen_ok(trials), pool_path=pool, resume=True
    )
    assert report["n_trials_skipped_already_done"] == 1
    assert report["resumed"] is True
    # the finished trial was never prompted again
    assert not any(trials[0].stimulus_text in c for c in calls)
    # and its original row survived
    assert len(read_jsonl(pool)) > done_before


def _gen_ok(trials):
    responder = _responder(trials)
    return lambda prompt, n: [responder(prompt, None)] * n


def test_fresh_round_rewrites_the_pool(cfg, selection, tmp_path):
    """Without resume, a round starts its pool clean -- STaR regenerates the split."""
    pool = tmp_path / "pool.jsonl"
    sample_round(cfg, selection.train, generate=_gen_ok(selection.train), pool_path=pool)
    n = len(read_jsonl(pool))
    sample_round(cfg, selection.train, generate=_gen_ok(selection.train), pool_path=pool)
    assert len(read_jsonl(pool)) == n


# --- pausing a real round ----------------------------------------------------
def test_round_pauses_on_budget_and_resumes(cfg, fake_tinker, selection, tmp_path):
    trials = selection.train + selection.eval
    fake_tinker.responder = _responder(trials)

    # A ceiling low enough to trip partway through sampling, but high enough that at
    # least one trial finishes first -- that finished trial is what resume must skip.
    tight = CostTracker(cfg.base_model, ledger_path=tmp_path / "cost.jsonl",
                        max_usd=0.0005, show_progress=False)
    session = TinkerSession(cfg, tracker=tight)
    session._tokenizer = fake_tinker.tokenizer

    with pytest.raises(BudgetExceeded):
        run_round(cfg, session, tmp_path, selection, round_n=1, sampler_path=None)

    pool = tmp_path / "rounds" / "round_1" / "sample_pool.jsonl"
    assert pool.is_file(), "work completed before the pause must be on disk"
    spent_rows = len(read_jsonl(pool))

    # Resume with no ceiling; already-decided trials are not re-sampled.
    roomy = CostTracker(cfg.base_model, show_progress=False)
    session2 = TinkerSession(cfg, tracker=roomy)
    session2._tokenizer = fake_tinker.tokenizer
    state = RunState(round=1, phase="sample")
    report = run_round(
        cfg, session2, tmp_path, selection, round_n=1, sampler_path=None,
        state=state, state_path=tmp_path / "progress.json",
    )
    assert report["checkpoint"]
    assert report["sample_report"]["n_trials_skipped_already_done"] >= 1
    assert len(read_jsonl(pool)) >= spent_rows


def test_run_star_writes_a_paused_marker(cfg, fake_tinker, tmp_path, monkeypatch):
    monkeypatch.setattr("rationales.star.out_root", lambda c, ts: tmp_path / ts)
    fake_tinker.responder = lambda prompt, model_path: P.build_completion("r", [1])

    summary = run_star(cfg, timestamp="TS", max_usd=1e-9)
    assert summary["paused"] is True
    assert "--resume" in summary["resume_with"]

    root = tmp_path / "TS"
    assert (root / "paused.json").is_file()
    state = json.loads((root / "progress.json").read_text())
    assert state["paused"] is True
    assert state["pause_reason"]
    assert (root / "cost_ledger.jsonl").is_file()


def test_completed_rounds_are_not_re_run(cfg, fake_tinker, tmp_path, monkeypatch):
    """Extending --rounds must not re-pay for a round that already finished."""
    monkeypatch.setattr("rationales.star.out_root", lambda c, ts: tmp_path / ts)
    root = tmp_path / "TS"
    (root / "rounds" / "round_1").mkdir(parents=True)
    (root / "rounds" / "round_1" / "eval.json").write_text(
        json.dumps({"round": 1, "checkpoint": "tinker://ckpt/r1"})
    )
    RunState(round=1, phase="done", completed_rounds=[1],
             sampler_path="tinker://ckpt/r1", checkpoint="tinker://ckpt/r1").save(
        root / "progress.json"
    )

    two = replace(cfg, rounds=2)
    fake_tinker.responder = lambda prompt, model_path: P.build_completion("r", [1])
    summary = run_star(two, timestamp="TS", resume=True, max_usd=1e-9)

    # round 2 started (and immediately hit the ceiling); round 1 was loaded, not re-run
    sampled_from = [c.model_path for c in fake_tinker.sampling_clients]
    assert sampled_from and sampled_from[0] == "tinker://ckpt/r1"
    assert summary["state"]["round"] == 2


def test_resume_requires_an_existing_run(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("rationales.star.out_root", lambda c, ts: tmp_path / ts)
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        run_star(cfg, timestamp="MISSING", resume=True)


# --- training checkpoints ----------------------------------------------------
def test_training_saves_state_periodically(cfg, fake_tinker, tmp_path):
    from rationales.train import train_round

    c = replace(cfg, steps_1=6, checkpoint_every=2)
    s = TinkerSession(c, tracker=CostTracker(c.base_model, show_progress=False))
    s._tokenizer = fake_tinker.tokenizer

    seen = []
    train_round(
        c, s, [{"prompt": "p", "completion": "c"}], round_n=1, out_dir=tmp_path,
        on_checkpoint=lambda step, path: seen.append((step, path)),
    )
    # steps 2 and 4 checkpoint; the final step saves sampler weights instead
    assert [step for step, _ in seen] == [2, 4]
    assert all(p.startswith("tinker://") for _, p in seen)


def test_resumed_training_skips_applied_steps(cfg, fake_tinker, tmp_path):
    from rationales.train import train_round

    c = replace(cfg, steps_1=6, checkpoint_every=0)
    s = TinkerSession(c, tracker=CostTracker(c.base_model, show_progress=False))
    s._tokenizer = fake_tinker.tokenizer

    log = train_round(
        c, s, [{"prompt": "p", "completion": "c"}], round_n=1, out_dir=tmp_path,
        start_step=4, resume_state_path="tinker://state/step4",
    )
    assert log["resumed_from_step"] == 4
    assert log["resume_state_path"] == "tinker://state/step4"
    assert [e["step"] for e in log["steps"]] == [4, 5], "steps 0-3 must not be re-run"


def test_checkpoint_every_zero_disables_mid_round_saves(cfg, fake_tinker, tmp_path):
    from rationales.train import train_round

    c = replace(cfg, steps_1=4, checkpoint_every=0)
    s = TinkerSession(c, tracker=CostTracker(c.base_model, show_progress=False))
    s._tokenizer = fake_tinker.tokenizer
    seen = []
    train_round(
        c, s, [{"prompt": "p", "completion": "c"}], round_n=1, out_dir=tmp_path,
        on_checkpoint=lambda step, path: seen.append(step),
    )
    assert seen == []


# --- eval resume -------------------------------------------------------------
def test_eval_resumes_from_scored_rows(cfg, rev_fail, fwd_fail, tmp_path):
    from rationales import evaluate as ev

    rows_path = tmp_path / "eval_rows.jsonl"
    AppendSink(rows_path).append(
        {
            "trial_id": rev_fail.trial_id, "direction": "reverse", "length": 4,
            "human_correct": False, "digits_presented": rev_fail.digits,
            "expected_digits": rev_fail.expected_digits,
            "human_digits": rev_fail.user_digits, "pred_digits": rev_fail.user_digits,
            "reasoning": "cached", "raw": "", "parse_errors": [],
            "human_match": True, "ground_truth_correct": False,
            "error_type_model": "truncation", "error_type_human": "truncation",
            "features_model": {"prefix_frac": 1.0, "len_delta": -1},
            "features_human": {"prefix_frac": 1.0, "len_delta": -1},
        }
    )

    calls = []

    def generate(prompt, n):
        calls.append(prompt)
        return [P.build_completion("fresh", fwd_fail.user_digits)]

    res = ev.run_eval(
        cfg, [rev_fail, fwd_fail], generate=generate, label="t", rows_path=rows_path
    )
    assert res["n_resumed"] == 1
    assert len(calls) == 1, "the cached trial must not be re-scored"
    assert rev_fail.stimulus_text not in calls[0]
    assert res["metrics"]["overall"]["n"] == 2
