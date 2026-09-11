"""Algorithm 1 lines 3-6: generation, rationalization, and corpus construction."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rationales import prompting as P
from rationales.filter import build_corpus
from rationales.sample import (
    GENERATION,
    RATIONALIZATION,
    path_stats,
    sample_round,
    training_pairs,
)


def _responder(mapping, default=""):
    """Return a generate() that answers based on a marker found in the prompt."""

    def generate(prompt, n):
        for marker, text in mapping.items():
            if marker in prompt:
                return [text] * n
        return [default] * n

    return generate


def _completion(trial, digits=None, reasoning="they lost the tail."):
    return P.build_completion(reasoning, digits if digits is not None else trial.user_digits)


# --- acceptance target is the HUMAN's response, not the correct answer -----
def test_fail_trial_accepts_only_the_humans_wrong_answer(cfg, fwd_fail, tmp_path):
    # Model outputs the CORRECT answer -> must be rejected on the generation path.
    gen = _responder({"": _completion(fwd_fail, fwd_fail.expected_digits)})
    report = sample_round(cfg, [fwd_fail], generate=gen, pool_path=tmp_path / "pool.jsonl")
    rows = json.loads("[" + ",".join(
        (tmp_path / "pool.jsonl").read_text().strip().splitlines()
    ) + "]")
    assert all(r["accepted"] is False for r in rows)
    assert report["stats"]["n_accepted"] == 0
    # ... and it fell through to rationalization
    assert any(r["via"] == RATIONALIZATION for r in rows)


def test_fail_trial_accepted_when_it_reproduces_the_human_error(cfg, fwd_fail, tmp_path):
    gen = _responder({"": _completion(fwd_fail)})
    report = sample_round(cfg, [fwd_fail], generate=gen, pool_path=tmp_path / "pool.jsonl")
    assert report["stats"]["n_accepted"] == 1
    assert report["stats"]["n_accepted_generation"] == 1
    assert report["stats"]["fail_side_generation_yield"] == 1.0


def test_rationalization_runs_only_after_generation_fails(cfg, fwd_success, fwd_fail, tmp_path):
    gen = _responder(
        {
            "[8, 4, 1]": _completion(fwd_success),  # matches the success trial only
        },
        default="",
    )
    # success trial: accepted on generation; fail trial: falls through
    sample_round(cfg, [fwd_success], generate=gen, pool_path=tmp_path / "a.jsonl")
    rows = [json.loads(l) for l in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert {r["via"] for r in rows} == {GENERATION}, "no hint path when generation succeeds"


def test_rationalization_prompt_is_used_on_the_second_attempt(cfg, rev_fail, tmp_path):
    seen = []

    def generate(prompt, n):
        seen.append(prompt)
        # only answer correctly when handed the hint
        if "actually responded" in prompt:
            return [_completion(rev_fail)] * n
        return ["<reasoning>guess</reasoning>\npress <<1>>."] * n

    report = sample_round(cfg, [rev_fail], generate=generate, pool_path=tmp_path / "p.jsonl")
    assert len(seen) == 2
    assert "actually responded" not in seen[0]
    assert "actually responded" in seen[1]
    assert report["stats"]["n_accepted_rationalization"] == 1
    assert report["stats"]["fail_side_generation_yield"] == 0.0


# --- what ends up in the training corpus -----------------------------------
def test_training_pair_strips_hint_and_fewshot(cfg, rev_fail, tmp_path):
    gen = _responder({"actually responded": _completion(rev_fail)}, default="nope")
    sample_round(cfg, [rev_fail], generate=gen, pool_path=tmp_path / "p.jsonl")

    from rationales.star import _reload_pool

    rows = _reload_pool(tmp_path / "p.jsonl", {rev_fail.trial_id: rev_fail})
    pairs = training_pairs(rows, cfg)
    assert len(pairs) == 1
    pair = pairs[0]
    assert "actually responded" not in pair["prompt"]
    assert pair["prompt"] == P.build_sample_prompt(rev_fail, fewshot=False)
    assert pair["completion"].startswith(P.REASONING_OPEN)
    assert pair["target_digits"] == rev_fail.user_digits
    assert pair["via"] == RATIONALIZATION


def test_only_accepted_rows_become_training_pairs(cfg, fwd_fail):
    from rationales.sample import SampleRow

    rows = [
        SampleRow(fwd_fail, GENERATION, 0, "p", "raw", "r", [9, 9, 9], False, []),
        SampleRow(fwd_fail, GENERATION, 1, "p", "raw", "r", fwd_fail.user_digits, True, []),
    ]
    assert len(training_pairs(rows, cfg)) == 1


def test_accepted_row_without_reasoning_is_not_trainable(cfg, fwd_fail):
    from rationales.sample import SampleRow

    rows = [SampleRow(fwd_fail, GENERATION, 0, "p", "raw", None, fwd_fail.user_digits, True, [])]
    assert training_pairs(rows, cfg) == []


def test_at_most_one_sample_is_accepted_per_trial(cfg, fwd_fail, tmp_path):
    many = replace(cfg, k=3, temperature=1.0)  # k>1 requires a temperature
    gen = _responder({"": _completion(fwd_fail)})
    sample_round(many, [fwd_fail], generate=gen, pool_path=tmp_path / "p.jsonl")
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert sum(r["accepted"] for r in rows) == 1
    assert len(rows) == 3


def test_pool_records_rejected_samples_for_inspection(cfg, fwd_fail, tmp_path):
    gen = _responder({"": "garbage with no block"})
    sample_round(cfg, [fwd_fail], generate=gen, pool_path=tmp_path / "p.jsonl")
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert rows and all(r["accepted"] is False for r in rows)
    assert all(r["parse_errors"] == ["no_reasoning_block"] for r in rows)
    assert {r["via"] for r in rows} == {GENERATION, RATIONALIZATION}


def test_fewshot_placeholders_block_sampling(fwd_fail, tmp_path, monkeypatch):
    """Unwritten demos must stop a run before it spends anything.

    The shipped files are filled in, so the placeholder state is stubbed here rather
    than asserted against them.
    """
    from rationales.config import StarConfig

    monkeypatch.setattr(P, "load_fewshot", lambda d: "PLACEHOLDER — write me")
    cfg_fs = StarConfig(use_fewshot=True, max_workers=1)
    with pytest.raises(RuntimeError, match="PLACEHOLDER"):
        sample_round(cfg_fs, [fwd_fail], generate=lambda p, n: [""], pool_path=tmp_path / "p.jsonl")


def test_shipped_fewshot_files_pass_the_gate():
    """The committed demos are ready to sample with."""
    from rationales.sample import check_fewshot_ready
    from rationales.config import StarConfig

    check_fewshot_ready(StarConfig(use_fewshot=True))


# --- filter ----------------------------------------------------------------
def _pair(trial, via, reasoning):
    return {
        "trial_id": trial.trial_id,
        "direction": trial.direction,
        "length": trial.length,
        "human_correct": trial.correct,
        "via": via,
        "prompt": "P",
        "completion": "C",
        "reasoning": reasoning,
        "target_digits": trial.user_digits,
    }


def test_filter_splits_D_n_and_D_rat_n(cfg, fwd_success, fwd_fail, tmp_path):
    pairs = [
        _pair(fwd_success, GENERATION, "clean recall"),
        _pair(fwd_fail, RATIONALIZATION, "the tail decayed"),
    ]
    corpus, stats = build_corpus(pairs, cfg, out_dir=tmp_path)
    assert stats["size_D_n"] == 1 and stats["size_D_rat_n"] == 1
    assert len(corpus) == 2
    assert stats["fail_share"] == 0.5


def test_filter_drops_leaked_rationales(cfg, fwd_fail, tmp_path):
    pairs = [_pair(fwd_fail, RATIONALIZATION, "The human actually responded 8 1 4.")]
    corpus, stats = build_corpus(pairs, cfg, out_dir=tmp_path)
    assert corpus == []
    assert stats["n_dropped_hint_leak"] == 1
    assert stats["dropped_hint_leak"][0]["trial_id"] == fwd_fail.trial_id


def test_leak_filter_can_be_disabled_to_match_the_paper(cfg, fwd_fail, tmp_path):
    off = replace(cfg, leak_filter=False)
    pairs = [_pair(fwd_fail, RATIONALIZATION, "The human actually responded 8 1 4.")]
    corpus, stats = build_corpus(pairs, off, out_dir=tmp_path)
    assert len(corpus) == 1 and stats["n_dropped_hint_leak"] == 0


def test_filter_does_not_cap_or_downsample(cfg, fwd_fail, tmp_path):
    """No per-cell cap: balance is decided once in select.py. A cap here would
    silently break the 1:1 ratio and is not part of STaR."""
    pairs = [
        dict(_pair(fwd_fail, GENERATION, f"reason {i}"), trial_id=f"t{i}")
        for i in range(25)
    ]
    corpus, stats = build_corpus(pairs, cfg, out_dir=tmp_path)
    assert len(corpus) == 25
    assert stats["by_cell"] == {"forward:3:fail": 25}


def test_accepted_jsonl_matches_returned_corpus(cfg, fwd_success, tmp_path):
    pairs = [_pair(fwd_success, GENERATION, "fine")]
    corpus, _ = build_corpus(pairs, cfg, out_dir=tmp_path)
    written = [json.loads(l) for l in (tmp_path / "accepted.jsonl").read_text().splitlines()]
    assert written == corpus


def test_path_stats_reports_per_cell_counts(cfg, fwd_success, rev_fail):
    from rationales.sample import SampleRow

    rows = [
        SampleRow(fwd_success, GENERATION, 0, "p", "r", "x", fwd_success.user_digits, True, []),
        SampleRow(rev_fail, GENERATION, 0, "p", "r", "x", [0], False, []),
        SampleRow(rev_fail, RATIONALIZATION, 0, "p", "r", "x", rev_fail.user_digits, True, []),
    ]
    stats = path_stats(rows)
    assert stats["cells"]["forward:success"] == {
        "n": 1, "accepted": 1, GENERATION: 1, RATIONALIZATION: 0
    }
    assert stats["cells"]["reverse:fail"][RATIONALIZATION] == 1
    assert stats["fail_side_generation_yield"] == 0.0
