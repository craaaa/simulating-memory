"""What actually gets handed to the LoRA trainer.

The loss mask is the whole correctness of the SFT step: weight 0 over the prompt,
weight 1 over the completion. If it is wrong the model is trained to reproduce the
prompt as well as the rationale, and nothing downstream would notice.
"""
from __future__ import annotations

import pytest

from rationales.tinker_client import TinkerSession, Usage, offline_token_estimate, require_api_key


def test_datum_masks_prompt_and_trains_completion(session, fake_tinker):
    prompt, completion = "PROMPT", "DONE"
    datum, meta = session.build_datum(prompt, completion)

    tok = fake_tinker.tokenizer
    n_prompt = len(tok.encode(prompt))
    n_completion = len(tok.encode(completion)) + 1  # + EOS

    assert meta["n_prompt_tokens"] == n_prompt
    assert meta["n_tokens"] == n_prompt + n_completion

    # inputs/targets are shifted by one
    assert len(datum.tokens) == len(datum.target_tokens) == n_prompt + n_completion - 1
    assert datum.target_tokens == tok.encode(prompt + completion)[1:] + [tok.eos_token_id]

    # exactly the completion positions carry weight
    assert set(datum.weights) == {0.0, 1.0}
    assert datum.weights[: n_prompt - 1] == [0.0] * (n_prompt - 1)
    assert datum.weights[n_prompt - 1 :] == [1.0] * n_completion
    assert sum(datum.weights) == n_completion


def test_weighted_targets_decode_back_to_the_completion(session, fake_tinker):
    """The tokens actually being learned must be the completion, nothing else."""
    prompt, completion = "the prompt text", "<reasoning>r</reasoning>\npress <<4>>."
    datum, _ = session.build_datum(prompt, completion)
    learned = [t for t, w in zip(datum.target_tokens, datum.weights) if w == 1.0]
    assert fake_tinker.decode(learned) == completion


def test_eos_is_appended_so_generation_can_stop(session, fake_tinker):
    datum, _ = session.build_datum("p", "c")
    assert datum.target_tokens[-1] == fake_tinker.tokenizer.eos_token_id
    assert datum.weights[-1] == 1.0


def test_long_examples_are_truncated_to_max_seq_length(cfg, fake_tinker):
    from dataclasses import replace

    small = replace(cfg, max_seq_length=10)
    s = TinkerSession(small)
    s._tokenizer = fake_tinker.tokenizer
    datum, meta = s.build_datum("x" * 50, "y" * 50)
    assert meta["n_tokens"] == 10
    assert len(datum.tokens) == 9
    # truncation cuts the completion -- flagged so it cannot pass silently
    assert meta["truncated"] is True
    assert meta["dropped_tokens"] == 91  # 50 + 50 + EOS - 10


def test_examples_within_budget_are_not_flagged(session):
    _, meta = session.build_datum("short prompt", "short completion")
    assert meta["truncated"] is False
    assert meta["dropped_tokens"] == 0


def test_default_budget_fits_a_real_prompt_plus_a_long_rationale(cfg):
    """The probe produced rationales up to ~3400 characters; the default budget must
    hold one alongside a real training prompt.

    Measured in estimated real tokens, not through the char-level fake tokenizer --
    a real BPE runs ~4 chars/token, so the fake would overstate this by ~4x.
    """
    from rationales.config import StarConfig
    from rationales.data import load_all
    from rationales.prompting import build_completion, build_sample_prompt
    from rationales.tinker_client import offline_token_estimate as est

    trial = load_all(["reverse"])[0][0]
    prompt = build_sample_prompt(trial, fewshot=False)
    completion = build_completion("s" * 3400, trial.user_digits)

    needed = est(prompt) + est(completion)
    assert needed <= StarConfig().max_seq_length, (
        f"default max_seq_length={StarConfig().max_seq_length} is below the "
        f"~{needed} tokens a worst-case example needs"
    )


def test_training_client_is_always_created_from_the_base_model(session, fake_tinker, cfg):
    a = session.training_client()
    b = session.training_client()
    assert a.base_model == b.base_model == cfg.base_model
    assert a is not b, "each round must get a fresh client, not a resumed one"
    assert [c.rank for c in fake_tinker.training_clients] == [cfg.lora_rank] * 2


def test_sampling_client_targets_base_or_checkpoint(session, cfg):
    base = session.sampling_client(None)
    tuned = session.sampling_client("tinker://ckpt/round1")
    assert base.model_path is None and base.base_model == cfg.base_model
    assert tuned.model_path == "tinker://ckpt/round1"


def test_sample_returns_text_and_counts_tokens(session, fake_tinker, cfg):
    fake_tinker.responder = lambda prompt, model_path: "ANSWER"
    client = session.sampling_client(None)
    out = session.sample(client, "PROMPT", num_samples=2, temperature=0.0, max_tokens=16)

    assert out == ["ANSWER", "ANSWER"]
    assert session.usage.prefill_tokens == len("PROMPT") * 2
    assert session.usage.sample_tokens == len("ANSWER") * 2


def test_missing_api_key_is_refused(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        require_api_key()


def test_usage_costs_use_tinker_rates():
    u = Usage()
    u.add(prefill=1_000_000, sample=1_000_000, train=1_000_000)
    s = u.summary("Qwen/Qwen3-8B")
    assert s["estimated_cost_usd"] == pytest.approx(0.195 + 0.60 + 0.44)
    assert Usage().summary("some/unpriced-model")["estimated_cost_usd"] is None


def test_offline_estimate_uses_the_measured_ratio():
    from rationales.config import CHARS_PER_TOKEN

    assert offline_token_estimate("") == 1
    assert offline_token_estimate("a" * 3720) == int(3720 / CHARS_PER_TOKEN)
    assert offline_token_estimate("a" * 800) > offline_token_estimate("a" * 400)
    # the old flat 4-chars rule under-counted; the measured ratio must not
    assert offline_token_estimate("a" * 3337) > 3337 // 4
