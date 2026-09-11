"""Live cost tracking: ledger rows, running totals, and the spend ceiling."""
from __future__ import annotations

import json

import pytest

from rationales.tinker_client import BudgetExceeded, CostTracker, TinkerSession


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_ledger_row_per_call_with_running_total(tmp_path):
    t = CostTracker("Qwen/Qwen3-8B", ledger_path=tmp_path / "cost.jsonl", show_progress=False)
    t.record("sample", prefill=1_000_000, sample=0)
    t.record("sample", prefill=0, sample=1_000_000)
    t.record("train", train=1_000_000)

    rows = _rows(tmp_path / "cost.jsonl")
    assert [r["kind"] for r in rows] == ["sample", "sample", "train"]
    # cumulative total climbs and matches the list prices
    totals = [r["cumulative"]["cost_usd"] for r in rows]
    assert totals == sorted(totals)
    assert totals[0] == pytest.approx(0.195)
    assert totals[1] == pytest.approx(0.195 + 0.60)
    assert totals[2] == pytest.approx(0.195 + 0.60 + 0.44)
    assert rows[-1]["cumulative"]["requests"] == 3


def test_extra_fields_land_in_the_ledger(tmp_path):
    t = CostTracker("Qwen/Qwen3-8B", ledger_path=tmp_path / "c.jsonl", show_progress=False)
    t.record("train", train=10, round=2, step=7)
    row = _rows(tmp_path / "c.jsonl")[0]
    assert row["round"] == 2 and row["step"] == 7


def test_budget_ceiling_aborts_the_run(tmp_path):
    t = CostTracker(
        "Qwen/Qwen3-8B", ledger_path=tmp_path / "c.jsonl", max_usd=0.10, show_progress=False
    )
    t.record("sample", prefill=100_000)          # $0.0195, under
    with pytest.raises(BudgetExceeded, match="exceeded the \\$0.10 ceiling"):
        t.record("sample", prefill=1_000_000)    # crosses it

    # the crossing call is still recorded, so the ledger explains the abort
    assert len(_rows(tmp_path / "c.jsonl")) == 2


def test_no_ceiling_means_no_abort(tmp_path):
    t = CostTracker("Qwen/Qwen3-8B", ledger_path=tmp_path / "c.jsonl", show_progress=False)
    for _ in range(5):
        t.record("sample", prefill=10_000_000)
    assert t.cost_usd == pytest.approx(9.75)


def test_unpriced_model_never_blocks_a_run(tmp_path):
    t = CostTracker("some/unpriced", ledger_path=tmp_path / "c.jsonl", max_usd=0.01, show_progress=False)
    t.record("sample", prefill=10_000_000)   # cost is None -> cannot compare to ceiling
    assert t.cost_usd is None
    assert _rows(tmp_path / "c.jsonl")[0]["cumulative"]["cost_usd"] is None


def test_tracker_works_without_a_ledger():
    t = CostTracker("Qwen/Qwen3-8B", show_progress=False)
    t.record("sample", prefill=1_000_000)
    assert t.cost_usd == pytest.approx(0.195)
    assert t.summary()["ledger"] is None


def test_summary_carries_the_reconciliation_note():
    t = CostTracker("Qwen/Qwen3-8B", show_progress=False)
    s = t.summary()
    assert "billing usage" in s["pricing_note"]
    assert s["estimated_cost_usd"] == 0


# --- integration with the session --------------------------------------------
def test_sampling_records_exact_token_counts(cfg, fake_tinker, tmp_path):
    tracker = CostTracker(
        cfg.base_model, ledger_path=tmp_path / "c.jsonl", show_progress=False
    )
    s = TinkerSession(cfg, tracker=tracker)
    s._tokenizer = fake_tinker.tokenizer
    fake_tinker.responder = lambda prompt, model_path: "ANSWER"

    client = s.sampling_client(None)
    s.sample(client, "PROMPT", num_samples=2, temperature=0.0, max_tokens=16)

    row = _rows(tmp_path / "c.jsonl")[0]
    assert row["kind"] == "sample"
    assert row["prefill_tokens"] == len("PROMPT") * 2
    assert row["sample_tokens"] == len("ANSWER") * 2   # from the returned tokens, not an estimate
    assert row["num_samples"] == 2


def test_training_run_is_tagged_for_billing_reconciliation(cfg, fake_tinker):
    s = TinkerSession(cfg, user_metadata={"pipeline": "rationales-star", "run": "TS"})
    s._tokenizer = fake_tinker.tokenizer
    s.training_client()
    assert s.user_metadata["run"] == "TS"


def test_train_steps_are_priced_as_they_happen(cfg, fake_tinker, tmp_path):
    from rationales.train import train_round

    tracker = CostTracker(
        cfg.base_model, ledger_path=tmp_path / "c.jsonl", show_progress=False
    )
    s = TinkerSession(cfg, tracker=tracker)
    s._tokenizer = fake_tinker.tokenizer

    corpus = [{"prompt": "p" * 40, "completion": "c" * 40} for _ in range(4)]
    train_round(cfg, s, corpus, round_n=1, out_dir=tmp_path)

    rows = [r for r in _rows(tmp_path / "c.jsonl") if r["kind"] == "train"]
    assert len(rows) == cfg.steps_for_round(1), "one priced row per step, not one at the end"
    assert all(r["train_tokens"] > 0 for r in rows)
    assert rows[-1]["cumulative"]["cost_usd"] > rows[0]["cumulative"]["cost_usd"]


def test_budget_abort_leaves_completed_work_on_disk(cfg, fake_tinker, tmp_path):
    from rationales.train import train_round

    tracker = CostTracker(
        cfg.base_model, ledger_path=tmp_path / "c.jsonl", max_usd=1e-9, show_progress=False
    )
    s = TinkerSession(cfg, tracker=tracker)
    s._tokenizer = fake_tinker.tokenizer

    with pytest.raises(BudgetExceeded):
        train_round(cfg, s, [{"prompt": "p", "completion": "c"}], round_n=1, out_dir=tmp_path)
    assert (tmp_path / "c.jsonl").exists()
