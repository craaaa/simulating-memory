"""The prompt probe: format checking with a stubbed model, no network."""
from __future__ import annotations

import pytest

from rationales import probe as pr
from rationales import prompting as P


def test_pick_trials_covers_every_cell():
    trials = pr.pick_trials(2, seed=42)
    cells = {(t.direction, t.correct) for t in trials}
    assert cells == {("forward", True), ("forward", False), ("reverse", True), ("reverse", False)}
    assert all(4 <= t.length <= 8 for t in trials)
    assert len({t.trial_id for t in trials}) == len(trials)


def test_pick_trials_is_deterministic():
    a = [t.trial_id for t in pr.pick_trials(2, seed=42)]
    b = [t.trial_id for t in pr.pick_trials(2, seed=42)]
    c = [t.trial_id for t in pr.pick_trials(2, seed=7)]
    assert a == b and a != c


def test_probe_runs_both_prompt_kinds_per_trial():
    seen = []

    def generate(prompt):
        seen.append(prompt)
        return P.build_completion("because", [1, 2, 3])

    report = pr.probe(model="stub/model", n_per_cell=1, fewshot=False, generate=generate)
    assert report["n_trials"] == 4
    assert report["n_calls"] == 8 == len(seen)
    assert sum("actually responded" in p for p in seen) == 4
    kinds = [r["kind"] for r in report["rows"]]
    assert kinds.count("sample") == kinds.count("rationalize") == 4


def test_probe_flags_well_formed_output():
    def generate(prompt):
        return P.build_completion("they lost the tail", [4, 5])

    report = pr.probe(model="stub/model", n_per_cell=1, fewshot=False, generate=generate)
    assert report["summary"]["well_formed_rate"] == 1.0
    assert report["summary"]["median_reasoning_chars"] > 0
    assert report["summary"]["parse_error_counts"] == {}


def test_probe_flags_malformed_output():
    report = pr.probe(
        model="stub/model", n_per_cell=1, fewshot=False, generate=lambda p: "press <<4>>."
    )
    s = report["summary"]
    assert s["well_formed_rate"] == 0.0
    assert s["parse_error_counts"] == {"no_reasoning_block": 8}
    assert s["rationalize_hit_rate"] == 0.0


def test_probe_detects_a_working_hint():
    """Echoing the hinted response must show up as a rationalize hit, and the plain
    sample path must not get the same credit."""
    import re

    def generate(prompt):
        if "actually responded" in prompt:
            hinted = re.search(
                r"The human actually responded:\n((?:press <<\d>>\.\n?)+)", prompt
            ).group(1)
            digits = [int(d) for d in re.findall(r"press <<(\d)>>\.", hinted)]
            return P.build_completion("the tail decayed", digits)
        return P.build_completion("a guess", [0])

    report = pr.probe(model="stub/model", n_per_cell=1, fewshot=False, generate=generate)
    s = report["summary"]
    assert s["rationalize_hit_rate"] == 1.0
    assert s["rationalize_hit_rate_fail_trials"] == 1.0
    assert s["sample_human_match_fail_trials"] == 0.0


def test_probe_reports_hint_leaks():
    def generate(prompt):
        return P.build_completion("The human actually responded that way.", [1])

    report = pr.probe(model="stub/model", n_per_cell=1, fewshot=False, generate=generate)
    assert report["summary"]["rationalize_hint_leak_rate"] == 1.0
    # leak detection applies to the hinted path only
    assert all(r["hint_leak"] is None for r in report["rows"] if r["kind"] == "sample")


def test_probe_refuses_to_run_on_unwritten_fewshot_demos(tmp_path, monkeypatch):
    """The probe spends real money rendering the real generation prompt, so it is
    gated exactly as sampling is -- without this it would report a format verdict for
    a prompt nobody intends to run."""
    from rationales.listening import prompting as lp
    from rationales.tasks.listening_qa import ListeningQATask

    path = tmp_path / "fewshot_listening.txt"
    path.write_text(lp.load_fewshot().replace("they", "PLACEHOLDER", 1), encoding="utf-8")
    monkeypatch.setattr(lp, "FEWSHOT_PATH", path)

    with pytest.raises(RuntimeError, match="PLACEHOLDER"):
        pr.probe(
            model="stub/model",
            n_per_cell=1,
            fewshot=True,
            generate=lambda p: "",
            task=ListeningQATask(),
        )


def test_probe_never_produces_a_training_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "PKG_DIR", tmp_path)
    report = pr.probe(
        model="stub/model", n_per_cell=1, fewshot=False,
        generate=lambda p: P.build_completion("r", [1]),
    )
    path = pr.write_report(report, timestamp="20260911T000000Z")
    assert path.exists()
    written = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"20260911T000000Z_stub_model.json"}
    assert "accepted.jsonl" not in written and "sample_pool.jsonl" not in written


def test_report_carries_the_off_policy_caveat():
    report = pr.probe(
        model="stub/model", n_per_cell=1, fewshot=False, generate=lambda p: "x"
    )
    assert "off-policy" in report["caveat"].lower()
    assert "nothing here is used as training data" in report["caveat"].lower()
    assert report["backend"] == "openrouter"


def test_summarize_on_no_rows_is_safe():
    s = pr.summarize([])
    assert s["well_formed_rate"] is None
    assert s["parse_error_counts"] == {}
    assert s["median_reasoning_chars"] is None
