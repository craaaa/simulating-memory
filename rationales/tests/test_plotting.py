"""Figures render from an eval payload without a display or a live run."""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from rationales import evaluate as ev
from rationales import plotting
from rationales import prompting as P


def _payload(cfg, trials, make_digits):
    def gen(prompt, n):
        for t in trials:
            if t.stimulus_text in prompt:
                return [P.build_completion("r", make_digits(t))]
        return [""]

    base = ev.run_eval(cfg, trials, generate=gen, label="base")
    tuned = ev.run_eval(cfg, trials, generate=gen, label="tuned")
    return {"base": base["metrics"], "tuned": tuned["metrics"]}


def test_round_figures_are_written(cfg, fwd_fail, rev_fail, tmp_path):
    payload = _payload(cfg, [fwd_fail, rev_fail], lambda t: t.user_digits)
    paths = plotting.plot_round(payload, tmp_path, round_n=1)
    assert paths and all(p.exists() and p.stat().st_size > 0 for p in paths)
    names = {p.name for p in paths}
    assert "round1_forward_exact_by_span.png" in names
    assert "round1_forward_human_match_by_span.png" in names
    assert "round1_error_profile.png" in names


def test_figures_cover_both_directions(cfg, fwd_fail, rev_fail, tmp_path):
    payload = _payload(cfg, [fwd_fail, rev_fail], lambda t: t.user_digits)
    names = {p.name for p in plotting.plot_round(payload, tmp_path, round_n=2)}
    assert any("forward" in n for n in names)
    assert any("reverse" in n for n in names)
    assert all(n.startswith("round2_") for n in names)


def test_bootstrap_signal_handles_missing_values(tmp_path):
    summary = {
        "bootstrap_signal": [
            {"round": 1, "fail_side_generation_yield": None},
            {"round": 2, "fail_side_generation_yield": 0.25},
        ]
    }
    path = plotting.plot_bootstrap_signal(summary, tmp_path)
    assert path.exists() and path.stat().st_size > 0
