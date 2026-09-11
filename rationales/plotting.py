"""Figures for a STaR round: per-length curves and error-type profiles.

Both figures put the human curve on the same axes as the model's, because the claim
being tested is "closer to human", not "higher".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from bench.core.plotting import save_fig

from . import errors as err


def _lengths(by_length: Dict[str, Any], direction: str) -> List[int]:
    return sorted(
        int(k.split(":")[1]) for k in by_length if k.split(":")[0] == direction
    )


def plot_by_length(
    eval_payload: Dict[str, Any], figures_dir: Path, *, round_n: int
) -> List[Path]:
    """Exact-match-by-span for base, tuned, and human on one axis, per direction.

    This is the figure that detects the length shortcut: under global 1:1 balancing,
    fails concentrate at long spans, so a model that has only learned "long -> fail"
    shows a step rather than the human's gradual decline.
    """
    import matplotlib.pyplot as plt

    out: List[Path] = []
    base_len = eval_payload["base"]["by_length"]
    tuned_len = eval_payload["tuned"]["by_length"]
    directions = sorted({k.split(":")[0] for k in tuned_len})

    for direction in directions:
        lengths = _lengths(tuned_len, direction)
        if not lengths:
            continue
        keys = [f"{direction}:{n}" for n in lengths]

        fig = plt.figure()
        plt.plot(lengths, [base_len[k]["model_exact_rate"] for k in keys if k in base_len],
                 marker="o", label="base model")
        plt.plot(lengths, [tuned_len[k]["model_exact_rate"] for k in keys],
                 marker="s", label=f"STaR round {round_n}")
        plt.plot(lengths, [tuned_len[k]["human_exact_rate"] for k in keys],
                 marker="^", linestyle="--", label="human")
        plt.xlabel("Span length")
        plt.ylabel("Exact match rate (vs. correct answer)")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{direction} digit span: accuracy by span (round {round_n})")
        plt.legend()
        path = figures_dir / f"round{round_n}_{direction}_exact_by_span.png"
        save_fig(fig, path)
        out.append(path)

        fig = plt.figure()
        plt.plot(lengths, [tuned_len[k]["human_match_rate"] for k in keys], marker="s")
        plt.xlabel("Span length")
        plt.ylabel("Match to the human's own response")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{direction}: human-match by span (round {round_n})")
        path = figures_dir / f"round{round_n}_{direction}_human_match_by_span.png"
        save_fig(fig, path)
        out.append(path)

    return out


def plot_error_profile(
    eval_payload: Dict[str, Any], figures_dir: Path, *, round_n: int
) -> Path:
    """Model vs. human error-type distribution.

    Read this rather than the exact-match rate: which specific error a human makes is
    stochastic, so the distribution is the learnable target, not the individual token.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tuned = eval_payload["tuned"]["overall"]
    base = eval_payload["base"]["overall"]
    cats = [c for c in err.CATEGORIES if
            tuned["error_profile_model"].get(c) or tuned["error_profile_human"].get(c)
            or base["error_profile_model"].get(c)]

    x = np.arange(len(cats))
    width = 0.27
    fig = plt.figure(figsize=(max(6, len(cats) * 1.1), 4))
    plt.bar(x - width, [base["error_profile_model"].get(c, 0) for c in cats], width, label="base")
    plt.bar(x, [tuned["error_profile_model"].get(c, 0) for c in cats], width,
            label=f"STaR round {round_n}")
    plt.bar(x + width, [tuned["error_profile_human"].get(c, 0) for c in cats], width,
            label="human")
    plt.xticks(x, cats, rotation=30, ha="right")
    plt.ylabel("Share of trials")
    tv = tuned.get("error_profile_tv_distance")
    plt.title(f"Error types, round {round_n} (TV vs. human: {tv:.3f})" if tv is not None
              else f"Error types, round {round_n}")
    plt.legend()
    path = figures_dir / f"round{round_n}_error_profile.png"
    save_fig(fig, path)
    return path


def plot_bootstrap_signal(summary: Dict[str, Any], figures_dir: Path) -> Path:
    """fail_side_generation_yield per round.

    Flat near zero across rounds means rationalization is carrying everything and
    nothing is transferring to the unhinted prompt -- the stop/pivot signal.
    """
    import matplotlib.pyplot as plt

    points = [(p["round"], p["fail_side_generation_yield"] or 0.0)
              for p in summary["bootstrap_signal"]]
    fig = plt.figure()
    plt.plot([r for r, _ in points], [y for _, y in points], marker="o")
    plt.xlabel("STaR round")
    plt.ylabel("Fail-trial rationales from the unhinted path")
    plt.ylim(-0.02, 1.02)
    plt.title("Bootstrap signal")
    path = figures_dir / "bootstrap_signal.png"
    save_fig(fig, path)
    return path


def plot_round(eval_payload: Dict[str, Any], figures_dir: Path, *, round_n: int) -> List[Path]:
    paths = plot_by_length(eval_payload, figures_dir, round_n=round_n)
    paths.append(plot_error_profile(eval_payload, figures_dir, round_n=round_n))
    return paths
