"""Prompt sanity probe: do the sampling prompts elicit well-formed rationales?

This is a **format check**, not data generation. It answers three questions on a
handful of trials:

  1. Does the model emit a `<reasoning>` block followed by parseable `press <<D>>.`
     lines? (format validity)
  2. On the plain generation prompt, does it ever reproduce the human's response?
     (expected: often on success trials, rarely on fail trials)
  3. On the rationalization prompt, does it reproduce the hinted response?
     (if not, the hint wording is not working and STaR has no fail-side signal)

It runs through OpenRouter because it needs no Tinker checkpoint and costs cents.
That makes it **off-policy** -- a different serving stack, and usually a different
model, from the one being trained -- so its accept rates are NOT an estimate of round-1
yield and its output is never written to a corpus. Nothing here feeds training; the
probe writes a single timestamped JSON report under `out/probe/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bench.core.io import run_timestamp, write_json

from .config import PKG_DIR
from .data import HumanTrial, load_all
from .prompting import (
    build_rationalize_prompt,
    build_sample_prompt,
    hint_leak,
    parse_rationale,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def pick_trials(n_per_cell: int = 2, *, seed: int = 42) -> List[HumanTrial]:
    """A few trials from each (direction x success/fail) cell, biased toward mid spans.

    Very short trials are trivial and very long ones are rare, so neither tells you
    much about whether the prompt works.
    """
    import random

    rng = random.Random(seed)
    trials, _ = load_all(["forward", "reverse"])
    out: List[HumanTrial] = []
    for direction in ("forward", "reverse"):
        for correct in (True, False):
            cell = [
                t
                for t in trials
                if t.direction == direction and t.correct is correct and 4 <= t.length <= 8
            ]
            cell.sort(key=lambda t: t.trial_id)
            out.extend(rng.sample(cell, min(n_per_cell, len(cell))))
    return out


def _row(trial: HumanTrial, kind: str, prompt: str, raw: str) -> Dict[str, Any]:
    reasoning, digits, parse_errors = parse_rationale(raw)
    return {
        "trial_id": trial.trial_id,
        "direction": trial.direction,
        "length": trial.length,
        "human_correct": trial.correct,
        "kind": kind,
        # Tracked separately from parse failures: an empty completion means the
        # provider returned nothing (budget spent on hidden reasoning, filtered, etc.),
        # which is a serving problem, not a prompt problem.
        "empty_response": not (raw or "").strip(),
        "well_formed": bool(reasoning) and bool(digits),
        "parse_errors": parse_errors,
        "reasoning": reasoning,
        "reasoning_chars": len(reasoning or ""),
        "pred_digits": digits,
        "human_digits": trial.user_digits,
        "expected_digits": trial.expected_digits,
        "matches_human": digits == trial.user_digits,
        "matches_ground_truth": digits == trial.expected_digits,
        # Length is judged against y_i (the human's response), not the correct answer:
        # faithfully reproducing a human's truncation has the "wrong" length by design,
        # and scoring it against gold made a success look like a failure.
        "right_length": len(digits) == len(trial.user_digits),
        "right_length_vs_gold": len(digits) == len(trial.expected_digits),
        "hint_leak": hint_leak(reasoning or "") if kind == "rationalize" else None,
        "prompt_chars": len(prompt),
        "raw": raw,
    }


# Thinking models spend their completion budget on hidden reasoning tokens before
# emitting any content -- observed on qwen3.8-flash, where 373 of 480 completion tokens
# were reasoning and longer thinks returned an EMPTY string. That looks exactly like a
# broken prompt in the report, so thinking is disabled by default: the probe is meant
# to measure the prompt, not the provider's reasoning budget.
NO_THINKING_EXTRA_BODY: Dict[str, Any] = {
    "enable_thinking": False,
    "reasoning": {"enabled": False},
}


def probe(
    *,
    model: str,
    n_per_cell: int = 2,
    fewshot: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 42,
    thinking: bool = False,
    generate=None,
) -> Dict[str, Any]:
    """Run both prompt kinds over a few trials and summarize format validity.

    ``generate(prompt) -> str`` can be injected for tests; otherwise an OpenRouter
    client is built.
    """
    if generate is None:
        from bench.core.llm_openai import OpenAIChatLLM

        llm = OpenAIChatLLM(
            model=model,
            base_url=OPENROUTER_BASE_URL,
            extra_body=None if thinking else NO_THINKING_EXTRA_BODY,
        )

        def generate(prompt: str) -> str:  # noqa: F811
            return llm.generate(
                prompt, temperature=temperature, max_tokens=max_tokens, top_p=1.0
            ).text

        usage_source = llm
    else:
        usage_source = None

    trials = pick_trials(n_per_cell, seed=seed)
    rows: List[Dict[str, Any]] = []

    for t in trials:
        gen_prompt = build_sample_prompt(t, fewshot=fewshot)
        rows.append(_row(t, "sample", gen_prompt, generate(gen_prompt)))

        rat_prompt = build_rationalize_prompt(t, fewshot=fewshot)
        rows.append(_row(t, "rationalize", rat_prompt, generate(rat_prompt)))

    report = {
        "model": model,
        "backend": "openrouter",
        "fewshot": fewshot,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "n_trials": len(trials),
        "n_calls": len(rows),
        "summary": summarize(rows),
        "rows": rows,
        "caveat": (
            "Off-policy format check only: a different serving stack (and usually a "
            "different model) from the Tinker base being trained. Accept rates here "
            "do not estimate round-1 yield, and nothing here is used as training data."
        ),
    }
    if usage_source is not None and hasattr(usage_source, "usage_summary"):
        report["llm_usage"] = usage_source.usage_summary()
    return report


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def frac(sel, pred) -> Optional[float]:
        sel = list(sel)
        return (sum(1 for r in sel if pred(r)) / len(sel)) if sel else None

    sample = [r for r in rows if r["kind"] == "sample"]
    rat = [r for r in rows if r["kind"] == "rationalize"]

    return {
        # (1) does the format hold up at all
        "well_formed_rate": frac(rows, lambda r: r["well_formed"]),
        "well_formed_rate_sample": frac(sample, lambda r: r["well_formed"]),
        "well_formed_rate_rationalize": frac(rat, lambda r: r["well_formed"]),
        "empty_response_rate": frac(rows, lambda r: r["empty_response"]),
        "right_length_rate": frac(rows, lambda r: r["right_length"]),
        "right_length_rate_rationalize": frac(rat, lambda r: r["right_length"]),
        "median_reasoning_chars": _median([r["reasoning_chars"] for r in rows]),
        "parse_error_counts": _counts(e for r in rows for e in r["parse_errors"]),
        # (2) unhinted behavior, split the way it matters
        "sample_human_match_success_trials": frac(
            (r for r in sample if r["human_correct"]), lambda r: r["matches_human"]
        ),
        "sample_human_match_fail_trials": frac(
            (r for r in sample if not r["human_correct"]), lambda r: r["matches_human"]
        ),
        "sample_ground_truth_rate": frac(sample, lambda r: r["matches_ground_truth"]),
        # (3) is the hint actually working
        "rationalize_hit_rate": frac(rat, lambda r: r["matches_human"]),
        "rationalize_hit_rate_fail_trials": frac(
            (r for r in rat if not r["human_correct"]), lambda r: r["matches_human"]
        ),
        "rationalize_hint_leak_rate": frac(rat, lambda r: bool(r["hint_leak"])),
        "reading": (
            "Check empty_response_rate FIRST: anything above 0 is a serving problem "
            "(a thinking model spending its budget on hidden reasoning, say), not a "
            "prompt problem -- raise --max-tokens or keep thinking disabled. Then "
            "well_formed_rate near 1.0 means the prompt works. A low "
            "rationalize_hit_rate_fail_trials is the real warning sign: the hint is "
            "being ignored, so STaR would get no fail-side training signal."
        ),
    }


def _median(xs: Sequence[int]) -> Optional[float]:
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    return float(xs[mid]) if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _counts(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items()))


def write_report(report: Dict[str, Any], *, timestamp: Optional[str] = None) -> Path:
    ts = timestamp or run_timestamp()
    slug = report["model"].replace("/", "_")
    path = PKG_DIR / "out" / "probe" / f"{ts}_{slug}.json"
    write_json(path, report)
    return path
