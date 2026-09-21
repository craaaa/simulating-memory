"""Held-out evaluation against the human trials.

Read this with the ceiling in mind: on fail trials the model has to predict WHICH
error a human made, and that is genuinely stochastic (two people failing the same
8-digit sequence transpose different positions). So exact match to the human response
is the right STaR *filter* but a poor headline metric -- its ceiling on the fail half
is set by the entropy of human errors, not by model quality. The distributional
metrics (per-length curve, error-type profile) are what to read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from bench.core.io import write_json
from bench.core.parallel import map_participants, resolve_worker_count

from . import errors as err
from .config import StarConfig
from .data import HumanTrial
from .resume import AppendSink

# (A TASK_NAMES dict used to sit here mapping direction -> bench task name. Nothing ever
# read it, and it read like a task registry without being one. The registry is now
# rationales/task.py.)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def digit_span_eval_row(
    t: HumanTrial,
    reasoning: Optional[str],
    digits: List[int],
    raw: str,
    parse_errors: List[str],
) -> Dict[str, Any]:
    """One scored digit-span trial. Lifted verbatim out of ``run_eval`` so the
    digit-span task adapter can supply it through the task seam."""
    return {
        "trial_id": t.trial_id,
        "direction": t.direction,
        "length": t.length,
        "human_correct": t.correct,
        "digits_presented": t.digits,
        "expected_digits": t.expected_digits,
        "human_digits": t.user_digits,
        "pred_digits": digits,
        "reasoning": reasoning,
        "raw": raw,
        "parse_errors": parse_errors,
        "human_match": digits == t.user_digits,
        "ground_truth_correct": digits == t.expected_digits,
        "error_type_model": err.classify_error(t.expected_digits, digits),
        "error_type_human": err.classify_error(t.expected_digits, t.user_digits),
        "features_model": err.error_features(t.expected_digits, digits),
        "features_human": err.error_features(t.expected_digits, t.user_digits),
    }


def run_eval(
    cfg: StarConfig,
    trials: Sequence[Any],
    *,
    generate: Callable[[str, int], List[str]],
    label: str,
    rows_path: Optional[Path] = None,
    task: Any = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Sample once per eval trial with the zero-shot training-time prompt.

    Few-shot is off here on purpose: it is a data-generation scaffold, and the point of
    the fine-tune is that the format and the behavior are internalized.

    When ``rows_path`` is given, each scored trial is appended there as it completes and
    trials already present are skipped -- so a paused run resumes mid-eval instead of
    re-scoring (and re-paying for) what it already has.
    """
    if task is None:
        from .task import default_task

        task = default_task()

    sink = AppendSink(rows_path) if rows_path is not None else None
    done_rows = sink.rows() if sink else []

    # Rows are only reusable if the SAME model produced them. Resume keyed on trial_id
    # alone is right for a pause mid-eval and wrong the moment the checkpoint changes
    # underneath it -- continuing a round's training and re-running eval silently
    # re-reported the previous model's scores, with zero API calls to show for it.
    if sink is not None and model_id is not None:
        stale = [r for r in done_rows if r.get("model_id") != model_id]
        if stale:
            done_rows = [r for r in done_rows if r.get("model_id") == model_id]
            # Rewrite rather than append past them, or the file accumulates two models'
            # rows and the metrics average across both.
            sink.replace(done_rows)
            print(
                f"[eval:{label}] dropped {len(stale)} row(s) scored by a different "
                f"model; re-scoring those trials against {model_id}"
            )

    done_ids = {r["trial_id"] for r in done_rows}
    pending = [t for t in trials if task.item_id(t) not in done_ids]
    workers = resolve_worker_count(max(len(pending), 1), max_parallel=cfg.max_workers)

    def _one(idx: int) -> Dict[str, Any]:
        t = pending[idx]
        prompt = task.build_sample_prompt(t, fewshot=False)
        raws = generate(prompt, 1)
        raw = raws[0] if raws else ""
        reasoning, answer, parse_errors = task.parse(raw)
        row = task.eval_row(t, reasoning, answer, raw, parse_errors)
        # Stamped so a later resume can tell whose scores these are.
        return dict(row, model_id=model_id) if model_id is not None else row

    def _scored(idx: int) -> Dict[str, Any]:
        row = _one(idx)
        if sink:
            sink.append(row)
        return row

    fresh: List[Dict[str, Any]] = []
    if pending:
        fresh = list(map_participants(list(range(len(pending))), _scored, max_workers=workers))

    rows = done_rows + fresh
    return {
        "label": label,
        "rows": rows,
        "n_resumed": len(done_rows),
        "metrics": task.summarize(rows),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def subset(pred) -> List[Dict[str, Any]]:
        return [r for r in rows if pred(r)]

    def block(sel: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not sel:
            return {"n": 0}
        model_pairs = [(r["expected_digits"], r["pred_digits"]) for r in sel]
        human_pairs = [(r["expected_digits"], r["human_digits"]) for r in sel]
        model_profile = err.profile(model_pairs)
        human_profile = err.profile(human_pairs)
        return {
            "n": len(sel),
            # Primary STaR-filter metric. Low on the fail half by construction.
            "human_match_rate": _mean([float(r["human_match"]) for r in sel]),
            # Must FALL toward the human level; staying ~1.0 means nothing was learned
            # about forgetting.
            "ground_truth_accuracy": _mean(
                [float(r["ground_truth_correct"]) for r in sel]
            ),
            "human_ground_truth_accuracy": _mean(
                [float(r["human_digits"] == r["expected_digits"]) for r in sel]
            ),
            "mean_prefix_frac_model": _mean(
                [r["features_model"]["prefix_frac"] for r in sel]
            ),
            "mean_prefix_frac_human": _mean(
                [r["features_human"]["prefix_frac"] for r in sel]
            ),
            "mean_len_delta_model": _mean(
                [float(r["features_model"]["len_delta"]) for r in sel]
            ),
            "mean_len_delta_human": _mean(
                [float(r["features_human"]["len_delta"]) for r in sel]
            ),
            "error_profile_model": model_profile,
            "error_profile_human": human_profile,
            "error_profile_tv_distance": err.total_variation(model_profile, human_profile),
            "parse_failure_rate": _mean([float(bool(r["parse_errors"])) for r in sel]),
        }

    by_length: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = f"{r['direction']}:{r['length']}"
        cell = by_length.setdefault(
            key, {"n": 0, "model_exact": 0, "human_exact": 0, "human_match": 0}
        )
        cell["n"] += 1
        cell["model_exact"] += int(r["ground_truth_correct"])
        cell["human_exact"] += int(r["human_digits"] == r["expected_digits"])
        cell["human_match"] += int(r["human_match"])
    for cell in by_length.values():
        n = cell["n"]
        cell["model_exact_rate"] = cell["model_exact"] / n
        cell["human_exact_rate"] = cell["human_exact"] / n
        cell["human_match_rate"] = cell["human_match"] / n

    return {
        "overall": block(rows),
        "by_direction": {
            d: block(subset(lambda r, d=d: r["direction"] == d))
            for d in sorted({r["direction"] for r in rows})
        },
        # The interesting split: successes are near-trivial (the digits are in the
        # prompt), fails are where human-likeness actually lives.
        "human_success_trials": block(subset(lambda r: r["human_correct"])),
        "human_fail_trials": block(subset(lambda r: not r["human_correct"])),
        "by_length": dict(sorted(by_length.items())),
    }


def compare(base: Dict[str, Any], tuned: Dict[str, Any]) -> Dict[str, Any]:
    """Deltas that decide whether the round worked."""

    def get(block: Dict[str, Any], *path: str) -> Optional[float]:
        cur: Any = block
        for p in path:
            cur = (cur or {}).get(p) if isinstance(cur, dict) else None
        return cur if isinstance(cur, (int, float)) else None

    def delta(*path: str) -> Optional[float]:
        a, b = get(base["metrics"], *path), get(tuned["metrics"], *path)
        return (b - a) if (a is not None and b is not None) else None

    return {
        "human_match_rate_delta": delta("overall", "human_match_rate"),
        "human_match_rate_delta_fail_trials": delta(
            "human_fail_trials", "human_match_rate"
        ),
        "ground_truth_accuracy_delta": delta("overall", "ground_truth_accuracy"),
        "error_profile_tv_delta": delta("overall", "error_profile_tv_distance"),
        "reading": (
            "success = human_match up, ground_truth_accuracy DOWN toward the human "
            "level, error_profile_tv_distance DOWN. A modest absolute fail-side match "
            "rate is expected (see module docstring) and is not itself failure."
        ),
    }


def write_eval(out_dir: Path, name: str, payload: Dict[str, Any]) -> None:
    write_json(out_dir / f"{name}.json", payload)
