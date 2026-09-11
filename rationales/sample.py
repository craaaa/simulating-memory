"""STaR Algorithm 1 lines 3-4: rationale generation, then rationalization.

Line 3 decodes greedily, one sample per problem ("STaR approximates J by greedily
decoding samples of (r̂_i, ŷ_i) to reduce variance of this estimate"). cfg.k > 1 or
cfg.temperature > 0 is a deviation and is recorded as such.

Line 4 hands the model the answer as a hint. Here the "answer" is the human's own
response, so a rationalized rationale explains why a human would make that specific
error. The hint and the few-shot block are both stripped from the stored training
prompt -- "as if the model had come up with the rationale without the hint".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from bench.core.io import JsonlSink
from bench.core.parallel import map_participants, resolve_worker_count

from .config import StarConfig
from .data import HumanTrial
from .prompting import (
    build_completion,
    build_rationalize_prompt,
    build_sample_prompt,
    fewshot_has_placeholders,
    parse_rationale,
)
from .resume import AppendSink, completed_trial_ids, read_jsonl

GENERATION = "generation"
RATIONALIZATION = "rationalization"


@dataclass
class SampleRow:
    trial: HumanTrial
    via: str
    sample_index: int
    prompt: str
    raw: str
    reasoning: Optional[str]
    digits: List[int]
    accepted: bool
    parse_errors: List[str]

    def to_json(self) -> Dict[str, Any]:
        t = self.trial
        return {
            "trial_id": t.trial_id,
            "direction": t.direction,
            "participant_id": t.participant_id,
            "length": t.length,
            "sequence_index": t.sequence_index,
            "human_correct": t.correct,
            "digits_presented": t.digits,
            "expected_digits": t.expected_digits,
            "target_digits": t.user_digits,   # y_i: the human's own response
            "via": self.via,
            "sample_index": self.sample_index,
            "prompt": self.prompt,
            "raw": self.raw,
            "reasoning": self.reasoning,
            "pred_digits": self.digits,
            "accepted": self.accepted,
            "parse_errors": self.parse_errors,
        }


def check_fewshot_ready(cfg: StarConfig) -> None:
    if not cfg.use_fewshot:
        return
    unready = [d for d in cfg.directions if fewshot_has_placeholders(d)]
    if unready:
        raise RuntimeError(
            "few-shot rationale demos still contain PLACEHOLDER text for: "
            f"{', '.join(unready)}. Write them (rationales/prompts/fewshot_*.txt) "
            "or pass --no-fewshot."
        )


def _attempt(
    trial: HumanTrial,
    prompt: str,
    via: str,
    raws: Sequence[str],
) -> List[SampleRow]:
    """Parse every raw sample for one prompt; accept the first digit-exact match."""
    rows: List[SampleRow] = []
    accepted_any = False
    for i, raw in enumerate(raws):
        reasoning, digits, errs = parse_rationale(raw)
        # y_i comparison is int-list vs int-list. Comparing against the raw response
        # string would be silently always False and would empty the generation path.
        ok = (not accepted_any) and reasoning is not None and digits == trial.user_digits
        if ok:
            accepted_any = True
        rows.append(
            SampleRow(
                trial=trial,
                via=via,
                sample_index=i,
                prompt=prompt,
                raw=raw,
                reasoning=reasoning,
                digits=digits,
                accepted=ok,
                parse_errors=errs,
            )
        )
    return rows


def sample_round(
    cfg: StarConfig,
    trials: Sequence[HumanTrial],
    *,
    generate: Callable[[str, int], List[str]],
    pool_path: Path,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run lines 3-4 over the whole train split and write every sample to pool_path.

    ``generate(prompt, num_samples) -> list[str]`` is injected so this module never
    imports the Tinker SDK (and so tests can pass a stub).

    With ``resume=True`` the existing pool is kept and trials already decided are
    skipped, so a paused run does not pay twice for the same rationales. Rows are
    appended rather than truncated. Without it the pool is rewritten from scratch --
    which is what a fresh round wants, since STaR regenerates the whole split each
    round.
    """
    check_fewshot_ready(cfg)

    already: set = set()
    if resume:
        prior = read_jsonl(pool_path)
        already = completed_trial_ids(prior, cfg.k)
        sink = AppendSink(pool_path)
    else:
        sink = JsonlSink(pool_path)

    pending = [t for t in trials if t.trial_id not in already]
    skipped = len(trials) - len(pending)
    trials = pending
    workers = resolve_worker_count(max(len(trials), 1), max_parallel=cfg.max_workers)

    def _one(idx: int) -> List[SampleRow]:
        trial = trials[idx]
        # Line 3: rationale generation.
        gen_prompt = build_sample_prompt(trial, fewshot=cfg.use_fewshot)
        rows = _attempt(trial, gen_prompt, GENERATION, generate(gen_prompt, cfg.k))
        # Flushed per attempt, not per trial: if the run pauses between the generation
        # and rationalization calls, the generation result is still on disk (and the
        # trial correctly reads as not-yet-exhausted, so it is retried rather than
        # silently dropped).
        for r in rows:
            sink.append(r.to_json())

        if not any(r.accepted for r in rows):
            # Line 4: rationalization, only for problems line 3 failed. Line 4 runs
            # for all i in the paper, but line 6 keeps only failures, so restricting
            # it here is equivalent and cheaper.
            rat_prompt = build_rationalize_prompt(trial, fewshot=cfg.use_fewshot)
            rat_rows = _attempt(
                trial, rat_prompt, RATIONALIZATION, generate(rat_prompt, cfg.k)
            )
            for r in rat_rows:
                sink.append(r.to_json())
            rows += rat_rows
        return rows

    all_rows: List[SampleRow] = []
    if trials:
        for rows in map_participants(list(range(len(trials))), _one, max_workers=workers):
            all_rows.extend(rows)

    return {
        "n_trials_sampled": len(trials),
        "n_trials_skipped_already_done": skipped,
        "n_samples": len(all_rows),
        "workers": workers,
        "resumed": resume,
        "pool_path": str(pool_path),
        # Stats come from the full pool on disk, not just this invocation, so a resumed
        # round reports the same numbers an uninterrupted one would.
        "stats": pool_stats(pool_path),
    }


def pool_stats(pool_path: Path) -> Dict[str, Any]:
    """Accept counts over every row written to the pool, across pauses."""
    rows = read_jsonl(pool_path)
    by_trial: Dict[str, str] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        meta[r["trial_id"]] = r
        if r.get("accepted"):
            by_trial[r["trial_id"]] = r["via"]

    cells: Dict[str, Dict[str, int]] = {}
    for tid, r in meta.items():
        key = f"{r['direction']}:{'success' if r['human_correct'] else 'fail'}"
        cell = cells.setdefault(key, {"n": 0, "accepted": 0, GENERATION: 0, RATIONALIZATION: 0})
        cell["n"] += 1
        via = by_trial.get(tid)
        if via:
            cell["accepted"] += 1
            cell[via] += 1

    fail_n = sum(c["n"] for k, c in cells.items() if k.endswith(":fail"))
    fail_gen = sum(c[GENERATION] for k, c in cells.items() if k.endswith(":fail"))
    return {
        "cells": dict(sorted(cells.items())),
        "n_trials_in_pool": len(meta),
        "n_accepted": len(by_trial),
        "n_accepted_generation": sum(1 for v in by_trial.values() if v == GENERATION),
        "n_accepted_rationalization": sum(1 for v in by_trial.values() if v == RATIONALIZATION),
        "fail_side_generation_yield": (fail_gen / fail_n) if fail_n else None,
    }


def path_stats(rows: Sequence[SampleRow]) -> Dict[str, Any]:
    """Accept counts split by path and by (direction, human_correct).

    fail_side_generation_yield is the bootstrap signal: if fail trials only ever
    accept via the hint path, round over round, STaR is not bootstrapping and the fix
    is not more rounds.
    """
    by_trial: Dict[str, str] = {}
    for r in rows:
        if r.accepted:
            by_trial[r.trial.trial_id] = r.via

    trials = {r.trial.trial_id: r.trial for r in rows}
    cells: Dict[str, Dict[str, int]] = {}
    for tid, trial in trials.items():
        key = f"{trial.direction}:{'success' if trial.correct else 'fail'}"
        cell = cells.setdefault(
            key, {"n": 0, "accepted": 0, GENERATION: 0, RATIONALIZATION: 0}
        )
        cell["n"] += 1
        via = by_trial.get(tid)
        if via:
            cell["accepted"] += 1
            cell[via] += 1

    fail_n = sum(c["n"] for k, c in cells.items() if k.endswith(":fail"))
    fail_gen = sum(c[GENERATION] for k, c in cells.items() if k.endswith(":fail"))
    return {
        "cells": dict(sorted(cells.items())),
        "n_accepted": len(by_trial),
        "n_accepted_generation": sum(1 for v in by_trial.values() if v == GENERATION),
        "n_accepted_rationalization": sum(
            1 for v in by_trial.values() if v == RATIONALIZATION
        ),
        "fail_side_generation_yield": (fail_gen / fail_n) if fail_n else None,
    }


def training_pairs(rows: Sequence[SampleRow], cfg: StarConfig) -> List[Dict[str, Any]]:
    """Accepted rows -> (training prompt, completion) pairs.

    The training prompt is always zero-shot and hint-free, whichever path produced the
    rationale.
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not r.accepted or r.reasoning is None:
            continue
        out.append(
            {
                "trial_id": r.trial.trial_id,
                "direction": r.trial.direction,
                "length": r.trial.length,
                "human_correct": r.trial.correct,
                "via": r.via,
                "prompt": build_sample_prompt(r.trial, fewshot=False),
                "completion": build_completion(r.reasoning, r.digits),
                "reasoning": r.reasoning,
                "target_digits": r.trial.user_digits,
            }
        )
    return out
