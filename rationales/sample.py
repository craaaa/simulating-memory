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
from .resume import AppendSink, completed_trial_ids, read_jsonl
from .task import default_task

GENERATION = "generation"
RATIONALIZATION = "rationalization"


@dataclass
class SampleRow:
    """One decoded sample.

    ``trial`` is the task's item and ``digits`` the parsed answer -- both named for
    digit span, which is the only task that existed when this was written. They are
    opaque to this module: a listening item and a set of option numbers ride in the
    same two slots. The field order is positional in the tests, so it is left alone;
    ``item`` and ``answer`` are readable aliases for new code.
    """

    trial: Any
    via: str
    sample_index: int
    prompt: str
    raw: str
    reasoning: Optional[str]
    digits: Any
    accepted: bool
    parse_errors: List[str]

    @property
    def item(self) -> Any:
        return self.trial

    @property
    def answer(self) -> Any:
        return self.digits

    def to_json(self, task: Any = None, sampler_id: Optional[str] = None) -> Dict[str, Any]:
        task = task or default_task()
        return {
            "trial_id": task.item_id(self.trial),
            # Which model produced this completion. Resume reuses rows only from the
            # same sampler: within one round it never changes, but a round that is
            # re-trained and then resumed would otherwise silently reuse rows drawn
            # from the superseded checkpoint. The eval cache had this exact bug.
            **({"sampler_id": sampler_id} if sampler_id is not None else {}),
            **task.item_fields(self.trial),
            "via": self.via,
            "sample_index": self.sample_index,
            "prompt": self.prompt,
            "raw": self.raw,
            "reasoning": self.reasoning,
            **task.answer_fields(self.digits),
            "accepted": self.accepted,
            "parse_errors": self.parse_errors,
        }


def check_fewshot_ready(cfg: StarConfig, task: Any = None) -> None:
    (task or default_task()).check_ready(cfg)


def _attempt(
    trial: Any,
    prompt: str,
    via: str,
    raws: Sequence[str],
    task: Any = None,
) -> List[SampleRow]:
    """Parse every raw sample for one prompt; accept the first exact match to y_i."""
    task = task or default_task()
    rows: List[SampleRow] = []
    accepted_any = False
    for i, raw in enumerate(raws):
        reasoning, answer, errs = task.parse(raw)
        # The STaR filter. Note what it compares against: the human's own response,
        # not the correct answer. Type discipline matters here -- digit span compares
        # int list to int list, and comparing a parsed answer against a raw response
        # *string* would be silently always False, emptying the generation path.
        ok = (not accepted_any) and reasoning is not None and task.accepts(trial, answer)
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
                digits=answer,
                accepted=ok,
                parse_errors=errs,
            )
        )
    return rows


def sample_round(
    cfg: StarConfig,
    trials: Sequence[Any],
    *,
    generate: Callable[[str, int], List[str]],
    pool_path: Path,
    resume: bool = False,
    task: Any = None,
    sampler_id: Optional[str] = None,
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
    task = task or default_task()
    check_fewshot_ready(cfg, task)

    already: set = set()
    if resume:
        prior = read_jsonl(pool_path)
        if sampler_id is not None:
            stale = [r for r in prior if r.get("sampler_id") != sampler_id]
            if stale:
                # Off-policy rows: drawn from a checkpoint this round no longer uses.
                # Keeping them would train on another model's completions.
                kept = [r for r in prior if r.get("sampler_id") == sampler_id]
                AppendSink(pool_path).replace(kept)
                print(
                    f"[sample] dropped {len(stale)} pool row(s) drawn from a different "
                    f"sampler; re-sampling those trials from {sampler_id or 'base'}"
                )
                prior = kept
        already = completed_trial_ids(prior, cfg.k)
        sink = AppendSink(pool_path)
    else:
        sink = JsonlSink(pool_path)

    pending = [t for t in trials if task.item_id(t) not in already]
    skipped = len(trials) - len(pending)
    trials = pending
    workers = resolve_worker_count(max(len(trials), 1), max_parallel=cfg.max_workers)

    def _one(idx: int) -> List[SampleRow]:
        trial = trials[idx]
        # Line 3: rationale generation.
        gen_prompt = task.build_sample_prompt(trial, fewshot=cfg.use_fewshot)
        rows = _attempt(trial, gen_prompt, GENERATION, generate(gen_prompt, cfg.k), task)
        # Flushed per attempt, not per trial: if the run pauses between the generation
        # and rationalization calls, the generation result is still on disk (and the
        # trial correctly reads as not-yet-exhausted, so it is retried rather than
        # silently dropped).
        for r in rows:
            sink.append(r.to_json(task, sampler_id))

        if not any(r.accepted for r in rows):
            # Line 4: rationalization, only for problems line 3 failed. Line 4 runs
            # for all i in the paper, but line 6 keeps only failures, so restricting
            # it here is equivalent and cheaper.
            rat_prompt = task.build_rationalize_prompt(trial, fewshot=cfg.use_fewshot)
            rat_rows = _attempt(
                trial, rat_prompt, RATIONALIZATION, generate(rat_prompt, cfg.k), task
            )
            for r in rat_rows:
                sink.append(r.to_json(task, sampler_id))
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
        "stats": pool_stats(pool_path, task),
    }


def pool_stats(pool_path: Path, task: Any = None) -> Dict[str, Any]:
    """Accept counts over every row written to the pool, across pauses."""
    task = task or default_task()
    rows = read_jsonl(pool_path)
    by_trial: Dict[str, str] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        meta[r["trial_id"]] = r
        if r.get("accepted"):
            by_trial[r["trial_id"]] = r["via"]

    cells: Dict[str, Dict[str, int]] = {}
    for tid, r in meta.items():
        key = task.pool_cell_key(r)
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


def path_stats(rows: Sequence[SampleRow], task: Any = None) -> Dict[str, Any]:
    """Accept counts split by path and by human-success/human-fail cell.

    fail_side_generation_yield is the bootstrap signal: if fail trials only ever
    accept via the hint path, round over round, STaR is not bootstrapping and the fix
    is not more rounds.
    """
    task = task or default_task()
    by_trial: Dict[str, str] = {}
    for r in rows:
        if r.accepted:
            by_trial[task.item_id(r.trial)] = r.via

    trials = {task.item_id(r.trial): r.trial for r in rows}
    cells: Dict[str, Dict[str, int]] = {}
    for tid, trial in trials.items():
        key = task.pool_cell_key(task.item_fields(trial))
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


def training_pairs(
    rows: Sequence[SampleRow], cfg: StarConfig, task: Any = None
) -> List[Dict[str, Any]]:
    """Accepted rows -> (training prompt, completion) pairs.

    The training prompt is always zero-shot and hint-free, whichever path produced the
    rationale -- "as if the model had come up with the rationale without the hint".
    """
    task = task or default_task()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not r.accepted or r.reasoning is None:
            continue
        out.append(
            {
                "trial_id": task.item_id(r.trial),
                **task.pair_fields(r.trial, r.digits),
                "via": r.via,
                "prompt": task.build_sample_prompt(r.trial, fewshot=False),
                "completion": task.build_completion(r.reasoning, r.digits),
                "reasoning": r.reasoning,
            }
        )
    return out
