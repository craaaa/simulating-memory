"""Score every model under ``runs/`` against the human data and write a .txt
summary of per-task accuracy + humanlikeness per prompting condition.

Usage
-----

    # Score everything under runs/prompting + runs/compactor (default):
    python src/score.py

    # Score one specific model directory (e.g. a new run produced by run.py):
    python src/score.py --model-dir runs/my-model

    # Override output path:
    python src/score.py --out tables/my-table.txt

Layout expected per model directory (matches what ``bench`` produces and what
``run.py`` writes):

    <model_dir>/
        tasks/
            digit_span_forward.jsonl          # TaskPr / HumPr / MemPr rows
            digit_span_reverse.jsonl
            nback.jsonl
            word_recognition.jsonl
            variable_mapping.jsonl
            factual_qa.jsonl
            narrative_qa.jsonl
            semantic_story_recall.jsonl
            map_task.jsonl
            craft_task.jsonl
            wm_digit_span_forward.jsonl       # Compactor rows (optional)
            wm_digit_span_reverse.jsonl
            ...

Rows can carry the prompting conditions ``C1`` (TaskPr), ``C2`` (HumPr),
``C3`` (MemPr); compactor rows under ``wm_<task>.jsonl`` use ``C2``.

The output table has one row per (task, condition) and columns
``mean_score`` (per-participant normalized score averaged over the model)
and ``humanlikeness`` (= 1 - W_1 between the model's score distribution
and the human distribution loaded from ``runs/human/``).

One exception to "per participant": since 2026-09-29 the ``nback`` unit is the
(participant, n-level) cell on BOTH sides, with the human lead-in trials dropped. See
the block comment above ``nback_human_by_level``. n-back humanlikeness computed before
that date is NOT comparable with these figures; ``nback_human_scores_legacy_pooled``
reproduces the old shape for reading historical records.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HUMAN_ROOT = RUNS / "human"
PROMPTING_ROOT = RUNS / "prompting"
COMPACTOR_ROOT = RUNS / "compactor"

TASKS = [
    "digit_span_forward",
    "digit_span_reverse",
    "nback",
    "word_recognition",
    "variable_mapping",
    "factual_qa",
    "narrative_qa",
    "semantic_story_recall",
    "map_task",
    "craft_task",
]

TASK_DISPLAY = {
    "digit_span_forward": "Digit Span",
    "digit_span_reverse": "Reverse Digit Span",
    "nback": "N-Back",
    "word_recognition": "Word Recognition",
    "variable_mapping": "Variable Mapping",
    "factual_qa": "Factual QA",
    "narrative_qa": "Narrative QA",
    "semantic_story_recall": "Narrative Free Recall",
    "map_task": "Map Task",
    "craft_task": "Craft Task",
}

HUMAN_TASK_DIR = {
    "digit_span_forward": "working-memory-digit-span",
    "digit_span_reverse": "working-memory-reverse-digit-span",
    "nback": "working-memory-nback",
    "word_recognition": "working-memory-word-recognition",
    "variable_mapping": "working-memory-variable-mapping",
    "factual_qa": "factual-qa",
    "narrative_qa": "narrative-qa",
    "semantic_story_recall": "semantic-memory-story-recall",
    "map_task": "procedure-memory-map-task",
    "craft_task": "procedure-memory-craft-task",
}

# Per-task denominator so that 1.0 = perfect. For the two SURVIVAL_TASKS below it is
# instead the censoring point: 1.0 means "never stopped".
TASK_DENOM = {
    "digit_span_forward": 20.0,
    "digit_span_reverse": 20.0,
    "nback": 1.0,
    "word_recognition": 100.0,
    "variable_mapping": 16.0,   # PINNED survival scale, NOT the question count -- see below
    "factual_qa": 10.0,
    "narrative_qa": 10.0,
    "semantic_story_recall": 1.0,
    "map_task": 15.0,
    "craft_task": 15.0,
}

# ---------------------------------------------------------------------------
# Survival length, for the two tasks whose human protocol stops on a fixed error count
# ---------------------------------------------------------------------------
# DECISION 2026-09-29 [USER]: "Use survival length on both sides."
#
# Neither of these two tasks has a human accuracy, and no denominator choice creates one.
# Both human protocols end a session after a fixed number of mistakes, so the number of
# errors is pinned by the protocol and the only free quantity is WHEN the last one landed:
#
#   word_recognition  `trialsCompleted - correctResponses == 3` for 53 of 53 human records,
#                     so the old `correctResponses / 100` was identically (n_survived - 3)/100.
#                     Human n_survived: mean 34.49, median 32, min 4, max 102.
#   variable_mapping  exactly one error in 152 of 152 human records, always the participant's
#                     LAST question, so the old `correct_count / 10` was identically
#                     (n_survived - 1)/10. Human n_survived: mean 4.99, median 5, min 2, max 16.
#
# So both sides now report survival length -- the number of items the participant was
# presented before stopping -- divided by TASK_DENOM, which for these two tasks is a fixed
# scale and clipping point rather than a perfect score.
#
# THE variable_mapping DENOMINATOR IS PINNED AT 16 AND MUST NOT TRACK THE QUESTION COUNT.
# 16 is the human maximum: over 152 records the most questions anyone answered is 16, and no
# human record is censored, because every one ends in an error.
#
# Why pinned rather than "however many questions bench asks". Wasserstein-1 scales with the
# denominator, so a denominator that follows the schedule length would make humanlikeness rise
# by roughly 0.02 for a longer task at identical behaviour -- a metric that improves when you
# lengthen the task. The scale is therefore fixed once, at the human maximum, and survival
# beyond it clips to 1.0.
#
# History: this was 10 until 2026-09-29, matching a `bench` schedule of 10 questions that
# censored the model at 10 while leaving humans uncensored. `bench` now asks 20
# (`variable_mapping.N_QUESTIONS`), so model censoring moves from 10 to 16 and the ~5 of 150
# rows that used to pile on the ceiling are resolved. **Humanlikeness computed at denominator
# 10 is NOT comparable with 16** -- rescore from the run dirs rather than comparing across it.
#
# word_recognition is nearly symmetric and needs no such pinning: the human list is 100 long
# (one record reports 102, clipped here) and the model's is 100.
SURVIVAL_TASKS = ("word_recognition", "variable_mapping")


def _survival(n_presented: float | None, denom: float) -> float | None:
    """Survival length as a fraction of the censoring point, clipped to 1.0."""
    if n_presented is None:
        return None
    return min(float(n_presented) / denom, 1.0)


PROMPT_CONDITIONS = ["C1", "C2", "C3"]
CONDITION_DISPLAY = {
    "C1": "TaskPr",
    "C2": "HumPr",
    "C3": "MemPr",
    "compactor": "Compactor",
}


# ---------- Per-participant scoring ------------------------------------------


def _best_span_from_trials(trials: list[dict[str, Any]]) -> int:
    by_span: dict[int, list[bool]] = defaultdict(list)
    for t in trials:
        length = t.get("length")
        if length is None:
            continue
        by_span[int(length)].append(bool(t.get("correct", False)))
    best = 0
    for span in sorted(by_span.keys()):
        if any(by_span[span]):
            best = span
        else:
            break
    return best


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _score_human_record(task: str, rec: dict[str, Any]) -> float | None:
    summary = rec.get("summary") or {}
    payload = rec.get("payload") or {}
    denom = TASK_DENOM[task]

    if task in ("digit_span_forward", "digit_span_reverse"):
        trials = payload.get("trials") or []
        if trials:
            return _best_span_from_trials(trials) / denom
        best = summary.get("bestSpan")
        return None if best is None else float(best) / denom

    if task == "nback":
        # LEGACY SHAPE, SUPERSEDED 2026-09-29 [USER, "Option D"]. This is the pooled,
        # lead-in-included human n-back score: one value per participant, averaged over
        # all three n-levels, counting the n lead-in trials of each block. `human_scores`
        # no longer calls it -- see `nback_human_by_level` and the block comment there for
        # the two defects (M6, M7) it embodies. It is kept, unchanged, only so the n-back
        # figures already recorded in meta_harness/logs/evolution_summary.jsonl remain
        # reproducible; reach it through `nback_human_scores_legacy_pooled()`.
        trials = payload.get("trials") or []
        scored = [t for t in trials if t.get("phase") != "practice"]
        if scored:
            correct = sum(1 for t in scored if t.get("correct"))
            return correct / len(scored)
        pct = summary.get("accuracyPercent")
        return None if pct is None else float(pct) / 100.0

    if task == "word_recognition":
        # Survival length: words presented before the 3-strike stop. See SURVIVAL_TASKS.
        return _survival(summary.get("trialsCompleted"), denom)

    if task == "variable_mapping":
        # Survival length: questions presented before the stop-on-first-error.
        questions = payload.get("questions") or []
        if questions:
            return _survival(len(questions), denom)
        # No per-question payload. `turnsCompleted` counts turns, not questions, and
        # `bestScore` is the old correct-count; neither is a survival length, so refuse
        # rather than mix quantities inside one distribution.
        return None

    if task in ("factual_qa", "narrative_qa"):
        total = summary.get("totalQuestions") or 10
        correct = summary.get("correctAnswers")
        return None if correct is None else float(correct) / float(total)

    if task == "semantic_story_recall":
        # Paper text says BLEU but the web app's BLEU is ~0 across participants
        # (heavy smoothing); embeddingSimilarity is the metric that lines up
        # with the published Figure 2 visually.
        sim = summary.get("embeddingSimilarity")
        return None if sim is None else float(sim)

    if task in ("map_task", "craft_task"):
        total = summary.get("totalQuestions") or 0
        correct = summary.get("totalCorrect") or 0
        return None if total <= 0 else float(correct) / float(total)

    return None


# ---------------------------------------------------------------------------
# N-Back: per (participant, n-level), human lead-in trials dropped
# ---------------------------------------------------------------------------
# DECISION 2026-09-29 [USER], "Option D": fix BOTH denominator and granularity at once.
# Before this, the two sides of the n-back comparison were unmatched in two ways.
#
#   M6  UNMATCHED DENOMINATORS. A human block holds exactly 14 non-practice trials
#       INCLUDING the n lead-in trials, `trial` indexed from 1 -- verified for 49 of 49
#       records that carry a `level` field, at each of n=1,2,3. The model is presented its
#       n lead-in letters separately (`buffer_letters`, responses in
#       `model_parsed_buffer`) and they are excluded from `acc_over_14`. So the human was
#       scored over 14 trials, 3 of them lead-in at n=3, and the model over 14 genuine
#       trials.
#
#       The human lead-in trials are not measurements. Over the 318 of them (53 records x
#       (1+2+3)) accuracy is flat at 0.9182 at every level while real accuracy falls
#       0.9492 -> 0.8553 -> 0.7496, and 112 of the 318 are logged `target: true`, which
#       cannot happen when no letter n back exists. Worked case, the first record's n=3
#       block, letters X W M X V D Z G X Z D X W D: trial 3 (letter M) is logged
#       `target: true`, the participant answered "target", and it is scored
#       `correct: true` -- but the block starts at trial 1, so there is no letter three
#       back. Those trials are dropped here, on the side that has them.
#
#   M7  UNMATCHED GRANULARITY. The model is one observation per (participant, n_level) --
#       150 rows -- while the human was one pooled observation over all three of that
#       participant's levels. The model's distribution therefore spanned the whole level
#       effect while the human's averaged it away, and W_1 between them was inflated by
#       that alone. Both sides are now scored per (participant, level).
#
# THE MODEL SIDE NEEDS NO CHANGE. `llm_scores("nback", ...)` already emits one value per
# (participant, level) from `acc_over_14`, which already excludes the lead-in. Option D is
# entirely a human-side fix.
#
# Measured effect on the three `meta_harness/runs/iter12stage2` arms, humanlikeness
# (= 1 - W_1, in [0,1]), mean over arms, with `wasserstein_1d` below:
#
#   legacy shape (pooled human, lead-in included)   0.9340
#   M6 only  (pooled human, lead-in excluded)       0.9419
#   M7 only  (per-level both sides, lead-in in)     0.9627
#   M6 + M7  (this code)                            0.9633
#
# M7 dominates: +0.0287 on its own. M6's increment on top of it is +0.0006, which is
# smaller than the measured run-to-run spread for n-back (0.0061, score_candidate.
# RUN_TO_RUN_SPREAD) and whose SIGN is not identified across specifications -- it is
# -0.0020 if the human pool is restricted to the 49 level-carrying records. M6 is a
# correctness fix, not a scoring gain, and must not be reported as one.
#
# AND M6 MAKES THE n=3 DISTRIBUTION MATCH SLIGHTLY WORSE while closing the mean gap:
# dropping the lead-in widens the human n=3 spread (sd 0.1492 -> 0.1657 at the 49-record
# pool) while the model sits at sd 0.0970, so closing the mean gap widens the dispersion
# gap. Any statement of the n=3 mean gap has to carry that caveat.
NBACK_LEVELS = (1, 2, 3)
_NBACK_PRACTICE_BLOCK_PREFIX = "training"


def _nback_level(trial: dict[str, Any]) -> int | None:
    """The n-level of one human trial, or None if it is not a scored 1/2/3-back trial.

    Reads `level` when present and falls back to the leading integer of `block`
    ("2-back" -> 2). The fallback is not cosmetic: 4 of the 57 human records carry
    `level: null` on all 42 of their scored trials, and those same 4 also carry
    `phase: null` rather than `phase: "scored"`. Their blocks are named 1-back / 2-back /
    3-back with exactly 14 trials indexed 1..14 each, structurally identical to the other
    49, so the level is recoverable and they belong in the sample. This is the same field
    source `meta_harness/error_structure.nback_human_trials` already uses for the M3
    measure, and it is why both agree on 318 lead-in trials and 112 bogus `target: true`.

    FILTER ON `phase != "practice"`, NEVER ON `phase == "scored"`: the latter silently
    drops those 4 records and lands the human n at 49 instead of 53 -- a number that looks
    plausible and matches the pre-decision measurements, so nothing would flag it.
    """
    block = str(trial.get("block") or "")
    if trial.get("phase") == "practice" or block.startswith(_NBACK_PRACTICE_BLOCK_PREFIX):
        return None
    lvl = trial.get("level")
    if isinstance(lvl, int) and lvl in NBACK_LEVELS:
        return lvl
    m = re.match(r"(\d+)", block)
    if m and int(m.group(1)) in NBACK_LEVELS:
        return int(m.group(1))
    return None


def _nback_is_leadin(trial: dict[str, Any], level: int) -> bool:
    """True for the first `level` trials of a block, where no letter n back exists.

    `trial` is 1-based, so trials 1..n are lead-in. A record with no `trial` index is
    treated as non-lead-in rather than guessed at; none of the 53 usable records is in
    that state (all have `trial` 1..14 at every level).
    """
    idx = trial.get("trial")
    return idx is not None and int(idx) <= level


def nback_human_by_level(*, exclude_leadin: bool = True) -> dict[int, list[float]]:
    """Per-participant n-back accuracy at each n-level, in reading order of the records.

    One list per level, each entry the proportion of that participant's scored trials at
    that level answered correctly. Practice trials are always dropped; the n lead-in
    trials of each block are dropped when ``exclude_leadin`` (the default, and the
    Option D shape). Returns 53 values per level on the released human data -- the 4
    remaining records of the 57 carry an empty `payload` AND an empty `summary`, so they
    have no trials and no accuracy to score on any shape, legacy included.
    """
    folder = HUMAN_ROOT / HUMAN_TASK_DIR["nback"]
    acc: dict[int, list[float]] = {n: [] for n in NBACK_LEVELS}
    if not folder.is_dir():
        return acc
    for path in sorted(folder.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                rec = json.load(f)
        except json.JSONDecodeError:
            continue
        by_level: dict[int, list[bool]] = defaultdict(list)
        for t in (rec.get("payload") or {}).get("trials") or []:
            lvl = _nback_level(t)
            if lvl is None:
                continue
            if exclude_leadin and _nback_is_leadin(t, lvl):
                continue
            by_level[lvl].append(bool(t.get("correct")))
        for lvl, vals in by_level.items():
            if vals:
                acc[lvl].append(sum(vals) / len(vals))
    return acc


def nback_human_scores(*, exclude_leadin: bool = True) -> np.ndarray:
    """The n-back human distribution `human_scores("nback")` returns: one value per
    (participant, level), levels concatenated in order 1, 2, 3."""
    by_level = nback_human_by_level(exclude_leadin=exclude_leadin)
    out: list[float] = []
    for n in NBACK_LEVELS:
        out.extend(by_level[n])
    return np.asarray(out, dtype=np.float64)


def nback_human_scores_legacy_pooled() -> np.ndarray:
    """SUPERSEDED 2026-09-29. The pre-Option-D human n-back distribution: one value per
    participant, pooled over levels, lead-in trials INCLUDED.

    Only for reading historical records. Every n-back humanlikeness in
    meta_harness/logs/evolution_summary.jsonl written before 2026-09-29 was computed
    against this, so it must keep producing the same numbers; do not "fix" it.
    """
    return _human_scores_from_records("nback")


def _human_scores_from_records(task: str) -> np.ndarray:
    """`_score_human_record` over every record in a task's human folder."""
    folder = HUMAN_ROOT / HUMAN_TASK_DIR[task]
    out: list[float] = []
    if not folder.is_dir():
        return np.asarray(out, dtype=np.float64)
    for path in sorted(folder.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                rec = json.load(f)
        except json.JSONDecodeError:
            continue
        s = _score_human_record(task, rec)
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=np.float64)


def human_scores(task: str) -> np.ndarray:
    """The human per-participant score distribution humanlikeness is measured against.

    n-back is the one task whose unit is NOT the participant: it is the
    (participant, n-level) cell, matching the granularity the model rows already have.
    See the `nback_human_by_level` block comment (decision 2026-09-29, "Option D").
    """
    if task == "nback":
        return nback_human_scores(exclude_leadin=True)
    return _human_scores_from_records(task)


def _score_llm_row(task: str, row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics") or {}
    denom = TASK_DENOM[task]

    if task in ("digit_span_forward", "digit_span_reverse"):
        # Aggregated upstream via _digit_span_participant_scores.
        ex = metrics.get("exact")
        return None if ex is None else float(ex)

    if task == "nback":
        acc = metrics.get("accuracy")
        if acc is not None:
            return float(acc)
        acc = row.get("acc_over_14")
        if acc is None:
            acc = row.get("acc_over_answered")
        return None if acc is None else float(acc)

    if task == "word_recognition":
        # Survival length: trials presented before the 3-strike stop. `trials_presented` is
        # written by the post-2026-09-29 one-word-per-turn harness; before that the harness
        # answered all 100 at once and `score_game` truncated `per_trial` at the third error,
        # which lands on the same index. `metrics.first_error_at` is NOT usable here -- it is
        # misnamed and holds the third error's index, not the first.
        n = row.get("trials_presented")
        if n is None:
            per_trial = row.get("per_trial")
            n = len(per_trial) if per_trial else None
        return _survival(n, denom)

    if task == "variable_mapping":
        # Survival length: the first error's position, or the full question count if the run
        # never erred (censored). Here `first_error_at` really is the first error: verified
        # against an independent re-grade from `parsed_answers` + `options` + `correct_city`,
        # 150 of 150 rows agreeing, 0 unparsed answers.
        fe = metrics.get("first_error_at")
        if fe:
            return _survival(fe, denom)
        n_q = metrics.get("n_questions") or len(row.get("questions") or []) or None
        return _survival(n_q, denom)

    if task in ("factual_qa", "narrative_qa"):
        acc = metrics.get("accuracy")
        if acc is None:
            correct = metrics.get("correct")
            total = metrics.get("total")
            if correct is not None and total:
                acc = float(correct) / float(total)
        return None if acc is None else float(acc)

    if task == "semantic_story_recall":
        sim = metrics.get("embeddingSimilarity")
        return None if sim is None else float(sim)

    if task in ("map_task", "craft_task"):
        acc = metrics.get("accuracy")
        if acc is not None:
            return float(acc)
        correct = metrics.get("correct") or metrics.get("totalCorrect")
        total = metrics.get("total") or metrics.get("totalQuestions")
        if correct is not None and total:
            return float(correct) / float(total)
        return None

    return None


def _digit_span_participant_scores(rows: Iterable[dict[str, Any]], condition: str) -> list[float]:
    # Group rows by (participant_id, sequence_index_bucket). The prompting
    # protocol runs N participants x 2 sequences per span; the compactor
    # protocol typically runs 1 participant x M sequences per span. To keep
    # both protocols comparable we group on (pid, seq_idx) when a single
    # participant carries many sequences, so each seq becomes one observation.
    rows = list(rows)
    rows = [r for r in rows
            if (r.get("condition_id") or r.get("condition")) == condition]
    pid_set = {int(r.get("participant_id", 0)) for r in rows}
    seq_set = {int(r.get("sequence_index", 1)) for r in rows}
    # If there's only one participant but many sequences, split on sequence_index.
    split_by_seq = len(pid_set) <= 1 and len(seq_set) > 2

    buckets: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pid = int(r.get("participant_id", 0))
        seq = int(r.get("sequence_index", 1)) if split_by_seq else 0
        span = r.get("span_length")
        ex = (r.get("metrics") or {}).get("exact")
        if span is None or ex is None:
            continue
        buckets[(pid, seq)][int(span)].append(float(ex))
    out: list[float] = []
    for by_span in buckets.values():
        best = 0
        for s in sorted(by_span):
            if any(v == 1.0 for v in by_span[s]):
                best = s
            else:
                break
        out.append(best / 20.0)
    return out


def _resolve_jsonl(task: str, model_dir: Path, *, compactor: bool) -> Path | None:
    fname = f"wm_{task}.jsonl" if compactor else f"{task}.jsonl"
    p = model_dir / "tasks" / fname
    if not compactor and task == "variable_mapping":
        rp = model_dir / "tasks" / "variable_mapping.reparsed.jsonl"
        if rp.is_file():
            p = rp
    return p if p.is_file() else None


def llm_scores(task: str, model_dir: Path, condition: str) -> np.ndarray:
    """Per-participant scores from a model directory.

    ``condition`` is one of: ``C1``, ``C2``, ``C3`` (prompting) or
    ``compactor``. For the compactor case we read ``wm_<task>.jsonl``
    where the synthetic condition id is ``C2``.
    """
    compactor = condition == "compactor"
    path = _resolve_jsonl(task, model_dir, compactor=compactor)
    if path is None:
        return np.asarray([], dtype=np.float64)
    cond_id = "C2" if compactor else condition
    rows = _read_jsonl(path)
    if task in ("digit_span_forward", "digit_span_reverse"):
        return np.asarray(
            _digit_span_participant_scores(rows, cond_id), dtype=np.float64
        )
    out = []
    for r in rows:
        row_cond = r.get("condition_id") or r.get("condition")
        if row_cond != cond_id:
            continue
        s = _score_llm_row(task, r)
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=np.float64)


# ---------- Statistics -------------------------------------------------------


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    if grid.size == 1:
        return float(abs(np.mean(x) - np.mean(y)))
    edges = np.concatenate([[grid[0]], (grid[:-1] + grid[1:]) / 2.0, [grid[-1]]])
    total = 0.0
    for i in range(len(edges) - 1):
        mid = 0.5 * (edges[i] + edges[i + 1])
        fx = np.searchsorted(x, mid, side="right") / x.size
        fy = np.searchsorted(y, mid, side="right") / y.size
        total += abs(fx - fy) * (edges[i + 1] - edges[i])
    return float(total)


def humanlikeness(human: np.ndarray, model: np.ndarray) -> float:
    w = wasserstein_1d(human, model)
    return float("nan") if np.isnan(w) else 1.0 - w


# ---------- Model directory discovery ----------------------------------------


def discover_models(*, prompting_root: Path = PROMPTING_ROOT,
                    compactor_root: Path = COMPACTOR_ROOT,
                    explicit: list[Path] | None = None) -> list[tuple[str, Path, bool]]:
    """Return [(display_name, model_dir, is_compactor)] entries.

    If ``explicit`` is provided, treat each as a self-contained model dir
    that may carry both prompting jsonls and ``wm_*`` (compactor) jsonls.
    """
    found: list[tuple[str, Path, bool]] = []
    if explicit:
        for d in explicit:
            d = Path(d)
            if not d.is_dir():
                continue
            has_prompt = any((d / "tasks").glob("*.jsonl")) if (d / "tasks").is_dir() else False
            if has_prompt:
                found.append((d.name, d, False))
            has_compactor = (
                (d / "tasks").is_dir()
                and any((d / "tasks").glob("wm_*.jsonl"))
            )
            if has_compactor:
                found.append((d.name, d, True))
        return found

    if prompting_root.is_dir():
        for d in sorted(prompting_root.iterdir()):
            if d.is_dir() and (d / "tasks").is_dir():
                found.append((d.name, d, False))
    if compactor_root.is_dir():
        for d in sorted(compactor_root.iterdir()):
            if d.is_dir() and (d / "tasks").is_dir():
                found.append((d.name, d, True))
    return found


# ---------- Reporting --------------------------------------------------------


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "  n/a "
    return f"{v: .3f}"


def build_table(model_entries: list[tuple[str, Path, bool]],
                human_dists: dict[str, np.ndarray]) -> str:
    """Render a fixed-width table with per-(task, condition) rows."""
    out: list[str] = []
    out.append("# Per-model task scores and humanlikeness")
    out.append("")
    out.append("Each block reports, for one model and one prompting condition, "
               "per-task mean normalized score (1.0 = perfect) and humanlikeness "
               "(= 1 - W_1 on per-participant score distributions vs. humans).")
    out.append("")

    # First, per-task human means for reference
    out.append("## Humans (reference)")
    out.append("")
    out.append(f"{'task':<26}  {'n':>5}  {'mean_score':>10}")
    out.append("-" * 46)
    for t in TASKS:
        h = human_dists[t]
        out.append(f"{TASK_DISPLAY[t]:<26}  {h.size:>5}  {_fmt(np.mean(h) if h.size else float('nan')):>10}")
    out.append("")

    for display_name, model_dir, is_compactor in model_entries:
        suffix = " [compactor]" if is_compactor else ""
        out.append(f"## {display_name}{suffix}")
        out.append(f"_source: {model_dir}_")
        out.append("")

        conditions = ["compactor"] if is_compactor else PROMPT_CONDITIONS
        for cond in conditions:
            out.append(f"### {CONDITION_DISPLAY[cond]} ({cond})")
            out.append("")
            out.append(
                f"{'task':<26}  {'n':>5}  {'mean_score':>10}  {'humanlikeness':>13}"
            )
            out.append("-" * 64)
            for t in TASKS:
                m = llm_scores(t, model_dir, cond)
                h = human_dists[t]
                mean_s = float(np.mean(m)) if m.size else float("nan")
                hl = humanlikeness(h, m) if m.size else float("nan")
                out.append(
                    f"{TASK_DISPLAY[t]:<26}  {m.size:>5}  {_fmt(mean_s):>10}  {_fmt(hl):>13}"
                )
            out.append("")
        out.append("")

    return "\n".join(out) + "\n"


# ---------- CLI --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help="Path to a model run directory (repeatable). If omitted, scores "
             "every model found under runs/prompting and runs/compactor.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "src" / "scores.txt",
        help="Output .txt path (default: src/scores.txt).",
    )
    args = parser.parse_args()

    explicit = [Path(p) for p in args.model_dir] if args.model_dir else None
    entries = discover_models(explicit=explicit)
    if not entries:
        raise SystemExit("No model directories found. Pass --model-dir <path>.")

    human_dists = {t: human_scores(t) for t in TASKS}

    text = build_table(entries, human_dists)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
