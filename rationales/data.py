"""Load human digit-span trials as STaR problems (x_i, y_i).

y_i is the participant's actual typed response, not the correct answer -- the one
deliberate deviation from STaR (Zelikman et al. 2022).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .config import REPO_ROOT

HUMAN_ROOT = REPO_ROOT / "runs" / "human"

# Same mapping as src/score.py:HUMAN_TASK_DIR.
HUMAN_TASK_DIR: Dict[str, str] = {
    "forward": "working-memory-digit-span",
    "reverse": "working-memory-reverse-digit-span",
}


@dataclass(frozen=True)
class HumanTrial:
    """One human digit-span trial.

    ``user_digits`` and ``expected_digits`` are int lists so they compare directly
    against ``bench.tasks.digit_span_forward.parse_pressed_digits`` output. Comparing a
    parsed int list against a raw response *string* is silently always False, which
    would empty the generation path and look like a modeling failure.

    For reverse trials ``expected``/``user_response`` are already in response
    (reversed) order as recorded by the task page -- nothing re-reverses them here.
    """

    direction: str
    participant_id: str
    run_id: str
    trial: int
    length: int
    sequence_index: int
    digits: List[int]           # presented sequence, presentation order
    expected_digits: List[int]  # correct response
    user_digits: List[int]      # what the human actually typed
    expected: str
    user_response: str
    correct: bool

    @property
    def trial_id(self) -> str:
        return f"{self.direction}:{self.participant_id}:{self.run_id}:t{self.trial}"

    @property
    def stimulus_text(self) -> str:
        """Matches bench's stimulus wording exactly, incl. list formatting."""
        return f"The digits are the following: {self.digits}"


def _digits(s: str) -> List[int]:
    return [int(c) for c in (s or "").strip() if c.isdigit()]


def _load_run(path: Path, direction: str) -> List[HumanTrial]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "completed":
        return []
    trials = (rec.get("payload") or {}).get("trials") or []
    if not trials:
        return []

    out: List[HumanTrial] = []
    for t in trials:
        expected = str(t["expectedResponse"])
        user = str(t["userResponse"])
        stored_correct = bool(t["correct"])
        derived_correct = expected.strip() == user.strip()
        if stored_correct != derived_correct:
            raise ValueError(
                f"{path.name} trial {t.get('trial')}: stored correct={stored_correct} "
                f"but expected={expected!r} user={user!r}"
            )
        out.append(
            HumanTrial(
                direction=direction,
                participant_id=str(rec["participant_id"]),
                run_id=str(rec["run_id"]),
                trial=int(t["trial"]),
                length=int(t["length"]),
                sequence_index=int(t["sequenceIndex"]),
                digits=_digits(str(t["presentedSequence"])),
                expected_digits=_digits(expected),
                user_digits=_digits(user),
                expected=expected,
                user_response=user,
                correct=stored_correct,
            )
        )
    return out


def load_human_trials(direction: str) -> tuple[List[HumanTrial], Dict[str, object]]:
    """Load one direction's trials plus a provenance/exclusion report.

    Excludes ``.claude/worktrees`` (a byte-identical duplicate tree -- a recursive
    glob there double-counts every trial) and de-dupes participants with more than
    one run by keeping the earliest ``started_at``.
    """
    task_dir = HUMAN_TASK_DIR[direction]
    root = HUMAN_ROOT / task_dir
    if not root.is_dir():
        raise FileNotFoundError(f"missing human data dir: {root}")

    # Non-recursive glob on purpose: keeps the worktree copy out.
    paths = sorted(root.glob("run-*.json"))

    by_participant: Dict[str, List[tuple[str, Path]]] = {}
    skipped_incomplete: List[str] = []
    for p in paths:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("status") != "completed" or not (rec.get("payload") or {}).get("trials"):
            skipped_incomplete.append(p.name)
            continue
        pid = str(rec["participant_id"])
        by_participant.setdefault(pid, []).append((str(rec.get("started_at") or ""), p))

    trials: List[HumanTrial] = []
    dropped_dupes: List[str] = []
    for pid, runs in sorted(by_participant.items()):
        runs.sort(key=lambda r: r[0])
        keep = runs[0][1]
        for _, extra in runs[1:]:
            dropped_dupes.append(f"{pid}:{extra.name}")
        trials.extend(_load_run(keep, direction))

    report = {
        "direction": direction,
        "files_seen": len(paths),
        "skipped_incomplete": skipped_incomplete,
        "dropped_duplicate_runs": dropped_dupes,
        "participants": len(by_participant),
        "trials": len(trials),
        "correct": sum(1 for t in trials if t.correct),
        "incorrect": sum(1 for t in trials if not t.correct),
    }
    return trials, report


def load_all(directions: Sequence[str]) -> tuple[List[HumanTrial], List[Dict[str, object]]]:
    trials: List[HumanTrial] = []
    reports: List[Dict[str, object]] = []
    for d in directions:
        t, r = load_human_trials(d)
        trials.extend(t)
        reports.append(r)
    return trials, reports


def by_length(trials: Sequence[HumanTrial]) -> Dict[tuple, Dict[str, int]]:
    """(direction, length) -> {n, n_correct}. Digit-span accuracy is length-dominated,
    so any balance or eval claim needs this table, not a pooled mean."""
    out: Dict[tuple, Dict[str, int]] = {}
    for t in trials:
        cell = out.setdefault((t.direction, t.length), {"n": 0, "n_correct": 0})
        cell["n"] += 1
        cell["n_correct"] += int(t.correct)
    return dict(sorted(out.items()))
