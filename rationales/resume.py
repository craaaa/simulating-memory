"""Pause/resume bookkeeping.

A round is a sequence of phases, each of which appends its results to disk as it goes:

    sample -> filter -> train -> eval_base -> eval_tuned -> done

``progress.json`` records where the run got to. On resume every phase skips work that is
already on disk -- sampled trials keep their rationales, training restarts from the last
saved optimizer state, eval keeps the rows it already scored -- so nothing already paid
for is paid for twice.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PHASES = ("sample", "filter", "train", "eval_base", "eval_tuned", "done")


class AppendSink:
    """Thread-safe JSONL appender.

    Unlike ``bench.core.io.JsonlSink``, this does NOT truncate on construction -- that
    is the whole point: a resumed phase must add to the rows a paused run already paid
    for, not replace them.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    def rows(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        return read_jsonl(self.path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class RunState:
    """Where the run got to, and what it can pick back up from."""

    round: int = 1
    phase: str = "sample"
    train_step: int = 0                      # steps already applied this round
    train_state_path: Optional[str] = None   # tinker:// full state (weights + optimizer)
    sampler_path: Optional[str] = None       # checkpoint the NEXT round samples from
    checkpoint: Optional[str] = None         # this round's sampler checkpoint
    completed_rounds: List[int] = field(default_factory=list)
    paused: bool = False
    pause_reason: Optional[str] = None
    spend_usd_at_pause: Optional[float] = None

    @classmethod
    def load(cls, path: Path) -> "RunState":
        if not Path(path).is_file():
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def advance(self, phase: str) -> None:
        self.phase = phase

    def start_round(self, n: int) -> None:
        self.round = n
        self.phase = "sample"
        self.train_step = 0
        self.train_state_path = None
        self.checkpoint = None

    def finish_round(self) -> None:
        if self.round not in self.completed_rounds:
            self.completed_rounds.append(self.round)
        self.sampler_path = self.checkpoint
        self.phase = "done"

    def is_done_with(self, phase: str) -> bool:
        """True when ``phase`` already ran to completion in this round."""
        return PHASES.index(self.phase) > PHASES.index(phase)


def completed_trial_ids(pool_rows: List[Dict[str, Any]], k: int) -> set:
    """Trials whose sampling is finished and must not be re-paid for.

    A trial is finished when a rationale was accepted, or when both the generation and
    the rationalization path have been tried to exhaustion (k samples each). Anything
    else -- a trial cut off mid-flight by a pause -- is re-sampled.
    """
    accepted = {r["trial_id"] for r in pool_rows if r.get("accepted")}

    tried: Dict[str, Dict[str, int]] = {}
    for r in pool_rows:
        tried.setdefault(r["trial_id"], {}).setdefault(r["via"], 0)
        tried[r["trial_id"]][r["via"]] += 1

    exhausted = {
        tid
        for tid, vias in tried.items()
        if vias.get("generation", 0) >= k and vias.get("rationalization", 0) >= k
    }
    return accepted | exhausted
