"""The task seam.

STaR's control flow -- generate, rationalize the failures, filter, train from base,
eval tuned against base -- is task-independent. Everything that *is* task-specific
(what an item is, how it is rendered into a prompt, how a completion is parsed, and
above all what counts as matching the human) lives behind this protocol.

The one thing worth stating plainly: ``accepts`` is the STaR filter, and in this
package it compares against the HUMAN's response, not the correct answer. A task that
implements ``accepts`` as "matches ground truth" is doing standard STaR; the tasks here
deliberately do not.

Callers pass ``task=`` explicitly. Where it is omitted, ``default_task()`` supplies the
digit-span implementation, so every pre-existing call site keeps its exact behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class RationaleTask(Protocol):
    """What STaR needs to know about a task.

    ``Item`` is whatever the task's loader returns (a ``HumanTrial``, a
    ``ListeningItem``, ...); this module never inspects it. ``Answer`` is whatever
    ``parse`` extracts from a completion and ``accepts`` compares -- a digit list, a set
    of option numbers. Both are opaque here and must be JSON-serializable.
    """

    name: str

    # --- data ---------------------------------------------------------------
    def load(self, cfg: Any) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Every item, plus per-source provenance reports for run_config.json."""

    def select(self, items: Sequence[Any], *, seed: int, eval_frac: float) -> Any:
        """Build D and split it. Returns a ``select.Selection``."""

    def restore(self, items: Sequence[Any], report: Dict[str, Any]) -> Any:
        """Rebuild a Selection from a saved selection.json, so every round sees the
        identical D."""

    def item_id(self, item: Any) -> str:
        """Stable per-item key. Used for resume, for pinning the split, and as the
        ``trial_id`` column in every jsonl the run writes."""

    # --- prompts ------------------------------------------------------------
    def check_ready(self, cfg: Any) -> None:
        """Raise if the task cannot run yet (e.g. few-shot demos still PLACEHOLDER)."""

    def build_sample_prompt(self, item: Any, *, fewshot: bool) -> str:
        """Algorithm 1 line 3. With ``fewshot=False`` this is also the training and
        eval prompt -- zero-shot and hint-free."""

    def build_rationalize_prompt(self, item: Any, *, fewshot: bool) -> str:
        """Algorithm 1 line 4: the same problem with the human's answer given as a hint."""

    def build_completion(self, reasoning: str, answer: Any) -> str:
        """The training target: reasoning block followed by the answer."""

    def prompt_additions(self) -> Dict[str, str]:
        """Every string this task adds on top of the shared bench conditions, recorded
        with the run so a prompt change is never silent."""

    # --- parsing and the STaR filter ----------------------------------------
    def parse(self, text: str) -> Tuple[Optional[str], Any, List[str]]:
        """(reasoning, answer, parse_errors) from one raw completion."""

    def accepts(self, item: Any, answer: Any) -> bool:
        """The STaR filter: does this answer match the human's own response?"""

    def answer_from_row(self, row: Dict[str, Any]) -> Any:
        """Inverse of ``answer_fields``, for rebuilding a written pool."""

    def human_answer(self, item: Any) -> Any:
        """y_i -- what the participant actually did. What ``accepts`` compares against."""

    def gold_answer(self, item: Any) -> Any:
        """The correct answer. Never the filter target here; used only to report how
        far the model has moved away from being right and toward being human."""

    def probe_items(self, n_per_cell: int, *, seed: int) -> List[Any]:
        """A handful of items per cell for the off-policy format probe."""

    # --- row shapes ---------------------------------------------------------
    def item_fields(self, item: Any) -> Dict[str, Any]:
        """Task-specific columns describing the problem: stimulus, the human's own
        response (y_i), and whatever the corpus is balanced on."""

    def answer_fields(self, answer: Any) -> Dict[str, Any]:
        """Task-specific columns describing what the model produced."""

    def pair_fields(self, item: Any, answer: Any) -> Dict[str, Any]:
        """Task-specific columns for a training pair."""

    def cell_key(self, row: Dict[str, Any]) -> str:
        """Corpus-balance cell for filter_stats.json."""

    def pool_cell_key(self, row: Dict[str, Any]) -> str:
        """Coarser cell for sample-pool accept rates. This is where the bootstrap
        signal is read, so it must split human-success from human-fail items."""

    # --- evaluation ---------------------------------------------------------
    def eval_row(
        self,
        item: Any,
        reasoning: Optional[str],
        answer: Any,
        raw: str,
        parse_errors: List[str],
    ) -> Dict[str, Any]:
        """One scored held-out item."""

    def summarize(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Metrics over scored rows.

        Must emit the four key paths ``evaluate.compare`` reads --
        ``overall.human_match_rate``, ``human_fail_trials.human_match_rate``,
        ``overall.ground_truth_accuracy`` and ``overall.error_profile_tv_distance`` --
        or every headline delta in eval.json comes out null, silently.
        """

    def plot_round(self, payload: Dict[str, Any], figures_dir: Path, *, round_n: int) -> List[Path]:
        """Per-round figures. Never raises into the round; star.py catches, but a task
        should not rely on that."""


_REGISTRY: Dict[str, str] = {
    "digit_span": "rationales.tasks.digit_span:TASK",
    "listening_qa": "rationales.tasks.listening_qa:TASK",
}

DEFAULT_TASK = "digit_span"


def resolve(name: str) -> RationaleTask:
    """Look up a task by name. Imported lazily: loading the listening task pulls in
    yaml and the listening item bank, which a digit-span run has no reason to need."""
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; known tasks: {', '.join(sorted(_REGISTRY))}"
        ) from None
    module_name, attr = target.split(":")
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def default_task() -> RationaleTask:
    """The task assumed by call sites that predate this seam."""
    return resolve(DEFAULT_TASK)


def task_names() -> List[str]:
    return sorted(_REGISTRY)
