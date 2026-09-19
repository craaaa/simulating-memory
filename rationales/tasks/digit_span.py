"""Digit span as a RationaleTask.

A pure adapter. Every method delegates to the module that already implemented it --
``data``, ``select``, ``prompting``, ``evaluate``, ``plotting`` -- and no logic moved
here. The point is that pointing STaR at a second task must not be able to change what
the four committed digit-span runs would do; if a digit-span test has to change, the
seam leaked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import evaluate as ev
from .. import plotting, prompting, select as sel_mod
from ..data import HumanTrial, load_all


class DigitSpanTask:
    name = "digit_span"

    # StarConfig's defaults ARE the digit-span defaults -- they were tuned on it.
    defaults: Dict[str, Any] = {}

    def for_config(self, cfg: Any) -> "DigitSpanTask":
        # Nothing about digit span's prompts is configurable per run beyond the flags
        # StarConfig already carries (use_fewshot, directions), which the methods below
        # read from cfg directly.
        return self

    # --- data ---------------------------------------------------------------
    def load(self, cfg: Any) -> Tuple[List[HumanTrial], List[Dict[str, Any]]]:
        return load_all(cfg.directions)

    def select(self, items: Sequence[HumanTrial], *, seed: int, eval_frac: float):
        return sel_mod.select(items, seed=seed, eval_frac=eval_frac)

    def restore(self, items: Sequence[HumanTrial], report: Dict[str, Any]):
        return sel_mod.restore(items, report)

    def item_id(self, item: HumanTrial) -> str:
        return item.trial_id

    # --- prompts ------------------------------------------------------------
    def check_ready(self, cfg: Any) -> None:
        if not cfg.use_fewshot:
            return
        unready = [d for d in cfg.directions if prompting.fewshot_has_placeholders(d)]
        if unready:
            raise RuntimeError(
                "few-shot rationale demos still contain PLACEHOLDER text for: "
                f"{', '.join(unready)}. Write them (rationales/prompts/fewshot_*.txt) "
                "or pass --no-fewshot."
            )

    def build_sample_prompt(self, item: HumanTrial, *, fewshot: bool) -> str:
        return prompting.build_sample_prompt(item, fewshot=fewshot)

    def build_rationalize_prompt(self, item: HumanTrial, *, fewshot: bool) -> str:
        return prompting.build_rationalize_prompt(item, fewshot=fewshot)

    def build_completion(self, reasoning: str, answer: List[int]) -> str:
        return prompting.build_completion(reasoning, answer)

    def prompt_additions(self) -> Dict[str, str]:
        return prompting.prompt_additions()

    # --- parsing and the STaR filter ----------------------------------------
    def parse(self, text: str) -> Tuple[Optional[str], List[int], List[str]]:
        return prompting.parse_rationale(text)

    def accepts(self, item: HumanTrial, answer: List[int]) -> bool:
        # int list vs int list. Comparing a parsed list against the raw response
        # *string* is silently always False and empties the generation path.
        return answer == item.user_digits

    def answer_from_row(self, row: Dict[str, Any]) -> List[int]:
        return row.get("pred_digits") or []

    def human_answer(self, item: HumanTrial) -> List[int]:
        return item.user_digits

    def gold_answer(self, item: HumanTrial) -> List[int]:
        return item.expected_digits

    def probe_items(self, n_per_cell: int, *, seed: int) -> List[HumanTrial]:
        from ..probe import pick_trials

        return pick_trials(n_per_cell, seed=seed)

    def hint_leak(self, reasoning: str) -> Optional[str]:
        return prompting.hint_leak(reasoning)

    # --- row shapes ---------------------------------------------------------
    def item_fields(self, item: HumanTrial) -> Dict[str, Any]:
        return {
            "direction": item.direction,
            "participant_id": item.participant_id,
            "length": item.length,
            "sequence_index": item.sequence_index,
            "human_correct": item.correct,
            "digits_presented": item.digits,
            "expected_digits": item.expected_digits,
            "target_digits": item.user_digits,  # y_i: the human's own response
        }

    def answer_fields(self, answer: List[int]) -> Dict[str, Any]:
        return {"pred_digits": answer}

    def pair_fields(self, item: HumanTrial, answer: List[int]) -> Dict[str, Any]:
        return {
            "direction": item.direction,
            "length": item.length,
            "human_correct": item.correct,
            "target_digits": item.user_digits,
        }

    def cell_key(self, row: Dict[str, Any]) -> str:
        return f"{row['direction']}:{row['length']}:{'success' if row['human_correct'] else 'fail'}"

    def pool_cell_key(self, row: Dict[str, Any]) -> str:
        return f"{row['direction']}:{'success' if row['human_correct'] else 'fail'}"

    # --- evaluation ---------------------------------------------------------
    def eval_row(
        self,
        item: HumanTrial,
        reasoning: Optional[str],
        answer: List[int],
        raw: str,
        parse_errors: List[str],
    ) -> Dict[str, Any]:
        return ev.digit_span_eval_row(item, reasoning, answer, raw, parse_errors)

    def summarize(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return ev.summarize(rows)

    def plot_round(self, payload: Dict[str, Any], figures_dir: Path, *, round_n: int) -> List[Path]:
        return plotting.plot_round(payload, figures_dir, round_n=round_n)


TASK = DigitSpanTask()
