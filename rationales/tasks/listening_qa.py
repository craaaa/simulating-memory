"""Listening QA as a RationaleTask.

y_i is the set of options the participant actually endorsed. An item is one
(participant, topic, question); see rationales/listening/data.py for why.

``sibling_context`` is the one knob here that is not in StarConfig. The prompt shows
the participant's answers to the other four questions of the same passage, which
individuates the person being simulated -- but the question bank builds each topic's
fourth question as a 2x2 combination of two earlier questions' content, so for those
items the siblings let a model DERIVE the answer instead of simulating a memory. The
flag exists so the base model can be evaluated both ways and the difference read off
before any of it is attributed to the fine-tune.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..listening import evaluate as lev
from ..listening import plotting as lplot
from ..listening import prompting as lp
from ..listening import select as lsel
from ..listening.data import ListeningItem, load_items


class ListeningQATask:
    name = "listening_qa"

    # StarConfig defaults that were tuned on digit span and are wrong here. An explicit
    # CLI flag still wins; see rationales/cli.py:_cfg.
    #
    #   max_seq_length  headroom, not a fix for a known overflow. Measured over 300
    #                   items with the real Qwen tokenizer: the zero-shot training
    #                   prompt runs 698-945 tokens, so even with a generous completion
    #                   a datum is ~1120 and 2048 would in fact have held. 4096 costs
    #                   nothing (training is billed on actual datum length, not the
    #                   cap) and buys room for a longer stimulus or a fifth sibling.
    #                   It matters because build_datum truncates from the RIGHT, so an
    #                   overflow would remove the answer line the loss covers rather
    #                   than failing.
    #   steps_1         60 steps x batch 8 = 480 examples, tuned for a ~560-item
    #                   corpus. Here the corpus is ~7x that, so 60 steps is well under
    #                   one epoch. 200 is a starting point; set the real value off the
    #                   round-1 loss curve.
    defaults = {"max_seq_length": 4096, "steps_1": 200}

    def __init__(self, *, sibling_context: bool = True) -> None:
        self.sibling_context = sibling_context

    def for_config(self, cfg: Any) -> "ListeningQATask":
        want = getattr(cfg, "sibling_context", True)
        return self if want == self.sibling_context else ListeningQATask(sibling_context=want)

    # --- data ---------------------------------------------------------------
    def load(self, cfg: Any) -> Tuple[List[ListeningItem], List[Dict[str, Any]]]:
        items, report = load_items()
        return items, [report]

    def select(self, items: Sequence[ListeningItem], *, seed: int, eval_frac: float):
        return lsel.select(items, seed=seed, eval_frac=eval_frac)

    def restore(self, items: Sequence[ListeningItem], report: Dict[str, Any]):
        return lsel.restore(items, report)

    def item_id(self, item: ListeningItem) -> str:
        return item.item_id

    # --- prompts ------------------------------------------------------------
    def check_ready(self, cfg: Any) -> None:
        if not cfg.use_fewshot:
            return
        if lp.fewshot_has_placeholders():
            raise RuntimeError(
                "few-shot rationale demos still contain PLACEHOLDER text: "
                f"{lp.FEWSHOT_PATH}. Write them or pass --no-fewshot."
            )

    def build_sample_prompt(self, item: ListeningItem, *, fewshot: bool) -> str:
        return lp.build_sample_prompt(
            item, fewshot=fewshot, sibling_context=self.sibling_context
        )

    def build_rationalize_prompt(self, item: ListeningItem, *, fewshot: bool) -> str:
        return lp.build_rationalize_prompt(
            item, fewshot=fewshot, sibling_context=self.sibling_context
        )

    def build_completion(self, reasoning: str, answer: List[int]) -> str:
        return lp.build_completion(reasoning, answer)

    def prompt_additions(self) -> Dict[str, str]:
        return dict(lp.prompt_additions(), sibling_context=str(self.sibling_context))

    # --- parsing and the STaR filter ----------------------------------------
    def parse(self, text: str) -> Tuple[Optional[str], List[int], List[str]]:
        return lp.parse_rationale(text)

    def accepts(self, item: ListeningItem, answer: List[int]) -> bool:
        # Set equality, not list equality: option order in the answer line carries no
        # meaning, and a list comparison would reject "3,1" for "1,3".
        return set(answer or []) == set(item.endorsed)

    def answer_from_row(self, row: Dict[str, Any]) -> List[int]:
        return row.get("pred_options") or []

    def human_answer(self, item: ListeningItem) -> List[int]:
        return sorted(item.endorsed)

    def gold_answer(self, item: ListeningItem) -> List[int]:
        return sorted(item.gold)

    def probe_items(self, n_per_cell: int, *, seed: int) -> List[ListeningItem]:
        """A few items per (level x human-success/fail) cell.

        Level is the cell that matters for the probe: the distractor passage is where
        a rationale has the most to explain, and control is where the prompt is most
        likely to look fine while doing nothing.
        """
        import random

        rng = random.Random(seed)
        items, _ = load_items()
        out: List[ListeningItem] = []
        for level in ("control", "repeat_short", "repeat_long", "distractor"):
            for correct in (True, False):
                cell = sorted(
                    (i for i in items if i.level == level and i.correct is correct),
                    key=lambda i: i.item_id,
                )
                out.extend(rng.sample(cell, min(n_per_cell, len(cell))))
        return out

    def hint_leak(self, reasoning: str) -> Optional[str]:
        return lp.hint_leak(reasoning)

    # --- row shapes ---------------------------------------------------------
    def item_fields(self, item: ListeningItem) -> Dict[str, Any]:
        return {
            "respondent_id": item.respondent_id,
            "topic": item.topic,
            "level": item.level,
            "question_id": item.question_id,
            "human_correct": item.correct,
            "gold_options": sorted(item.gold),
            "target_options": sorted(item.endorsed),  # y_i: the human's own selection
        }

    def answer_fields(self, answer: List[int]) -> Dict[str, Any]:
        return {"pred_options": sorted(answer or [])}

    def pair_fields(self, item: ListeningItem, answer: List[int]) -> Dict[str, Any]:
        return {
            "topic": item.topic,
            "level": item.level,
            "question_id": item.question_id,
            "human_correct": item.correct,
            "target_options": sorted(item.endorsed),
        }

    def cell_key(self, row: Dict[str, Any]) -> str:
        return (
            f"{row['topic']}:{row['level']}:{row['question_id']}:"
            f"{'success' if row['human_correct'] else 'fail'}"
        )

    def pool_cell_key(self, row: Dict[str, Any]) -> str:
        # Must split success from fail: the fail side is where the bootstrap signal is
        # read. Level is the other axis worth watching, since the distractor passage is
        # where rationalization is most likely to be carrying the round.
        return f"{row['level']}:{'success' if row['human_correct'] else 'fail'}"

    # --- evaluation ---------------------------------------------------------
    def eval_row(
        self,
        item: ListeningItem,
        reasoning: Optional[str],
        answer: List[int],
        raw: str,
        parse_errors: List[str],
    ) -> Dict[str, Any]:
        return lev.eval_row(item, reasoning, answer, raw, parse_errors)

    def summarize(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return lev.summarize(rows)

    def plot_round(self, payload: Dict[str, Any], figures_dir: Path, *, round_n: int) -> List[Path]:
        return lplot.plot_round(payload, figures_dir, round_n=round_n)


TASK = ListeningQATask()
