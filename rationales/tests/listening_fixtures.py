"""Synthetic listening items.

Deliberately not built from the real bank: a fixture that reads responses.csv would
couple every prompt and filter test to a gitignored build artifact, and would go
missing on a fresh checkout. The tests that must see real data say so and load it
themselves (and skip when it is absent).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from rationales.listening.data import ListeningItem, SiblingAnswer

# Deliberately NOT the few-shot practice passage: tests assert that the training prompt
# contains no few-shot text, and a shared stimulus would make that assertion unfalsifiable.
PASSAGE = (
    "The Skerry Light stands on a granite stack off the northern headland. It was lit "
    "in 1904, and its lamp turns on a bath of mercury so that one push moves it."
)

OPTIONS: Dict[int, str] = {
    1: "It stands on a granite stack",
    2: "It was lit in 1904",
    3: "Its lamp turns on a bath of mercury",
    4: "It is built of stacked timber",
    5: "None of the above",
}

LABELS = {
    1: ("true", "paraphrase"),
    2: ("true", "exact"),
    3: ("true", "exact"),
    4: ("false_interference", "paraphrase"),
    5: ("false_plain", "paraphrase"),
}


def make_item(
    *,
    respondent: str = "r0001",
    topic: str = "lighthouses",
    level: str = "control",
    question_id: str = "QV01",
    gold: Sequence[int] = (1, 2, 3),
    endorsed: Sequence[int] = (1, 2, 3),
    siblings: Optional[List[SiblingAnswer]] = None,
    group_id: Optional[str] = "1",
    position: Optional[int] = 1,
) -> ListeningItem:
    if siblings is None:
        siblings = [
            SiblingAnswer(
                question_id="QV02",
                question="How is the light reached? Select all that apply.",
                options={1: "By helicopter", 2: "On foot at low tide", 5: "None of the above"},
                endorsed=[1],
            )
        ]
    return ListeningItem(
        respondent_id=respondent,
        topic=topic,
        level=level,
        question_id=question_id,
        position=position,
        group_id=group_id,
        passage=PASSAGE,
        question="Which of the following describe the Skerry Light? Select all that apply.",
        options=dict(OPTIONS),
        gold=list(gold),
        endorsed=list(endorsed),
        other_answers=list(siblings),
        option_labels=dict(LABELS),
    )


def make_cohort(n_respondents: int = 12, *, groups: int = 4) -> List[ListeningItem]:
    """A small population: every respondent answers two questions on one topic, with
    group and level assigned round-robin so a stratified split has something to work
    with."""
    levels = ["control", "repeat_short", "repeat_long", "distractor"]
    items: List[ListeningItem] = []
    for i in range(n_respondents):
        rid = f"r{i:04d}"
        level = levels[i % len(levels)]
        group = str(i % groups + 1)
        # One correct, one wrong, so both strata exist for every respondent.
        for q, endorsed in (("QV01", (1, 2, 3)), ("QV02", (1, 4))):
            items.append(
                make_item(
                    respondent=rid,
                    level=level,
                    question_id=q,
                    endorsed=endorsed,
                    group_id=group,
                )
            )
    return items
