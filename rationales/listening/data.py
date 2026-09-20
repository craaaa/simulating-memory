"""Load human listening-QA responses as STaR problems (x_i, y_i).

y_i is the set of options the participant actually endorsed, not the correct answer --
the same deliberate deviation from STaR (Zelikman et al. 2022) that the digit-span
loader makes, for the same reason: the object of study is which errors a human makes,
so the target has to be the human's own response.

One item is one (participant, topic, question). x carries the passage transcript, the
question's five options, and that participant's answers to the other four questions of
the same topic; y is their endorsement set for this question. 201 respondents x 4 topics
x 5 content questions = 4020 items.

Three sources are joined:

  analysis/data/processed/responses.csv   the option-level long table (what was endorsed,
                                          plus the blind option_type / cue_match labels)
  data/<topic>/questions.yaml             question and option text, and the gold answer
  data/<topic>/texts/<level>.md           the transcript of the passage that participant
                                          heard

Note the asymmetry the study carries and this loader inherits: humans heard audio, and
the transcript is what a model is shown. That is decision D5 in the analysis README, not
something introduced here.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from application.listening_qa.data import LEVELS, Question, Topic, load_topics
from application.listening_qa.prompting import content_questions

from ..config import REPO_ROOT

ITEM_BANK = REPO_ROOT / "application" / "listening_qa" / "data"
RESPONSES_CSV = (
    REPO_ROOT / "application" / "listening_qa" / "analysis" / "data" / "processed" / "responses.csv"
)

# responses.csv is a build artifact of the R pipeline's ingest step, and it is
# gitignored. If it is missing, this is the command that makes it.
REGENERATE_CMD = "make -C application/listening_qa/analysis reconcile"

# responses.csv holds 106k model rows alongside the 20.1k human ones. Every read is
# filtered on this; pulling model rows in would be silent and would poison the corpus.
HUMAN_AGENT = "human"

TRUE_STRINGS = {"true", "True", "TRUE", "1", "1.0"}


def _flag(value: str) -> bool:
    return str(value).strip() in TRUE_STRINGS


@dataclass(frozen=True)
class SiblingAnswer:
    """One of the other four questions of the same passage, as that participant
    answered it. This is context for the target question, never the target itself."""

    question_id: str
    question: str
    options: Dict[int, str]
    endorsed: List[int]
    gold: List[int] = field(default_factory=list)

    def endorsed_text(self) -> List[str]:
        return [self.options[n] for n in self.endorsed if n in self.options]

    @property
    def correct(self) -> bool:
        """Exact set match, the same definition ListeningItem.correct uses."""
        return set(self.endorsed) == set(self.gold)


@dataclass(frozen=True)
class ListeningItem:
    respondent_id: str
    topic: str
    level: str
    question_id: str
    position: Optional[int]        # where this topic fell in that participant's sequence
    group_id: Optional[str]        # counterbalancing group; the split stratifies on it
    passage: str
    question: str
    options: Dict[int, str]
    gold: List[int]                # correct option numbers
    endorsed: List[int]            # y_i -- what this human selected
    other_answers: List[SiblingAnswer] = field(default_factory=list)
    option_labels: Dict[int, Tuple[str, str]] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return f"{self.respondent_id}:{self.topic}:{self.question_id}"

    @property
    def correct(self) -> bool:
        """Exact set match against the gold answer. The bank scores per statement too,
        but the STaR filter is exact-match on the human's set, so `correct` is defined
        the same way to keep the two readings of "fail" from diverging."""
        return set(self.endorsed) == set(self.gold)

    @property
    def n_options(self) -> int:
        return len(self.options)


def _read_human_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. It is a gitignored build artifact of the analysis "
            f"ingest step; regenerate it with:\n    {REGENERATE_CMD}"
        )
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("agent") == HUMAN_AGENT]
    if not rows:
        raise ValueError(f"{path}: no rows with agent=={HUMAN_AGENT!r}")
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _check_bank_agrees(
    q: Question, opt_rows: Dict[int, Dict[str, str]], where: str
) -> None:
    """The CSV carries its own option_is_true flags, derived independently of the item
    bank. If the two ever disagree, one of them has been edited and every downstream
    correctness number is wrong -- so this is an error, not a warning."""
    csv_true = {n for n, r in opt_rows.items() if _flag(r["option_is_true"])}
    bank_true = set(q.answer)
    if csv_true != bank_true:
        raise ValueError(
            f"{where}: responses.csv marks options {sorted(csv_true)} true but "
            f"{ITEM_BANK}/{q.q_id} says {sorted(bank_true)}. The CSV and the question "
            "bank have diverged; re-run the ingest step before trusting either."
        )


def load_items(
    *,
    responses_csv: Path = RESPONSES_CSV,
    item_bank: Path = ITEM_BANK,
) -> Tuple[List[ListeningItem], Dict[str, Any]]:
    """Every human item, plus a provenance/exclusion report."""
    topics = {t.topic_id: t for t in load_topics(item_bank)}
    questions_by_topic: Dict[str, Dict[str, Question]] = {
        tid: {q.q_id: q for q in content_questions(t.questions)}
        for tid, t in topics.items()
    }

    rows = _read_human_rows(responses_csv)

    # (respondent, topic, question) -> option number -> row
    grouped: Dict[Tuple[str, str, str], Dict[int, Dict[str, str]]] = defaultdict(dict)
    for r in rows:
        key = (r["respondent_id"], r["topic"], r["question_id"])
        grouped[key][int(float(r["option_n"]))] = r

    skipped_attention: List[str] = []
    skipped_unknown: List[str] = []
    skipped_failed_attention: List[str] = []
    partial_option_sets: List[str] = []

    # First pass: one raw record per (respondent, topic, question).
    raw: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (respondent, topic, q_id), opt_rows in grouped.items():
        bank = questions_by_topic.get(topic)
        if bank is None or q_id not in bank:
            # Attention checks are dropped from the task entirely (they are also
            # already absent from responses.csv); anything else unknown is a join bug.
            (skipped_attention if q_id.startswith("AT_") else skipped_unknown).append(
                f"{topic}:{q_id}"
            )
            continue
        q = bank[q_id]
        if set(opt_rows) != set(q.options):
            partial_option_sets.append(f"{respondent}:{topic}:{q_id}")
            continue
        any_row = next(iter(opt_rows.values()))
        if not _flag(any_row.get("attn_pass_all", "True")):
            skipped_failed_attention.append(respondent)
            continue

        _check_bank_agrees(q, opt_rows, where=f"{respondent}:{topic}:{q_id}")
        raw[(respondent, topic)][q_id] = {
            "question": q,
            "endorsed": sorted(n for n, r in opt_rows.items() if _flag(r["endorsed"])),
            # Level is per (respondent, topic) and counterbalanced -- read, never assumed.
            "level": any_row["level"],
            "position": any_row.get("position"),
            "group_id": any_row.get("group_id"),
            "labels": {
                n: (r.get("option_type", ""), r.get("cue_match", ""))
                for n, r in opt_rows.items()
            },
        }

    items: List[ListeningItem] = []
    level_conflicts: List[str] = []
    for (respondent, topic), by_q in sorted(raw.items()):
        levels = {rec["level"] for rec in by_q.values()}
        if len(levels) != 1:
            level_conflicts.append(f"{respondent}:{topic}:{sorted(levels)}")
            continue
        level = levels.pop()
        if level not in LEVELS:
            level_conflicts.append(f"{respondent}:{topic}:unknown level {level!r}")
            continue
        passage = topics[topic].levels[level]

        for q_id, rec in sorted(by_q.items()):
            q: Question = rec["question"]
            siblings = [
                SiblingAnswer(
                    question_id=other_id,
                    question=other["question"].question,
                    options=dict(other["question"].options),
                    endorsed=list(other["endorsed"]),
                    gold=list(other["question"].answer),
                )
                for other_id, other in sorted(by_q.items())
                if other_id != q_id
            ]
            items.append(
                ListeningItem(
                    respondent_id=respondent,
                    topic=topic,
                    level=level,
                    question_id=q_id,
                    position=_maybe_int(rec["position"]),
                    group_id=_maybe_group(rec["group_id"]),
                    passage=passage,
                    question=q.question,
                    options=dict(q.options),
                    gold=list(q.answer),
                    endorsed=list(rec["endorsed"]),
                    other_answers=siblings,
                    option_labels=dict(rec["labels"]),
                )
            )

    items.sort(key=lambda it: it.item_id)
    report = {
        "source_responses_csv": str(responses_csv),
        "source_responses_csv_sha256": sha256(responses_csv),
        "source_item_bank": str(item_bank),
        "source_item_bank_sha256": _bank_digest(item_bank),
        "human_rows_read": len(rows),
        "n_items": len(items),
        "n_respondents": len({it.respondent_id for it in items}),
        "n_topics": len({it.topic for it in items}),
        "n_correct": sum(1 for it in items if it.correct),
        "n_incorrect": sum(1 for it in items if not it.correct),
        "skipped_attention_check_questions": sorted(set(skipped_attention)),
        "skipped_unknown_questions": sorted(set(skipped_unknown)),
        "skipped_respondents_failed_attention": sorted(set(skipped_failed_attention)),
        "skipped_partial_option_sets": partial_option_sets[:50],
        "n_skipped_partial_option_sets": len(partial_option_sets),
        "level_conflicts": level_conflicts[:50],
        "n_level_conflicts": len(level_conflicts),
    }
    return items, report


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _maybe_group(value: Any) -> Optional[str]:
    """group_id arrives as a float-formatted string ("16.0"). Normalize it so the
    stratified split does not treat "16.0" and "16" as different groups."""
    n = _maybe_int(value)
    return None if n is None else str(n)


def _bank_digest(item_bank: Path) -> str:
    """One digest over every questions.yaml and passage in the bank, so a prompt-
    affecting edit to a stimulus shows up in run_config.json."""
    h = hashlib.sha256()
    for path in sorted(item_bank.rglob("*")):
        if path.is_file() and path.suffix in {".yaml", ".md"}:
            h.update(path.relative_to(item_bank).as_posix().encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def by_cell(items: Sequence[ListeningItem]) -> Dict[tuple, Dict[str, int]]:
    """(topic, level, question_id) -> {n, n_correct}.

    Pooled accuracy hides what matters here: across these 80 cells the exact-correct
    share runs from 0.09 to 0.93. A cell pinned near 0 or 1 teaches question shape,
    not memory decay, so any balance or eval claim needs this table rather than a mean.
    """
    out: Dict[tuple, Dict[str, int]] = {}
    for it in items:
        cell = out.setdefault((it.topic, it.level, it.question_id), {"n": 0, "n_correct": 0})
        cell["n"] += 1
        cell["n_correct"] += int(it.correct)
    return dict(sorted(out.items()))
