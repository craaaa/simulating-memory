from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from bench.tasks.human_simulation_prefixes import (
    HUMAN_SIM_INTRO_C2,
    HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN,
)
from bench.tasks.stimulus_prompt_shell import (
    DIGIT_SPAN_SEP_BEFORE_RULES,
    wrap_stimulus_prompt,
)

from .data import Question

ANSWER_RE = re.compile(
    r"^question\s+(\d+):\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*$", re.IGNORECASE
)
ANSWER_TEXT_RE = re.compile(r"^question\s+(\d+):\s*(.+?)\s*$", re.IGNORECASE)
PASSAGE_DIFF_RE = re.compile(r"^passage\s+difficulty:\s*([0-9]{1,2})\s*$", re.IGNORECASE)
QUESTION_DIFF_RE = re.compile(r"^question\s+difficulty:\s*([0-9]{1,2})\s*$", re.IGNORECASE)

TASK_DESC = "Listen to an audio passage, answer all multiple-select questions, then rate passage and question difficulty."
HUMAN_PROMPT = "The human will listen to an audio recording once, after which the audio cannot be replayed. The human will then be asked to answer questions about the recording based on their memory, selecting all options that apply for each question, and rate passage and question difficulty out of 10."

FORMAT_RULES = """Output ONLY lines in this exact format:
Question 1: 1,3
Question 2: 2
...
Passage Difficulty: 7
Question Difficulty: 7
(List every correct option number for a question, separated by commas. If none apply, use the "None of the above" option number.)
"""

# Editable prompting-method conditions (TaskPr / HumPr / MemPr), mirrors reading_qa/prompting.py.
CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {
        "name": "Just describe the task",
        "prompt_prefix": TASK_DESC,
    },
    "C2": {
        "name": "Task + simulate human",
        "prompt_prefix": HUMAN_SIM_INTRO_C2 + HUMAN_PROMPT,
    },
    "C3": {
        "name": "Task + human + limited memory",
        "prompt_prefix": HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN + HUMAN_PROMPT,
    },
    # Keep C4 for compatibility with existing experiment structure.
    "C4": {
        "name": "Task + human + limited memory",
        "prompt_prefix": HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN + HUMAN_PROMPT,
    },
}


def build_prompt(
    condition_id: str, listening_text: str, questions: List[Question]
) -> str:
    lines: List[str] = []
    lines.append("Transcript:")
    lines.append(listening_text)
    lines.append("")
    lines.append("Questions:")
    for i, q in enumerate(questions, start=1):
        opts = " ".join(f"{key}) {q.options[key]}" for key in sorted(q.options))
        lines.append(f"Question {i}: {q.question} {opts}")
    lines.append("")
    lines.append("On a scale 1-10, rate how difficult the passage was, and how difficult the questions were.")

    stimulus = "\n".join(lines)
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    return wrap_stimulus_prompt(
        prefix,
        condition_id,
        stimulus,
        FORMAT_RULES,
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def parse_answers_and_difficulty(text: str) -> Dict[str, Any]:
    answers: Dict[int, List[int]] = {}
    text_answers: Dict[int, str] = {}
    passage_difficulty: Optional[int] = None
    question_difficulty: Optional[int] = None
    parse_errors: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        answer_match = ANSWER_RE.match(line)
        if answer_match:
            q_idx = int(answer_match.group(1))
            opts = [int(o.strip()) for o in answer_match.group(2).split(",")]
            answers[q_idx] = sorted(set(opts))
            continue
        text_match = ANSWER_TEXT_RE.match(line)
        if text_match and not PASSAGE_DIFF_RE.match(line) and not QUESTION_DIFF_RE.match(line):
            # Model answered with option text instead of a number (e.g. "None of
            # the above") — resolved against question.options by the caller,
            # which has the per-question option text this parser doesn't.
            text_answers[int(text_match.group(1))] = text_match.group(2).strip()
            continue
        passage_diff_match = PASSAGE_DIFF_RE.match(line)
        if passage_diff_match:
            passage_difficulty = int(passage_diff_match.group(1))
            continue
        question_diff_match = QUESTION_DIFF_RE.match(line)
        if question_diff_match:
            question_difficulty = int(question_diff_match.group(1))

    if passage_difficulty is None:
        parse_errors.append("passage_difficulty_missing")
    elif passage_difficulty < 1 or passage_difficulty > 10:
        parse_errors.append("passage_difficulty_out_of_range")
        passage_difficulty = None

    if question_difficulty is None:
        parse_errors.append("question_difficulty_missing")
    elif question_difficulty < 1 or question_difficulty > 10:
        parse_errors.append("question_difficulty_out_of_range")
        question_difficulty = None

    return {
        "answers": answers,
        "text_answers": text_answers,
        "difficulty": {"passage": passage_difficulty, "question": question_difficulty},
        "parse_errors": parse_errors,
    }


def content_questions(questions: List[Question]) -> List[Question]:
    """Attention-check questions are excluded from the task entirely — not asked,
    not scored. (Kept in the source data/yaml for provenance, just filtered here.)"""
    return [q for q in questions if q.qtype != "attention_check"]


def resolve_text_answers(
    questions: List[Question], answers: Dict[int, List[int]], text_answers: Dict[int, str]
) -> Dict[int, List[int]]:
    """Fold free-text answers (e.g. "Question 5: None of the above") into ``answers``
    by matching the text against that question's option text. Only applied to
    questions the numeric parse missed, so a real numeric answer always wins."""
    resolved = dict(answers)
    for q_idx, text in text_answers.items():
        if q_idx in resolved:
            continue
        if q_idx < 1 or q_idx > len(questions):
            continue
        q = questions[q_idx - 1]
        norm = text.strip().lower().rstrip(".")
        for opt_num, opt_text in q.options.items():
            if opt_text.strip().lower().rstrip(".") == norm:
                resolved[q_idx] = [opt_num]
                break
    return resolved


def score_topic(
    questions: List[Question], predicted: Dict[int, List[int]]
) -> Dict[str, Any]:
    """Score per-statement (multi_v6's native ``scoring: per_statement``).
    ``questions`` must already be attention-check-filtered (see ``content_questions``);
    this only scores what it's given."""
    stmt_correct = 0
    stmt_total = 0
    exact_correct = 0
    exact_total = 0

    for i, q in enumerate(questions, start=1):
        pred_set = set(predicted.get(i, []))
        true_set = set(q.answer)
        stmt_correct += sum(
            1 for opt in q.options if (opt in pred_set) == (opt in true_set)
        )
        stmt_total += len(q.options)
        exact_total += 1
        exact_correct += 1 if pred_set == true_set else 0

    return {
        "per_statement_correct": stmt_correct,
        "per_statement_total": stmt_total,
        "per_statement_accuracy": stmt_correct / stmt_total if stmt_total else 0.0,
        "exact_match_correct": exact_correct,
        "exact_match_total": exact_total,
        "exact_match_accuracy": exact_correct / exact_total if exact_total else 0.0,
    }
