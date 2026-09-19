from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


LEVELS = ["control", "repeat_short", "repeat_long", "distractor"]


@dataclass(frozen=True)
class Question:
    q_id: str
    question: str
    options: Dict[int, str]
    answer: List[int]
    qtype: str  # "content" | "attention_check"


@dataclass(frozen=True)
class Topic:
    topic_id: str
    title: str
    levels: Dict[str, str]
    questions: List[Question]


def _parse_questions(raw_questions: List[Dict[str, Any]], path: Path) -> List[Question]:
    parsed: List[Question] = []
    for idx, q in enumerate(raw_questions, start=1):
        question_text = str(q.get("question", "")).strip()
        options_raw = q.get("options", {})
        answer_raw = q.get("answer", [])
        if not question_text:
            raise ValueError(f"{path}: question {idx} is missing text")
        if not isinstance(options_raw, dict) or not options_raw:
            raise ValueError(f"{path}: question {idx} options must be a non-empty object")
        options: Dict[int, str] = {}
        for key, val in options_raw.items():
            options[int(key)] = str(val).strip()
        if not isinstance(answer_raw, list) or not answer_raw:
            raise ValueError(f"{path}: question {idx} answer must be a non-empty list")
        answer = [int(a) for a in answer_raw]
        for a in answer:
            if a not in options:
                raise ValueError(f"{path}: question {idx} answer option {a} not in options")
        metadata = q.get("metadata", {}) or {}
        qtype = str(metadata.get("type", "content")).strip() or "content"
        parsed.append(
            Question(
                q_id=str(q.get("q_id", f"Q{idx:02d}")),
                question=question_text,
                options=options,
                answer=answer,
                qtype=qtype,
            )
        )
    if not parsed:
        raise ValueError(f"{path}: no questions found")
    return parsed


def _load_topic(topic_dir: Path) -> Topic:
    yaml_path = topic_dir / "questions.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    title = str(raw.get("title", "")).strip() or topic_dir.name

    levels: Dict[str, str] = {}
    for level in LEVELS:
        text_path = topic_dir / "texts" / f"{level}.md"
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{text_path}: level {level!r} is empty")
        levels[level] = text

    questions = _parse_questions(raw.get("questions", []), yaml_path)
    return Topic(topic_id=topic_dir.name, title=title, levels=levels, questions=questions)


def load_topics(data_dir: Path) -> List[Topic]:
    topics: List[Topic] = []
    for topic_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not (topic_dir / "questions.yaml").exists():
            continue
        topics.append(_load_topic(topic_dir))
    if not topics:
        raise ValueError(f"No topics found in {data_dir}")
    return topics
