from __future__ import annotations

import hashlib
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from application.listening_qa.data import LEVELS, Topic, Question, load_topics
from application.listening_qa.prompting import (
    FORMAT_RULES,
    HUMAN_PROMPT,
    TASK_DESC,
    content_questions,
    parse_answers_and_difficulty,
    score_topic,
)

from ..core.io import JsonlSink, write_json
from ..core.llm import LLM
from ..core.parallel import map_participants, resolve_worker_count
from .wm_mcq_common import run_wm_mcq_trial
from .wm_prompt_parts import (
    CONDITIONS,
    wm_mcq_recall_preamble,
    wm_system_prompts,
)

TASK_NAME = "wm_application_listening_qa"

RECALL_PREAMBLE = wm_mcq_recall_preamble(
    task_prompt=TASK_DESC,
    answer_focus="about the audio you listened to",
)

WM_SYSTEM_PROMPTS = wm_system_prompts(
    task_prompt=TASK_DESC,
    human_task_prompt=HUMAN_PROMPT,
)


def _rng_for_item_draw(stimuli_seed: int, participant_index: int, cond_id: str) -> random.Random:
    payload = f"{TASK_NAME}:{stimuli_seed}:{participant_index}:{cond_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return random.Random(seed_int)


def _format_questions(questions: List[Question]) -> str:
    lines = []
    for i, q in enumerate(questions, start=1):
        opts = " ".join(f"{key}) {q.options[key]}" for key in sorted(q.options))
        lines.append(f"Question {i}: {q.question} {opts}")
    lines.append("")
    lines.append("On a scale 1-10, rate how difficult the passage was, and how difficult the questions were.")
    return "\n".join(lines)


def _question_rows(questions: List[Question]) -> List[Dict[str, Any]]:
    return [
        {
            "q_id": q.q_id,
            "question": q.question,
            "options": q.options,
            "answer": q.answer,
            "qtype": q.qtype,
        }
        for q in questions
    ]


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _difficulty_histogram(values: List[int]) -> Dict[str, int]:
    hist = {str(k): 0 for k in range(1, 11)}
    for val in values:
        if 1 <= val <= 10:
            hist[str(val)] += 1
    return hist


def _parse_error_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for err in row.get("parse_errors", []):
            counts[str(err)] = counts.get(str(err), 0) + 1
    return counts


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stmt_acc_values = [float(r["metrics"]["per_statement_accuracy"]) for r in rows]
    exact_acc_values = [float(r["metrics"]["exact_match_accuracy"]) for r in rows]
    slot_values = [float(r["metrics"]["slot_utilization"]) for r in rows]
    passage_diff_values = [
        int(r["difficulty"]["passage"])
        for r in rows
        if isinstance(r.get("difficulty"), dict) and isinstance(r["difficulty"].get("passage"), int)
    ]
    question_diff_values = [
        int(r["difficulty"]["question"])
        for r in rows
        if isinstance(r.get("difficulty"), dict) and isinstance(r["difficulty"].get("question"), int)
    ]
    return {
        "n": len(rows),
        "metrics": {
            "per_statement_accuracy_mean": (
                sum(stmt_acc_values) / len(stmt_acc_values) if stmt_acc_values else 0.0
            ),
            "exact_match_accuracy_mean": (
                sum(exact_acc_values) / len(exact_acc_values) if exact_acc_values else 0.0
            ),
            "slot_utilization": sum(slot_values) / len(slot_values) if slot_values else 0.0,
        },
        "per_statement_accuracy": _summary_stats(stmt_acc_values),
        "difficulty": {
            "passage": {
                **_summary_stats([float(v) for v in passage_diff_values]),
                "valid_n": len(passage_diff_values),
                "histogram_1_to_10": _difficulty_histogram(passage_diff_values),
            },
            "question": {
                **_summary_stats([float(v) for v in question_diff_values]),
                "valid_n": len(question_diff_values),
                "histogram_1_to_10": _difficulty_histogram(question_diff_values),
            },
        },
        "parse_errors": _parse_error_counts(rows),
    }


def _summarize_condition_level(
    rows: List[Dict[str, Any]],
    condition_ids: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cond_id in condition_ids:
        out[cond_id] = {}
        for level in LEVELS:
            cell_rows = [
                r for r in rows if r["condition_id"] == cond_id and r["level"] == level
            ]
            out[cond_id][level] = _summarize_rows(cell_rows)
    return out


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    documents_dir: str = "application/listening_qa/data",
    n_participants: int = 1,
    n_docs: Optional[int] = None,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    topics = load_topics(Path(documents_dir))
    if not topics:
        raise ValueError(f"No topics in {documents_dir}.")

    rng = random.Random(stimuli_seed)
    pool = list(topics)
    if n_docs is not None:
        pool = rng.sample(pool, min(int(n_docs), len(pool)))

    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(
        n_participants, max_parallel=max_parallel_participants
    )
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in CONDITIONS:

        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            pick_rng = _rng_for_item_draw(stimuli_seed, pid, cond_id)
            topic = pick_rng.choice(pool)
            level = pick_rng.choice(LEVELS)
            passage = topic.levels[level]
            questions = content_questions(topic.questions)
            generic_topic = topic.topic_id.replace("_", " ")
            encode_content = f"Topic: {generic_topic}\n\n{passage}"
            questions_text = _format_questions(questions)

            result = run_wm_mcq_trial(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                encode_content=encode_content,
                questions_text=questions_text,
                recall_preamble=RECALL_PREAMBLE,
                format_rules=FORMAT_RULES,
                system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
            )

            parsed = parse_answers_and_difficulty(result["recall_raw"])
            answer_map = parsed["answers"]
            scored = score_topic(questions, answer_map)
            scored["slot_utilization"] = result["slot_utilization"]

            if debug:
                print(
                    f"  {TASK_NAME} | {cond_id} | p{pid} | topic={topic.topic_id} | level={level}"
                )
                print(f"  answers: {answer_map}")
                print(f"  difficulty: {parsed['difficulty']}")
                print(f"  per_statement_accuracy: {scored['per_statement_accuracy']:.2f}")
                print()

            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{pid}:{topic.topic_id}:{level}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "topic_id": topic.topic_id,
                "title": topic.title,
                "level": level,
                "encoding_log": result["encoding_log"],
                "final_kv": result["final_kv"],
                "recall_raw": result["recall_raw"],
                "parsed_answers": {str(k): v for k, v in answer_map.items()},
                "difficulty": parsed["difficulty"],
                "parse_errors": parsed["parse_errors"],
                "questions": _question_rows(questions),
                "metrics": scored,
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = {
            "condition": cond_id,
            "condition_name": CONDITIONS[cond_id]["name"],
            **_summarize_rows(rows),
        }
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    by_level: Dict[str, Any] = {}
    for level in LEVELS:
        by_level[level] = _summarize_rows(
            [r for r in all_rows if r["level"] == level]
        )

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "stimuli_seed": int(stimuli_seed),
        "documents_dir": str(documents_dir),
        "conditions": cond_summaries,
        "overall": _summarize_rows(all_rows),
        "by_level": by_level,
        "by_condition_level": _summarize_condition_level(
            all_rows, list(CONDITIONS.keys())
        ),
    }
    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
