"""Full-grid runner for the working-memory (compactor) listening QA task.

WM only ever runs condition C2 (see ``bench/tasks/wm_prompt_parts.py``), so
this enumerates every (topic, level) cell and runs a fixed number of repeats
per cell. Row and summary schema match the existing
``runs/compactor/<model_slug>/tasks/wm_application_listening_qa_full_grid.jsonl``
(and its ``_summary.json``) produced for meta-llama/llama-3.1-8b-instruct, so
downstream consumers (e.g. ``level_pair_preference_alignment.py``) work
unmodified against a new model's output.

Usage:
    python -m application.listening_qa.wm_full_grid_cli run \\
        --model openai/gpt-4.1-mini --backend openai \\
        --n-repeats-per-cell 30 \\
        --out-dir runs/compactor/openai_gpt-4.1-mini/tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from bench.core.io import ensure_dir, estimate_cost_usd, git_provenance, run_timestamp, write_json, write_jsonl
from bench.core.parallel import map_participants, resolve_worker_count
from bench.tasks.wm_application_listening_qa import (
    FORMAT_RULES,
    RECALL_PREAMBLE,
    WM_SYSTEM_PROMPTS,
    _format_questions,
    _question_rows,
)
from bench.tasks.wm_mcq_common import run_wm_mcq_trial
from bench.tasks.wm_prompt_parts import CONDITIONS

from .data import LEVELS, load_topics
from .prompting import content_questions, parse_answers_and_difficulty, score_topic

TASK_NAME = "wm_application_listening_qa"
COND_ID = "C2"

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Full-grid WM listening QA runner (every topic x level cell, C2 only, fixed repeats/cell)."""


@app.command()
def run(
    model: str = typer.Option(..., "--model", help="Model id."),
    backend: str = typer.Option("openai", "--backend", help="openai or anthropic"),
    n_repeats_per_cell: int = typer.Option(30, "--n-repeats-per-cell", min=1),
    documents_dir: str = typer.Option(
        "application/listening_qa/data",
        "--documents-dir",
        help="Directory with topic subdirs (questions.yaml + texts/*.md).",
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        help="Directory to write tasks/ output into. Default: application/out/<model_slug>/tasks",
    ),
    temperature: float = typer.Option(0.0, "--temperature"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    max_parallel: Optional[int] = typer.Option(
        None, "--max-parallel", min=1, help="Max concurrent trials. Default: all trials at once."
    ),
):
    model_slug = model.replace("/", "_").replace("\\", "_")
    tasks_dir = Path(out_dir) if out_dir else Path("runs/compactor") / model_slug / run_timestamp() / "tasks"
    ensure_dir(tasks_dir)

    if backend.lower() == "anthropic":
        from bench.core.llm_anthropic import AnthropicChatLLM

        llm = AnthropicChatLLM(model=model, base_url=base_url)
    else:
        from bench.core.llm_openai import OpenAIChatLLM

        llm = OpenAIChatLLM(model=model, base_url=base_url)

    topics = load_topics(Path(documents_dir))
    cells = [(topic, level) for topic in topics for level in LEVELS]
    jobs = [
        (topic, level, repeat_index)
        for (topic, level) in cells
        for repeat_index in range(1, n_repeats_per_cell + 1)
    ]
    workers = resolve_worker_count(len(jobs), max_parallel=max_parallel)

    def _run_one(job):
        topic, level, repeat_index = job
        passage = topic.levels[level]
        questions = content_questions(topic.questions)
        generic_topic = topic.topic_id.replace("_", " ")
        encode_content = f"Topic: {generic_topic}\n\n{passage}"
        questions_text = _format_questions(questions)

        result = run_wm_mcq_trial(
            llm=llm,
            condition_id=COND_ID,
            temperature=temperature,
            debug=False,
            encode_content=encode_content,
            questions_text=questions_text,
            recall_preamble=RECALL_PREAMBLE,
            format_rules=FORMAT_RULES,
            system_prompt_override=WM_SYSTEM_PROMPTS[COND_ID],
        )

        parsed = parse_answers_and_difficulty(result["recall_raw"])
        answer_map = parsed["answers"]
        scored = score_topic(questions, answer_map)
        scored["slot_utilization"] = result["slot_utilization"]

        return {
            "id": f"{TASK_NAME}:{COND_ID}:r{repeat_index}:{topic.topic_id}:{level}",
            "condition_id": COND_ID,
            "condition_name": CONDITIONS[COND_ID]["name"],
            "repeat_index": repeat_index,
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

    rows = map_participants(jobs, _run_one, max_workers=workers)

    cell_summaries = {}
    for topic, level in cells:
        key = f"{topic.topic_id}:{level}:{COND_ID}"
        cell_rows = [r for r in rows if r["topic_id"] == topic.topic_id and r["level"] == level]
        n = len(cell_rows)
        exact_mean = sum(r["metrics"]["exact_match_accuracy"] for r in cell_rows) / n if n else None
        stmt_mean = sum(r["metrics"]["per_statement_accuracy"] for r in cell_rows) / n if n else None
        parse_errors = sum(1 for r in cell_rows if r.get("parse_errors"))
        cell_summaries[key] = {
            "n": n,
            "exact_match_mean": exact_mean,
            "per_statement_mean": stmt_mean,
            "parse_error_count": parse_errors,
        }

    usage = llm.usage_summary()
    usage["estimated_cost_usd"] = estimate_cost_usd(
        model, usage["prompt_tokens"], usage["completion_tokens"]
    )

    summary = {
        "task": TASK_NAME,
        "model": model,
        "condition_id": COND_ID,
        "n_repeats_per_cell": n_repeats_per_cell,
        "n_cells": len(cells),
        "n_total_trials": len(jobs),
        "llm_usage": usage,
        "git_provenance": git_provenance(),
        "cells": cell_summaries,
    }

    write_jsonl(tasks_dir / f"{TASK_NAME}_full_grid.jsonl", rows)
    write_json(tasks_dir / f"{TASK_NAME}_full_grid_summary.json", summary)
    typer.echo(f"Saved: {tasks_dir / f'{TASK_NAME}_full_grid.jsonl'}")
    cost = usage["estimated_cost_usd"]
    cost_str = f"${cost:.4f}" if cost is not None else "unknown (model not in pricing table)"
    typer.echo(
        f"LLM usage: {usage['request_count']} requests, "
        f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens. "
        f"Estimated cost: {cost_str}"
    )
    typer.echo(f"Saved: {tasks_dir / f'{TASK_NAME}_full_grid_summary.json'}")


if __name__ == "__main__":
    app()
