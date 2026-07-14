"""Full-grid runner for the listening QA prompting task (C1-C4).

Unlike ``cli.py run`` (random topic/level sample per repeat), this enumerates
every (topic, level, condition) cell and runs a fixed number of repeats per
cell. Row and summary schema match the existing
``runs/prompting/<model_slug>/tasks/application_listening_qa_full_grid.jsonl``
(and its ``_summary.json``) produced for meta-llama/llama-3.1-8b-instruct, so
downstream consumers (e.g. ``level_pair_preference_alignment.py``) work
unmodified against a new model's output.

Usage:
    python -m application.listening_qa.full_grid_cli run \\
        --model openai/gpt-4.1-mini --backend openai \\
        --n-repeats-per-cell 30 \\
        --out-dir runs/prompting/openai_gpt-4.1-mini/tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from bench.core.io import ensure_dir, git_provenance, write_json, write_jsonl
from bench.core.parallel import map_participants, resolve_worker_count

from .data import LEVELS, load_topics
from .prompting import CONDITIONS, build_prompt, content_questions, parse_answers_and_difficulty, score_topic

TASK_NAME = "application_listening_qa"
CONDITION_IDS = ["C1", "C2", "C3", "C4"]

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Full-grid listening QA runner (every topic x level x C1-C4 cell, fixed repeats/cell)."""


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
    max_tokens: int = typer.Option(2048, "--max-tokens"),
    top_p: float = typer.Option(1.0, "--top-p"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    max_parallel: Optional[int] = typer.Option(
        None, "--max-parallel", min=1, help="Max concurrent trials. Default: all trials at once."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build prompts and write jsonl without calling the model."
    ),
):
    model_slug = model.replace("/", "_").replace("\\", "_")
    tasks_dir = Path(out_dir) if out_dir else Path("application/out") / model_slug / "tasks"
    ensure_dir(tasks_dir)

    llm = None
    if not dry_run:
        if backend.lower() == "anthropic":
            from bench.core.llm_anthropic import AnthropicChatLLM

            llm = AnthropicChatLLM(model=model, base_url=base_url)
        else:
            from bench.core.llm_openai import OpenAIChatLLM

            llm = OpenAIChatLLM(model=model, base_url=base_url)

    topics = load_topics(Path(documents_dir))
    cells = [
        (topic, level, cond_id)
        for topic in topics
        for level in LEVELS
        for cond_id in CONDITION_IDS
    ]
    jobs = [
        (topic, level, cond_id, repeat_index)
        for (topic, level, cond_id) in cells
        for repeat_index in range(1, n_repeats_per_cell + 1)
    ]
    workers = resolve_worker_count(len(jobs), max_parallel=max_parallel)

    def _run_one(job):
        topic, level, cond_id, repeat_index = job
        listening_text = topic.levels[level]
        questions = content_questions(topic.questions)
        prompt = build_prompt(cond_id, listening_text, questions)

        row = {
            "id": f"{TASK_NAME}:{cond_id}:r{repeat_index}:{topic.topic_id}:{level}",
            "condition_id": cond_id,
            "condition_name": CONDITIONS[cond_id]["name"],
            "repeat_index": repeat_index,
            "topic_id": topic.topic_id,
            "title": topic.title,
            "level": level,
            "prompt": prompt,
            "raw_response": None,
            "parsed_answers": None,
            "difficulty": None,
            "parse_errors": [],
            "metrics": None,
        }

        if dry_run:
            return row

        resp = llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        parsed = parse_answers_and_difficulty(resp.text)
        answers = parsed["answers"]
        metrics = score_topic(questions, answers)

        row.update(
            {
                "raw_response": resp.text,
                "parsed_answers": {str(k): v for k, v in answers.items()},
                "difficulty": parsed["difficulty"],
                "parse_errors": parsed["parse_errors"],
                "metrics": metrics,
            }
        )
        return row

    rows = map_participants(jobs, _run_one, max_workers=workers)

    cell_summaries = {}
    for topic, level, cond_id in cells:
        key = f"{topic.topic_id}:{level}:{cond_id}"
        cell_rows = [
            r
            for r in rows
            if r["topic_id"] == topic.topic_id and r["level"] == level and r["condition_id"] == cond_id
        ]
        scored = [r for r in cell_rows if r.get("metrics") is not None]
        n = len(cell_rows)
        exact_mean = sum(r["metrics"]["exact_match_accuracy"] for r in scored) / len(scored) if scored else None
        stmt_mean = sum(r["metrics"]["per_statement_accuracy"] for r in scored) / len(scored) if scored else None
        parse_errors = sum(1 for r in cell_rows if r.get("parse_errors"))
        cell_summaries[key] = {
            "n": n,
            "exact_match_mean": exact_mean,
            "per_statement_mean": stmt_mean,
            "parse_error_count": parse_errors,
        }

    summary = {
        "task": TASK_NAME,
        "model": model,
        "conditions": CONDITION_IDS,
        "n_repeats_per_cell": n_repeats_per_cell,
        "n_cells": len(cells),
        "n_total_trials": len(jobs),
        "documents_dir": str(documents_dir),
        "dry_run": dry_run,
        "git_provenance": git_provenance(),
        "cells": cell_summaries,
    }

    write_jsonl(tasks_dir / f"{TASK_NAME}_full_grid.jsonl", rows)
    write_json(tasks_dir / f"{TASK_NAME}_full_grid_summary.json", summary)
    typer.echo(f"Saved: {tasks_dir / f'{TASK_NAME}_full_grid.jsonl'}")
    typer.echo(f"Saved: {tasks_dir / f'{TASK_NAME}_full_grid_summary.json'}")


if __name__ == "__main__":
    app()
