# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Code + data for the paper *Simulating Human Memory with Language Models* (arXiv:2605.25680). Runs ten classic psychology memory tasks (digit span, n-back, word recognition, variable mapping, factual/narrative QA, semantic story recall, map task, craft task) against LLMs and compares score distributions to released human data.

## Common commands

Bench CLI (canonical entrypoint):

```bash
# List all task names
python -m bench.cli list-tasks

# Run tasks from a YAML config
python -m bench.cli run --config <config.yaml> [-t <task> ...] [--model NAME] \
    [--out-dir DIR] [--repeat N] [--qwen-thinking true|false] \
    [--extra-body-json '{...}']

# Regenerate CSV/HTML/markdown tables from an existing run
python -m bench.cli tables runs/<run_dir>
```

Convenience wrapper (assembles defaults matching released `runs/`):

```bash
python src/run.py --model gpt-4o --out-dir runs/my-run
python src/run.py --model gpt-4o-mini --out-dir runs/quick \
    --tasks digit_span_forward,word_recognition --repeat 5 --include-compactor
```

Scoring (computes per-task mean score + humanlikeness = 1 - W_1 vs `runs/human/`):

```bash
python src/score.py                          # everything under runs/prompting + runs/compactor
python src/score.py --model-dir runs/my-model
python src/score.py --out tables/my-table.txt
```

Application reading-QA experiment (independent of `bench`):

```bash
python -m application.reading_qa.cli run --model openai/gpt-4.1-mini --n-repeat 20
python -m application.reading_qa.cli plot --summary-json <path>
```

Auth: set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY` with `--base-url https://openrouter.ai/api/v1`); Anthropic backend via `model.backend: anthropic` in YAML.

## Architecture

Three task families, three storage roots:

- **Prompting tasks** (e.g. `digit_span_forward`) — bare prompt, no memory module. Outputs land in `runs/prompting/<model_slug>/tasks/<task>.jsonl`. Rows carry conditions `C1` (TaskPr), `C2` (HumPr), `C3` (MemPr).
- **Compactor / WM tasks** (`wm_<task>`) — same stimuli wrapped in a 4-slot key-value working-memory agent (`bench/core/working_memory.py`, `MAX_KEYS=4` per Cowan 2001; agent loop in `bench/core/wm_agent.py`). Outputs in `runs/compactor/<model_slug>/tasks/wm_<task>.jsonl`, condition `C2`.
- **Summarizer tasks** (`sum_<task>`) — alternative compression baseline, registered alongside `wm_` variants.

The single `TASKS` dict in `bench/cli.py` maps every task name to its `evaluate` (or `evaluate_summarizer`) function in `bench/tasks/`. Each `evaluate` is invoked with `(llm, out_dir, model_cfg=..., **task_specific_kwargs)`; `cli.py` builds a per-task closure that pulls kwargs from `task_config.<task>` in the YAML and forwards `--repeat`, `--story`, `--debug`, and `max_parallel_participants`. Adding a task = add the module to `bench/tasks/`, import `evaluate` in `cli.py`, register in `TASKS`, add an `elif tname == "..."` closure block.

Model I/O is unified behind two clients in `bench/core/`: `llm_openai.OpenAIChatLLM` (default; OpenAI-compatible incl. OpenRouter, vLLM) and `llm_anthropic.AnthropicChatLLM` (selected when `model.backend: anthropic`). `model.extra_body` is merged from YAML + `--extra-body-json` + the `--qwen-thinking` shortcut (only fires when model id contains `qwen3-8b`).

Output layout per run (`out_dir` from YAML, with `{date}` template expansion + automatic `<model_slug>` nesting; slug appends `_true`/`_false` when `extra_body.enable_thinking` is set):

```
<out_dir>/<model_slug>/
    config_snapshot.json       # exact merged config used
    summary.json               # aggregate metrics across tasks
    tasks/<task>.jsonl         # per-participant rows (what score.py reads)
    metrics_<task>.txt         # per-task condition table
    metrics_all_tasks.md
```

`bench/core/runner.py` runs task functions sequentially by default, or in a thread pool when `run.max_parallel_tasks > 1`; participant-level parallelism is separate (`run.max_parallel_participants`, threaded inside each task).

`src/run.py` and `src/score.py` are thin convenience wrappers: `run.py` builds a YAML config from defaults that reproduce the released `runs/` data and shells out to `python -m bench.cli run`; `score.py` walks `runs/prompting/`, `runs/compactor/`, and any `--model-dir`, expects the JSONL layout above, and joins against `runs/human/<task>/` for the humanlikeness score.

The `application/` tree is a separate reading-QA experiment (its own CLI under `application/reading_qa/`, its own outputs, its own plots) plus standalone analysis scripts (`plot_*.py`, `compare_wasserstein_reading_qa.py`) that consume both `runs/` and `application/` outputs.

Static stimuli live in `data/` (`words.json`, `names.json`, `city.json`, `maps.json`, `narrative_QA.json`, `craft_task.json`, `wikipedia_10docs_questions.json`, plus the four story transcripts for `semantic_story_recall`). Defaults in `bench/cli.py` hard-code these paths; override via `task_config.<task>.<*>_json_path`.
