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

Application listening-QA experiment (also independent of `bench`, parallel to reading-QA — separate stimuli, own prompting + WM full-grid runners):

```bash
python -m application.listening_qa.cli run --model openai/gpt-4.1-mini --n-repeat 20
# enumerate every (topic, level, condition) cell instead of random sampling — used for the released runs/ data
python -m application.listening_qa.full_grid_cli run --model <id> --backend openai \
    --n-repeats-per-cell 20 --out-dir runs/prompting/<model_slug>/tasks
python -m application.listening_qa.wm_full_grid_cli run --model <id> --backend openai \
    --n-repeats-per-cell 20 --out-dir runs/compactor/<model_slug>/tasks
# human-vs-WM agreement scoring, consumed by application/plot_listening_qa_*.py
python -m application.listening_qa.level_pair_preference_alignment \
    --standalone-jsonl <prompting.jsonl> --wm-jsonl <compactor.jsonl> --out-json <path>
```

Auth: set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY` with `--base-url https://openrouter.ai/api/v1`); Anthropic backend via `model.backend: anthropic` in YAML.

## Architecture

Three task families, three storage roots:

- **Prompting tasks** (e.g. `digit_span_forward`) — bare prompt, no memory module. Outputs land in `runs/prompting/<model_slug>/tasks/<task>.jsonl`. Rows carry conditions `C1` (TaskPr), `C2` (HumPr), `C3` (MemPr).
- **Compactor / WM tasks** (`wm_<task>`) — same stimuli wrapped in a 4-slot key-value working-memory agent (`bench/core/working_memory.py`, `MAX_KEYS=4` per Cowan 2001; agent loop in `bench/core/wm_agent.py`). Outputs in `runs/compactor/<model_slug>/tasks/wm_<task>.jsonl`, condition `C2`. Two encode modes: `encode()` (batch, whole passage in one turn, tool-call cap `max(6, tool_interactions*1.5)`) and `encode_streaming()` (condition `C2-stream`, segment-by-segment with no lookback — each segment gets a fresh message list built from the current KV snapshot only, forcing commit-as-you-go writes/deletes; tool calls capped per-segment (default 4) *and* per-trial (default 12), `parallel_tool_calls=False`). `wm_application_listening_qa` currently uses the batch path only.
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

The `application/` tree holds two independent experiments — reading-QA (`application/reading_qa/`) and listening-QA (`application/listening_qa/`, its own CLI + full-grid/WM-full-grid runners + human-alignment scoring) — plus `application/compactor/`, `application/prolific_study/` (human-data collection), `application/comparisons/` (cross-model comparison JSON/plots consumed by `SendUserFile`-delivered charts), and standalone analysis scripts (`plot_*.py`, `compare_wasserstein_reading_qa.py`) at the top level that consume both `runs/` and `application/` outputs.

Static stimuli live in `data/` (`words.json`, `names.json`, `city.json`, `maps.json`, `narrative_QA.json`, `craft_task.json`, `wikipedia_10docs_questions.json`, plus the four story transcripts for `semantic_story_recall`). Defaults in `bench/cli.py` hard-code these paths; override via `task_config.<task>.<*>_json_path`.
