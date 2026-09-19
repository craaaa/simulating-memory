# Handoff: does prompt-based vs. compactor-based pattern hold on open-source models?

Everything in `COMPACTOR_ITERATION_NOTES.md` was measured on gpt-4.1 only. Open question: can
we replicate behavior on an open-source model show the same pattern? Specifically, prompt-based
methods (C1-C4) should be at ceiling, while compactor performs worse.


## What to run

Current code state: `bench/tasks/wm_prompt_parts.py`, `bench/core/wm_agent.py`, and
`bench/core/working_memory.py` are at the **v10 baseline** (`write_memory`/`delete_key`,
sequential, `parallel_tool_calls=False`, filtered `to_recall_text()` snapshot) — this is
NOT the `replace_key` design; see `git log --oneline -- bench/core/wm_agent.py` and the
`exp/compactor-v10*` tags to navigate between the two if a replace_key comparison run is
also wanted (`git checkout exp/compactor-v12c -- bench/core/wm_agent.py bench/core/working_memory.py bench/tasks/wm_prompt_parts.py`
gets the cleanest replace_key baseline; remember to fix `wm_mcq_common.py`'s
`hasattr(agent.wm, "slot_utilization")` compatibility shim stays either way — it works with both).

For a chosen open-source model (`$MODEL`, an OpenRouter route id):

```bash
set -a; source .env; set +a

# standalone/prompting baseline (C1-C4), needed for the human-alignment comparison
python -m application.listening_qa.full_grid_cli run \
    --model "$MODEL" --backend openai --base-url https://openrouter.ai/api/v1 \
    --n-repeats-per-cell 5

# v10-style compactor, paragraph segmentation
python -m application.listening_qa.wm_full_grid_cli run \
    --model "$MODEL" --backend openai --base-url https://openrouter.ai/api/v1 \
    --streaming --segment-unit paragraph --n-repeats-per-cell 5

# v10-style compactor, sentence segmentation + raised cap (to test the distractor-collapse finding)
python -m application.listening_qa.wm_full_grid_cli run \
    --model "$MODEL" --backend openai --base-url https://openrouter.ai/api/v1 \
    --streaming --segment-unit sentence --trial-tool-call-cap 25 --n-repeats-per-cell 5
```

Compare accuracy (`exact_match_accuracy` mean across trials). Per [[feedback_alignment_metric_reporting]] 
(session memory): don't recompute human-alignment after every small run — use accuracy/error-mechanics for
exploratory passes, save alignment for a confirmatory run once there's a stable finding worth reporting.

## Cost note

gpt-4.1 sentence-segmented pilots ran $2.50-$5.50 for 32-80 trials (see
COMPACTOR_ITERATION_NOTES.md's cost table). Open-source models via OpenRouter are typically
much cheaper per token, but check current OpenRouter pricing for the chosen model before a
large sweep — smaller/cheaper models may also need more retries or produce more parse errors,
which increases effective request count.
