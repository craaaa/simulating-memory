#!/bin/bash
#SBATCH --job-name=mh-heldout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_287_cds
#SBATCH --output=/scratch/cl5625/meta-harness-compactor/meta_harness/logs/%j.out
#SBATCH --error=/scratch/cl5625/meta-harness-compactor/meta_harness/logs/%j.err
#
# No --mail-type / --mail-user: the user does not want job email.
#
# Held-out evaluation: finalists + baseline on llama-3.3-70b, all 10 tasks.
#
# Two GPUs, not one. The weights are 132G on disk and ~141GB in bf16, which does
# not fit one H200's 141GB alongside a usable KV cache, so tensor-parallel-size 2.
# Verified present: /scratch/cl5625/.cache/huggingface/hub/models--meta-llama--
# Llama-3.3-70B-Instruct, 30 safetensors shards.
#
# Submit (CAND is a space-separated list; baseline MUST be included, because the
# search baseline was measured on qwen and cannot be compared across models):
#   ssh torch 'cd /scratch/cl5625/meta-harness-compactor && \
#     sbatch --export=ALL,CAND="baseline <finalist> ..." meta_harness/cluster/heldout.sbatch'
set -uo pipefail

: "${CAND:?set CAND=\"baseline <finalist> ...\"}"

module purge

export UV_CACHE_DIR=/scratch/cl5625/.cache/uv
export UV_DATA_DIR=/scratch/cl5625/.local/share/uv
export HF_HOME=/scratch/cl5625/.cache/huggingface
export TORCH_HOME=/scratch/cl5625/.cache/torch
export PYTHONNOUSERSITE=True
export LIBRARY_PATH=/usr/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}

REPO=/scratch/cl5625/meta-harness-compactor
PY="$REPO/.venv/bin/python"
export PATH="$REPO/.venv/bin:$PATH"   # FlashInfer shells out to ninja

export FLASHINFER_WORKSPACE_BASE=/scratch/cl5625/.cache/flashinfer
export FLASHINFER_CACHE_DIR=/scratch/cl5625/.cache/flashinfer
export TORCHINDUCTOR_CACHE_DIR=/scratch/cl5625/.cache/torchinductor
export TRITON_CACHE_DIR=/scratch/cl5625/.cache/triton
mkdir -p "$FLASHINFER_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Dummy key: the client will not construct without one even against localhost, and
# setting it explicitly means no real key from the environment can be used, so this
# job cannot reach a paid endpoint.
export OPENAI_API_KEY=local-vllm-no-key-needed
unset OPENROUTER_API_KEY
unset ANTHROPIC_API_KEY

MODEL=meta-llama/Llama-3.3-70B-Instruct
PORT=8000
LOGS="$REPO/meta_harness/logs"
mkdir -p "$LOGS"
VLLM_LOG="$LOGS/vllm_${SLURM_JOB_ID}.log"

# All ten: the eight search tasks plus the two held out from the search entirely.
TASKS=(
  wm_digit_span_forward wm_digit_span_reverse wm_nback wm_word_recognition
  wm_variable_mapping wm_narrative_qa wm_semantic_story_recall wm_craft_task
  wm_map_task wm_factual_qa
)
TASK_ARGS=()
for t in "${TASKS[@]}"; do TASK_ARGS+=(-t "$t"); done

cd "$REPO" || exit 1
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  cand: ${CAND}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== HEAD: $(git rev-parse HEAD)"

"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --host 127.0.0.1 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "=== tearing down vLLM"; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

# 70B on 2 GPUs loads far slower than 30B on 1; allow 60 minutes, not 30.
echo "=== waiting for vLLM on :$PORT"
for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "=== ready after ${i}0s"; break; }
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!!! vLLM died. exception lines:"
    grep -E "(FileNotFoundError|ImportError|RuntimeError|OSError|ValueError|CalledProcessError):" "$VLLM_LOG" | tail -5
    tail -200 "$VLLM_LOG"; exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "!!! never ready"; exit 1; }
grep -iE "GPU KV cache size|Maximum concurrency" "$VLLM_LOG" || true

# The compactor is a tool-calling agent, so a served model that cannot emit a tool
# call produces a run of silent zeros rather than an error. Fail fast instead.
# NOTE the parser differs from the qwen jobs: llama-3.3 uses llama3_json, not
# hermes. A wrong parser returns 200 with the call buried in content.
echo "=== preflight tool call"
TC=$(curl -sS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Store 7 under key seven with write_memory.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_memory\",\"parameters\":{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"},\"value\":{\"type\":\"string\"}},\"required\":[\"key\",\"value\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":128,\"temperature\":0}")
printf '%s' "$TC" | head -c 300; echo
printf '%s' "$TC" | grep -q tool_calls || { echo "!!! no tool_calls; check --tool-call-parser"; exit 1; }

# Each candidate in its own process: injection mutates imported bench modules, so
# two candidates in one process would contaminate each other.
RC=0
for c in $CAND; do
  echo
  echo "############ candidate: $c"
  OUT_DIR="meta_harness/runs/heldout/${c}"
  "$PY" meta_harness/run_candidate.py \
    --candidate "meta_harness/candidates/${c}/harness.py" \
    --config meta_harness/cluster/heldout_set.yaml \
    --out-dir "$OUT_DIR" \
    "${TASK_ARGS[@]}"
  rc=$?
  echo "=== $c run_candidate exit: $rc"
  [ "$rc" -ne 0 ] && RC="$rc"
  find "$OUT_DIR" -name '*.jsonl' 2>/dev/null | while read -r f; do
    echo "  $(wc -l < "$f") $f"
  done
done

echo "=== all candidates done, worst exit: $RC"
exit "$RC"
