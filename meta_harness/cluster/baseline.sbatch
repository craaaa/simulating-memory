#!/bin/bash
#SBATCH --job-name=mh-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_287_cds
#SBATCH --output=/scratch/cl5625/meta-harness-compactor/meta_harness/logs/%j.out
#SBATCH --error=/scratch/cl5625/meta-harness-compactor/meta_harness/logs/%j.err
#
# No --mail-type / --mail-user: the user does not want job email.
#
# Candidate 0: the unmodified compactor on all 8 search tasks, served locally.
#
# This exists because the reproduction gate FAILED: the released
# runs/compactor/qwen_qwen3-30b-a3b-instruct-2507 rows came from OpenRouter and
# do not reproduce under local bf16 serving (W_1 0.149 vs a 0.075 noise floor on
# wm_word_recognition). Baseline and candidates must share one serving stack, so
# the number every candidate is compared against has to be measured here.
#
# Submit:
#   ssh torch 'cd /scratch/cl5625/meta-harness-compactor && sbatch meta_harness/cluster/baseline.sbatch'
set -uo pipefail

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
# venv bin on PATH so FlashInfer can find ninja for its JIT compile.
export PATH="$REPO/.venv/bin:$PATH"

export FLASHINFER_WORKSPACE_BASE=/scratch/cl5625/.cache/flashinfer
export FLASHINFER_CACHE_DIR=/scratch/cl5625/.cache/flashinfer
export TORCHINDUCTOR_CACHE_DIR=/scratch/cl5625/.cache/torchinductor
export TRITON_CACHE_DIR=/scratch/cl5625/.cache/triton
mkdir -p "$FLASHINFER_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Dummy key: the OpenAI client will not construct without one even against
# localhost, and setting it explicitly guarantees no real key from the
# environment can be used, so this job cannot reach a paid endpoint.
export OPENAI_API_KEY=local-vllm-no-key-needed
unset OPENROUTER_API_KEY
unset ANTHROPIC_API_KEY

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
PORT=8000
LOGS="$REPO/meta_harness/logs"
mkdir -p "$LOGS"
VLLM_LOG="$LOGS/vllm_${SLURM_JOB_ID}.log"
OUT_DIR="meta_harness/runs/iter0/baseline"

cd "$REPO" || exit 1
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== HEAD: $(git rev-parse HEAD)"

"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "=== tearing down vLLM"; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

echo "=== waiting for vLLM on :$PORT"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "=== ready after ${i}0s"; break; }
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!!! vLLM died. exception lines:"
    grep -E "(FileNotFoundError|ImportError|RuntimeError|OSError|CalledProcessError):" "$VLLM_LOG" | tail -5
    tail -200 "$VLLM_LOG"; exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "!!! never ready"; exit 1; }

grep -iE "GPU KV cache size|Maximum concurrency" "$VLLM_LOG" || true

# Tool calling must work; the compactor is a tool-calling agent.
echo "=== preflight tool call"
TC=$(curl -sS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Store 7 under key seven with write_memory.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_memory\",\"parameters\":{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"},\"value\":{\"type\":\"string\"}},\"required\":[\"key\",\"value\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":128,\"temperature\":0}")
printf '%s' "$TC" | head -c 300; echo
printf '%s' "$TC" | grep -q tool_calls || { echo "!!! no tool_calls; check parser"; exit 1; }

# No --repeat: search_set.yaml sizes every task explicitly, because --repeat is
# global and cannot both give 50 participants and a reduced digit span.
echo "=== running the 8-task search set through the baseline candidate"
"$PY" meta_harness/run_candidate.py \
  --candidate meta_harness/candidates/baseline/harness.py \
  --config meta_harness/cluster/search_set.yaml \
  --out-dir "$OUT_DIR"
RC=$?
echo "=== run_candidate exit: $RC"

echo "=== rows written"
find "$OUT_DIR" -name '*.jsonl' | while read -r f; do echo "  $(wc -l < "$f") $f"; done
exit "$RC"
