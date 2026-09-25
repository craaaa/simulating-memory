#!/bin/bash
# Serve a base model under vLLM for Meta-Harness candidate evaluation.
#
# Run this INSIDE a GPU compute node allocation, never on a login node:
#
#   srun --pty --gres=gpu:h200:1 --mem=96G -c 16 \
#        --account=torch_pr_287_cds --time=8:00:00 /bin/bash
#   bash meta_harness/cluster/serve_vllm.sh search
#
# GRES is typed here, and h100/a100 are unreachable for both of this user's
# accounts.  h200 (141GB) via the torch_pr_287_cds account schedules fastest --
# see the notes in gate.sbatch for measured queue times.
#
# Models are already cached under /scratch/cl5625/.cache/huggingface, so this
# does not download anything.  bf16 only -- quantization changes model
# behaviour, and behaviour is the quantity being compared to human data, so an
# fp8 run would not be comparable to the released baselines.
set -euo pipefail

ROLE="${1:-search}"

case "$ROLE" in
  search)
    # ~61GB weights.  One H200 (141GB) leaves ~80GB for KV cache, enough for the
    # released max_parallel_participants: 50.  On l40s instead, use TP=2 (92GB).
    MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
    TP="${TP:-1}"
    # Qwen3 instruct emits Hermes-style <tool_call> JSON.
    PARSER="hermes"
    ;;
  holdout)
    # ~141GB weights: exactly one H200's capacity, so no room for KV.  Use 2x
    # H200 (282GB).  On l40s that would be 4x (184GB), but l40s x4 queues ~15h.
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    TP="${TP:-2}"
    PARSER="llama3_json"
    ;;
  *)
    echo "usage: $0 [search|holdout]" >&2
    exit 2
    ;;
esac

export HF_HOME=/scratch/cl5625/.cache/huggingface
export PYTHONNOUSERSITE=True
# Documented Triton libcuda.so.1 fix for vLLM on this cluster.
export LIBRARY_PATH=/usr/lib64
export VLLM_LOGGING_LEVEL=INFO

REPO=/scratch/cl5625/meta-harness-compactor
LOG_DIR="$REPO/meta_harness/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/vllm_${ROLE}_${SLURM_JOB_ID:-nojob}.log"

echo "serving $MODEL  role=$ROLE  tp=$TP  on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "log -> $LOG"

# --served-model-name keeps the id stable for bench configs regardless of role.
"$REPO/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "$PARSER" \
  --host 127.0.0.1 \
  --port 8000 \
  2>&1 | tee "$LOG"
