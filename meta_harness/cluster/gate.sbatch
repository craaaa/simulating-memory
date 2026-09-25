#!/bin/bash
#SBATCH --job-name=mh-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# GRES here is TYPED (sinfo: gpu:h200:8, gpu:l40s:4), so the GPU type belongs in
# --gres and --constraint is unnecessary; a plain --gres=gpu:2 is rejected with
# "No partition fits this job request".
#
# Neither account can reach h100 or a100 (those partitions are _tandon / _cds
# /_cilvr with no _public equivalent, and both accounts fail --test-only on
# them).  Reachable GPUs are l40s (46GB) and h200 (141GB), and the h200_cds
# partition -- available only on the torch_pr_287_cds account -- was the fastest
# to schedule by a wide margin when this was measured 2026-09-24:
#   h200 x1/x2/x4 on cds      -> ~30 min
#   l40s x2 on general        -> ~1 h
#   l40s x4 on general        -> ~15 h
#   h200 x2 on general/public -> ~20 h
# One H200 holds qwen3-30b (~61GB) with ~80GB left for KV, which beats the 2x
# H100 plan this replaces.
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_287_cds
#SBATCH --output=/scratch/cl5625/meta-harness-compactor/meta_harness/logs/%j.out
#SBATCH --error=/scratch/cl5625/meta-harness-compactor/meta_harness/logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5625@nyu.edu
#
# Reproduction gate: stand up vLLM, confirm it serves, run wm_word_recognition
# through bench, tear the server down.  Self-contained so nothing depends on an
# interactive session staying alive.
#
# Submit with:  ssh torch 'cd /scratch/cl5625/meta-harness-compactor && sbatch meta_harness/cluster/gate.sbatch'
set -uo pipefail

module purge

export UV_CACHE_DIR=/scratch/cl5625/.cache/uv
export UV_DATA_DIR=/scratch/cl5625/.local/share/uv
export HF_HOME=/scratch/cl5625/.cache/huggingface
export TORCH_HOME=/scratch/cl5625/.cache/torch
export PYTHONNOUSERSITE=True
# Documented Triton libcuda.so.1 fix for vLLM on this cluster.
export LIBRARY_PATH=/usr/lib64

REPO=/scratch/cl5625/meta-harness-compactor
PY="$REPO/.venv/bin/python"
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
PORT=8000
LOGS="$REPO/meta_harness/logs"
mkdir -p "$LOGS"
VLLM_LOG="$LOGS/vllm_${SLURM_JOB_ID}.log"

cd "$REPO" || exit 1

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== HEAD: $(git rev-parse HEAD)"

# --- start the server ------------------------------------------------------
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port "$PORT" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "=== tearing down vLLM (pid $VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT

# --- wait for readiness, but fail fast if the server died ------------------
echo "=== waiting for vLLM on :$PORT (log: $VLLM_LOG)"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "=== server ready after ${i}0s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!!! vLLM exited before becoming ready; last 40 lines:"
    tail -40 "$VLLM_LOG"
    exit 1
  fi
  sleep 10
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
  echo "!!! vLLM never became ready in 30min; last 40 lines:"
  tail -40 "$VLLM_LOG"
  exit 1
fi

# Record what vLLM actually allocated for KV cache -- the concurrency figure in
# the spec is an estimate until this is read off a real startup.
echo "=== measured KV cache / concurrency from vLLM startup:"
grep -iE "GPU KV cache size|Maximum concurrency|KV cache blocks" "$VLLM_LOG" || \
  echo "(no KV cache line matched; check $VLLM_LOG)"

# --- run the gate ----------------------------------------------------------
echo "=== running wm_word_recognition gate"
"$PY" -m bench.cli run \
  --config meta_harness/cluster/gate_word_recognition.yaml \
  -t wm_word_recognition
RC=$?

echo "=== bench exit code: $RC"
echo "=== gate outputs:"
find meta_harness/runs/gate -name '*.jsonl' -newermt '-4 hours' 2>/dev/null | head
exit "$RC"
