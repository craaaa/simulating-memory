#!/bin/bash
#SBATCH --job-name=mh-cand
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
# Evaluate one candidate on the 8-task search set under local vLLM.
#
# Submit:
#   ssh torch 'cd /scratch/cl5625/meta-harness-compactor && \
#     sbatch --export=ALL,CAND=<id>,ITER=<n> meta_harness/cluster/candidate.sbatch'
#
# CAND is a directory name under meta_harness/candidates/.
# Several candidates can share one server by passing CAND="a b c".
set -uo pipefail

: "${CAND:?set CAND=<candidate id> (or a space-separated list)}"
ITER="${ITER:-0}"

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

# Dummy key: the client will not construct without one even against localhost,
# and setting it explicitly means no real key from the environment can be used,
# so this job cannot reach a paid endpoint.
export OPENAI_API_KEY=local-vllm-no-key-needed
unset OPENROUTER_API_KEY
unset ANTHROPIC_API_KEY

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
PORT=8000
LOGS="$REPO/meta_harness/logs"
mkdir -p "$LOGS"
VLLM_LOG="$LOGS/vllm_${SLURM_JOB_ID}.log"

TASKS=(
  wm_digit_span_forward wm_digit_span_reverse wm_nback wm_word_recognition
  wm_variable_mapping wm_narrative_qa wm_semantic_story_recall wm_craft_task
)
TASK_ARGS=()
for t in "${TASKS[@]}"; do TASK_ARGS+=(-t "$t"); done

cd "$REPO" || exit 1
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  iter: ${ITER}  cand: ${CAND}"
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

echo "=== preflight tool call"
TC=$(curl -sS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Store 7 under key seven with write_memory.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_memory\",\"parameters\":{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"},\"value\":{\"type\":\"string\"}},\"required\":[\"key\",\"value\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":128,\"temperature\":0}")
printf '%s' "$TC" | head -c 300; echo
printf '%s' "$TC" | grep -q tool_calls || { echo "!!! no tool_calls; check parser"; exit 1; }

# Each candidate runs in its own process: injection mutates imported bench
# modules, so two candidates in one process would contaminate each other.
RC=0
for c in $CAND; do
  echo
  echo "############ candidate: $c"

  # Most arms are a directory name under candidates/ run over all 8 tasks. The
  # exception is an ABLATION ARM: the same source file run under a different
  # environment, which exists so a candidate can be compared against the one
  # thing it changed rather than against a baseline it invalidated.
  #
  # serial_recognition_open is that arm for serial_recognition. It is the same
  # harness.py with MH_SERIAL_RECOGNITION_MASK=0, so the recall prompt still
  # shows one trial per call with the identical directive, identical stitching and
  # identical max_tokens, but leaves the other 99 trial lines visible. It answers
  # the objection that would otherwise sink the masked arm -- "you asked the model
  # to judge one word with no context, that is framing, not memory". Only
  # word_recognition differs between the arms, so it runs that task alone.
  SRC="$c"
  ARM_TASKS=("${TASK_ARGS[@]}")
  unset MH_SERIAL_RECOGNITION_MASK
  case "$c" in
    serial_recognition_open)
      SRC="serial_recognition"
      export MH_SERIAL_RECOGNITION_MASK=0
      ARM_TASKS=(-t wm_word_recognition)
      echo "  ablation arm: source=${SRC}, MASK=0, word_recognition only"
      ;;
  esac

  OUT_DIR="meta_harness/runs/iter${ITER}/${c}"
  "$PY" meta_harness/run_candidate.py \
    --candidate "meta_harness/candidates/${SRC}/harness.py" \
    --config meta_harness/cluster/search_set.yaml \
    --out-dir "$OUT_DIR" \
    "${ARM_TASKS[@]}"
  rc=$?
  echo "=== $c run_candidate exit: $rc"
  [ "$rc" -ne 0 ] && RC="$rc"
  find "$OUT_DIR" -name '*.jsonl' 2>/dev/null | while read -r f; do
    echo "  $(wc -l < "$f") $f"
  done
done

echo "=== all candidates done, worst exit: $RC"
exit "$RC"
