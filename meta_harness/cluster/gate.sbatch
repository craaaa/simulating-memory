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
#
# No --mail-type / --mail-user: the user does not want job email.
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
# LIBRARY_PATH is the nyu-hpc skill's documented Triton fix; kept because it is
# harmless, but note it was NOT the cause of the job-18488082 failure.
export LIBRARY_PATH=/usr/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
#
# Root cause of that failure, established by diagnosis rather than the skill's
# gotcha table: the system interpreter /usr/bin/python3.12 has NO development
# headers -- /usr/include/python3.12/Python.h does not exist.  Triton compiles
# cuda_utils.c into a Python extension module at runtime, so engine-core init
# dies with an InductorError wrapping a gcc failure.  libcuda was a red herring:
# libcuda.so.1 is present in BOTH /usr/lib64 and /lib64 and a direct
# `gcc -l:libcuda.so.1 -L/usr/lib64` link test succeeds on the GPU node.
#
# Fix: build the venv against a uv-managed CPython, which ships its own headers
# (uv python install 3.12 -> .../cpython-3.12.13-linux-x86_64-gnu with
# include/python3.12/Python.h).  Nothing here needs to set include paths; the
# venv's sysconfig points Triton at the right place automatically.

REPO=/scratch/cl5625/meta-harness-compactor
PY="$REPO/.venv/bin/python"

# The venv's bin MUST be on PATH, not just used via $PY.  FlashInfer JIT-compiles
# its sampling kernels and shells out to `ninja`; ninja is installed in the venv,
# but with only $PY invoked directly, shutil.which("ninja") returns None and
# engine startup dies with FileNotFoundError: 'ninja' (job 18489568).
export PATH="$REPO/.venv/bin:$PATH"

# Keep FlashInfer's JIT artifacts off $HOME, which has a 30k inode limit that
# compile output would exhaust.
export FLASHINFER_WORKSPACE_BASE=/scratch/cl5625/.cache/flashinfer
export FLASHINFER_CACHE_DIR=/scratch/cl5625/.cache/flashinfer
export TORCHINDUCTOR_CACHE_DIR=/scratch/cl5625/.cache/torchinductor
export TRITON_CACHE_DIR=/scratch/cl5625/.cache/triton
mkdir -p "$FLASHINFER_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# The OpenAI client bench uses requires *some* api_key even when base_url points
# at localhost, and refuses to construct without one (job 18489855).  Set an
# obvious dummy: vLLM ignores it, and setting it explicitly also guarantees a
# real key inherited from the environment can never be used, so there is no path
# from this job to a paid endpoint.
export OPENAI_API_KEY=local-vllm-no-key-needed
unset OPENROUTER_API_KEY
unset ANTHROPIC_API_KEY
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
  `# --enforce-eager keeps this gate on the cluster's known-good path (the` \
  `# skill records it as the fix for Triton trouble) at the cost of CUDA` \
  `# graphs.  The gate only needs to be correct, not fast.  For the 40-candidate` \
  `# search, drop it and measure -- per-candidate wall-clock compounds there.` \
  --enforce-eager \
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
    echo "!!! vLLM exited before becoming ready."
    # The APIServer traceback is generic ("Engine core initialization failed")
    # and the real cause is in the EngineCore block ABOVE it, so a short tail
    # hides it -- that cost a debugging round on job 18489568.  Surface the
    # actual exception lines first, then a long tail.
    echo "--- exception lines:"
    grep -E "^\(EngineCore.*(Error|Exception|error:)" "$VLLM_LOG" | tail -5
    grep -E "(FileNotFoundError|ImportError|RuntimeError|OSError|CalledProcessError|AssertionError):" "$VLLM_LOG" | tail -5
    echo "--- last 200 lines:"
    tail -200 "$VLLM_LOG"
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

# --- preflight: the endpoint must actually complete, not just answer /health --
echo "=== preflight chat completion"
PREFLIGHT=$(curl -sS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":8,\"temperature\":0}" 2>&1)
echo "$PREFLIGHT" | head -c 600
echo
if ! printf '%s' "$PREFLIGHT" | grep -q '"choices"'; then
  echo "!!! endpoint did not return a completion; aborting before bench"
  exit 1
fi

# --- run the gate ----------------------------------------------------------
echo "=== running wm_word_recognition gate"
# --repeat 50 MUST be passed on the command line.  bench/cli.py:179 only WRITES
# the CLI --repeat into run_cfg; it never reads run.repeat back out, so a YAML
# `run: repeat:` is ignored for task sizing.  Without this flag the task falls
# back to task_config n_repeat (5) and the gate would compare 5 participants
# against the released run's 50, which is not a comparison at all.
#
# 50 is what the released runs used, recovered from their row counts:
# word_recognition 50 rows; digit span 19 spans x (2*50) = 1900;
# story recall 4 stories x 50 = 200; nback 3 levels x 50 = 150.
"$PY" -m bench.cli run \
  --config meta_harness/cluster/gate_word_recognition.yaml \
  --repeat 50 \
  -t wm_word_recognition
RC=$?

echo "=== bench exit code: $RC"
echo "=== gate outputs:"
find meta_harness/runs/gate -name '*.jsonl' -newermt '-4 hours' 2>/dev/null | head
exit "$RC"
