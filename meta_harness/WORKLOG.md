# Worklog — autonomous session, 2026-09-25

Running unattended at the user's request. No money spent: everything is local
vLLM on Torch GPU hours, no paid API calls.

## Where things stood at the start of this stretch

- Onboarding complete, `domain_spec.md` written and revised.
- Cluster env built and verified on compute nodes.
- Reproduction gate (wm_word_recognition on local vLLM vs the released
  OpenRouter baseline) still unanswered after three infrastructure failures.
- Scaffolding deliberately not started: a gate failure would change the
  evaluation contract and the code would be rewritten.

## Gate attempts

| job | result | cause |
|---|---|---|
| 18488082 | FAILED | `Python.h` absent — `/usr/bin/python3.12` ships no dev headers, so Triton cannot build `cuda_utils.c`. Looked like the skill's documented libcuda gotcha; it is not. libcuda.so.1 is in both `/usr/lib64` and `/lib64` and links fine. |
| 18489568 | FAILED | `FileNotFoundError: 'ninja'`. FlashInfer JIT-compiles sampling kernels via ninja. Ninja *was* installed in the venv, but the job called `$REPO/.venv/bin/python` directly and never put `.venv/bin` on `PATH`. |
| 18489855 | running | — |

All three surfaced as the same generic `RuntimeError: Engine core
initialization failed`, with the real exception in the EngineCore block *above*
the APIServer traceback — so the job's own `tail -40` hid it and cost a round.
Now greps exception lines first, then tails 200.

## Measured, not estimated

From 18489568 startup on `gh117` (H200):

- model 56.93 GiB, available KV cache 67.15 GiB
- GPU KV cache 733,488 tokens
- **maximum concurrency 89.54x** at 8192 tokens/request

Above the `max_parallel_participants: 50` the released configs use, so one H200
at `--tensor-parallel-size 1` is enough. Supersedes my earlier derived
"~25-30 concurrent" H100 figure.

Cluster facts established by testing:

- h100 and a100 are unreachable for both of this user's accounts.
- `h200_cds` on `torch_pr_287_cds` schedules in ~30 min; l40s x2 ~1 h;
  l40s x4 ~15 h; h200 on general/public ~20 h.
- GRES is typed: `--gres=gpu:h200:1`, and `--constraint` is then unnecessary.
- Both models already cached on scratch, so nothing downloads.
- Compute nodes DO have outbound internet (huggingface.co 200 in 0.034s).

## Analysis completed while waiting

Axis work, all on released data, no API cost:

- **A1 was a schedule artifact.** The human staircase is 2 trials/span,
  ascending, stopping on a double failure; the model ran all 19 spans. Matched
  up, opus sub-span leak is 0.077 vs humans' 0.087 — already human-like. The
  earlier "supra-span hit 0.574 vs 0.000" is withdrawn. A1 becomes a
  constraint, leak in [0.05, 0.12].
- **A2 is the real axis, and it is universal.** All eight baselined models
  over-false-alarm: miss/FA ratio 0.000–1.329 against humans' 6.09, so even the
  closest is 4.6x off.
- **A3 is weak on the chosen substrate.** Only opus (BLEU 0.199) and mildly
  gpt-5.4 (0.017) are verbatim; qwen3-30b is already at human BLEU and near
  human length. Most models *under*-recall (82–97 words vs 137). A3 becomes a
  constraint too.

Consequence worth the user's attention: **on qwen3-30b only A2 has headroom.**
Searching on opus would give headroom on both A2 and A3. That is a real cost of
the no-spend plan, not an argument to override it.

## Running notes

(appended below as things happen)
