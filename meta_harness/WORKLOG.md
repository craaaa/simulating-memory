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
| 18489855 | FAILED | vLLM **served successfully** ("server ready after 280s"), then bench died: the OpenAI client refuses to construct without an `api_key` even against localhost. Fixed with an explicit dummy, which also guarantees no real key can be picked up from the environment. |
| 18490375 | FAILED | Served in 160s, plain completion preflight returned "OK", bench started and wrote a JSONL — then every tool call 400'd: `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. |
| 18491047 | running | — |

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

Pattern worth noting: every one of these was a *different* cause reported
through the same generic `Engine core initialization failed` or a bench
traceback, and each one was masked until the previous was fixed. The job script
now front-loads cheap checks — exception-line grep before the log tail, a plain
completion preflight, and a tool-call preflight — so the next failure identifies
itself in seconds rather than costing a queue cycle.

## Two bugs found by reading code, not by running it

Both would have produced plausible-looking but wrong numbers:

1. **`run.repeat` in YAML is inert.** `bench/cli.py:179` only *writes* the CLI
   `--repeat` into `run_cfg`; nothing reads `run.repeat` back. The gate would
   have silently used `task_config n_repeat: 5` and compared **5** participants
   against the released run's 50. Row counts confirm the released runs used
   `--repeat 50`: word_recognition 50, digit span 19 spans x (2*50) = 1900,
   story recall 4 stories x 50 = 200, nback 3 levels x 50 = 150.
2. **Injection cannot patch one module.** `MAX_KEYS` is bound in **10** modules
   and `WorkingMemoryAgent` in 8, because each `wm_*` task does
   `from ..core.wm_agent import ...` at import. Rebinding only the defining
   module would have left every task running the original harness — and every
   candidate would have scored exactly like the baseline with nothing in the
   output to reveal it.

## Scaffolding built (all tested offline, no API cost)

- `inject.py` — rebinds a candidate's overrides everywhere they are bound, and
  reports where, so a result traces to the code that produced it.
- `verify_interface.py` — offline compliance check against a scripted stub LLM
  using the OpenAI tool-call shape `wm_agent` actually reads.
- `candidates/baseline/` — released compactor unmodified; control and plumbing
  test. **Passes.**
- `candidates/random_decay/` — the adversary. Seeded from a hash of the stimulus
  content, not `id(self)`, because an address-derived seed would make the control
  irreproducible. Its test asserts reproducibility, per-participant variation,
  and a rate spread of 0.001–0.755 across 30 participants. **Passes.**
- `run_candidate.py` — runs bench in-process (required for injection to apply).
- `score_candidate.py` — the whole evaluation contract: per-task humanlikeness
  vector, mean over search tasks, A2 with CI and the trials covariate, A1/A3
  guards, deltas with floor violations, and a noise-floor flag. Verified against
  released data.
- `cluster/search_set.yaml` — 8 search tasks, digit span cut to ~190 rows.

## Running notes

(appended below as things happen)
