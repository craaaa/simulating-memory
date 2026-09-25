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
| 18491047 | **COMPLETED** | Full pipeline ran: server, both preflights, bench exit 0, 50 rows. Needed `--enable-auto-tool-choice --tool-call-parser hermes`. |

## GATE RESULT: FAIL — and it caught something real

`check_gate.py` on job 18491047's output:

| | local vLLM bf16 | released OpenRouter |
|---|---|---|
| paired score mean | **71.58** | **86.38** |
| miss rate | 0.056 | 0.001 |
| false-alarm rate | 0.185 | 0.064 |
| A2 miss/fa ratio | 0.302 | 0.018 |
| trials attempted | 72.9 | 87.5 |

W_1(local, released) = **0.149**, against a human split-half noise floor of
0.075 for this task. Mean paired delta -14.80; local worse on 16/50 participants,
better on 10, tied on 24.

Ruled out first, because it would have been the boring explanation: **the stimuli
are identical for all 50 paired participants.** An earlier version of this
diagnosis claimed they differed, which was my bug — rows are written in
completion order under 50-way parallelism, so index 0 in one file is not the same
participant as index 0 in the other. Paired by `id`, the stimuli match exactly,
`words.json` and the task code each have a single commit, and `generate_one_game`
is seeded deterministically. So the difference is in the model's **responses**.

**Conclusion:** the released `runs/compactor/qwen_qwen3-30b-a3b-instruct-2507`
numbers describe *OpenRouter's deployment* of that model — plausibly quantized or
with a different chat template — not the model served at bf16. The 0.167 headroom
figure for qwen3-30b therefore does not transfer to local serving.

**This does not sink the plan.** The human data is fixed, so humanlikeness
against humans stays meaningful. What breaks is using the *released* run as the
baseline. The fix is to re-measure the baseline locally, so baseline and
candidates share one serving stack. That is candidate 0 on the 8-task search set:
GPU hours, $0.

Worth noting for the paper independently of this project: released per-model
numbers are serving-stack-dependent, and the OpenRouter-served rows are not
reproducible from the model weights alone.

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

## Blocker for the user: how the proposer runs

The reference implementation invokes the proposer as the `claude` CLI once per
iteration, and deliberately strips `ANTHROPIC_API_KEY` so it authenticates
against the Claude *subscription* rather than the paid API
(`reference_examples/text_classification/meta_harness.py`, `propose_claude`).

So the outer loop costs no API dollars, but ~20 iterations of an Opus coding
agent at `effort="max"` does consume the user's plan. The instruction for this
stretch was to spend no money; subscription usage is not dollar spend, but it is
not obviously free either, and spawning that unattended was not authorized. **Not
started.** Everything up to it is built, so the loop can begin as soon as the
user decides:

- let me drive iterations as the proposer inside this session, or
- authorize spawning fresh `claude` CLI sessions per iteration (matches the
  reference, gives each iteration a clean context), or
- run iterations manually with me preparing each candidate.

`PROPOSER.md` holds the instructions either way: the inverted objective, the
noise-injection trap and its control, the hard constraints (no reading
`runs/human/`, `MAX_KEYS` needs a psychological argument), the axes with their
numbers, the per-task noise floor, and the infrastructure gotchas already paid
for.

## Running notes

**Job 18491587 — local baseline (candidate 0), 8 search tasks.** Submitted after
the gate failure, since the baseline must be measured on the same serving stack as
the candidates. Running. When it lands:

    python meta_harness/score_candidate.py meta_harness/runs/iter0/baseline \
        --json-out meta_harness/logs/baseline_scores.json

That produces the per-task vector every later `delta_vs_baseline` and per-task
floor is computed against.

**Seed population now.** Three of the spec's four baselines exist and all pass
the offline interface check:

- `baseline` — released compactor unmodified; the number to beat and a plumbing
  test.
- `random_decay` — adversary, lower anchor. Must score well on humanlikeness and
  badly on A2, or the axes are not discriminating.
- `full_context` — control, upper anchor. Bottleneck removed, stimulus kept
  verbatim. Isolates how much humanlikeness comes from the memory module rather
  than the base model. Expected worst; if it is not, the 4-slot bottleneck is not
  doing the work it is credited with, which would be a finding in its own right.

The fourth, the summarizer, already exists in `bench` as `SummarizerAgent` with
its own `sum_*` task registrations, so it is run as a separate family rather than
injected as a candidate.

**Not started, and why:** the outer loop. See the proposer-auth blocker above.
Everything it needs is built and tested.
