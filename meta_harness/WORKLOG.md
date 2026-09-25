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

## Wave 0 — the validity wave

Purpose: decide whether the Pareto setup discriminates at all, *before* spending
proposer iterations on it. Three runs, all on the same local vLLM stack
(Qwen3-30B-A3B-Instruct-2507, bf16, `--repeat 50`), 8 search tasks.

| id | mean | ds_fwd | ds_rev | nback | word_rec | var_map | narr_qa | story | craft | A2 dist | A1 leak | A3 bleu | A3 words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **0.7861** | 0.886 | 0.967 | 0.791 | 0.495 | 0.355 | 0.957 | 0.947 | 0.891 | 5.754 | 0.130 | 0.0031 | 121 |
| random_decay (adversary) | 0.6977 | 0.686 | 0.735 | 0.782 | 0.438 | 0.351 | 0.939 | 0.792 | 0.858 | 5.429 | 0.500 | 0.0001 | 65 |
| full_context (upper anchor) | 0.6387 | 0.598 | 0.568 | 0.948 | 0.364 | 0.347 | 0.865 | 0.608 | 0.813 | 6.056 | 0.032 | 0.3047 | 411 |
| *human* | — | — | — | — | — | — | — | — | — | 0 | 0.087 | 0.002 | 137 |

Three things fall out of this, and none of them were guaranteed:

**1. The objective really does invert, and the anchor proves it rather than
asserting it.** `full_context` — capacity 10 000, stimulus kept verbatim — is the
*most capable* harness in the table and the *least* humanlike, 0.6387 against the
baseline's 0.7861. The 4-slot bottleneck is doing the work it is credited with.
Had `full_context` come out on top, the whole premise would have been wrong.

**2. The A3 guard is not vacuous.** `full_context` regurgitates: story-recall BLEU
0.3047 against a human 0.002, and 411 words against a human 137. Both guards fire.
This is the exact failure mode a mean-humanlikeness score alone would have priced
at merely "somewhat worse" instead of "not recall at all", so the guard is earning
its place.

**3. The real headroom is two tasks, and they pull opposite ways on capacity.**
Five of eight tasks are already at 0.89–0.97 on the baseline and have nothing left
to win. What is actually open:

- `variable_mapping` **0.355** — flat across all three candidates (0.347–0.355).
  I first read this as "an interesting non-capacity problem". That reading was
  wrong; see *The leak* below. It is not reachable from the memory module at all.
- `word_recognition` **0.495**, and it carries A2, the one axis with headroom
  (distance 5.754 from the human miss/FA ratio of 6.09).
- `nback` **0.791** — and `full_context` takes it to **0.948**. N-Back is the task
  where the model is *worse* than humans, so it is the one task that wants *more*
  capacity, and removing the bottleneck duly fixes it while destroying everything
  else.

That last point is the design tension worth handing the proposer: a single global
`MAX_KEYS` cannot satisfy both N-Back and digit span. A mechanism that is
demand-sensitive — interference or displacement rather than a hard slot count —
could in principle get both, and that is a psychological claim with a literature
behind it rather than a knob-twiddle.

**Axis validation is still open**, pending `random_decay_v2` (job 18494067).
`random_decay` v1 failed as an adversary in the informative direction: it did not
game the metric, it overshot the human distribution badly (digit-span best span
18.4 → 2.0 against a human 6.88, A1 leak 0.500 against 0.087) and *lost* 0.088 of
humanlikeness. A weak adversary leaves the axes unvalidated, not validated, so v2
retries the attack calibrated — uniform(0, 0.35), applied once after encode.
Pass condition, written before the run: mean humanlikeness ≥ baseline + 0.05
*while* A2 distance stays ≥ 2× the baseline's. Passing both invalidates the Pareto
setup and it must be redesigned before any iteration is spent; failing the first
again is evidence that matching the human distribution with pure noise is harder
than assumed, which is the result in the metric's favour.

## The leak: two of the eight tasks do not test the memory module

Found while checking whether `variable_mapping`'s flatness was a real mechanism or
an artifact. It was an artifact, and a structural one.

`WorkingMemoryAgent` keeps conversation history across `step()` calls.
`reset_messages()` exists but **no task ever calls it**. `recall()`, by contrast,
calls `llm.generate` with a single fresh prompt built from the KV store alone. So
tasks split into two regimes by which method they answer through:

| regime | route | tasks | mean HL |
|---|---|---|---|
| bottlenecked | `encode()` → `recall()`, KV only | ds_fwd .886, ds_rev .967, word_rec .495, story .947, craft .891, narr .957 | **0.854** |
| leaky | answers via `step()`, full study history in context | nback .791, variable_mapping .355 | **0.573** |

(`craft_task` and `narrative_qa` reach `recall()` through `run_wm_mcq_trial`.)

**The two least humanlike tasks are exactly the two where the 4-slot bottleneck is
bypassed.** Measured directly on `variable_mapping`, conditioning each question on
what the store actually held at the time:

| store state at question time | n | accuracy |
|---|---|---|
| queried name present, value correct | 795 | 1.000 |
| queried name present but value **stale** | 31 | 0.935 |
| queried name **absent** — store had evicted it | 674 | 0.985 |

45% of questions ask about a name the store no longer holds, and the model answers
them at 0.985. When the store actively contradicts the truth, the model overrides
it and is still right 93.5% of the time. The store is decorative here; the answer
comes from the dialogue history.

That fully explains the flatness: capacity 4, capacity 10 000 and random decay all
give 0.992, because none of them are on the causal path. It also means
**`variable_mapping`'s apparent 0.645 of headroom — the largest single block in the
table, 0.081 of the 8-task mean — is unreachable by any change to the memory
module.** A proposer aiming there would burn a budgeted iteration for nothing, or
worse, discover that the only way to move it is to manufacture failures.

Two further protocol facts about `variable_mapping`, which compound the above:

- The human task terminates at **3 strikes**; the model answers all 10 questions.
  Humans answered a mean of **4.93** questions (95% fewer than 10), but score is
  `correct/10` either way. So the human 0.394 mixes lower accuracy with shorter
  exposure. Accuracy *among answered* questions is **0.731** against the model's
  0.992 — a real gap, but much smaller than the score gap implies.
- Protocol-matching the 3-strike rule onto model trials changes the model's score
  by nothing (0.992 → 0.992), because at 0.8% error it almost never accumulates 3
  strikes. So unlike the digit-span case, protocol-matching does not rescue this
  comparison; the gap genuinely requires the model to fail.

### A4, and why it had to exist before any iteration was spent

`variable_mapping` had the largest headroom, the *best* measurement precision
(min credible delta 0.017), and no error-structure axis touching it — A1 is digit
span, A2 word recognition, A3 story recall. That is precisely the shape of a cell
that gets gamed, and the adversary control had no teeth there.

`meta_harness/interference.py` closes it, using two facts about human errors:

- **Humans make interference errors.** Of 152 human errors: 23.0% picked a city
  previously assigned to *that same name* (stale binding), 46.1% a city belonging
  to *another* name, 30.9% a city never assigned. So 69.1% are intrusions against a
  57.7% chance rate from option composition — a real but modest +0.114, ~3 SE.
- **Human errors concentrate on high-interference items.** `relationCount` averages
  **6.18 on errors vs 4.46 on correct**, a ratio of **1.386**.

The ratio is the primary statistic because it is the one noise cannot fake: random
key-dropping produces errors independent of interference load, so its ratio tends
to 1.0. The guard is conditional — a candidate that leaves `variable_mapping` alone
owes nothing, but one that improves it past the noise floor must show either ≥30
errors with `rc_ratio ≥ 1.15`, or it is rejected as an unstructured gain.

At baseline the model makes 12 errors in 1500 questions, so its own A4 is
unmeasurable and reports `trustworthy: false`. That is by design: A4 says nothing
about the baseline and everything about any candidate that starts failing.

## Measurement precision was wrong, and it mattered

The `NOISE_FLOOR` table was human split-half only — uncertainty in the reference
distribution, nothing about the model side being a finite sample too.
`meta_harness/metric_noise.py` bootstraps both and reports the SE of a *difference*
between two runs, which is what a per-task floor actually has to respect:

| task | n_model | min credible \|Δ\| | old floor |
|---|---|---|---|
| digit_span_forward | **10** | **0.140** | 0.036 |
| word_recognition | 50 | **0.121** | 0.075 |
| nback | 150 | 0.060 | 0.025 |
| digit_span_reverse | 10 | 0.059 | 0.026 |
| narrative_qa | 50 | 0.030 | 0.053 |
| craft_task | 150 | 0.025 | 0.041 |
| variable_mapping | 150 | 0.017 | 0.041 |
| semantic_story_recall | 200 | 0.011 | 0.042 |

The digit-span figure is the bad one: `search_set.yaml` cuts those tasks to 190
rows for a 4x throughput win, and `score.py` resolves 190 rows into **10** model
participants. A floor of 0.03 on a quantity with a 0.140 noise band was measuring
nothing. Effective floor is now `max(FLOOR, min_credible_delta[task])`.

Deliberately **not** fixed by re-running at larger n. The primary objective is the
8-task mean, and averaging already fixes it: `sqrt(Σ SE²)/8 = 0.013`, so a mean
delta of **0.026** is credible — inside the existing 0.03 floor. Re-measuring the
baseline at 3.5x cost would have bought precision the objective does not need.
Instead, digit-span regressions are watched via `best_span`, a scalar over all 190
trials rather than 10 pseudo-participants, which caught `random_decay` at 18.4 → 2.0.

That last number is worth stating on its own: **the baseline reaches span 18.4
where humans stop at 6.88.** Digit-span humanlikeness reads 0.886 because the
marginal score distributions happen to line up, while the underlying staircase
behaviour does not. Digit span is less closed than 0.886 suggests, and A1's
`sub_span_leak` cannot see it — `full_context` hit best_span 20.0, never
terminated, had no failures to leak, and so scored *closer* to human on A1 than the
baseline purely by being uninformative. A1 now reports `at_ceiling` and
`best_span_distance` so that cannot be misread again.

### Bug found while scoring wave 0

`score_candidate.py` looked for `manifest.json` in the run dir it was handed, but
that is bench's per-model output dir (`.../<cand>/<model>/`) while
`run_candidate.py` writes the manifest one level up. The lookup always missed, so
every record fell back to `run_dir.name` — the *model* name, identical for every
candidate. With `--record` replacing rows by id, scoring a second candidate would
silently overwrite the first. It was masked until now only because the earlier two
rows happened to be written with an explicit `--id`. Fixed to check both
locations; all three wave-0 rows re-scored so each carries manifest provenance.
