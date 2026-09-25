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

### Verdict: the axes discriminate. Search is meaningful. Proceeding.

Both controls turned out to be threats of different kinds, and a different guard
caught each one.

**`random_decay_v2` — the calibrated-noise threat, caught by A1.** This is the
textbook case the error-structure axes were built for:

| | best_span | A1 sub-span leak | mean HL |
|---|---|---|---|
| human | **6.88** | **0.087** | — |
| baseline | 18.4 | 0.130 | 0.7861 |
| random_decay v1 (uniform 0–0.9) | 2.0 | 0.500 | 0.6977 |
| **random_decay v2 (uniform 0–0.35)** | **8.2** | **0.249** | 0.7727 |

Calibrated noise got the *aggregate* statistic nearly right — best_span 18.4 → 8.2
against a human 6.88, far better than the baseline — while making the *error
structure* twice as wrong, leaking 0.249 below its own span against a human 0.087.
That is the signature of stochastic dropping, and A1 fired on it precisely.

The mean tells the rest of the story: v2 is at 0.7727 against the baseline's
0.7861, a delta of −0.013, which is **inside** the 0.026 min credible mean delta.
So on humanlikeness alone v2 is indistinguishable from the baseline, while on A1 it
is clearly distinguishable and clearly worse. Had the evaluation been mean
humanlikeness only, this candidate would have passed as a wash and its broken error
structure would have been invisible. It also failed the per-task floor on
digit_span_reverse (−0.070 against a 0.059 threshold).

It failed its own pass condition (needed mean ≥ 0.836) for the third time across
two dial settings, so **no v3.** Two calibrations bracketing the human distribution
were enough: undirected noise does not match a human score distribution, because
the noise has to be the right size on every task simultaneously.

**`full_context` — the non-noise threat, caught by A3.** The more convincing of the
two validations, and I under-claimed it initially. This was not a metric attack at
all: it is a legitimately *better* harness that genuinely improved a real task
(N-Back +0.157, to 0.948 against a human 0.866) — and the axes rejected it anyway,
on A3 BLEU 0.3047 against a human 0.002 and 411 recall words against 137. An axis
doing discriminating work against a serious candidate is stronger evidence than an
axis rejecting deliberate noise.

So: humanlikeness alone is gameable and was gamed; the axes caught both threats by
different routes; the search can proceed.
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

## Iteration 1 — `displacement`: mechanism confirmed, candidate rejected

Job 18496607, 18m18s. One method body changed: overflow displaces the least recently
refreshed entry instead of refusing the write. Capacity, prompts, tools and the
tool-call cap all untouched.

**The mechanism claim is confirmed, and cleanly.** The proposer's diagnosis was that
the baseline's refuse-on-full state jams the store at n=3 and suppresses *responses*
rather than degrading judgement. Pre-registered thresholds, and what happened:

| prediction | threshold | observed | |
|---|---|---|---|
| n=3 `answered` rises | ≥ 12 of 14 (baseline 6.82) | **14.00** | PASS |
| n=3 `acc_over_answered` unchanged | 0.687–0.787 | 0.744 | PASS |
| n=3 slots still saturated | ≥ 3.5 of 4 | 3.98 | PASS |
| digit span untouched | — | A1 and best_span *identical* (0.1298, 18.4) | PASS |
| story recall regresses | Δ < 0 | −0.0509 | PASS (as predicted) |

The "answers more eagerly" objection is ruled out arithmetically. N-Back is 2AFC, so
chance is 0.5. If the 7.18 additional responses were guesses, `acc_over_answered`
would be (6.82·0.737 + 7.18·0.5)/14 = **0.616**; observed is **0.744**, implying the
additional responses alone were answered at **0.751** — slightly *better* than the
trials the baseline already answered. The extra responses are informed.

The predicted story-recall mechanism is also confirmed, in the task it lost. Most
common surviving keys:

    baseline       setting, beginning, first encounter, pie man, main_character
    displacement   pie man, second appearance, key_moment, emotional_impact, ...

`setting` and `beginning` drop out entirely: the kept chunks shift from primacy and
orienting information to later events, exactly as pre-registered. **Primacy
protection is therefore the named iteration-2 ingredient, with direct evidence
rather than a guess.**

**But the candidate is rejected, and the reason matters more than the headline.**

    all 8 search tasks                     0.7861 -> 0.8122   +0.0261
    excluding word_recognition             0.8277 -> 0.8378   +0.0100
    the 5 tasks that measure the store     0.9296 -> 0.9148   -0.0148
    the 3-task gist subgroup               0.9317 -> 0.9025   -0.0293

The headline +0.0261 — which only just clears the 0.026 min credible mean delta — is
bought entirely by the two leaky tasks. It fails the per-task floor on
semantic_story_recall (−0.0509).

> **CORRECTED in iteration 2, twice.** Two of the three rejection grounds stated
> here were wrong, though the verdict stands.
>
> **The −0.0148 is not a finding.** That subgroup's mean delta has
> SE = sqrt(ΣSE²)/5 = 0.0157 from `logs/metric_noise.json`, so its min credible
> delta is **0.0314** and −0.0148 is comfortably inside noise. Worse, **93.3% of
> that subgroup's variance is the two digit-span cells** (SE 0.0698 and 0.0294),
> which displacement moved by 0.0000 and +0.0138 — so the subgroup was mostly
> measuring the two tasks the candidate provably did not touch. Drop them and the
> 3-task gist subgroup (story, craft, narrative_qa) has min credible delta 0.0136,
> against which **−0.0293 is credible**. That is the defensible version of the
> claim. Found by iteration 2a's proposer; verified here.
>
> **The A3 guard firing was an artifact.** BLEU's brevity penalty made A3's
> enforced quantity a length proxy: pooled over 1000 story-recall rows,
> Spearman(recall length, BLEU) = 0.822, and all 121 rows under 60 words score
> exactly 0.0000, min and max alike. So the BLEU guard punished lengthening toward
> the human mean while the A3 *word* guard rewarded it — the two halves of A3
> contradicted each other, and displacement tripped that contradiction. Under the
> replacement axis (clipped 4-gram precision, no brevity penalty, medians)
> displacement **passes all guards**, at distance 0.0153 against the baseline's
> 0.0221 — it is closer to the human median than the baseline is. `full_context` is
> still rejected, now doubly, at precision 1.0000.
>
> **What the rejection actually rests on:** the semantic_story_recall floor
> violation (−0.0509 against −0.03) and the credible 3-task gist regression
> (−0.0293 against 0.0136). Both survive. Nothing else cited here does.

**word_recognition's +0.1388 is a fake gain, and the axes caught it.** This is the
clearest vindication of building error-structure axes at all:

| | miss | false alarm | A2 ratio | A2 distance | trials attempted |
|---|---|---|---|---|---|
| human | 0.272 | 0.045 | **6.09** | — | 34.5 |
| baseline | 0.043 | 0.127 | 0.340 | 5.754 | 82.9 |
| displacement | 0.037 | **0.211** | 0.174 | **5.920 — worse** | 68.9 |

Humanlikeness on that task rose because false alarms nearly doubled: the model claims
"old" for new words *more* often, which is the exact opposite of the human
conservative bias it is supposed to be approaching. The aggregate moved toward humans
while the error structure moved away, and the covariate moved too. A mean-only
evaluation would have banked this as the candidate's second-best result.

I had guessed the opposite — that displacement would make the model conservative by
telling it what it had lost, raising the miss rate. A2 refuted that outright.

**Two prediction failures, of different kinds.**

1. **`word_recognition` moved 0.1388 against a predicted < 0.121.** The proposer said
   a large move here would disconfirm the third-leak reading. On the evidence it does
   not: the ceiling group only fell from 36 to 30 of 50, so the ~43 participants
   reading the studied list off the recall prompt are still doing exactly that. What
   changed is that the store now actively misleads the handful that consult it. The
   leak stands; the prediction was miscalibrated because on a distribution this
   bimodal, six participants moving is worth a lot of W_1.

2. **The buffer-compliance test was void, not failed.** The proposer called it "the
   sharpest one" because `answered` can rise for dull reasons and phase compliance
   cannot. It turns out the n=3 buffer response is a fixed *positional* pattern, near
   identical in both runs — slot 1 `No response` ×50, slot 3 `Different` ×50, slot 2
   mostly `Same` (43 vs 48). It cannot move, so it tested nothing. My checker reports
   FAIL faithfully, which is correct behaviour for a badly specified prediction, but
   the honest reading is "inert metric", and the guessing-floor arithmetic above is
   the replacement. Worth noting n=2 buffer compliance actually *declined*
   (`No response` at slot 1: 27 → 16).

**Unexplained and carried forward:** A3 BLEU rose 0.0031 → 0.0413 (guard fired) with
recall length 121 → 146 words, which moved *toward* human 137.2. The candidate
asserted displacement "cannot cause regurgitation — it strictly reduces what is
retained." That was wrong, and why is not yet established. Likeliest reading: chunks
kept by recency are later and less abstracted than chunks kept by primacy, so what
survives is closer to surface form. Iteration 2 must account for it.

> **RESOLVED in iteration 2, and my "likeliest reading" above was wrong.** It is
> not that recency-kept chunks are less abstracted. Measured 4-gram precision of
> each chunk against the story by write position is flat — 0.066, 0.068, 0.045,
> 0.073, 0.029, 0.113 for positions 1–6 — so position 5 is the *least* verbatim and
> position 1 is as verbatim as position 6. There is no primacy/recency gradient in
> verbatimness to appeal to.
>
> The rise is a **brevity-penalty artifact of recall length**, i.e. a defect in the
> axis rather than a fact about the candidate. Holding displacement's own written
> value text fixed and varying only which subset is retained gives a 7× BLEU swing
> from identical sentences: refuse 0.0048 @ 122 words, LRU 0.0335 @ 141, ACT-R
> 0.0087 @ 124. Within a single row, adding chunks: 39 words → 3.5e-9, 58 → 5.7e-6,
> 95 → 0.0034, 119 → 0.0282. Across runs the relation is monotone in length alone:
> 65 → 0.0001, 121 → 0.0031, 146 → 0.0413, 411 → 0.3047.
>
> Verified independently within runs rather than only between them: pooled
> Spearman(length, BLEU) = 0.822 over 1000 rows, and in the 0–60 word bin all 121
> participants score exactly 0.0000, min and max alike — BLEU cannot represent a
> verbatim short recall at all. A3's enforced field is now brevity-penalty-free
> 4-gram precision, on which the human within-record Spearman(length, precision) is
> 0.128, i.e. humans vary in length and verbatimness independently. Found by
> iteration 2a's proposer; the independent within-run check and the axis replacement
> are mine.

**Net:** the search's first real mechanism result is a confirmed diagnosis of a
harness defect — a capacity limit implemented as refusal produces response omission,
which is a failure humans never show — obtained without touching capacity, prompts,
tools or compute. The candidate that demonstrates it is not an improvement in
humanlikeness on the tasks that measure memory.

## Per-task comparability audit (all ten tasks)

Four of the first four tasks I looked at had a defect in which the human and model
sides were not measuring the same thing. That rate made it worth auditing the rest
systematically rather than waiting to trip over them. Doing so *before* reading a
humanlikeness number as a fact about memory is the single most useful habit this
project produced.

| task | comparable? | defect, and what fixed it |
|---|---|---|
| digit_span_forward | after correction | human staircase terminates on double failure, model ran all 19 spans → `protocol_match.py` |
| digit_span_reverse | after correction | same |
| nback | after correction | model scored per `(participant, n_level)`, human pooled over levels → `nback_levels.py`, per level on both sides |
| word_recognition | **no** | recall prompt embeds the studied list verbatim; 36/50 participants read the answer off it. Not fixable without changing `recall()`'s context construction |
| variable_mapping | **no** | store is off the causal path (674/1500 questions answered at 0.985 with the key evicted); human task also terminates at 3 strikes with a `/10` denominator while the model answers all 10 |
| semantic_story_recall | **yes** | none — verified, see below |
| narrative_qa | yes | none; fixed 10-question denominator, no early termination |
| craft_task | yes | none; fixed 15-question denominator, all 54 humans completed 3 trials |
| map_task | yes | none; same structure as craft |
| factual_qa | yes | none; same structure as narrative |

`semantic_story_recall` was the one I most expected to be broken, because the human
`embeddingSimilarity` is a number precomputed by the web app while the model's is
computed by `bench` with `all-MiniLM-L6-v2`. Different embedders would make the two
distributions incomparable while looking perfectly fine. Tested directly by
recomputing the human similarity from `payload.recallText` and the story transcript
with MiniLM and comparing against the stored value, over all 53 usable human records:

    stored (web app)   mean 0.6041  sd 0.1282
    recomputed MiniLM  mean 0.5911  sd 0.1220
    mean |diff| 0.0346   max |diff| 0.1054   corr 0.9431

Same embedder, small residual attributable to text preprocessing. So A3 and the
story-recall humanlikeness are sound, and the four remaining MCQ-style tasks are
sound by construction.

**Net: three of the eight search tasks do not measure the memory module** —
`nback` and `variable_mapping` because the study history stays in context, and
`word_recognition` because the studied list is re-presented at recall. Five do.
That is the real search space, and it was not visible from the released numbers.

### Bug found while scoring wave 0

`score_candidate.py` looked for `manifest.json` in the run dir it was handed, but
that is bench's per-model output dir (`.../<cand>/<model>/`) while
`run_candidate.py` writes the manifest one level up. The lookup always missed, so
every record fell back to `run_dir.name` — the *model* name, identical for every
candidate. With `--record` replacing rows by id, scoring a second candidate would
silently overwrite the first. It was masked until now only because the earlier two
rows happened to be written with an explicit `--id`. Fixed to check both
locations; all three wave-0 rows re-scored so each carries manifest provenance.

---

## Iteration 2 — four candidates, four disjoint surfaces, and five corrections to
## the evaluation contract

The wave is four candidates plus one ablation arm, each owning one mechanism
surface, partitioned in advance so single-mechanism attribution survives. All four
are built on the plain baseline rather than on `displacement`, so exactly one method
body separates each from the control.

| candidate | surface | what it tests |
|---|---|---|
| `primacy` | `WorkingMemory.write_key` eviction order | recovers displacement's story-recall loss; evicts by lowest ACT-R base-level activation, so the middle dies first and eviction proceeds outward |
| `chunk_limit` | within-slot content | bounds enumerated elements per value at `MAX_KEYS`, so a slot is a chunk rather than a list |
| `serial_recognition` | `recall()` trial-list presentation | closes leak 1 — the studied list is visible at recall |
| `episodic_reset` | `step()` conversation history | closes leaks 2 and 3 — full history retained, so the store is decorative on nback and variable_mapping |
| `serial_recognition_open` | ablation of the above | same source, mask off: distinguishes memory from framing |

Jobs 18522417 (four arms) and 18523331 (`chunk_limit`). `primacy` completed in
859.9s with all row counts correct.

### Two candidates are not scoreable in the ordinary way, and that is recorded in code

`serial_recognition` is recorded with `--instrument`, which keeps the row in full
but excludes it from frontier derivation. Closing the word-recognition leak makes
that task's humanlikeness rise **arithmetically**: with the baseline at 0.816
against a human 0.315, even a point mass at zero scores about 0.685. Both frontier
axes break at once — the mean is measured against a reference the candidate
invalidated, and A2 stops meaning for that row what it means for every other row,
because pre-closure A2 is a mixture of two populations doing different tasks
(ceiling group miss 0.001 / fa 0.011, floor group miss 0.400 / fa 0.767). The
candidate's own author asked that its headline not be quoted as a gain.

The exclusion is validated in both directions in `test_history.py`, against a
fixture that is deliberately the strongest row in the set: it beats every other row
on mean *and* on A2 and passes all guards, so only the flag can exclude it. With the
flag it is absent from the frontier; with the flag stripped it is the sole frontier
member, dominating all three others.

`chunk_limit` is the **control** for it. The ceiling group was measured to be
reading the studied list off the prompt — 36 of 50 participants score a perfect
100/100 while their own store decides at most 0.640 of old trials, mean 0.226 — so
removing 74% of the store's content should leave word_recognition unmoved. Its P13
asserts exactly that at the same 0.121 threshold iteration 1 failed.

### Five defects in my own evaluation contract, all found by the proposers and all
### verified here before acceptance

**1. A3 guarded recall length, not verbatimness.** See the correction inline above.
Enforced field is now brevity-penalty-free clipped 4-gram precision on medians;
BLEU is retained with `bleu_enforced: false` because five runs are scored against it
and `full_context`'s 0.3047 is real regurgitation.

**2. The 5-task "store-measuring" subgroup claim was inside noise.** See above.

**3. A4's assignment window was undercounted about twofold.** `model_trials`
computed `turn = q.get("turn", q.get("question_index", 0))` and filtered assignments
by `a["turn"] < turn`. Model question records carry no `turn` field, so it always
fell back to `question_index` (1–10) and compared it against assignment turns
(1–20). Assignments are interleaved, not front-loaded: `variable_mapping.py:145`
emits a question every `TURNS_PER_QUESTION = 2` assignments, so question *k* follows
assignment turn 2*k* and the participant has seen all of 1..2*k*. The filter kept
*k*−1, so intrusions from the unseen half were misclassified as `novel_guess`.
`TURNS_PER_QUESTION` is now imported from the task so the assumption cannot drift.
The proposer's suggested fix was itself off by one pair.

Fixing it showed raw `intrusion_share` is not comparable between sides at all: it
saturated at 1.000 for the baseline against a human 0.691, which reads as the model
being more intrusion-prone when it is only more exposed. Model chance is 0.889
against the humans' 0.577. Above chance the sides agree — model +0.111, human
+0.114 — so `intrusion_above_chance` is now the comparable quantity.

**4. A4's `rc_ratio` has an error-count-dependent ceiling, so the old guard could
never have rejected anything.** `relation_count` is `len(mapping)` and saturates at
10, so the attainable maximum rises with the error count: 1.2525 at 12 errors,
1.2575 at 35, 1.2857 at 150, 1.3750 at 400. `displacement` scored **exactly** 1.2575
at **exactly** 35 errors — its arithmetic maximum, `rc_mean_error` 10.000 — and
`full_context` scored exactly 1.2525 at exactly 12. A fixed 1.15 threshold on the
raw ratio is therefore not scale-free. A4 now reports `rc_ratio_ceiling` from the
run's own distribution and `rc_ratio_normalized` between noise (0.0) and that
ceiling, and the guard compares the normalized value.

Recomputing the human reference on the same footing **inverts the interpretation**:
humans score 1.3867 against a ceiling of 2.0372 over 152 errors, i.e. normalized
**0.3728**. Humans sit at 37% of their attainable interference structure, not near
it. `displacement` sits at 1.0 and the baseline at 0.67 — so the model's errors are
*more* load-ordered than humans', the opposite of what the raw ratios suggested.
"Approaching the human 1.386" is withdrawn as a target for any single model run.

Also withdrawn: my brief to iteration 2c claimed A4 had never fired on a real run.
False — `displacement` produced 35 errors with `trustworthy: true`.

**5. variable_mapping's two sides are scored by different formulas.** The fifth
comparability defect and the most severe kind: not a protocol mismatch but two
different quantities sharing a denominator. Human is `sum(q.correct)/10`, a correct
count; model is `relation_count` of the last consecutively correct question, which
saturates by question 5, so **every error after question 5 is invisible to the
model's score**. `displacement` erred on 25 runs and 24 of them still score 1.0.

This reframes what the task was telling us. The "99% at 1.0, two unique values"
point mass, recorded earlier as evidence the store is off the causal path, is
substantially a **scoring** artifact — the errors exist and the formula discards
them. On the human formula: baseline {1.0: 138, 0.9: 12}, displacement
{1.0: 125, 0.9: 18, 0.8: 4, 0.7: 3}. Reported as
`axes()["variable_mapping_matched"]` beside the raw figure, since redefining the
raw one would make the history incomparable. Analysis-side, not in the harness: a
harness must not know the scoring protocol.

### A premise of mine that a proposer disproved: the chunk defect is not task-general

I briefed `chunk_limit` that unbounded value length was "task-general rather than
specific to one leak", and that digit-span values were "exactly the kind of list
your bound would cut". Measured elements per value on the baseline's `final_kv`,
reproduced independently here:

| task | elements/value (max) | elements/store (max) | can the bound bite? |
|---|---|---|---|
| word_recognition | **13.21 (99)** | **44.90 (180)** | extreme |
| semantic_story_recall | 3.89 (9) | 13.98 (23) | already at the bound |
| narrative_qa | 3.76 (9) | 13.54 (25) | already at the bound |
| digit span fwd/rev | 1.94 (13) | 6.75 (25) | weak, about one digit at spans ≥15 |
| nback | 1.13 (**2**) | 2.45 (6) | **no** |
| variable_mapping | 1.00 (**1**) | 3.51 (4) | **no** |
| craft_task | 1.00 (**1**) | 3.33 (4) | **no** |

One extreme task, two already at the bound, one weak, three that cannot move — and
the extreme one is exactly the leaky task whose score is uncreditable. So the
8-task mean is not expected to move credibly, which the candidate states *before*
the run. Digit span is disqualified as its control by measurement; `craft_task` is
the derived control, with 0 of 150 rows touched and store and tool results
byte-identical.

### The n-back leak is worse than recorded

Share of n-back `final_kv` values containing a bare capital letter, baseline:
n=1 50/50, n=2 77/77, **n=3 4/198**. At n=3 the store holds no letters at all and
the agent is still 0.737 accurate, so those answers come from the dialogue history.
`displacement`'s n=3 humanlikeness of 0.9387 is therefore leak-derived, and "n-back
per level is closed" was never the same claim as "the memory module closed n-back".
`full_context` is the contrast: 53/53 values carry letters.

### Iteration 2 results — no candidate joins the frontier, and three of the four
### failed in ways that taught us something

Jobs 18522417 (47m, four arms) and 18523331 (17m), all exit 0.

    arm                        mean     floor   guards   instrument
    serial_recognition_open  0.8631      ok       ok       YES (partial: 1 task)
    serial_recognition       0.8134      ok       ok       YES
    primacy                  0.7945    FAIL       ok        -
    episodic_reset           0.7892    FAIL       ok        -
    baseline                 0.7861      ok       ok        -
    chunk_limit              0.7799    FAIL       ok        -

**The baseline remains the sole frontier member.** Note that the two top rows by mean
are the instrument rows, on a result their own ablation voided — the self-enforcing
marking earned its keep on its first real use.

**`primacy` — mechanism confirmed, rejected on two floors.** The ACT-R eviction rule
did exactly what it claimed. U-shaped survival, the fraction of overflowing story rows
retaining both the first- and last-written key, went **0.000 → 1.000** where both
`displacement` and the baseline scored exactly 0.000, with the retention minimum at a
middle write position as predicted. `semantic_story_recall` recovered 0.8964 → 0.9477,
inside its pre-registered two-sided band, and it did so while keeping displacement's
response fix intact (n=3 answered 14.0, keys_held 4.0, accuracy 0.76). Digit span was
untouched. But `craft_task` fell 0.0451 and `narrative_qa` 0.0309, both past their
floors, so recovering story recall cost two other gist tasks. 6 PASS / 3 FAIL / 4
INCONCLUSIVE.

**`episodic_reset` — the strongest mechanism result in the project, and VOID by its
own pre-registered control.** Closing the history leak worked, unambiguously:

    n=3 store carries letter identity   0.0202 -> 1.0000   (full_context is 1.0)
    variable_mapping A4 n_errors            12 -> 509      (trustworthy at scale)
    A4 rc_ratio                         1.1682 -> 1.3170   (ceiling 1.434, normalized 0.7302)
    vm share at ceiling                 0.9867 -> 0.2733,  2 -> 4 distinct scores
    vm humanlikeness                    0.3554 -> 0.6764   (+0.3210)
    vm humanlikeness, matched formula   0.3587 -> 0.7295

That is a +0.32 gain on the task the candidate targeted, the point mass broken, and
A4 finally trustworthy at 509 errors rather than 12. And it is all VOID, because the
candidate pre-registered that a failure of its n=1 control invalidates P1–P7
regardless of their values — and n=1 failed hard: **36 of 50 participants answered
nothing at all**, mean answered 2.06 of 14, keys_held 0.28. At n=1 a single
overwritten key suffices, so that can only be broken mechanics, not a capacity limit.
`nback` fell 0.2939, violating its floor. 1 PASS / 2 FAIL / 2 INCONCLUSIVE / **6
VOID**.

This is the discipline paying for itself. Without the control the honest-looking
+0.32 would have been bankable. The proposer also pre-committed the diagnosis and the
fix: a P9 failure localised to buffer-period turns means lost sequence position, whose
remedy is pinning the instruction turn. That is iteration 3's first move.

**`serial_recognition` — VOID via its own ablation, which is exactly why the ablation
was required.** The masked arm looked like a triumph: word_recognition +0.2439,
ceiling group 36 → 0, mean score 0.816 → 0.055. But the **open arm collapsed too** —
ceiling 1 of 50, mean score 0.177 — and the open arm is identical in every respect
except that the other 99 trial lines stay visible. So the collapse is caused by the
serial framing, the call pattern or the stitching, not by removing information.
Without that arm this wave would have recorded a fake +0.244 as "closing the leak
works", and the arm cost 105 seconds. 9 PASS / 2 FAIL / 3 INCONCLUSIVE / 3 VOID, with
P13 and P14 tagged ENTAILED so the mean cannot be cited as a gain.

**`chunk_limit` — the bound worked exactly, and it independently confirmed the leak
diagnosis that `serial_recognition` could not.** Elements per value went 13.21 → 3.267
with max exactly 4, store max 99 → 16, and distinct studied words in the store
17.78 → 4.92. Max elements per value is ≤ 4 on **all eight tasks**. `craft_task` is
bit-identical to the baseline, as predicted.

The decisive row is P13. After removing 74% of the store's content and 75% of its
distinct studied words, **word_recognition moved −0.0051**, far inside its 0.121
floor. The score does not depend on the store at all, because it is read off the
studied list in the recall prompt. That is the cleanest possible confirmation of the
leak, obtained from a candidate that was not trying to close it. Rejected on
`narrative_qa` (−0.0310); P11 failed because the bound cost three spans of digit span
(best_span 18.4 → 15.4) rather than the predicted one digit, and P12 failed because A2
moved *away* from the human conservative bias — consistent with P13, since a task read
off the prompt cannot respond to store content. 11 PASS / 3 FAIL / 1 VOID.

### ~~The serving stack is deterministic, which makes every movement attributable~~

> **WRONG. Retracted in iteration 3, and this was my error, not a proposer's.** The
> claim below rests on scored humanlikeness agreeing exactly, and that does not
> establish determinism: humanlikeness is a Wasserstein distance over a score
> *distribution*, so it is invariant both to which participant got which score and to
> any generation change that does not alter the score. Two runs can differ in content
> on dozens of rows and produce an identical statistic.
>
> Iteration 3a challenged it and was right. Comparing generated text with the
> server-assigned tool-call ids stripped — `chatcmpl-tool-<hex>`, minted fresh per
> call, which differ between any two runs and are not generation content — on
> `craft_task` against the baseline:
>
>     chunk_limit         provable no-op on craft    7 of 150 rows differ   delta  0.0000
>     serial_recognition  provable no-op on craft   12 of 150 rows differ   delta  0.0000
>     episodic_reset                               24 of 150 rows differ   delta -0.0280
>     primacy                                      57 of 150 rows differ   delta -0.0451
>
> The divergences are real: the model writes `"rule1": "A and B make D"` in one run and
> `"A + B -> E"` in another. vLLM with continuous batching at
> `max_parallel_participants: 50` is not bitwise reproducible at temperature 0, because
> batch composition shifts reduction order.
>
> **Consequences, as amended by iteration 3b — which corrected this correction.** My
> first version of this retraction said "a no-op arm moved craft −0.0280, so craft's
> floor is understated". That is wrong: **`episodic_reset` is not a no-op on craft.**
> `WorkingMemoryAgent.encode()` calls `self.step()` internally (`wm_agent.py:337`,
> comment "Use step() internally"), and `step()` is exactly the surface
> `episodic_reset` rewrites. Verified at the source. So its craft move IS attributable
> to it after all, and craft's empirical no-op band is **0.0000 and 0.0000** — the two
> genuine no-ops, `chunk_limit` and `serial_recognition`, produced 7 and 12 rows of
> phrasing variation and moved the scored statistic not at all.
>
> So the effect of generation noise on a scored statistic is **strongly
> task-dependent**, and that is the durable lesson rather than a blanket floor
> inflation:
>
>     craft_task     19 rows of phrasing variation across two no-ops -> 0.0000 twice.
>                    A 5-question 2AFC score is too coarse to register a rewording.
>     narrative_qa   serial_recognition (a genuine no-op here) changes 29 of 50 rows'
>                    scores for a delta of -0.0034. Least characterised cell in the set.
>     story recall   188 of 200 rows differ for -0.0026. Continuous score, tiny effect.
>
> `metric_noise.py` still cannot see generation-level variation — it resamples the model
> side of a FIXED row set — but craft is evidence that the consequence is sometimes
> exactly zero, not that every floor is too tight. Job 18536738, a repeat baseline, is
> measuring the real run-to-run band, and narrative_qa is the quantity that matters: it
> is the cell on which three of four iteration-2 candidates were charged a violation.
>
> `primacy`'s craft regression is real — 57 differing rows against a no-op band of 7–12,
> and a scored move of −0.0451 against 0.0000.
>
> **MEASURED, job 18536738 — and it settles the question in the conservative
> direction.** A repeat baseline, same harness, config and seed, so every per-task
> delta is pure run-to-run variation. Full table in `logs/run_to_run_floor.json`:
>
>     craft_task              0.0000    0 of 150 rows differ    floor 0.030
>     digit_span_reverse      0.0000    0 of 190                floor 0.059
>     word_recognition        0.0000    0 of 50                 floor 0.121
>     semantic_story_recall  -0.0012  156 of 200                floor 0.030
>     nback                  -0.0023                            floor 0.060
>     variable_mapping       -0.0044    1 of 150                floor 0.030
>     narrative_qa           -0.0065   13 of 50                 floor 0.030
>     digit_span_forward     +0.0152                            floor 0.140
>     8-task mean            +0.0001                     min credible 0.026
>
> **No floor needs loosening.** Every enforced floor exceeds the measured variation,
> most by a wide margin. My worry that the instrument was too tight, and iteration 3a's
> second-order claim that craft's floor understates run-to-run variation, are both
> refuted — craft is exactly reproducible at the score level.
>
> The durable lesson is that **generation variation does not propagate to the score
> uniformly.** Story recall has 156 of 200 rows differing for −0.0012, because a
> continuous embedding similarity averages the variation out. Craft and word recognition
> have ZERO rows differing, because a coarse multiple-choice score cannot register a
> rewording. `narrative_qa` is the sensitive cell: 13 of 50 rows for −0.0065, the largest
> scored effect per differing row in the set. So a differing-row count is still the more
> sensitive mechanism signal, but it must be compared against this table — 156 of 200 is
> the *no-change* baseline on story recall, not evidence of anything.
>
> Two attributions change. `narrative_qa`'s real noise is 0.0065, so `primacy`'s −0.0309
> is about five times it and is a genuine effect; it violates the floor by 0.0009 only
> because 0.030 is conservative, not because the effect is marginal. And
> `digit_span_forward`'s +0.0152 run-to-run is **exactly** the value `primacy` reported
> there, with `chunk_limit` at +0.0183 — so those digit-span movements were variation,
> not mechanism, which confirms `primacy`'s P4 no-change claim for a better reason than
> the one it gave.
>
> **A separate arithmetic error of mine, also caught by 3b:** the enforced floor is
> `max(FLOOR=0.03, NOISE_FLOOR[task])`, so **craft's is 0.030, not the 0.025 I quoted
> throughout.** Consequences: `displacement`'s craft −0.0280 *passed* and I reported it
> as a cost it did not incur, and `primacy`'s narrative_qa violation is **−0.0309
> against 0.030, i.e. by 0.0009** — which is not a result any instrument here can
> support as a rejection ground. Four iteration-2 candidates on four unrelated surfaces
> all landed narrative between −0.0089 and −0.0310.
>
> The original text is kept below because the error and its correction are both part of
> the record.

Worth recording because it licenses reading small deltas at all, and because it
refutes the obvious objection to `episodic_reset`'s P10 failure. Independent
candidates reproduce the baseline **bit-exactly**: `digit_span_reverse` in all four
arms, `craft_task` in two, `digit_span_forward` in two. So there is no run-to-run
nondeterminism to blame for movement elsewhere, the per-task floors are not too tight,
and `episodic_reset`'s craft (−0.0280) and narrative (−0.0100) moves are genuine
effects of its `step()` rewrite. Its claim that the six `encode()`→`recall()` tasks
would be untouched is empirically wrong.

### The VOID machinery fired three times on real data

It was built because iteration 1 scored FAIL on a prediction that could not move.
This wave it caught: `primacy` P2 against `displacement` (both-ends retention 0.000 in
both runs), `chunk_limit` P5 (median 4-gram precision identical to the baseline's
0.0414 to full precision), and the two precondition cascades above. A checker that can
only say PASS or FAIL would have reported six false verdicts here.

---

## Iteration 3 — one creditable gain, one mechanism that cannot reach its target

Jobs 18538066 (three arms) and 18536738 (repeat baseline), all exit 0. The baseline
remains the sole frontier member for the third wave running.

    arm                  mean     nback      vm     craft   narrative   floor
    episodic_reset_v2  0.7993    0.5601  0.6854    0.8907      0.9572    FAIL nback
    primacy_v2         0.7944    0.9442  0.3530    0.8487      0.9423    FAIL craft
    episodic_primacy   0.7943    0.5893  0.6507    0.8472      0.9563    FAIL both
    baseline           0.7861    0.7909  0.3554    0.8907      0.9572

### `episodic_reset_v2` — the precondition passed, so the +0.33 is finally creditable

This is the first time the project's largest single-task gain has been attributable.
n=1 is fully repaired: answered 13.88 against a baseline 13.98, accuracy 0.9964, **zero
silent participants** where `episodic_reset` had 36 of 50, keys_held back to 1.00. P1
was the pre-registered precondition, so P2–P7 are scored rather than voided.

    variable_mapping, raw formula      0.3554 -> 0.6854   (+0.3300)
    variable_mapping, matched formula  0.3587 -> 0.7568
    A4 n_errors                            12 -> 550      trustworthy
    A4 rc_ratio_normalized             0.6661 -> 0.7362   (humans 0.3728)
    n=3 store carries letter identity  0.0202 -> 0.9843

And the structural claim held **exactly**: `craft_task`, `narrative_qa`,
`digit_span_forward` and `digit_span_reverse` all moved by precisely 0.0000. That is
worth noting against its predecessor, which moved craft −0.0280 and narrative −0.0100
while making the same structural claim — so whatever caused those moves was specific to
v1's implementation and is now gone. Unresolved, and carried forward.

**Rejected, because the absorbing state relocated rather than disappearing:**

    level   baseline        episodic_reset      episodic_reset_v2
    n=1     13.98, 0 silent   2.06, 36 silent    13.88,  0 silent   <- fixed
    n=2     13.24, 0 silent  10.16,  0 silent     6.80,  3 silent   <- worse
    n=3      6.82, 0 silent   9.08,  0 silent     2.12, 12 silent   <- much worse

`nback` fell 0.2308, violating its floor. So the control-state block cures the failure
at the level where the store holds one item and induces it where the store is nearly
full. P7 also failed and is the clue: 38 of 1500 variable_mapping answers were
unparseable against a baseline 0, with mean reply length 15.91 characters against 13.10
and a maximum of 147. Prepending four lines to a prompt whose instruction is "output
ONLY one line" has a cost, and the anti-verbosity row was written to catch exactly that.

### `primacy_v2` — correctly implemented, and unreachable on the task it targets

The batch rule never fired. Craft's overflowing rows settled at mean occupancy **4.0**,
which is `primacy`'s value, so P2 — the mechanism-confirmation row — failed, and craft
came in at 0.8487 against `primacy`'s 0.8456. The candidate was measuring its parent.

The rule is not broken. Driving the real `step()` with a stub that emits five
`write_memory` calls in one assistant message settles the store at **3**; five separate
messages settle it at **4**. Both paths behave as designed.

What fails is the premise. 3b argued that encode writes arrive as one parallel
assistant message, from 67 of 200 story rows showing two consecutive refusals with no
`delete_key` between them — the agent could not have seen the first refusal before
issuing the second. But craft's actual pattern is different:

    baseline     w1..w4 written -> w5 "memory is full" -> delete_key rule1
    primacy_v2   w1..w4 written -> w5 written (displaced) -> delete_key -> "not found"

The delete immediately following the refusal proves the agent *did* see the result
before acting, so arrival on craft is serial and `_batch_writes()` correctly returns 1.
A mechanism conditioned on simultaneous arrival therefore cannot engage on the one task
it was designed to repair. The premise was measured on a minority pattern, on a
different task.

~~Two genuine gains survive, neither the headline.~~ `narrative_qa` came in at −0.0149
against `primacy`'s −0.0309, and A3 `precision_distance` at 0.0082 against `primacy`'s
0.0196 and the baseline's 0.0221.

> **RETRACTED within the hour, and this one was mine.** I wrote those up as gains
> attributable to a key-naming side effect and proposed a control run to confirm the
> tool-result string was responsible. The control was unnecessary, because **`primacy`
> and `primacy_v2` were the same harness in this run.**
>
> Two facts settle it. For a *single* eviction `primacy_v2`'s tool-result string is
> byte-identical to `primacy`'s — the plural form only exists for a double eviction, and
> the batch rule never fired, so a double eviction never happened. And replaying the
> real write sequences from both live runs through both memory classes under serial
> arrival gives **zero divergences over 874 rows and 3610 write calls**, comparing store
> contents and returned strings at every call.
>
> So every difference between the two arms is run-to-run variation, which makes the pair
> a free second measurement of it — and a more useful one than the repeat baseline,
> because this harness overflows and the baseline does not:
>
>     narrative_qa            0.9263 vs 0.9423    0.0160
>     digit_span_forward      0.9012 vs 0.8860    0.0152
>     A3 precision_distance   0.0196 vs 0.0082    0.0114
>     word_recognition        0.4706 vs 0.4637    0.0069
>     craft_task              0.8456 vs 0.8487    0.0031
>     8-task mean             0.7945 vs 0.7944    0.0001
>
> **This revises the earlier claim that every floor is amply conservative.** narrative's
> noise reaches **0.0160**, over half its 0.030 floor, not the 0.0065 the repeat baseline
> showed. So one measurement of run-to-run variation was not enough, and a harness that
> overflows is noisier than one that refuses. `primacy`'s narrative violation of −0.0309
> is about twice the observed spread rather than five times it — still probably real, but
> the margin is much thinner than I stated, and no narrative effect below about 0.02
> should be claimed by anything.
>
> The A3 figure is the sharpest illustration: `precision_distance` moved 0.0196 → 0.0082
> between two runs of identical code, so "closest to the human median any candidate has
> reached" described nothing at all.

### `episodic_primacy` — the composition could not be evaluated, by design

Its C1 precondition failed on the same craft occupancy of 4.0, so C2 (additivity) and
C4 (anti-regression) are VOID. That is the import-time assertion and the precondition
row doing their job: the arm reports that it was measuring one mechanism rather than
quietly presenting a composed result. `nback` −0.2016 and `craft` −0.0435 are both
inherited failures.

What is still visible, as C3 reported rather than scored: `nback` in composition
(0.5893) sits between `episodic_reset_v2` alone (0.5601) and `primacy_v2` alone
(0.9442), and `variable_mapping` (0.6507) below `episodic_reset_v2` alone (0.6854). So
the two mechanisms interact rather than add, but with one of them inactive this run
cannot say how.

### Where iteration 4 should go

The leak closure is now established and creditable, and its remaining cost is localised
to a single mechanism: silence at n≥2 when the store is nearly full. That is a narrower
problem than either previous wave faced, and `episodic_reset_v2`'s own P7 failure points
at the cause — 38 unparseable answers and a mean reply of 15.91 characters against the
baseline's 13.10, from prepending four lines to a prompt that demands one line of output.
That is iteration 4.

`primacy_v2`'s batch condition should be **dropped rather than repaired**: serial arrival
is what the agent actually does once writes are admitted, so a rule conditioned on
simultaneous arrival can never engage. And there are no narrative or A3 gains to
re-derive — see the retraction above.

The methodological lesson, which has now cost three separate claims of mine: **a single
measurement of run-to-run variation is not enough to license reading a delta.** The
repeat baseline gave narrative 0.0065 and the primacy pair gave 0.0160 on the same task.
Any future floor argument needs the noise measured on a harness of the same family as the
candidate, not on the baseline.
