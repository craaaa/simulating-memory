# `rationales/` — STaR against human responses

Training-time lever for the gap this repo measures: out-of-the-box LMs remember far
better than humans do. `bench/` attacks that with prompting (C1–C4) and a compactor;
this package fine-tunes instead, using **STaR** (Zelikman et al. 2022,
[arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) through the
[Tinker](https://tinker-docs.thinkingmachines.ai) API.

The model learns to emit a `<reasoning>` block and then a response. A rationale is kept
only if that response **matches what a real human actually did** — not what was correct.
On trials the human got right the two coincide; on the ones they got wrong, the model
has to reproduce *that person's specific error*.

## Two tasks

| `--task` | An item is | y_i | Human data |
|---|---|---|---|
| `digit_span` (default) | one presented sequence | the digits that participant typed | `runs/human/working-memory-{digit-span,reverse-digit-span}/` |
| `listening_qa` | one (participant, topic, question) | the options that participant endorsed | `application/listening_qa/data/` + the analysis pipeline's `responses.csv` |

Everything the loop does is task-independent; everything task-specific sits behind the
`RationaleTask` protocol in `task.py`. The two implementations are thin adapters in
`tasks/` — over `data.py` / `select.py` / `prompting.py` / `errors.py` / `evaluate.py` /
`plotting.py` for digit span, and over the parallel modules in `listening/` for
listening QA.

The balance story differs by task, and is worth knowing before reading any result:

* **digit span** subsamples to 1:1 success/fail, because otherwise "always recall
  correctly" satisfies the objective. Fails then concentrate at long spans, which is a
  shortcut the per-length eval curve exists to detect.
* **listening QA** subsamples nothing — the pool is already 46.7% exact-correct — and
  splits **by participant**, not by item, because each prompt carries that
  participant's answers to the passage's other four questions. Pooled 46.7% is not the
  reading: across the 80 (topic, level, question) cells the correct share runs 0.09 to
  0.93, and `select-data` flags the five outside 0.15–0.85, since a cell pinned near 0
  or 1 is learnable from the question's shape alone.

---

## Fidelity to STaR, and the deviations

Algorithm 1 from the paper:

```
Input M: a pretrained LLM; dataset D = {(x_i, y_i)}  (w/ few-shot prompts)
1: M_0 <- M
2: for n in 1...N do
3:   (r̂_i, ŷ_i)      <- M_{n-1}(x_i)                  # rationale generation
4:   (r̂rat_i, ŷrat_i) <- M_{n-1}(add_hint(x_i, y_i))   # rationalization
5:   D_n     <- {(x_i, r̂_i, y_i)    | ŷ_i = y_i}
6:   D^rat_n <- {(x_i, r̂rat_i, y_i) | ŷ_i != y_i AND ŷrat_i = y_i}
7:   M_n <- train(M, D_n u D^rat_n)                    # finetune the ORIGINAL model
8: end for
```

Honored here, and easy to get wrong:

| Paper | Here |
|---|---|
| line 3 decodes **greedily, one sample per problem** | `k=1, temperature=0.0` by default |
| line 4 hint is removed from the training prompt | training prompt is zero-shot and hint-free |
| line 7 trains **M**, not `M_{n-1}` | a fresh LoRA client off `base_model` every round |
| line 7 uses **that round's** `D_n u D^rat_n` | never accumulated; the whole split is regenerated each round |
| few-shot prompts used every round | `prompts/fewshot_*.txt`, injected at sampling only |
| step budget grows 20%/round after warmup | `steps_for_round(n) = steps_1 * 1.2**(n-1)` |

Deliberate deviations, all recorded in `run_config.json` under `deviations`:

1. **`y_i` is the human's response, not ground truth.** On success trials the human's
   response *is* the gold answer, so that half is vanilla STaR; only the fail half
   deviates.
2. **1:1 success/fail balancing** of `D` (the paper takes `D` as given).
3. **Hint-leak filter** on rationales (the paper only removes the hint from the
   prompt). `--no-leak-filter` restores paper behavior.

`k > 1` or `temperature > 0` is available but is logged as a deviation — k-sample
rejection is RFT/ReST, not STaR.

### Read the results with this ceiling in mind

Which *specific* transposition a human makes is stochastic, so exact-match-to-human on
the fail half is bounded by the entropy of human errors, not by model quality. Exact
match is the right STaR **filter**; it is a poor headline **metric**. Read the
per-length curve and the error-type profile instead. A modest absolute fail-side match
rate is expected and is not, by itself, failure.

Second caveat: with the digits printed in the prompt, forward recall is a copy task, so
success-trial rationales are near-vacuous. The detectors are ground-truth accuracy
failing to fall and the per-length curve staying flat.

---

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `StarConfig` (every tunable, placeholders marked), Tinker price table verified 2026-09-10, `tinker_cost_usd()`, output-path helpers. |
| `task.py` | The `RationaleTask` protocol and the task registry. Everything the loop needs to know about a task: how to load and split items, build the three prompt variants, parse a completion, and — above all — `accepts()`, the STaR filter. |
| `tasks/` | The two implementations. `digit_span.py` is a pure adapter over the modules below; `listening_qa.py` over `listening/`. |
| `data.py` | Loads `runs/human/working-memory-{digit-span,reverse-digit-span}/run-*.json` into `HumanTrial`. Excludes `.claude/worktrees/` (byte-identical copy — a recursive glob double-counts), drops `status != completed` runs, de-dupes repeat participants by earliest `started_at`, and re-derives `correct` from the strings as an assertion. **Stores `user_digits`/`expected_digits` as `list[int]`** so they compare directly against parsed output. |
| `select.py` | Builds `D`: all fail trials + an equal random sample of successes (global 1:1), then a trial-level split stratified on (direction, correct). Drops eval trials whose digit sequence also appears in train. `restore()` re-reads `selection.json` so every round uses the identical `D`. |
| `prompting.py` | Assembles prompts from bench's **C3** condition (imported, not copied) plus an explicit task line and a `<reasoning>` block. Three variants: generation, rationalization (hint), and the zero-shot training prompt. `parse_rationale()` splits on `</reasoning>` and applies bench's `PRESS_RE` **to the answer only**. `hint_leak()` flags rationales that give away the hint. |
| `prompts/` | User-authored few-shot rationale demos. See `prompts/README.md`. |
| `errors.py` | Error taxonomy (`truncation`/`omission`/`insertion`/`substitution`/`transposition`/`mixed`), graded `error_features()` for the ~44% of human errors that combine several, and `total_variation()` between profiles. Applied identically to humans and models. |
| `sample.py` | Algorithm 1 lines 3–4. Greedy generation, then rationalization only for trials that failed. Writes every sample — accepted or not — to `sample_pool.jsonl`. Reports `fail_side_generation_yield`. |
| `filter.py` | Lines 5–6: partitions accepted rows into `D_n` / `D^rat_n`, applies the leak filter, writes `accepted.jsonl` + `filter_stats.json`. **No cap or downsampling** — balance is decided once, in `select.py`. |
| `tinker_client.py` | The only module that imports `tinker`. Sampling/training clients, `build_datum()` with loss weights 0 over the prompt and 1 over the completion, token/cost accounting. |
| `train.py` | Line 7. Fresh LoRA client from the base model, warmup-then-constant LR, fixed step budget, `save_weights_for_sampler`. |
| `evaluate.py` | Held-out eval: human-match rate, ground-truth accuracy, per-length curves, error profiles, and `compare()` deltas between base and tuned. |
| `plotting.py` | Per-length curves (base/tuned/human on one axis), error-profile bars, bootstrap-signal plot. Figure failures never lose a completed round. |
| `star.py` | The outer loop and `dry_run()` cost planning. Round N+1 *samples* from round N's checkpoint; training still starts from base. |
| `cli.py` | Typer entry points (below). |
| `tests/` | 261 tests. `conftest.py` fakes the Tinker SDK and torch, so masking, model routing, and round-over-round behavior are all testable with no API key and no spend. |

### `listening/` — the listening-QA task

| Module | Responsibility |
|---|---|
| `listening/data.py` | Joins `analysis/data/processed/responses.csv` (what was endorsed, plus the blind `option_type` / `cue_match` labels), `data/<topic>/questions.yaml` (question, options, gold) and `data/<topic>/texts/<level>.md` (the transcript) into 4020 `ListeningItem`s. Asserts `agent == "human"` — the CSV holds ~106k model rows too — and raises if the CSV and the question bank disagree about which options are correct. Records a sha256 of every source in `run_config.json`. |
| `listening/select.py` | Splits **by participant**, stratified on the counterbalancing group. Deliberately not `select.select`: its stimulus-overlap drop would empty the eval split (sixteen stimuli, all in train) and its 1:1 subsample would discard half the data. Reports every (topic, level, question) cell and flags degenerate ones. |
| `listening/prompting.py` | The listening task's **C3** condition (imported from `application/listening_qa/prompting.py`, not copied) plus a task line, a `<reasoning>` block, and the sibling-answer context. One question per prompt, answered as `Answer: 1,3`. |
| `listening/evaluate.py` | Exact set match against the human, per-option agreement, and the endorsement profile over `option_type × cue_match` — the listening analogue of digit span's error taxonomy, reusing labels the analysis pipeline already assigns blind. |
| `listening/plotting.py` | By-level curves (base/tuned/human on one axis) and endorsement-profile bars. |

**The open question about this task.** Each topic's fourth question is built as a 2×2
combination of two earlier questions' content, so for those items the sibling answers
may let a model *derive* the answer rather than simulate a memory. Nothing in the design
separates "learned human-like forgetting" from "inferred it from the siblings". Run
`probe` both ways before reading any fine-tune result:

```bash
python -m rationales.cli probe --task listening_qa --n-per-cell 10
python -m rationales.cli probe --task listening_qa --n-per-cell 10 --no-sibling-context
```

If the base model already matches humans well *with* siblings, the tuned-vs-base
comparison is not interpretable and `--no-sibling-context` is the lever.

## Output layout

```
rationales/out/<model_slug>/<run_timestamp>/
    run_config.json            # config + deviations + git_provenance + prompt_additions
    selection.json             # the pinned D, reused by every round
    summary.json               # per-round reports + bootstrap_signal
    figures/
    rounds/round_<n>/
        sample_pool.jsonl      # every sample, accepted or not, with parse_errors
        accepted.jsonl         # D_n u D^rat_n for THIS round
        filter_stats.json
        train_log.json         # per-step loss; records trained_from = base model
        checkpoint.txt
        eval.json              # base vs tuned, with comparison + usage
```

`sample_pool.jsonl` is truncated on each run (`JsonlSink`), so re-running a round
replaces its pool rather than appending. Rounds never share a directory, and pools are
never merged across runs or configs.

---

## CLI

Run as `python -m rationales.cli <command>`.

### `select-data`
Builds `D` and prints the split plus the fail-by-length histogram. No API calls.
```
python -m rationales.cli select-data [--seed 42] [--eval-frac 0.15] [--out PATH]
```
The histogram is worth reading: global 1:1 concentrates fails at long spans, which is
the shortcut the per-length eval curve exists to detect.

### `show-prompt`
Renders one fully-assembled prompt for review. No API calls.
```
python -m rationales.cli show-prompt [--direction forward|reverse]
                                     [--kind sample|rationalize|training]
                                     [--fewshot/--no-fewshot]
```
Use it to confirm the training prompt carries neither the few-shot block nor the hint.

### `dry-run`
Plans a round — request counts, token estimate, cost estimate. No API calls.
```
python -m rationales.cli dry-run [--base-model Qwen/Qwen3-8B] [--rounds 1]
                                 [--k 1] [--temperature 0.0] [--fewshot/--no-fewshot]
```
Warns if the estimate exceeds $5 or the model has no price on file. Estimates use a
~4-chars/token heuristic and are labelled as such.

### `probe`
Checks that the sampling prompts actually elicit well-formed rationales, using
OpenRouter. **Format check only** — a few cents, and its output is never written to a
corpus.
```
set -a && . ./.env && set +a          # repo-root .env, already gitignored
python -m rationales.cli probe
    [--model qwen/qwen3.8-flash] [--n-per-cell 2] [--max-tokens 1024]
    [--fewshot/--no-fewshot] [--thinking/--no-thinking]
    [--temperature 0.0] [--seed 42]
```

**Thinking is off by default, and it matters.** The first probe run on
`qwen3.8-flash` returned `well_formed_rate: 0.44` with 7 of 16 completions *entirely
empty* — 373 of 480 completion tokens had gone to hidden reasoning, so longer thinks
produced no content at all. That reads as a broken prompt in the report when it is
purely a serving artifact. Check `empty_response_rate` before anything else.
Runs both prompt kinds over `n_per_cell` trials from each (direction × success/fail)
cell — 16 calls at the default, ≈$0.004 on `qwen3.8-flash` — and reports:

- `well_formed_rate` — did a `<reasoning>` block and parseable `press <<D>>.` lines
  come back? Near 1.0 means the prompt works.
- `rationalize_hit_rate_fail_trials` — **the one to watch.** If the model ignores the
  hint, STaR gets no fail-side training signal and the loop has nothing to bootstrap
  from.
- `sample_human_match_{success,fail}_trials` — expect high on successes (the digits are
  in the prompt) and low on failures.
- `rationalize_hint_leak_rate` — rationales that give away that they were handed the
  answer; those get dropped by `filter.py`.

This is **off-policy**: a different serving stack, usually a different model, from the
Tinker base being trained. It tells you the prompts and parser work; it does *not*
estimate round-1 yield. Report lands in `out/probe/<timestamp>_<model>.json`.

#### What the probe found (2026-09-11, `qwen3.8-flash`, 16 calls)

| metric | value |
|---|---|
| `well_formed_rate` | 1.00 |
| `empty_response_rate` | 0.00 |
| `rationalize_hit_rate_fail_trials` | **1.00** — the hint works; STaR has fail-side signal |
| `rationalize_hint_leak_rate` | 0.00 |
| `median_reasoning_chars` | 319 |
| `sample_human_match_success_trials` | 0.75 |
| `sample_human_match_fail_trials` | 0.25 |

Two fixes came out of it, both now in the code:

1. **Reasoning was unbounded** — up to ~3400 characters, and one completion never
   closed its tag before the token limit. The prompt now caps reasoning at three
   sentences and forbids step-by-step working; median length dropped 1291 → 319 chars.
2. **`max_seq_length` was 1024**, too small for a real prompt plus a long rationale.
   Truncation cuts from the right — off the *completion*, which is the part the loss
   covers — so it would have silently trained on half an answer. Now 2048, with
   `build_datum` flagging truncation and `train_log.json` recording
   `n_truncated_examples`.

Worth noting from the unhinted path: this model already errs on its own under C3
(`sample_ground_truth_rate` 0.375), so success trials will not all accept on the
generation path either — expect rationalization to carry some of the success half too.

### `cost`
Reads the live cost ledger a running (or finished) round writes.
```
python -m rationales.cli cost [--run-dir PATH] [--watch] [--interval 5]
```
Defaults to the most recent run under `out/`. `--watch` refreshes until `summary.json`
appears.

```
20260911T051018Z  (1092 calls)
  sample   calls=1092  prefill=  931,689 sample= 124,488 train=        0
  train    calls=60    prefill=        0 sample=       0 train=  248,445
  spend: $0.3657
```

### `prepare`
Creates the run directory with `selection.json` and git provenance, without calling the
API.
```
python -m rationales.cli prepare [--base-model ...] [--seed 42]
```

### `star`
The full loop: sample → filter → train → eval, per round. **This spends money.**
```
python -m rationales.cli star [--base-model Qwen/Qwen3-8B] [--rounds 1]
                              [--k 1] [--temperature 0.0]
                              [--lora-rank 16] [--learning-rate 1e-4]
                              [--steps-1 60] [--batch-size 8] [--seed 42]
                              [--fewshot/--no-fewshot] [--leak-filter/--no-leak-filter]
                              [--yes]
```
Prints a cost estimate and asks for confirmation unless `--yes`.

---

## Setup

```bash
uv pip install -e ".[rationales,dev]"
export TINKER_API_KEY=...        # never committed
python -m pytest rationales/tests -q
```

Sampling and training both run through Tinker. There is **no OpenRouter path**: neither
`Qwen3-8B` nor `Qwen3.5-4B` is served there (checked against OpenRouter's model list),
and sampling a different model than the one being trained would make the corpus
off-policy.

## Cost

Tinker bills prefill, sample, and train separately. Verified 2026-09-10 from
[models.json](https://tinker-docs.thinkingmachines.ai/tinker/models.json), USD per 1M
tokens — **note the 8B is cheaper than the 3.5-4B**:

| model | prefill | sample | train |
|---|---|---|---|
| `Qwen/Qwen3-8B` | 0.195 | 0.60 | 0.44 |
| `Qwen/Qwen3.5-4B` | 0.33 | 1.005 | 0.737 |
| `openai/gpt-oss-20b` | 0.18 | 0.45 | 0.396 |

### Live cost tracking

Tinker has **no live spend API**, so cost is priced client-side as calls happen:

| source | live? | USD? |
|---|---|---|
| `tinker billing usage` | no — "up to a few hours" behind | no, tokens / GB-hours |
| Session metrics (console panels, `tinker session export-trace`) | yes, token totals | no |
| SDK responses (`len(seq.tokens)`, datum sizes) | yes, immediate | only once priced |

`CostTracker` prices every call against the table above and writes one row per call to
`<run>/cost_ledger.jsonl` with a running total, plus a single-line stderr readout:

```
spend $0.1842 | 604 reqs | prefill 531,204 sample 63,384 train 0
```

Counts are exact, not estimated — prefill is the tokens actually sent, sample is
`len(seq.tokens)` off the response, train is the datum tokens per step (priced per
step, not once at the end).

Watch it from another terminal while a round runs:
```
python -m rationales.cli cost --watch          # or: tail -f <run>/cost_ledger.jsonl
```

### Pause on spend, resume without re-paying

`--max-usd` **pauses** the run rather than killing it. Every phase flushes as it goes,
`progress.json` records where it stopped, and `paused.json` carries the resume command:

```
python -m rationales.cli star --max-usd 2.00
...
PAUSED: spend $2.0013 exceeded the $2.00 ceiling after 1447 requests.
resume: python -m rationales.cli star --resume --run-dir <run> --max-usd <higher>
```

Resuming picks up mid-phase, and nothing already paid for is bought twice:

| phase | what resume reuses | granularity |
|---|---|---|
| `sample` | `sample_pool.jsonl` — trials already accepted, or exhausted on both paths, are skipped | per trial, flushed **per attempt**, so a trial interrupted between its generation and rationalization calls is correctly retried rather than silently dropped |
| `filter` | recomputed from the pool (free, no API) | — |
| `train` | `save_state` checkpoints (weights **and** optimizer momentum) every `--checkpoint-every` steps; applied steps are skipped | ≤ `checkpoint_every` steps of rework |
| `eval_base` / `eval_tuned` | `eval_rows_*.jsonl` — scored trials are skipped | per trial |

The training resume uses `create_training_client_from_state_with_optimizer`, so momentum
is restored rather than reset — restarting at step 0 would both re-pay for the steps and
change the schedule the model actually saw.

Note the split between the two kinds of checkpoint: `save_state` is for *continuing an
interrupted round*, while a new round still starts from the base model, since Algorithm 1
line 7 trains `M`, not `M_{n-1}`.

**Reconciling with real billing.** Every training run is tagged with
`user_metadata={"pipeline": "rationales-star", "run": <timestamp>, ...}`. Once billing
catches up (hours later), join the export back to the run:
```
tinker billing usage 2026-09-11T00:00:00Z 2026-09-12T00:00:00Z \
    --csv usage.csv --sessions-csv sessions.csv
```
`sessions.csv` carries `session_id` + `user_metadata`; join it to `usage.csv` on
`session_id` to get the authoritative token counts for this run, then compare against
`cost_ledger.jsonl`. Billing reports tokens, not USD, so the list-price table is still
what converts it.

Prices move (prefill/sample rose ~50% and train ~10% on 2026-07-17) — re-check the live
file before a run. `dry-run` on the current split (556 train / 94 eval, 1022
generations, 60 train steps) estimates **$0.35** for round 1 on Qwen3-8B. Always run it
first; it also warns when a model has no price on file.

## Suggested first run

Digit span:

```bash
python -m rationales.cli select-data                  # sanity-check D
# write the rationales in prompts/fewshot_{forward,reverse}.txt
python -m rationales.cli show-prompt --kind sample --direction reverse
python -m rationales.cli dry-run                      # cost gate
git tag exp/star-v1
python -m rationales.cli star --rounds 1
```

Listening QA:

```bash
python -m rationales.cli select-data --task listening_qa      # 4020 items; check the
                                                              # flagged degenerate cells
python -m rationales.cli show-prompt --task listening_qa --kind training

# the sibling-context control -- cents, and the gate on whether a fine-tune result
# can be read at all (see "The open question about this task")
python -m rationales.cli probe --task listening_qa --n-per-cell 10
python -m rationales.cli probe --task listening_qa --n-per-cell 10 --no-sibling-context

python -m rationales.cli dry-run --task listening_qa           # ~$3.50/round on Qwen3-8B
git tag exp/listening-star-v1
python -m rationales.cli star --task listening_qa --rounds 1 --max-usd <ceiling>
```

Then read, in order: `filter_stats.json` (is `D_n` non-empty?),
`fail_side_generation_yield` (if human-fail items only ever accept via the hint path,
round over round, STaR is not bootstrapping and more rounds will not fix it), the
by-length / by-level figure, and the error-profile figure.
