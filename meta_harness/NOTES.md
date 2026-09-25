# Meta-Harness onboarding notes (pre-spec)

Measurements taken from released data only. No API spend.
Regenerate with the scripts referenced below.

## 1. The objective inverts

Humanlikeness is `1 - W_1` between the model's and humans' per-participant
score distributions (`src/score.py`). The `claude-opus-4-6` compactor is
*better* than humans on 8/10 tasks and *worse* on N-Back. So this is a
**calibrated-degradation search**, not a capability search, and the direction
is not uniform: a single "forget harder" knob improves 8 tasks and worsens
N-Back.

## 2. Noise floor and headroom

Split-half W_1 within the ~50 human participants per task gives the
irreducible floor. See `baseline_humanlikeness.txt`.

| base model | mean humanlikeness | recoverable headroom |
|---|---|---|
| claude-opus-4-6 | 0.730 | **0.225** |
| qwen3-30b-a3b-instruct-2507 | 0.788 | 0.167 |
| llama-3.3-70b-instruct | 0.791 | 0.164 |
| llama-3-8b-instruct | 0.798 | 0.157 |
| gpt-5.4 | 0.806 | 0.148 |
| qwen3-30b-a3b-thinking-2507 | 0.825 | 0.130 |
| qwen3-next-80b-a3b-instruct | 0.832 | 0.122 |
| qwen3-8b | 0.883 | 0.071 |

Mean attainable ceiling across tasks: **0.955**.

Weak models are already human-like *by incapacity*. qwen3-8b's 0.071 mean
headroom is at or inside the noise floor on 4/10 tasks (Free Recall 0.009,
Reverse Digit Span 0.020, Narrative QA 0.021, Digit Span 0.027 against floors
of 0.026-0.042), so it is not a usable search substrate.

Per-task headroom on `claude-opus-4-6`: Variable Mapping 0.569, Digit Span
0.307, Factual QA 0.276, N-Back 0.250, Map Task 0.221, Word Recognition 0.164,
Craft Task 0.147, Narrative QA 0.118, Reverse Digit Span 0.106, Free Recall
0.090. Variable Mapping alone is 25% of the total, so a mean-only objective
lets the proposer buy the whole improvement there.

## 3. Error structure is required, not optional

`1 - W_1` on scores is gameable: a harness that draws a random per-participant
decay rate matches the human distribution while being scientifically empty.
With `MAX_KEYS` and decay inside the search space, that candidate is reachable.
So error structure must be a hard Pareto axis. See `error_structure.txt`.

- **A2 word recognition, error asymmetry.** Humans are conservative:
  miss 0.272, false-alarm 0.045, ratio **6.09**. `claude-opus-4-6`: miss 0.000,
  false-alarm 0.533, ratio **0.00** -- the error structure is fully inverted,
  yet its word-recognition humanlikeness is 0.760. This is the clearest
  demonstration that score-distribution match is insufficient.
- **A3 story recall, verbatim vs gist.** Humans BLEU 0.002 at 137 words;
  `claude-opus-4-6` BLEU 0.199 at 366 words (verbatim regurgitation, 2.7x too
  long). qwen3-30b BLEU 0.003 at 128 words is already human-like here.
- **A1 digit span, threshold sharpness.** Humans fail at a clean threshold
  (sub-span failure 0.087). Models fail raggedly: supra-threshold hit rate
  0.574 (opus), 0.551 (qwen3-30b).
  **Caveat:** not yet apples-to-apples. The human protocol is adaptive and
  terminates (~12 trials, ~2 per span), while the model protocol runs all 19
  span lengths, which inflates the model's supra-threshold opportunities. A1
  needs a protocol-matched version (subsample model trials onto the human
  adaptive schedule) before it can serve as a Pareto axis.

## 4. Cost and wall-clock per candidate

Token usage is **not** logged in the released runs (`encoding_log` stores
turns and content but no `usage`), so cost is proxied from 19,940 logged LLM
turns / 6.6M chars across 4,750 participant-rows per full 10-task evaluation:

| base model | ~per candidate | ~40 candidates |
|---|---|---|
| claude-opus-4-6 | $370 | $14,800 |
| gpt-4.1 | $46 | $1,850 |
| gpt-4.1-mini | $9 | $370 |
| local on Torch | $0 | $0 (GPU hours only) |

Digit span forward + reverse are **85% of all turns** (16,975 of 19,940) for
0.307 + 0.106 headroom. Cutting their repeats from 1900 rows to ~200 each
drops a full evaluation to ~4,700 turns -- a 4x speedup for little lost
signal. Do this for the search set; keep full repeats for final scoring.

Released qwen configs run `max_parallel_participants: 50`,
`max_parallel_tasks: 10`. On a local vLLM serving qwen3-30b-a3b (MoE, ~3B
active), one reduced-set candidate is roughly 10-30 min of single-GPU time,
so 40 candidates fits in a normal allocation.

## 5. Recommended plan

Search locally on **qwen3-30b-a3b-instruct-2507** via vLLM on Torch: best
headroom (0.167) of any free option, $0 API spend. Then confirm only the
Pareto-frontier finalists (3-5) on `claude-opus-4-6`, which has the real
headroom (0.225) and is the baseline worth reporting -- ~$1-2k, needs explicit
approval under the cost gate.

## 6. Open items

- Working vLLM setup + GPU allocation on Torch: **unknown**, user to confirm.
- Confirm-stage dollar approval: **unknown**, user to confirm.
- `MAX_KEYS != 4` is in scope by the user's choice, but it breaks the
  Cowan-2001 grounding the compactor is built on. Any winning candidate that
  moves it needs that argued explicitly rather than silently accepted.
- Splits: participant split (half the humans as search reference, half held
  out) is nearly free and should be on regardless. Task split vs held-out-model
  split is a claim question still to settle.

## Scripts

Kept in the session scratchpad; copy into `meta_harness/` when the spec lands.

- `noise_floor.py <model_dir>` -- split-half floor, observed W_1, headroom
- `all_baselines.py` -- the table in section 2
- `error_structure.py [model_dirs...]` -- the axes in section 3
- `call_count.py` -- turn counts and cost proxy
