# Level-pair alignment methods: from individual-sample draws to hierarchical Bradley-Terry

Context: does the WM compactor's per-level accuracy pattern (control / repeat_short /
repeat_long / distractor) match humans' the way plain prompting (C1-C3) doesn't? This
tracks three successive methods used to answer that, why each one was replaced, and
what the final method found. Companion to `COMPACTOR_ITERATION_NOTES.md` (which covers
the compactor's own prompt/tool design, not the alignment metric).

## Method 1: individual-sample-draw pairwise reranking (`level_pair_preference_alignment.py`)

For each of the 24 within-topic level pairs, repeatedly draw one random sample from
each side's raw trials and record which side wins; agreement = how often the model's
per-draw winner matches the human's per-draw winner (Monte Carlo estimate of the
underlying probability-of-superiority). Original method, still used for the
`listening_level_pair_alignment_{model}.png` / `listening_qa_pairwise_reranking_slopeplot.png`
charts.

## Method 2: mean/Mann-Whitney U cell-ranking (`mean_level_pair_alignment.py`)

Started as a **mean comparison**: rank each (topic, level) cell by its raw mean score,
score agreement 1/0/0.5(tie) against human's mean-ranking. Switched to **Mann-Whitney U
probability of superiority** (`PS = U / (n_a * n_b)`, via `scipy.stats.mannwhitneyu`)
per user request ("look for a principled way to handle variance without using means") --
PS is the *exact closed-form* version of what Method 1 approximates by Monte Carlo, so
it uses each cell's full distribution instead of collapsing to one number, with no
resampling noise. Ties (PS==0.5) score 0.5, not a coin flip -- deterministic, no RNG,
same expected value.

Two scopes: `within_topic` (24 base pairs) and `all` (120, includes cross-topic pairs).
Human split-half reliability baseline: repeatedly split participants in half, score
half A vs half B with the same method, average over 20 splits -- the noise ceiling any
model could hit given this sample size.

**Limitation found**: bootstrap CIs (resampling the 24/120 base-pair scores) were very
wide. This method estimates each of the 24 comparisons in complete isolation -- no
sharing of evidence across topics.

## Method 3: Bradley-Terry pooling (`bradley_terry_level_strength.py`, `bayesian_bt_level_strength.py`)

### Full pooling (WLS + bootstrap)

Fits ONE `beta_level` per level, pooling all 4 topics' pairwise MWU evidence via
weighted least squares on `logit(PS) = beta_i - beta_j`, weight `n_i*n_j/(n_i+n_j)`.
Bootstrap CI resamples raw per-trial data within each cell and refits.

**Root-caused why THIS method's bootstrap CI was too narrow to be trusted**: full
pooling has no parameter for "topic 3 disagrees with the other 3" -- it treats all 24
rows as exchangeable evidence for one shared value. The bootstrap only captures
within-cell sampling noise, never topic-to-topic disagreement, because the model has
no slot to represent that disagreement at all. This is the standard "pooling bias"
failure mode.

### Partial pooling (hierarchical Bayesian, PyMC/NUTS)

Each topic gets its own `beta_topic_level ~ Normal(beta_global_level, tau)`, `tau`
estimated from the data. `tau` inflates when the 4 topics genuinely disagree, and that
uncertainty correctly propagates into `beta_global`'s credible interval
(`Var(beta_global) -> tau^2/k` as within-cell noise -> 0, k=4 topics). This is the
statistically honest version; full pooling was silently **understating** uncertainty
the whole time.

**Diagnostic finding**: every model's `tau` (topic-to-topic heterogeneity) is 3-24x
larger than human's (human tau=0.13, near-zero -- the level effect is consistent
across topics for humans). `sigma_obs` (per-comparison noise) is also inflated
2-26x for every model. Checked whether more samples/cell would shrink the CIs:
computed each model's ratio of `current_CI_width / floor_width` where
`floor_width = 1.96 * tau` (the width more per-trial samples can never shrink below,
since `tau` reflects genuine cross-topic disagreement, not sampling noise) --
**all 6 models are already within ~20% of their tau floor**. More samples per cell
would buy almost nothing for the 5 models already at n=20; the one model at n=3
(GPT-4.1) had the most headroom.

**Confirmed empirically**: reran GPT-4.1's WM condition at n=20 (was n=3). Its
`repeat_short`/`repeat_long` point estimates **flipped sign** -- from disagreeing with
human direction (likely n=3 sampling noise) to agreeing with it, joining
Gemma-4-31B-it and Qwen2.5-32B-Instruct. `distractor` stayed correctly negative at
both n. Lesson: an outlier result on a model with much smaller n than its peers is a
red flag to investigate before treating it as a real finding.

### Visualization iteration

First attempt: 7-series line/slope plot at fixed x-positions per level -- too much
whisker overlap once real (wide, honest) Bayesian CIs were added. Fixed with
per-series x-offset jitter within each level group (vertical dividers mark group
boundaries), keeping the connecting-line "slope" shape the user preferred over a
forest-plot layout (which was tried first, reads well for CI-heavy data but the
user preferred the line-plot form here).

## Cross-model findings (WM condition only, human-agreement direction on beta sign)

| model | repeat_short/long direction | distractor direction | notes |
|---|---|---|---|
| Human | positive/positive | negative | reference |
| Gemma-4-31B-it | positive | negative | most human-aligned point estimates, but also largest tau/sigma_obs (least stable reasoning) |
| Qwen2.5-32B-Instruct | positive | ~0 (not sig.) | |
| GPT-4.1 (n=20) | positive | negative | flipped from n=3's wrong-direction result |
| Kimi-K2-0905 | ~0 (not sig.) | negative | closest tau/sigma_obs to human (most consistent across topics) |
| Qwen2.5-72B-Instruct | ~0/negative (not sig.) | negative | |
| Command-A | positive/negative (mixed) | negative | |
| Gemini-3.1-Pro-Preview | positive (not sig.) | negative (largest magnitude, -4.49, clearly significant) | strongest distractor signal of any model |

**Distractor direction is the most consistent finding**: every model agrees with
human direction (repetition-irrelevant distractor content hurts recall), varying only
in magnitude/significance. `repeat_short`/`repeat_long` (does repetition help
recall) is where models diverge from each other and from human -- roughly half show
the human-matching positive direction, half don't, and few reach significance given
current sample sizes.

## Other models with compactor data (held off, insufficient n)

`deepseek-v3.2`, `llama-4-maverick`, `qwen3.5-397b-a17b` have compactor runs but only
n=5/cell (smaller than even GPT-4.1's original problematic n=3). Quick MLE check
showed no clean pattern (llama-4-maverick was essentially all-tied, like a
prompting-ceiling condition). Held off on the full Bayesian treatment until more
samples are collected -- not worth trusting at this n given what n=3 -> n=20 already
taught us about GPT-4.1.

## Infra notes

- `scipy`, `matplotlib`, `pymc` (+`arviz` pinned `<1.0` to fix a resolve that grabbed
  an API-incompatible split-package version) were never in `pyproject.toml` --
  prior work in this session had silently been running against a stray, ABI-mismatched
  conda python instead of the project's `uv`-managed venv the whole time.
- Bar chart colors for `C1/C2/C3/WM` conditions now match
  `application/listening_qa/plot_level_pair_alignment.py`'s existing palette (blue
  shades for prompting, green for compactor) instead of an ad hoc scheme -- and both
  scripts/shared color constants were later centralized into `application/plot_style.py`.
