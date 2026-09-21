# STaR results

Two tasks. [Digit span](#star-on-digit-span--round-1-results-2026-09-11) below;
listening QA first, as the more recent.

---

# STaR on listening QA — round 1 (2026-09-20)

Base `Qwen/Qwen3-8B` via Tinker, one run, `out/Qwen_Qwen3-8B/20260920T223000Z/`.
4020 items from 201 respondents, split by participant (167 train / 34 eval → 3340 /
680 items). Sibling context on. Spend $5.91.

Tagged `exp/listening-star-v1`. The round was trained twice from one corpus — 60
steps, then continued to the intended 200 — and **both evals are kept**, as
`eval_step60.json` and `eval.json`.

| metric | base | 60 step | 200 step | human |
|---|---|---|---|---|
| human_match_rate | 0.332 | 0.384 | **0.444** | — |
| ground_truth_accuracy | 0.728 | **0.566** | 0.663 | 0.474 |
| error_profile_tv_distance | 0.057 | **0.044** | 0.049 | — |
| mean options endorsed | 1.897 | **1.446** | 1.574 | 1.404 |
| fail-only human_match | 0.014 | **0.182** | 0.151 | — |
| fail-only profile TV | 0.310 | 0.279 | **0.235** | — |

## What it shows

**The model learns to forget here too, and to forget as a specific person.** Every
success criterion moves: exact match to the participant's own selection up,
ground-truth accuracy down toward the human level, endorsement profile closer to the
human distribution. On items the human got *wrong* the tuned model reproduces their
exact selection 11–13× more often than base (0.014 → 0.151–0.182).

**More training is not uniformly better.** The 200-step model matches more exact
selections; the 60-step model is closer to human on every memory-fidelity metric —
accuracy descent, endorsement count, fail-side match. Longer training appears to
partly undo the forgetting, which is plausible given roughly half the corpus is
human-success items where the human was simply right. Which model is preferred
depends on the claim being made.

**Rationalization carried the corpus, and that was still enough.** Fail-side
generation yield was **0.018**: of 2324 accepted rationales, 1427 came from the hinted
path. Training on them transferred to the unhinted eval prompt anyway. A near-zero
round-1 bootstrap signal did not mean the round was wasted — but whether round 2
raises that yield is the open question, and the digit-span runs below suggest it
climbs only slightly (0.011 → 0.018 there).

## Caveats

- **Noise floor ~0.01.** Base was re-scored between the two evals and moved 0.326 →
  0.332 at temperature 0 (about four items of 680). Serving is not bit-deterministic.
  The fail-side gap between the two tuned models (0.182 vs 0.151, ~11 items of 358)
  sits close to this floor; the overall gap does not.
- **Off-policy probe.** The pre-run probe used `qwen/qwen3.8-flash` via OpenRouter and
  reported clean parsing. It could not have caught that Tinker returns `<|im_end|>`
  inside the completion, which emptied the first corpus entirely — OpenRouter strips
  control tokens server-side.
- **Sibling context earns its place on this evidence only weakly.** The probe's paired
  arms showed it makes no difference on human-fail items (6/40 either way) and helps
  only where matching the human means being correct. It was kept on for this run; a
  `--no-sibling-context` run is the obvious comparison and has not been done.
- **Five degenerate cells** (correct share outside 0.15–0.85) were flagged by
  `select-data` and left in.

---

# STaR on digit span — round 1 results (2026-09-11)

Base model `Qwen/Qwen3-8B` via Tinker. Three round-1 runs plus one round-2 continuation.
Total spend $1.30. Raw outputs under `out/Qwen_Qwen3-8B/<timestamp>/`.

## Runs

| run | seed | mean per-cell gap | base gap | fail-side yield | gt acc | error TV | fail-trial match |
|---|---|---|---|---|---|---|---|
| 20260911T055700Z | 42 | 0.150 | 0.411 | 0.011 | 0.638 | 0.330 | 0.000 |
| 20260911T063609Z | 42 (replicate) | 0.236 | 0.405 | 0.014 | 0.670 | 0.287 | 0.000 |
| 20260911T064029Z | 5566 | 0.347 | 0.388 | 0.004 | 0.653 | 0.337 | 0.000 |
| 20260911T055700Z r2 | 42 | 0.451 | — | 0.018 | 0.702 | 0.287 | 0.021 |

Human reference: ground-truth accuracy **0.489**. "Gap" = mean over (direction × span)
cells of |model exact rate − human exact rate|; lower is more human-like.

## What replicates

**The model learns to forget.** Ground-truth accuracy falls from ~0.85 (base) to
**0.654 ± 0.016** across three runs, moving toward the human 0.489. Tight spread.

**Error-type distribution moves toward humans.** TV distance 0.447 (base) →
**0.318 ± 0.028**. Consistent across runs.

## What does not replicate

**The per-cell gap.** 0.150 / 0.236 / 0.347 — a 2.3× spread, and the worst run (0.347)
is barely better than the base model's 0.388–0.411. The two seed-42 runs share an
identical corpus (|D_n| 210 vs 210, |D^rat_n| 335 vs 332) and an identical eval split,
so this is **training nondeterminism on Tinker, not seed variance**, amplified by an
eval set where several span cells have n=1–2 (one cell flipping at n=1 moves the mean
by ~0.06).

Any claim about matching the human *forgetting curve* needs a larger eval split and
n≥3 runs. The single-run 0.150 originally reported was not a reliable estimate.

## What is flatly zero

**Fail-trial exact human-match: 0.000 in all three round-1 runs** (0 of 48 held-out
fail trials each). The model never reproduces a specific human's specific error. Round 2
scored 0.021 — one trial out of 48.

This is the entropy ceiling the plan predicted: which digits a given human transposes is
stochastic, so exact match on the fail half is not a learnable target.

## STaR did not bootstrap

Fail-side generation yield (rationales accepted on the *unhinted* path) across rounds:
**0.011 → 0.018**, with round-1 replicates at 0.014 and 0.004. Between 97.5% and 99.6%
of fail-side corpus entries came from rationalization in every run. The hinted
rationales never transferred to the unhinted prompt, which is the mechanism STaR
depends on.

## Round 2 made things worse

Mean gap 0.150 → 0.451, worse than the untuned base (0.388–0.411). It over-forgets at
short spans (forward:5 → 0.11 where humans sit at 0.78) and under-forgets at long spans
(reverse:6–8 back to near-perfect). Aggregate metrics hid this: overall accuracy and TV
distance both looked fine.

Likely cause: round 2's corpus is self-generated by the round-1 model, so that model's
over-forgetting gets relabeled as "correct" wherever it coincidentally matched a human,
then trained on. Nothing corrects the drift, because the fail-side signal is ~99%
synthetic.

## Recommendation

Stop the outer loop. Two changes are worth more than a round 3:

1. **Make the filter distributional.** Accept a rationale when it reproduces the
   human's *error type* (truncation / transposition / omission) rather than their exact
   digits. The exact-match filter is entropy-limited and produces an almost entirely
   hint-derived fail half.
2. **Fix the eval before trusting any gap.** Larger held-out split, n≥3 runs, and report
   a range. Ground-truth accuracy and error-type TV are the metrics that currently
   survive replication; the per-cell gap is not.

Secondary: the reverse direction stays too accurate at spans 6–8 (0.57–1.00 vs human
0.00) in every run. That is where the remaining "memory too good" lives.
