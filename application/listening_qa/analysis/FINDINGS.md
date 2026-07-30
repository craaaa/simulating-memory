# Findings — Human–Model Alignment (listening-QA)

Scope: **text-model vs audio-human** (no text-human control batch, D5). 201 humans
(whole-participant attention pass); 6 models × {prompting C2, compactor C2-stream}, n=20/cell.
Lead with descriptive rates; GLMM is rigor on top. Results **never averaged across models**.

## Headline

**The manipulation moves models differently from humans, and the compactor is not
statistically equivalent to humans on any stratum (Δ = ±0.22 log-odds). Where models err,
they err by being *too accurate*, not by making human-like mistakes.**

## 1. Descriptive endorsement (separation-free, decisive)

Endorsement rate on true / false options, pooled over levels:

| system | true | false | selection count / question |
|---|---|---|---|
| human | 0.57–0.72 | 0.07–0.12 | 1.38 (peaks at 1) |
| prompting | **0.998–1.000** | **0.000** | 1.65 (peaks at 2 — over-selects) |
| compactor | 0.59–0.85 | 0.03–0.09 | 1.43 (≈ human) |

- **Prompting is categorically non-human**: near-perfect on true, zero on false → the GLMM is
  *completely separated* there (log-odds → ±∞). The separation *is* the result.
- **Compactor is far closer to human** on both endorsement rate and selection-count shape
  (1.43 vs human 1.38), but sits slightly *above* human on true and *below* on false — the
  "too good" direction, milder than prompting.

## 2. Effect alignment — confirmatory GLMM (§4), TOST at Δ=±0.22

Fit per model as two 2-level `system` models (human+prompting, human+compactor); contrasts
extracted **within each option_type** (see DEVIATIONS.md for why — separation forced this;
these contrasts are therefore reported **exploratory**).

- **No compactor−human contrast is equivalent** at Δ=0.22 on any (level × option_type) stratum
  — every estimable verdict is *different* or *inconclusive*, none *equivalent* (fig 2).
- Direction is consistent across models:
  - **true options**: compactor − human mostly **positive** (+0.4 to +2.6 log-odds) → compactor
    endorses correct options *more* than humans.
  - **false_interference**: strongly **negative or separated** → compactor endorses the
    trait-swap interference foils *less* than humans. It does **not** fall for the interference
    the design was built to probe; humans do.
  - **false_plain**: small, mixed (some inconclusive within the band, none formally equivalent).
- **command-a** compactor is itself near-ceiling (true 0.967) → its true & false_interference
  strata are separated and reported **ceiling-limited** (the recovery gate showed the sparse
  false-stratum fallback is unreliable under ceiling; see DEVIATIONS.md).
- Prompting−human: separated on essentially all strata → categorically different (not testable,
  and does not need to be).

Convergence: all fits converged at rung 1 (gemma/kimi one side at rung 2). Δ sensitivity
(literal 0.056) does not change the verdict — nothing is equivalent even at the wider 0.22.

## 3. Error alignment (§5, exploratory)

- **κ (error consistency, Framing B)**: prompting κ≈0 (too accurate to share human error
  structure); compactor κ 0.02–0.17 vs human–human ceiling ≈0.20 — compactor's error pattern is
  *measurably more human-like than prompting's*, but still below the human ceiling.
- **False-option profile Spearman**: undefined for prompting at easy levels (zero variance —
  no false endorsements). Compactor computable but weak/near-zero on interference → confirms
  compactor does not reproduce the human interference profile.

## 4. Difficulty alignment (§6, exploratory)

- Compactor option-facility vs human: Spearman 0.28–0.80 (ratio 0.30–0.87 of the 0.92 human
  split-half ceiling). Highest for command-a (partly ceiling-driven). **Distractor level is
  consistently weakest** (GPT-4.1 distractor r≈0) — models find *different* options hard under
  distraction than humans do. Caveat: split-half r bounds R², not r (fig 5 caption).

## 5. Diagnostics (§7)

- **Selection count**: compactor ≈ human (1.43 vs 1.38); prompting over-selects (1.65).
- **Model reliability** (within-model κ across 20 samples): reported per model.
- **Modality** unseparated (audio-human vs text-model) — a scope caveat, not measured (D5).

## Bottom line

The compactor tracks humans far better than bare prompting on *gross behavior* (how many
options, overall endorsement shape), but on the **fine structure the study was built to test —
the interference manipulation — it is not human-like: it resists the trait-swap foils humans
fall for, and over-endorses correct options.** Equivalence to humans is not established for any
model at Δ=0.22. The compactor≈human hypothesis is **not supported**; the gap is in the
"too accurate" direction, a milder version of prompting's ceiling behavior.
