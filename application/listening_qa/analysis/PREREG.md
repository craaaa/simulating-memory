# PREREG.md — READY TO FREEZE (all decisions resolved; tag `prereg-v1` at commit)

> **Status: complete.** All open items resolved as of 2026-07-30. Freeze = commit this file
> + `config/` + `DECISIONS.md` + `DEVIATIONS.md` and tag `prereg-v1` **before the first fit
> on `multi_v6`** (P4). Do not fit §4 before the tag exists.

## Frozen-by-decision (see DECISIONS.md)

- **Confirmatory axis (D2):** effect alignment (§4). Error (§5) and difficulty (§6) exploratory.
- **Language (D1):** R for all modeling; Python only for ingest (interchange = parquet/csv).
- **Model-sample framing (D3):** A for effect/difficulty; B for error consistency.
- **Human set:** whole-participant attention pass (n=201).
- **Model set (D6):** 5 confirmatory paired models (prompting C2 + compactor C2-stream);
  GPT-4.1 exploratory. Never averaged across models.
- **Multiplicity:** Holm across the 3 non-reference `level` contrasts, confirmatory axis only.
- **Convergence ladder (§4):** rung 1 full → 2 `(level||option_id)` → 3 `(1|option_id)` → 4 BFGS; log the rung.
- **Seeds:** `config/analysis.yaml`.

## Primary model (§4)

```
endorsed ~ system * level * option_type + position_c + topic
         + (1 | respondent) + (level | option_id)
```
family = binomial; `system` treatment-coded, **ref = human** (3 levels: human, prompting,
compactor); level treatment (ref=control); topic sum-coded. Fit **per underlying model** on
rows where system ∈ {human, prompting(M), compactor(M)}; humans shared across fits.

Two confirmatory contrasts vs human, on the `system × level` marginal effects (log-odds):
- **prompting − human**: ordinary difference test (expected non-null).
- **compactor − human**: **TOST equivalence** against ±Δ (expected equivalent).

## RESOLVED

- **`system` factor.** 3-level `system`, ref=human; per-model fit; two contrasts above.
  Frozen in `config/analysis.yaml` + `DECISIONS.md` (2026-07-30, user).
- **Equivalence bound Δ (D4) = ±0.2213 log-odds** (≈ OR 1.25). 0.5 × smallest *reliable*
  human level effect (repeat_long 0.4426, p=0.005) on per-option correctness, multi_v5 GLM
  (n=21). Distractor excluded as noise — deviation logged in `DEVIATIONS.md`. Applies to the
  compactor − human contrast. Sensitivity row at literal Δ=0.056 reported at P4.

No open items remain.

## Reproducibility

`make all` (from `analysis/`) rebuilds `responses.parquet`, reconciles against run-time
ground truths, and validates — from read-only `data/raw/`. P1 is complete and green.
