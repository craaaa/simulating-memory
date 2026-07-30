# Human–Model Alignment Analysis (listening-QA)

Implements `application/listening_qa/ANALYSIS_PLAN.md`. Confirmatory axis: **effect
alignment** (does the passage manipulation move models as it moves humans?).

## Status

All phases P0–P7 implemented and run; tags `prereg-v1`, `recovery-gate-passed`.

- **P0** ✅ scaffold, `DECISIONS.md` (D1–D6), `config/`.
- **P1** ✅ option-level `responses.{parquet,csv}` + `reconcile.py` (matches run-time ground
  truths) + `02_validate.R` gates + `cell_counts.csv` (reviewed).
- **P2** ✅ `03_simulate.R` recovery gate PASSED (CI coverage ≥0.95; TOST logic; ceiling
  separation-flagging; difficulty/error recovery).
- **P3** ✅ `PREREG.md` frozen + tagged; deviations in `DEVIATIONS.md`.
- **P4** ✅ `04_glmm_effect.R` — 2-level split GLMM (separation fix), per-option_type
  contrasts + TOST. Result: no compactor equivalent to human at Δ=0.22 (see `FINDINGS.md`).
- **P5–P7** ✅ error/difficulty/diagnostics (`05`–`07`) + figures (`08`, `09`).

**Reproducibility caveat:** each stage has been run and verified individually; the one-shot
`make all` (which includes the ~15-min `simulate` gate) has not yet been executed start-to-finish
from an empty `outputs/`. The DAG is wired; run `make all` for a clean end-to-end rebuild.

## Run

```bash
cd application/listening_qa/analysis
make p1        # ingest (Python) → reconcile → validate (R); reproduces from data/raw/
```

## Layout

- `py/01_ingest.py` — builds the long table from the raw Qualtrics TSV (humans, option-level
  via the combinatorial `parse_response`) + model/compactor JSONL (`parsed_answers`).
- `tests/reconcile.py` — hard-fails unless derived `endorsed`/`option_is_true` match both
  run-time ground truths (model `parsed_answers`+`metrics`; human `longdata_strict`).
- `R/02_validate.R` — structural gates (§2) + `cell_counts.csv`.
- `config/` — `analysis.yaml` (seeds, bounds, GLMM formula), `models.yaml` (explicit paths).
- `stimuli/option_labels.yaml` — blind `option_type`/`cue_match` labels.
- `data/raw/` — read-only inputs (Qualtrics TSV, question banks, longdata).

## Key facts

- 201 humans (whole-participant attention pass, audio). 6 models × {prompting C2, compactor
  C2-stream}, n=20 draws/cell (GPT-4.1 exploratory). 4 topics × 4 levels × 5 questions × 5 options.
- Claim scope: **text-model vs audio-human** (no text-human control batch; D5).
- Results are never averaged across models.
