# Human–Model Alignment Analysis (listening-QA)

Implements `application/listening_qa/ANALYSIS_PLAN.md`. Confirmatory axis: **effect
alignment** (does the passage manipulation move models as it moves humans?).

## Status

- **P0** ✅ scaffold, `DECISIONS.md` (D1–D6), `config/`.
- **P1** ✅ option-level data contract + reconciliation + validation.
  - `data/processed/responses.{parquet,csv}` — one row per (respondent, text, question, option).
  - `outputs/tables/cell_counts.csv` — **awaiting human review before P2**.
- **P2–P7** ⏳ not started (simulate → GLMM → error → difficulty → diagnostics → figures).
- `PREREG.md` is a **draft** — freeze + tag `prereg-v1` before the first `multi_v6` fit.

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
