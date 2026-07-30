# Makefile — reproduce the human–model alignment analysis from data/raw/.
# Run from application/listening_qa/analysis/.  `make all` is the full pipeline.
# Phases beyond P1 are stubbed until PREREG.md is tagged (P3) and R packages installed.

PY      := PYTHONWARNINGS=ignore python3
RSCRIPT := Rscript
PROC    := data/processed
TABLES  := outputs/tables

.PHONY: all ingest validate clean p1 help
.DEFAULT_GOAL := help

help:
	@echo "Targets: ingest validate p1 | simulate glmm error difficulty diagnostics figures | all clean"
	@echo "  all = p1 -> recovery gate -> glmm+error+difficulty+diagnostics -> figures"

# ── P1: data contract + validation ──
$(PROC)/responses.csv: py/01_ingest.py config/models.yaml stimuli/option_labels.yaml \
                       data/raw/human_results_multi_v6.tsv
	$(PY) py/01_ingest.py

ingest: $(PROC)/responses.csv

# reconcile derived endorsed/option_is_true against run-time ground truths
# (model parsed_answers+metrics ; human longdata_strict correct+partial). Gate before validate.
.reconcile.ok: tests/reconcile.py $(PROC)/responses.csv
	$(PY) tests/reconcile.py && touch $@

reconcile: .reconcile.ok

$(TABLES)/cell_counts.csv: R/02_validate.R $(PROC)/responses.csv .reconcile.ok
	$(RSCRIPT) R/02_validate.R

validate: $(TABLES)/cell_counts.csv

p1: reconcile validate

# ── P2: synthetic-data recovery gate (must pass before any real fit) ──
outputs/logs/03_simulate_recovery.json: R/03_simulate.R R/lib_alignment.R
	$(RSCRIPT) R/03_simulate.R
	@grep -q '"passed":true' $@ || (echo "RECOVERY GATE FAILED"; exit 1)
simulate: outputs/logs/03_simulate_recovery.json
.PHONY: simulate glmm error difficulty diagnostics figures

# ── P4-P6: analyses (real fits gated on the recovery gate) ──
$(TABLES)/glmm_contrasts.csv: R/04_glmm_effect.R R/lib_alignment.R $(PROC)/responses.csv outputs/logs/03_simulate_recovery.json
	$(RSCRIPT) R/04_glmm_effect.R
glmm: $(TABLES)/glmm_contrasts.csv

$(TABLES)/error_kappa.csv: R/05_error_alignment.R $(PROC)/responses.csv
	$(RSCRIPT) R/05_error_alignment.R
error: $(TABLES)/error_kappa.csv

$(TABLES)/difficulty_corr.csv: R/06_difficulty.R $(PROC)/responses.csv
	$(RSCRIPT) R/06_difficulty.R
difficulty: $(TABLES)/difficulty_corr.csv

$(TABLES)/selection_counts_raw.csv: R/07_diagnostics.R $(PROC)/responses.csv
	$(RSCRIPT) R/07_diagnostics.R
diagnostics: $(TABLES)/selection_counts_raw.csv

# ── P7: figures ──
figures: R/08_figures.R glmm error difficulty diagnostics
	$(RSCRIPT) R/08_figures.R

all: p1 simulate glmm error difficulty diagnostics figures
	@echo "Pipeline complete. Figures in $(ANALYSIS_FIG); tables in $(TABLES)."

clean:
	rm -f $(PROC)/responses.csv $(PROC)/responses.parquet $(TABLES)/*.csv .reconcile.ok \
	      outputs/logs/03_simulate_recovery.json outputs/figures/*.png
