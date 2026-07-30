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
	@echo "Targets: ingest  validate  p1 (=ingest+validate)  clean"
	@echo "Later (post-PREREG, needs R pkgs): simulate glmm error difficulty diagnostics figures"

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

# ── later phases (stubs; wired in P2+) ──
# simulate:    $(RSCRIPT) R/03_simulate.R
# glmm:        $(RSCRIPT) R/04_glmm_effect.R
# error:       $(RSCRIPT) R/05_error_alignment.R
# difficulty:  $(RSCRIPT) R/06_difficulty.R
# diagnostics: $(RSCRIPT) R/07_diagnostics.R
# figures:     $(RSCRIPT) R/08_figures.R

all: p1
	@echo "P1 complete. Human review of $(TABLES)/cell_counts.csv required before P2."

clean:
	rm -f $(PROC)/responses.csv $(PROC)/responses.parquet $(TABLES)/cell_counts.csv .reconcile.ok
