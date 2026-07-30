#!/usr/bin/env Rscript
# compute_delta — estimate the equivalence bound Δ (D4) from the multi_v5 pilot.
# Human-only binomial GLM of per-option correctness on `level` (ref=control), pooled
# over topics/options. Δ = 0.5 × smallest |level coefficient| (log-odds). Base R only.
#
# Run after: python analysis/py/compute_delta_from_multi_v5.py

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
d <- read.csv(file.path(ANALYSIS, "data", "interim", "multi_v5_human_options.csv"))

d$level <- relevel(factor(d$level), ref = "control")

# per-option correctness ~ level (the manipulation's effect, on log-odds)
m <- glm(correct ~ level, family = binomial, data = d)
co <- summary(m)$coefficients
lvl_rows <- grep("^level", rownames(co))
effects <- co[lvl_rows, "Estimate", drop = TRUE]
names(effects) <- sub("^level", "", rownames(co)[lvl_rows])

cat("multi_v5 human level effects on per-option correctness (log-odds, ref=control):\n")
for (nm in names(effects)) cat(sprintf("  %-14s % .4f  (p=%.3g)\n", nm, effects[[nm]],
                                       co[paste0("level", nm), "Pr(>|z|)"]))
smallest <- min(abs(effects))
smallest_nm <- names(effects)[which.min(abs(effects))]
delta <- 0.5 * smallest
cat(sprintf("\nSmallest |level effect| = %.4f (%s)\n", smallest, smallest_nm))
cat(sprintf("Equivalence bound  Δ = 0.5 × %.4f = %.4f log-odds  (±%.4f)\n",
            smallest, delta, delta))
cat(sprintf("\nProvenance string for config/analysis.yaml:\n"))
cat(sprintf("  \"0.5 x smallest human level effect (%.4f, %s) on per-option correctness, multi_v5 GLM, %s\"\n",
            smallest, smallest_nm, format(Sys.Date())))
