#!/usr/bin/env Rscript
# 04_glmm_effect — confirmatory effect-alignment GLMM (§4) on the REAL data.
# GATE: do NOT run until 03_simulate.R (recovery) passes and PREREG is tagged (prereg-v1).
#
# One fit per underlying model M, on rows where system in {human, prompting(M),
# compactor(M)} (humans shared). Contrasts vs human per level; TOST equivalence on the
# compactor - human contrast against ±delta (config), Holm across the 3 non-control levels.
#
# Usage:  Rscript R/04_glmm_effect.R [model_name ...]   (default: all in models.yaml)
# Writes: outputs/tables/glmm_contrasts.csv , glmm_convergence.csv ,
#         outputs/logs/04_sessionInfo.txt

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
suppressWarnings(suppressMessages(library(TOSTER)))

cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
DELTA <- as.numeric(read_cfg_scalar(cfg, "bound_logodds"))
ALPHA <- as.numeric(read_cfg_scalar(cfg, "alpha", 0.05))
stopifnot(is.finite(DELTA))

# ── load canonical table (parquet via arrow, else csv) ──
load_responses <- function() {
  pq <- file.path(ANALYSIS, "data", "processed", "responses.parquet")
  csvf <- file.path(ANALYSIS, "data", "processed", "responses.csv")
  if (requireNamespace("arrow", quietly = TRUE) && file.exists(pq)) {
    as.data.frame(arrow::read_parquet(pq))
  } else {
    read.csv(csvf, stringsAsFactors = FALSE)
  }
}

d0 <- load_responses()
d0$endorsed <- as.integer(d0$endorsed %in% c(TRUE, "True", "true", 1, "1"))
# centred serial position: humans 1..4 -> -1.5..1.5 ; models have none -> 0
d0$position_c <- ifelse(is.na(d0$position), 0, as.numeric(d0$position) - 2.5)

# underlying models = model_name without the __wm suffix, intersect prompting & compactor
prompting_models <- unique(d0$model_name[d0$system == "prompting"])
cli <- commandArgs(trailingOnly = TRUE)
models <- if (length(cli)) cli else prompting_models

prep <- function(M) {
  keep <- d0[(d0$system == "human") |
             (d0$system == "prompting" & d0$model_name == M) |
             (d0$system == "compactor" & d0$model_name == paste0(M, "__wm")), ]
  if (!any(keep$system == "compactor")) return(NULL)  # need all 3 systems
  keep$system <- relevel(factor(keep$system, levels = c("human", "prompting", "compactor")), ref = "human")
  keep$level  <- relevel(factor(keep$level, levels = c("control","repeat_short","repeat_long","distractor")), ref = "control")
  keep$option_type <- factor(keep$option_type)
  keep$topic <- factor(keep$topic)
  keep$respondent <- factor(keep$respondent_id)
  keep$option_id <- factor(keep$option_id)
  keep
}

all_ct <- list(); conv <- list(); pred_all <- list()
for (M in models) {
  d <- prep(M)
  if (is.null(d)) { message(sprintf("skip %s (no compactor rows)", M)); next }
  message(sprintf("fitting %s (n=%d rows) ...", M, nrow(d)))
  f <- fit_effect_glmm(d, verbose = TRUE)
  conv[[M]] <- data.frame(model = M, converged = f$ok, rung = f$rung,
                          n_rows = nrow(d), stringsAsFactors = FALSE)
  if (!f$ok) { message(sprintf("  DID NOT CONVERGE at any rung: %s", M)); next }
  ct <- system_level_contrasts(f$fit)
  ct$model <- M
  # predicted endorsement (response scale) by system × level × option_type — feeds figure 1
  pr <- tryCatch({
    emmp <- emmeans(f$fit, ~ system | level * option_type,
                    at = list(position_c = 0), type = "response")
    p <- as.data.frame(summary(emmp))
    names(p)[names(p) == "prob"] <- "pred"
    p$model <- M
    p
  }, error = function(e) NULL)
  if (!is.null(pr)) pred_all[[M]] <- pr
  # TOST only on compactor - human, non-control levels ; Holm across the 3 levels
  is_comp <- grepl("compactor", ct$contrast) & ct$level != "control"
  tost_res <- mapply(function(e, s) {
    r <- tost(e, s, DELTA, ALPHA); c(r$p_tost, as.integer(r$equivalent))
  }, ct$estimate[is_comp], ct$SE[is_comp])
  p_holm <- p.adjust(tost_res[1, ], method = "holm")
  ct$tost_p <- NA_real_; ct$tost_p_holm <- NA_real_; ct$equivalent <- NA
  ct$tost_p[is_comp] <- tost_res[1, ]
  ct$tost_p_holm[is_comp] <- p_holm
  ct$equivalent[is_comp] <- p_holm < ALPHA
  all_ct[[M]] <- ct
}

TABLES <- file.path(ANALYSIS, "outputs", "tables")
LOGS <- file.path(ANALYSIS, "outputs", "logs")
dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)
dir.create(LOGS, showWarnings = FALSE, recursive = TRUE)
if (length(all_ct)) write.csv(do.call(rbind, all_ct),
                              file.path(TABLES, "glmm_contrasts.csv"), row.names = FALSE)
if (length(conv)) write.csv(do.call(rbind, conv),
                            file.path(TABLES, "glmm_convergence.csv"), row.names = FALSE)
if (length(pred_all)) write.csv(do.call(rbind, pred_all),
                                file.path(TABLES, "glmm_predictions.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(LOGS, "04_sessionInfo.txt"))
cat(sprintf("\nWrote glmm_contrasts.csv (%d models) + glmm_convergence.csv. delta=%.4f\n",
            length(all_ct), DELTA))
