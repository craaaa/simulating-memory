#!/usr/bin/env Rscript
# 04_glmm_effect — confirmatory effect-alignment GLMM (§4) on the REAL data.
# GATE: run only after 03_simulate.R (incl. the ceiling/separation scenario) passes.
#
# DEVIATION from the frozen 3-level marginal spec (see DEVIATIONS.md, dated post-data):
# prompting models are COMPLETELY separated (endorse ~100% true / ~0% false) — ML log-odds
# are non-identified there, and marginalizing the contrast over option_type lets one ceiling
# stratum blow up the whole estimate. So per underlying model we fit TWO 2-level `system`
# models — (human + prompting) and (human + compactor) — and extract system×level contrasts
# WITHIN each option_type stratum. No shrinkage prior on the compactor fit (a prior shrinks
# toward "equivalent" = anti-conservative for the compactor≈human hypothesis). Contrasts from
# separated strata are flagged and reported as ceiling-limited (exploratory), not equivalenced.
#
# Usage: Rscript R/04_glmm_effect.R [model_name ...]
# Writes: outputs/tables/{glmm_contrasts.csv, glmm_convergence.csv, glmm_predictions.csv}

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
DELTA <- as.numeric(read_cfg_scalar(cfg, "bound_logodds"))
ALPHA <- as.numeric(read_cfg_scalar(cfg, "alpha", 0.05))
stopifnot(is.finite(DELTA))

load_responses <- function() {
  pq <- file.path(ANALYSIS, "data", "processed", "responses.parquet")
  csvf <- file.path(ANALYSIS, "data", "processed", "responses.csv")
  if (requireNamespace("arrow", quietly = TRUE) && file.exists(pq)) as.data.frame(arrow::read_parquet(pq))
  else read.csv(csvf, stringsAsFactors = FALSE)
}
d0 <- load_responses()
d0$endorsed <- as.integer(d0$endorsed %in% c(TRUE, "True", "true", 1, "1"))
d0$position_c <- ifelse(is.na(d0$position), 0, as.numeric(d0$position) - 2.5)
prompting_models <- unique(d0$model_name[d0$system == "prompting"])
cli <- commandArgs(trailingOnly = TRUE)
models <- if (length(cli)) cli else prompting_models

# build a 2-level (human + one system) dataset for model M
prep2 <- function(M, sysname) {
  mm <- if (sysname == "prompting") M else paste0(M, "__wm")
  keep <- d0[d0$system == "human" | (d0$system == sysname & d0$model_name == mm), ]
  if (!any(keep$system == sysname)) return(NULL)
  keep$system <- relevel(factor(keep$system, levels = c("human", sysname)), ref = "human")
  keep$level  <- relevel(factor(keep$level, levels = c("control","repeat_short","repeat_long","distractor")), ref = "control")
  keep$option_type <- factor(keep$option_type); keep$topic <- factor(keep$topic)
  keep$respondent <- factor(keep$respondent_id); keep$option_id <- factor(keep$option_id)
  keep
}

all_ct <- list(); conv <- list(); pred_all <- list()
for (M in models) {
  for (sysname in c("prompting", "compactor")) {
    d <- prep2(M, sysname); if (is.null(d)) next
    message(sprintf("fitting %s : human vs %s (n=%d) ...", M, sysname, nrow(d)))
    f <- fit_effect_glmm(d)
    conv[[paste(M, sysname)]] <- data.frame(model = M, system = sysname,
      converged = f$ok, rung = f$rung, n_rows = nrow(d), stringsAsFactors = FALSE)
    if (!f$ok) { message(sprintf("  NON-CONVERGENT: %s %s", M, sysname)); next }
    ct <- system_level_contrasts_by_ot(f$fit, ref = "human")
    if (is.null(ct)) next
    ct$model <- M
    # TOST only on the compactor - human contrast, non-control levels, ESTIMABLE strata.
    ct$tost_p <- NA_real_; ct$equivalent <- NA
    if (sysname == "compactor") {
      est_ok <- !ct$separated & ct$level != "control"
      if (any(est_ok)) {
        tp <- mapply(function(e, s) tost(e, s, DELTA, ALPHA)$p_tost,
                     ct$estimate[est_ok], ct$SE[est_ok])
        ph <- p.adjust(tp, method = "holm")
        ct$tost_p[est_ok] <- tp; ct$equivalent[est_ok] <- ph < ALPHA
      }
    }
    all_ct[[paste(M, sysname)]] <- ct
    # predictions for fig1 (this system + human), response scale, by level×option_type
    pr <- tryCatch({
      emmp <- emmeans(f$fit, ~ system | level * option_type, at = list(position_c = 0), type = "response")
      p <- as.data.frame(summary(emmp)); names(p)[names(p) == "prob"] <- "pred"; p$model <- M; p
    }, error = function(e) NULL)
    if (!is.null(pr)) pred_all[[paste(M, sysname)]] <- pr
  }
}

TABLES <- file.path(ANALYSIS, "outputs", "tables"); LOGS <- file.path(ANALYSIS, "outputs", "logs")
dir.create(TABLES, showWarnings = FALSE, recursive = TRUE); dir.create(LOGS, showWarnings = FALSE, recursive = TRUE)
if (length(all_ct)) {
  ctab <- do.call(rbind, all_ct)
  # dedupe human predictions duplicated across the two fits (keep both systems' predicted rows)
  write.csv(ctab, file.path(TABLES, "glmm_contrasts.csv"), row.names = FALSE)
}
if (length(conv)) write.csv(do.call(rbind, conv), file.path(TABLES, "glmm_convergence.csv"), row.names = FALSE)
if (length(pred_all)) {
  pr <- do.call(rbind, pred_all)
  pr <- pr[!(pr$system == "human" & duplicated(pr[, c("model","level","option_type","system")])), ]
  write.csv(pr, file.path(TABLES, "glmm_predictions.csv"), row.names = FALSE)
}
writeLines(capture.output(sessionInfo()), file.path(LOGS, "04_sessionInfo.txt"))
cat(sprintf("\nWrote glmm_contrasts.csv + convergence + predictions. delta=%.4f\n", DELTA))
