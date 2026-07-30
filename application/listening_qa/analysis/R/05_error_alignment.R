#!/usr/bin/env Rscript
# 05_error_alignment — EXPLORATORY (D2). §5 of the plan. Base R only.
#   5b (primary, trait-swap design): false-option endorsement-rate profile correlation,
#       human vs each model, per level, decomposed by option_type (interference vs plain).
#   5a: error consistency (Cohen's kappa) over option-level correctness, Framing B
#       (each model sample vs each human), per (topic,level), with human–human noise ceiling
#       and respondent-bootstrap CIs.
#
# Usage: Rscript R/05_error_alignment.R
# Writes: outputs/tables/error_profile_corr.csv , error_kappa.csv ,
#         outputs/tables/error_profile_points.csv (for the §8 scatter)

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
NBOOT <- as.integer(read_cfg_scalar(cfg, "n_bootstrap", 2000))
SEED  <- as.integer(read_cfg_scalar(cfg, "bootstrap", 11081))
LEVELS <- c("control", "repeat_short", "repeat_long", "distractor")

d <- read.csv(file.path(ANALYSIS, "data", "processed", "responses.csv"), stringsAsFactors = FALSE)
d$endorsed <- as.integer(d$endorsed %in% c(TRUE, "True", "true", 1, "1"))
d$option_is_true <- d$option_is_true %in% c(TRUE, "True", "true", 1, "1")
d$correct <- as.integer(d$endorsed == as.integer(d$option_is_true))
models <- unique(d$model_name[d$system == "prompting"])

# ── 5b: false-option endorsement-rate profiles ──
endorse_rate <- function(sub) tapply(sub$endorsed, sub$option_id, mean)
scor <- function(a, b) suppressWarnings(cor(a, b, method = "spearman", use = "complete.obs"))

prof_rows <- list(); point_rows <- list()
human <- d[d$agent == "human", ]
for (M in models) {
  for (sysname in c("prompting", "compactor")) {
    mname <- if (sysname == "prompting") M else paste0(M, "__wm")
    md <- d[d$model_name == mname & !is.na(d$model_name), ]
    if (!nrow(md)) next
    for (lv in LEVELS) {
      for (otype in c("false_interference", "false_plain", "both_false")) {
        pick <- function(x) if (otype == "both_false") !x$option_is_true else x$option_type == otype
        hs <- human[human$level == lv & pick(human), ]
        ms <- md[md$level == lv & pick(md), ]
        hr <- endorse_rate(hs); mr <- endorse_rate(ms)
        k <- intersect(names(hr), names(mr))
        if (length(k) < 3) next
        prof_rows[[length(prof_rows)+1]] <- data.frame(
          model = M, system = sysname, level = lv, option_type = otype,
          n_options = length(k), spearman = scor(hr[k], mr[k]), stringsAsFactors = FALSE)
        if (otype == "both_false")
          point_rows[[length(point_rows)+1]] <- data.frame(
            model = M, system = sysname, level = lv, option_id = k,
            human_rate = hr[k], model_rate = mr[k],
            option_type = d$option_type[match(k, d$option_id)], stringsAsFactors = FALSE)
      }
    }
  }
}
TABLES <- file.path(ANALYSIS, "outputs", "tables"); dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)
write.csv(do.call(rbind, prof_rows), file.path(TABLES, "error_profile_corr.csv"), row.names = FALSE)
write.csv(do.call(rbind, point_rows), file.path(TABLES, "error_profile_points.csv"), row.names = FALSE)

# ── 5a: error consistency kappa (Framing B), per (topic,level) ──
# kappa between two correctness vectors over the same option set.
kappa_pair <- function(a, b) {
  p_obs <- mean(a == b)
  a1 <- mean(a); a2 <- mean(b)
  p_exp <- a1 * a2 + (1 - a1) * (1 - a2)
  if (p_exp >= 1) return(NA_real_)
  (p_obs - p_exp) / (1 - p_exp)
}
# correctness vector for one respondent over a cell's options, keyed by option_id
resp_vecs <- function(sub) {
  split_ids <- split(sub, sub$respondent_id)
  lapply(split_ids, function(s) setNames(s$correct, s$option_id))
}
mean_kappa_cross <- function(vlist_a, vlist_b, same = FALSE) {
  ks <- c()
  for (i in seq_along(vlist_a)) {
    js <- seq_along(vlist_b)
    for (j in js) {
      if (same && j <= i) next
      va <- vlist_a[[i]]; vb <- vlist_b[[j]]
      k <- intersect(names(va), names(vb))
      if (length(k) < 3) next
      ks <- c(ks, kappa_pair(va[k], vb[k]))
    }
  }
  mean(ks, na.rm = TRUE)
}

# Precompute per-pair κ MATRICES per (topic) so the respondent-level bootstrap (plan §5a:
# "Bootstrap CIs, respondent-level resampling, report CIs always") is cheap: resample human
# columns and average. MH = model-sample × human ; HH = human × human (self excluded).
kappa_matrix <- function(rowvecs, colvecs) {
  m <- matrix(NA_real_, length(rowvecs), length(colvecs))
  for (i in seq_along(rowvecs)) for (j in seq_along(colvecs)) {
    k <- intersect(names(rowvecs[[i]]), names(colvecs[[j]]))
    if (length(k) >= 3) m[i, j] <- kappa_pair(rowvecs[[i]][k], colvecs[[j]][k])
  }
  m
}
set.seed(SEED)
kappa_rows <- list()
topics <- unique(d$topic)
for (M in models) {
  for (sysname in c("prompting", "compactor")) {
    mname <- if (sysname == "prompting") M else paste0(M, "__wm")
    for (lv in LEVELS) {
      MH <- list(); HH <- list()   # per-topic matrices
      for (tp in topics) {
        hs <- human[human$level == lv & human$topic == tp, ]
        ms <- d[d$model_name == mname & !is.na(d$model_name) & d$level == lv & d$topic == tp, ]
        if (!nrow(hs) || !nrow(ms)) next
        hv <- resp_vecs(hs); mv <- resp_vecs(ms)
        MH[[tp]] <- kappa_matrix(mv, hv)
        hh <- kappa_matrix(hv, hv); diag(hh) <- NA         # exclude self-comparisons
        HH[[tp]] <- hh
      }
      if (!length(MH)) next
      point_mh   <- mean(sapply(MH, function(x) mean(x, na.rm = TRUE)), na.rm = TRUE)
      point_ceil <- mean(sapply(HH, function(x) mean(x, na.rm = TRUE)), na.rm = TRUE)
      # respondent (human) bootstrap: resample human columns within each topic, re-average
      bmh <- bceil <- numeric(NBOOT)
      for (b in 1:NBOOT) {
        mh_t <- ceil_t <- c()
        for (tp in names(MH)) {
          nh <- ncol(MH[[tp]]); cols <- sample.int(nh, nh, replace = TRUE)
          mh_t <- c(mh_t, mean(MH[[tp]][, cols], na.rm = TRUE))
          hh <- HH[[tp]][cols, cols]
          ceil_t <- c(ceil_t, mean(hh[upper.tri(hh)], na.rm = TRUE))
        }
        bmh[b] <- mean(mh_t, na.rm = TRUE); bceil[b] <- mean(ceil_t, na.rm = TRUE)
      }
      qm <- quantile(bmh, c(.025, .975), na.rm = TRUE); qc <- quantile(bceil, c(.025, .975), na.rm = TRUE)
      kappa_rows[[length(kappa_rows)+1]] <- data.frame(
        model = M, system = sysname, level = lv,
        kappa_model_human = point_mh, mh_lo = qm[1], mh_hi = qm[2],
        kappa_human_human_ceiling = point_ceil, ceil_lo = qc[1], ceil_hi = qc[2],
        stringsAsFactors = FALSE)
    }
  }
}
write.csv(do.call(rbind, kappa_rows), file.path(TABLES, "error_kappa.csv"), row.names = FALSE)
cat(sprintf("Wrote error_profile_corr.csv, error_profile_points.csv, error_kappa.csv (%d models)\n",
            length(models)))
