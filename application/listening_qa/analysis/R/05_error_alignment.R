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

set.seed(SEED)
kappa_rows <- list()
topics <- unique(d$topic)
for (M in models) {
  for (sysname in c("prompting", "compactor")) {
    mname <- if (sysname == "prompting") M else paste0(M, "__wm")
    per_level <- list()
    for (lv in LEVELS) {
      cell_k <- c(); cell_ceiling <- c()
      for (tp in topics) {
        hs <- human[human$level == lv & human$topic == tp, ]
        ms <- d[d$model_name == mname & !is.na(d$model_name) & d$level == lv & d$topic == tp, ]
        if (!nrow(hs) || !nrow(ms)) next
        hv <- resp_vecs(hs); mv <- resp_vecs(ms)
        cell_k <- c(cell_k, mean_kappa_cross(mv, hv))            # model↔human
        cell_ceiling <- c(cell_ceiling, mean_kappa_cross(hv, hv, same = TRUE))  # human↔human
      }
      per_level[[lv]] <- c(model_human = mean(cell_k, na.rm = TRUE),
                           ceiling = mean(cell_ceiling, na.rm = TRUE))
    }
    for (lv in LEVELS) kappa_rows[[length(kappa_rows)+1]] <- data.frame(
      model = M, system = sysname, level = lv,
      kappa_model_human = per_level[[lv]]["model_human"],
      kappa_human_human_ceiling = per_level[[lv]]["ceiling"],
      stringsAsFactors = FALSE)
  }
}
write.csv(do.call(rbind, kappa_rows), file.path(TABLES, "error_kappa.csv"), row.names = FALSE)
cat(sprintf("Wrote error_profile_corr.csv, error_profile_points.csv, error_kappa.csv (%d models)\n",
            length(models)))
