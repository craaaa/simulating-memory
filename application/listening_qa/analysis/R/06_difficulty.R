#!/usr/bin/env Rscript
# 06_difficulty — EXPLORATORY (D2). §6 of the plan, OPTION LEVEL ONLY. Base R only.
# Per-option human endorsement rate vs model endorsement probability; Spearman.
# Noise ceiling: split humans in half N times, correlate group-mean rates, Spearman–Brown
# corrected (2r/(1+r)), averaged. Report r, ceiling, and r/ceiling — all bootstrapped
# (respondent-level resampling).
#
# GUARDRAIL (stated in outputs): the split-half r bounds explainable variance (R^2), not r.
# We report the raw ratio r/ceiling and flag this convention; do not read the ratio as
# "fraction of achievable correlation" without that caveat.
#
# Usage: Rscript R/06_difficulty.R
# Writes: outputs/tables/difficulty_corr.csv , difficulty_points.csv

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
NBOOT <- as.integer(read_cfg_scalar(cfg, "n_bootstrap", 2000))
NSPLIT <- as.integer(read_cfg_scalar(cfg, "n_splithalf", 100))
SEED_B <- as.integer(read_cfg_scalar(cfg, "bootstrap", 11081))
SEED_S <- as.integer(read_cfg_scalar(cfg, "splithalf", 22162))
LEVELS <- c("control", "repeat_short", "repeat_long", "distractor")

d <- read.csv(file.path(ANALYSIS, "data", "processed", "responses.csv"), stringsAsFactors = FALSE)
d$endorsed <- as.integer(d$endorsed %in% c(TRUE, "True", "true", 1, "1"))
models <- unique(d$model_name[d$system == "prompting"])
scor <- function(a, b) suppressWarnings(cor(a, b, method = "spearman", use = "complete.obs"))

rate_by_option <- function(sub, ids = NULL) {
  if (!is.null(ids)) sub <- sub[sub$respondent_id %in% ids, ]
  tapply(sub$endorsed, sub$option_id, mean)
}

# split-half human ceiling on option endorsement rates (pooled over levels within option)
human_ceiling <- function(hsub, nsplit, seed) {
  set.seed(seed)
  ids <- unique(hsub$respondent_id)
  vals <- numeric(0)
  for (s in 1:nsplit) {
    g1 <- sample(ids, length(ids) %/% 2)
    r1 <- rate_by_option(hsub, g1); r2 <- rate_by_option(hsub, setdiff(ids, g1))
    k <- intersect(names(r1), names(r2))
    r <- scor(r1[k], r2[k]); if (!is.na(r) && r > -1) vals <- c(vals, 2 * r / (1 + r))
  }
  mean(vals, na.rm = TRUE)
}

human <- d[d$agent == "human", ]
point_rows <- list(); rows <- list()
for (M in models) {
  for (sysname in c("prompting", "compactor")) {
    mname <- if (sysname == "prompting") M else paste0(M, "__wm")
    md <- d[d$model_name == mname & !is.na(d$model_name), ]
    if (!nrow(md)) next
    for (lv in LEVELS) {
      hs <- human[human$level == lv, ]; ms <- md[md$level == lv, ]
      hr <- rate_by_option(hs); mr <- rate_by_option(ms)
      k <- intersect(names(hr), names(mr)); if (length(k) < 5) next
      r_point <- scor(hr[k], mr[k])
      ceiling <- human_ceiling(hs, NSPLIT, SEED_S + which(LEVELS == lv))
      # respondent-bootstrap CI on r and ratio
      set.seed(SEED_B + which(LEVELS == lv))
      hids <- unique(hs$respondent_id); mids <- unique(ms$respondent_id)
      boot_r <- numeric(NBOOT)
      for (b in 1:NBOOT) {
        hb <- rate_by_option(hs, sample(hids, replace = TRUE))
        mb <- rate_by_option(ms, sample(mids, replace = TRUE))
        kb <- intersect(names(hb), names(mb)); boot_r[b] <- scor(hb[kb], mb[kb])
      }
      ci <- quantile(boot_r, c(.025, .975), na.rm = TRUE)
      rows[[length(rows)+1]] <- data.frame(
        model = M, system = sysname, level = lv, n_options = length(k),
        spearman = r_point, ci_lo = ci[1], ci_hi = ci[2],
        human_ceiling = ceiling, ratio = r_point / ceiling, stringsAsFactors = FALSE)
      point_rows[[length(point_rows)+1]] <- data.frame(
        model = M, system = sysname, level = lv, option_id = k,
        human_rate = hr[k], model_rate = mr[k],
        option_is_true = d$option_is_true[match(k, d$option_id)], stringsAsFactors = FALSE)
    }
  }
}
TABLES <- file.path(ANALYSIS, "outputs", "tables"); dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)
write.csv(do.call(rbind, rows), file.path(TABLES, "difficulty_corr.csv"), row.names = FALSE)
write.csv(do.call(rbind, point_rows), file.path(TABLES, "difficulty_points.csv"), row.names = FALSE)
cat(sprintf("Wrote difficulty_corr.csv, difficulty_points.csv (%d models). NBOOT=%d NSPLIT=%d\n",
            length(models), NBOOT, NSPLIT))
cat("NOTE: split-half r bounds R^2, not r; ratio r/ceiling reported with that caveat.\n")
