#!/usr/bin/env Rscript
# 03_simulate — synthetic-data recovery gate (ANALYSIS_PLAN.md §3).
# MUST pass before any fit on data/raw/ (P2 gate). Validates, on data with KNOWN parameters:
#   (a) GLMM recovers the true system×level interaction within its CI in >=90% of sims
#       (both compactor and prompting);
#   (b) NULL world (no interaction) is not manufactured into a difference — the prompting−human
#       "significant difference" rate is ~alpha, and the true 0 is covered >=90%;
#   (c) TOST equivalence logic behaves: declares equivalence when the estimate lies inside
#       ±delta and not when outside (deterministic unit tests — power-independent);
#   (d) error-overlap (§5b) and difficulty (§6) Spearman correlations recover known signal.
#
# The GLMM recovery is decoupled from TOST power on purpose: whether the REAL data can
# actually power the compactor≈human equivalence test at delta is a data question, assessed
# at P4 — not something a simulation can pre-ordain. Here we verify the machinery is correct.
#
# Usage:  Rscript R/03_simulate.R [nsim]      (nsim default 60)
# Writes: outputs/logs/03_simulate_recovery.json , 03_sessionInfo.txt

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))

cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
DELTA <- as.numeric(read_cfg_scalar(cfg, "bound_logodds", 0.2213))
SEED  <- as.integer(read_cfg_scalar(cfg, "simulate", 20260730))
ALPHA <- as.numeric(read_cfg_scalar(cfg, "alpha", 0.05))
cli <- commandArgs(trailingOnly = TRUE)
NSIM <- if (length(cli) >= 1) as.integer(cli[1]) else 60L

LEVELS <- c("control", "repeat_short", "repeat_long", "distractor")
TOPICS <- c("astronomy", "fruits", "martial_arts", "fabrics")

# ── one synthetic dataset. Interaction (system×level, on TRUE options) is the recovery
#    target: for non-control levels, compactor gets +int_c, prompting +int_p, human 0. ──
simulate_one <- function(seed, int_c, int_p, n_humans = 200, n_draws = 25,
                         b0 = -0.4, b_true = 2.0, sd_opt = 0.8, lvl_main = 0.3) {
  set.seed(seed)
  opts <- expand.grid(topic = TOPICS, q = 1:3, o = 1:4, stringsAsFactors = FALSE)
  opts$question_id <- paste0(opts$topic, "_Q", opts$q)
  opts$option_id   <- paste0(opts$question_id, "_o", opts$o)
  opts$option_is_true <- opts$o %in% c(1, 2)
  opts$option_type <- ifelse(opts$option_is_true, "true",
                       ifelse(opts$o == 3, "false_interference", "false_plain"))
  u_o <- rnorm(nrow(opts), 0, sd_opt); names(u_o) <- opts$option_id
  posv <- c(-1.5, -0.5, 0.5, 1.5)
  int <- list(human = 0, prompting = int_p, compactor = int_c)

  mk <- function(sys, specs) {
    do.call(rbind, lapply(specs, function(sp) {
      sub <- opts[opts$topic == sp$topic, ]
      shift <- if (sp$level == "control") 0 else int[[sys]]
      main  <- if (sp$level == "control") 0 else lvl_main
      lin <- b0 + b_true * sub$option_is_true + u_o[sub$option_id] +
             (main + shift) * sub$option_is_true
      data.frame(respondent = sp$id, system = sys, level = sp$level, topic = sp$topic,
                 question_id = sub$question_id, option_id = sub$option_id,
                 option_is_true = sub$option_is_true, option_type = sub$option_type,
                 position_c = sp$pos, endorsed = rbinom(nrow(sub), 1, plogis(lin)),
                 stringsAsFactors = FALSE)
    }))
  }
  hs <- lapply(seq_len(n_humans), function(h) {
    lv <- sample(LEVELS); tp <- sample(TOPICS)
    lapply(1:4, function(k) list(id = paste0("H", h), level = lv[k], topic = tp[k],
                                 pos = posv[k]))
  })
  hs <- unlist(hs, recursive = FALSE)
  ms <- function(sys) {
    out <- list()
    for (tp in TOPICS) for (lv in LEVELS) for (r in 1:n_draws)
      out[[length(out)+1]] <- list(id = paste0(sys,"_",tp,"_",lv,"_r",r),
                                   level = lv, topic = tp, pos = 0)
    out
  }
  d <- rbind(mk("human", hs), mk("prompting", ms("prompting")), mk("compactor", ms("compactor")))
  d$system <- relevel(factor(d$system), ref = "human")
  d$level  <- relevel(factor(d$level),  ref = "control")
  d$option_type <- factor(d$option_type); d$topic <- factor(d$topic)
  d$respondent <- factor(d$respondent); d$option_id <- factor(d$option_id)
  list(d = d, u_o = u_o, opts = opts)
}

int_on_true <- function(fit) {
  emm <- emmeans(fit, ~ system | level, at = list(position_c = 0, option_type = "true"))
  as.data.frame(summary(contrast(emm, "trt.vs.ctrl", ref = "human"), infer = c(TRUE, FALSE)))
}

# ── (a)+(b) GLMM recovery over NSIM sims ──
message(sprintf("Recovery gate: NSIM=%d, delta=%.4f, seed=%d", NSIM, DELTA, SEED))
TRUE_C <- 0.20; TRUE_P <- 0.60      # true interactions at non-control levels
cov_c <- cov_p <- fp_null <- cov_null <- 0; n_ok <- 0L
for (i in 1:NSIM) {
  # alternate: even i = signal world (int as above); odd i = null world (both 0)
  null_world <- (i %% 2 == 1)
  ic <- if (null_world) 0 else TRUE_C; ip <- if (null_world) 0 else TRUE_P
  sim <- simulate_one(SEED + i, int_c = ic, int_p = ip)
  f <- fit_effect_glmm(sim$d)
  if (!f$ok) next
  ct <- tryCatch(int_on_true(f$fit), error = function(e) NULL)
  if (is.null(ct)) next
  n_ok <- n_ok + 1L
  ct <- ct[ct$level != "control", ]
  cc <- ct[grepl("compactor", ct$contrast), ]; pp <- ct[grepl("prompting", ct$contrast), ]
  if (null_world) {
    cov_null <- cov_null + mean(cc$asymp.LCL <= 0 & 0 <= cc$asymp.UCL)
    # difference-test false positives: fraction of prompting contrasts whose CI excludes 0
    fp_null <- fp_null + mean(!(pp$asymp.LCL <= 0 & 0 <= pp$asymp.UCL))
  } else {
    cov_c <- cov_c + mean(cc$asymp.LCL <= TRUE_C & TRUE_C <= cc$asymp.UCL)
    cov_p <- cov_p + mean(pp$asymp.LCL <= TRUE_P & TRUE_P <= pp$asymp.UCL)
  }
}
n_signal <- sum((1:NSIM) %% 2 == 0); n_null <- NSIM - n_signal
rec <- list(n_ok = n_ok,
            ci_cov_compactor = cov_c / n_signal, ci_cov_prompting = cov_p / n_signal,
            ci_cov_null = cov_null / n_null, null_false_pos = fp_null / n_null)

# ── (c) TOST logic unit tests (deterministic) ──
t_inside  <- tost(estimate = 0.00, se = 0.05, delta = DELTA, alpha = ALPHA)
t_bound_in<- tost(estimate = DELTA * 0.5, se = 0.02, delta = DELTA, alpha = ALPHA)
t_outside <- tost(estimate = DELTA * 3,   se = 0.05, delta = DELTA, alpha = ALPHA)
t_wide    <- tost(estimate = 0.00, se = DELTA,       delta = DELTA, alpha = ALPHA) # underpowered → not equiv
tost_ok <- t_inside$equivalent && t_bound_in$equivalent && !t_outside$equivalent && !t_wide$equivalent

# ── (d) error-overlap (§5b) + difficulty (§6) recovery ──
recover_corr <- function(seed) {
  sim <- simulate_one(seed, int_c = 0, int_p = 0.6, n_humans = 150, n_draws = 30)
  d <- sim$d
  rate <- function(sys, only_false = FALSE) {
    dd <- d[d$system == sys, ]; if (only_false) dd <- dd[!dd$option_is_true, ]
    # droplevels so absent option_id levels don't become NA entries in tapply
    tapply(dd$endorsed, droplevels(dd$option_id), mean)
  }
  scor <- function(a, b) suppressWarnings(cor(a, b, method = "spearman", use = "complete.obs"))
  h <- rate("human"); m <- rate("compactor"); k <- intersect(names(h), names(m))
  hf <- rate("human", TRUE); mf <- rate("compactor", TRUE); kf <- intersect(names(hf), names(mf))
  set.seed(seed + 7)
  hh <- d[d$system == "human", ]; ids <- unique(hh$respondent); g1 <- sample(ids, length(ids)%/%2)
  r1 <- tapply(hh$endorsed[hh$respondent %in% g1], droplevels(hh$option_id[hh$respondent %in% g1]), mean)
  r2 <- tapply(hh$endorsed[!hh$respondent %in% g1], droplevels(hh$option_id[!hh$respondent %in% g1]), mean)
  kk <- intersect(names(r1), names(r2)); sh <- scor(r1[kk], r2[kk])
  list(difficulty_spearman = scor(h[k], m[k]),
       error_false_spearman = scor(hf[kf], mf[kf]),
       splithalf = sh, ceiling = 2*sh/(1+sh))
}
corr <- recover_corr(SEED + 999L)

# ── (e) CEILING / SEPARATION scenario ─────────────────────────────────────────────
# A 2-level (human + model) world where the model is at ceiling on TRUE options
# (~99% endorse → quasi-separation) but has a KNOWN interaction on the FALSE stratum.
# Validates the refit strategy (per-option_type contrasts): the separated `true` stratum
# must be flagged, and the `false_*` contrast must stay estimable and recover the truth.
simulate_ceiling <- function(seed, int_false, n_humans = 150, n_draws = 25) {
  set.seed(seed)
  opts <- expand.grid(topic = TOPICS, q = 1:3, o = 1:4, stringsAsFactors = FALSE)
  opts$question_id <- paste0(opts$topic, "_Q", opts$q); opts$option_id <- paste0(opts$question_id, "_o", opts$o)
  opts$option_is_true <- opts$o %in% c(1, 2)
  opts$option_type <- ifelse(opts$option_is_true, "true", ifelse(opts$o == 3, "false_interference", "false_plain"))
  u_o <- rnorm(nrow(opts), 0, 0.8); names(u_o) <- opts$option_id
  mk <- function(sys, specs, ceil) do.call(rbind, lapply(specs, function(sp) {
    sub <- opts[opts$topic == sp$topic, ]
    if (ceil) {  # model: PERFECT separation on true (deterministic 1); interaction on FALSE
      shift_f <- if (sp$level == "control") 0 else int_false
      pf <- plogis(-2.6 + u_o[sub$option_id] + shift_f)
      end <- ifelse(sub$option_is_true, 1L, rbinom(nrow(sub), 1, pf))  # true always endorsed → separation
    } else {     # human: moderate on true, ~0.1 on false
      lin <- ifelse(sub$option_is_true, 0.6 + u_o[sub$option_id], -2.2 + u_o[sub$option_id])
      end <- rbinom(nrow(sub), 1, plogis(lin))
    }
    data.frame(respondent = sp$id, system = sys, level = sp$level, topic = sp$topic,
               question_id = sub$question_id, option_id = sub$option_id,
               option_is_true = sub$option_is_true, option_type = sub$option_type,
               position_c = sp$pos, endorsed = end, stringsAsFactors = FALSE)
  }))
  hs <- unlist(lapply(seq_len(n_humans), function(h) { lv <- sample(LEVELS); tp <- sample(TOPICS)
    lapply(1:4, function(k) list(id = paste0("H", h), level = lv[k], topic = tp[k], pos = c(-1.5,-.5,.5,1.5)[k])) }),
    recursive = FALSE)
  ms <- list(); for (tp in TOPICS) for (lv in LEVELS) for (r in 1:n_draws)
    ms[[length(ms)+1]] <- list(id = paste0("M_",tp,"_",lv,"_r",r), level = lv, topic = tp, pos = 0)
  d <- rbind(mk("human", hs, FALSE), mk("compactor", ms, TRUE))
  d$system <- relevel(factor(d$system), ref = "human"); d$level <- relevel(factor(d$level), ref = "control")
  d$option_type <- factor(d$option_type); d$topic <- factor(d$topic)
  d$respondent <- factor(d$respondent); d$option_id <- factor(d$option_id)
  d
}
TRUE_FALSE_INT <- 0.5
ceil_true_sep <- ceil_false_cov <- 0; n_ceil <- 0L
for (i in 1:20) {
  d <- simulate_ceiling(SEED + 5000L + i, int_false = TRUE_FALSE_INT)
  f <- fit_effect_glmm(d); if (!f$ok) next
  ct <- tryCatch(system_level_contrasts_by_ot(f$fit, ref = "human"), error = function(e) NULL)
  if (is.null(ct)) next
  n_ceil <- n_ceil + 1L
  tr <- ct[ct$option_type == "true" & ct$level != "control", ]
  fa <- ct[ct$option_type %in% c("false_interference","false_plain") & ct$level != "control", ]
  ceil_true_sep <- ceil_true_sep + mean(tr$separated)                                  # true stratum flagged
  fa_ok <- fa[!fa$separated, ]
  if (nrow(fa_ok)) ceil_false_cov <- ceil_false_cov + mean(fa_ok$lower <= TRUE_FALSE_INT & TRUE_FALSE_INT <= fa_ok$upper)
}
ceiling <- list(true_stratum_flagged = ceil_true_sep / n_ceil,
                false_stratum_coverage = ceil_false_cov / n_ceil, n = n_ceil)

# ── assertions ──
fails <- character(0); gate <- function(ok, m) if (!isTRUE(ok)) fails <<- c(fails, m)
gate(rec$ci_cov_compactor >= 0.90, sprintf("CI coverage compactor int = %.3f (<.90)", rec$ci_cov_compactor))
gate(rec$ci_cov_prompting >= 0.90, sprintf("CI coverage prompting int = %.3f (<.90)", rec$ci_cov_prompting))
gate(rec$ci_cov_null >= 0.90,      sprintf("CI coverage null (0) = %.3f (<.90)", rec$ci_cov_null))
gate(rec$null_false_pos <= 0.15,   sprintf("null difference false-positive rate = %.3f (>.15)", rec$null_false_pos))
gate(tost_ok, sprintf("TOST logic unit tests failed (inside=%s bound=%s outside=%s wide=%s)",
                      t_inside$equivalent, t_bound_in$equivalent, t_outside$equivalent, t_wide$equivalent))
gate(corr$difficulty_spearman >= 0.5, sprintf("difficulty Spearman not recovered = %.3f", corr$difficulty_spearman))
gate(corr$error_false_spearman >= 0.3, sprintf("false-option overlap Spearman not recovered = %.3f", corr$error_false_spearman))
# HARD gate: a ceiling (separated) true stratum must be flagged, not silently estimated.
gate(ceiling$true_stratum_flagged >= 0.90, sprintf("ceiling: true stratum not flagged separated = %.3f (<.90)", ceiling$true_stratum_flagged))
# DIAGNOSTIC (not a hard gate): coverage of the sparse false-stratum fallback under ceiling.
# Poor coverage here is itself a FINDING — it means a near-ceiling compactor's equivalence
# read (which must fall back to the false stratum) is unreliable and is reported ceiling-limited.
if (ceiling$false_stratum_coverage < 0.80)
  cat(sprintf("WARNING: ceiling false-stratum coverage = %.3f (<.80) — near-ceiling models' equivalence contrasts are unreliable; report them ceiling-limited/exploratory.\n",
              ceiling$false_stratum_coverage))

LOGS <- file.path(ANALYSIS, "outputs", "logs"); dir.create(LOGS, showWarnings = FALSE, recursive = TRUE)
fmt <- function(x) formatC(x, digits = 4, format = "f")
jsonify <- function(x) {
  if (is.list(x)) paste0("{", paste(sprintf('"%s":%s', names(x), sapply(x, jsonify)), collapse=","), "}")
  else if (is.character(x)) paste0("\"", paste(x, collapse="; "), "\"")
  else if (is.logical(x)) tolower(as.character(x))
  else if (length(x) == 1) fmt(x) else paste0("[", paste(sapply(x, jsonify), collapse=","), "]")
}
out <- list(nsim = NSIM, delta = DELTA, seed = SEED, n_ok = rec$n_ok,
            passed = length(fails) == 0, failures = if (length(fails)) fails else "none",
            recovery = rec, tost_logic_ok = tost_ok, corr = corr, ceiling = ceiling)
writeLines(jsonify(out), file.path(LOGS, "03_simulate_recovery.json"))
writeLines(capture.output(sessionInfo()), file.path(LOGS, "03_sessionInfo.txt"))

cat("\n=== recovery summary ===\n")
cat(sprintf("CI coverage: compactor=%.3f prompting=%.3f null=%.3f | null false-pos=%.3f\n",
            rec$ci_cov_compactor, rec$ci_cov_prompting, rec$ci_cov_null, rec$null_false_pos))
cat(sprintf("TOST logic ok=%s | difficulty r=%.3f error-false r=%.3f\n",
            tost_ok, corr$difficulty_spearman, corr$error_false_spearman))
cat(sprintf("ceiling: true-stratum flagged=%.3f  false-stratum coverage=%.3f (n=%d)\n",
            ceiling$true_stratum_flagged, ceiling$false_stratum_coverage, ceiling$n))
cat("========================\n")
if (length(fails) == 0) { cat("RECOVERY GATE PASSED.\n"); quit(status = 0) }
cat("RECOVERY GATE FAILED:\n"); for (f in fails) cat("  -", f, "\n"); quit(status = 1)
