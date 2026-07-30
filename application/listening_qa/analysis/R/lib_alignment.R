# lib_alignment.R — shared fitting/inference for the effect-alignment pipeline (§4).
# Sourced by 03_simulate.R (recovery gate) and 04_glmm_effect.R (real fit).
# Depends: glmmTMB, emmeans. TOST implemented inline (two one-sided z-tests on the
# contrast estimate/SE) so it needs no TOSTER at fit time; TOSTER used only for reporting.

suppressWarnings(suppressMessages({
  library(glmmTMB)
  library(emmeans)
}))

# ── tiny YAML scalar reader (avoids an R yaml dependency for a few scalars) ──
read_cfg_scalar <- function(path, key, default = NA) {
  txt <- readLines(path, warn = FALSE)
  hit <- grep(sprintf("^\\s*%s\\s*:", key), txt, value = TRUE)
  if (length(hit) == 0) return(default)
  v <- sub(sprintf("^\\s*%s\\s*:\\s*", key), "", hit[1])
  v <- trimws(sub("#.*$", "", v))
  num <- suppressWarnings(as.numeric(v))
  if (!is.na(num)) num else gsub('^"|"$', "", v)
}

# ── convergence ladder (§4): rung 1 full → 2 (level||option_id) → 3 (1|option_id) → 4 BFGS ──
LADDER <- list(
  list(rung = 1, re = "(1 | respondent) + (level | option_id)", opt = "nlminb"),
  list(rung = 2, re = "(1 | respondent) + (level || option_id)", opt = "nlminb"),
  list(rung = 3, re = "(1 | respondent) + (1 | option_id)",      opt = "nlminb"),
  list(rung = 4, re = "(1 | respondent) + (1 | option_id)",      opt = "BFGS")
)

fixed_part <- "endorsed ~ system * level * option_type + position_c + topic"

#' Fit the effect-alignment GLMM, descending the convergence ladder until one converges.
#' Returns list(fit, rung, ok). `d` must have columns:
#'   endorsed (0/1), system (factor, ref=human), level (factor, ref=control),
#'   option_type (factor), position_c (numeric, 0 for models), topic (factor),
#'   respondent (factor), option_id (factor).
fit_effect_glmm <- function(d, verbose = FALSE) {
  for (step in LADDER) {
    form <- as.formula(paste(fixed_part, "+", step$re))
    ctrl <- glmmTMBControl(optimizer = optim, optArgs = list(method = step$opt))
    if (step$opt == "nlminb") ctrl <- glmmTMBControl()
    fit <- tryCatch(
      suppressWarnings(glmmTMB(form, family = binomial, data = d, control = ctrl)),
      error = function(e) NULL)
    if (is.null(fit)) next
    conv <- fit$fit$convergence
    # Accept on: optimizer converged AND all FIXED-effect SEs finite. A boundary/near-zero
    # random-effect variance (singular RE) is NOT a failure — glmmTMB handles it and the
    # fixed-effect (system×level) inference we need stays valid. We still descend the ladder
    # when the optimizer fails or the Hessian is degenerate (non-finite SEs).
    ses <- tryCatch(sqrt(diag(vcov(fit)$cond)), error = function(e) NA)
    sdok <- length(ses) > 0 && all(is.finite(ses))
    pdh <- tryCatch(fit$sdr$pdHess, error = function(e) FALSE)
    if (!is.null(conv) && conv == 0 && sdok && isTRUE(pdh)) {
      if (verbose) message(sprintf("converged at rung %d (%s)", step$rung, step$opt))
      return(list(fit = fit, rung = step$rung, ok = TRUE))
    }
  }
  list(fit = fit, rung = NA, ok = FALSE)
}

#' System×level contrasts vs human on the log-odds scale, marginal over option_type/topic.
#' Returns a data.frame: contrast (system diff), level, estimate, SE, lower, upper.
system_level_contrasts <- function(fit) {
  emm <- emmeans(fit, ~ system | level, at = list(position_c = 0))
  ct <- contrast(emm, method = "trt.vs.ctrl", ref = "human")  # each system - human, per level
  s <- as.data.frame(summary(ct, infer = c(TRUE, FALSE)))
  data.frame(contrast = s$contrast, level = s$level,
             estimate = s$estimate, SE = s$SE,
             lower = s$asymp.LCL, upper = s$asymp.UCL,
             stringsAsFactors = FALSE)
}

#' System×level contrasts vs a reference, computed WITHIN each option_type stratum
#' (NOT marginalized over option_type — a single ceiling stratum would otherwise drive the
#' marginal to ±Inf and collapse the SDT decomposition). Returns rows with an `option_type`
#' column and a `separated` flag (|estimate|>10 or SE>10 => non-identified / quasi-separation).
system_level_contrasts_by_ot <- function(fit, ref = "human") {
  ots <- levels(fit$frame$option_type)
  out <- list()
  for (ot in ots) {
    emm <- tryCatch(emmeans(fit, ~ system | level,
                            at = list(position_c = 0, option_type = ot)),
                    error = function(e) NULL)
    if (is.null(emm)) next
    s <- tryCatch(as.data.frame(summary(contrast(emm, "trt.vs.ctrl", ref = ref),
                                         infer = c(TRUE, FALSE))),
                  error = function(e) NULL)
    if (is.null(s)) next
    out[[ot]] <- data.frame(contrast = s$contrast, level = s$level, option_type = ot,
                            estimate = s$estimate, SE = s$SE,
                            lower = s$asymp.LCL, upper = s$asymp.UCL,
                            separated = (abs(s$estimate) > 10 | s$SE > 10),
                            stringsAsFactors = FALSE)
  }
  do.call(rbind, out)
}

#' TOST equivalence on a single contrast estimate/SE against symmetric bound ±delta.
#' Equivalent iff BOTH one-sided tests reject at alpha (i.e. the (1-2a) CI ⊂ (-delta,delta)).
tost <- function(estimate, se, delta, alpha = 0.05) {
  z_low  <- (estimate - (-delta)) / se   # H0: <= -delta
  z_high <- (estimate - ( delta)) / se   # H0: >=  delta
  p_low  <- pnorm(z_low, lower.tail = FALSE)  # reject if estimate > -delta
  p_high <- pnorm(z_high, lower.tail = TRUE)  # reject if estimate <  delta
  p_tost <- max(p_low, p_high)
  ci_mult <- qnorm(1 - alpha)
  lo <- estimate - ci_mult * se
  hi <- estimate + ci_mult * se
  list(p_tost = p_tost, equivalent = (p_tost < alpha),
       inner_lo = lo, inner_hi = hi, delta = delta)
}
