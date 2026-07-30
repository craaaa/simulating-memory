#!/usr/bin/env Rscript
# 07_diagnostics — §7. Cheap descriptive checks + one gated GLMM (prosody cue leakage).
# Base R + glmmTMB (only for the cue-leakage fit). Each pre-empts a reviewer question.
#
# Usage: Rscript R/07_diagnostics.R
# Writes: outputs/tables/{selection_counts.csv, model_reliability.csv,
#         position_effect.csv, prosody_cue_leakage.csv}

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
LEVELS <- c("control", "repeat_short", "repeat_long", "distractor")

d <- read.csv(file.path(ANALYSIS, "data", "processed", "responses.csv"), stringsAsFactors = FALSE)
d$endorsed <- as.integer(d$endorsed %in% c(TRUE, "True", "true", 1, "1"))
d$option_is_true <- d$option_is_true %in% c(TRUE, "True", "true", 1, "1")
d$correct <- as.integer(d$endorsed == as.integer(d$option_is_true))
TABLES <- file.path(ANALYSIS, "outputs", "tables"); dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)

agent_label <- function(df) ifelse(df$agent == "human", "human", paste0(df$system))

# ── (1) Selection-count distribution: # options endorsed per question, per system ──
d$sys_label <- agent_label(d)
sel <- aggregate(endorsed ~ sys_label + respondent_id + topic + level + question_id, data = d, FUN = sum)
selsum <- aggregate(endorsed ~ sys_label, data = sel,
                    FUN = function(x) c(mean = mean(x), sd = sd(x),
                                        p0 = mean(x == 0), p1 = mean(x == 1),
                                        p2 = mean(x == 2), p3plus = mean(x >= 3)))
sc <- do.call(rbind, lapply(seq_len(nrow(selsum)), function(i)
  data.frame(system = selsum$sys_label[i], t(selsum$endorsed[i, ]))))
write.csv(sc, file.path(TABLES, "selection_counts.csv"), row.names = FALSE)
# full per-question counts for the fig-6 histogram
write.csv(sel, file.path(TABLES, "selection_counts_raw.csv"), row.names = FALSE)

# ── (2) Model reliability: within-model κ across the 20 samples (option-level correctness) ──
kappa_pair <- function(a, b) {
  p_obs <- mean(a == b); a1 <- mean(a); a2 <- mean(b)
  pe <- a1 * a2 + (1 - a1) * (1 - a2); if (pe >= 1) return(NA_real_); (p_obs - pe) / (1 - pe)
}
rel_rows <- list()
for (mname in unique(d$model_name[!is.na(d$model_name)])) {
  md <- d[d$model_name == mname & !is.na(d$model_name), ]
  ks <- c()
  for (tp in unique(md$topic)) for (lv in LEVELS) {
    cell <- md[md$topic == tp & md$level == lv, ]
    vs <- split(cell, cell$respondent_id)
    vs <- lapply(vs, function(s) setNames(s$correct, s$option_id))
    ids <- seq_along(vs)
    for (i in ids) for (j in ids) if (j > i) {
      k <- intersect(names(vs[[i]]), names(vs[[j]]))
      if (length(k) >= 3) ks <- c(ks, kappa_pair(vs[[i]][k], vs[[j]][k]))
    }
  }
  rel_rows[[length(rel_rows)+1]] <- data.frame(model_name = mname,
    system = md$system[1], within_model_kappa = mean(ks, na.rm = TRUE))
}
write.csv(do.call(rbind, rel_rows), file.path(TABLES, "model_reliability.csv"), row.names = FALSE)

# ── (3) Position effect (humans only): mean correctness by serial position ──
h <- d[d$agent == "human" & !is.na(d$position), ]
pe <- aggregate(correct ~ position, data = h, FUN = mean)
pe_n <- aggregate(respondent_id ~ position, data = h, FUN = function(x) length(unique(x)))
pe <- merge(pe, pe_n); names(pe)[3] <- "n_humans"
write.csv(pe, file.path(TABLES, "position_effect.csv"), row.names = FALSE)

# ── (4) Prosody cue leakage (gated GLMM): does the audio-human advantage concentrate on
#    cue_match==exact vs paraphrase? Fit on TRUE options (where cue_match is meaningful):
#    endorsed ~ agent * cue_match + (1|respondent) + (1|option_id). agent = human vs model. ──
tp_ok <- tryCatch({
  dt <- d[d$option_is_true & d$cue_match %in% c("exact", "paraphrase"), ]
  dt$agent <- relevel(factor(dt$agent), ref = "model")
  dt$cue_match <- factor(dt$cue_match)
  dt$respondent <- factor(dt$respondent_id); dt$option_id <- factor(dt$option_id)
  m <- suppressWarnings(glmmTMB(endorsed ~ agent * cue_match + (1 | respondent) + (1 | option_id),
                                family = binomial, data = dt))
  co <- as.data.frame(summary(m)$coefficients$cond)
  co$term <- rownames(co)
  write.csv(co[, c("term","Estimate","Std. Error","Pr(>|z|)")],
            file.path(TABLES, "prosody_cue_leakage.csv"), row.names = FALSE)
  TRUE
}, error = function(e) { message("prosody fit failed: ", conditionMessage(e)); FALSE })

cat(sprintf("Diagnostics written. selection_counts, model_reliability, position_effect%s\n",
            if (tp_ok) ", prosody_cue_leakage" else " (prosody fit skipped)"))
cat("MODALITY: audio-human vs text-model is unseparated (no text-human batch, D5) — scope claims accordingly.\n")
