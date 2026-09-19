#!/usr/bin/env Rscript
# 02_validate — hard-fail validation gates for the option-level long table
# (ANALYSIS_PLAN.md §2). Base R only (reads responses.csv; no package installs).
# Emits outputs/tables/cell_counts.csv for the human-review gate (P1).
#
# Run:  Rscript analysis/R/02_validate.R
# Exits non-zero on any gate failure.

suppressWarnings({
  args <- commandArgs(trailingOnly = FALSE)
  here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
})
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
PROC <- file.path(ANALYSIS, "data", "processed")
TABLES <- file.path(ANALYSIS, "outputs", "tables")
LOGS <- file.path(ANALYSIS, "outputs", "logs")
dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)
dir.create(LOGS, showWarnings = FALSE, recursive = TRUE)

LEVELS <- c("control", "repeat_short", "repeat_long", "distractor")
TOPICS <- c("astronomy", "fruits", "martial_arts", "fabrics")
MIN_HUMANS_PER_CELL <- 25
N_CONTENT_Q <- 5
N_OPT <- 5

fail <- character(0)
note <- function(...) cat(sprintf(...), "\n")
gate <- function(ok, msg) if (!isTRUE(ok)) fail <<- c(fail, msg)
as_bool <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "1", "yes", "t")
}

csv <- file.path(PROC, "responses.csv")
if (!file.exists(csv)) { cat("FATAL: responses.csv not found — run 01_ingest.py first.\n"); quit(status = 2) }
d <- read.csv(csv, stringsAsFactors = FALSE, na.strings = c("", "NA"))
note("Loaded %d rows, %d columns from responses.csv", nrow(d), ncol(d))

d$option_is_true <- as_bool(d$option_is_true)
d$endorsed       <- as_bool(d$endorsed)
d$attn_pass_all  <- as_bool(d$attn_pass_all)

# ── Gate: required columns ──
req <- c("respondent_id","agent","system","model_name","sample_idx","group_id",
         "position","modality","topic","level","question_id","option_id",
         "option_is_true","option_type","cue_match","endorsed","attn_pass_all")
gate(all(req %in% names(d)), paste("missing columns:", paste(setdiff(req, names(d)), collapse=", ")))

# ── Gate: vocab ──
gate(all(d$topic %in% TOPICS), "unexpected topic value(s)")
gate(all(d$level %in% LEVELS), "unexpected level value(s)")
gate(all(d$agent %in% c("human","model")), "unexpected agent value(s)")

# ── Gate: no missing endorsed ──
gate(!any(is.na(d$endorsed)), "missing endorsed values present")

# ── Gate: option_type fully labeled + in allowed set ──
gate(!any(is.na(d$option_type)), "option_type has NA (run tagging + re-ingest)")
gate(all(d$option_type %in% c("true","false_interference","false_plain")),
     "option_type has out-of-set values")
gate(all(d$cue_match[!is.na(d$cue_match)] %in% c("exact","paraphrase")),
     "cue_match has out-of-set values")

# ── Gate: every (topic, question_id, option_id) appears in all 4 levels, identical truth ──
key_lvls <- tapply(d$level, list(paste(d$topic, d$question_id, d$option_id)),
                   function(x) length(unique(x)))
gate(all(key_lvls == length(LEVELS)),
     sprintf("%d (topic,question,option) keys not present in all 4 levels",
             sum(key_lvls != length(LEVELS))))
truth_consistency <- tapply(d$option_is_true,
                            list(paste(d$topic, d$question_id, d$option_id)),
                            function(x) length(unique(x)))
gate(all(truth_consistency == 1),
     sprintf("%d option keys have inconsistent option_is_true across levels",
             sum(truth_consistency != 1)))

# ── Gate: content-question / option structure ──
q_per_topic <- tapply(d$question_id, d$topic, function(x) length(unique(x)))
gate(all(q_per_topic == N_CONTENT_Q),
     paste("topic(s) without exactly", N_CONTENT_Q, "content questions"))
opt_per_q <- tapply(d$option_id, paste(d$topic, d$question_id), function(x) length(unique(x)))
gate(all(opt_per_q == N_OPT), paste("question(s) without exactly", N_OPT, "options"))

# ── Humans ──
h <- d[d$agent == "human", ]
# one respondent = 4 distinct topics & 4 distinct levels
h_topics <- tapply(h$topic, h$respondent_id, function(x) length(unique(x)))
h_levels <- tapply(h$level, h$respondent_id, function(x) length(unique(x)))
gate(all(h_topics == 4), sprintf("%d humans not seeing exactly 4 topics", sum(h_topics != 4)))
gate(all(h_levels == 4), sprintf("%d humans not seeing exactly 4 levels", sum(h_levels != 4)))

# whole-participant attention exclusion is applied at ingest → every human row must be a pass
gate(all(h$attn_pass_all), "human rows present that did not pass ALL attention checks (ingest bug)")
ha <- h[h$attn_pass_all, ]
# ≥25 humans per (topic, level) cell
hcell <- aggregate(respondent_id ~ topic + level, data = ha,
                   FUN = function(x) length(unique(x)))
names(hcell)[3] <- "n_humans_attnpass"
below <- hcell[hcell$n_humans_attnpass < MIN_HUMANS_PER_CELL, ]
gate(nrow(below) == 0,
     sprintf("%d (topic,level) cells below %d humans", nrow(below), MIN_HUMANS_PER_CELL))

# (topic, level, position) non-empty for humans with known position
hp <- ha[!is.na(ha$position), ]
posgrid <- expand.grid(topic = TOPICS, level = LEVELS, position = 1:4, stringsAsFactors = FALSE)
present <- unique(hp[, c("topic","level","position")])
missing_pos <- nrow(posgrid) - nrow(merge(posgrid, present))
gate(missing_pos == 0, sprintf("%d (topic,level,position) human cells empty", missing_pos))

# ── Models: exactly N samples per (model_name, topic, level, question_id) ──
m <- d[d$agent == "model", ]
# one "sample" = one option-row set per (respondent_id); count distinct sample_idx per cell/question
msamp <- aggregate(sample_idx ~ model_name + topic + level + question_id, data = m,
                   FUN = function(x) length(unique(x)))
names(msamp)[5] <- "n_samples"
samp_by_model <- tapply(msamp$n_samples, msamp$model_name, function(x) paste(sort(unique(x)), collapse=","))
note("Model samples/cell (distinct counts per model):")
for (nm in names(samp_by_model)) note("  %-28s %s", nm, samp_by_model[[nm]])
# hard gate: every model has a single, uniform sample count across all its cells
uniform <- tapply(msamp$n_samples, msamp$model_name, function(x) length(unique(x)) == 1)
gate(all(uniform), sprintf("%d model(s) have non-uniform samples/cell", sum(!uniform)))

# ── cell_counts.csv (human review artifact) ──
hcnt <- aggregate(respondent_id ~ topic + level, data = h,
                  FUN = function(x) length(unique(x)))
names(hcnt)[3] <- "n_humans_all"
hcnt <- merge(hcnt, hcell, all.x = TRUE)
hcnt$agent <- "human"; hcnt$model_name <- NA; hcnt$system <- "human"
hcnt$n_samples <- hcnt$n_humans_all

mcnt <- aggregate(sample_idx ~ model_name + system + topic + level, data = m,
                 FUN = function(x) length(unique(x)))
names(mcnt)[5] <- "n_samples"
mcnt$agent <- "model"; mcnt$n_humans_all <- NA; mcnt$n_humans_attnpass <- NA

common <- c("agent","system","model_name","topic","level","n_samples",
            "n_humans_all","n_humans_attnpass")
cell_counts <- rbind(hcnt[, common], mcnt[, common])
cell_counts <- cell_counts[order(cell_counts$agent, cell_counts$model_name,
                                 cell_counts$topic, cell_counts$level), ]
write.csv(cell_counts, file.path(TABLES, "cell_counts.csv"), row.names = FALSE)
note("Wrote %s (%d rows)", file.path("outputs","tables","cell_counts.csv"), nrow(cell_counts))

# human cell summary to console
note("\nHuman respondents per (topic, level) [all / attn-pass]:")
wide <- reshape(hcnt[, c("topic","level","n_humans_all")], idvar="topic",
                timevar="level", direction="wide")
print(wide, row.names = FALSE)

# ── verdict ──
cat("\n===============================\n")
if (length(fail) == 0) {
  cat("VALIDATION PASSED — all gates green.\n")
  cat("NEXT: a human must review outputs/tables/cell_counts.csv before P2.\n")
  quit(status = 0)
} else {
  cat("VALIDATION FAILED:\n")
  for (f in fail) cat("  -", f, "\n")
  quit(status = 1)
}
