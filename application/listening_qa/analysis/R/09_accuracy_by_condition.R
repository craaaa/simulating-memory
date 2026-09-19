#!/usr/bin/env Rscript
# 09_accuracy_by_condition — per-entity exact-match accuracy by condition, styled after
# multi_v6/v6_accuracy_by_condition.png (4-colour condition palette, chance line, value
# labels, CIs). One panel per entity: human + each model's prompting and compactor stream.
#
# Exact match = a question is correct iff EVERY option matches ground truth (endorsed==truth).
# Chance for a 5-option multi-select = (1/2)^5 = 1/32.
#
# Usage: Rscript R/09_accuracy_by_condition.R
# Writes: outputs/figures/fig7_accuracy_by_condition.png , outputs/tables/accuracy_by_condition.csv

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
suppressWarnings(suppressMessages(library(ggplot2)))
cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
NBOOT <- as.integer(read_cfg_scalar(cfg, "n_bootstrap", 2000))
SEED  <- as.integer(read_cfg_scalar(cfg, "bootstrap", 11081))
LV <- c("control", "repeat_short", "repeat_long", "distractor")
LV_LABELS <- c(control = "Control", repeat_short = "Repeat Once",
               repeat_long = "Repeat Twice", distractor = "Distractor")
COND_COL <- c(Control = "#8a97b3", `Repeat Once` = "#f2a24e",
              `Repeat Twice` = "#e86f10", Distractor = "#a396e0")
CHANCE <- 1/32
SURF <- "#fcfcfb"

d <- read.csv(file.path(ANALYSIS, "data", "processed", "responses.csv"), stringsAsFactors = FALSE)
d$endorsed <- as.integer(d$endorsed %in% c(TRUE, "True", "true", 1, "1"))
d$option_is_true <- d$option_is_true %in% c(TRUE, "True", "true", 1, "1")
d$ok <- as.integer(d$endorsed == as.integer(d$option_is_true))

# entity = who: human, or a model stream. Nice label.
d$entity <- ifelse(d$agent == "human", "Human", d$model_name)
label_entity <- function(e) {
  e <- sub("__wm$", " (WM)", e)
  ifelse(e == "Human", "Human", ifelse(grepl(" \\(WM\\)$", e), e, paste0(e, " (prompt)")))
}

# exact-match per (entity, respondent, topic, level, question): all options in the question correct
qok <- aggregate(ok ~ entity + respondent_id + topic + level + question_id, data = d,
                 FUN = function(x) as.integer(all(x == 1)))
# per-respondent accuracy within (entity, level) = mean over that respondent's questions
racc <- aggregate(ok ~ entity + respondent_id + level, data = qok, FUN = mean)

# styled after open_source_compactor_by_level_bar.png: grouped bars, Human + each model's
# COMPACTOR (WM) stream, conditions as the 4-colour groups, Human split off by a dotted line.
racc <- racc[racc$entity == "Human" | grepl("__wm$", racc$entity), ]
set.seed(SEED)
rows <- list()
for (e in unique(racc$entity)) for (lv in LV) {
  v <- racc$ok[racc$entity == e & racc$level == lv]; if (!length(v)) next
  boot <- replicate(NBOOT, mean(sample(v, length(v), replace = TRUE)))
  ci <- quantile(boot, c(.025, .975), na.rm = TRUE)
  rows[[length(rows)+1]] <- data.frame(entity = e, level = lv, acc = mean(v),
    lo = ci[1], hi = ci[2], n = length(v), stringsAsFactors = FALSE)
}
acc <- do.call(rbind, rows)
TABLES <- file.path(ANALYSIS, "outputs", "tables"); dir.create(TABLES, showWarnings = FALSE, recursive = TRUE)
write.csv(acc, file.path(TABLES, "accuracy_by_condition.csv"), row.names = FALSE)

disp <- function(e) ifelse(e == "Human", "Human", sub("__wm$", "", sub("cohere_", "", e)))
acc$xlab <- disp(acc$entity)
# order: Human first, then models by descending control accuracy (readable)
ctrl <- acc[acc$level == "control", ]; morder <- ctrl$xlab[ctrl$xlab != "Human"][order(-ctrl$acc[ctrl$xlab != "Human"])]
acc$xlab <- factor(acc$xlab, levels = c("Human", morder))
acc$cond <- factor(LV_LABELS[acc$level], levels = LV_LABELS[LV])
dodge <- position_dodge(0.8)

g <- ggplot(acc, aes(xlab, acc, fill = cond)) +
  geom_hline(yintercept = CHANCE, linetype = "dashed", color = "#9aa0a6", linewidth = 0.3) +
  geom_vline(xintercept = 1.5, linetype = "dotted", color = "#52514e", linewidth = 0.4) +
  geom_col(position = dodge, width = 0.75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = 0.2, linewidth = 0.3, color = "#333333") +
  scale_fill_manual(values = COND_COL, name = NULL) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2)) +
  labs(title = "Compactor exact-match accuracy by condition (vs. human)",
       subtitle = "proportion of questions fully correct; dashed = chance (1/32); error bars = 95% bootstrap CI",
       x = NULL, y = "Exact-match accuracy") +
  theme_minimal(base_size = 11) +
  theme(plot.background = element_rect(fill = SURF, color = NA),
        panel.background = element_rect(fill = SURF, color = NA),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "#e6e5e2", linewidth = 0.3),
        axis.text.x = element_text(angle = 25, hjust = 1),
        legend.position = "top", text = element_text(color = "#0b0b0b"))
ggsave(file.path(ANALYSIS, "outputs", "figures", "fig7_compactor_accuracy_by_condition.png"),
       g, width = 13, height = 5.5, dpi = 150, bg = SURF)
cat(sprintf("wrote fig7_compactor_accuracy_by_condition.png (%d entities)\n", length(unique(acc$xlab))))
