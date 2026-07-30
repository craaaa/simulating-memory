#!/usr/bin/env Rscript
# 08_figures — §8. Six figures, all per-model, all with CIs. ggplot2 -> PNG (light surface).
# Palette (dataviz skill, CVD-validated as a set): human=blue, prompting=green, compactor=magenta.
# Each figure is guarded on its source table existing; missing inputs are skipped with a note.
#
# Usage: Rscript R/08_figures.R
# Writes: outputs/figures/fig1..fig6 *.png

args_all <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
if (length(here) == 0 || here == "") here <- "analysis/R"
ANALYSIS <- normalizePath(file.path(here, ".."))
source(file.path(ANALYSIS, "R", "lib_alignment.R"))
suppressWarnings(suppressMessages(library(ggplot2)))
cfg <- file.path(ANALYSIS, "config", "analysis.yaml")
DELTA <- as.numeric(read_cfg_scalar(cfg, "bound_logodds", 0.2213))
TAB <- file.path(ANALYSIS, "outputs", "tables")
FIG <- file.path(ANALYSIS, "outputs", "figures"); dir.create(FIG, showWarnings = FALSE, recursive = TRUE)
LV <- c("control", "repeat_short", "repeat_long", "distractor")

SYS_COL <- c(human = "#2a78d6", prompting = "#008300", compactor = "#e87ba4")
OT_COL  <- c(false_interference = "#eb6834", false_plain = "#4a3aa7", `true` = "#2a78d6")
SURF <- "#fcfcfb"
theme_viz <- function() theme_minimal(base_size = 11) + theme(
  plot.background = element_rect(fill = SURF, color = NA),
  panel.background = element_rect(fill = SURF, color = NA),
  panel.grid.minor = element_blank(),
  panel.grid.major = element_line(color = "#e6e5e2", linewidth = 0.3),
  text = element_text(color = "#0b0b0b"),
  axis.text = element_text(color = "#52514e"),
  strip.text = element_text(color = "#0b0b0b", face = "bold"),
  legend.position = "top")
rd <- function(f) if (file.exists(file.path(TAB, f))) read.csv(file.path(TAB, f), stringsAsFactors = FALSE) else NULL
lvf <- function(x) factor(x, levels = LV)
save_png <- function(p, name, w = 9, h = 6) {
  ggsave(file.path(FIG, name), p, width = w, height = h, dpi = 150, bg = SURF)
  cat("wrote", name, "\n")
}
made <- c()

# ── Fig 1: predicted endorsement by level × option_type, per system, per model ──
p1 <- rd("glmm_predictions.csv")
if (!is.null(p1)) {
  p1$level <- lvf(p1$level)
  p1$system <- factor(p1$system, levels = names(SYS_COL))
  # separated strata: CI spans ~the whole scale (non-identified, e.g. prompting at ceiling/floor).
  # Draw error bars only where estimable; keep the point elsewhere.
  p1$estimable <- (p1$asymp.UCL - p1$asymp.LCL) <= 0.9
  g <- ggplot(p1, aes(level, pred, color = system, group = system)) +
    geom_line(linewidth = 0.6) + geom_point(size = 1.8) +
    geom_errorbar(data = p1[p1$estimable, ],
                  aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.15, linewidth = 0.4) +
    facet_grid(model ~ option_type) +
    scale_color_manual(values = SYS_COL) +
    labs(title = "Predicted endorsement probability by level × option type",
         subtitle = "human vs prompting vs compactor (GLMM marginal means, 95% CI; bars omitted where separated/non-identified)",
         x = NULL, y = "P(endorse)", color = NULL) +
    theme_viz() + theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_png(g, "fig1_predicted_endorsement.png", 9, 2 + 1.6 * length(unique(p1$model))); made <- c(made, 1)
} else cat("skip fig1 (glmm_predictions.csv missing — run 04)\n")

# ── Fig 2: compactor−human equivalence forest, by option_type ──
p2 <- rd("glmm_contrasts.csv")
if (!is.null(p2)) {
  XL <- 4; EDGE <- 3.9
  all2 <- p2[grepl("compactor", p2$contrast) & p2$level != "control", ]
  sepflag <- if ("separated" %in% names(all2)) as.logical(all2$separated %in% c(TRUE,"TRUE","True")) else rep(FALSE, nrow(all2))
  est <- all2[!sepflag, ]; sepd <- all2[sepflag, ]      # estimable vs separated (floored)
  est$level <- lvf(est$level); sepd$level <- lvf(sepd$level)
  est$verdict <- ifelse(est$upper < DELTA & est$lower > -DELTA, "equivalent",
                 ifelse(est$lower > DELTA | est$upper < -DELTA, "different", "inconclusive"))
  est$xc <- pmax(pmin(est$estimate, EDGE), -EDGE)        # clamp point into view
  est$off <- abs(est$estimate) > EDGE                    # estimate beyond ±4 (e.g. kimi −4.95)
  # separated cells are floored (compactor ≈ 0 endorsement) → mark at the left edge
  sepd$xc <- -EDGE
  vcol <- c(equivalent = "#008300", different = "#e34948", inconclusive = "#52514e")
  g <- ggplot(est, aes(xc, level, color = verdict)) +
    annotate("rect", xmin = -DELTA, xmax = DELTA, ymin = -Inf, ymax = Inf, fill = "#9aa0a6", alpha = 0.20) +
    geom_vline(xintercept = 0, color = "#9aa0a6", linewidth = 0.3) +
    geom_errorbarh(aes(xmin = pmax(lower, -XL), xmax = pmin(upper, XL)), height = 0.2, linewidth = 0.5) +
    geom_point(aes(shape = off), size = 2) +
    # separated / floored strata: hollow grey marker at the left edge (absence made visible)
    { if (nrow(sepd)) geom_point(data = sepd, aes(xc, level), inherit.aes = FALSE,
                                 shape = 4, color = "#9aa0a6", size = 2.2, stroke = 0.9) } +
    facet_grid(model ~ option_type) +
    scale_color_manual(values = vcol, name = NULL) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 17), guide = "none") +
    coord_cartesian(xlim = c(-XL, XL)) +
    labs(title = "Compactor − human contrast (log-odds), by level × option type",
         subtitle = sprintf("shaded = ±%.2f equivalence band (Δ); ▲ estimate beyond ±%g; grey ✕ = floored (compactor ≈0, non-identified). None equivalent.", DELTA, XL),
         x = "compactor − human (log-odds; CIs clipped to ±4)", y = NULL) + theme_viz()
  save_png(g, "fig2_contrast_forest.png", 9, 2 + 1.5 * length(unique(all2$model))); made <- c(made, 2)
} else cat("skip fig2 (glmm_contrasts.csv missing — run 04)\n")

# ── Fig 3: error consistency kappa by level, per model, with human ceiling band ──
p3 <- rd("error_kappa.csv")
if (!is.null(p3)) {
  p3$level <- lvf(p3$level)
  p3$system <- factor(p3$system, levels = c("prompting", "compactor"))
  # human–human ceiling is model-independent and ~flat across levels → one shaded band
  cy <- c(mean(p3$ceil_lo, na.rm = TRUE), mean(p3$ceil_hi, na.rm = TRUE))
  dodge <- position_dodge(0.35)
  g <- ggplot(p3, aes(level, kappa_model_human, color = system, group = system)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = cy[1], ymax = cy[2],
             fill = "#9aa0a6", alpha = 0.25) +
    geom_line(position = dodge, linewidth = 0.6) +
    geom_errorbar(aes(ymin = mh_lo, ymax = mh_hi), width = 0.15, linewidth = 0.4, position = dodge) +
    geom_point(size = 1.8, position = dodge) +
    facet_wrap(~ model) +
    scale_color_manual(values = c(prompting = "#008300", compactor = "#e87ba4")) +
    labs(title = "Error consistency (Cohen's κ) vs humans, by level",
         subtitle = "points = model↔human κ (95% bootstrap CI); grey band = human–human noise ceiling (95% CI)",
         x = NULL, y = "κ (model ↔ human)", color = NULL) +
    theme_viz() + theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_png(g, "fig3_error_kappa.png", 10, 6); made <- c(made, 3)
} else cat("skip fig3 (error_kappa.csv missing — run 05)\n")

# ── Fig 4: false-option endorsement scatter, faceted by level, coloured by option_type ──
p4 <- rd("error_profile_points.csv")
if (!is.null(p4)) {
  p4 <- p4[p4$system == "compactor", ]         # compactor is the human-like hypothesis
  p4$level <- lvf(p4$level)
  g <- ggplot(p4, aes(human_rate, model_rate, color = option_type)) +
    geom_abline(slope = 1, intercept = 0, color = "#9aa0a6", linewidth = 0.3) +
    geom_point(size = 1.3, alpha = 0.8) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.5, formula = y ~ x) +
    facet_grid(model ~ level) +
    scale_color_manual(values = OT_COL) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "False-option endorsement: human vs compactor",
         subtitle = "each point a false option; identity line grey; coloured by option type",
         x = "human endorsement rate", y = "compactor endorsement rate", color = NULL) +
    theme_viz()
  save_png(g, "fig4_false_option_scatter.png", 10, 2 + 1.6 * length(unique(p4$model))); made <- c(made, 4)
} else cat("skip fig4 (error_profile_points.csv missing — run 05)\n")

# ── Fig 5: option facility scatter with ceiling-normalised r annotated ──
p5 <- rd("difficulty_points.csv"); c5 <- rd("difficulty_corr.csv")
if (!is.null(p5)) {
  p5 <- p5[p5$system == "compactor", ]; p5$level <- lvf(p5$level)
  ann <- NULL
  if (!is.null(c5)) { c5 <- c5[c5$system == "compactor", ]
    c5$level <- lvf(c5$level)
    ann <- data.frame(model = c5$model, level = c5$level,
                      lab = sprintf("r=%.2f\nr/ceil=%.2f", c5$spearman, c5$ratio)) }
  g <- ggplot(p5, aes(human_rate, model_rate)) +
    geom_abline(slope = 1, intercept = 0, color = "#9aa0a6", linewidth = 0.3) +
    geom_point(aes(color = as.logical(option_is_true)), size = 1.3, alpha = 0.8) +
    facet_grid(model ~ level) +
    scale_color_manual(values = c(`TRUE` = "#2a78d6", `FALSE` = "#eb6834"),
                       labels = c(`TRUE` = "true", `FALSE` = "false"), name = "option") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Option facility: human vs compactor endorsement rate",
         subtitle = "Spearman r and ceiling-normalised ratio annotated (see caption caveat: split-half bounds R², not r)",
         x = "human endorsement rate", y = "compactor endorsement rate") + theme_viz()
  if (!is.null(ann)) g <- g + geom_text(data = ann, aes(x = 0.03, y = 0.95, label = lab),
                                        hjust = 0, vjust = 1, size = 2.6, color = "#0b0b0b")
  save_png(g, "fig5_option_facility.png", 10, 2 + 1.6 * length(unique(p5$model))); made <- c(made, 5)
} else cat("skip fig5 (difficulty_points.csv missing — run 06)\n")

# ── Fig 6: selection-count distribution per system ──
p6 <- rd("selection_counts_raw.csv")
if (!is.null(p6)) {
  p6$endorsed <- pmin(p6$endorsed, 4)  # bucket 4+
  p6$system <- factor(p6$sys_label, levels = names(SYS_COL))
  tab <- as.data.frame(prop.table(table(p6$system, p6$endorsed), 1))
  names(tab) <- c("system", "n_selected", "frac")
  g <- ggplot(tab, aes(n_selected, frac, fill = system)) +
    geom_col(position = position_dodge(0.8), width = 0.75) +
    scale_fill_manual(values = SYS_COL) +
    labs(title = "Options endorsed per question", subtitle = "distribution by system (4 = 4+)",
         x = "# options selected", y = "fraction of question-responses", fill = NULL) + theme_viz()
  save_png(g, "fig6_selection_counts.png", 8, 5); made <- c(made, 6)
} else cat("skip fig6 (selection_counts_raw.csv missing — run 07)\n")

cat(sprintf("\nFigures produced: %s\n", if (length(made)) paste0("fig", made, collapse = ", ") else "none"))
