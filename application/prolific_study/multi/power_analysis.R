library(brms)
library(tidyverse)
library(tidybayes)

set.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────
d <- read_csv("pilot_long.csv") |>
  mutate(
    condition = factor(condition,
                       levels = c("control", "repeat_short", "repeat_long", "distractor")),
    topic     = factor(topic),
    participant = factor(participant)
  )

# Condition contrasts matching directional predictions:
#   C1: repeat_long > distractor
#   C2: repeat_short > control
#   C3: {control, repeat_short} > {distractor, repeat_long}
contrasts(d$condition) <- cbind(
  repL_vs_dist  = c( 0,  0,  1, -1),   # repeat_long - distractor
  repS_vs_ctrl  = c(-1,  1,  0,  0),   # repeat_short - control
  rep_vs_noRep  = c(-1,  1,  1, -1)    # (repS + repL) - (ctrl + dist), unscaled
)

# ── Pilot model ───────────────────────────────────────────────────────────────
# Outcome: proportion correct (exact match), 5 questions per topic
# Using binomial family (n_correct / n_questions) with crossed random effects.
# Beta-binomial would be more appropriate for overdispersion, but start simple.

fit_pilot <- brm(
  n_correct | trials(n_questions) ~ condition + (1 | participant) + (1 | topic),
  data   = d,
  family = binomial(link = "logit"),
  prior  = c(
    prior(normal(0, 1.5), class = Intercept),
    prior(normal(0, 0.5), class = b),           # modest shrinkage on condition effects
    prior(exponential(1), class = sd)
  ),
  chains = 4, iter = 4000, warmup = 1000,
  cores  = 4, seed  = 42,
  file   = "fit_pilot"                          # cache to disk
)

summary(fit_pilot)
plot(fit_pilot, variable = "^b_", regex = TRUE)

# Posterior contrasts
posterior_samples <- as_draws_df(fit_pilot)

contrasts_post <- posterior_samples |>
  transmute(
    repL_vs_dist = b_conditionrepL_vs_dist,
    repS_vs_ctrl = b_conditionrepS_vs_ctrl,
    rep_vs_noRep = b_conditionrep_vs_noRep / 2  # scale to per-condition unit
  )

# Posterior probability each contrast > 0 (directional)
cat("\nPosterior P(contrast > 0):\n")
print(summarise(contrasts_post,
  across(everything(), ~ mean(. > 0), .names = "P>{.col}")
))

# 95% credible intervals
cat("\n95% CrI:\n")
print(summarise(contrasts_post,
  across(everything(), list(
    median = median,
    lo95   = ~ quantile(., 0.025),
    hi95   = ~ quantile(., 0.975)
  ))
))

# ── Simulation-based power analysis ──────────────────────────────────────────
# Strategy:
#   1. Extract posterior means for all parameters from pilot fit
#   2. Simulate datasets at varying N (participants)
#   3. For each dataset, check if 95% CrI of each contrast excludes 0 in
#      the predicted direction (i.e. lower bound > 0)
#   4. Power = proportion of simulations where this holds

simulate_dataset <- function(n_participants, params, topics, n_q = 5) {
  # params: list with intercept, b_repL_dist, b_repS_ctrl, b_rep_noRep,
  #         sd_participant, sd_topic
  # Latin square: assign conditions to topics cycling through groups
  # Each participant gets all 4 conditions, one per topic
  groups <- rep(1:24, length.out = n_participants)
  latin  <- list(
    `1`  = c(birds="control",   fruits="repeat_short", astronomy="repeat_long",  musical_instruments="distractor"),
    `8`  = c(birds="repeat_short", fruits="control",   astronomy="distractor",   musical_instruments="repeat_long"),
    `17` = c(birds="repeat_long",  fruits="distractor",astronomy="control",      musical_instruments="repeat_short"),
    `24` = c(birds="distractor",   fruits="repeat_long",astronomy="repeat_short",musical_instruments="control")
  )
  # Simplify: just cycle condition assignments
  cond_matrix <- matrix(
    c("control","repeat_short","repeat_long","distractor",
      "repeat_short","control","distractor","repeat_long",
      "repeat_long","distractor","control","repeat_short",
      "distractor","repeat_long","repeat_short","control"),
    nrow = 4, byrow = TRUE,
    dimnames = list(NULL, topics)
  )

  participant_rfx <- rnorm(n_participants, 0, params$sd_participant)
  topic_rfx       <- rnorm(length(topics),  0, params$sd_topic)
  names(topic_rfx) <- topics

  rows <- list()
  for (i in seq_len(n_participants)) {
    g    <- ((i - 1) %% 4) + 1
    for (ti in seq_along(topics)) {
      t    <- topics[ti]
      cond <- cond_matrix[g, t]
      b_cond <- switch(cond,
        control      =  0,
        repeat_short =  params$b_repS_ctrl,
        repeat_long  =  params$b_repL_dist,
        distractor   = -params$b_repL_dist
      )
      eta   <- params$intercept + b_cond + participant_rfx[i] + topic_rfx[ti]
      prob  <- plogis(eta)
      n_cor <- rbinom(1, n_q, prob)
      rows[[length(rows)+1]] <- list(
        participant = paste0("P", i), topic = t,
        condition = cond, n_correct = n_cor, n_questions = n_q
      )
    }
  }
  bind_rows(rows) |>
    mutate(
      condition   = factor(condition, levels = c("control","repeat_short","repeat_long","distractor")),
      participant = factor(participant),
      topic       = factor(topic)
    ) |>
    (\(df) { contrasts(df$condition) <- cbind(
        repL_vs_dist = c(0,0,1,-1),
        repS_vs_ctrl = c(-1,1,0,0),
        rep_vs_noRep = c(-1,1,1,-1)); df })()
}

# Extract pilot posterior means
pilot_params <- list(
  intercept        = mean(posterior_samples$b_Intercept),
  b_repL_dist      = mean(posterior_samples$b_conditionrepL_vs_dist),
  b_repS_ctrl      = mean(posterior_samples$b_conditionrepS_vs_ctrl),
  b_rep_noRep      = mean(posterior_samples$b_conditionrep_vs_noRep),
  sd_participant   = mean(posterior_samples$sd_participant__Intercept),
  sd_topic         = mean(posterior_samples$sd_topic__Intercept)
)
cat("\nPilot parameter estimates:\n")
print(pilot_params)

# Power simulation
N_SIMS     <- 200          # simulations per N (increase to 500+ for final)
N_SIZES    <- c(40, 60, 80, 100, 120, 150, 200)
TOPICS_VEC <- c("birds", "fruits", "astronomy", "musical_instruments")

# Quick model template (reuse structure, refit data only)
power_results <- map_dfr(N_SIZES, function(n_p) {
  cat(sprintf("\nSimulating N=%d ...\n", n_p))
  successes <- c(repL_vs_dist=0, repS_vs_ctrl=0, rep_vs_noRep=0)
  for (sim in seq_len(N_SIMS)) {
    sim_d <- simulate_dataset(n_p, pilot_params, TOPICS_VEC)
    fit_s <- tryCatch(
      brm(
        n_correct | trials(n_questions) ~ condition + (1|participant) + (1|topic),
        data   = sim_d,
        family = binomial(link = "logit"),
        prior  = c(
          prior(normal(0, 1.5), class = Intercept),
          prior(normal(0, 0.5), class = b),
          prior(exponential(1), class = sd)
        ),
        chains = 2, iter = 2000, warmup = 500,
        cores  = 2, seed  = sim,
        refresh = 0, silent = 2
      ),
      error = function(e) NULL
    )
    if (is.null(fit_s)) next
    ps <- as_draws_df(fit_s)
    # "success" = 95% CrI lower bound > 0 for each contrast
    successes["repL_vs_dist"] <- successes["repL_vs_dist"] +
      (quantile(ps$b_conditionrepL_vs_dist, 0.05) > 0)
    successes["repS_vs_ctrl"] <- successes["repS_vs_ctrl"] +
      (quantile(ps$b_conditionrepS_vs_ctrl, 0.05) > 0)
    successes["rep_vs_noRep"] <- successes["rep_vs_noRep"] +
      (quantile(ps$b_conditionrep_vs_noRep, 0.05) > 0)
  }
  tibble(
    n_participants = n_p,
    power_repL_dist  = successes["repL_vs_dist"] / N_SIMS,
    power_repS_ctrl  = successes["repS_vs_ctrl"] / N_SIMS,
    power_rep_noRep  = successes["rep_vs_noRep"] / N_SIMS
  )
})

cat("\n\nPower results:\n")
print(power_results)
write_csv(power_results, "power_results.csv")

# Plot
power_long <- power_results |>
  pivot_longer(-n_participants, names_to="contrast", values_to="power") |>
  mutate(contrast = recode(contrast,
    power_repL_dist = "distractor < repeat_long",
    power_repS_ctrl = "control < repeat_short",
    power_rep_noRep = "{dist,repL} < {ctrl,repS}"
  ))

ggplot(power_long, aes(n_participants, power, color=contrast)) +
  geom_line() + geom_point() +
  geom_hline(yintercept=0.80, linetype="dashed") +
  geom_hline(yintercept=0.90, linetype="dotted") +
  scale_y_continuous(labels=scales::percent, limits=c(0,1)) +
  labs(x="N participants (attention-pass)", y="Power",
       title="Simulation-based power analysis",
       subtitle=sprintf("Based on pilot n=%d; %d sims per N; binomial GLMM", nrow(d)/4, N_SIMS),
       color="Contrast") +
  theme_bw()

ggsave("power_curve.png", width=8, height=5, dpi=150)
cat("Saved power_curve.png\n")
