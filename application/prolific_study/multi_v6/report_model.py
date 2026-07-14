#!/usr/bin/env python3
"""Reporting for the maximal Bayesian model: contrasts, forest plot, LOO
comparison against simpler nested variants, and a sensitivity refit on the
strict (whole-participant) attention-check exclusion rule.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bayesian_model import load_data, build_sanity_model, build_maximal_model, RESULTS_DIR, HERE

CONDS_NONREF = ['repeat_short', 'repeat_long', 'distractor']


def logit_to_prob(x):
    return 1 / (1 + np.exp(-x))


def report_primary():
    idata = az.from_netcdf(RESULTS_DIR / 'maximal_model.nc')

    print('=== Fixed-effect posteriors (logit scale, 94% CI) ===')
    summ = az.summary(idata, var_names=['Intercept', 'condition', 'serial_position'], ci_prob=0.95)
    print(summ)

    post = idata.posterior
    intercept = post['Intercept'].values.flatten()

    print('\n=== Condition contrasts vs. control (probability scale) ===')
    print('  control baseline P(correct) [at serial_position=1, average pid/topic/item]: '
          f'{logit_to_prob(intercept).mean():.3f}')
    cond_draws = {}
    for i, name in enumerate(CONDS_NONREF):
        draws = post['condition'].sel(condition_dim=name).values.flatten()
        cond_draws[name] = draws
        p_control = logit_to_prob(intercept)
        p_cond = logit_to_prob(intercept + draws)
        diff = p_cond - p_control
        print(f'  {name}: logit coef mean={draws.mean():+.3f} sd={draws.std():.3f} | '
              f'P(correct) diff vs control = {diff.mean():+.3f} '
              f'[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}] | '
              f'P(direction) = {(draws > 0).mean() if draws.mean() > 0 else (draws < 0).mean():.3f}')

    print('\n=== Pairwise contrasts among non-reference conditions ===')
    pairs = [('distractor', 'repeat_long'), ('distractor', 'repeat_short'), ('repeat_long', 'repeat_short')]
    for a, b in pairs:
        d = cond_draws[a] - cond_draws[b]
        print(f'  {a} - {b}: mean={d.mean():+.3f} sd={d.std():.3f} '
              f'P(direction)={(d > 0).mean() if d.mean() > 0 else (d < 0).mean():.3f}')

    print('\n=== Serial position contrasts vs. position 1 (probability scale) ===')
    for pos in ['2', '3', '4']:
        draws = post['serial_position'].sel(serial_position_dim=pos).values.flatten()
        p_1 = logit_to_prob(intercept)
        p_pos = logit_to_prob(intercept + draws)
        diff = p_pos - p_1
        print(f'  position {pos} vs 1: logit coef mean={draws.mean():+.3f} | '
              f'P(correct) diff = {diff.mean():+.3f} '
              f'[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]')

    # Forest plot
    pc = az.plot_forest(idata, var_names=['condition', 'serial_position'], combined=True,
                         ci_probs=[0.5, 0.95])
    pc.add_title('Fixed effects (logit scale)\ncondition ref=control, serial_position ref=1st')
    pc.savefig(RESULTS_DIR / 'forest_fixed_effects.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved: {RESULTS_DIR / "forest_fixed_effects.png"}')

    return idata


def plot_caterpillar(idata):
    pc = az.plot_trace(idata, var_names=['Intercept', 'condition', 'serial_position'])
    pc.savefig(RESULTS_DIR / 'caterpillar_fixed_effects.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {RESULTS_DIR / "caterpillar_fixed_effects.png"}')
    plt.close('all')


def loo_comparison():
    print('\n=== LOO comparison: maximal vs. simpler nested variants ===')
    df = load_data(HERE / 'longdata_strict.csv')

    idata_maximal = az.from_netcdf(RESULTS_DIR / 'maximal_model.nc')

    # Simpler variant: drop the (1+condition|topic) slope down to (1|topic).
    # Use the same target_accept/tune that got the maximal model clean (0.99/2000) —
    # the looser 0.95/1500 settings left this variant with divergences too.
    import bambi as bmb
    formula_simple = ('correct ~ condition + serial_position '
                       '+ (1 + condition | pid) + (1 | topic) + (1 | question_id)')
    model_simple = bmb.Model(formula_simple, data=df, family='bernoulli', noncentered=True)
    model_simple.build()
    idata_simple = model_simple.fit(draws=1000, tune=2000, chains=4, target_accept=0.99,
                                     random_seed=0, idata_kwargs={'log_likelihood': True})
    n_div_simple = int(idata_simple.sample_stats.diverging.sum())
    print(f'  simple model: {n_div_simple} divergences, max r_hat = '
          f'{az.summary(idata_simple)["r_hat"].max()}')
    idata_simple.to_netcdf(RESULTS_DIR / 'simple_topic_intercept_model.nc')

    # need log_likelihood on the maximal model too for a fair comparison;
    # refit isn't necessary if it was computed already, otherwise recompute
    if 'log_likelihood' not in idata_maximal.groups:
        print('  (maximal_model.nc has no log_likelihood group — refitting with it enabled)')
        model_maximal, _ = build_maximal_model(df)
        model_maximal.build()
        idata_maximal = model_maximal.fit(draws=1000, tune=2000, chains=4, target_accept=0.99,
                                           random_seed=0, idata_kwargs={'log_likelihood': True})
        idata_maximal.to_netcdf(RESULTS_DIR / 'maximal_model_with_loglik.nc')

    comp = az.compare({'maximal (topic slope)': idata_maximal, 'simple (topic intercept only)': idata_simple})
    print(comp)
    comp.to_csv(RESULTS_DIR / 'loo_comparison.csv')
    print(f'Saved: {RESULTS_DIR / "loo_comparison.csv"}')
    return comp


def sensitivity_strict():
    print('\n=== Sensitivity check: refit maximal model on longdata_strict.csv ===')
    df_strict = load_data(HERE / 'longdata_strict.csv')
    import bambi as bmb
    model, formula = build_maximal_model(df_strict)
    model.build()
    idata = model.fit(draws=1000, tune=2000, chains=4, target_accept=0.99, random_seed=0)
    idata.to_netcdf(RESULTS_DIR / 'maximal_model_strict.nc')
    print(f'Saved: {RESULTS_DIR / "maximal_model_strict.nc"}')

    print('n divergent:', int(idata.sample_stats.diverging.sum()))
    s = az.summary(idata, var_names=['Intercept', 'condition', 'serial_position'], ci_prob=0.95)
    print(s[['mean', 'sd', 'r_hat', 'ess_bulk']])
    return idata


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-loo', action='store_true')
    ap.add_argument('--skip-sensitivity', action='store_true')
    args = ap.parse_args()

    idata = report_primary()
    plot_caterpillar(idata)
    if not args.skip_loo:
        loo_comparison()
    if not args.skip_sensitivity:
        sensitivity_strict()
