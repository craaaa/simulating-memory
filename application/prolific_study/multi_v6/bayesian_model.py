#!/usr/bin/env python3
"""Bayesian mixed-effects (GLMM) model of multi_v6 item-level accuracy.

Stages (run with --stage):
  sanity   - fixed-effects-only model (correct ~ condition + serial_position),
             no random effects. Confirms the pipeline runs end-to-end quickly.
  maximal  - the full model justified by the design:
             correct ~ condition + serial_position
               + (1 + condition | pid) + (1 + condition | topic) + (1 | question_id)

See /Users/cl5625/.claude/plans/i-want-to-run-imperative-stroustrup.md for the
full rationale behind this model specification.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az

HERE = Path(__file__).parent
RESULTS_DIR = HERE / 'model_results'
RESULTS_DIR.mkdir(exist_ok=True)

COND_CATEGORIES = ['control', 'repeat_short', 'repeat_long', 'distractor']


def load_data(path=HERE / 'longdata.csv'):
    df = pd.read_csv(path)
    df['condition'] = pd.Categorical(df['condition'], categories=COND_CATEGORIES, ordered=False)
    df['serial_position'] = pd.Categorical(df['serial_position'].astype('Int64').astype(str),
                                            categories=['1', '2', '3', '4'], ordered=False)
    df['topic'] = df['topic'].astype('category')
    df['pid'] = df['pid'].astype('category')
    df['question_id'] = df['question_id'].astype('category')
    df = df.dropna(subset=['correct', 'condition', 'serial_position'])
    df['correct'] = df['correct'].astype(int)
    return df


def log_priors(model, label):
    """Explicitly print every prior the model will use, so it's clear whether
    any deviate from bambi's automatic defaults (per user request)."""
    print(f'\n--- Priors for {label} model ---')
    print(model)
    print('(all priors above are bambi auto-generated defaults, scaled to this '
          'data — none are hand-overridden in this script. If that changes, this '
          'function is where the override would be logged.)')


def build_sanity_model(df):
    formula = 'correct ~ condition + serial_position'
    model = bmb.Model(formula, data=df, family='bernoulli')
    return model, formula


def build_maximal_model(df):
    # condition|pid dropped: each participant sees each condition exactly once
    # (Latin-square design), so per-participant condition slopes aren't identifiable.
    formula = (
        'correct ~ condition + serial_position '
        '+ (1 | pid) + (1 + condition | topic) + (1 | question_id)'
    )
    model = bmb.Model(formula, data=df, family='bernoulli', noncentered=True)
    return model, formula


def prior_predictive_check(model, label, draws=500):
    idata = model.prior_predictive(draws=draws)
    ppc = idata.prior_predictive['correct'].values.flatten()
    p_correct = ppc.mean()
    print(f'\n--- Prior-predictive check ({label}) ---')
    print(f'  Prior-implied mean P(correct): {p_correct:.3f} '
          f'(sane range: not near 0.0 or 1.0; chance-level for a 5-option-ish item ~0.2-0.5)')
    return idata


def fit_sanity(df):
    model, formula = build_sanity_model(df)
    model.build()
    log_priors(model, 'sanity')
    prior_predictive_check(model, 'sanity')
    print(f'\nFitting sanity model: {formula}')
    idata = model.fit(draws=500, tune=500, chains=4, random_seed=0, progressbar=True)
    print(az.summary(idata))
    idata.to_netcdf(RESULTS_DIR / 'sanity_model.nc')
    print(f'Saved: {RESULTS_DIR / "sanity_model.nc"}')
    return model, idata


def fit_maximal(df, draws=1000, tune=1000, chains=4, target_accept=0.9):
    model, formula = build_maximal_model(df)
    model.build()
    log_priors(model, 'maximal')
    prior_predictive_check(model, 'maximal')
    print(f'\nFitting maximal model: {formula}')
    print(f'  draws={draws} tune={tune} chains={chains} target_accept={target_accept}')
    idata = model.fit(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                       random_seed=0, progressbar=True)
    idata.to_netcdf(RESULTS_DIR / 'maximal_model.nc')
    print(f'Saved: {RESULTS_DIR / "maximal_model.nc"}')
    return model, idata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['sanity', 'maximal'], default='sanity')
    ap.add_argument('--data', default=str(HERE / 'longdata_strict.csv'))
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--target-accept', type=float, default=0.9)
    args = ap.parse_args()

    df = load_data(args.data)
    print(f'Loaded {len(df)} rows, {df["pid"].nunique()} participants, '
          f'{df["topic"].nunique()} topics, {df["question_id"].nunique()} items')

    if args.stage == 'sanity':
        fit_sanity(df)
    else:
        fit_maximal(df, draws=args.draws, tune=args.tune, chains=args.chains,
                    target_accept=args.target_accept)


if __name__ == '__main__':
    main()
