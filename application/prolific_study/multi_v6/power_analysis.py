#!/usr/bin/env python3
"""Bayesian sample-size projection for multi_v6's condition contrasts.

Approach: analytic projection, not full re-simulation-and-refit. Posterior SD
for a fixed effect in this kind of model shrinks roughly as 1/sqrt(N) once the
likelihood dominates the (weakly-informative) prior, which is already the case
here (see the current model's ESS/r_hat — it's well past the prior-dominated
regime). Calibrate the proportionality constant from the CURRENT fit, then
project forward to hypothetical larger sample sizes.

This assumes:
  - The true effect size stays near the current posterior mean as N grows
    (i.e., no assumption that the effect will get bigger or smaller with more
    data - just tighter around wherever it currently sits).
  - Topic/condition/item balance stays roughly what it's been.
  - The 1/sqrt(N) scaling is a standard asymptotic approximation for this
    regime, anchored to match the current fit exactly at N=n_now by
    construction - NOT independently validated against real new data. Attempts
    to validate it in this session (--simulate, --simulate-full) both turned
    out to rely on bootstrap-resampling existing participants' rows as stand-ins
    for "new" participants, which is pseudo-replication (duplicated rows share
    the same real random-effect information, so it doesn't add the genuine new
    information the 1/sqrt(N) scaling assumes) - so those checks are NOT valid
    evidence for or against the scaling assumption. Treat the projection here
    as a standard, defensible approximation, not something empirically confirmed
    in this dataset. A real check would require genuine posterior-predictive
    simulation (fresh random-effect draws + fresh Bernoulli outcomes for
    simulated new participants), which is more involved than what's built here.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
from scipy import stats

HERE = Path(__file__).parent
RESULTS_DIR = HERE / 'model_results'

CONDS_NONREF = ['repeat_short', 'repeat_long', 'distractor']


def logit_to_prob(x):
    return 1 / (1 + np.exp(-x))


def current_posterior_stats_prob_scale(idata):
    """Return {condition: (mean, sd, q_lo, q_hi)} on the PROBABILITY-CORRECT
    scale (P(cond) - P(control)), matching the scale the ROPE/precision target
    was defined on, rather than the logit scale."""
    post = idata.posterior
    intercept = post['Intercept'].values.flatten()
    p_control = logit_to_prob(intercept)
    stats_by_cond = {}
    for name in CONDS_NONREF:
        coef = post['condition'].sel(condition_dim=name).values.flatten()
        diff = logit_to_prob(intercept + coef) - p_control
        stats_by_cond[name] = (diff.mean(), diff.std(),
                                np.percentile(diff, 2.5), np.percentile(diff, 97.5))
    return stats_by_cond


def current_posterior_stats(idata):
    """Return {condition: (mean, sd, q_lo, q_hi)} on the logit scale — q_lo/q_hi
    are the actual empirical 2.5%/97.5% quantiles of the posterior draws (not a
    normal approximation), so the "current N" row of the projection matches the
    real fitted CI exactly rather than an approximation that can disagree with
    it right at the boundary (this happened for repeat_long: normal-approx CI
    excluded zero at N=122 while the true empirical CI did not)."""
    post = idata.posterior
    stats_by_cond = {}
    for name in CONDS_NONREF:
        draws = post['condition'].sel(condition_dim=name).values.flatten()
        stats_by_cond[name] = (draws.mean(), draws.std(),
                                np.percentile(draws, 2.5), np.percentile(draws, 97.5))
    return stats_by_cond


def n_participants_from_longdata(path=HERE / 'longdata.csv'):
    df = pd.read_csv(path)
    return df['pid'].nunique()


def project(mean, q_lo, q_hi, n_now, n_target):
    """Scale the empirical quantile deviations from the mean by sqrt(n_now/n_target),
    preserving the current posterior's skew/shape instead of assuming normality.
    Matches the real fitted CI exactly at n_target == n_now."""
    scale = np.sqrt(n_now / n_target)
    ci_lo = mean + (q_lo - mean) * scale
    ci_hi = mean + (q_hi - mean) * scale
    sd_equiv = (ci_hi - ci_lo) / (2 * 1.96)  # for reporting only
    p_direction = stats.norm.cdf(abs(mean) / sd_equiv) if sd_equiv > 0 else np.nan
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    return sd_equiv, ci_lo, ci_hi, p_direction, excludes_zero


def n_needed_for_significance(mean, q_lo, q_hi, n_now, target_p_direction=0.975):
    """Total N at which the empirical-quantile-scaled 95% CI would just exclude
    zero, assuming the true effect stays at `mean`. Solves scale*deviation = mean
    on whichever side is closest to zero (the binding constraint)."""
    if mean == 0:
        return np.inf
    # whichever bound is on the same side as zero is the binding one
    bound = q_hi if mean < 0 else q_lo
    deviation = bound - mean
    if deviation == 0:
        return np.inf
    # need: mean + deviation*scale == 0  =>  scale == -mean/deviation
    scale_needed = -mean / deviation
    if scale_needed <= 0:
        # already excludes zero at any N, or moving further away — no finite N flips it
        return np.nan
    n_target = n_now / scale_needed ** 2
    return n_target


def analytic_projection(idata_path=RESULTS_DIR / 'maximal_model.nc',
                         longdata_path=HERE / 'longdata.csv',
                         target_ns=None):
    idata = az.from_netcdf(idata_path)
    stats_by_cond = current_posterior_stats(idata)
    n_now = n_participants_from_longdata(longdata_path)

    if target_ns is None:
        target_ns = [n_now, int(n_now * 1.25), int(n_now * 1.5), int(n_now * 2),
                     int(n_now * 3), int(n_now * 4)]

    print(f'Current N (participants): {n_now}\n')
    print('Assumption: true effect size = current posterior mean; the empirical '
          'posterior quantiles (not a normal approximation) are scaled by '
          'sqrt(n_now/n_target), so the "current" row below matches the actual '
          'fitted CI exactly.\n')

    for name in CONDS_NONREF:
        mean, sd_now, q_lo, q_hi = stats_by_cond[name]
        n_req = n_needed_for_significance(mean, q_lo, q_hi, n_now)
        req_str = f'{n_req:.0f} participants ({"+" if n_req > n_now else ""}{n_req - n_now:.0f} more than current)' \
            if np.isfinite(n_req) else 'not reachable by adding data at this effect size (already excludes zero, or bound moving away from zero)'
        print(f'=== {name} vs. control (current: mean={mean:+.3f}, sd={sd_now:.3f}, '
              f'logit scale) ===')
        print(f'  Projected N for 95% CI to exclude zero (if true effect stays at '
              f'{mean:+.3f}): {req_str}')
        print(f'  {"N":>6} {"~SD":>8} {"95% CI":>20} {"P(direction)":>13} {"excludes 0":>11}')
        for n_target in target_ns:
            sd_t, lo, hi, p_dir, excl = project(mean, q_lo, q_hi, n_now, n_target)
            marker = ' <- current (matches actual fit)' if n_target == n_now else ''
            print(f'  {n_target:>6} {sd_t:>8.3f}   [{lo:+.3f}, {hi:+.3f}]'
                  f'   {p_dir:>11.3f}   {str(excl):>11}{marker}')
        print()

    print('Caveat: this is an analytic 1/sqrt(N) projection anchored to the '
          'current fit, not a full re-simulation. It assumes the true effect '
          'size is exactly today\'s posterior mean — if the true effect is '
          'smaller, more data will be needed than this table suggests; if '
          'larger, less. Use --simulate for a (slower, approximate) check '
          'against actual posterior-predictive simulation.')

    return stats_by_cond, n_now


def diminishing_returns(idata_path=RESULTS_DIR / 'maximal_model.nc',
                         longdata_path=HERE / 'longdata.csv',
                         step=50, n_max=None, relative_threshold=0.05):
    """Diminishing-returns check, in the spirit of sapbenchmark's stopping rule:
    they picked their 10ms SD target by running a prospective power analysis and
    observing where adding more participants stopped meaningfully shrinking the
    item-level posterior SD. This does the analogous thing here: project
    probability-scale posterior SD in steps of `step` participants, report the
    marginal (not cumulative) SD reduction per step, and flag the first N where
    that marginal reduction drops below `relative_threshold` (default 5%) of
    the SD at that point - a rough empirical "elbow" rather than a fixed target
    picked in advance from a ROPE.
    """
    idata = az.from_netcdf(idata_path)
    stats_by_cond = current_posterior_stats_prob_scale(idata)
    n_now = n_participants_from_longdata(longdata_path)
    n_max = n_max or n_now * 8

    ns = list(range(n_now, n_max + 1, step))

    print(f'\n=== Diminishing-returns check (probability-correct scale, step={step}) ===')
    print(f'Marginal SD reduction per +{step} participants; "elbow" = first step where '
          f'the marginal reduction drops below {relative_threshold:.0%} of the SD at that point.\n')

    elbows = {}
    for name in CONDS_NONREF:
        mean, sd_now, q_lo, q_hi = stats_by_cond[name]
        sds = [sd_now * np.sqrt(n_now / n) for n in ns]
        print(f'{name} (current SD={sd_now:.4f} at N={n_now}):')
        elbow_n = None
        for i, n in enumerate(ns):
            if i == 0:
                print(f'  N={n:>5}  SD={sds[i]:.4f}   (baseline)')
                continue
            marginal = sds[i - 1] - sds[i]
            rel = marginal / sds[i - 1]
            flag = ''
            if elbow_n is None and rel < relative_threshold:
                elbow_n = n
                flag = '  <- elbow (marginal gain below threshold)'
            print(f'  N={n:>5}  SD={sds[i]:.4f}   marginal Δ={marginal:.4f} '
                  f'({rel:.1%} of prior SD){flag}')
        elbows[name] = elbow_n
        print()

    print('Suggested stopping N per contrast (first N where each extra '
          f'{step} participants buys < {relative_threshold:.0%} further SD reduction):')
    for name, elbow_n in elbows.items():
        print(f'  {name}: {elbow_n if elbow_n else f">{n_max} (not reached in this range)"}')
    overall = max(v for v in elbows.values() if v is not None) if any(elbows.values()) else None
    if overall:
        print(f'\nOverall suggested stopping point (slowest contrast to reach its elbow): '
              f'N ≈ {overall}')
    print('\nCaveat: this uses the same anchored 1/sqrt(N) scaling as the main '
          'projection (see module docstring) - it is not independently validated '
          'against real new data in this session.')

    return elbows


def cohens_d_target_n(idata_path=RESULTS_DIR / 'maximal_model.nc',
                       longdata_path=HERE / 'longdata.csv',
                       d_target=0.2):
    """N needed for each contrast's 95% CI width, expressed in Cohen's d, to be
    <= d_target. Uses the pooled binary-outcome SD at the model's baseline
    P(correct) to convert between the probability scale and d - no extra
    halving/judgment-call step, just the direct Cohen's-d convention (0.2 =
    small, 0.5 = medium)."""
    idata = az.from_netcdf(idata_path)
    n_now = n_participants_from_longdata(longdata_path)
    post = idata.posterior
    intercept = post['Intercept'].values.flatten()
    p_baseline = logit_to_prob(intercept).mean()
    sd_pooled = np.sqrt(p_baseline * (1 - p_baseline))

    n_per_contrast = {}
    for name in CONDS_NONREF:
        coef = post['condition'].sel(condition_dim=name).values.flatten()
        diff = logit_to_prob(intercept + coef) - logit_to_prob(intercept)
        d_now = diff.mean() / sd_pooled
        sd_d_now = diff.std() / sd_pooled
        sd_d_target = d_target / 3.92
        n_req = n_now * (sd_d_now / sd_d_target) ** 2
        n_per_contrast[name] = n_req
        print(f'  {name}: current d={d_now:+.3f}  N for CI width (in d) <= {d_target}: {n_req:.0f}')

    return n_per_contrast, n_now


def recommended_stopping_n(elbow_step=50, elbow_threshold=0.05, d_target=0.2):
    """Combined stopping rule: N = min(diminishing-returns elbow, N needed for
    Cohen's-d CI width <= d_target across ALL contrasts (worst case)). Prints
    both components and the final recommendation."""
    print('\n=== Combined stopping recommendation ===')
    print(f'Rule: N = min(diminishing-returns elbow, N for small-Cohen\'s-d '
          f'(d={d_target}) precision on the slowest contrast)\n')

    elbows = diminishing_returns(step=elbow_step, relative_threshold=elbow_threshold)
    elbow_n = max(v for v in elbows.values() if v is not None)

    print(f'\n--- Cohen\'s d (target={d_target}) precision requirement ---')
    n_per_contrast, n_now = cohens_d_target_n(d_target=d_target)
    d_n = max(n_per_contrast.values())

    final_n = min(elbow_n, d_n)
    print(f'\nDiminishing-returns elbow (worst contrast): N ≈ {elbow_n}')
    print(f'Cohen\'s d (target={d_target}) precision (worst contrast): N ≈ {d_n:.0f}')
    print(f'\n>>> Recommended stopping N = min({elbow_n}, {d_n:.0f}) = {min(elbow_n, d_n):.0f} <<<')
    print(f'    ({"elbow" if elbow_n <= d_n else "Cohen\'s d precision"} is the binding constraint)')
    print(f'    Current N = {n_now}; additional participants needed ≈ '
          f'{max(0, final_n - n_now):.0f}')

    return final_n


def simulate_check(n_target_list, n_reps=3, draws=500, tune=500, chains=2):
    """Approximate verification: simulate new item-level data from the current
    model's posterior-predictive distribution at hypothetical total N's, fit a
    FAST fixed-effects-only model (correct ~ condition + serial_position, no
    random effects — skipped for speed, not for the final analysis) on each,
    and check whether the resulting CI widths track the analytic projection.
    """
    import bambi as bmb
    from bayesian_model import load_data, build_sanity_model

    idata = az.from_netcdf(RESULTS_DIR / 'maximal_model.nc')
    df_real = load_data(HERE / 'longdata.csv')
    n_now = df_real['pid'].nunique()

    print(f'\n=== Simulation-based check (fixed-effects-only model, {n_reps} reps per N) ===')
    print('(Faster approximation for verification purposes only — the real '
          'inference should always use the full maximal model.)\n')

    rng = np.random.default_rng(0)
    pids = df_real[['pid', 'topic', 'condition', 'serial_position']].drop_duplicates(subset=['pid', 'topic'])

    for n_target in n_target_list:
        extra_needed = max(0, n_target - n_now)
        widths = {c: [] for c in CONDS_NONREF}
        for rep in range(n_reps):
            if extra_needed > 0:
                # Resample existing (pid, topic) blocks with replacement to stand
                # in for "new" participant-topic instances, keeping their real
                # condition/serial_position/item outcomes (bootstrap resampling
                # of observed rows, not a true posterior-predictive draw — a
                # deliberate simplification for this fast approximate check).
                sampled_pids = pids.sample(n=extra_needed, replace=True,
                                            random_state=rng.integers(1_000_000_000))
                sim_rows = []
                for i, row in enumerate(sampled_pids.itertuples()):
                    src = df_real[(df_real['pid'] == row.pid) & (df_real['topic'] == row.topic)]
                    new_block = src.copy()
                    new_block['pid'] = f'sim_{rep}_{i}'
                    sim_rows.append(new_block)
                df_sim_extra = pd.concat(sim_rows, ignore_index=True) if sim_rows else df_real.iloc[0:0]
                df_full = pd.concat([df_real, df_sim_extra], ignore_index=True)
            else:
                df_full = df_real

            model, _ = build_sanity_model(df_full)
            model.build()
            idata_sim = model.fit(draws=draws, tune=tune, chains=chains, random_seed=rep,
                                   progressbar=False)
            for c in CONDS_NONREF:
                d = idata_sim.posterior['condition'].sel(condition_dim=c).values.flatten()
                widths[c].append(float(np.percentile(d, 97.5) - np.percentile(d, 2.5)))

        print(f'N={n_target}:')
        for c in CONDS_NONREF:
            print(f'  {c}: mean 95% CI width across {n_reps} reps = {np.mean(widths[c]):.3f} '
                  f'(sd {np.std(widths[c]):.3f})')


def simulate_check_full(n_target_list, draws=1000, tune=2000, chains=4, target_accept=0.99):
    """Real verification: refit the FULL maximal model (all random effects
    included) on bootstrap-resampled larger-N data. One fit per N, no repeats
    (kept to 1 rep to bound runtime — this is a rough sanity check on the
    analytic projection, not a precise estimate in its own right). Slow
    (minutes per N) but doesn't share the fast-model's bias toward
    underestimating uncertainty.
    """
    from bayesian_model import load_data, build_maximal_model

    df_real = load_data(HERE / 'longdata.csv')
    n_now = df_real['pid'].nunique()

    print(f'\n=== Full-model simulation check ({len(n_target_list)} N values, 1 fit each) ===')

    rng = np.random.default_rng(0)
    pids = df_real[['pid', 'topic', 'condition', 'serial_position']].drop_duplicates(subset=['pid', 'topic'])

    for n_target in n_target_list:
        extra_needed = max(0, n_target - n_now)
        if extra_needed > 0:
            sampled_pids = pids.sample(n=extra_needed, replace=True,
                                        random_state=rng.integers(1_000_000_000))
            sim_rows = []
            for i, row in enumerate(sampled_pids.itertuples()):
                src = df_real[(df_real['pid'] == row.pid) & (df_real['topic'] == row.topic)]
                new_block = src.copy()
                new_block['pid'] = f'sim_{i}'
                sim_rows.append(new_block)
            df_sim_extra = pd.concat(sim_rows, ignore_index=True) if sim_rows else df_real.iloc[0:0]
            df_full = pd.concat([df_real, df_sim_extra], ignore_index=True)
        else:
            df_full = df_real

        print(f'\nFitting full maximal model at N={n_target} '
              f'({len(df_full)} rows, {df_full["pid"].nunique()} participants)...')
        model, _ = build_maximal_model(df_full)
        model.build()
        idata_sim = model.fit(draws=draws, tune=tune, chains=chains,
                               target_accept=target_accept, random_seed=0, progressbar=True)
        idata_sim.to_netcdf(RESULTS_DIR / f'sim_full_N{n_target}.nc')

        n_div = int(idata_sim.sample_stats.diverging.sum())
        s = az.summary(idata_sim, var_names=['condition'])
        print(f'N={n_target}: {n_div} divergences, max r_hat={s["r_hat"].max()}')
        for c in CONDS_NONREF:
            d = idata_sim.posterior['condition'].sel(condition_dim=c).values.flatten()
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            print(f'  {c}: mean={d.mean():+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  width={hi - lo:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simulate', action='store_true',
                     help='Also run the (slower) simulation-based verification.')
    ap.add_argument('--sim-ns', type=int, nargs='+', default=None,
                     help='Target total N values for --simulate (default: current, 1.5x, 2x).')
    ap.add_argument('--simulate-full', action='store_true',
                     help='Real verification: refit the full maximal model (slow, ~minutes per N).')
    ap.add_argument('--sim-full-ns', type=int, nargs='+', default=None,
                     help='Target total N values for --simulate-full (default: current, 2x).')
    ap.add_argument('--diminishing-returns', action='store_true',
                     help='Run the diminishing-returns elbow check (probability scale).')
    ap.add_argument('--dr-step', type=int, default=50)
    ap.add_argument('--dr-threshold', type=float, default=0.05)
    ap.add_argument('--dr-max', type=int, default=None)
    ap.add_argument('--recommend', action='store_true',
                     help='Print the combined stopping recommendation: '
                          'min(diminishing-returns elbow, Cohen\'s-d-0.2 precision N).')
    args = ap.parse_args()

    if args.recommend:
        recommended_stopping_n(elbow_step=args.dr_step, elbow_threshold=args.dr_threshold)
        return

    stats_by_cond, n_now = analytic_projection()

    if args.diminishing_returns:
        diminishing_returns(step=args.dr_step, n_max=args.dr_max,
                             relative_threshold=args.dr_threshold)

    if args.simulate:
        sim_ns = args.sim_ns or [n_now, int(n_now * 1.5), int(n_now * 2)]
        simulate_check(sim_ns)

    if args.simulate_full:
        sim_full_ns = args.sim_full_ns or [n_now, int(n_now * 2)]
        simulate_check_full(sim_full_ns)


if __name__ == '__main__':
    main()
