#!/usr/bin/env python3
"""Analysis of multi_v6 pilot results (4 topics, no musical_instruments)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_prep import (
    HERE, DATA_FILE, TOPICS, TOPIC_LABEL, CONDS, COND_LABEL,
    TOPIC_CONTENT_DIRNAME, FL_TOPIC, col_correct, col_options, col_qid,
    content_cols, parse_response, load_records,
)

colors       = ['#94a3b8', '#fb923c', '#f97316', '#a78bfa']
topic_colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ec4899']

records, pass_recs = load_records()
for rec in records:
    flags = {t: rec[t]['cond'][:4] for t in TOPICS}
    accs  = {t: f"{rec[t]['acc']:.2f}" for t in TOPICS}
    print(f"  {'P' if rec['attn_pass'] else 'F'} g={rec['group']:>2} | " +
          ' | '.join(f"{t[:3]}({flags[t]})={accs[t]}" for t in TOPICS))


_boot_rng = np.random.RandomState(42)

def bootstrap_ci(vals, n_boot=2000):
    """Returns (lower_err, upper_err) relative to mean for asymmetric yerr."""
    if len(vals) < 2:
        return 0.0, 0.0
    arr = np.array(vals, dtype=float)
    boot_means = [_boot_rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    m = arr.mean()
    return float(m - lo), float(hi - m)


def within_subject_ci(cond_accs_dict, conds, n_boot=2000):
    """Cousineau-Morey within-subject 95% CIs (bootstrap).

    Removes between-participant variance before computing CIs.
    Applies Morey (2008) correction factor sqrt(J/(J-1)).
    cond_accs_dict must have same-length lists (index i = same participant).
    Returns dict: condition -> (lo_err, hi_err) relative to mean.
    """
    J = len(conds)
    n = len(cond_accs_dict[conds[0]])
    # participant × condition matrix
    Y = np.array([[cond_accs_dict[c][i] for c in conds] for i in range(n)], dtype=float)
    mu_p = Y.mean(axis=1, keepdims=True)
    mu   = Y.mean()
    Y_norm = Y - mu_p + mu
    morey  = np.sqrt(J / (J - 1))
    result = {}
    for ci, c in enumerate(conds):
        lo, hi = bootstrap_ci(Y_norm[:, ci], n_boot=n_boot)
        result[c] = (lo * morey, hi * morey)
    return result


def dot_offsets(ys, dy=0.07, dx=0.04):
    from collections import defaultdict
    bins = defaultdict(list)
    for i, y in enumerate(ys):
        bins[round(float(y) / dy)].append(i)
    offsets = np.zeros(len(ys))
    for idxs in bins.values():
        n  = len(idxs)
        xs = np.arange(n) * dx - (n - 1) * dx / 2
        for k, idx in enumerate(idxs):
            offsets[idx] = xs[k]
    return offsets


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Accuracy by condition (pooled + by topic × condition)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
# Build participant × condition dict (Latin square: each participant contributes
# exactly one value per condition, in pass_recs order).
cond_accs = {c: [] for c in CONDS}
for rec in pass_recs:
    for t in TOPICS:
        c = rec[t]['cond']
        if c:
            cond_accs[c].append(rec[t]['acc'])
ws_ci  = within_subject_ci(cond_accs, CONDS)
means  = [np.mean(cond_accs[c]) if cond_accs[c] else 0 for c in CONDS]
ci_lo  = [ws_ci[c][0] for c in CONDS]
ci_hi  = [ws_ci[c][1] for c in CONDS]
bars   = ax.bar([COND_LABEL[c] for c in CONDS], means, yerr=[ci_lo, ci_hi], capsize=5,
                color=colors, edgecolor='#334155', linewidth=0.8)
ax.set_ylim(0, 1); ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by condition\n(attn-pass n={len(pass_recs)})')
ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8, label='Chance (1/32)')
ax.plot([], [], color='#334155', linewidth=0, label='Error bars: within-subj 95% CI\n(Cousineau-Morey)')
ax.legend(fontsize=8)
for bar, m, hi in zip(bars, means, ci_hi):
    ax.text(bar.get_x() + bar.get_width() / 2, m + hi + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
x = np.arange(len(TOPICS)); width = 0.2
for i, c in enumerate(CONDS):
    tm, tlo, thi = [], [], []
    for t in TOPICS:
        accs = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        tm.append(np.mean(accs) if accs else 0)
        lo, hi = bootstrap_ci(accs)
        tlo.append(lo); thi.append(hi)
    ax.bar(x + i * width, tm, width, yerr=[tlo, thi], capsize=3,
           label=COND_LABEL[c].replace('\n', ' '), color=colors[i], edgecolor='#334155', linewidth=0.6)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_ylim(0, 1); ax.set_ylabel('Proportion correct')
ax.set_title(f'Accuracy by topic × condition\n(attn-pass n={len(pass_recs)})')
ax.legend(fontsize=8); ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(HERE / 'v6_accuracy_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v6_accuracy_by_condition.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Per-question accuracy (aggregate, per topic)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, topic in zip(axes.flat, TOPICS):
    cols = content_cols[topic]
    qlabels = [col_qid[c] for c in cols]
    means_pass = [np.mean([rec[topic]['scores'][cols.index(c)] for rec in pass_recs]) for c in cols]
    x = np.arange(len(cols))
    ax.bar(x, means_pass, 0.5, color='#3b82f6', edgecolor='#334155', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(qlabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Proportion correct')
    ax.set_title(f'{TOPIC_LABEL[topic]} — per-question accuracy (n={len(pass_recs)})')
plt.tight_layout()
plt.savefig(HERE / 'v6_per_question.png', dpi=150, bbox_inches='tight')
print('Saved: v6_per_question.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 2b: Per-question accuracy by condition
# ══════════════════════════════════════════════════════════════════════════
n_conds = len(CONDS); width = 0.18
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, topic in zip(axes.flat, TOPICS):
    cols = content_cols[topic]
    qlabels = [col_qid[c] for c in cols]
    x = np.arange(len(cols))
    for ci, c in enumerate(CONDS):
        recs_c = [rec for rec in pass_recs if rec[topic]['cond'] == c]
        means = [np.mean([rec[topic]['scores'][qi] for rec in recs_c]) if recs_c else 0
                 for qi in range(len(cols))]
        offset = (ci - (n_conds - 1) / 2) * width
        ax.bar(x + offset, means, width, label=COND_LABEL[c].replace('\n', ' '),
               color=colors[ci], edgecolor='#334155', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(qlabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Proportion correct')
    ax.set_title(f'{TOPIC_LABEL[topic]} — per-question accuracy by condition')
    if topic == TOPICS[0]:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v6_per_question_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v6_per_question_by_condition.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 3a: Violin — accuracy by condition
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
cond_data  = {c: [] for c in CONDS}
cond_tcols = {c: [] for c in CONDS}
for rec in pass_recs:
    for ti, t in enumerate(TOPICS):
        c = rec[t]['cond']
        if c:
            cond_data[c].append(rec[t]['acc'])
            cond_tcols[c].append(topic_colors[ti])
positions = list(range(len(CONDS)))
nonempty = [c for c in CONDS if len(cond_data[c]) >= 2]
if nonempty:
    parts = ax.violinplot([cond_data[c] for c in nonempty],
                          positions=[CONDS.index(c) for c in nonempty],
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], [colors[CONDS.index(c)] for c in nonempty]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(2)
for i, c in enumerate(CONDS):
    offs = dot_offsets(cond_data[c])
    ax.scatter(np.full(len(cond_data[c]), i) + offs, cond_data[c],
               s=18, color=cond_tcols[c], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ti, t in enumerate(TOPICS):
    ax.scatter([], [], s=18, color=topic_colors[ti], edgecolor='#334155', linewidth=0.5, label=TOPIC_LABEL[t])
ax.set_xticks(positions); ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by condition\n(attn-pass n={len(pass_recs)})')
ax.legend(fontsize=8, title='Topic', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v6_violin_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v6_violin_by_condition.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 3b: Violin — accuracy by topic × condition
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
vwidth = 0.18
offsets_4 = np.array([-0.3, -0.1, 0.1, 0.3])
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        vals = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets_4[ci]
        if len(vals) > 1:
            parts = ax.violinplot(vals, positions=[pos], widths=vwidth, showmedians=True, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[ci]); parts['bodies'][0].set_alpha(0.7)
            parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(1.5)
        offs = dot_offsets(vals, dx=0.03)
        ax.scatter(np.full(len(vals), pos) + offs, vals,
                   s=14, color=colors[ci], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ci, c in enumerate(CONDS):
    ax.scatter([], [], s=18, color=colors[ci], edgecolor='#334155', linewidth=0.5, label=COND_LABEL[c].replace('\n', ' '))
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by topic × condition (attn-pass n={len(pass_recs)})')
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v6_violin_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v6_violin_by_topic.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 3c: Partial credit violin by topic × condition
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        vals = [rec[t]['partial_acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets_4[ci]
        if len(vals) > 1:
            parts = ax.violinplot(vals, positions=[pos], widths=vwidth, showmedians=True, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[ci]); parts['bodies'][0].set_alpha(0.7)
            parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(1.5)
        offs = dot_offsets(vals, dx=0.03)
        ax.scatter(np.full(len(vals), pos) + offs, vals,
                   s=14, color=colors[ci], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ci, c in enumerate(CONDS):
    ax.scatter([], [], s=18, color=colors[ci], edgecolor='#334155', linewidth=0.5, label=COND_LABEL[c].replace('\n', ' '))
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('Partial credit score')
ax.set_title(f'Partial credit by topic × condition (n={len(pass_recs)})')
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v6_violin_partial_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v6_violin_partial_by_topic.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Completion time vs. accuracy (participant-level)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for ti, t in enumerate(TOPICS):
    xs = [rec[t]['t_passage'] + rec[t]['t_qs'] for rec in pass_recs if rec[t]['t_passage'] > 0]
    ys = [rec[t]['acc'] for rec in pass_recs if rec[t]['t_passage'] > 0]
    ax.scatter(xs, ys, s=28, color=topic_colors[ti], alpha=0.7, edgecolor='#334155', linewidth=0.4, label=TOPIC_LABEL[t])
all_x = [rec[t]['t_passage'] + rec[t]['t_qs'] for rec in pass_recs for t in TOPICS if rec[t]['t_passage'] > 0]
all_y = [rec[t]['acc'] for rec in pass_recs for t in TOPICS if rec[t]['t_passage'] > 0]
if len(all_x) > 2:
    corr = np.corrcoef(all_x, all_y)[0, 1]
    ax.set_title(f'Time on passage+questions vs. accuracy\n(per topic-instance, r={corr:.2f})')
else:
    ax.set_title('Time on passage+questions vs. accuracy')
ax.set_xlabel('Time on passage + questions (s)'); ax.set_ylabel('Proportion correct')
ax.legend(fontsize=8)

ax = axes[1]
part_dur = [rec['duration'] for rec in pass_recs]
part_acc = [np.mean([rec[t]['acc'] for t in TOPICS]) for rec in pass_recs]
ax.scatter(part_dur, part_acc, s=40, color='#3b82f6', alpha=0.8, edgecolor='#334155', linewidth=0.5)
if len(part_dur) > 2:
    corr2 = np.corrcoef(part_dur, part_acc)[0, 1]
    ax.set_title(f'Total survey duration vs. mean accuracy\n(per participant, r={corr2:.2f})')
else:
    ax.set_title('Total survey duration vs. mean accuracy')
ax.set_xlabel('Total survey duration (s)'); ax.set_ylabel('Mean accuracy across topics')

plt.tight_layout()
plt.savefig(HERE / 'v6_time_vs_accuracy.png', dpi=150, bbox_inches='tight')
print('Saved: v6_time_vs_accuracy.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Completion order — viewing order per participant + accuracy by
# serial position (order effects / fatigue check)
# ══════════════════════════════════════════════════════════════════════════
COND_ABBREV = {'control': 'ctrl', 'repeat_short': 'r-sh', 'repeat_long': 'r-lo', 'distractor': 'dist'}
topic_colors_map = {t: c for t, c in zip(TOPICS, topic_colors)}
recs_with_order = [r for r in pass_recs if r['fl_order']]
ordered_recs = sorted(recs_with_order, key=lambda r: int(r['group']) if str(r['group']).isdigit() else 99)
n_part = len(ordered_recs)

fig = plt.figure(figsize=(14, max(5, n_part * 0.4 + 1.5)))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])

ax = fig.add_subplot(gs[0])
for pi, rec in enumerate(ordered_recs):
    y = n_part - 1 - pi
    order = rec['fl_order']
    for pos, topic in enumerate(order):
        col = topic_colors_map.get(topic, '#cccccc')
        rect = plt.Rectangle([pos, y - 0.4], 1, 0.8, color=col, linewidth=0.8, edgecolor='#334155')
        ax.add_patch(rect)
        cond = rec.get(topic, {}).get('cond', '') if isinstance(rec.get(topic), dict) else ''
        ax.text(pos + 0.5, y + 0.12, TOPIC_LABEL.get(topic, topic)[:3], ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold')
        ax.text(pos + 0.5, y - 0.17, COND_ABBREV.get(cond, cond[:4]), ha='center', va='center',
                fontsize=6, color='white', alpha=0.9)
    ax.text(-0.15, y, f"g={rec['group']}", ha='right', va='center', fontsize=7, color='#64748b')
ax.set_xlim(-1.5, len(TOPICS) + 0.2); ax.set_ylim(-0.7, n_part - 0.3)
ax.set_xticks([i + 0.5 for i in range(len(TOPICS))])
ax.set_xticklabels([f'{i+1}{"stndrdth"[0 if i==0 else 1 if i==1 else 2 if i==2 else 3:0+1]}' for i in range(len(TOPICS))], fontsize=9)
ax.set_xticklabels(['1st', '2nd', '3rd', '4th'][:len(TOPICS)], fontsize=10)
ax.set_yticks([]); ax.set_xlabel('Position in study')
ax.set_title(f'Topic viewing order per participant (n={n_part})')
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
for t in TOPICS:
    ax.scatter([], [], s=60, color=topic_colors_map[t], label=TOPIC_LABEL[t], edgecolor='#334155', linewidth=0.5)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

ax2 = fig.add_subplot(gs[1])
pos_accs = {i: [] for i in range(len(TOPICS))}
for rec in recs_with_order:
    for pos, topic in enumerate(rec['fl_order']):
        pos_accs[pos].append(rec[topic]['acc'])
positions = list(range(len(TOPICS)))
means  = [np.mean(pos_accs[p]) if pos_accs[p] else 0 for p in positions]
ci_lo  = [bootstrap_ci(pos_accs[p])[0] for p in positions]
ci_hi  = [bootstrap_ci(pos_accs[p])[1] for p in positions]
bars = ax2.bar([f'{p+1}{["st","nd","rd","th"][min(p,3)]}' for p in positions], means, yerr=[ci_lo, ci_hi], capsize=5,
               color='#3b82f6', edgecolor='#334155', linewidth=0.8)
ax2.set_ylim(0, 1); ax2.set_ylabel('Proportion correct (exact match)')
ax2.set_xlabel('Serial position in study')
ax2.set_title(f'Accuracy by serial position\n(order effect check, n={len(recs_with_order)})')
for bar, m, hi in zip(bars, means, ci_hi):
    ax2.text(bar.get_x() + bar.get_width() / 2, m + hi + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(HERE / 'v6_completion_order.png', dpi=150, bbox_inches='tight')
print('Saved: v6_completion_order.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Difficulty ratings — by topic, and by topic × condition
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (dkey, dlabel) in zip(axes[0], [('diff_passage', 'Passage difficulty'), ('diff_qs', 'Question difficulty')]):
    means, ci_lo, ci_hi = [], [], []
    for t in TOPICS:
        vals = [rec[t][dkey] for rec in pass_recs if not np.isnan(rec[t][dkey])]
        means.append(np.mean(vals) if vals else 0)
        lo, hi = bootstrap_ci(vals)
        ci_lo.append(lo); ci_hi.append(hi)
    bars = ax.bar([TOPIC_LABEL[t] for t in TOPICS], means, yerr=[ci_lo, ci_hi], capsize=5,
                  color=topic_colors, edgecolor='#334155', linewidth=0.8)
    ax.set_ylim(0, 10); ax.set_ylabel('Rating (0-10, higher = harder)')
    ax.set_title(f'{dlabel} by topic (n={len(pass_recs)})')
    for bar, m, hi in zip(bars, means, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, m + hi + 0.15, f'{m:.1f}', ha='center', va='bottom', fontsize=9)

for ax, (dkey, dlabel) in zip(axes[1], [('diff_passage', 'Passage difficulty'), ('diff_qs', 'Question difficulty')]):
    x = np.arange(len(TOPICS)); width = 0.2
    for ci, c in enumerate(CONDS):
        tm, tlo, thi = [], [], []
        for t in TOPICS:
            vals = [rec[t][dkey] for rec in pass_recs if rec[t]['cond'] == c and not np.isnan(rec[t][dkey])]
            tm.append(np.mean(vals) if vals else 0)
            lo, hi = bootstrap_ci(vals)
            tlo.append(lo); thi.append(hi)
        ax.bar(x + ci * width, tm, width, yerr=[tlo, thi], capsize=3,
               label=COND_LABEL[c].replace('\n', ' '), color=colors[ci], edgecolor='#334155', linewidth=0.6)
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
    ax.set_ylim(0, 10); ax.set_ylabel('Rating (0-10, higher = harder)')
    ax.set_title(f'{dlabel} by topic × condition')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(HERE / 'v6_difficulty.png', dpi=150, bbox_inches='tight')
print('Saved: v6_difficulty.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Per-participant line graph across conditions, faceted by "group"
# (the fixed topic-to-condition permutation branch), with jitter to separate
# overlapping trajectories (accuracy only takes 6 discrete values).
# Group determines which topic lands in the distractor slot, so rows = distractor
# topic, groups sorted within each row — makes the topic-level pattern legible.
# Each participant sees all 4 conditions (one per topic, Latin-square design),
# so this is a genuine within-subject repeated-measures view.
# ══════════════════════════════════════════════════════════════════════════
rng = np.random.RandomState(0)
x = np.arange(len(CONDS))
JITTER_X, JITTER_Y = 0.06, 0.02

def group_dist_topic(g):
    recs = [rec for rec in pass_recs if rec['group'] == g]
    for t in TOPICS:
        if recs and recs[0][t]['cond'] == 'distractor':
            return t
    return '?'

groups_all = sorted({rec['group'] for rec in pass_recs}, key=lambda g: int(g) if str(g).isdigit() else 99)
groups_by_topic = {t: [g for g in groups_all if group_dist_topic(g) == t] for t in TOPICS}
ncols = max(len(v) for v in groups_by_topic.values())
nrows = len(TOPICS)
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharey=True, sharex=True)

for row, t in enumerate(TOPICS):
    row_groups = groups_by_topic[t]
    for col in range(ncols):
        ax = axes[row, col]
        if col >= len(row_groups):
            ax.axis('off')
            continue
        g = row_groups[col]
        if col == 0:
            ax.set_ylabel(f'Distractor =\n{TOPIC_LABEL[t]}', fontsize=10)
        facet_recs = [rec for rec in pass_recs if rec['group'] == g]

        for rec in facet_recs:
            ys = []
            for c in CONDS:
                matches = [rec[tt]['acc'] for tt in TOPICS if rec[tt]['cond'] == c]
                ys.append(matches[0] if matches else np.nan)
            xj = x + rng.uniform(-JITTER_X, JITTER_X, size=len(x))
            yj = np.array(ys) + rng.uniform(-JITTER_Y, JITTER_Y, size=len(x))
            ax.plot(xj, yj, color='#94a3b8', alpha=0.5, linewidth=1, marker='o', markersize=3, zorder=1)

        mean_ys = []
        for c in CONDS:
            vals = []
            for rec in facet_recs:
                matches = [rec[tt]['acc'] for tt in TOPICS if rec[tt]['cond'] == c]
                if matches:
                    vals.append(matches[0])
            mean_ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(x, mean_ys, color='#334155', linewidth=2.5, marker='o', markersize=6, zorder=3)

        ax.set_xticks(x); ax.set_xticklabels(['C', 'RS', 'RL', 'D'], fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f'g={g} (n={len(facet_recs)})', fontsize=9)

fig.suptitle(f'Per-participant accuracy across conditions, faceted by group (rows = distractor topic)\n'
             f'(within-subject; C=control, RS=repeat_short, RL=repeat_long, D=distractor; total n={len(pass_recs)})', y=1.01)
plt.tight_layout()
plt.savefig(HERE / 'v6_participant_lines.png', dpi=150, bbox_inches='tight')
print('Saved: v6_participant_lines.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Print summary tables
# ══════════════════════════════════════════════════════════════════════════
print('\n=== Per-question accuracy (attn-pass) ===')
for topic in TOPICS:
    cols = content_cols[topic]
    print(f'\n{TOPIC_LABEL[topic]}:')
    for col in cols:
        scores = [rec[topic]['scores'][cols.index(col)] for rec in pass_recs]
        qid = col_qid.get(col, col)
        print(f'  {col} ({qid}): {np.mean(scores):.0%} ({sum(scores)}/{len(scores)})')

print('\n=== Mean accuracy by condition (attn-pass, pooled across topics) ===')
for c in CONDS:
    accs = [rec[t]['acc'] for rec in pass_recs for t in TOPICS if rec[t]['cond'] == c]
    print(f'  {c}: {np.mean(accs):.2f} (n={len(accs)})')

print('\n=== Sample sizes per topic × condition (finished-with-cond) ===')
from collections import Counter
for t in TOPICS:
    ctr = Counter(rec[t]['cond'] for rec in records if rec[t]['cond'])
    print(f'  {t}: ' + ', '.join(f'{c}={ctr.get(c,0)}' for c in CONDS))

print('\n=== Sample sizes per topic × condition (attn-pass) ===')
for t in TOPICS:
    ctr = Counter(rec[t]['cond'] for rec in pass_recs if rec[t]['cond'])
    print(f'  {t}: ' + ', '.join(f'{c}={ctr.get(c,0)}' for c in CONDS))
min_cell = min(
    sum(1 for rec in pass_recs if rec[t]['cond'] == c)
    for t in TOPICS for c in CONDS
)
print(f'  Minimum cell n (attn-pass): {min_cell}')

print('\n=== Duration (all finished-with-cond) ===')
durs = [r['duration'] for r in records]
print(f'  Mean: {np.mean(durs)/60:.1f} min  Median: {np.median(durs)/60:.1f} min  '
      f'Range: {min(durs)/60:.1f}-{max(durs)/60:.1f} min')

print('\n=== Attention check pass rate ===')
print(f'  {len(pass_recs)}/{len(records)} ({len(pass_recs)/len(records):.0%})')

print('\n=== Overall accuracy (attn-pass, all topics pooled) ===')
all_scores = [s for rec in pass_recs for t in TOPICS for s in rec[t]['scores']]
print(f'  {np.mean(all_scores):.2%} (n_items={len(all_scores)})')

print('\n=== Difficulty ratings (attn-pass, 0-10 scale) ===')
for t in TOPICS:
    dp = [rec[t]['diff_passage'] for rec in pass_recs if not np.isnan(rec[t]['diff_passage'])]
    dq = [rec[t]['diff_qs'] for rec in pass_recs if not np.isnan(rec[t]['diff_qs'])]
    print(f'  {t}: passage={np.mean(dp):.1f}  questions={np.mean(dq):.1f}')
