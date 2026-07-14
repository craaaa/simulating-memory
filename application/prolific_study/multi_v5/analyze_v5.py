#!/usr/bin/env python3
"""Analysis of multi_v5 pilot results (first batch)."""
import csv, yaml
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA_FILE = HERE / 'multi_v5_July+9,+2026_09.28.tsv'

TOPICS = ['martial_arts', 'fruits_v2', 'astronomy', 'fabrics', 'musical_instruments']
TOPIC_LABEL = {
    'martial_arts': 'Martial Arts', 'fruits_v2': 'Fruits v2',
    'astronomy': 'Astronomy', 'fabrics': 'Fabrics',
    'musical_instruments': 'Musical Instr.',
}
CONDS = ['control', 'repeat_short', 'repeat_long', 'distractor']
COND_LABEL = {
    'control': 'Control', 'repeat_short': 'Repeat\nShort',
    'repeat_long': 'Repeat\nLong', 'distractor': 'Distractor',
}

# musical_instruments content lives in multi_v2, not multi_v5
CONTENT_DIR = {
    'martial_arts': HERE, 'fruits_v2': HERE / '..' / 'multi_v5',
    'astronomy': HERE, 'fabrics': HERE,
    'musical_instruments': HERE / '..' / 'multi_v2',
}
TOPIC_CONTENT_DIRNAME = {
    'martial_arts': 'martial_arts', 'fruits_v2': 'fruits',
    'astronomy': 'astronomy', 'fabrics': 'fabrics',
    'musical_instruments': 'musical_instruments',
}

col_correct = {}
col_options = {}
col_qid     = {}
content_cols = {}

for topic in TOPICS:
    base = CONTENT_DIR[topic] / TOPIC_CONTENT_DIRNAME[topic]
    with open(base / 'questions.yaml') as f:
        qdata = yaml.safe_load(f)
    topic_content_cols = []
    for q in qdata['questions']:
        if q['metadata']['type'] != 'content':
            continue
        qid = q['q_id']
        tsvcol = f'{topic}_{qid}'
        opts = q['options']
        ans  = q['answer'] if isinstance(q['answer'], list) else [q['answer']]
        col_correct[tsvcol] = frozenset(opts[a] for a in ans)
        col_options[tsvcol] = list(opts.values())
        col_qid[tsvcol]     = qid
        topic_content_cols.append(tsvcol)
    content_cols[topic] = topic_content_cols


def parse_response(raw, all_opts):
    if not raw:
        return frozenset()
    for r in range(len(all_opts) + 1):
        for combo in combinations(all_opts, r):
            if ','.join(combo) == raw:
                return frozenset(combo)
    for r in range(len(all_opts) + 1):
        for combo in combinations(all_opts, r):
            if ', '.join(combo) == raw:
                return frozenset(combo)
    return frozenset(s.strip() for s in raw.split(','))


with open(DATA_FILE, encoding='utf-16') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)
data = rows[2:]  # skip Qualtrics description rows

# Dedupe by PROLIFIC_PID: keep the longest-duration row per participant (repeat
# rows are typically an aborted quick restart followed by the real completed run).
best_by_pid = {}
for r in data:
    pid = r.get('PROLIFIC_PID', '').strip()
    if not pid:
        continue
    dur = float(r.get('Duration (in seconds)', 0) or 0)
    if pid not in best_by_pid or dur > float(best_by_pid[pid].get('Duration (in seconds)', 0) or 0):
        best_by_pid[pid] = r
n_before = len([r for r in data if r.get('PROLIFIC_PID', '').strip()])
data = list(best_by_pid.values()) + [r for r in data if not r.get('PROLIFIC_PID', '').strip()]
print(f'Deduped by PROLIFIC_PID: {n_before} rows -> {len(best_by_pid)} unique participants')

records = []
for r in data:
    if r.get('Finished', '').strip() != 'True':
        continue
    # Skip preview/test rows with no condition assigned
    if not r.get('martial_arts_cond', '').strip():
        continue

    attn_pass = True
    for t in TOPICS:
        at_col = f'{t}_AT_{t}' if t != 'astronomy' else f'{t}_AT_{t}'
        # AT column names: martial_arts_AT_martial_arts, fruits_v2_AT_fruits_v2,
        # astronomy_AT_astronomy, fabrics_AT_fabrics, musical_instruments_AT_music
        at_col = {
            'martial_arts': 'martial_arts_AT_martial_arts',
            'fruits_v2': 'fruits_v2_AT_fruits_v2',
            'astronomy': 'astronomy_AT_astronomy',
            'fabrics': 'fabrics_AT_fabrics',
            'musical_instruments': 'musical_instruments_AT_music',
        }[t]
        correct_col = f'{t}_AT_correct'
        raw = r.get(at_col, '').strip()
        correct_raw = r.get(correct_col, '').strip()
        correct = frozenset(correct_raw.split(',')) if correct_raw else frozenset()
        given = frozenset(raw.split(',')) if raw else frozenset()
        if given != correct:
            attn_pass = False

    rec = {'attn_pass': attn_pass, 'group': r.get('group', '?'),
           'duration': float(r.get('Duration (in seconds)', 0) or 0)}

    for topic in TOPICS:
        cols = content_cols[topic]
        scores, partial = [], []
        for col in cols:
            raw   = r.get(col, '').strip()
            given = parse_response(raw, col_options[col])
            exact = 1 if given == col_correct[col] else 0
            n_correct_opts  = len(col_correct[col])
            n_given_correct = len(given & col_correct[col])
            n_given_wrong   = len(given - col_correct[col])
            pc = max(0, (n_given_correct - n_given_wrong) / n_correct_opts) if n_correct_opts else 0
            scores.append(exact)
            partial.append(pc)

        d_passage = r.get(f'{topic}_difficulty_1', '').strip()
        d_qs      = r.get(f'{topic}_difficulty_2', '').strip()

        rec[topic] = {
            'cond':         r.get(f'{topic}_cond', '').strip(),
            'scores':       scores,
            'partial':      partial,
            'acc':          np.mean(scores),
            'partial_acc':  np.mean(partial),
            'diff_passage': float(d_passage) if d_passage else np.nan,
            'diff_qs':      float(d_qs) if d_qs else np.nan,
        }
    records.append(rec)

pass_recs = [r for r in records if r['attn_pass']]
print(f'Finished (with cond): {len(records)}, attention-pass: {len(pass_recs)}')
for rec in records:
    flags = {t: rec[t]['cond'][:4] for t in TOPICS}
    accs  = {t: f"{rec[t]['acc']:.2f}" for t in TOPICS}
    print(f"  {'P' if rec['attn_pass'] else 'F'} g={rec['group']:>2} | " +
          ' | '.join(f"{t[:3]}({flags[t]})={accs[t]}" for t in TOPICS))

colors = ['#94a3b8', '#fb923c', '#f97316', '#a78bfa']

# ── Figure 1: Accuracy by condition (pooled across topics) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
cond_accs = {c: [] for c in CONDS}
for rec in pass_recs:
    for t in TOPICS:
        c = rec[t]['cond']
        if c:
            cond_accs[c].append(rec[t]['acc'])
means = [np.mean(cond_accs[c]) if cond_accs[c] else 0 for c in CONDS]
sems  = [np.std(cond_accs[c]) / np.sqrt(len(cond_accs[c])) if len(cond_accs[c]) > 1 else 0 for c in CONDS]
bars  = ax.bar([COND_LABEL[c] for c in CONDS], means, yerr=sems, capsize=5,
               color=colors, edgecolor='#334155', linewidth=0.8)
ax.set_ylim(0, 1); ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by condition\n(attn-pass n={len(pass_recs)})')
for bar, m, se in zip(bars, means, sems):
    ax.text(bar.get_x() + bar.get_width() / 2, m + se + 0.02,
            f'{m:.2f}', ha='center', va='bottom', fontsize=9)
for c in CONDS:
    print(f'  {c}: n={len(cond_accs[c])}')

ax = axes[1]
x = np.arange(len(TOPICS)); width = 0.2
for i, c in enumerate(CONDS):
    tm, ts = [], []
    for t in TOPICS:
        accs = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        tm.append(np.mean(accs) if accs else 0)
        ts.append(np.std(accs) / np.sqrt(len(accs)) if len(accs) > 1 else 0)
    ax.bar(x + i * width, tm, width, yerr=ts, capsize=3,
           label=COND_LABEL[c].replace('\n', ' '), color=colors[i], edgecolor='#334155', linewidth=0.6)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_ylim(0, 1); ax.set_ylabel('Proportion correct')
ax.set_title(f'Accuracy by topic × condition\n(attn-pass n={len(pass_recs)})')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / 'v5_accuracy_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v5_accuracy_by_condition.png')
plt.close()

# ── Figure 2: Per-question accuracy ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, topic in zip(axes.flat, TOPICS):
    cols = content_cols[topic]
    qlabels = [col_qid[c] for c in cols]
    means_pass = []
    for col in cols:
        scores_pass = [rec[topic]['scores'][cols.index(col)] for rec in pass_recs]
        means_pass.append(np.mean(scores_pass) if scores_pass else 0)
    x = np.arange(len(cols))
    ax.bar(x, means_pass, 0.5, color='#3b82f6', edgecolor='#334155', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(qlabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Proportion correct')
    ax.set_title(f'{TOPIC_LABEL[topic]} — per-question accuracy (n={len(pass_recs)})')
axes.flat[-1].axis('off')
plt.tight_layout()
plt.savefig(HERE / 'v5_per_question.png', dpi=150, bbox_inches='tight')
print('Saved: v5_per_question.png')
plt.close()

# ── Figure 2b: Per-question accuracy by condition ────────────────────────────
n_conds = len(CONDS)
width = 0.18
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
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
axes.flat[-1].axis('off')
plt.tight_layout()
plt.savefig(HERE / 'v5_per_question_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v5_per_question_by_condition.png')
plt.close()

# ── Violin helpers ────────────────────────────────────────────────────────────
topic_colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ec4899', '#a855f7']


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


# ── Figure 3a: Violin — accuracy by condition ────────────────────────────────
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
    ax.scatter([], [], s=18, color=topic_colors[ti], edgecolor='#334155', linewidth=0.5,
               label=TOPIC_LABEL[t])
ax.set_xticks(positions); ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by condition\n(attn-pass n={len(pass_recs)})')
ax.legend(fontsize=7, title='Topic', title_fontsize=7)
plt.tight_layout()
plt.savefig(HERE / 'v5_violin_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v5_violin_by_condition.png')
plt.close()

# ── Figure 3b: Violin — accuracy by topic × condition ────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
vwidth = 0.16
offsets_4 = np.array([-0.3, -0.1, 0.1, 0.3])
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        vals = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets_4[ci]
        if len(vals) > 1:
            parts = ax.violinplot(vals, positions=[pos], widths=vwidth,
                                  showmedians=True, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[ci]); parts['bodies'][0].set_alpha(0.7)
            parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(1.5)
        offs = dot_offsets(vals, dx=0.03)
        ax.scatter(np.full(len(vals), pos) + offs, vals,
                   s=14, color=colors[ci], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ci, c in enumerate(CONDS):
    ax.scatter([], [], s=18, color=colors[ci], edgecolor='#334155', linewidth=0.5,
               label=COND_LABEL[c].replace('\n', ' '))
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by topic × condition (attn-pass n={len(pass_recs)})')
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v5_violin_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v5_violin_by_topic.png')
plt.close()

# ── Figure 3c: Partial credit violin by topic × condition ────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        vals = [rec[t]['partial_acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets_4[ci]
        if len(vals) > 1:
            parts = ax.violinplot(vals, positions=[pos], widths=vwidth,
                                  showmedians=True, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[ci]); parts['bodies'][0].set_alpha(0.7)
            parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(1.5)
        offs = dot_offsets(vals, dx=0.03)
        ax.scatter(np.full(len(vals), pos) + offs, vals,
                   s=14, color=colors[ci], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ci, c in enumerate(CONDS):
    ax.scatter([], [], s=18, color=colors[ci], edgecolor='#334155', linewidth=0.5,
               label=COND_LABEL[c].replace('\n', ' '))
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('Partial credit score')
ax.set_title(f'Partial credit by topic × condition (n={len(pass_recs)})')
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v5_violin_partial_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v5_violin_partial_by_topic.png')
plt.close()

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

print('\n=== Duration (all finished-with-cond) ===')
durs = [r['duration'] for r in records]
print(f'  Mean: {np.mean(durs)/60:.1f} min  Median: {np.median(durs)/60:.1f} min  '
      f'Range: {min(durs)/60:.1f}–{max(durs)/60:.1f} min')
