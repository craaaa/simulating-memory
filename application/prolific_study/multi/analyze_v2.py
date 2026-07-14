#!/usr/bin/env python3
"""Analysis of multi_v2 pilot results."""
import csv, yaml
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA_FILE = HERE / 'multi_v2_July+1,+2026_16.57.tsv'

TOPICS = ['birds', 'fruits', 'astronomy', 'musical_instruments']
TOPIC_LABEL = {
    'birds': 'Birds', 'fruits': 'Fruits',
    'astronomy': 'Astronomy', 'musical_instruments': 'Musical Instr.'
}
CONDS = ['control', 'repeat_short', 'repeat_long', 'distractor']
COND_LABEL = {
    'control': 'Control', 'repeat_short': 'Repeat\nShort',
    'repeat_long': 'Repeat\nLong', 'distractor': 'Distractor'
}

# Column prefix for content questions and attention checks
Q_PREFIX   = {'birds': 'QB', 'fruits': 'QF', 'astronomy': 'QA', 'musical_instruments': 'QM'}
ATTN_COL   = {'birds': 'QB_A', 'fruits': 'QF_A', 'astronomy': 'QA_A', 'musical_instruments': 'QI_A'}
ATTN_ANS   = {
    'birds':               'Green versus black',
    'fruits':              'Purple',
    'astronomy':           'What the objects are used to study',
    'musical_instruments': 'One instrument is plucked with fingers, the other with a pick',
}
COND_COL   = {t: f'{t}_cond' for t in TOPICS}
DIFF_COLS  = {
    'birds':               ('QB_passage_1', 'QB_passage_2'),
    'fruits':              ('QF_passage_1', 'QF_passage_2'),
    'astronomy':           ('QA_passage_1', 'QA_passage_2'),
    'musical_instruments': ('QI_passage_1', 'QI_passage_2'),
}

# Columns present in this export for each topic (QA9 absent from survey)
PRESENT_COLS = {
    'birds':               ['QB1','QB2','QB3','QB4','QB5','QB6','QB7','QB8'],
    'fruits':              ['QF1','QF2','QF3','QF4','QF5','QF6','QF7','QF8'],
    'astronomy':           ['QA1','QA2','QA3','QA4','QA5','QA6','QA7','QA8'],
    'musical_instruments': ['QM1','QM2','QM3','QM4','QM5','QM6','QM7','QM8'],
}

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

# Load correct answers from YAML
# Maps column name (e.g. 'QB1') -> (frozenset of correct texts, list of all option texts)
col_correct = {}
col_options = {}
col_qid     = {}   # col -> q_id
for topic in TOPICS:
    with open(HERE / topic / 'questions.yaml') as f:
        qdata = yaml.safe_load(f)
    content_qs = [q for q in qdata['questions'] if q['metadata']['type'] == 'content']
    cols = PRESENT_COLS[topic]
    for i, col in enumerate(cols):
        if i >= len(content_qs):
            break
        q = content_qs[i]
        opts = q['options']
        ans  = q['answer'] if isinstance(q['answer'], list) else [q['answer']]
        col_correct[col] = frozenset(opts[a] for a in ans)
        col_options[col] = list(opts.values())
        col_qid[col]     = q['q_id']

# Override: survey options differ from local YAML for QB8, QA8, QM8
# Determined from actual response data in multi_v2 export
QB8_OPTS = ['Its scientific name is Fringilla rufa', 'It eats fruits and seeds',
            'It has gray feathers', 'Its call is a sharp, descending whistle', 'None of the above']
col_options['QB8']  = QB8_OPTS
col_correct['QB8']  = frozenset(['Its scientific name is Fringilla rufa', 'It eats fruits and seeds'])
col_qid['QB8']      = 'QB08-survey'

QA8_OPTS = ['Through telescopes sensitive to red-light emission', 'Using radio equipment',
            'By tracking how its brightness fluctuates over time', 'With the naked eye on a clear night', 'None of the above']
col_options['QA8']  = QA8_OPTS
col_correct['QA8']  = frozenset(['Through telescopes sensitive to red-light emission'])
col_qid['QA8']      = 'QA08-survey'

QM8_OPTS = ['The Belvero was documented by a luthier based in Spain',
            'The Belvero was documented in the early seventeenth century',
            'The Belvero was documented by a Bavarian instrument-maker',
            'The Belvero was documented in the early nineteenth century', 'None of the above']
col_options['QM8']  = QM8_OPTS
col_correct['QM8']  = frozenset(['The Belvero was documented by a luthier based in Spain',
                                  'The Belvero was documented in the early seventeenth century'])
col_qid['QM8']      = 'QM08-survey'

# Load data
with open(DATA_FILE, encoding='utf-16') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)
data = rows[2:]  # skip description rows

FL_TOPIC = {'FL_32': 'birds', 'FL_33': 'fruits', 'FL_34': 'astronomy', 'FL_35': 'musical_instruments'}

records = []
for r in data:
    if r.get('Finished','').strip() != 'True':
        continue
    attn = all(r.get(ATTN_COL[t],'').strip() == ATTN_ANS[t] for t in TOPICS)
    fl_raw = r.get('FL_27_DO', '').strip()
    fl_order = [FL_TOPIC.get(code) for code in fl_raw.split('|') if code in FL_TOPIC]
    rec = {'attn_pass': attn, 'group': r.get('group','?'), 'fl_order': fl_order}
    for topic in TOPICS:
        cond = r.get(COND_COL[topic],'').strip()
        cols = PRESENT_COLS[topic]
        scores, partial = [], []
        for col in cols:
            raw   = r.get(col,'').strip()
            given = parse_response(raw, col_options[col])
            exact = 1 if given == col_correct[col] else 0
            n_correct_opts = len(col_correct[col])
            n_given_correct = len(given & col_correct[col])
            n_given_wrong   = len(given - col_correct[col])
            pc = max(0, (n_given_correct - n_given_wrong) / n_correct_opts) if n_correct_opts else 0
            scores.append(exact)
            partial.append(pc)
        pcol, qcol = DIFF_COLS[topic]
        try:
            p_diff = float(r.get(pcol, '').strip()) if r.get(pcol, '').strip() else None
            q_diff = float(r.get(qcol, '').strip()) if r.get(qcol, '').strip() else None
        except ValueError:
            p_diff = q_diff = None
        rec[topic] = {
            'cond': cond,
            'scores': scores,
            'partial': partial,
            'acc': np.mean(scores),
            'partial_acc': np.mean(partial),
            'passage_diff': p_diff,
            'qs_diff': q_diff,
        }
    records.append(rec)

pass_recs = [r for r in records if r['attn_pass']]
print(f'Finished: {len(records)}, attention-pass: {len(pass_recs)}')

for rec in records:
    flags = {t: rec[t]['cond'][:4] for t in TOPICS}
    accs  = {t: f"{rec[t]['acc']:.2f}" for t in TOPICS}
    print(f"  {'P' if rec['attn_pass'] else 'F'} g={rec['group']:>2} | " +
          " | ".join(f"{t[:3]}({flags[t]})={accs[t]}" for t in TOPICS))

# ── Figure 1: Accuracy by condition ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#94a3b8', '#fb923c', '#f97316', '#a78bfa']

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
ax.axhline(1/32, color='#94a3b8', linestyle='--', linewidth=0.8, label='Chance (1/32)')
ax.legend(fontsize=8)
for bar, m, se in zip(bars, means, sems):
    ax.text(bar.get_x() + bar.get_width()/2, m + se + 0.02,
            f'{m:.2f}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
x = np.arange(len(TOPICS)); width = 0.2
for i, c in enumerate(CONDS):
    tm, ts = [], []
    for t in TOPICS:
        accs = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        tm.append(np.mean(accs) if accs else 0)
        ts.append(np.std(accs)/np.sqrt(len(accs)) if len(accs) > 1 else 0)
    ax.bar(x + i*width, tm, width, yerr=ts, capsize=3,
           label=COND_LABEL[c].replace('\n',' '), color=colors[i], edgecolor='#334155', linewidth=0.6)
ax.set_xticks(x + width*1.5); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_ylim(0, 1); ax.set_ylabel('Proportion correct')
ax.set_title(f'Accuracy by topic × condition\n(attn-pass n={len(pass_recs)})')
ax.legend(fontsize=8); ax.axhline(1/32, color='#94a3b8', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(HERE / 'v2_accuracy_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v2_accuracy_by_condition.png')
plt.close()

# ── Figure 2: Per-question accuracy ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, topic in zip(axes.flat, TOPICS):
    cols = PRESENT_COLS[topic]
    n_orig = 5  # first 5 are original questions
    q_labels = [col_qid.get(c, c) for c in cols]
    means_all, means_pass = [], []
    for col in cols:
        scores_all  = []
        scores_pass = []
        for rec in records:
            raw   = None
            # find raw from records
        # Recalculate from pass_recs
        for rec in records:
            scores_all.append(rec[topic]['scores'][cols.index(col)])
        for rec in pass_recs:
            scores_pass.append(rec[topic]['scores'][cols.index(col)])
        means_all.append(np.mean(scores_all) if scores_all else 0)
        means_pass.append(np.mean(scores_pass) if scores_pass else 0)

    x = np.arange(len(cols))
    ax.bar(x[:n_orig] - 0.2, means_all[:n_orig],   0.35, label='All', color='#94a3b8', edgecolor='#334155', linewidth=0.6)
    ax.bar(x[:n_orig] + 0.15, means_pass[:n_orig],  0.35, label='Attn-pass', color='#3b82f6', edgecolor='#334155', linewidth=0.6)
    ax.bar(x[n_orig:] - 0.2, means_all[n_orig:],   0.35, color='#fca5a5', edgecolor='#334155', linewidth=0.6)
    ax.bar(x[n_orig:] + 0.15, means_pass[n_orig:], 0.35, color='#f97316', edgecolor='#334155', linewidth=0.6)
    ax.axvline(n_orig - 0.5, color='#64748b', linestyle=':', linewidth=1)
    ax.text(n_orig - 0.5, 0.97, 'new →', ha='left', va='top', fontsize=8, color='#64748b')
    ax.set_xticks(x); ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Proportion correct')
    ax.set_title(f'{TOPIC_LABEL[topic]} — per-question accuracy')
    ax.axhline(1/32, color='#94a3b8', linestyle='--', linewidth=0.8)
    if topic == 'birds':
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / 'v2_per_question.png', dpi=150, bbox_inches='tight')
print('Saved: v2_per_question.png')
plt.close()

# ── Figure 3: Difficulty (partial credit) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(TOPICS)); width = 0.2
for i, c in enumerate(CONDS):
    means_pc = []
    for t in TOPICS:
        pcs = [rec[t]['partial_acc'] for rec in pass_recs if rec[t]['cond'] == c]
        means_pc.append(np.mean(pcs) if pcs else 0)
    ax.bar(x + i*width, means_pc, width, label=COND_LABEL[c].replace('\n',' '),
           color=colors[i], edgecolor='#334155', linewidth=0.6)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS])
ax.set_ylim(0, 1); ax.set_ylabel('Partial credit score')
ax.set_title(f'Partial credit by topic × condition (attn-pass n={len(pass_recs)})')
ax.legend(fontsize=9); ax.axhline(0.5, color='#94a3b8', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(HERE / 'v2_partial_credit.png', dpi=150, bbox_inches='tight')
print('Saved: v2_partial_credit.png')
plt.close()

# ── Figure 4a: Violin — accuracy by condition ────────────────────────────────
colors = ['#94a3b8', '#fb923c', '#f97316', '#a78bfa']
topic_colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ec4899']

def dot_offsets(ys, dy=0.07, dx=0.04):
    """Stack dots horizontally within y-bins (vertical histogram style)."""
    from collections import defaultdict
    bins = defaultdict(list)
    for i, y in enumerate(ys):
        bins[round(float(y) / dy)].append(i)
    offsets = np.zeros(len(ys))
    for idxs in bins.values():
        n = len(idxs)
        xs = np.arange(n) * dx - (n - 1) * dx / 2
        for k, idx in enumerate(idxs):
            offsets[idx] = xs[k]
    return offsets

fig4a, ax = plt.subplots(figsize=(7, 5))
cond_data  = {c: [] for c in CONDS}
cond_tcols = {c: [] for c in CONDS}
for rec in pass_recs:
    for ti, t in enumerate(TOPICS):
        c = rec[t]['cond']
        if c:
            cond_data[c].append(rec[t]['acc'])
            cond_tcols[c].append(topic_colors[ti])
positions = list(range(len(CONDS)))
parts = ax.violinplot([cond_data[c] for c in CONDS], positions=positions,
                      showmedians=True, showextrema=False)
for pc, col in zip(parts['bodies'], colors):
    pc.set_facecolor(col); pc.set_alpha(0.7)
parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(2)
for i, c in enumerate(CONDS):
    offs = dot_offsets(cond_data[c])
    ax.scatter(np.full(len(cond_data[c]), i) + offs, cond_data[c],
               s=18, color=cond_tcols[c], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ti, t in enumerate(TOPICS):
    ax.scatter([], [], s=18, color=topic_colors[ti], edgecolor='#334155', linewidth=0.5, label=TOPIC_LABEL[t])
ax.set_xticks(positions)
ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by condition\n(attn-pass n={len(pass_recs)})')
ax.axhline(1/32, color='#94a3b8', linestyle='--', linewidth=0.8)
ax.legend(fontsize=8, title='Topic', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v2_violin_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v2_violin_by_condition.png')
plt.close()

# ── Figure 4b: Violin — accuracy by topic × condition ────────────────────────
fig4b, ax = plt.subplots(figsize=(9, 5))
vwidth = 0.18
offsets = np.array([-0.3, -0.1, 0.1, 0.3])
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        data = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets[ci]
        if len(data) > 1:
            parts = ax.violinplot(data, positions=[pos], widths=vwidth,
                                  showmedians=True, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[ci]); parts['bodies'][0].set_alpha(0.7)
            parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(1.5)
        offs = dot_offsets(data, dx=0.03)
        ax.scatter(np.full(len(data), pos) + offs, data,
                   s=14, color=colors[ci], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ci, c in enumerate(CONDS):
    ax.scatter([], [], s=18, color=colors[ci], edgecolor='#334155', linewidth=0.5,
               label=COND_LABEL[c].replace('\n', ' '))
ax.set_xticks(range(len(TOPICS)))
ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4)
ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by topic × condition (attn-pass n={len(pass_recs)})')
ax.axhline(1/32, color='#94a3b8', linestyle='--', linewidth=0.8)
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v2_violin_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v2_violin_by_topic.png')
plt.close()

# ── Figure 5: Partial credit violin by topic × condition ─────────────────────
fig5, ax = plt.subplots(figsize=(9, 5))
vwidth = 0.18
offsets = np.array([-0.3, -0.1, 0.1, 0.3])
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        data = [rec[t]['partial_acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets[ci]
        if len(data) > 1:
            parts = ax.violinplot(data, positions=[pos], widths=vwidth,
                                  showmedians=True, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[ci]); parts['bodies'][0].set_alpha(0.7)
            parts['cmedians'].set_color('#334155'); parts['cmedians'].set_linewidth(1.5)
        offs = dot_offsets(data, dx=0.03)
        ax.scatter(np.full(len(data), pos) + offs, data,
                   s=14, color=colors[ci], edgecolor='#334155', linewidth=0.5, zorder=3, alpha=0.8)
for ci, c in enumerate(CONDS):
    ax.scatter([], [], s=18, color=colors[ci], edgecolor='#334155', linewidth=0.5,
               label=COND_LABEL[c].replace('\n', ' '))
ax.set_xticks(range(len(TOPICS)))
ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4)
ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Partial credit score')
ax.set_title(f'Partial credit by topic × condition (n={len(pass_recs)})')
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v2_violin_partial_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v2_violin_partial_by_topic.png')
plt.close()

# ── Figure 6: Viewing order per participant ──────────────────────────────────
topic_colors_map = {t: c for t, c in zip(TOPICS, topic_colors)}
ordered_recs = sorted(records, key=lambda r: (not r['attn_pass'], int(r['group']) if str(r['group']).isdigit() else 99))
n_part = len(ordered_recs)
fig6, ax = plt.subplots(figsize=(6, max(4, n_part * 0.45 + 1.5)))

for pi, rec in enumerate(ordered_recs):
    y = n_part - 1 - pi  # top = first participant
    order = rec['fl_order']
    COND_ABBREV = {'control': 'ctrl', 'repeat_short': 'r-sh',
                   'repeat_long': 'r-lo', 'distractor': 'dist'}
    for pos, topic in enumerate(order):
        col = topic_colors_map.get(topic, '#cccccc')
        rect = plt.Rectangle([pos, y - 0.4], 1, 0.8, color=col,
                              linewidth=0.8, edgecolor='#334155')
        ax.add_patch(rect)
        cond = rec.get(topic, {}).get('cond', '') if isinstance(rec.get(topic), dict) else ''
        cond_abbrev = COND_ABBREV.get(cond, cond[:4])
        ax.text(pos + 0.5, y + 0.12, TOPIC_LABEL.get(topic, topic)[:3],
                ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
        ax.text(pos + 0.5, y - 0.17, cond_abbrev,
                ha='center', va='center', fontsize=6, color='white', alpha=0.9)
    # mark attn-pass with a star on left
    marker = '★' if rec['attn_pass'] else '☆'
    ax.text(-0.15, y, marker, ha='right', va='center', fontsize=9,
            color='#334155' if rec['attn_pass'] else '#94a3b8')
    label = f"g={rec['group']}"
    ax.text(-0.25, y, label, ha='right', va='center', fontsize=7, color='#64748b')

ax.set_xlim(-1.5, 4.2)
ax.set_ylim(-0.7, n_part - 0.3)
ax.set_xticks([0.5, 1.5, 2.5, 3.5])
ax.set_xticklabels(['1st', '2nd', '3rd', '4th'], fontsize=10)
ax.set_yticks([])
ax.set_xlabel('Position in study')
ax.set_title(f'Topic viewing order per participant\n(★ = attention-pass, n={len(pass_recs)}/{n_part})')
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# legend
for t in TOPICS:
    ax.scatter([], [], s=60, color=topic_colors_map[t], label=TOPIC_LABEL[t],
               edgecolor='#334155', linewidth=0.5)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig(HERE / 'v2_viewing_order.png', dpi=150, bbox_inches='tight')
print('Saved: v2_viewing_order.png')
plt.close()

# ── Figure 7: Perceived difficulty by topic × condition ──────────────────────
fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5))
diff_labels = ['Passage', 'Questions']
diff_keys   = ['passage_diff', 'qs_diff']

for ax, dkey, dlabel in zip(axes7, diff_keys, diff_labels):
    x = np.arange(len(TOPICS))
    for ci, c in enumerate(CONDS):
        means_d, sems_d = [], []
        for t in TOPICS:
            vals = [rec[t][dkey] for rec in pass_recs
                    if rec[t]['cond'] == c and rec[t][dkey] is not None]
            means_d.append(np.mean(vals) if vals else 0)
            sems_d.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(x + ci * width, means_d, width, yerr=sems_d, capsize=3,
               label=COND_LABEL[c].replace('\n', ' '), color=colors[ci],
               edgecolor='#334155', linewidth=0.6)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
    ax.set_ylim(0, 10)
    ax.set_ylabel('Difficulty rating (0–10)')
    ax.set_title(f'{dlabel} difficulty by topic × condition\n(attn-pass n={len(pass_recs)})')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / 'v2_difficulty.png', dpi=150, bbox_inches='tight')
print('Saved: v2_difficulty.png')
plt.close()

# ── Print summary table ───────────────────────────────────────────────────────
print('\n=== Per-question accuracy (attn-pass) ===')
for topic in TOPICS:
    cols = PRESENT_COLS[topic]
    print(f'\n{TOPIC_LABEL[topic]}:')
    for col in cols:
        scores = [rec[topic]['scores'][cols.index(col)] for rec in pass_recs]
        qid = col_qid.get(col, col)
        new_marker = ' *NEW*' if cols.index(col) >= 5 else ''
        print(f'  {col} ({qid}){new_marker}: {np.mean(scores):.0%} ({sum(scores)}/{len(scores)})')
