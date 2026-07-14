#!/usr/bin/env python3
"""Analysis of multi_v4 pilot results."""
import csv, re, yaml
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA_FILE = HERE / 'multi_v4_July+8,+2026_12.08.tsv'

TOPICS = ['martial_arts', 'fruits_v2', 'astronomy', 'fabrics']
TOPIC_LABEL = {
    'martial_arts': 'Martial Arts', 'fruits_v2': 'Fruits v2',
    'astronomy': 'Astronomy', 'fabrics': 'Fabrics',
}
CONDS = ['control', 'repeat_short', 'repeat_long', 'distractor']
COND_LABEL = {
    'control': 'Control', 'repeat_short': 'Repeat\nShort',
    'repeat_long': 'Repeat\nLong', 'distractor': 'Distractor',
}

FL_TOPIC = {'FL_32': 'martial_arts', 'FL_33': 'fruits_v2', 'FL_34': 'astronomy', 'FL_35': 'fabrics'}

# ── Read TSV header to map each topic's content questions to actual columns ──
# (survey column codes can diverge from the YAML q_id, e.g. fabrics uses QM01.. not QFB01..)
with open(DATA_FILE, encoding='utf-16') as f:
    header = next(csv.reader(f, delimiter='\t'))

def topic_q_cols(topic):
    at_col = f'{topic}_AT_{topic}'
    pat = re.compile(rf'^{re.escape(topic)}_Q(?!.*_DO$)')
    cols = [h for h in header if pat.match(h) and not h.endswith('_DO') and h != at_col]
    return cols

# ── Load questions from YAML ──────────────────────────────────────────────────
col_correct = {}
col_options = {}
col_qid     = {}
content_cols = {}  # topic -> [tsv_col, ...]
attn_col     = {}  # topic -> tsv_col for attention check
attn_correct = {}  # topic -> frozenset of correct texts

for topic in TOPICS:
    with open(HERE / topic / 'questions.yaml') as f:
        qdata = yaml.safe_load(f)
    content_qs = [q for q in qdata['questions'] if q['metadata']['type'] == 'content']
    tsv_cols   = topic_q_cols(topic)
    assert len(tsv_cols) == len(content_qs), f'{topic}: {len(tsv_cols)} cols vs {len(content_qs)} questions'
    for tsvcol, q in zip(tsv_cols, content_qs):
        opts = q['options']
        ans  = q['answer'] if isinstance(q['answer'], list) else [q['answer']]
        col_correct[tsvcol] = frozenset(opts[a] for a in ans)
        col_options[tsvcol] = list(opts.values())
        col_qid[tsvcol]     = q['q_id']
    content_cols[topic] = tsv_cols

    at_q = next(q for q in qdata['questions'] if q['metadata']['type'] == 'attention_check')
    attn_col[topic]     = f'{topic}_AT_{topic}'
    ans = at_q['answer'] if isinstance(at_q['answer'], list) else [at_q['answer']]
    attn_correct[topic] = frozenset(at_q['options'][a] for a in ans)

# Override: multi_v4 was administered with "meters" wording in QF_V2_04's options
# (units bug vs. control passage's "900 to 4000 feet", fixed in questions.yaml post-hoc).
# Score this dataset against what respondents actually saw.
QF_V2_04_COL = 'fruits_v2_QF_V2_04'
col_options[QF_V2_04_COL] = [o.replace('feet', 'meters') for o in col_options[QF_V2_04_COL]]
col_correct[QF_V2_04_COL] = frozenset(o.replace('feet', 'meters') for o in col_correct[QF_V2_04_COL])

print('Attention check correct answers:')
for t in TOPICS:
    print(f'  {attn_col[t]}: {attn_correct[t]}')
print()


def parse_response(raw, all_opts):
    """Parse a (possibly multi-select) response string into a frozenset of option texts."""
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


# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_FILE, encoding='utf-16') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)
data = rows[2:]  # skip description rows

records = []
for r in data:
    if r.get('Finished', '').strip() != 'True':
        continue

    attn_pass = True
    for t in TOPICS:
        col = attn_col[t]
        raw = r.get(col, '').strip()
        given = parse_response(raw, list(attn_correct[t]) + [])
        if given != attn_correct[t]:
            attn_pass = False

    fl_raw   = r.get('FL_27_DO', '').strip()
    fl_order = [FL_TOPIC.get(code) for code in fl_raw.split('|') if code in FL_TOPIC]

    rec = {'attn_pass': attn_pass, 'group': r.get('group', '?'), 'fl_order': fl_order,
           'duration': float(r.get('Duration (in seconds)', 0) or 0)}

    for topic in TOPICS:
        cols   = content_cols[topic]
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

        t_passage = float(r.get(f'{topic}_timing_passage_Page Submit', 0) or 0)
        t_qs      = float(r.get(f'{topic}_timing_qs_Page Submit', 0) or 0)

        d_passage = r.get(f'{topic}_difficulty_1', '').strip()
        d_qs      = r.get(f'{topic}_difficulty_2', '').strip()

        rec[topic] = {
            'cond':         r.get(f'{topic}_cond', '').strip(),
            'scores':       scores,
            'partial':      partial,
            'acc':          np.mean(scores),
            'partial_acc':  np.mean(partial),
            't_passage':    t_passage,
            't_qs':         t_qs,
            'diff_passage': float(d_passage) if d_passage else np.nan,
            'diff_qs':      float(d_qs) if d_qs else np.nan,
        }
    records.append(rec)

pass_recs = [r for r in records if r['attn_pass']]
print(f'Finished: {len(records)}, attention-pass: {len(pass_recs)}')
for rec in records:
    flags = {t: rec[t]['cond'][:4] for t in TOPICS}
    accs  = {t: f"{rec[t]['acc']:.2f}" for t in TOPICS}
    print(f"  {'P' if rec['attn_pass'] else 'F'} g={rec['group']:>2} | " +
          ' | '.join(f"{t[:3]}({flags[t]})={accs[t]}" for t in TOPICS))

colors       = ['#94a3b8', '#fb923c', '#f97316', '#a78bfa']
topic_colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ec4899']


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


# ── Figure 1: Accuracy by condition ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

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
ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8, label='Chance (1/32)')
ax.legend(fontsize=8)
for bar, m, se in zip(bars, means, sems):
    ax.text(bar.get_x() + bar.get_width() / 2, m + se + 0.02,
            f'{m:.2f}', ha='center', va='bottom', fontsize=9)

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
ax.legend(fontsize=8); ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(HERE / 'v4_accuracy_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v4_accuracy_by_condition.png')
plt.close()

# ── Figure 2: Per-question accuracy by condition ─────────────────────────────
n_conds = len(CONDS)
width   = 0.18
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, topic in zip(axes.flat, TOPICS):
    cols    = content_cols[topic]
    qlabels = [col_qid[c] for c in cols]
    x = np.arange(len(cols))
    for ci, c in enumerate(CONDS):
        recs_c = [rec for rec in pass_recs if rec[topic]['cond'] == c]
        means  = [np.mean([rec[topic]['scores'][qi] for rec in recs_c]) if recs_c else 0
                  for qi in range(len(cols))]
        offset = (ci - (n_conds - 1) / 2) * width
        ax.bar(x + offset, means, width, label=COND_LABEL[c].replace('\n', ' '),
               color=colors[ci], edgecolor='#334155', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(qlabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Proportion correct')
    ax.set_title(f'{TOPIC_LABEL[topic]} — per-question accuracy by condition')
    ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8)
    if topic == 'martial_arts':
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / 'v4_per_question.png', dpi=150, bbox_inches='tight')
print('Saved: v4_per_question.png')
plt.close()

# ── Figure 3: Partial credit by topic × condition ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(TOPICS)); width = 0.2
for i, c in enumerate(CONDS):
    means_pc = []
    for t in TOPICS:
        pcs = [rec[t]['partial_acc'] for rec in pass_recs if rec[t]['cond'] == c]
        means_pc.append(np.mean(pcs) if pcs else 0)
    ax.bar(x + i * width, means_pc, width, label=COND_LABEL[c].replace('\n', ' '),
           color=colors[i], edgecolor='#334155', linewidth=0.6)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS])
ax.set_ylim(0, 1); ax.set_ylabel('Partial credit score')
ax.set_title(f'Partial credit by topic × condition (attn-pass n={len(pass_recs)})')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(HERE / 'v4_partial_credit.png', dpi=150, bbox_inches='tight')
print('Saved: v4_partial_credit.png')
plt.close()

# ── Figure 4a: Violin — accuracy by condition ────────────────────────────────
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
ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8)
ax.legend(fontsize=8, title='Topic', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v4_violin_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: v4_violin_by_condition.png')
plt.close()

# ── Figure 4b: Violin — accuracy by topic × condition ────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
vwidth = 0.18; offsets_4 = np.array([-0.3, -0.1, 0.1, 0.3])
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        data = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets_4[ci]
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
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('Proportion correct (exact match)')
ax.set_title(f'Accuracy by topic × condition (attn-pass n={len(pass_recs)})')
ax.axhline(1 / 32, color='#94a3b8', linestyle='--', linewidth=0.8)
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v4_violin_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v4_violin_by_topic.png')
plt.close()

# ── Figure 5: Partial credit violin by topic × condition ─────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for ti, t in enumerate(TOPICS):
    for ci, c in enumerate(CONDS):
        data = [rec[t]['partial_acc'] for rec in pass_recs if rec[t]['cond'] == c]
        pos  = ti + offsets_4[ci]
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
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
ax.set_xlim(-0.6, len(TOPICS) - 0.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel('Partial credit score')
ax.set_title(f'Partial credit by topic × condition (n={len(pass_recs)})')
ax.legend(fontsize=8, title='Condition', title_fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v4_violin_partial_by_topic.png', dpi=150, bbox_inches='tight')
print('Saved: v4_violin_partial_by_topic.png')
plt.close()

# ── Figure 6: Viewing order per participant ──────────────────────────────────
topic_colors_map = {t: c for t, c in zip(TOPICS, topic_colors)}
COND_ABBREV = {'control': 'ctrl', 'repeat_short': 'r-sh',
               'repeat_long': 'r-lo', 'distractor': 'dist'}
ordered_recs = sorted(pass_recs,
                      key=lambda r: int(r['group']) if str(r['group']).isdigit() else 99)
n_part = len(ordered_recs)
fig, ax = plt.subplots(figsize=(6, max(4, n_part * 0.45 + 1.5)))
for pi, rec in enumerate(ordered_recs):
    y     = n_part - 1 - pi
    order = rec['fl_order']
    for pos, topic in enumerate(order):
        col  = topic_colors_map.get(topic, '#cccccc')
        rect = plt.Rectangle([pos, y - 0.4], 1, 0.8, color=col,
                              linewidth=0.8, edgecolor='#334155')
        ax.add_patch(rect)
        cond = rec.get(topic, {}).get('cond', '') if isinstance(rec.get(topic), dict) else ''
        ax.text(pos + 0.5, y + 0.12, TOPIC_LABEL.get(topic, topic)[:3],
                ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
        ax.text(pos + 0.5, y - 0.17, COND_ABBREV.get(cond, cond[:4]),
                ha='center', va='center', fontsize=6, color='white', alpha=0.9)
    ax.text(-0.15, y, f"g={rec['group']}", ha='right', va='center', fontsize=7, color='#64748b')
ax.set_xlim(-1.5, 4.2); ax.set_ylim(-0.7, n_part - 0.3)
ax.set_xticks([0.5, 1.5, 2.5, 3.5])
ax.set_xticklabels(['1st', '2nd', '3rd', '4th'], fontsize=10)
ax.set_yticks([]); ax.set_xlabel('Position in study')
ax.set_title(f'Topic viewing order per participant (n={len(pass_recs)})')
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
for t in TOPICS:
    ax.scatter([], [], s=60, color=topic_colors_map[t], label=TOPIC_LABEL[t],
               edgecolor='#334155', linewidth=0.5)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
plt.tight_layout()
plt.savefig(HERE / 'v4_viewing_order.png', dpi=150, bbox_inches='tight')
print('Saved: v4_viewing_order.png')
plt.close()

# ── Figure 7: Difficulty ratings by condition ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
diff_labels = ['Passage difficulty', 'Question difficulty']
diff_keys   = ['diff_passage', 'diff_qs']
for ax, label, key in zip(axes, diff_labels, diff_keys):
    x = np.arange(len(TOPICS)); width = 0.2
    for i, c in enumerate(CONDS):
        tm, ts = [], []
        for t in TOPICS:
            vals = [rec[t][key] for rec in pass_recs
                    if rec[t]['cond'] == c and not np.isnan(rec[t][key])]
            tm.append(np.mean(vals) if vals else 0)
            ts.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(x + i * width, tm, width, yerr=ts, capsize=3,
               label=COND_LABEL[c].replace('\n', ' '), color=colors[i],
               edgecolor='#334155', linewidth=0.6)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
    ax.set_ylim(0, 10); ax.set_ylabel('Rating (0–10)')
    ax.set_title(f'{label} by topic × condition')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v4_difficulty.png', dpi=150, bbox_inches='tight')
print('Saved: v4_difficulty.png')
plt.close()

# ── Figure 8: Timing by condition ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
time_labels = ['Passage reading time (s)', 'Question time (s)']
time_keys   = ['t_passage', 't_qs']
for ax, label, key in zip(axes, time_labels, time_keys):
    x = np.arange(len(TOPICS)); width = 0.2
    for i, c in enumerate(CONDS):
        tm, ts = [], []
        for t in TOPICS:
            vals = [rec[t][key] for rec in pass_recs if rec[t]['cond'] == c and rec[t][key] > 0]
            tm.append(np.mean(vals) if vals else 0)
            ts.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(x + i * width, tm, width, yerr=ts, capsize=3,
               label=COND_LABEL[c].replace('\n', ' '), color=colors[i],
               edgecolor='#334155', linewidth=0.6)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([TOPIC_LABEL[t] for t in TOPICS], rotation=15, ha='right')
    ax.set_ylabel(label); ax.set_title(f'{label} by topic × condition')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE / 'v4_timing.png', dpi=150, bbox_inches='tight')
print('Saved: v4_timing.png')
plt.close()

# ── Print summary ─────────────────────────────────────────────────────────────
print('\n=== Per-question accuracy (attn-pass) ===')
for topic in TOPICS:
    cols = content_cols[topic]
    print(f'\n{TOPIC_LABEL[topic]}:')
    for ci, col in enumerate(cols):
        scores = [rec[topic]['scores'][ci] for rec in pass_recs]
        qid    = col_qid[col]
        print(f'  {qid}: {np.mean(scores):.0%} ({sum(scores)}/{len(scores)})')

print('\n=== Mean accuracy by condition (attn-pass) ===')
for c in CONDS:
    accs = [rec[t]['acc'] for rec in pass_recs for t in TOPICS if rec[t]['cond'] == c]
    print(f'  {c}: {np.mean(accs):.2f} (n={len(accs)})')

print('\n=== Duration (attn-pass) ===')
durs = [r['duration'] for r in pass_recs]
print(f'  Mean: {np.mean(durs)/60:.1f} min  Median: {np.median(durs)/60:.1f} min  '
      f'Range: {min(durs)/60:.1f}–{max(durs)/60:.1f} min')
