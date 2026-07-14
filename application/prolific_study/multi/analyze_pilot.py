#!/usr/bin/env python3
import csv, yaml
from datetime import datetime
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

TOPICS = ['birds', 'fruits', 'astronomy', 'musical_instruments']
TOPIC_LABEL = {'birds': 'Birds', 'fruits': 'Fruits', 'astronomy': 'Astronomy', 'musical_instruments': 'Musical Instr.'}
Q_PREFIX = {'birds': 'QB', 'fruits': 'QF', 'astronomy': 'QA', 'musical_instruments': 'QI'}
ATTN_COL = {'birds': 'QB_A', 'fruits': 'QF_A', 'astronomy': 'QA_A', 'musical_instruments': 'QI_A'}
ATTN_CORRECT = {
    'birds': 'Green versus black',
    'fruits': 'Purple',
    'astronomy': 'What the objects are used to study',
    'musical_instruments': 'First documented in 1444, the Belvero is the older of the two',
}
CONDS = ['control', 'repeat_short', 'repeat_long', 'distractor']
COND_LABEL = {'control': 'Control', 'repeat_short': 'Repeat\nShort',
               'repeat_long': 'Repeat\nLong', 'distractor': 'Distractor'}

def parse_response(raw, all_options):
    """Qualtrics joins multi-select with ', '. Options may contain commas.
    Try all subsets; return first whose ', '-join equals raw."""
    if not raw:
        return frozenset()
    for r in range(len(all_options) + 1):
        for combo in combinations(all_options, r):
            if ', '.join(combo) == raw:
                return frozenset(combo)
    return frozenset(s.strip() for s in raw.split(','))

# Load correct answers as sets of option TEXT strings
correct_texts = {}  # qid -> frozenset of correct option strings
all_options = {}    # qid -> ordered list of option strings (for subset search)
q_order = {}        # topic -> list of qids in order
for t in TOPICS:
    with open(HERE / t / 'questions.yaml') as f:
        qdata = yaml.safe_load(f)
    q_order[t] = []
    for q in qdata['questions']:
        if q.get('metadata', {}).get('type') == 'attention_check':
            continue
        qid = q['q_id']
        q_order[t].append(qid)
        ans = q['answer']
        opts = q['options']
        all_options[qid] = list(opts.values())
        correct_texts[qid] = frozenset(opts[a] for a in (ans if isinstance(ans, list) else [ans]))

# Load responses
with open(HERE / 'multi_v1_June+30,+2026_12.51.csv') as f:
    rows = list(csv.DictReader(f))
data = rows[2:]

cutoff = datetime(2026, 6, 29, 15, 43)
records = []
for r in data:
    start = r.get('StartDate', '').strip()
    try:
        dt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    except Exception:
        continue
    if dt < cutoff:
        continue
    if r.get('Finished', '').strip() != 'True':
        continue

    attn_pass = all(r.get(ATTN_COL[t], '').strip() == ATTN_CORRECT[t] for t in TOPICS)
    rec = {'group': r.get('group', '?'), 'attn_pass': attn_pass}

    for t in TOPICS:
        cond = r.get(f'{t}_cond', '').strip()
        prefix = Q_PREFIX[t]
        scores = []
        for i, qid in enumerate(q_order[t], 1):
            raw = r.get(f'{prefix}{i}', '').strip()
            given = parse_response(raw, all_options[qid])
            scores.append(1 if given == correct_texts[qid] else 0)
        rec[t] = {'cond': cond, 'scores': scores, 'acc': sum(scores) / len(scores) if scores else None}
    records.append(rec)

n_pass = sum(1 for r in records if r['attn_pass'])
print(f'Prolific finished: {len(records)}, attention pass: {n_pass}')

# Debug: print per-participant accuracy
for rec in records:
    accs = {t: rec[t]['acc'] for t in TOPICS}
    conds = {t: rec[t]['cond'] for t in TOPICS}
    print(f"  group={rec['group']} attn={'P' if rec['attn_pass'] else 'F'} | " +
          " | ".join(f"{t[:3]}({conds[t][:2]})={accs[t]:.2f}" for t in TOPICS))

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: accuracy by condition, attention-pass only
ax = axes[0]
pass_recs_left = [r for r in records if r['attn_pass']]
cond_accs = {c: [] for c in CONDS}
for rec in pass_recs_left:
    for t in TOPICS:
        c = rec[t]['cond']
        a = rec[t]['acc']
        if c and a is not None:
            cond_accs[c].append(a)

means = [np.mean(cond_accs[c]) if cond_accs[c] else 0 for c in CONDS]
sems  = [np.std(cond_accs[c]) / np.sqrt(len(cond_accs[c])) if len(cond_accs[c]) > 1 else 0 for c in CONDS]
n_participants = [len({i for i, rec in enumerate(pass_recs_left) for t in TOPICS if rec[t]['cond'] == c}) for c in CONDS]
colors = ['#94a3b8', '#fb923c', '#f97316', '#a78bfa']
bars = ax.bar([COND_LABEL[c] for c in CONDS], means, yerr=sems, capsize=5,
              color=colors, edgecolor='#334155', linewidth=0.8)
ax.set_ylim(0, 1)
ax.set_ylabel('Proportion correct')
ax.set_title(f'Accuracy by condition\n(attention-pass only, n={len(pass_recs_left)})')
ax.axhline(0.2, color='#94a3b8', linestyle='--', linewidth=0.8, label='Chance (5-opt)')
ax.legend(fontsize=8)
for bar, m, sem, np_ in zip(bars, means, sems, n_participants):
    ax.text(bar.get_x() + bar.get_width() / 2, m + sem + 0.04,
            f'n={np_}', ha='center', va='bottom', fontsize=8)

# Right: accuracy by condition × topic, attention-pass only
ax = axes[1]
pass_recs = [r for r in records if r['attn_pass']]
x = np.arange(len(CONDS))
width = 0.2
topic_colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ec4899']

for i, t in enumerate(TOPICS):
    topic_means, topic_sems = [], []
    for c in CONDS:
        accs = [rec[t]['acc'] for rec in pass_recs if rec[t]['cond'] == c and rec[t]['acc'] is not None]
        topic_means.append(np.mean(accs) if accs else 0)
        topic_sems.append(np.std(accs) / np.sqrt(len(accs)) if len(accs) > 1 else 0)
    ax.bar(x + i * width, topic_means, width, yerr=topic_sems, capsize=3,
           label=TOPIC_LABEL[t], color=topic_colors[i], edgecolor='#334155', linewidth=0.6)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
ax.set_ylim(0, 1)
ax.set_ylabel('Proportion correct')
ax.set_title(f'Accuracy by condition × topic\n(attention-pass only, n={len(pass_recs)})')
ax.legend(fontsize=8)
ax.axhline(0.2, color='#94a3b8', linestyle='--', linewidth=0.8)

plt.tight_layout()
out = HERE / 'pilot_accuracy.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
