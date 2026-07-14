#!/usr/bin/env python3
"""Per-option selection frequency for fabrics content questions, by condition (attn-pass, multi_v4)."""
import csv, re, yaml
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA_FILE = HERE / 'multi_v4_July+8,+2026_12.08.tsv'
TOPIC = 'fabrics'

with open(DATA_FILE, encoding='utf-16') as f:
    header = next(csv.reader(f, delimiter='\t'))

def topic_q_cols(topic):
    at_col = f'{topic}_AT_{topic}'
    pat = re.compile(rf'^{re.escape(topic)}_Q(?!.*_DO$)')
    return [h for h in header if pat.match(h) and not h.endswith('_DO') and h != at_col]

with open(HERE / TOPIC / 'questions.yaml') as f:
    qdata = yaml.safe_load(f)
content_qs = [q for q in qdata['questions'] if q['metadata']['type'] == 'content']
tsv_cols = topic_q_cols(TOPIC)

ALL_TOPICS = ['martial_arts', 'fruits_v2', 'astronomy', 'fabrics']
all_attn = {}
for t in ALL_TOPICS:
    with open(HERE / t / 'questions.yaml') as f:
        d = yaml.safe_load(f)
    q = next(x for x in d['questions'] if x['metadata']['type'] == 'attention_check')
    ans = q['answer'] if isinstance(q['answer'], list) else [q['answer']]
    all_attn[t] = (f'{t}_AT_{t}', frozenset(q['options'][a] for a in ans))


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
data = rows[2:]

pass_rows = []
for r in data:
    if r.get('Finished', '').strip() != 'True':
        continue
    ok = True
    for t, (col, correct) in all_attn.items():
        raw = r.get(col, '').strip()
        given = parse_response(raw, list(correct))
        if given != correct:
            ok = False
    if ok:
        pass_rows.append(r)

print(f'attn-pass n={len(pass_rows)}')

CONDS = ['control', 'repeat_short', 'repeat_long', 'distractor']
COND_COLORS = {'control': '#94a3b8', 'repeat_short': '#fb923c', 'repeat_long': '#f97316', 'distractor': '#a78bfa'}
COND_COL = f'{TOPIC}_cond'

fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for ax, tsvcol, q in zip(axes, tsv_cols, content_qs):
    opts = q['options']
    all_texts = list(opts.values())
    ans = q['answer'] if isinstance(q['answer'], list) else [q['answer']]

    counts = {c: {i: 0 for i in opts} for c in CONDS}
    ns = {c: 0 for c in CONDS}
    for r in pass_rows:
        c = r.get(COND_COL, '').strip()
        raw = r.get(tsvcol, '').strip()
        if c not in CONDS or not raw:
            continue
        given = parse_response(raw, all_texts)
        ns[c] += 1
        for i, text in opts.items():
            if text in given:
                counts[c][i] += 1

    x = np.arange(len(opts)); width = 0.2
    for ci, c in enumerate(CONDS):
        vals = [counts[c][i] / ns[c] if ns[c] else 0 for i in opts]
        ax.bar(x + (ci - 1.5) * width, vals, width, label=c, color=COND_COLORS[c],
               edgecolor='#334155', linewidth=0.5)
    for i in opts:
        if i in ans:
            ax.axvspan(list(opts).index(i) - 0.5, list(opts).index(i) + 0.5, color='#22c55e', alpha=0.12, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in opts])
    ax.set_ylim(0, 1)
    ax.set_title(f"{q['q_id']}\n{q['question'][:45]}{'...' if len(q['question'])>45 else ''}", fontsize=9)
    ax.set_ylabel('Fraction selecting option')

axes[0].legend(fontsize=7, title='Condition', title_fontsize=7)
plt.suptitle(f'Fabrics — option selection frequency by condition (attn-pass n={len(pass_rows)}; '
             f'green band = correct option)', y=1.04)
plt.tight_layout()
plt.savefig(HERE / 'fabrics_mistakes_by_condition.png', dpi=150, bbox_inches='tight')
print('Saved: fabrics_mistakes_by_condition.png')
