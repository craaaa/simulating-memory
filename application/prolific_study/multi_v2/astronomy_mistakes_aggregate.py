#!/usr/bin/env python3
"""Per-option selection frequency for astronomy content questions (attn-pass, multi_v4)."""
import csv, re, yaml
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA_FILE = HERE / 'multi_v4_July+8,+2026_12.08.tsv'
TOPIC = 'astronomy'

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

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, tsvcol, q in zip(axes, tsv_cols, content_qs):
    opts = q['options']
    all_texts = list(opts.values())
    ans = q['answer'] if isinstance(q['answer'], list) else [q['answer']]
    correct = frozenset(opts[a] for a in ans)

    counts = {i: 0 for i in opts}
    n = 0
    for r in pass_rows:
        raw = r.get(tsvcol, '').strip()
        if not raw:
            continue
        given = parse_response(raw, all_texts)
        n += 1
        for i, text in opts.items():
            if text in given:
                counts[i] += 1

    labels = [f'{i}' for i in opts]
    vals = [counts[i] / n if n else 0 for i in opts]
    bar_colors = ['#22c55e' if i in ans else '#f87171' for i in opts]
    ax.bar(labels, vals, color=bar_colors, edgecolor='#334155', linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_title(f"{q['q_id']}\n{q['question'][:45]}{'...' if len(q['question'])>45 else ''}", fontsize=9)
    ax.set_ylabel('Fraction selecting option')
    for i, v in zip(labels, vals):
        ax.text(int(i) - 1, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

axes[0].legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color='#22c55e', label='Correct option'),
    plt.Rectangle((0, 0), 1, 1, color='#f87171', label='Wrong option'),
], fontsize=8, loc='upper right')

plt.suptitle(f'Astronomy — option selection frequency per question (attn-pass n={len(pass_rows)})', y=1.03)
plt.tight_layout()
plt.savefig(HERE / 'astronomy_mistakes.png', dpi=150, bbox_inches='tight')
print('Saved: astronomy_mistakes.png')
