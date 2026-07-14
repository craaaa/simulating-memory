"""
Pilot v4 analysis — plotting script.
Modeled on the v3 analyze_pilot_v3.ipynb structure.
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/Users/cl5625/simulating-memory/.claude/worktrees/compare-task/application/prolific_study/pilot_v4')
CSV_PATH = HERE / 'results.csv'

# ── Conditions ──────────────────────────────────────────────────────────────
CONDITIONS = ['control', 'repeat_short', 'repeat_long', 'distractor']
COND_COLORS = {
    'control':      '#1f77b4',
    'repeat_short': '#2ca02c',
    'repeat_long':  '#ff7f0e',
    'distractor':   '#d62728',
}
COND_LABELS = {
    'control':      'control',
    'repeat_short': 'repeat_short',
    'repeat_long':  'repeat_long',
    'distractor':   'distractor',
}

# ── Question definitions ─────────────────────────────────────────────────────
# Each question: col name, short title, options dict {1: text, ...}, correct set
QUESTIONS = [
    {
        'col':     'QP01',
        'qid':     'QP01',
        'title':   "Ashen Skreel's plumage?",
        'question': "Which of the following describe the Ashen Skreel's plumage?",
        'options': {
            1: "Mostly gray, with a lighter patch at the throat",
            2: "Rust orange, with a lighter patch at the throat",
            3: "Faint streaking along the back",
            4: "A reddish wash across the breast",
            5: "None of the above",
        },
        'answer': {1},
    },
    {
        'col':     'QP02',
        'qid':     'QP02',
        'title':   "Copperhook Finch diet?",
        'question': "Which of the following does the Copperhook Finch eat?",
        'options': {
            1: "Fruit",
            2: "Seeds",
            3: "Grubs",
            4: "None of the above",
        },
        'answer': {1, 2, 3},
    },
    {
        'col':     'QP03',
        'qid':     'QP03',
        'title':   "Difference between the two birds?",
        'question': "Which statements correctly describe a difference between the two birds?",
        'options': {
            1: "The ashen skreel eats carrion, while the Copperhook finch eats seeds",
            2: "The ashen skreel is mostly gray, whereas the Copperhook finch is mostly orange",
            3: "The Copperhook finch eats seeds, unlike the fruit-eating ashen skreel",
            4: "The ashen skreel has black wingtips, while the Copperhook finch has a pale throat",
            5: "None of the above",
        },
        'answer': {1, 2},
    },
    {
        'col':     'QP04',
        'qid':     'QP04',
        'title':   "Bird matched to Latin name?",
        'question': "Which of the following correctly matches the bird to its Latin name?",
        'options': {
            1: "Ashen skreel -- Fringilla rufa",
            2: "Ashen skreel -- Plumbea rufa",
            3: "Copperhook finch -- Fringilla vorax",
            4: "Copperhook finch -- Plumbea rufa",
            5: "None of the above",
        },
        'answer': {5},
    },
    {
        'col':     'QP05',
        'qid':     'QP05',
        'title':   "Copperhook Finch's plumage?",
        'question': "Which of the following describe the Copperhook Finch's plumage?",
        'options': {
            1: "A rust-orange body",
            2: "Dark tips to the wings",
            3: "A bright yellow crest",
            4: "Bluish bars across the tail",
        },
        'answer': {1, 2},
    },
]

QIDS = [q['qid'] for q in QUESTIONS]
Q_BY_ID = {q['qid']: q for q in QUESTIONS}

# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_selected(response_text, options_dict):
    """
    Greedy prefix matching: try to match each option text in the response string.
    Sort options longest-first to avoid prefix ambiguity.
    """
    if not response_text or not response_text.strip():
        return set()
    resp = response_text.strip()
    selected = set()
    # Sort by length descending to match longest option first
    sorted_opts = sorted(options_dict.items(), key=lambda kv: -len(kv[1]))
    for key_int, opt_text in sorted_opts:
        canon = opt_text.strip()
        if canon in resp:
            selected.add(key_int)
    return selected


def all_or_nothing(selected, answer):
    return int(set(selected) == set(answer))


def edit_distance(selected, answer):
    return len(set(selected).symmetric_difference(set(answer)))


# ── Load & filter CSV ─────────────────────────────────────────────────────────

with CSV_PATH.open(newline='', encoding='utf-8-sig') as f:
    raw_rows = list(csv.reader(f))

# Row 0 = column labels, 1 = display labels, 2 = importIds, data from row 3
headers = raw_rows[0]
data_rows = raw_rows[3:]

pid_idx      = headers.index('PROLIFIC_PID')
finished_idx = headers.index('Finished')
doc_idx      = headers.index('assigned_doc')
q34_idx      = headers.index('Q34')
q23_idx      = headers.index('Q23')

DATE_FILTER  = '2026-06-23'
AT1_CORRECT  = 'Green versus black'
AT2_CORRECT  = 'Social structure'
date_idx = headers.index('RecordedDate')
at1_idx  = headers.index('AT1')
at2_idx  = headers.index('AT2')

# Filter: Jun 23 + non-empty PROLIFIC_PID + Finished + pass attention checks
filtered = [
    r for r in data_rows
    if r[date_idx].startswith(DATE_FILTER)
    and r[pid_idx].strip()
    and r[finished_idx] == 'True'
    and r[at1_idx].strip() == AT1_CORRECT
    and r[at2_idx].strip() == AT2_CORRECT
]

# Deduplicate by PROLIFIC_PID: keep row with most non-empty question answers
def count_answers(row):
    return sum(1 for q in QUESTIONS if row[headers.index(q['col'])].strip())

pid_best = {}
for r in filtered:
    pid = r[pid_idx].strip()
    if pid not in pid_best or count_answers(r) > count_answers(pid_best[pid]):
        pid_best[pid] = r
filtered = list(pid_best.values())

print(f'Loaded {len(filtered)} respondents after filtering/deduplication')
print('Conditions:', Counter(r[doc_idx] for r in filtered))

# Group by condition
cond_rows = {c: [r for r in filtered if r[doc_idx] == c] for c in CONDITIONS}

# ── Score respondents ─────────────────────────────────────────────────────────

respondent_data = []
for cond in CONDITIONS:
    for r in cond_rows[cond]:
        rec = {
            'cond': cond,
            'pid':  r[pid_idx][:12],
        }
        for q in QUESTIONS:
            col = q['col']
            qid = q['qid']
            col_idx = headers.index(col)
            sel = parse_selected(r[col_idx], q['options'])
            rec[qid] = {
                'selected':  sel,
                'aon':       all_or_nothing(sel, q['answer']),
                'edit_dist': edit_distance(sel, q['answer']),
            }
        rec['total_aon']       = sum(rec[qid]['aon']       for qid in QIDS)
        rec['mean_edit_dist']  = np.mean([rec[qid]['edit_dist'] for qid in QIDS])
        # Difficulty ratings
        try:
            rec['q34'] = float(r[q34_idx]) if r[q34_idx].strip() else None
        except ValueError:
            rec['q34'] = None
        try:
            rec['q23'] = float(r[q23_idx]) if r[q23_idx].strip() else None
        except ValueError:
            rec['q23'] = None
        respondent_data.append(rec)

print(f'Scored {len(respondent_data)} respondents')

# ── Layout helpers ────────────────────────────────────────────────────────────

n_cond  = len(CONDITIONS)
bar_w   = 0.7 / n_cond
offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * bar_w
x_q     = np.arange(len(QIDS))


def wrap(s, n=22):
    words, lines, line = s.split(), [], ''
    for w in words:
        if len(line) + len(w) + 1 > n:
            lines.append(line); line = w
        else:
            line = (line + ' ' + w).strip()
    if line:
        lines.append(line)
    return '\n'.join(lines)


def bootstrap_ci(vals, n_boot=2000, ci=95, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0
    boot_means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                           for _ in range(n_boot)])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return vals.mean() - lo, hi - vals.mean()


rng_boot = np.random.default_rng(42)

# ── 1. Answer distributions: one plot per question ────────────────────────────

for q in QUESTIONS:
    qid      = q['qid']
    opts     = q['options']
    ans_set  = q['answer']
    opt_keys = list(opts.keys())
    xq       = np.arange(len(opt_keys))

    fig, ax = plt.subplots(figsize=(9, 5))

    # Highlight correct option columns
    for i, k in enumerate(opt_keys):
        if k in ans_set:
            ax.axvspan(i - 0.5, i + 0.5, color='#fffacd', zorder=0)

    for ci, cond in enumerate(CONDITIONS):
        recs  = [rec for rec in respondent_data if rec['cond'] == cond]
        n     = len(recs)
        rates = [
            sum(1 for rec in recs if k in rec[qid]['selected']) / n * 100 if n else 0
            for k in opt_keys
        ]
        ax.bar(xq + offsets[ci], rates, width=bar_w * 0.9,
               color=COND_COLORS[cond], label=f'{COND_LABELS[cond]} (N={n})', zorder=2)

    ax.set_title(
        f'{qid}  |  correct: {sorted(ans_set)}\n{q["question"]}',
        fontsize=10, loc='left'
    )
    ax.set_xticks(xq)
    ax.set_xticklabels([f'{k}\n{wrap(opts[k])}' for k in opt_keys], fontsize=7)
    ax.set_ylim(0, 110)
    ax.set_ylabel('% selected', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=7, ncol=n_cond, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.01, 0.97, 'yellow = correct option',
            transform=ax.transAxes, fontsize=7, va='top', color='#555')

    plt.tight_layout()
    out = HERE / f'answer_dist_{qid}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out.name}')

# ── 2. Correctness grids: 1 row × 4 cols ─────────────────────────────────────

from matplotlib.colors import ListedColormap
cmap_pad = ListedColormap(['#ffffff', '#eeeeee', '#2ca02c'])

max_n = max((len(cond_rows[c]) for c in CONDITIONS), default=1)
fig, axes = plt.subplots(1, n_cond, figsize=(3 * n_cond, max(4, 0.42 * max_n)))

for ax, cond in zip(axes, CONDITIONS):
    recs = sorted(
        [rec for rec in respondent_data if rec['cond'] == cond],
        key=lambda r: -r['total_aon']
    )
    n = len(recs)
    if n == 0:
        ax.axis('off')
        continue

    M        = np.array([[rec[qid]['aon'] for qid in QIDS] for rec in recs])
    row_accs = M.mean(axis=1)
    col_accs = M.mean(axis=0)
    pids     = [r['pid'] or f'R{i}' for i, r in enumerate(recs)]

    pad   = np.full((max_n - n, len(QIDS)), -1) if n < max_n else np.empty((0, len(QIDS)))
    M_pad = np.vstack([M, pad]) if len(pad) else M

    ax.imshow(M_pad, aspect='auto', cmap=cmap_pad, vmin=-1, vmax=1, interpolation='nearest')

    xlabels = [f'{qid}\n({a:.0%})' for qid, a in zip(QIDS, col_accs)]
    ax.set_xticks(range(len(QIDS)))
    ax.set_xticklabels(xlabels, fontsize=5.5, rotation=45, ha='right')
    ylabels = [f'{p} {a:.0%}' for p, a in zip(pids, row_accs)] + [''] * max(0, max_n - n)
    ax.set_yticks(range(max_n))
    ax.set_yticklabels(ylabels, fontsize=5.5)
    ax.set_title(f'{COND_LABELS[cond]}\n(N={n})', fontsize=8,
                 color=COND_COLORS[cond], fontweight='bold')

    ax.set_xticks(np.arange(-0.5, len(QIDS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', length=0)

plt.suptitle('Correctness grids by condition  |  green = correct (all-or-nothing)',
             fontsize=10, y=1.02)
plt.tight_layout()
out = HERE / 'correctness_grids.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out.name}')

# ── 3. Score comparison: 4 rows × 2 cols ─────────────────────────────────────

fig, axes = plt.subplots(n_cond, 2, figsize=(10, 2.5 * n_cond), sharex='col')

bins_aon  = np.arange(-0.5, len(QIDS) + 1.5, 1)
bins_edit = np.arange(-0.5, 5.5 + 0.01, 0.5)

for row, cond in enumerate(CONDITIONS):
    recs  = [rec for rec in respondent_data if rec['cond'] == cond]
    n     = len(recs)
    color = COND_COLORS[cond]
    label = COND_LABELS[cond]

    aon_scores  = [rec['total_aon']      for rec in recs]
    edit_scores = [rec['mean_edit_dist'] for rec in recs]

    ax = axes[row, 0]
    if aon_scores:
        ax.hist(aon_scores, bins=bins_aon, color=color, alpha=0.85)
        ax.axvline(np.mean(aon_scores), color='black', linestyle='--', linewidth=1.2)
        ax.text(0.97, 0.95, f'μ={np.mean(aon_scores):.1f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8)
    ax.set_ylabel(f'{label}\n(N={n})', fontsize=8, color=color, fontweight='bold')
    if row == 0:
        ax.set_title(f'All-or-nothing (of {len(QIDS)})', fontsize=9)
    if row == n_cond - 1:
        ax.set_xlabel('# correct', fontsize=8)
    ax.set_xlim(-0.5, len(QIDS) + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[row, 1]
    if edit_scores:
        ax.hist(edit_scores, bins=bins_edit, color=color, alpha=0.85)
        ax.axvline(np.mean(edit_scores), color='black', linestyle='--', linewidth=1.2)
        ax.text(0.97, 0.95, f'μ={np.mean(edit_scores):.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8)
    if row == 0:
        ax.set_title('Mean edit distance (lower = better)', fontsize=9)
    if row == n_cond - 1:
        ax.set_xlabel('Edit distance', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Score distributions by condition  |  dashed = mean', fontsize=11, y=1.01)
plt.tight_layout()
out = HERE / 'score_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out.name}')

# ── 4 & 5. Per-question accuracy and edit distance ────────────────────────────

def bar_plot_q(metric, ylabel, fname, ylim, title, scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 5))
    for ci, cond in enumerate(CONDITIONS):
        recs  = [rec for rec in respondent_data if rec['cond'] == cond]
        n     = len(recs)
        means, lo_errs, hi_errs = [], [], []
        for qid in QIDS:
            vals = np.array([rec[qid][metric] for rec in recs]) * scale
            m    = vals.mean() if n else 0
            lo, hi = bootstrap_ci(vals, rng=rng_boot)
            means.append(m); lo_errs.append(lo); hi_errs.append(hi)
        ax.bar(x_q + offsets[ci], means, width=bar_w * 0.9,
               color=COND_COLORS[cond], label=f'{COND_LABELS[cond]} (N={n})', zorder=2)
        ax.errorbar(x_q + offsets[ci], means, yerr=[lo_errs, hi_errs],
                    fmt='none', ecolor='#444', elinewidth=1, capsize=3, capthick=1, zorder=3)
    ax.set_xticks(x_q)
    ax.set_xticklabels(QIDS, fontsize=10)
    ax.set_xlim(-0.6, len(QIDS) - 0.4)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_title(f'{title}\nerror bars = 95% bootstrap CI', fontsize=10)
    ax.legend(fontsize=8, ncol=n_cond, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = HERE / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out.name}')


bar_plot_q('aon',       '% correct (all-or-nothing)',          'per_question_accuracy.png',  (0, 110), 'Per-question accuracy by condition',      scale=100)
bar_plot_q('edit_dist', 'Mean edit distance (lower = better)', 'per_question_edit_dist.png', (0, 5),   'Per-question edit distance by condition', scale=1)

# ── 6. Reported difficulty ────────────────────────────────────────────────────

diff_cols = {
    'Passage difficulty (Q34)':  'q34',
    'Question difficulty (Q23)': 'q23',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
rng_jitter = np.random.default_rng(42)

for ax, (label, key) in zip(axes, diff_cols.items()):
    for ci, cond in enumerate(CONDITIONS):
        vals = np.array([rec[key] for rec in respondent_data
                         if rec['cond'] == cond and rec[key] is not None], dtype=float)
        if len(vals) == 0:
            continue
        color = COND_COLORS[cond]

        ax.boxplot(vals, positions=[ci], widths=0.5,
                   patch_artist=True, zorder=2,
                   boxprops=dict(facecolor=color, alpha=0.35, linewidth=1.2),
                   medianprops=dict(color=color, linewidth=2),
                   whiskerprops=dict(color=color, linewidth=1.2),
                   capprops=dict(color=color, linewidth=1.2),
                   flierprops=dict(marker='', linestyle='none'))

        jitter = rng_jitter.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(ci + jitter, vals, color=color, alpha=0.7, s=25, zorder=3)
        ax.scatter(ci, vals.mean(), marker='D', color=color, s=50,
                   edgecolors='white', linewidths=0.8, zorder=4)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=8, rotation=15, ha='right')
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('Difficulty (0=easy, 10=hard)', fontsize=9)
    ax.set_ylim(-0.5, 10.5)
    ax.set_yticks(range(11))
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for ci, cond in enumerate(CONDITIONS):
        vals = np.array([rec[key] for rec in respondent_data
                         if rec['cond'] == cond and rec[key] is not None], dtype=float)
        if len(vals):
            ax.text(ci, vals.mean() + 0.45, f'{vals.mean():.1f}',
                    ha='center', fontsize=7.5, color=COND_COLORS[cond], fontweight='bold')

plt.suptitle('Reported difficulty by condition  |  diamond=mean  |  box=IQR',
             fontsize=10, y=1.02)
plt.tight_layout()
out = HERE / 'reported_difficulty.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out.name}')

# ── 7. Per-respondent accuracy ────────────────────────────────────────────────

from matplotlib.patches import Patch

sorted_recs = sorted(respondent_data, key=lambda r: -r['total_aon'])
accs   = [r['total_aon'] / len(QIDS) * 100 for r in sorted_recs]
colors = [COND_COLORS[r['cond']] for r in sorted_recs]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(sorted_recs)), accs, color=colors, edgecolor='white', linewidth=0.4)

legend_handles = [Patch(facecolor=COND_COLORS[c], label=COND_LABELS[c]) for c in CONDITIONS]
ax.legend(handles=legend_handles, fontsize=8, ncol=len(CONDITIONS), loc='upper right')

ax.set_xlabel('Respondent (sorted by accuracy)', fontsize=9)
ax.set_ylabel('% correct (all-or-nothing)', fontsize=9)
ax.set_ylim(0, 110)
ax.set_xticks([])
ax.set_title(f'Per-respondent accuracy  (N={len(sorted_recs)}, {len(QIDS)} questions)', fontsize=10)
if accs:
    ax.axhline(np.mean(accs), color='black', linestyle='--', linewidth=1)
    ax.text(len(sorted_recs) - 0.5, np.mean(accs) + 1.5,
            f'mean={np.mean(accs):.0f}%', ha='right', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, acc in enumerate(accs):
    ax.text(i, acc + 1, f'{acc:.0f}',
            ha='center', va='bottom', fontsize=6.5, rotation=90)

plt.tight_layout()
out = HERE / 'per_respondent_accuracy.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out.name}')

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(f'{"Condition":<15} {"N":>4}  {"Mean AoN /5":>11}  {"Mean edit dist":>14}')
print('-' * 50)
for cond in CONDITIONS:
    recs = [rec for rec in respondent_data if rec['cond'] == cond]
    n    = len(recs)
    if not recs:
        continue
    mean_aon  = np.mean([r['total_aon']      for r in recs])
    mean_edit = np.mean([r['mean_edit_dist'] for r in recs])
    print(f'{COND_LABELS[cond]:<15} {n:>4}  {mean_aon:>11.2f}  {mean_edit:>14.3f}')

print(f'\nEdit distance = |selected Δ answer| per question, averaged over {len(QIDS)} questions.')
print('Range 0 (perfect) – 5 (worst). Lower = better.')
