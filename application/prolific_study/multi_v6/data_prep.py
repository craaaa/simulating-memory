#!/usr/bin/env python3
"""Reusable data loading/scoring for multi_v6 pilot results.

Extracted from analyze_v6.py so both the descriptive-plots script and the
Bayesian modeling pipeline share one canonical parsing implementation.
"""
import csv, yaml
from pathlib import Path
from itertools import combinations
import numpy as np

HERE = Path(__file__).parent
DATA_FILE = HERE / 'results.tsv'

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
TOPIC_CONTENT_DIRNAME = {
    'martial_arts': 'martial_arts', 'fruits_v2': 'fruits',
    'astronomy': 'astronomy', 'fabrics': 'fabrics',
}
FL_TOPIC = {'FL_32': 'martial_arts', 'FL_33': 'fruits_v2', 'FL_34': 'astronomy', 'FL_35': 'fabrics'}

AT_COL = {
    'martial_arts': 'martial_arts_AT_martial_arts',
    'fruits_v2':    'fruits_v2_AT_fruits_v2',
    'astronomy':    'astronomy_AT_astronomy',
    'fabrics':      'fabrics_AT_fabrics',
}

# ── Load question bank (facts about the survey, not the data — safe at import time) ──
col_correct  = {}
col_options  = {}
col_qid      = {}
content_cols = {}

for topic in TOPICS:
    base = HERE / TOPIC_CONTENT_DIRNAME[topic]
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


def load_records(data_file=DATA_FILE, verbose=True):
    """Load the Qualtrics TSV export, dedupe by PID, score responses.

    Returns (records, pass_recs) — records is every finished-with-condition
    submission; pass_recs is the subset that passed all 4 topics' attention
    checks (whole-participant AND-across-topics rule).
    """
    with open(data_file, encoding='utf-16') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    data = rows[2:]

    best_by_pid = {}
    no_pid = []
    for r in data:
        pid = r.get('PROLIFIC_PID', '').strip()
        if not pid:
            no_pid.append(r)
            continue
        dur = float(r.get('Duration (in seconds)', 0) or 0)
        if pid not in best_by_pid or dur > float(best_by_pid[pid].get('Duration (in seconds)', 0) or 0):
            best_by_pid[pid] = r
    n_raw = len(data)
    data = list(best_by_pid.values()) + no_pid
    if verbose:
        print(f'Deduped by PROLIFIC_PID: {n_raw} rows -> {len(data)} unique submissions')

    records = []
    for r in data:
        if r.get('Finished', '').strip() != 'True':
            continue
        if not r.get('martial_arts_cond', '').strip():
            continue

        attn_pass = True
        topic_attn_pass = {}
        for t in TOPICS:
            raw = r.get(AT_COL[t], '').strip()
            correct_raw = r.get(f'{t}_AT_correct', '').strip()
            correct = frozenset(correct_raw.split(',')) if correct_raw else frozenset()
            given = frozenset(raw.split(',')) if raw else frozenset()
            topic_attn_pass[t] = (given == correct)
            if given != correct:
                attn_pass = False

        fl_raw   = r.get('FL_27_DO', '').strip()
        fl_order = [FL_TOPIC.get(code) for code in fl_raw.split('|') if code in FL_TOPIC]

        rec = {
            'attn_pass': attn_pass, 'group': r.get('group', '?'), 'fl_order': fl_order,
            'duration': float(r.get('Duration (in seconds)', 0) or 0),
            'pid': r.get('PROLIFIC_PID', '').strip(),
        }

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
                'attn_pass':    topic_attn_pass[topic],
            }
        records.append(rec)

    pass_recs = [r for r in records if r['attn_pass']]
    if verbose:
        print(f'Finished (with cond): {len(records)}, attention-pass: {len(pass_recs)}')

    return records, pass_recs
