#!/usr/bin/env python3
"""Build long-format (one row per participant x topic x question) datasets
for the Bayesian mixed-effects model, from data_prep.load_records().

Writes two CSVs:
  longdata.csv        - per-topic attention-check exclusion (primary)
  longdata_strict.csv - whole-participant attention-check exclusion (sensitivity)
"""
from pathlib import Path
import pandas as pd

from data_prep import TOPICS, CONDS, content_cols, col_qid, load_records

HERE = Path(__file__).parent

COND_CATEGORIES = ['control', 'repeat_short', 'repeat_long', 'distractor']


def build_rows(records, exclusion):
    """exclusion: 'per_topic' (drop only the failing topic's rows) or
    'strict' (drop all of a participant's rows if they fail any topic)."""
    rows = []
    for rec in records:
        if exclusion == 'strict' and not rec['attn_pass']:
            continue
        for topic in TOPICS:
            sub = rec[topic]
            if exclusion == 'per_topic' and not sub['attn_pass']:
                continue
            if not rec['fl_order']:
                serial_position = None
            else:
                serial_position = rec['fl_order'].index(topic) + 1 if topic in rec['fl_order'] else None
            cols = content_cols[topic]
            for qi, col in enumerate(cols):
                rows.append({
                    'pid':             rec['pid'],
                    'group':           rec['group'],
                    'topic':           topic,
                    'question_id':     col_qid[col],
                    'condition':       sub['cond'],
                    'correct':         sub['scores'][qi],
                    'partial':         sub['partial'][qi],
                    'serial_position': serial_position,
                    't_passage':       sub['t_passage'],
                    't_qs':            sub['t_qs'],
                    'diff_passage':    sub['diff_passage'],
                    'diff_qs':         sub['diff_qs'],
                    'duration':        rec['duration'],
                })
    df = pd.DataFrame(rows)
    df['condition'] = pd.Categorical(df['condition'], categories=COND_CATEGORIES, ordered=False)
    df['serial_position'] = pd.Categorical(df['serial_position'], categories=[1, 2, 3, 4], ordered=False)
    return df


def main():
    records, pass_recs = load_records()

    df_primary = build_rows(records, exclusion='per_topic')
    df_strict = build_rows(records, exclusion='strict')

    assert not df_primary.duplicated(subset=['pid', 'topic', 'question_id']).any(), \
        'duplicate (pid, topic, question_id) rows in longdata.csv'
    assert not df_strict.duplicated(subset=['pid', 'topic', 'question_id']).any(), \
        'duplicate (pid, topic, question_id) rows in longdata_strict.csv'
    assert list(df_primary['condition'].cat.categories) == COND_CATEGORIES
    assert df_primary['condition'].cat.categories[0] == 'control'

    df_primary.to_csv(HERE / 'longdata.csv', index=False)
    df_strict.to_csv(HERE / 'longdata_strict.csv', index=False)

    n_participants_primary = df_primary['pid'].nunique()
    n_participants_strict = df_strict['pid'].nunique()
    print(f'longdata.csv:        {len(df_primary)} rows, {n_participants_primary} participants '
          f'(per-topic AT exclusion)')
    print(f'longdata_strict.csv: {len(df_strict)} rows, {n_participants_strict} participants '
          f'(whole-participant AT exclusion)')
    print()
    print('Rows per topic x condition (longdata.csv):')
    print(df_primary.groupby(['topic', 'condition'], observed=True)['pid'].nunique().unstack())


if __name__ == '__main__':
    main()
