"""Make digit-span error structure (axis A1) comparable between humans and model.

The two protocols differ, and the difference invalidates the naive comparison:

* Humans get an **adaptive, terminating** staircase.  Inferred from all 52
  released participants: 2 trials per span, ascending from span 2, stop as soon
  as both trials at a span fail (every participant's terminal span is exactly
  2 trials / 0 correct).  Best span = highest span with at least one correct.
* The compactor runs **all 19 span lengths** (2..20), one trial per
  (span, sequence_index), 100 sequences.

Consequence: the model gets many trials far above its ceiling that a human
would never have been administered, which is what produced the apparent
"supra-span hit rate 0.574 vs 0.000" gap in the first pass.  That number was an
artifact of the schedule, not a psychological difference.

Under a matched protocol `supra_span_hit` is 0 on BOTH sides by construction --
the staircase stops at the first double-failure, so the only administered
supra-ceiling trials are the two failed ones.  So it cannot be an axis.  What
survives as a genuine discriminator is **sub-span leakage**: how often a
participant fails a span at or below their own ceiling.  Humans are not
perfectly reliable below threshold (0.087) but they are not random either; a
harness that drops items stochastically will show a much higher rate.

This module emulates the human staircase on model trials by pairing adjacent
sequence_index values into 2-trial-per-span blocks (100 sequences -> 50
emulated participants), then applying the real stop rule.  No resampling and no
synthetic trials: every trial used is one the model actually produced.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "runs/human/working-memory-digit-span"

TRIALS_PER_SPAN = 2


def _staircase(by_span: dict[int, list[bool]]) -> dict | None:
    """Apply the human stop rule to a span -> [outcomes] mapping.

    Returns the administered trials, the ceiling, and the sub-span leak rate.
    """
    administered: list[tuple[int, bool]] = []
    best = 0
    for span in sorted(by_span):
        block = by_span[span][:TRIALS_PER_SPAN]
        if len(block) < TRIALS_PER_SPAN:
            break
        administered.extend((span, ok) for ok in block)
        if any(block):
            best = span
        else:
            break  # both failed -> staircase terminates
    if not administered or best == 0:
        return None
    sub = [ok for span, ok in administered if span <= best]
    sup = [ok for span, ok in administered if span > best]
    return {
        "best_span": best,
        "n_administered": len(administered),
        "sub_span_fail": 1.0 - float(np.mean(sub)) if sub else np.nan,
        "supra_span_hit": float(np.mean(sup)) if sup else np.nan,
    }


def human_participants() -> list[dict]:
    out = []
    for f in sorted(HUMAN_DIR.glob("run-*.json")):
        trials = json.load(open(f)).get("payload", {}).get("trials", [])
        by_span: dict[int, list[bool]] = defaultdict(list)
        for t in trials:
            if t.get("length") is None:
                continue
            by_span[int(t["length"])].append(bool(t["correct"]))
        rec = _staircase(by_span)
        if rec:
            out.append(rec)
    return out


def model_participants(jsonl_path: Path, condition: str = "C2") -> list[dict]:
    """Pair adjacent sequence_index values into 2-trial-per-span blocks.

    The released rows carry a constant participant_id, so sequence_index is the
    participant unit (the same grouping src/score.py uses).  Pairing sequences
    2k-1 and 2k gives the two trials per span the human staircase requires.
    """
    by_seq: dict[int, dict[int, bool]] = defaultdict(dict)
    for line in open(jsonl_path):
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != condition:
            continue
        seq = int(r["sequence_index"])
        by_seq[seq][int(r["span_length"])] = bool(r.get("metrics", {}).get("exact", 0) >= 1.0)

    seqs = sorted(by_seq)
    out = []
    for a, b in zip(seqs[0::2], seqs[1::2]):
        merged: dict[int, list[bool]] = defaultdict(list)
        for span in sorted(set(by_seq[a]) | set(by_seq[b])):
            for src in (a, b):
                if span in by_seq[src]:
                    merged[span].append(by_seq[src][span])
        rec = _staircase(merged)
        if rec:
            out.append(rec)
    return out


def summarize(label: str, recs: Iterable[dict]) -> dict:
    recs = list(recs)
    return {
        "label": label,
        "n": len(recs),
        "best_span": float(np.mean([r["best_span"] for r in recs])),
        "n_administered": float(np.mean([r["n_administered"] for r in recs])),
        "sub_span_fail": float(np.nanmean([r["sub_span_fail"] for r in recs])),
        "supra_span_hit": float(np.nanmean([r["supra_span_hit"] for r in recs])),
    }


def main() -> None:
    import sys

    models = sys.argv[1:] or [
        "runs/compactor/claude-opus-4-6",
        "runs/compactor/qwen_qwen3-30b-a3b-instruct-2507",
        "runs/compactor/qwen_qwen3-8b_false",
    ]

    rows = [summarize("HUMANS", human_participants())]
    for md in models:
        p = ROOT / md / "tasks/wm_digit_span_forward.jsonl"
        if not p.exists():
            print(f"  (missing {p})")
            continue
        rows.append(summarize(Path(md).name, model_participants(p)))

    print("A1  DIGIT SPAN, protocol-matched staircase "
          f"({TRIALS_PER_SPAN} trials/span, stop on double failure)")
    print(f"    {'source':<44}{'n':>5}{'best_span':>11}{'trials':>8}"
          f"{'sub_span_fail':>15}{'supra_hit':>11}")
    for r in rows:
        sup = "0.000*" if np.isnan(r["supra_span_hit"]) else f"{r['supra_span_hit']:.3f}"
        print(f"    {r['label']:<44}{r['n']:>5}{r['best_span']:>11.2f}"
              f"{r['n_administered']:>8.1f}{r['sub_span_fail']:>15.3f}{sup:>11}")
    print()
    print("    * supra-ceiling trials are 0 by construction under the matched")
    print("      staircase, on both sides -- so sub_span_fail is the axis, and")
    print("      the earlier 0.574-vs-0.000 gap was a schedule artifact.")


if __name__ == "__main__":
    main()
