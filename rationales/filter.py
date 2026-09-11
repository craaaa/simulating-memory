"""STaR Algorithm 1 lines 5-6: build D_n and D^rat_n.

  D_n     = {(x_i, r̂_i, y_i)    | ŷ_i = y_i}
  D^rat_n = {(x_i, r̂rat_i, y_i) | ŷ_i != y_i AND ŷrat_i = y_i}

Acceptance itself happens in sample.py (a row is accepted iff its parsed digits equal
y_i). This module partitions accepted rows into the two sets, applies the one non-STaR
addition (the hint-leak filter), and writes the corpus.

Deliberately absent: any per-cell cap or downsampling. Balance is decided once, in
select.py; capping here would silently break the 1:1 ratio and is not part of STaR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from bench.core.io import write_json, write_jsonl

from .config import StarConfig
from .prompting import hint_leak
from .sample import GENERATION, RATIONALIZATION


def build_corpus(
    pairs: Sequence[Dict[str, Any]],
    cfg: StarConfig,
    *,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Partition into D_n / D^rat_n, drop leaked rationales, write accepted.jsonl.

    Returns (corpus, stats) where corpus is exactly what line 7 trains on.
    """
    kept: List[Dict[str, Any]] = []
    leaked: List[Dict[str, Any]] = []

    for p in pairs:
        leak = hint_leak(p["reasoning"]) if cfg.leak_filter else None
        if leak:
            leaked.append({"trial_id": p["trial_id"], "via": p["via"], "matched": leak})
            continue
        kept.append(p)

    d_n = [p for p in kept if p["via"] == GENERATION]
    d_rat = [p for p in kept if p["via"] == RATIONALIZATION]

    write_jsonl(out_dir / "accepted.jsonl", kept)

    def _cells(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            key = f"{r['direction']}:{r['length']}:{'success' if r['human_correct'] else 'fail'}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    stats = {
        "n_accepted_total": len(kept),
        "size_D_n": len(d_n),
        "size_D_rat_n": len(d_rat),
        "leak_filter_enabled": cfg.leak_filter,
        "n_dropped_hint_leak": len(leaked),
        "dropped_hint_leak": leaked[:50],
        "by_cell": _cells(kept),
        "fail_share": (
            sum(1 for p in kept if not p["human_correct"]) / len(kept) if kept else None
        ),
    }
    write_json(out_dir / "filter_stats.json", stats)
    return kept, stats
