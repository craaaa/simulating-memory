"""Merge a supplementary full-grid batch into an existing full-grid jsonl.

Validated against the Kimi-K2-0905 pilot: the dest file (e.g. n_repeats_per_cell=5,
320 rows for prompting / 80 for compactor) gets a fresh batch appended
(e.g. n_repeats_per_cell=15, freshly numbered repeat_index 1..15) with
repeat_index (and the matching ``:r{n}:`` segment inside ``id``) offset so the
combined file has non-colliding, contiguous repeat_index values.

Usage:
    python -m application.listening_qa.merge_repeat_batch \\
        --dest runs/prompting/<model_slug>/tasks/application_listening_qa_full_grid.jsonl \\
        --batch /tmp/extra_batch/application_listening_qa_full_grid.jsonl \\
        --task-kind prompting   # or "compactor"

Writes a ``<dest>.bak_n5`` backup of the pre-merge dest (refusing to overwrite
an existing backup), remaps + appends the batch, rewrites dest, and recomputes
the sibling ``_summary.json`` (cell-level aggregates only; llm_usage from the
batch summary, if present, is added to the dest summary's usage counters).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _remap_row(row: Dict[str, Any], offset: int) -> Dict[str, Any]:
    r = dict(row)
    old_ri = int(r["repeat_index"])
    new_ri = old_ri + offset
    r["repeat_index"] = new_ri
    rid = r.get("id")
    if isinstance(rid, str):
        # id format: "<task>:<cond_id>:r{repeat_index}:<topic_id>:<level>"
        r["id"] = re.sub(rf":r{old_ri}:", f":r{new_ri}:", rid, count=1)
    return r


def _recompute_cell_summaries_prompting(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        key = f"{r['topic_id']}:{r['level']}:{r['condition_id']}"
        cells.setdefault(key, []).append(r)
    out = {}
    for key, cell_rows in cells.items():
        scored = [r for r in cell_rows if r.get("metrics") is not None]
        n = len(cell_rows)
        exact_mean = (
            sum(r["metrics"]["exact_match_accuracy"] for r in scored) / len(scored) if scored else None
        )
        stmt_mean = (
            sum(r["metrics"]["per_statement_accuracy"] for r in scored) / len(scored) if scored else None
        )
        parse_errors = sum(1 for r in cell_rows if r.get("parse_errors"))
        out[key] = {
            "n": n,
            "exact_match_mean": exact_mean,
            "per_statement_mean": stmt_mean,
            "parse_error_count": parse_errors,
        }
    return out


def _recompute_cell_summaries_compactor(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        key = f"{r['topic_id']}:{r['level']}:{r['condition_id']}"
        cells.setdefault(key, []).append(r)
    out = {}
    for key, cell_rows in cells.items():
        n = len(cell_rows)
        exact_mean = sum(r["metrics"]["exact_match_accuracy"] for r in cell_rows) / n if n else None
        stmt_mean = sum(r["metrics"]["per_statement_accuracy"] for r in cell_rows) / n if n else None
        parse_errors = sum(1 for r in cell_rows if r.get("parse_errors"))
        out[key] = {
            "n": n,
            "exact_match_mean": exact_mean,
            "per_statement_mean": stmt_mean,
            "parse_error_count": parse_errors,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="Existing full-grid jsonl to extend in place.")
    ap.add_argument("--batch", required=True, help="Freshly generated batch jsonl (repeat_index restarts at 1).")
    ap.add_argument("--task-kind", required=True, choices=["prompting", "compactor"])
    ap.add_argument(
        "--expect-dest-rows",
        type=int,
        default=None,
        help="Tripwire: abort if dest row count before merge != this.",
    )
    ap.add_argument(
        "--expect-total-rows",
        type=int,
        default=None,
        help="Tripwire: abort if merged row count != this.",
    )
    args = ap.parse_args()

    dest_path = Path(args.dest)
    batch_path = Path(args.batch)
    backup_path = dest_path.with_suffix(dest_path.suffix + ".bak_n5")

    dest_rows = _read_jsonl(dest_path)
    batch_rows = _read_jsonl(batch_path)

    if args.expect_dest_rows is not None and len(dest_rows) != args.expect_dest_rows:
        raise SystemExit(
            f"TRIPWIRE: dest {dest_path} has {len(dest_rows)} rows, expected {args.expect_dest_rows}. Stopping."
        )

    if backup_path.exists():
        raise SystemExit(f"Refusing to overwrite existing backup {backup_path}. Remove it first if intentional.")

    max_ri = max((int(r["repeat_index"]) for r in dest_rows if "repeat_index" in r), default=0)
    remapped = [_remap_row(r, max_ri) for r in batch_rows]

    merged = dest_rows + remapped

    # sanity: no duplicate ids, and every id's r-segment matches its repeat_index field
    ids = [r["id"] for r in merged]
    if len(ids) != len(set(ids)):
        raise SystemExit("TRIPWIRE: duplicate ids after remap. Aborting without writing.")
    for r in merged:
        m = re.search(r":r(\d+):", r["id"])
        if not m or int(m.group(1)) != int(r["repeat_index"]):
            raise SystemExit(f"TRIPWIRE: id/repeat_index mismatch for row {r['id']!r}. Aborting without writing.")

    if args.expect_total_rows is not None and len(merged) != args.expect_total_rows:
        raise SystemExit(
            f"TRIPWIRE: merged row count {len(merged)} != expected {args.expect_total_rows}. Aborting without writing."
        )

    ri_values = sorted({int(r["repeat_index"]) for r in merged})
    print(f"dest rows: {len(dest_rows)}, batch rows: {len(batch_rows)}, merged rows: {len(merged)}")
    print(f"repeat_index range: {ri_values[0]}..{ri_values[-1]} (n={len(ri_values)} distinct)")

    # back up pre-merge dest, then write merged file
    dest_path.rename(backup_path)
    _write_jsonl(dest_path, merged)

    # recompute summary
    summary_path = dest_path.with_name(dest_path.stem + "_summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    if args.task_kind == "prompting":
        summary["cells"] = _recompute_cell_summaries_prompting(merged)
    else:
        summary["cells"] = _recompute_cell_summaries_compactor(merged)
    summary["n_repeats_per_cell"] = len(ri_values)
    summary["n_total_trials"] = len(merged)

    # fold in batch usage counters (compactor only) if present
    batch_summary_path = batch_path.with_name(batch_path.stem + "_summary.json")
    if args.task_kind == "compactor" and batch_summary_path.exists():
        batch_summary = json.loads(batch_summary_path.read_text())
        batch_usage = batch_summary.get("llm_usage") or {}
        base_usage = summary.get("llm_usage") or {}
        merged_usage = dict(base_usage)
        for k, v in batch_usage.items():
            if isinstance(v, (int, float)) and isinstance(base_usage.get(k), (int, float)):
                merged_usage[k] = base_usage[k] + v
            elif k not in merged_usage:
                merged_usage[k] = v
        summary["llm_usage"] = merged_usage

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Backed up pre-merge dest to {backup_path}")
    print(f"Wrote merged dest to {dest_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
