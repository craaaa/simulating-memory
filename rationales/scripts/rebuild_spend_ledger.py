"""Rebuild the lifetime spend ledger from the authoritative per-run ledgers.

Use when the global ledger has picked up rows that do not correspond to real usage
(the test suite wrote synthetic million-token calls into it before the isolation
fixture existed). Per-run `cost_ledger.jsonl` files are the source of truth: they are
written only by real runs.

    python -m rationales.scripts.rebuild_spend_ledger [--apply]

Without --apply it only reports what would change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rationales.config import GLOBAL_LEDGER, PKG_DIR, tinker_cost_usd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write the rebuilt ledger.")
    args = ap.parse_args()

    rebuilt = []
    for ledger in sorted(PKG_DIR.glob("out/*/*/cost_ledger.jsonl")):
        run = ledger.parent.name
        model = ledger.parent.parent.name.replace("_", "/", 1)
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rebuilt.append(
                {
                    "model": model,
                    "run": run,
                    "kind": r["kind"],
                    "prefill_tokens": r["prefill_tokens"],
                    "sample_tokens": r["sample_tokens"],
                    "train_tokens": r["train_tokens"],
                    "cost_usd": tinker_cost_usd(
                        model,
                        prefill_tokens=r["prefill_tokens"],
                        sample_tokens=r["sample_tokens"],
                        train_tokens=r["train_tokens"],
                    ),
                }
            )

    # Probe spend goes through OpenRouter, not Tinker, so it has no per-run
    # cost_ledger.jsonl. Its source of truth is the probe report, which records what
    # OpenRouter actually billed. Rebuilt from there rather than carried over, so a
    # rebuild neither drops it nor double-counts it.
    from rationales.probe import probe_ledger_row

    for report_path in sorted(PKG_DIR.glob("out/probe/*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = probe_ledger_row(report, timestamp=report_path.stem.split("_")[0])
        if row is not None:
            rebuilt.append(row)

    old_rows = []
    if GLOBAL_LEDGER.is_file():
        old_rows = [json.loads(l) for l in GLOBAL_LEDGER.read_text().splitlines() if l.strip()]

    def total(rows):
        return sum(r.get("cost_usd") or 0 for r in rows)

    print(f"current ledger: {len(old_rows):5d} rows  ${total(old_rows):.4f}")
    print(f"rebuilt:        {len(rebuilt):5d} rows  ${total(rebuilt):.4f}")

    if not args.apply:
        print("\n(dry run -- pass --apply to write)")
        return

    backup = GLOBAL_LEDGER.with_suffix(".jsonl.bak")
    if GLOBAL_LEDGER.is_file():
        backup.write_text(GLOBAL_LEDGER.read_text())
        print(f"backed up to {backup}")
    GLOBAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LEDGER.open("w", encoding="utf-8") as f:
        for r in rebuilt:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {GLOBAL_LEDGER}")


if __name__ == "__main__":
    main()
