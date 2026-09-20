"""Re-derive a round's accept decisions from the completions already on disk.

For when a parser bug, not a model failure, emptied a corpus. Every sample_pool.jsonl
row keeps the raw completion, so the accept decision can be recomputed for free --
sampling is the expensive phase and it has already been paid for.

    python -m rationales.scripts.reparse_pool --run-dir rationales/out/<model>/<ts> [--apply]

Without --apply it only reports what would change. After applying, resume the round:

    python -m rationales.cli star --task <task> --resume --run-dir <run-dir>

which skips sampling and picks up at the filter step.

This rewrites `reasoning`, the answer column and `accepted` on every row, and refreshes
sample_report.json's stats. It does not re-sample and cannot change what the model said.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rationales.sample import pool_stats
from rationales.task import resolve, task_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--task", default=None, help=f"One of: {', '.join(task_names())}")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--apply", action="store_true", help="Rewrite the pool in place.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    round_dir = run_dir / "rounds" / f"round_{args.round}"
    pool_path = round_dir / "sample_pool.jsonl"
    if not pool_path.is_file():
        raise SystemExit(f"no sample pool at {pool_path}")

    task_name = args.task
    if task_name is None:
        cfg_path = run_dir / "run_config.json"
        task_name = json.loads(cfg_path.read_text(encoding="utf-8")).get("task", "digit_span")
        print(f"task from run_config.json: {task_name}")
    task = resolve(task_name)

    rows = [json.loads(l) for l in pool_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    answer_key = next(iter(task.answer_fields([])))

    was_accepted = sum(1 for r in rows if r.get("accepted"))
    out_rows = []
    accepted_by_trial: dict = {}
    for r in rows:
        reasoning, answer, errors = task.parse(r.get("raw"))
        # Same rule sample._attempt applies: at most one accept per trial, first wins.
        ok = (
            reasoning is not None
            and r["trial_id"] not in accepted_by_trial
            and _matches_target(r, answer)
        )
        if ok:
            accepted_by_trial[r["trial_id"]] = r["via"]
        out_rows.append(
            dict(r, reasoning=reasoning, accepted=bool(ok), parse_errors=errors, **{answer_key: answer})
        )

    now_accepted = sum(1 for r in out_rows if r["accepted"])
    print(f"rows:            {len(rows)}")
    print(f"accepted before: {was_accepted}")
    print(f"accepted after:  {now_accepted}  ({len(accepted_by_trial)} distinct trials)")

    if not args.apply:
        print("\n(dry run -- pass --apply to rewrite the pool)")
        return

    backup = pool_path.with_suffix(".jsonl.bak")
    backup.write_text(pool_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pool_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nbacked up to {backup}\nrewrote {pool_path}")

    report_path = round_dir / "sample_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["stats"] = pool_stats(pool_path, task)
        report["reparsed"] = True
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"refreshed {report_path}")
        print(f"fail-side generation yield: {report['stats']['fail_side_generation_yield']}")


def _matches_target(row: dict, answer) -> bool:
    """Compare against y_i as stored on the row, without reloading the dataset.

    Every task writes the human's own response under a `target_*` column (digit span's
    `target_digits`, listening's `target_options`). Sets for the tasks whose answer is
    unordered, exact equality otherwise -- the same comparison `accepts` makes.
    """
    target_key = next(k for k in row if k.startswith("target_"))
    target = row[target_key]
    if isinstance(target, list) and target_key == "target_options":
        return set(answer or []) == set(target or [])
    return list(answer or []) == list(target or [])


if __name__ == "__main__":
    main()
