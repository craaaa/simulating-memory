"""Query the search history. This is the proposer's main view of the past.

File layout follows the Meta-Harness reference example so the conventions are
familiar:

    meta_harness/logs/evolution_summary.jsonl   one row per candidate evaluation
    meta_harness/logs/frontier.json             current Pareto frontier
    meta_harness/logs/pending_eval.json         candidates the proposer wants run
    meta_harness/runs/<iter>/<candidate>/       per-candidate artifacts

The paper's point is that the proposer reads raw traces off the filesystem
rather than a compressed summary, so this tool is for orientation and ranking --
`trace` deliberately prints one full episode rather than statistics over many.

Commands:
    list [--sort mean|a2|iteration]   one line per candidate
    show <id>                        manifest, scores, per-task vector, guards
    diff <id_a> <id_b>               per-task delta with the noise floor marked
    frontier                         current Pareto frontier
    regressions <id>                 tasks violating the per-task floor
    trace <id> --task T [--n N]      one episode's encode/recall trace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "meta_harness/logs"
SUMMARY = LOGS / "evolution_summary.jsonl"
FRONTIER = LOGS / "frontier.json"

NOISE_FLOOR = {
    "digit_span_forward": 0.036, "digit_span_reverse": 0.026, "nback": 0.025,
    "word_recognition": 0.075, "variable_mapping": 0.041, "factual_qa": 0.061,
    "narrative_qa": 0.053, "semantic_story_recall": 0.042, "map_task": 0.054,
    "craft_task": 0.041,
}


def load_rows() -> list[dict[str, Any]]:
    if not SUMMARY.exists():
        return []
    rows = []
    for line in SUMMARY.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _a2(row: dict) -> float | None:
    a2 = (row.get("axes") or {}).get("A2") or {}
    return a2.get("distance")


def _mean(row: dict) -> float | None:
    return row.get("mean_humanlikeness_search")


def cmd_list(args: argparse.Namespace) -> int:
    rows = load_rows()
    if not rows:
        print(f"no history yet ({SUMMARY} absent or empty)")
        return 0
    key = {"mean": lambda r: -(_mean(r) or -1),
           "a2": lambda r: (_a2(r) if _a2(r) is not None else 1e9),
           "iteration": lambda r: r.get("iteration", 0)}[args.sort]
    rows.sort(key=key)
    print(f"{'iter':>4} {'candidate':<24} {'mean_HL':>8} {'A2_dist':>8} "
          f"{'floor':>6} {'guards':>7}  parent")
    for r in rows:
        m, a = _mean(r), _a2(r)
        print(f"{r.get('iteration', 0):>4} {str(r.get('id'))[:24]:<24} "
              f"{(f'{m:.4f}' if m is not None else 'n/a'):>8} "
              f"{(f'{a:.3f}' if a is not None else 'n/a'):>8} "
              f"{('ok' if r.get('passes_floor', True) else 'FAIL'):>6} "
              f"{('ok' if r.get('passes_guards', True) else 'FAIL'):>7}  "
              f"{r.get('parent') or '-'}")
    return 0


def _find(cid: str) -> dict[str, Any] | None:
    for r in load_rows():
        if str(r.get("id")) == cid:
            return r
    return None


def cmd_show(args: argparse.Namespace) -> int:
    r = _find(args.id)
    if r is None:
        print(f"no candidate {args.id!r} in history")
        return 1
    print(json.dumps(r, indent=2))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a, b = _find(args.a), _find(args.b)
    if a is None or b is None:
        print("one or both candidates not found")
        return 1
    ta = a.get("humanlikeness_by_task", {})
    tb = b.get("humanlikeness_by_task", {})
    print(f"{'task':<26}{args.a[:12]:>12}{args.b[:12]:>12}{'delta':>9}  note")
    for task in sorted(set(ta) | set(tb)):
        va, vb = ta.get(task), tb.get(task)
        if va is None or vb is None:
            print(f"{task:<26}{str(va):>12}{str(vb):>12}{'':>9}  incomparable")
            continue
        d = vb - va
        note = "within noise floor" if abs(d) < NOISE_FLOOR.get(task, 0.05) else ""
        print(f"{task:<26}{va:>12.4f}{vb:>12.4f}{d:>+9.4f}  {note}")
    ma, mb = _mean(a), _mean(b)
    if ma is not None and mb is not None:
        print(f"\n{'mean (search tasks)':<26}{ma:>12.4f}{mb:>12.4f}{mb - ma:>+9.4f}")
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    if FRONTIER.exists():
        print(FRONTIER.read_text())
        return 0
    # Derive it if the file is not written yet: maximize mean humanlikeness,
    # minimize A2 distance, and only among candidates that pass the guards.
    #
    # Instrument candidates are excluded. A candidate that changes what a task
    # MEASURES, rather than how well the harness does on it, is incomparable on
    # both frontier axes at once: its mean is computed against a reference the
    # candidate itself invalidated, and A2 -- the second axis -- no longer means
    # what it means for every other row. `serial_recognition` is the case that
    # forced this: closing the word-recognition leak moves that task from a
    # mixture of 36 prompt-readers and 7 store-consulters to store-consulters
    # only, so its word_recognition humanlikeness rises arithmetically (a point
    # mass at zero already scores ~0.685 against a baseline of 0.816 and a human
    # 0.315) without the harness having become more humanlike at anything.
    rows = [r for r in load_rows()
            if _mean(r) is not None and _a2(r) is not None
            and r.get("passes_guards", True) and r.get("passes_floor", True)
            and not r.get("instrument", False)]
    front = []
    for r in rows:
        if not any(o is not r and _mean(o) >= _mean(r) and _a2(o) <= _a2(r)
                   and (_mean(o) > _mean(r) or _a2(o) < _a2(r)) for o in rows):
            front.append(r)
    front.sort(key=lambda r: -_mean(r))
    print(f"derived frontier ({len(front)} of {len(rows)} eligible):")
    for r in front:
        print(f"  {str(r.get('id')):<24} mean_HL {_mean(r):.4f}  A2_dist {_a2(r):.3f}")
    return 0


def cmd_regressions(args: argparse.Namespace) -> int:
    r = _find(args.id)
    if r is None:
        print(f"no candidate {args.id!r}")
        return 1
    viol = r.get("floor_violations") or []
    guards = r.get("guard_violations") or []
    if not viol and not guards:
        print("no floor or guard violations")
        return 0
    for v in viol:
        print(f"floor  {v['task']}: delta {v['delta']:+.4f} (limit {v['floor']:+.4f})")
    for g in guards:
        print(f"guard  {g}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    r = _find(args.id)
    if r is None:
        print(f"no candidate {args.id!r}")
        return 1
    run_dir = Path(r.get("run_dir", ""))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    path = run_dir / f"tasks/wm_{args.task}.jsonl"
    if not path.exists():
        print(f"no trace file at {path}")
        return 1
    for i, line in enumerate(open(path)):
        if i != args.n:
            continue
        row = json.loads(line)
        print(json.dumps(row, indent=2)[:args.max_chars])
        return 0
    print(f"row {args.n} not found in {path}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list"); p.add_argument(
        "--sort", choices=["mean", "a2", "iteration"], default="mean")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show"); p.add_argument("id"); p.set_defaults(fn=cmd_show)

    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("frontier"); p.set_defaults(fn=cmd_frontier)

    p = sub.add_parser("regressions"); p.add_argument("id")
    p.set_defaults(fn=cmd_regressions)

    p = sub.add_parser("trace"); p.add_argument("id")
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=8000)
    p.set_defaults(fn=cmd_trace)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
