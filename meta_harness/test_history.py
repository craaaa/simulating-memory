"""Exercise the history CLI's non-empty paths with synthetic rows.

Writes a temporary evolution_summary.jsonl, runs each subcommand, then restores
whatever was there before. Synthetic values only -- this tests the tool's logic,
not any real result.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
SUMMARY = ROOT / "meta_harness/logs/evolution_summary.jsonl"

ROWS = [
    # baseline: passes everything
    {"iteration": 0, "id": "baseline", "parent": None,
     "mean_humanlikeness_search": 0.7780,
     "humanlikeness_by_task": {"word_recognition": 0.4501, "nback": 0.8355,
                               "digit_span_forward": 0.8602},
     "axes": {"A2": {"distance": 6.076}}, "passes_floor": True,
     "passes_guards": True, "run_dir": "runs/compactor/qwen_qwen3-30b-a3b-instruct-2507"},
    # better mean AND better A2 -> should dominate baseline
    {"iteration": 1, "id": "cand_a", "parent": "baseline",
     "mean_humanlikeness_search": 0.8100,
     "humanlikeness_by_task": {"word_recognition": 0.6000, "nback": 0.8400,
                               "digit_span_forward": 0.8610},
     "axes": {"A2": {"distance": 3.500}}, "passes_floor": True,
     "passes_guards": True, "run_dir": "x"},
    # better mean, worse A2 -> also on the frontier (trade-off)
    {"iteration": 1, "id": "cand_b", "parent": "baseline",
     "mean_humanlikeness_search": 0.8300,
     "humanlikeness_by_task": {"word_recognition": 0.4600, "nback": 0.8360,
                               "digit_span_forward": 0.8605},
     "axes": {"A2": {"distance": 6.090}}, "passes_floor": True,
     "passes_guards": True, "run_dir": "x"},
    # dominated by cand_a, and fails a guard -> must be excluded from frontier
    {"iteration": 2, "id": "cand_c", "parent": "cand_a",
     "mean_humanlikeness_search": 0.7900,
     "humanlikeness_by_task": {"word_recognition": 0.5000, "nback": 0.8000,
                               "digit_span_forward": 0.8000},
     "axes": {"A2": {"distance": 4.000}}, "passes_floor": False,
     "passes_guards": False,
     "floor_violations": [{"task": "nback", "delta": -0.0355, "floor": -0.03}],
     "guard_violations": ["A1 sub-span leak 0.21 outside (0.05, 0.12)"],
     "run_dir": "x"},
]

backup = SUMMARY.read_text() if SUMMARY.exists() else None
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")

def run(*argv):
    print(f"\n$ history.py {' '.join(argv)}")
    p = subprocess.run([sys.executable, "meta_harness/history.py", *argv],
                       cwd=ROOT, capture_output=True, text=True)
    print(p.stdout.rstrip())
    if p.returncode != 0:
        print(f"[exit {p.returncode}] {p.stderr[-400:]}")
    return p

try:
    run("list")
    run("list", "--sort", "a2")
    run("diff", "baseline", "cand_a")
    front = run("frontier")
    run("regressions", "cand_c")

    out = front.stdout
    checks = {
        "cand_a on frontier": "cand_a" in out,
        "cand_b on frontier (trade-off)": "cand_b" in out,
        "baseline dominated, excluded": "baseline" not in out,
        "cand_c guard-failing, excluded": "cand_c" not in out,
    }
    print("\n=== frontier assertions")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
finally:
    if backup is None:
        SUMMARY.unlink(missing_ok=True)
    else:
        SUMMARY.write_text(backup)
    print(f"\nrestored {SUMMARY.name} (synthetic rows removed)")

sys.exit(0 if ok else 1)
