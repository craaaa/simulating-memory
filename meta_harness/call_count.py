"""Proxy the cost of one full compactor evaluation.

Token usage is not logged in the released runs, so count LLM turns from the
stored encoding logs instead and multiply by a per-turn token estimate.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "runs/compactor/claude-opus-4-6/tasks"

# character counts of everything the log stored, as a stand-in for context size
def turns_and_chars(row):
    turns, chars = 0, 0
    log = row.get("encoding_log")
    if isinstance(log, dict):
        for v in log.values():
            if isinstance(v, list):
                turns += len(v)
                chars += len(json.dumps(v))
            else:
                chars += len(json.dumps(v))
    elif isinstance(log, list):
        turns += len(log)
        chars += len(json.dumps(log))
    chars += len(json.dumps(row.get("questions", [])))
    return max(turns, 1) + 1, chars  # +1 for the recall turn


grand_turns = grand_chars = grand_rows = 0
print(f"{'task':<30}{'rows':>7}{'turns':>9}{'MChars':>9}")
print("-" * 55)
for f in sorted(TASKS_DIR.glob("wm_*.jsonl")):
    if "application" in f.name:
        continue
    n = t = c = 0
    for line in open(f):
        row = json.loads(line)
        tt, cc = turns_and_chars(row)
        n, t, c = n + 1, t + tt, c + cc
    print(f"{f.stem:<30}{n:>7}{t:>9}{c/1e6:>9.2f}")
    grand_rows, grand_turns, grand_chars = grand_rows + n, grand_turns + t, grand_chars + c

print("-" * 55)
print(f"{'TOTAL':<30}{grand_rows:>7}{grand_turns:>9}{grand_chars/1e6:>9.2f}")

# Rough cost: assume the prompt is re-sent each turn, so input tokens scale with
# turns x mean context. Use logged chars/4 as a floor on tokens actually seen.
tok_floor = grand_chars / 4
print(f"\nlogged-content tokens (chars/4)      : {tok_floor/1e6:.2f} M  (FLOOR, output-ish only)")
for name, inp, out in [
    ("claude-opus-4-6", 15.0, 75.0),
    ("gpt-4.1", 2.0, 8.0),
    ("gpt-4.1-mini", 0.40, 1.60),
]:
    # assume 10x the logged floor in input tokens (re-sent context per turn)
    cost = (tok_floor * 10 / 1e6) * inp + (tok_floor / 1e6) * out
    print(f"  one full eval on {name:<18}: ~${cost:,.0f}   (40 candidates: ~${cost*40:,.0f})")
