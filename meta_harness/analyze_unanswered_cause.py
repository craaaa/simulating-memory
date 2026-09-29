"""What actually leaves an n-back trial unanswered, post-fix?

UNITS. This script reports COUNTS and PROPORTIONS OF TRIALS, not humanlikeness. A
"proportion unanswered" of 0.22 means 22% of the scored n-back trials in that cell had no
parsed label. Nothing here is a humanlikeness score and nothing here is a delta.

WHY. The project's standing account of unanswered n-back trials was: the tool-call budget
runs dry, the model types the tool call out as text instead of answering, and the trial is
recorded unanswered. Route 3 in HANDOFF.md is filed under that chain. The post-fix Qwen runs
falsified its middle link -- spoken tool calls on n-back answer turns went 553/2100 to
0/2100 while trials answered went 10.86 to 10.85 of 14 -- so tool-call-as-text does not
cause unanswered trials, and route 3's mechanism of record is gone.

That leaves an unexplained fact: the post-fix baseline still leaves about 3.15 of 14 n-back
trials unanswered. This script cross-tabs each scored trial's answered/unanswered status
against the budget state of ITS OWN answer turn, so route 3 can be kept or dropped on
evidence already on disk rather than on a spent GPU arm.

Mapping: per_trial trial k is step n+k of step_log (step 0 is the instruction turn, steps
1..n are the lead-in letters where "no response" is correct).

Usage:
    python meta_harness/analyze_unanswered_cause.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = "Qwen_Qwen3-30B-A3B-Instruct-2507"

# "collection fix" here means eb3e96f (empty `tools` list omits the schemas), NOT the later
# instrument fix (turn-boundary reset + n-back store injection). Both predate any run listed
# here; nothing below was produced after `exp/compactor-prefix-v1`.
GENERATIONS = {
    "pre-collection-fix baseline": ["iter0/baseline", "iter8repA/baseline",
                                    "iter8repB/baseline"],
    "post-collection-fix baseline": ["iter9postA/baseline", "iter9postB/baseline"],
    "post-collection-fix respond_first_v2": ["iter9postA/respond_first_v2",
                                             "iter9postB/respond_first_v2"],
}


def trials(run: Path):
    """Yield (answered, budget_exhausted, cap_hit, n_level) per scored trial."""
    f = run / "tasks" / "wm_nback.jsonl"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n = int(r.get("n_level") or 1)
        per_trial = r.get("per_trial") or []
        steps = r.get("step_log") or []
        # Post-instrument-fix rows hold TWO step entries per letter (encode, answer), so
        # the old positional rule "scored trial k is step n+k" is only correct for rows
        # written before `exp/compactor-prefix-v1`. When the explicit mapping is present,
        # use it.
        by_pos = r.get("answer_step_by_position") or {}
        for k, t in enumerate(per_trial, start=1):
            if by_pos:
                i = by_pos.get(str(n + k), by_pos.get(n + k))
                if i is None:
                    continue
            else:
                i = n + k
            if i >= len(steps):
                continue
            st = steps[i]
            lab = t.get("model_label")
            answered = bool(lab) and str(lab).strip() != ""
            bb = st.get("tool_call_budget_before")
            yield (answered,
                   bb is not None and int(bb) <= 0,
                   bool(st.get("tool_call_cap_hit")),
                   n)


def main() -> int:
    print(__doc__.split("Usage:")[0].rstrip())
    for gen, dirs in GENERATIONS.items():
        rows = []
        for d in dirs:
            run = ROOT / "meta_harness" / "runs" / d / Q
            if not run.exists():
                print(f"  MISSING {d}")
                continue
            rows.extend(trials(run))
        if not rows:
            continue
        n = len(rows)
        unans = sum(1 for a, *_ in rows if not a)
        print(f"\n{'=' * 88}\n{gen}: {n} scored trials, {unans} unanswered "
              f"({unans / n:.3f} of trials)")

        # cross-tab: budget exhausted entering the answer turn x answered
        for name, idx in (("budget_before <= 0", 1), ("tool_call_cap_hit", 2)):
            a = sum(1 for r in rows if r[idx] and not r[0])       # flag & unanswered
            b = sum(1 for r in rows if r[idx] and r[0])           # flag & answered
            c = sum(1 for r in rows if not r[idx] and not r[0])   # no flag & unanswered
            d_ = sum(1 for r in rows if not r[idx] and r[0])      # no flag & answered
            pf = a / (a + b) if a + b else None
            pn = c / (c + d_) if c + d_ else None
            print(f"  {name:<22} flagged {a + b:>5} trials, "
                  f"unanswered {(f'{pf:.3f}' if pf is not None else '   -')} of them; "
                  f"unflagged {c + d_:>5} trials, unanswered "
                  f"{(f'{pn:.3f}' if pn is not None else '   -')} of them")
            if pf is not None and pn is not None:
                print(f"{'':<24}lift {pf - pn:+.3f} in proportion-of-trials-unanswered; "
                      f"{a} of {unans} unanswered trials ({a / unans:.3f}) are flagged")

        by_level = {}
        for r in rows:
            k = by_level.setdefault(r[3], [0, 0])
            k[0] += 1
            k[1] += 0 if r[0] else 1
        parts = " ".join(f"n={lv}: {v[1]}/{v[0]} ({v[1] / v[0]:.3f})"
                         for lv, v in sorted(by_level.items()))
        print(f"  unanswered by level  {parts}")
    print("\nRead: if 'budget_before <= 0' carries most of the unanswered trials and has a "
          "large\nlift, route 3 survives with a new mechanism. If the unanswered trials sit "
          "mostly in\nthe unflagged cell, budget exhaustion is not what stops the model "
          "answering and\nreshaping _tool_call_cap() cannot recover them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
