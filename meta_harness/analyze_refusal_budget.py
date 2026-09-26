"""Are the refusal loop and the tool-call-as-text failure ONE mechanism?

Hypothesis, now testable for the first time because n-back persists step_log:
  a full store REFUSES the write -> the agent spends calls on refusal/delete/retry
  -> the tool-call budget is exhausted -> step() stops offering tools
  -> the model still wants to write and emits the call as PLAIN TEXT
  -> that reply carries no classification and the trial scores unanswered.

If true, the n-back omission is downstream of the refusal loop, iteration 5 is one
fix rather than two, and it also explains the baseline's own n=3 omission.

Test: per turn, cross-tabulate whether the budget was exhausted against whether the
reply contained a tool-call string and whether it produced an answer.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
sys.path.insert(0, str(ROOT))
from bench.tasks.wm_nback import _parse_classification as pc  # noqa: E402

M = "Qwen_Qwen3-30B-A3B-Instruct-2507"
p = ROOT / f"meta_harness/runs/iter4/episodic_reset_v3/{M}/tasks/wm_nback.jsonl"

print(f"{'lvl':>4}{'turns':>7}{'budget=0':>10}{'tc-text':>9}"
      f"{'P(tc|b=0)':>11}{'P(tc|b>0)':>11}{'P(ans|b=0)':>12}{'P(ans|b>0)':>12}")
for level in (1, 2, 3):
    cells = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    ans = {0: [0, 0], 1: [0, 0]}      # budget_zero -> [answered, total]
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("n_level") != level:
            continue
        for st in (r.get("step_log") or [])[1:]:
            text = str(st.get("text") or "")
            bz = 1 if st.get("tool_call_budget_after") == 0 else 0
            tc = 1 if "<tool_call>" in text else 0
            cells[(bz, tc)] += 1
            stripped = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.S)
            ans[bz][1] += 1
            if pc(stripped) is not None:
                ans[bz][0] += 1
    turns = sum(cells.values())
    n_bz = cells[(1, 0)] + cells[(1, 1)]
    n_tc = cells[(0, 1)] + cells[(1, 1)]
    p_tc_bz = cells[(1, 1)] / n_bz if n_bz else float("nan")
    p_tc_nb = cells[(0, 1)] / (turns - n_bz) if turns - n_bz else float("nan")
    p_a_bz = ans[1][0] / ans[1][1] if ans[1][1] else float("nan")
    p_a_nb = ans[0][0] / ans[0][1] if ans[0][1] else float("nan")
    print(f"{level:>4}{turns:>7}{n_bz:>10}{n_tc:>9}{p_tc_bz:>11.3f}"
          f"{p_tc_nb:>11.3f}{p_a_bz:>12.3f}{p_a_nb:>12.3f}")

print("\nP(tc|b=0) vs P(tc|b>0): if the budget causes the tool-call-as-text reply,")
print("the first should be far larger than the second.")
print("P(ans|...) counts a GENUINE answer, parsed after stripping tool-call blocks.")

print("\n--- the refusal loop's own footprint, same run ---")
print(f"{'lvl':>4}{'memory-full results':>22}{'turns':>7}{'per turn':>10}")
for level in (1, 2, 3):
    full = turns = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("n_level") != level:
            continue
        for st in (r.get("step_log") or [])[1:]:
            turns += 1
            for tc in (st.get("tool_calls") or []):
                if "memory is full" in str(tc.get("result") or ""):
                    full += 1
    print(f"{level:>4}{full:>22}{turns:>7}{full / max(1, turns):>10.2f}")
