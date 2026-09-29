"""How much of the benchmark's data is lost to spoken tool calls, per task and per arm?

`WorkingMemoryAgent.step()` passes `tools=TOOLS` on EVERY request, including the two
branches that forbid calling them (`allow_tools=False`, and the cap-exhausted branch, both
of which send `tool_choice="none"`). The tool schemas are still rendered into the chat
template, so the model can and does emit a `<tool_call>` block as ordinary content; with
`tool_choice="none"` nothing parses it back out, so it stays in `resp.content`, is returned
by `step()` as the agent's reply, and reaches the task's answer parser -- which sees no
answer and records the trial as unanswered.

This measures the size of that loss on runs already on disk. It is the number that decides
whether fixing collection in `bench/` is worth more than another candidate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HERMES = "NousResearch_Hermes-4-70B"
QWEN = "Qwen_Qwen3-30B-A3B-Instruct-2507"

ARMS = {
    "hermes": [(a, ROOT / f"meta_harness/runs/heldout/{a}/{HERMES}") for a in (
        "baseline", "baseline_rep2", "respond_first", "evicting_reset")],
    "qwen": [
        ("baseline", ROOT / f"meta_harness/runs/iter0/baseline/{QWEN}"),
        ("respond_first", ROOT / f"meta_harness/runs/iter6/respond_first/{QWEN}"),
        ("evicting_reset", ROOT / f"meta_harness/runs/iter6/evicting_reset/{QWEN}"),
    ],
}

CALL_RE = re.compile(r"<tool_call>|</tool_call>|\"name\"\s*:\s*\"(write_memory|delete_key)\"")


def texts(row: dict) -> list[str]:
    """Every model reply this row recorded, wherever the task stored it."""
    out = []
    for k in ("answer_raw", "recall_raw", "response_raw", "text"):
        v = row.get(k)
        if isinstance(v, str):
            out.append(v)
    for st in (row.get("step_log") or []):
        for k in ("text", "maintenance_text"):
            v = st.get(k)
            if isinstance(v, str):
                out.append(v)
    # Each task stores its replies under its own name: variable_mapping keeps them in
    # step_logs[].answer_raw, story recall in recall_text and turn_logs[].content.
    for k in ("per_trial", "questions", "answers", "step_logs", "turn_logs"):
        seq = row.get(k)
        if isinstance(seq, list):
            for it in seq:
                if isinstance(it, dict):
                    for kk in ("answer_raw", "raw", "text", "reply", "content"):
                        v = it.get(kk)
                        if isinstance(v, str):
                            out.append(v)
    for k in ("recall_text", "judge_raw"):
        v = row.get(k)
        if isinstance(v, str):
            out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=sorted(ARMS), default="hermes")
    args = ap.parse_args()

    arms = [(n, p) for n, p in ARMS[args.substrate] if p.exists()]
    tasks = sorted({f.name for _, p in arms for f in (p / "tasks").glob("wm_*.jsonl")})

    print(f"substrate {args.substrate}: share of recorded model replies that contain a "
          f"spoken tool call\n")
    print(f"{'task':<28}" + "".join(f"{n[:14]:>16}" for n, _ in arms))
    for t in tasks:
        cells = []
        for _, p in arms:
            f = p / "tasks" / t
            if not f.exists():
                cells.append("-")
                continue
            n_tot = n_bad = 0
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for s in texts(row):
                    n_tot += 1
                    if CALL_RE.search(s):
                        n_bad += 1
            cells.append(f"{n_bad}/{n_tot}" if n_tot else "-")
        print(f"{t.replace('wm_', '').replace('.jsonl', ''):<28}"
              + "".join(f"{c:>16}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
