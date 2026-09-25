"""Run the noise-floor / headroom report for every model that has a full
10-task compactor baseline, and write it to meta_harness/baseline_humanlikeness.txt.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
SCRIPT = Path(__file__).with_name("noise_floor.py")
MODELS = [
    "claude-opus-4-6",
    "gpt-5.4",
    "meta-llama_llama-3.3-70b-instruct",
    "meta-llama_llama-3-8b-instruct",
    "qwen_qwen3-next-80b-a3b-instruct",
    "qwen_qwen3-30b-a3b-instruct-2507",
    "qwen_qwen3-30b-a3b-thinking-2507",
    "qwen_qwen3-8b_false",
]

out = ROOT / "meta_harness"
out.mkdir(exist_ok=True)
chunks = []
for m in MODELS:
    r = subprocess.run([sys.executable, str(SCRIPT), f"runs/compactor/{m}"],
                       cwd=ROOT, capture_output=True, text=True)
    chunks.append(r.stdout.strip() + "\n")
    head = [l for l in r.stdout.splitlines()
            if l.startswith("# model") or l.startswith("mean ")]
    print("\n".join(head), "\n")
(out / "baseline_humanlikeness.txt").write_text("\n".join(chunks))
print("wrote", out / "baseline_humanlikeness.txt")
