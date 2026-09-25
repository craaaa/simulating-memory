"""Split-half noise floor for the humanlikeness metric.

For each task: repeatedly split the human participants in half and compute
W_1(half_A, half_B).  That is the irreducible floor on the 1 - W_1 humanlikeness
score -- no harness can beat it.  Also resamples the existing compactor
participants to get the model-side run-to-run spread at the released n.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import score as S  # noqa: E402

RNG = np.random.default_rng(0)
N_BOOT = 2000
MODEL_DIR = ROOT / (sys.argv[1] if len(sys.argv) > 1
                    else "runs/compactor/claude-opus-4-6")
print(f"# model: {MODEL_DIR.name}")

print(f"{'task':<26}{'n_hum':>6}{'n_mod':>6}{'floor_W1':>10}{'floor_ci':>16}"
      f"{'model_W1':>10}{'model_ci':>16}{'gap':>8}")
print("-" * 98)

rows = []
for task in S.TASKS:
    hum = S.human_scores(task)
    mod = S.llm_scores(task, MODEL_DIR, "compactor")
    if hum.size == 0 or mod.size == 0:
        continue
    # --- human split-half floor, at the model's n for the comparison side ---
    floor = []
    for _ in range(N_BOOT):
        perm = RNG.permutation(hum.size)
        a, b = hum[perm[: hum.size // 2]], hum[perm[hum.size // 2 :]]
        floor.append(S.wasserstein_1d(a, b))
    floor = np.array(floor)
    # --- model-side sampling spread: bootstrap the model participants ---
    boot = []
    for _ in range(N_BOOT):
        m = RNG.choice(mod, size=mod.size, replace=True)
        h = RNG.choice(hum, size=hum.size, replace=True)
        boot.append(S.wasserstein_1d(h, m))
    boot = np.array(boot)
    obs = S.wasserstein_1d(hum, mod)
    fl, fh = np.percentile(floor, [2.5, 97.5])
    bl, bh = np.percentile(boot, [2.5, 97.5])
    print(f"{S.TASK_DISPLAY[task]:<26}{hum.size:>6}{mod.size:>6}"
          f"{floor.mean():>10.3f}  [{fl:.3f},{fh:.3f}]"
          f"{obs:>10.3f}  [{bl:.3f},{bh:.3f}]"
          f"{obs - floor.mean():>8.3f}")
    rows.append((task, floor.mean(), obs, obs - floor.mean()))

print()
print(f"mean humanlikeness (1-W1)      : {np.mean([1 - r[2] for r in rows]):.3f}")
print(f"mean attainable ceiling        : {np.mean([1 - r[1] for r in rows]):.3f}")
print(f"mean recoverable headroom      : {np.mean([r[3] for r in rows]):.3f}")
print()
print("tasks ranked by headroom above the noise floor:")
for task, fl, obs, gap in sorted(rows, key=lambda r: -r[3]):
    print(f"  {S.TASK_DISPLAY[task]:<26}{gap:>8.3f}")
