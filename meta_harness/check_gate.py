"""Decide whether a local vLLM run reproduces the released OpenRouter baseline.

This is the go/no-go before any harness search. If the serving stack differs --
sampling defaults, chat template, tokenizer revision -- then the headroom
figures the whole plan rests on are measuring something else, and 40 candidates
would be spent against the wrong contract.

Compares a fresh wm_word_recognition JSONL against
runs/compactor/qwen_qwen3-30b-a3b-instruct-2507 on the statistics that would
move if the stack differed:

  * score distribution     -- W_1 between the two runs' per-participant scores,
                              judged against the human split-half noise floor
                              (0.075 for this task).  A difference below the
                              floor is indistinguishable from sampling noise.
  * miss / false-alarm     -- the A2 axis, released value 0.018
  * trials attempted       -- released mean 87.5; this drives A2's denominator

Usage:
    python meta_harness/check_gate.py <fresh_wm_word_recognition.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import score as S  # noqa: E402

RELEASED = ROOT / "runs/compactor/qwen_qwen3-30b-a3b-instruct-2507/tasks/wm_word_recognition.jsonl"
NOISE_FLOOR = 0.075  # human split-half W_1 for word recognition
N_BOOT = 2000


def participant_scores(path: Path) -> np.ndarray:
    """Score rows with score.py's own scorer, so the gate can never measure a
    different quantity from the headline humanlikeness metric."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != "C2":
            continue
        s = S._score_llm_row("word_recognition", r)
        if s is not None:
            out.append(s)
    return np.asarray(out, dtype=float)


def a2(path: Path) -> tuple[float, float, float, float]:
    miss, fa, trials = [], [], []
    for line in open(path):
        r = json.loads(line)
        pt = r.get("per_trial") or []
        old = [t for t in pt if str(t["expected"]).lower().startswith("old")]
        new = [t for t in pt if str(t["expected"]).lower().startswith("new")]
        if not pt:
            continue
        if old:
            miss.append(np.mean([not str(t["model_response"]).lower().startswith("old")
                                 for t in old]))
        if new:
            fa.append(np.mean([str(t["model_response"]).lower().startswith("old")
                               for t in new]))
        trials.append(len(pt))
    m = float(np.nanmean(miss)) if miss else np.nan
    f = float(np.nanmean(fa)) if fa else np.nan
    ratio = m / f if f else np.nan
    return m, f, ratio, float(np.mean(trials)) if trials else np.nan


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    fresh = Path(sys.argv[1])
    if not fresh.exists():
        print(f"no such file: {fresh}")
        sys.exit(2)
    if not RELEASED.exists():
        print(f"missing released reference: {RELEASED}")
        sys.exit(2)

    a, b = participant_scores(fresh), participant_scores(RELEASED)
    w = S.wasserstein_1d(a, b)

    # Bootstrap spread, for context only.  Note this resamples BOTH sides
    # independently, so it does not collapse to 0 even for two identical files
    # (self-test: point estimate 0.000, CI [0.012, 0.124]).  It is therefore a
    # scale reference, NOT the decision rule -- the point estimate against the
    # noise floor is what decides pass/fail.
    rng = np.random.default_rng(0)
    boot = [S.wasserstein_1d(rng.choice(a, a.size, replace=True),
                             rng.choice(b, b.size, replace=True))
            for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    fm, ff, fr, ft = a2(fresh)
    rm, rf, rr, rt = a2(RELEASED)

    print("REPRODUCTION GATE -- local vLLM vs released OpenRouter baseline")
    print(f"  model: qwen3-30b-a3b-instruct-2507, task: wm_word_recognition\n")
    print(f"  {'metric':<26}{'local':>12}{'released':>12}{'delta':>12}")
    print("  " + "-" * 62)
    print(f"  {'n participants':<26}{a.size:>12}{b.size:>12}{'':>12}")
    print(f"  {'mean score':<26}{a.mean():>12.3f}{b.mean():>12.3f}{a.mean()-b.mean():>12.3f}")
    print(f"  {'miss rate':<26}{fm:>12.3f}{rm:>12.3f}{fm-rm:>12.3f}")
    print(f"  {'false-alarm rate':<26}{ff:>12.3f}{rf:>12.3f}{ff-rf:>12.3f}")
    print(f"  {'miss/fa ratio (A2)':<26}{fr:>12.3f}{rr:>12.3f}{fr-rr:>12.3f}")
    print(f"  {'trials attempted':<26}{ft:>12.1f}{rt:>12.1f}{ft-rt:>12.1f}")
    print()
    print(f"  W_1(local, released) = {w:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  human split-half noise floor for this task = {NOISE_FLOOR:.3f}")
    print()

    # The gate passes when the two runs are closer to each other than the human
    # data is to itself -- i.e. any difference hides inside sampling noise.
    if w <= NOISE_FLOOR:
        print("  PASS: local vLLM is within the noise floor of the released run.")
        print("        The serving stack is not shifting the measured behaviour,")
        print("        so the released headroom figures carry over.")
        rc = 0
    else:
        print("  FAIL: local vLLM differs from the released run by more than the")
        print("        human noise floor. Do NOT start the search -- the baseline")
        print("        and therefore the headroom figures do not carry over.")
        print("        Check sampling params, chat template, tokenizer revision.")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
