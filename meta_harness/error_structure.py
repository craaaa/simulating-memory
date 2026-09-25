"""Error-structure metrics: do model errors look like human errors, beyond
matching the score distribution?

Three axes, each computable from the released data for both humans and models,
and each one a noise-injecting harness would fail:

  A1  digit span, threshold sharpness
      sub_span_fail  = P(error | length <= that participant's best span)
      supra_span_hit = P(correct | length >  that participant's best span)
      Humans fail progressively around a threshold.  A random-drop harness
      scatters errors at short spans and occasionally nails long ones.

  A2  word recognition, error asymmetry
      miss_rate = P(say "new" | word was old),  fa_rate = P(say "old" | new)
      Humans show a characteristic miss/false-alarm ratio.  Random responding
      drives that ratio to 1.

  A3  story recall, verbatim vs gist
      BLEU of recall against the transcript.  Humans reconstruct gist
      (BLEU ~ 0); a harness that stores text verbatim scores high even when
      its coverage score matches humans.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/cl5625/simulating-memory/.claude/worktrees/meta-harness-compactor")
HUMAN = ROOT / "runs/human"


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# ---------------------------------------------------------------- A1 digit span
def a1_human():
    out = []
    for f in (HUMAN / "working-memory-digit-span").glob("run-*.json"):
        trials = json.load(open(f)).get("payload", {}).get("trials", [])
        out.append(_a1_one([(int(t["length"]), bool(t["correct"])) for t in trials
                            if t.get("length") is not None]))
    return [o for o in out if o]


def a1_model(model_dir):
    by_p = defaultdict(list)
    path = ROOT / model_dir / "tasks/wm_digit_span_forward.jsonl"
    for line in open(path):
        r = json.loads(line)
        if r.get("condition_id") not in (None, "C2"):
            continue
        correct = bool(r.get("metrics", {}).get("exact", 0) >= 1.0)
        # The released rows are 19 span lengths (2..20) x 100 sequence_index
        # values under a single constant participant_id, so sequence_index is
        # the participant -- same grouping score.py uses.
        by_p[r["sequence_index"]].append((int(r["span_length"]), correct))
    return [o for o in (_a1_one(v) for v in by_p.values()) if o]


def _a1_one(pairs):
    if not pairs:
        return None
    by_len = defaultdict(list)
    for length, ok in pairs:
        by_len[length].append(ok)
    best = 0
    for L in sorted(by_len):
        if any(by_len[L]):
            best = L
        else:
            break
    sub = [ok for L, ok in pairs if L <= best]
    sup = [ok for L, ok in pairs if L > best]
    return (
        1 - np.mean(sub) if sub else np.nan,   # sub-span failure rate
        np.mean(sup) if sup else np.nan,       # supra-span hit rate
    )


# --------------------------------------------------------- A2 word recognition
def a2_human():
    miss, fa, trials = [], [], []
    for f in (HUMAN / "working-memory-word-recognition").glob("run-*.json"):
        resp = json.load(open(f)).get("payload", {}).get("responses", [])
        pairs = [(str(r["expectedResponse"]).lower(), str(r["userResponse"]).lower())
                 for r in resp]
        m, fp = _a2_one(pairs)
        if m is not None:
            miss.append(m)
            fa.append(fp)
            trials.append(len(pairs))
    return miss, fa, trials


def a2_model(model_dir):
    miss, fa, trials = [], [], []
    for line in open(ROOT / model_dir / "tasks/wm_word_recognition.jsonl"):
        r = json.loads(line)
        pt = r.get("per_trial") or []
        pairs = [(str(t["expected"]).lower(), str(t["model_response"]).lower())
                 for t in pt]
        m, fp = _a2_one(pairs)
        if m is not None:
            miss.append(m)
            fa.append(fp)
            trials.append(len(pairs))
    return miss, fa, trials


def _a2_one(pairs):
    old = [(e, u) for e, u in pairs if e.startswith("old")]
    new = [(e, u) for e, u in pairs if e.startswith("new")]
    m = np.mean([not u.startswith("old") for _, u in old]) if old else None
    f = np.mean([u.startswith("old") for _, u in new]) if new else None
    if m is None and f is None:
        return None, None
    return (m if m is not None else np.nan, f if f is not None else np.nan)


# ------------------------------------------------------------ A3 story recall
def a3_human():
    out = []
    for f in (HUMAN / "semantic-memory-story-recall").glob("run-*.json"):
        s = json.load(open(f)).get("summary", {})
        if s.get("bleu") is not None:
            out.append((float(s["bleu"]), float(s.get("embeddingSimilarity", np.nan)),
                        float(s.get("recallWordCount", np.nan))))
    return out


def a3_model(model_dir):
    out = []
    for line in open(ROOT / model_dir / "tasks/wm_semantic_story_recall.jsonl"):
        m = json.loads(line).get("metrics", {})
        if m.get("bleuScore") is not None:
            out.append((float(m["bleuScore"]), float(m.get("embeddingSimilarity", np.nan)),
                        len(str(json.loads(line).get("recall_text", "")).split())))
    return out


# ---------------------------------------------------------------------- report
MODELS = sys.argv[1:] or [
    "runs/compactor/claude-opus-4-6",
    "runs/compactor/qwen_qwen3-30b-a3b-instruct-2507",
    "runs/compactor/qwen_qwen3-8b_false",
]

print("A1  DIGIT SPAN threshold sharpness")
print(f"    {'source':<44}{'n':>5}{'sub_span_fail':>15}{'supra_span_hit':>16}")
h = a1_human()
print(f"    {'HUMANS':<44}{len(h):>5}"
      f"{pct(np.nanmean([x[0] for x in h])):>15}{pct(np.nanmean([x[1] for x in h])):>16}")
for md in MODELS:
    v = a1_model(md)
    print(f"    {Path(md).name:<44}{len(v):>5}"
          f"{pct(np.nanmean([x[0] for x in v])):>15}{pct(np.nanmean([x[1] for x in v])):>16}")

def a2_ratio_ci(miss, fa, n_boot=2000, seed=0):
    """Bootstrap the miss/false-alarm ratio.

    The ratio's denominator is small: word recognition terminates at 3 strikes,
    so each participant contributes only the trials they attempted.  A harness
    change that shifts first_error_at moves this axis without fixing the
    asymmetry, so the CI and the trial count are reported alongside it.
    """
    rng = np.random.default_rng(seed)
    miss, fa = np.asarray(miss, float), np.asarray(fa, float)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, miss.size, miss.size)
        d = np.nanmean(fa[idx])
        out.append(np.nanmean(miss[idx]) / d if d else np.nan)
    return np.nanpercentile(out, [2.5, 97.5])


print("\nA2  WORD RECOGNITION error asymmetry")
print(f"    {'source':<44}{'n':>5}{'miss_rate':>11}{'fa_rate':>9}{'miss/fa':>9}"
      f"{'ratio_ci':>18}{'trials':>8}")
hm, hf, ht = a2_human()
lo, hi = a2_ratio_ci(hm, hf)
print(f"    {'HUMANS':<44}{len(hm):>5}{pct(np.nanmean(hm)):>11}{pct(np.nanmean(hf)):>9}"
      f"{pct(np.nanmean(hm)/np.nanmean(hf) if np.nanmean(hf) else np.nan):>9}"
      f"  [{lo:.2f},{hi:.2f}]".rjust(18) + f"{np.mean(ht):>8.1f}")
for md in MODELS:
    m, f, t = a2_model(md)
    r = np.nanmean(m)/np.nanmean(f) if f and np.nanmean(f) else np.nan
    lo, hi = a2_ratio_ci(m, f)
    print(f"    {Path(md).name:<44}{len(m):>5}{pct(np.nanmean(m)):>11}"
          f"{pct(np.nanmean(f)):>9}{pct(r):>9}"
          f"  [{lo:.2f},{hi:.2f}]".rjust(18) + f"{np.mean(t):>8.1f}")

print("\nA3  STORY RECALL verbatim vs gist")
print(f"    {'source':<44}{'n':>5}{'BLEU':>9}{'embed_sim':>11}{'words':>8}")
a = a3_human()
print(f"    {'HUMANS':<44}{len(a):>5}{pct(np.nanmean([x[0] for x in a])):>9}"
      f"{pct(np.nanmean([x[1] for x in a])):>11}{pct(np.nanmean([x[2] for x in a])):>8}")
for md in MODELS:
    v = a3_model(md)
    if not v:
        print(f"    {Path(md).name:<44}    - (no bleu logged)")
        continue
    print(f"    {Path(md).name:<44}{len(v):>5}{pct(np.nanmean([x[0] for x in v])):>9}"
          f"{pct(np.nanmean([x[1] for x in v])):>11}{pct(np.nanmean([x[2] for x in v])):>8}")
