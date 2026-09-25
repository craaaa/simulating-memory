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
      Humans reconstruct gist; a harness that stores text verbatim scores high
      even when its coverage score matches humans.

      Measured two ways, and only the second is enforced.  `bleu` is what the
      released data ships and is kept for continuity with the five runs already
      scored against it, but BLEU's brevity penalty makes it a length proxy
      rather than a verbatimness measure: pooled over 1000 model rows from five
      runs, Spearman(recall length, BLEU) = 0.822, and all 121 rows under 60
      words score exactly 0.0000 -- min and max alike.  It therefore cannot
      distinguish a perfectly verbatim short recall from an abstracted one, and
      it penalises any candidate that moves recall length toward the human
      135.6, which is the opposite of what the A3 word guard rewards.
      `verbatim_precision` is clipped modified 4-gram precision with no brevity
      penalty: the fraction of the recall's own 4-grams lifted from the source.
      Within the human records Spearman(length, precision) = 0.128, i.e. humans
      vary in length and in verbatimness independently, which is the property
      BLEU lacks.
"""
import collections
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "runs/human"
DATA = ROOT / "data"


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


def _num(v):
    """Some rows carry an explicit null for a judge metric (e.g. the judge
    failed or was skipped), so `.get(k, nan)` is not enough."""
    return np.nan if v is None else float(v)


def a3_model(model_dir):
    out = []
    for line in open(ROOT / model_dir / "tasks/wm_semantic_story_recall.jsonl"):
        row = json.loads(line)
        m = row.get("metrics") or {}
        if m.get("bleuScore") is not None:
            out.append((_num(m["bleuScore"]),
                        _num(m.get("embeddingSimilarity")),
                        len(str(row.get("recall_text") or "").split())))
    return out


# ------------------------------------- A3, length-free verbatimness (enforced)
_WORD = re.compile(r"[a-z0-9']+")
_TRANSCRIPT_CACHE: dict[str, str | None] = {}


def _toks(s):
    return _WORD.findall(str(s).lower())


def _ngrams(t, n):
    return collections.Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def _transcript(story_file):
    """Story transcripts live in data/ under their basename on both sides: the
    human records carry payload.storyFile like 'transcript/pieman_transcript.txt'
    while the model rows carry story_source_file, and only the basename is
    common to the two."""
    if not story_file:
        return None
    name = Path(str(story_file)).name
    if name not in _TRANSCRIPT_CACHE:
        p = DATA / name
        _TRANSCRIPT_CACHE[name] = p.read_text(errors="replace") if p.exists() else None
    return _TRANSCRIPT_CACHE[name]


def verbatim_precision(recall_text, story_file, n=4):
    """Clipped modified n-gram precision of the recall against its source, with
    NO brevity penalty, so a short recall is not forced to zero.

    Returns None when the recall is shorter than n tokens, which is the only case
    where the quantity is genuinely undefined rather than merely small.
    """
    src = _transcript(story_file)
    if src is None:
        return None
    cand, ref = _ngrams(_toks(recall_text), n), _ngrams(_toks(src), n)
    total = sum(cand.values())
    if total == 0:
        return None
    hit = sum(min(c, ref.get(g, 0)) for g, c in cand.items())
    return hit / total


def a3_precision_human(n=4):
    """Per-participant verbatim precision for the human records.

    Recomputed from payload.recallText rather than read from summary.bleu, which
    the web app precomputed and which carries the brevity penalty. This is the
    same recomputation path already validated for embeddingSimilarity (stored
    0.6041 vs recomputed 0.5911 over these records, correlation 0.943).
    """
    out = []
    for f in sorted((HUMAN / "semantic-memory-story-recall").glob("run-*.json")):
        payload = json.load(open(f)).get("payload") or {}
        text = (payload.get("recallText") or "").strip()
        if not text:
            continue
        p = verbatim_precision(text, payload.get("storyFile"), n)
        if p is not None:
            out.append(p)
    return out


def a3_precision_model(model_dir, n=4):
    """Per-participant verbatim precision for a model run."""
    out = []
    path = ROOT / model_dir / "tasks/wm_semantic_story_recall.jsonl"
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        p = verbatim_precision(row.get("recall_text") or "",
                               row.get("story_source_file") or row.get("story_name"), n)
        if p is not None:
            out.append(p)
    return out


def a2_ratio_ci(miss, fa, n_boot=2000, seed=0):
    """Bootstrap the miss/false-alarm ratio.

    The denominator is small: word recognition terminates at 3 strikes, so each
    participant contributes only the trials they attempted.  A harness change
    that shifts first_error_at moves this axis without fixing the asymmetry, so
    the CI and the trial count are reported alongside it.
    """
    rng = np.random.default_rng(seed)
    miss, fa = np.asarray(miss, float), np.asarray(fa, float)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, miss.size, miss.size)
        d = np.nanmean(fa[idx])
        out.append(np.nanmean(miss[idx]) / d if d else np.nan)
    return np.nanpercentile(out, [2.5, 97.5])


# ---------------------------------------------------------------------- report
def main() -> None:
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


if __name__ == "__main__":
    main()
