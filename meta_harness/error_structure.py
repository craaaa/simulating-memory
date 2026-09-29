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


# =============================================================================
# M3-M6: error-shape measures, REPORT ONLY.
#
# None of what follows gates, floors, or enters mean_humanlikeness_search, and
# none of it is added to score_candidate.axes(). Promotion is a later decision;
# `report_error_shape.py` is the only delivery surface.
#
# Each measure has its OWN unit and none of them is humanlikeness. Where
# per-participant values exist on both sides at matched granularity, the measure
# also reports `1 - W_1` OVER THOSE VALUES, in the measure's own unit -- that makes
# the error shape commensurable with the score metric without merging into it.
# Where per-participant values do not exist (or are one observation each), the
# measure reports pooled only, with n, and says so.
# =============================================================================


def one_minus_w1(human_vals, model_vals) -> float | None:
    """1 - W_1 between two sets of per-participant values, in THEIR unit.

    Deliberately not called `humanlikeness`: the acceptance metric is 1 - W_1 over
    per-participant *proportion correct*, and these are miss rates, error-class
    shares, lag-bin accuracies and cosine similarities. Same functional form, a
    different quantity, so it must not be averaged into the 8-task mean.
    """
    h = np.asarray([v for v in human_vals if v is not None and not np.isnan(v)], float)
    m = np.asarray([v for v in model_vals if v is not None and not np.isnan(v)], float)
    if h.size == 0 or m.size == 0:
        return None
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import score as _S
    w = _S.wasserstein_1d(h, m)
    return None if np.isnan(w) else round(1.0 - float(w), 4)


def _mean_or_none(vals, nd=4):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.mean(v)), nd) if v else None


# ------------------------------------------------- M3  n-back error structure
#
# UNITS.
#   miss_rate  = P(respond "different" | the letter IS an n-back target), in [0,1],
#                reference point the target trials that got a parseable answer.
#   fa_rate    = P(respond "same"      | the letter is NOT a target), same form.
#   ratio      = miss_rate / fa_rate, dimensionless, reported the way A2 reports it
#                (with a bootstrap CI from `a2_ratio_ci`). 1.0 is the value random
#                responding drives it to.
#   unanswered_rate = share of scored trials with no parseable label. Reported
#                SEPARATELY and never folded into miss_rate: at baseline the model
#                leaves a large fraction of trials unanswered, and counting silence
#                as a miss would let a candidate move the ratio by going quiet.
#   lure_fa_rate = fa_rate restricted to non-target trials whose letter matches at
#                lag n-1 or n+1. Pooled only (see MIN_LURE_TRIALS).
#   position profile = accuracy by third of the block's DEFINED trials.
#
# PROTOCOL FILTERS, and this is where the project has been wrong before:
#   1. Human `payload.trials` include a practice phase. Dropped: `phase ==
#      "practice"` and any block whose name starts with "training".
#   2. Human granularity. n-back humanlikeness pools a participant's three levels
#      into one score; M3 does NOT pool. It compares per (participant, level),
#      which is the granularity the model rows already have -- the same fix
#      `nback_levels.py` exists to make.
#   3. The first n trials of every block are LEAD-IN and are dropped on both sides.
#      On the model side they are not in `per_trial` at all (they live in
#      `buffer_letters`), so dropping them on the human side is what makes the two
#      positions line up: model scored trial k == human block trial n+k.
#      This is not a cosmetic filter. In the human records 112 of the 318 lead-in
#      trials are logged `target: true`, which is impossible -- there is no letter
#      n positions back -- so those trials are scored against an unsatisfiable
#      label. See `logs/error_shape_measures.md`.
#   4. `target` is recomputed from the letter sequence on both sides rather than
#      read from the log. Post-lead-in the recomputation agrees with the logged
#      flag on 1908/1908 human trials and 2100/2100 model trials, which is the
#      check that licenses using it.
MIN_LURE_TRIALS = 30       # below this, lure_fa_rate prints insufficient
# `ratio` is a quotient whose DENOMINATOR is fa_rate, and fa_rate is small: the
# human n=1 level has 12 false alarms in 438 non-target trials. A candidate that
# drives its own fa_rate toward zero manufactures an enormous ratio out of a
# handful of false alarms -- the same failure mode A4's fixed threshold had. So the
# point estimate is withheld below this many false alarms, and even above it the
# ratio must be read alongside `ratio_ci_over_cells`, never alone.
MIN_FALSE_ALARMS_FOR_RATIO = 30
NBACK_LEVELS = (1, 2, 3)
_LEAD_BLOCK = "training"


def _nback_response_label(resp) -> str | None:
    """Human `response` is 'target'/'nontarget'; model `model_label` is
    'Same'/'Different'/None. Mapped onto one vocabulary: same / different / None."""
    if resp is None:
        return None
    s = str(resp).strip().lower()
    if s in ("target", "same"):
        return "same"
    if s in ("nontarget", "non-target", "different"):
        return "different"
    return None


def _nback_trial_records(letters: list[str], responses: list, n: int,
                         participant) -> list[dict]:
    """One record per DEFINED trial: index n..len-1 of the letter stream.

    `letters` is the full stream including the n lead-in letters; `responses` is
    aligned to it, with None where no response is defined or recorded.
    """
    out = []
    n_defined = len(letters) - n
    for i in range(n, len(letters)):
        pos = i - n + 1                     # 1-based defined-trial index
        target = letters[i] == letters[i - n]
        lure = (not target) and any(
            0 <= i - d < i and letters[i] == letters[i - d]
            for d in (n - 1, n + 1) if d >= 1)
        out.append({
            "participant": participant, "level": n, "position": pos,
            "n_defined": n_defined,
            "letter": letters[i], "target": target, "lure": lure,
            "response": _nback_response_label(responses[i] if i < len(responses) else None),
        })
    return out


def nback_human_trials() -> list[dict]:
    """Defined, non-practice human n-back trials, one record per trial."""
    out: list[dict] = []
    for f in sorted((HUMAN / "working-memory-nback").glob("run-*.json")):
        trials = (json.load(open(f)).get("payload") or {}).get("trials") or []
        by_block: dict[str, list[dict]] = defaultdict(list)
        for t in trials:
            block = str(t.get("block") or "")
            if t.get("phase") == "practice" or block.startswith(_LEAD_BLOCK):
                continue
            by_block[block].append(t)
        for block, ts in by_block.items():
            # `level` is null in 4 of the 57 records, so it is read from the block
            # name ("2-back"), which is present in all of them.
            m = re.match(r"(\d+)", block)
            if not m:
                continue
            n = int(m.group(1))
            if n not in NBACK_LEVELS:
                continue
            ts = sorted(ts, key=lambda t: int(t.get("trial") or 0))
            letters = [str(t.get("letter") or "") for t in ts]
            resps = [t.get("response") for t in ts]
            out.extend(_nback_trial_records(letters, resps, n, Path(f).name))
    return out


def nback_model_trials(run_dir) -> list[dict]:
    """Defined model n-back trials. `buffer_letters` supplies the lead-in so the
    two sides' `position` means the same thing."""
    path = Path(run_dir) / "tasks/wm_nback.jsonl"
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != "C2":
            continue
        n = int(r.get("n_level") or 0)
        if n not in NBACK_LEVELS:
            continue
        letters = list(r.get("buffer_letters") or []) + list(r.get("trial_letters") or [])
        per = r.get("per_trial") or []
        resps = [None] * n + [t.get("model_label") for t in per]
        out.extend(_nback_trial_records(
            letters, resps, n, f"{r.get('participant_id')}"))
    return out


def nback_error_profile(trials: list[dict], min_lure: int = MIN_LURE_TRIALS) -> dict:
    """Miss / false-alarm / lure / position structure, overall and per level."""
    def _one(sel: list[dict]) -> dict:
        tg = [t for t in sel if t["target"]]
        nt = [t for t in sel if not t["target"]]
        tg_a = [t for t in tg if t["response"] is not None]
        nt_a = [t for t in nt if t["response"] is not None]
        miss = (float(np.mean([t["response"] == "different" for t in tg_a]))
                if tg_a else None)
        fa = (float(np.mean([t["response"] == "same" for t in nt_a]))
              if nt_a else None)
        lure = [t for t in nt_a if t["lure"]]
        nonlure = [t for t in nt_a if not t["lure"]]
        n_fa = sum(1 for t in nt_a if t["response"] == "same")
        ratio_ok = n_fa >= MIN_FALSE_ALARMS_FOR_RATIO
        return {
            "n_trials": len(sel),
            "n_target": len(tg), "n_nontarget": len(nt),
            "n_answered": len(tg_a) + len(nt_a),
            "n_false_alarms": n_fa,
            "unanswered_rate": (round(1.0 - (len(tg_a) + len(nt_a)) / len(sel), 4)
                                if sel else None),
            "miss_rate": round(miss, 4) if miss is not None else None,
            "fa_rate": round(fa, 4) if fa is not None else None,
            "ratio": (round(miss / fa, 4)
                      if (ratio_ok and miss is not None and fa) else None),
            "ratio_note": (None if ratio_ok else
                           f"insufficient (false alarms n={n_fa}); minimum "
                           f"{MIN_FALSE_ALARMS_FOR_RATIO} -- the ratio's denominator"),
            "n_lure_trials": len(lure),
            "lure_fa_rate": (round(float(np.mean([t["response"] == "same" for t in lure])), 4)
                             if len(lure) >= min_lure else None),
            "lure_note": (None if len(lure) >= min_lure
                          else f"insufficient (n={len(lure)}); minimum {min_lure}"),
            "nonlure_fa_rate": (round(float(np.mean([t["response"] == "same" for t in nonlure])), 4)
                                if nonlure else None),
        }

    # Per (participant, level), which is the matched granularity -- NOT pooled over
    # a participant's three levels, and NOT per-row-over-all-levels.
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        cells[(t["participant"], t["level"])].append(t)
    per_cell = {}
    for key, sel in cells.items():
        tg_a = [t for t in sel if t["target"] and t["response"] is not None]
        nt_a = [t for t in sel if not t["target"] and t["response"] is not None]
        thirds: list[list[bool]] = [[], [], []]
        for t in sel:
            if t["response"] is None:
                continue
            b = min(2, 3 * (t["position"] - 1) // max(t["n_defined"], 1))
            thirds[b].append((t["response"] == "same") == t["target"])
        per_cell[key] = {
            "miss": (float(np.mean([t["response"] == "different" for t in tg_a]))
                     if tg_a else None),
            "fa": (float(np.mean([t["response"] == "same" for t in nt_a]))
                   if nt_a else None),
            "thirds": [float(np.mean(b)) if b else None for b in thirds],
        }

    out: dict = {
        "unit": {
            "miss_rate": "P(respond different | target), over answered target trials",
            "fa_rate": "P(respond same | non-target), over answered non-target trials",
            "ratio": ("miss_rate / fa_rate, dimensionless (random responding -> 1.0); "
                      "withheld below 30 false alarms and to be read only alongside "
                      "ratio_ci_over_cells"),
            "unanswered_rate": "share of defined trials with no parseable label",
            "position_thirds": "accuracy over answered trials in each third of the block",
        },
        "overall": _one(trials),
        "by_level": {str(n): _one([t for t in trials if t["level"] == n])
                     for n in NBACK_LEVELS if any(t["level"] == n for t in trials)},
        "n_cells": len(per_cell),
        "per_cell": {f"{k[0]}|n{k[1]}": v for k, v in per_cell.items()},
    }
    miss = [v["miss"] for v in per_cell.values()]
    fa = [v["fa"] for v in per_cell.values()]
    if any(m is not None for m in miss) and any(f is not None for f in fa):
        lo, hi = a2_ratio_ci([np.nan if m is None else m for m in miss],
                             [np.nan if f is None else f for f in fa])
        out["ratio_ci_over_cells"] = [round(float(lo), 3), round(float(hi), 3)]
    return out


# -------------------------------- M4  word-recognition lag curve (task is broken)
#
# READ THIS BEFORE USING THE NUMBER. `word_recognition` does not isolate memory on
# either side. All 100 (human: up to ~100) test words are presented in one visible
# list, and "Old" means "this word appeared EARLIER IN THIS LIST". So a participant
# who simply re-reads the list can answer correctly with no retention at all, and a
# lag curve computed on it measures list inspection at least as much as memory
# decay. It is computed here because it is the only lag structure the released data
# supports, and it is labelled `task_is_broken: true` so no reader can take it for
# a clean recency curve.
#
# UNIT: accuracy on OLD trials (proportion correct), in [0,1], within a lag bin,
# where lag = trials elapsed since that word's FIRST appearance in the list.
#
# PROTOCOL FILTER: human sessions terminate at 3 strikes and run 4 to 102 trials
# (median 32); the model runs all 100. Lag coverage therefore differs per
# participant, so every bin is computed PER PARTICIPANT and then averaged, never
# pooled across unequal exposure. `n_participants` per bin is reported because a
# late bin is carried by the few long human sessions.
LAG_BINS = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9))
LAG_BIN_LABELS = ("1-2", "3-5", "6-10", "11-20", "21+")
MIN_LAG_PARTICIPANTS = 10


def _lag_bin(lag: int) -> int | None:
    for i, (lo, hi) in enumerate(LAG_BINS):
        if lo <= lag <= hi:
            return i
    return None


def _lag_one_session(words: list[str], correct: list[bool]) -> list[float | None]:
    """Per-bin accuracy on old trials for one participant."""
    bins: list[list[bool]] = [[] for _ in LAG_BINS]
    first: dict[str, int] = {}
    for i, w in enumerate(words, start=1):
        if w in first:
            b = _lag_bin(i - first[w])
            if b is not None:
                bins[b].append(bool(correct[i - 1]))
        else:
            first[w] = i
    return [float(np.mean(b)) if b else None for b in bins]


def wr_lag_human() -> list[list[float | None]]:
    out = []
    for f in sorted((HUMAN / "working-memory-word-recognition").glob("run-*.json")):
        resp = (json.load(open(f)).get("payload") or {}).get("responses") or []
        if not resp:
            continue
        out.append(_lag_one_session([str(r["word"]) for r in resp],
                                    [bool(r["correct"]) for r in resp]))
    return out


def wr_lag_model(run_dir) -> list[list[float | None]]:
    path = Path(run_dir) / "tasks/wm_word_recognition.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        per = r.get("per_trial") or []
        if not per:
            continue
        out.append(_lag_one_session([str(t["word"]) for t in per],
                                    [bool(t["correct"]) for t in per]))
    return out


def wr_lag_summary(sessions: list[list[float | None]],
                   min_participants: int = MIN_LAG_PARTICIPANTS) -> dict:
    bins = []
    for i, label in enumerate(LAG_BIN_LABELS):
        vals = [s[i] for s in sessions if s[i] is not None]
        bins.append({
            "lag": label,
            "n_participants": len(vals),
            "accuracy": (round(float(np.mean(vals)), 4)
                         if len(vals) >= min_participants else None),
            "note": (None if len(vals) >= min_participants
                     else f"insufficient (n={len(vals)} participants); "
                          f"minimum {min_participants}"),
            "per_participant": vals,
        })
    return {
        "task_is_broken": True,
        "caveat": ("all test words are visible in one list and 'Old' means "
                   "'appeared earlier in this list', so this curve measures list "
                   "inspection as well as memory"),
        "unit": "accuracy on old trials (proportion correct) within a lag bin",
        "n_sessions": len(sessions),
        "bins": bins,
    }


# ------------------------------- M5  which wrong option: distractor choice / index
#
# TWO DIFFERENT MEASURES, kept apart on purpose.
#
# (a) `distractor_choice` -- narrative_qa and factual_qa, which have FOUR options,
#     so "which of the three wrong options" is a real question.
#     UNITS, both probabilities in [0,1] with chance = 1/3 for three distractors:
#       within_side_agreement : P(two errors drawn from the same question, without
#                               replacement, chose the SAME distractor). Computed as
#                               sum_d c_d(c_d - 1) / sum_d c_d(C - 1), which is
#                               unbiased at small counts -- a plain Herfindahl is
#                               not (with 2 errors its minimum is 0.5).
#       cross_side_agreement  : P(a human error and a model error drawn from the
#                               same question chose the same distractor)
#                               = sum_q sum_d h_qd*m_qd / sum_q H_q*M_q.
#     This is the "do the two sides concentrate on the same distractors" number.
#
# (b) `error_index_profile` -- craft_task and map_task have exactly TWO options, so
#     picking the wrong one carries no information: distractor identity is forced.
#     Naming it a distractor measure there would be a category error. What the data
#     does support is WHERE in the 5-question block the error falls, as an error
#     rate per question index, per participant then averaged.
#
# PROTOCOL FILTERS.
#   * The model's chosen option is NOT persisted per question in the wm_* rows, so
#     it is recovered by re-running that task's own `parse_answers` over
#     `recall_raw`. Validated: the recovered choices reproduce the stored
#     `metrics.correct` on 50/50 narrative_qa rows and 150/150 craft_task rows of
#     `iter8repA/baseline`, asserted in `test_error_shape.py`. Persisting the chosen
#     letter is a one-line change in `bench/tasks/wm_*` and is worth making later;
#     it is deliberately NOT made here, since this work is evaluator-side only.
#   * narrative_qa humans saw TWO question banks: 471 of their 520 questions come
#     from `data/narrative_QA.json` and 49 from `data/narrative_QA_easy.json`,
#     while the model only ever sees the former. The comparison is restricted to
#     the shared bank and the 49 easy-bank questions are dropped. factual_qa has no
#     such split: all 530 human questions match `wikipedia_10docs_questions.json`.
#   * Option letters are NOT shuffled between the sides: the human record's
#     `correctAnswer` letter agrees with the bank's `answer` on 471/471 narrative
#     and 530/530 factual questions, which is what makes letter-level comparison
#     legitimate.
MIN_DISTRACTOR_QUESTIONS = 10
MIN_DISTRACTOR_PAIRS = 30

_MCQ_BANKS = {
    "narrative_qa": ("data/narrative_QA.json", "items", "questions", "options"),
    "factual_qa": ("data/wikipedia_10docs_questions.json", "results", "questions",
                   "choices"),
}


def mcq_bank(task: str) -> dict[str, dict]:
    """question text -> {'answer': letter, 'options': {letter: text}}."""
    rel, top, qkey, okey = _MCQ_BANKS[task]
    data = json.loads((ROOT / rel).read_text())
    out: dict[str, dict] = {}
    for it in data[top]:
        for q in it.get(qkey) or []:
            out[str(q["question"]).strip()] = {
                "answer": str(q.get("answer") or "").upper(),
                "options": q.get(okey) or {},
            }
    return out


_HUMAN_MCQ_DIR = {"narrative_qa": "narrative-qa", "factual_qa": "factual-qa"}


def distractor_counts_human(task: str) -> dict[str, collections.Counter]:
    """question text -> Counter over chosen wrong letters, humans."""
    bank = mcq_bank(task)
    out: dict[str, collections.Counter] = defaultdict(collections.Counter)
    for f in sorted((HUMAN / _HUMAN_MCQ_DIR[task]).glob("run-*.json")):
        payload = json.load(open(f)).get("payload") or {}
        for q in payload.get("responses") or []:
            text = str(q.get("question") or "").strip()
            if text not in bank:
                continue            # narrative_qa easy-bank question: dropped
            if q.get("correct"):
                continue
            sel = str(q.get("userAnswer") or "").strip().upper()
            if sel and sel != bank[text]["answer"]:
                out[text][sel] += 1
    return dict(out)


def distractor_counts_model(run_dir, task: str) -> dict[str, collections.Counter]:
    """question text -> Counter over chosen wrong letters, model.

    `recall_raw` is re-parsed with the task's own parser; see the filter note above.
    """
    parse = _mcq_parser(task)
    path = Path(run_dir) / f"tasks/wm_{task}.jsonl"
    out: dict[str, collections.Counter] = defaultdict(collections.Counter)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        answers = parse(r.get("recall_raw") or "")
        for i, q in enumerate(r.get("questions") or [], start=1):
            gold = str(q.get("answer") or "").strip().upper()
            sel = answers.get(i)
            if sel is None or sel == gold:
                continue
            out[str(q.get("question") or q.get("prompt") or "").strip()][sel] += 1
    return dict(out)


def _mcq_parser(task: str):
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    if task == "narrative_qa":
        from bench.tasks.narrative_qa import parse_answers
    elif task == "factual_qa":
        from bench.tasks.factual_qa import parse_answers
    elif task == "craft_task":
        from bench.tasks.craft_task import parse_answers
    elif task == "map_task":
        from bench.tasks.map_task import parse_answers
    else:
        raise KeyError(task)
    return parse_answers


def _within_agreement(counts: dict[str, collections.Counter]) -> tuple[float | None, int, int]:
    """Unbiased P(two errors on the same question chose the same distractor)."""
    num = den = 0
    nq = 0
    for c in counts.values():
        C = sum(c.values())
        if C < 2:
            continue
        nq += 1
        num += sum(v * (v - 1) for v in c.values())
        den += C * (C - 1)
    return (num / den if den else None), nq, den


def distractor_profile(counts: dict[str, collections.Counter],
                       n_distractors: int = 3) -> dict:
    agree, nq, pairs = _within_agreement(counts)
    total = sum(sum(c.values()) for c in counts.values())
    ok = nq >= MIN_DISTRACTOR_QUESTIONS and pairs >= MIN_DISTRACTOR_PAIRS
    return {
        "unit": (f"P(two errors on the same question chose the same distractor); "
                 f"chance = {1.0 / n_distractors:.4f} for {n_distractors} distractors"),
        "chance": round(1.0 / n_distractors, 4),
        "n_errors": total,
        "n_questions_with_errors": len(counts),
        "n_questions_with_2plus_errors": nq,
        "n_ordered_pairs": pairs,
        "within_side_agreement": round(agree, 4) if (ok and agree is not None) else None,
        "note": (None if ok else
                 f"insufficient (questions with 2+ errors n={nq}, ordered pairs "
                 f"n={pairs}); minimum {MIN_DISTRACTOR_QUESTIONS} questions and "
                 f"{MIN_DISTRACTOR_PAIRS} pairs"),
    }


def cross_side_distractor_agreement(human: dict[str, collections.Counter],
                                    model: dict[str, collections.Counter],
                                    n_distractors: int = 3) -> dict:
    num = den = 0
    shared = 0
    for text, hc in human.items():
        mc = model.get(text)
        if not mc:
            continue
        H, M = sum(hc.values()), sum(mc.values())
        if not H or not M:
            continue
        shared += 1
        num += sum(hc[d] * mc.get(d, 0) for d in hc)
        den += H * M
    ok = shared >= MIN_DISTRACTOR_QUESTIONS and den >= MIN_DISTRACTOR_PAIRS
    return {
        "unit": ("P(a human error and a model error on the same question chose the "
                 f"same distractor); chance = {1.0 / n_distractors:.4f}"),
        "chance": round(1.0 / n_distractors, 4),
        "n_shared_questions": shared,
        "n_pairs": den,
        "cross_side_agreement": round(num / den, 4) if (ok and den) else None,
        "note": (None if ok else
                 f"insufficient (shared questions n={shared}, pairs n={den})"),
    }


# (b) binary-choice tasks: error rate by question index within the block.
_HUMAN_INDEX_DIR = {"craft_task": "procedure-memory-craft-task",
                    "map_task": "procedure-memory-map-task"}


def error_index_human(task: str) -> list[list[float | None]]:
    """Per participant, error rate at each question index 1..N of a block."""
    out = []
    for f in sorted((HUMAN / _HUMAN_INDEX_DIR[task]).glob("run-*.json")):
        payload = json.load(open(f)).get("payload") or {}
        by_idx: dict[int, list[bool]] = defaultdict(list)
        for tr in payload.get("trials") or []:
            for q in tr.get("responses") or []:
                by_idx[int(q["index"])].append(not bool(q.get("correct")))
        if by_idx:
            n = max(by_idx)
            out.append([float(np.mean(by_idx[i])) if by_idx.get(i) else None
                        for i in range(1, n + 1)])
    return out


def error_index_model(run_dir, task: str) -> list[list[float | None]]:
    """Same, per model participant. Blocks are pooled within a participant, as on
    the human side (3 trials each)."""
    parse = _mcq_parser(task)
    path = Path(run_dir) / f"tasks/wm_{task}.jsonl"
    if not path.exists():
        return []
    by_p: dict[object, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        answers = parse(r.get("recall_raw") or "")
        pid = r.get("participant_id", r.get("repeat_index"))
        for i, q in enumerate(r.get("questions") or [], start=1):
            gold = str(q.get("answer") or "").strip().upper()
            by_p[pid][i].append(answers.get(i) != gold)
    out = []
    for pid, by_idx in by_p.items():
        n = max(by_idx)
        out.append([float(np.mean(by_idx[i])) if by_idx.get(i) else None
                    for i in range(1, n + 1)])
    return out


def error_index_summary(sessions: list[list[float | None]]) -> dict:
    n = max((len(s) for s in sessions), default=0)
    idx = []
    for i in range(n):
        vals = [s[i] for s in sessions if i < len(s) and s[i] is not None]
        idx.append({"index": i + 1, "n_participants": len(vals),
                    "error_rate": round(float(np.mean(vals)), 4) if vals else None,
                    "per_participant": vals})
    return {
        "unit": "error rate at that question index (proportion of that index's "
                "questions answered wrong), per participant then averaged",
        "not_a_distractor_measure": True,
        "why": "these tasks have exactly two options, so the wrong option is forced",
        "n_participants": len(sessions),
        "by_index": idx,
    }


# ----------------------- M6  story recall, gist similarity (complements A3)
#
# A3 enforces that recall is NOT verbatim (clipped 4-gram precision, human median
# 0.0192) and that it is roughly the human length (137.2 words). Neither says
# whether the CONTENT survived. `payload.evaluation.embeddingSimilarity` has been
# sitting in the human records unused; this is that quantity, made comparable.
#
# UNIT: cosine similarity between mean-pooled all-MiniLM-L6-v2 embeddings of the
# story and the recall, both truncated to the first 200 words, in [0,1]. Same
# embedder and same truncation as `bench.tasks.semantic_story_recall`.
#
# WHY RECOMPUTED ON THE HUMAN SIDE. The stored human number was produced by the web
# app, the model's by bench. They are the same embedder (verified: over the 53
# usable human records, stored mean 0.6041 / sd 0.1282 against recomputed 0.5911 /
# sd 0.1220, mean |diff| 0.0346, corr 0.9431) but not the same preprocessing, so the
# cache stores BOTH and the 1 - W_1 uses the recomputed column, which is the only
# one computed identically on the two sides.
#
# Reported per story as well as pooled, because the story mix differs: humans split
# 18 Eyespy / 15 Pieman / 13 Baseball / 7 Oregon Trail (plus 3 records with no
# story), while the model runs 50 of each.


def _embedder():
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from bench.tasks.semantic_story_recall import (  # noqa
        _load_sentence_transformer, embedding_similarity)
    return _load_sentence_transformer(), embedding_similarity


def gist_human(recompute: bool = True) -> list[dict]:
    """Per participant: stored and (optionally) recomputed gist similarity."""
    model = sim = None
    if recompute:
        try:
            model, sim = _embedder()
        except Exception:
            model = sim = None
    out = []
    for f in sorted((HUMAN / "semantic-memory-story-recall").glob("run-*.json")):
        rec = json.load(open(f))
        payload = rec.get("payload") or {}
        ev = payload.get("evaluation") or {}
        stored = ev.get("embeddingSimilarity")
        if stored is None:
            stored = (rec.get("summary") or {}).get("embeddingSimilarity")
        text = (payload.get("recallText") or "").strip()
        src = _transcript(payload.get("storyFile"))
        recomputed = None
        if sim is not None and model is not None and text and src:
            recomputed = sim(model, src, text)
        out.append({
            "participant": Path(f).name,
            "story": payload.get("storyName"),
            "stored": None if stored is None else float(stored),
            "recomputed": None if recomputed is None else float(recomputed),
        })
    return out


def gist_model(run_dir) -> list[dict]:
    path = Path(run_dir) / "tasks/wm_semantic_story_recall.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = (r.get("metrics") or {}).get("embeddingSimilarity")
        out.append({"participant": r.get("id"), "story": r.get("story_name"),
                    "stored": None if v is None else float(v),
                    "recomputed": None if v is None else float(v)})
    return out


def gist_summary(records: list[dict], column: str = "recomputed") -> dict:
    vals = [r[column] for r in records if r.get(column) is not None]
    by_story: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get(column) is not None:
            by_story[str(r.get("story"))].append(r[column])
    return {
        "unit": ("cosine similarity of all-MiniLM-L6-v2 embeddings of story vs "
                 "recall, both truncated to 200 words, in [0,1]"),
        "column": column,
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 4) if vals else None,
        "sd": round(float(np.std(vals)), 4) if vals else None,
        "by_story": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4)}
                     for k, v in sorted(by_story.items())},
        "per_participant": vals,
    }


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
