"""N-Back, compared at matched granularity per n-level.

The problem this fixes. `score.py` scores the model as one observation per
`(participant, n_level)` row — 150 points — while a human is one pooled
observation over all three of that participant's levels — 53 points. The model's
score distribution therefore spans the full level effect (n=1 0.993, n=2 0.779,
n=3 0.360) while the human's averages it away, and `W_1` between them is inflated
by that structural difference alone.

Measured size of the artifact on the baseline:

    per-row (what humanlikeness currently uses)   HL 0.7909,  sd 0.321
    pooled to human granularity                   HL 0.8443,  sd 0.102
    human                                                     sd 0.081

So the "model is bimodal on N-Back, humans are tightly clustered" reading was an
artifact of scoring granularity: 35% of rows sitting at exactly 1.0 are precisely
the 50 n=1 rows. Pooled, the model's shape matches the human shape and the deficit
is a clean mean shift, 0.710 against 0.866. This is the same family of defect as
the digit-span protocol mismatch and the variable-mapping exposure mismatch: the
two sides were not measuring the same thing.

Pooling both sides is one fix. Splitting both sides by level is the better one,
because the human records support it — `payload.trials` carries a `level` field with
14 scored trials at each of n=1,2,3 (plus 8 discarded `phase == "practice"` trials).
That gives three matched comparisons instead of one and localises the deficit
instead of averaging over it.

Usage:
    python meta_harness/nback_levels.py
    python meta_harness/nback_levels.py --run <run_dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import score as S  # noqa: E402

HUMAN_DIR = ROOT / "runs/human/working-memory-nback"
LEVELS = (1, 2, 3)


def human_by_level() -> dict[int, np.ndarray]:
    """Per-participant accuracy at each n-level, practice trials excluded."""
    acc: dict[int, list[float]] = {n: [] for n in LEVELS}
    for f in sorted(glob.glob(str(HUMAN_DIR / "*.json"))):
        trials = (json.load(open(f)).get("payload") or {}).get("trials") or []
        by: dict[int, list[bool]] = defaultdict(list)
        for t in trials:
            if t.get("phase") == "practice":
                continue
            lvl = t.get("level")
            if lvl in LEVELS:
                by[lvl].append(bool(t.get("correct")))
        for n, vals in by.items():
            if vals:
                acc[n].append(sum(vals) / len(vals))
    return {n: np.asarray(v, float) for n, v in acc.items()}


def model_by_level(run_dir: Path) -> dict[int, np.ndarray]:
    path = Path(run_dir) / "tasks/wm_nback.jsonl"
    acc: dict[int, list[float]] = {n: [] for n in LEVELS}
    if not path.exists():
        return {n: np.asarray([], float) for n in LEVELS}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != "C2":
            continue
        n = r.get("n_level")
        if n in LEVELS and r.get("acc_over_14") is not None:
            acc[n].append(float(r["acc_over_14"]))
    return {n: np.asarray(v, float) for n, v in acc.items()}


def diagnostics(run_dir: Path) -> dict[int, dict[str, float]]:
    """Answered / accuracy-over-answered / slot use per level.

    These separate *omission* from *error*: at n=3 the baseline answers only 6.82
    of 14 trials but is 0.737 accurate on the ones it does answer, so its low score
    is mostly silence rather than bad judgement.
    """
    path = Path(run_dir) / "tasks/wm_nback.jsonl"
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    def _nan(v):
        """None -> nan. These keys are present-but-null when undefined, so the
        default of `.get(k, np.nan)` never fires."""
        return np.nan if v is None else float(v)

    def _round_or_none(v, nd=4):
        """All-nan input means the quantity is undefined for every row here, which
        is a fact to report rather than a 0.0 to average into something."""
        return None if v is None or np.isnan(v) else round(float(v), nd)

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for n in LEVELS:
        sel = [r for r in rows if r.get("n_level") == n]
        if not sel:
            continue
        # model_parsed_buffer is a DICT keyed by buffer position ("1".."3"), not a
        # list. Iterating it directly yields the keys, which silently scores every
        # slot as non-compliant -- so take .values() explicitly.
        buf: list[str] = []
        for r in sel:
            mb = r.get("model_parsed_buffer") or {}
            buf.extend(str(v) for v in
                       (mb.values() if isinstance(mb, dict) else mb))
        # `.get(k, np.nan)` is not enough: these keys EXIST with an explicit null
        # when the quantity is undefined, and `None` is what comes back, which
        # np.nanmean cannot sum. That is not a corrupt row -- `acc_over_answered` is
        # null exactly when `answered` is 0, so a candidate that suppresses
        # responses entirely produces nulls legitimately. episodic_reset is the
        # first run to do so: at n=1, 36 of its 50 rows answered nothing.
        # Coercing to nan keeps those rows in the `answered` mean, where they
        # belong, while excluding them from the accuracy means, where they are
        # genuinely undefined.
        def _vals(key):
            return [_nan(r.get(key)) for r in sel]

        n_undefined = sum(1 for r in sel if r.get("acc_over_answered") is None)
        out[n] = {
            "answered": round(float(np.nanmean(_vals("answered"))), 2),
            "acc_over_answered": _round_or_none(np.nanmean(_vals("acc_over_answered"))),
            "acc_over_14": _round_or_none(np.nanmean(_vals("acc_over_14"))),
            # How many participants answered nothing at all. Without this an
            # acc_over_answered averaged over the few who did respond looks healthy
            # while most of the sample was silent.
            "n_rows": len(sel),
            "n_no_answers": n_undefined,
            "keys_held": round(
                float(np.mean([len(r.get("final_kv") or {}) for r in sel])), 2),
            # Compliance with the instruction to stay silent during the buffer
            # period. A rise here cannot be bought by answering more eagerly, so it
            # is the cleanest available check on whether behaviour really changed.
            "buffer_no_response_frac": (
                round(sum("no response" in b.lower() for b in buf) / len(buf), 4)
                if buf else None),
        }
    return out


def report(run_dir: Path) -> dict[str, Any]:
    h, m = human_by_level(), model_by_level(run_dir)
    per_level = {}
    for n in LEVELS:
        if h[n].size and m[n].size:
            per_level[n] = {
                "human_mean": round(float(h[n].mean()), 4),
                "model_mean": round(float(m[n].mean()), 4),
                "humanlikeness": round(float(S.humanlikeness(h[n], m[n])), 4),
                "n_human": int(h[n].size), "n_model": int(m[n].size),
            }

    # Pooled to the granularity score.py uses for humans, for comparison with the
    # per-row number that humanlikeness currently reports.
    rows = model_by_level(run_dir)
    path = Path(run_dir) / "tasks/wm_nback.jsonl"
    pooled = np.asarray([], float)
    if path.exists():
        by_pid: dict[Any, list[float]] = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("acc_over_14") is not None:
                by_pid[r.get("participant_id")].append(float(r["acc_over_14"]))
        pooled = np.asarray([np.mean(v) for v in by_pid.values()], float)

    flat = np.concatenate([rows[n] for n in LEVELS if rows[n].size]) if any(
        rows[n].size for n in LEVELS) else np.asarray([], float)
    hp = S.human_scores("nback")
    return {
        "per_level": per_level,
        "per_row_humanlikeness": (round(float(S.humanlikeness(hp, flat)), 4)
                                  if flat.size else None),
        "pooled_humanlikeness": (round(float(S.humanlikeness(hp, pooled)), 4)
                                 if pooled.size else None),
        "per_row_sd": round(float(flat.std()), 4) if flat.size else None,
        "pooled_sd": round(float(pooled.std()), 4) if pooled.size else None,
        "human_sd": round(float(hp.std()), 4),
        "diagnostics": diagnostics(run_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",
                    default="meta_harness/runs/iter0/baseline/"
                            "Qwen_Qwen3-30B-A3B-Instruct-2507")
    args = ap.parse_args()
    rep = report(Path(args.run))
    print(json.dumps(rep, indent=2))

    h = human_by_level()
    print("\nhuman per-level:", {n: round(float(h[n].mean()), 4) for n in LEVELS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
