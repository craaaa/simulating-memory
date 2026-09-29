"""Evaluate predictions P1-P7 for job 18781213 against the bands registered before it ran.

WRITTEN BEFORE THE RUN PRODUCED ANY OUTPUT, deliberately, so the bands cannot be tuned to the
result. The bands are copied verbatim from `logs/predictions_iter12stage2.md`; if a number here
disagrees with that file, that file is authoritative and this script has a bug.

Every prediction resolves to one of:
    SUPPORTS   the observation fell in the registered support band
    REJECTS    it fell in the registered rejection band
    INCONCL    it fell between them, or the quantity is absent

INCONCL is a real outcome, not a failure to compute: P1 in particular has a deliberate gap
between its bands, and "inconclusive" must be reported rather than rounded toward support.

Usage:
    python meta_harness/check_iter12_predictions.py <run_dir> [<run_dir> ...]

Several run dirs are averaged per task where averaging is meaningful (humanlikeness), and
pooled where it is not (survival distributions, error-shape counts).
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "meta_harness"))

import score as S  # noqa: E402
import error_structure as ES  # noqa: E402
import nback_levels as NL  # noqa: E402
import protocol_match as PM  # noqa: E402

SUPPORTS, REJECTS, INCONCL = "SUPPORTS", "REJECTS", "INCONCL"

# --- registered references, from iter11postfix ------------------------------
REF = {
    # 84.80 was registered, but that figure is iter10postfix; iter11postfix, the actual
    # comparator, is 80.82 words. Recorded as the correction in the predictions file. The
    # BANDS are unchanged -- re-fitting a band after seeing the reference voids the
    # pre-registration.
    "wr_survival_mean": 80.82,      # words, iter11postfix/baseline
    "wr_humanlikeness": 0.5364,
    "wr_lag_spread": 0.043,         # |max - min| across the five lag bins
    "a2_ratio": 0.3901,             # word_recognition miss/false-alarm ratio
    "vm_survival_median": 4.0,      # questions
    "vm_humanlikeness": 0.9643,
    "nback": 0.9344, "narrative_qa": 0.9444,
    "semantic_story_recall": 0.9470, "craft_task": 0.8679,
}
# P6: 2x the measured 3-repeat spread, doubled because 3 repeats pin a spread to ~a third.
P6_BAND = {"nback": 0.0068, "narrative_qa": 0.0368,
           "semantic_story_recall": 0.0230, "craft_task": 0.0714}

results: list[tuple[str, str, str, str]] = []


def add(pid: str, verdict: str, observed: str, band: str) -> None:
    results.append((pid, verdict, observed, band))


def _rows(run: Path, task: str) -> list[dict]:
    p = run / "tasks" / f"wm_{task}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main(runs: list[Path]) -> int:
    # ---------------- P1: word_recognition survival and humanlikeness -------
    surv: list[int] = []
    for r in runs:
        for row in _rows(r, "word_recognition"):
            n = row.get("trials_presented")
            if n is None:
                pt = row.get("per_trial")
                n = len(pt) if pt else None
            if n is not None:
                surv.append(int(n))
    hl_wr = [S.humanlikeness(S.human_scores("word_recognition"),
                             S.llm_scores("word_recognition", r, "compactor")) for r in runs]
    hl_wr = [h for h in hl_wr if h == h]
    if not surv or not hl_wr:
        add("P1 word_recognition", INCONCL, "absent", "survival <60 & HL >0.60")
    else:
        m_surv, m_hl = st.mean(surv), st.mean(hl_wr)
        obs = f"survival mean {m_surv:.2f} words (median {st.median(surv)}), HL {m_hl:.4f}"
        if m_surv < 60 and m_hl > 0.60:
            v = SUPPORTS
        elif m_surv > 80 or abs(m_hl - REF["wr_humanlikeness"]) <= 0.04:
            v = REJECTS
        else:
            v = INCONCL
        add("P1 word_recognition", v, obs,
            f"supports survival<60 & HL>0.60; rejects survival>80 or |HL-{REF['wr_humanlikeness']}|<=0.04")

    # ---------------- P2: a lag effect exists at all ------------------------
    try:
        lag = ES.wr_lag_summary(ES.wr_lag_model(str(runs[0])))
        # `bins` is a list of {lag, n_participants, accuracy, note, per_participant}.
        bins = [b for b in (lag.get("bins") or [])
                if isinstance(b.get("accuracy"), (int, float))]
        vals = [float(b["accuracy"]) for b in bins]
        if len(vals) < 3:
            add("P2 lag effect", INCONCL, f"only {len(vals)} bins readable", "|max-min| >0.10")
        else:
            spread = max(vals) - min(vals)
            shape = "  ".join(f"{b['lag']}={b['accuracy']:.4f}(n={b['n_participants']})"
                              for b in bins)
            v = SUPPORTS if spread > 0.10 else (REJECTS if spread < 0.05 else INCONCL)
            add("P2 lag effect", v,
                f"|max-min| {spread:.4f} over {len(vals)} bins (was {REF['wr_lag_spread']}); {shape}",
                "supports >0.10; rejects <0.05")
    except Exception as e:                                   # analysis must not hard-fail
        add("P2 lag effect", INCONCL, f"error: {type(e).__name__}: {e}", "|max-min| >0.10")

    # ---------------- P3: A2 miss/false-alarm ratio crosses 1.0 -------------
    try:
        miss, fa, trials = ES.a2_model(str(runs[0]))
        m, f = float(np.nanmean(miss)), float(np.nanmean(fa))
        ratio = m / f if f else float("nan")
        if ratio != ratio:
            add("P3 A2 ratio", INCONCL, "ratio undefined (no false alarms)", ">1.0")
        else:
            v = SUPPORTS if ratio > 1.0 else (REJECTS if ratio < 0.6 else INCONCL)
            add("P3 A2 ratio", v,
                f"miss {m:.4f}, fa {f:.4f}, ratio {ratio:.4f} (was {REF['a2_ratio']}, human 6.094), "
                f"trials attempted {float(np.mean(trials)):.1f}",
                "supports >1.0; rejects <0.6")
    except Exception as e:
        add("P3 A2 ratio", INCONCL, f"error: {type(e).__name__}: {e}", ">1.0")

    # ---------------- P4: variable_mapping at 20 questions ------------------
    vm_surv, n_rows, at_16 = [], 0, 0
    for r in runs:
        for row in _rows(r, "variable_mapping"):
            fe = (row.get("metrics") or {}).get("first_error_at")
            nq = (row.get("metrics") or {}).get("n_questions") or len(row.get("questions") or []) or 20
            s = int(fe) if fe else int(nq)
            vm_surv.append(s)
            n_rows += 1
            if s >= 16:
                at_16 += 1
    hl_vm = [S.humanlikeness(S.human_scores("variable_mapping"),
                             S.llm_scores("variable_mapping", r, "compactor")) for r in runs]
    hl_vm = [h for h in hl_vm if h == h]
    if not vm_surv or not hl_vm:
        add("P4 variable_mapping", INCONCL, "absent", "median 4-5, <=3 rows >=16, |dHL|<=0.02")
    else:
        med, m_hl = st.median(vm_surv), st.mean(hl_vm)
        per_run = at_16 / max(len(runs), 1)
        obs = (f"survival median {med} (mean {st.mean(vm_surv):.2f}), "
               f"{at_16}/{n_rows} rows >=16 ({per_run:.1f} per run), HL {m_hl:.4f}")
        ok = (4 <= med <= 5 and per_run <= 3
              and abs(m_hl - REF["vm_humanlikeness"]) <= 0.02)
        bad = per_run > 3 or abs(m_hl - REF["vm_humanlikeness"]) > 0.05
        add("P4 variable_mapping", SUPPORTS if ok else (REJECTS if bad else INCONCL), obs,
            "supports median 4-5 & <=3 rows>=16 per run & |dHL|<=0.02; "
            "rejects >3 rows>=16 per run or |dHL|>0.05")

    # ---------------- P5: digit-span matched n ------------------------------
    try:
        n_matched = len(PM.model_participants(runs[0] / "tasks/wm_digit_span_forward.jsonl"))
        rev = PM.model_span_trials(runs[0] / "tasks/wm_digit_span_reverse.jsonl")
        rev_err = sum(1 for p in rev for t in p if not t.get("correct"))
        obs = f"matched pseudo-participants {n_matched} (was 5); reverse model errors {rev_err} (was 20, need >=30)"
        v = SUPPORTS if n_matched == 20 else REJECTS
        add("P5 digit span matched n", v, obs, "supports n_model == 20")
    except Exception as e:
        add("P5 digit span matched n", INCONCL, f"error: {type(e).__name__}: {e}", "n_model == 20")

    # ---------------- P6: precondition on the four untouched tasks ---------
    for task, band in P6_BAND.items():
        vals = [S.humanlikeness(S.human_scores(task), S.llm_scores(task, r, "compactor"))
                for r in runs]
        vals = [v for v in vals if v == v]
        if not vals:
            add(f"P6 {task}", INCONCL, "absent", f"within +/-{band}")
            continue
        d = st.mean(vals) - REF[task]
        add(f"P6 {task}", SUPPORTS if abs(d) <= band else REJECTS,
            f"{st.mean(vals):.4f} vs {REF[task]} -> {d:+.4f}", f"within +/-{band}")

    # ---------------- report ------------------------------------------------
    width = max(len(p) for p, _, _, _ in results)
    print("PREDICTION CHECK, job 18781213 -- bands registered in "
          "logs/predictions_iter12stage2.md before the run\n")
    print("runs:", ", ".join(str(r) for r in runs), "\n")
    for pid, v, obs, band in results:
        print(f"  {pid:<{width}}  {v:<8}  {obs}")
        print(f"  {'':<{width}}            band: {band}")
    tally = collections.Counter(v for _, v, _, _ in results)
    print("\ntally:", dict(tally))
    p6 = [v for p, v, _, _ in results if p.startswith("P6")]
    if REJECTS in p6:
        print("\n*** P6 FAILED -- the precondition is broken and P1-P5 are VOID until the "
              "leak is traced. This is how the stage-1 precondition was treated. ***")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main([Path(a) for a in sys.argv[1:]]))
