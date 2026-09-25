"""Check a candidate's PRE-REGISTERED predictions against its run. Mechanically.

Why this exists. A candidate's manifest states what it expects to happen, including
what it expects to lose. After the run, it is very easy to read the numbers and
write a story in which the candidate did roughly what it meant to — the predictions
were qualitative, the evidence is a 40-field record, and nobody is checking. That is
how a search stops being able to tell a mechanism from a lucky number.

So the predictions are encoded here as executable assertions, written from the
candidate's docstring, and evaluated as PASS / FAIL / INCONCLUSIVE with the observed
value printed next to the threshold. A FAIL is not an argument for rejecting the
candidate — the per-task floor and the guards do that — it is a record that the
stated mechanism did not do what it claimed, which is the thing worth knowing for the
next iteration.

`displacement` (iteration 1) pre-registered, in its own words:

  primary      n=3 `answered` rises from 6.82 to >= 12 of 14, while
               `acc_over_answered` stays within +-0.05 of 0.737
  disconfirm   `answered` does not rise  -> the refusal diagnosis is dead
               `answered` rises but acc_over_answered < 0.65 -> the silence was
               incapacity and the store was never the issue
  capacity     n=3 slot utilisation stays ~1.0; displacement is not extra capacity
  compliance   n=3 buffer `No response` rises from 50/150 past 100/150
  leak test    word_recognition moves by LESS than its 0.121 noise floor; if it
               moves a lot under a change that only touches overflow semantics, the
               third-leak diagnosis is wrong
  exposed      semantic_story_recall regresses -- expected, names primacy
               protection as the iteration-2 ingredient

Usage:
    python meta_harness/check_predictions.py <candidate_id> <run_dir> \
        [--baseline <run_dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from meta_harness import nback_levels as NL  # noqa: E402
from meta_harness import score_candidate as SC  # noqa: E402

PASS, FAIL, INCONCL = "PASS", "FAIL", "INCONCLUSIVE"


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def displacement_checks(run_dir: Path, baseline: Path | None) -> list[dict[str, Any]]:
    rep = NL.report(run_dir)
    d3 = (rep.get("diagnostics") or {}).get(3) or {}
    lv3 = (rep.get("per_level") or {}).get(3) or {}

    answered = d3.get("answered")
    acc_ans = d3.get("acc_over_answered")
    keys = d3.get("keys_held")
    buf = d3.get("buffer_no_response_frac")

    out: list[dict[str, Any]] = []

    def add(name: str, verdict: str, observed: Any, threshold: str,
            note: str = "") -> None:
        out.append({"prediction": name, "verdict": verdict,
                    "observed": observed, "threshold": threshold, "note": note})

    # --- primary -------------------------------------------------------------
    if answered is None:
        add("n=3 answered >= 12/14", INCONCL, None, ">= 12",
            "no n=3 rows found")
    elif answered >= 12.0:
        add("n=3 answered >= 12/14", PASS, answered, ">= 12 (baseline 6.82)")
    elif answered > 6.82 + 1.0:
        add("n=3 answered >= 12/14", FAIL, answered, ">= 12 (baseline 6.82)",
            "rose but fell short of the stated threshold -- mechanism partially "
            "works; do not round this up to a confirmation")
    else:
        add("n=3 answered >= 12/14", FAIL, answered, ">= 12 (baseline 6.82)",
            "DISCONFIRMS the refusal diagnosis outright: the store jam was not "
            "what was suppressing responses")

    if acc_ans is None:
        add("n=3 acc_over_answered within +-0.05 of 0.737", INCONCL, None,
            "0.687 - 0.787")
    elif 0.687 <= acc_ans <= 0.787:
        add("n=3 acc_over_answered within +-0.05 of 0.737", PASS, acc_ans,
            "0.687 - 0.787")
    elif acc_ans < 0.65:
        add("n=3 acc_over_answered within +-0.05 of 0.737", FAIL, acc_ans,
            "0.687 - 0.787",
            "below the 0.65 disconfirmation line: extra responses are guesses, so "
            "any score gain here is production without judgement")
    else:
        add("n=3 acc_over_answered within +-0.05 of 0.737", FAIL, acc_ans,
            "0.687 - 0.787",
            "outside the band but above 0.65 -- judgement changed too, so the "
            "change is not purely to response production as claimed")

    # --- capacity: displacement must not smuggle in extra room ---------------
    if keys is None:
        add("n=3 slot utilisation ~1.0 (4 keys held)", INCONCL, None, ">= 3.5")
    elif keys >= 3.5:
        add("n=3 slot utilisation ~1.0 (4 keys held)", PASS, keys,
            ">= 3.5 of 4 (baseline 3.96)")
    else:
        add("n=3 slot utilisation ~1.0 (4 keys held)", FAIL, keys,
            ">= 3.5 of 4 (baseline 3.96)",
            "the store is no longer saturating, so something other than the "
            "overflow rule changed -- compare against full_context's 1.06")

    # --- compliance: the check that cannot be bought by answering eagerly ----
    # Compare against the baseline's MEASURED value rather than a literal, so a run
    # scored against itself is not described as having "improved" on itself.
    base_buf = None
    if baseline is not None:
        base_buf = ((NL.report(baseline).get("diagnostics") or {}).get(3) or {}
                    ).get("buffer_no_response_frac")
    ref = "baseline 0.333" if base_buf is None else f"baseline {base_buf:.3f}"

    if buf is None:
        add("n=3 buffer 'No response' > 100/150", INCONCL, None, "> 0.667")
    elif buf > 0.667:
        add("n=3 buffer 'No response' > 100/150", PASS, buf, f"> 0.667 ({ref})")
    elif base_buf is not None and buf > base_buf + 1e-9:
        add("n=3 buffer 'No response' > 100/150", FAIL, buf, f"> 0.667 ({ref})",
            "improved but under the stated bar")
    else:
        add("n=3 buffer 'No response' > 100/150", FAIL, buf, f"> 0.667 ({ref})",
            "phase tracking did not improve; a rise in `answered` without this is "
            "consistent with answering more eagerly rather than knowing where it is")

    # --- the leak test, and the expected loss -------------------------------
    if baseline is not None:
        cand = SC.humanlikeness_by_task(run_dir, SC.SEARCH_TASKS)
        base = SC.humanlikeness_by_task(baseline, SC.SEARCH_TASKS)

        wr_c, wr_b = cand.get("word_recognition"), base.get("word_recognition")
        if wr_c is None or wr_b is None:
            add("word_recognition moves < 0.121 (leak test)", INCONCL, None,
                "< 0.121")
        else:
            delta = abs(wr_c - wr_b)
            add("word_recognition moves < 0.121 (leak test)",
                PASS if delta < 0.121 else FAIL, round(delta, 4), "< 0.121",
                "" if delta < 0.121 else
                "word recognition moved a lot under a change that only touches "
                "overflow semantics -- the third-leak reading needs revisiting")

        sr_c, sr_b = cand.get("semantic_story_recall"), base.get("semantic_story_recall")
        if sr_c is None or sr_b is None:
            add("semantic_story_recall regresses (expected loss)", INCONCL, None,
                "delta < 0")
        else:
            delta = round(sr_c - sr_b, 4)
            # Either direction is informative here, so this is reported rather than
            # scored: the candidate predicted a loss and said what a loss would mean.
            add("semantic_story_recall regresses (expected loss)",
                PASS if delta < 0 else INCONCL, delta, "delta < 0",
                "as predicted; primacy protection is the named iteration-2 "
                "ingredient" if delta < 0 else
                "did NOT regress -- the primacy-vs-recency story is wrong or the "
                "refusal was not shaping which chunks were kept")

    return out


CHECKS: dict[str, Callable[[Path, Path | None], list[dict[str, Any]]]] = {
    "displacement": displacement_checks,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    fn = CHECKS.get(args.candidate)
    if fn is None:
        print(f"no pre-registered predictions encoded for '{args.candidate}'. "
              f"Known: {sorted(CHECKS)}")
        return 2

    rows = fn(Path(args.run_dir), Path(args.baseline) if args.baseline else None)

    width = max(len(r["prediction"]) for r in rows)
    for r in rows:
        print(f"{r['verdict']:13s} {r['prediction']:{width}s}  "
              f"observed {_fmt(r['observed'])}  (needs {r['threshold']})")
        if r["note"]:
            print(f"{'':13s} -> {r['note']}")

    n_pass = sum(r["verdict"] == PASS for r in rows)
    n_fail = sum(r["verdict"] == FAIL for r in rows)
    print(f"\n{n_pass} passed, {n_fail} failed, "
          f"{sum(r['verdict'] == INCONCL for r in rows)} inconclusive")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
