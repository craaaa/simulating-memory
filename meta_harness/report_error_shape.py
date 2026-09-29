"""Error-shape report: six measures, all tasks, REPORT ONLY.

What this is for. `mean_humanlikeness_search` is 1 - W_1 between the model's and
the humans' per-participant score distributions, averaged over 8 tasks. It is a
SCALAR PER TASK: it counts how many items were right and can say nothing about
which ones or how. Two runs with identical score distributions can fail in
completely different ways, and the four existing axes only look at three tasks
(A1 digit span, A2 word recognition, A3 story recall) plus A4 on variable mapping.
This module adds an error-shape measure for every task so the evaluation does not
hinge on one of them.

NOTHING HERE GATES. No floors, no guards, no change to `mean_humanlikeness_search`,
and deliberately no entry in `score_candidate.axes()` -- anything added there lands
in the record the proposer reads. Promotion is a later decision; this CLI is the
only delivery surface.

Every measure has its OWN unit and none of them is humanlikeness. The table prints
the human reference, the model value and the distance as separate columns, plus the
trial or participant count behind each. Where per-participant values exist at
matched granularity on both sides the measure also reports `1 - W_1` over them, in
that measure's own unit -- commensurable with the score metric in form, not the same
quantity, and never averaged into it.

The human references are cached in `logs/human_error_shape.json` with the exact
filters applied, so a reference can never be recomputed inline and silently
differently -- the failure mode that made A3 and A4 wrong the first time.
`test_error_shape.py` asserts the cache still matches a fresh recomputation.

Usage:
    python meta_harness/report_error_shape.py --rebuild-human-cache
    python meta_harness/report_error_shape.py <run_dir> [<run_dir> ...]
    python meta_harness/report_error_shape.py <run_dir> --json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_harness import error_structure as ES      # noqa: E402
from meta_harness import interference as IF         # noqa: E402
from meta_harness import protocol_match as PM       # noqa: E402

CACHE = ROOT / "meta_harness/logs/human_error_shape.json"

SEARCH_TASKS = ("variable_mapping", "digit_span_forward", "digit_span_reverse",
                "nback", "word_recognition", "narrative_qa", "craft_task",
                "semantic_story_recall")
HELDOUT_TASKS = ("map_task", "factual_qa")


# ------------------------------------------------------------ human references
def _f(predicate: str, n_after: int | None, unit: str) -> dict:
    """One filter step, as structured data rather than prose."""
    return {"predicate": predicate, "unit_of_n": unit, "n_after": n_after}


def human_reference() -> dict[str, Any]:
    """Recompute every human reference from the released records.

    This is the ONLY place a human reference is computed. `build_cache` writes its
    output to disk and the report reads the file.
    """
    out: dict[str, Any] = {"schema": 1, "measures": {}}
    M = out["measures"]

    # ---- M1 variable_mapping intrusion type
    vm = IF.human_trials()
    prof = IF.intrusion_profile(vm)
    M["M1_variable_mapping_intrusion"] = {
        "task": "variable_mapping",
        "pooled_only": True,
        "pooled_only_because": (
            "every human session terminates at its FIRST wrong answer -- all 152 "
            "participants with an error have exactly one, and it is always their "
            "last question -- so a per-participant share is one observation and "
            "1 - W_1 over them would measure rounding"),
        "unit": prof["unit"],
        "values": prof["shares"],
        "values4": prof["shares4"],
        "counts": prof["counts"],
        "n_errors": prof["n_errors"],
        "n_trials": prof["n_trials"],
        "min_errors": prof["min_errors"],
        "filters": [
            _f("runs/human/working-memory-variable-mapping/run-*.json", 154, "records"),
            _f("payload.questions[] with selectedCity is not None", prof["n_trials"],
               "question trials"),
            _f("errors only (correct is false)", prof["n_errors"], "error trials"),
            _f("assignment window: assignments with turn < question turn", None,
               "n/a -- classification input, not a filter"),
        ],
    }

    # ---- M2 digit span typology + serial position, both tasks
    for task, reverse in (("digit_span_forward", False), ("digit_span_reverse", True)):
        parts = PM.human_span_trials(reverse=reverse)
        typ = PM.span_typology(parts, reverse=reverse)
        sp = PM.serial_position(parts)
        M[f"M2_{task}_typology"] = {
            "task": task,
            "unit": typ["unit"],
            "values": typ["shares"],
            "counts": typ["counts"],
            "reversal_share_nonpalindromic": typ["reversal_share_nonpalindromic"],
            "n_errors": typ["n_errors"],
            "n_errors_nonpalindromic": typ["n_errors_nonpalindromic"],
            "n_trials": typ["n_trials"],
            "n_participants": typ["n_participants"],
            "min_errors": typ["min_errors"],
            "filters": [
                _f(("runs/human/working-memory-"
                    + ("reverse-digit-span" if reverse else "digit-span")
                    + "/run-*.json"), None, "records"),
                _f("payload.trials[] with length is not None", None, "trials"),
                _f("administered under the real staircase: 2 trials/span ascending, "
                   "stop on the first span where both fail", typ["n_trials"],
                   "administered trials"),
                _f("participants whose staircase has a defined ceiling (best_span > 0)",
                   typ["n_participants"], "participants"),
                _f("errors only", typ["n_errors"], "administered errors"),
                _f("reversal class only: non-palindromic presented sequences",
                   typ["n_errors_nonpalindromic"], "administered errors"),
            ],
        }
        M[f"M2_{task}_serial_position"] = {
            "task": task,
            "unit": sp["unit"],
            "values": {"primacy_third": sp["bin_means"][0],
                       "middle_third": sp["bin_means"][1],
                       "recency_third": sp["bin_means"][2]},
            "per_participant": sp["per_participant"],
            "absolute_curve": sp["absolute_curve"],
            "absolute_n": sp["absolute_n"],
            "n_participants": sp["n_participants"],
            "filters": [
                _f("same administered trials as the typology above", None,
                   "administered trials"),
                _f("position i of a length-L target -> bin min(2, 3*i//L)", None,
                   "n/a -- binning rule"),
            ],
        }

    # ---- M3 n-back
    nb = ES.nback_human_trials()
    prof = ES.nback_error_profile(nb)
    M["M3_nback_error_structure"] = {
        "task": "nback",
        "unit": prof["unit"],
        "values": {k: prof["overall"][k] for k in
                   ("miss_rate", "fa_rate", "ratio", "unanswered_rate",
                    "lure_fa_rate", "nonlure_fa_rate")},
        "ratio_ci_over_cells": prof.get("ratio_ci_over_cells"),
        "by_level": {k: {kk: v[kk] for kk in
                         ("miss_rate", "fa_rate", "ratio", "lure_fa_rate",
                          "nonlure_fa_rate", "n_lure_trials", "lure_note")}
                     for k, v in prof["by_level"].items()},
        "n_trials": prof["overall"]["n_trials"],
        "n_target": prof["overall"]["n_target"],
        "n_nontarget": prof["overall"]["n_nontarget"],
        "n_lure_trials": prof["overall"]["n_lure_trials"],
        "n_cells": prof["n_cells"],
        "per_cell_miss": [v["miss"] for v in prof["per_cell"].values()],
        "per_cell_fa": [v["fa"] for v in prof["per_cell"].values()],
        "per_cell_thirds": [v["thirds"] for v in prof["per_cell"].values()],
        "min_lure_trials": ES.MIN_LURE_TRIALS,
        "filters": [
            _f("runs/human/working-memory-nback/run-*.json", 57, "records"),
            _f("drop phase == 'practice' and blocks named training-*", None, "trials"),
            _f("level read from the block name ('2-back'), because payload level is "
               "null in 4 of the 57 records", None, "n/a -- field source"),
            _f("drop the first n trials of each block (lead-in: no letter n back, "
               "and 112 of those 318 trials are nonetheless logged target=true)",
               prof["overall"]["n_trials"], "defined trials"),
            _f("target recomputed as letter[i] == letter[i-n]; agrees with the "
               "logged flag on 1908/1908 of these trials", None, "n/a -- check"),
            _f("granularity: one cell per (participant, level), NOT pooled over a "
               "participant's three levels", prof["n_cells"], "cells"),
            _f("lure = non-target whose letter matches at lag n-1 or n+1",
               prof["overall"]["n_lure_trials"], "lure trials"),
        ],
    }

    # ---- M4 word-recognition lag curve
    sessions = ES.wr_lag_human()
    lag = ES.wr_lag_summary(sessions)
    M["M4_word_recognition_lag"] = {
        "task": "word_recognition",
        "task_is_broken": True,
        "caveat": lag["caveat"],
        "unit": lag["unit"],
        "values": {b["lag"]: b["accuracy"] for b in lag["bins"]},
        "n_participants_per_bin": {b["lag"]: b["n_participants"] for b in lag["bins"]},
        "per_participant": {b["lag"]: b["per_participant"] for b in lag["bins"]},
        "n_sessions": lag["n_sessions"],
        "min_participants": ES.MIN_LAG_PARTICIPANTS,
        "filters": [
            _f("runs/human/working-memory-word-recognition/run-*.json", 53, "records"),
            _f("records with a non-empty payload.responses", lag["n_sessions"],
               "sessions"),
            _f("old trials only (the word appeared earlier in this list)", None,
               "old trials"),
            _f("lag = trials elapsed since that word's FIRST appearance", None,
               "n/a -- binning rule"),
            _f("every bin computed PER PARTICIPANT then averaged, because sessions "
               "terminate at 3 strikes and run 4-102 trials (median 32)", None,
               "n/a -- aggregation rule"),
        ],
    }

    # ---- M5 distractor choice (4-option tasks) and error-index profile (binary)
    for task in ("narrative_qa", "factual_qa"):
        counts = ES.distractor_counts_human(task)
        prof = ES.distractor_profile(counts)
        bank_note = (
            "471 of the 520 human questions come from data/narrative_QA.json and 49 "
            "from data/narrative_QA_easy.json; the model only ever sees the former, "
            "so the 49 easy-bank questions are dropped"
            if task == "narrative_qa" else
            "all 530 human questions match data/wikipedia_10docs_questions.json")
        M[f"M5_{task}_distractor"] = {
            "task": task,
            "unit": prof["unit"],
            "values": {"within_side_agreement": prof["within_side_agreement"]},
            "chance": prof["chance"],
            "n_errors": prof["n_errors"],
            "n_questions_with_errors": prof["n_questions_with_errors"],
            "n_questions_with_2plus_errors": prof["n_questions_with_2plus_errors"],
            "n_ordered_pairs": prof["n_ordered_pairs"],
            "per_question_counts": {k: dict(v) for k, v in counts.items()},
            "pooled_only": True,
            "pooled_only_because": ("a human participant answers 10 questions and "
                                    "makes 1-3 errors, so a per-participant "
                                    "distractor distribution does not exist"),
            "filters": [
                _f(f"runs/human/{ES._HUMAN_MCQ_DIR[task]}/run-*.json", None, "records"),
                _f("payload.responses[] whose question text is in the shared bank; "
                   + bank_note, None, "question trials"),
                _f("errors only, with a parseable chosen letter different from the "
                   "bank answer", prof["n_errors"], "error trials"),
                _f("option letters verified unshuffled: the record's correctAnswer "
                   "agrees with the bank answer on every matched question", None,
                   "n/a -- check"),
            ],
        }
    for task in ("craft_task", "map_task"):
        sess = ES.error_index_human(task)
        prof = ES.error_index_summary(sess)
        M[f"M5b_{task}_error_index"] = {
            "task": task,
            "unit": prof["unit"],
            "not_a_distractor_measure": True,
            "why": prof["why"],
            "values": {str(b["index"]): b["error_rate"] for b in prof["by_index"]},
            "n_participants_per_index": {str(b["index"]): b["n_participants"]
                                         for b in prof["by_index"]},
            "per_participant": [b["per_participant"] for b in prof["by_index"]],
            "n_participants": prof["n_participants"],
            "filters": [
                _f(f"runs/human/{ES._HUMAN_INDEX_DIR[task]}/run-*.json", None,
                   "records"),
                _f("payload.trials[].responses[], all blocks pooled within a "
                   "participant", None, "question trials"),
                _f("error rate per question index, per participant then averaged",
                   prof["n_participants"], "participants"),
            ],
        }

    # ---- M6 story recall gist similarity
    recs = ES.gist_human(recompute=True)
    rec = ES.gist_summary(recs, "recomputed")
    sto = ES.gist_summary(recs, "stored")
    M["M6_story_recall_gist"] = {
        "task": "semantic_story_recall",
        "unit": rec["unit"],
        "values": {"gist_similarity_recomputed": rec["mean"],
                   "gist_similarity_stored_web_app": sto["mean"]},
        "sd_recomputed": rec["sd"],
        "sd_stored": sto["sd"],
        "n_recomputed": rec["n"],
        "n_stored": sto["n"],
        "by_story_recomputed": rec["by_story"],
        "per_participant": rec["per_participant"],
        "comparable_column": "recomputed",
        "comparable_column_because": (
            "the stored human value came from the web app and the model's from "
            "bench; they are the same embedder (all-MiniLM-L6-v2, 200-word "
            "truncation) but not the same preprocessing, so only the recomputed "
            "column is computed identically on the two sides"),
        "filters": [
            _f("runs/human/semantic-memory-story-recall/run-*.json", 56, "records"),
            _f("records with a non-empty payload.recallText and a resolvable "
               "payload.storyFile in data/", rec["n"], "records"),
            _f("story and recall each truncated to their first 200 words, then "
               "mean-pooled all-MiniLM-L6-v2 cosine", None, "n/a -- definition"),
        ],
    }
    return out


def build_cache(path: Path = CACHE) -> dict[str, Any]:
    ref = human_reference()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n")
    return ref


def load_cache(path: Path = CACHE) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing human reference cache {path}; run with "
                         f"--rebuild-human-cache")
    return json.loads(path.read_text())


# ------------------------------------------------------------- model side
def model_measures(run_dir: Path) -> dict[str, Any]:
    """Every measure on one run dir. Absent tasks are simply absent."""
    out: dict[str, Any] = {}
    if (run_dir / "tasks/wm_variable_mapping.jsonl").exists():
        out["M1_variable_mapping_intrusion"] = IF.intrusion_model(run_dir)
    for task, reverse, fn in (("digit_span_forward", False, "wm_digit_span_forward.jsonl"),
                              ("digit_span_reverse", True, "wm_digit_span_reverse.jsonl")):
        p = run_dir / "tasks" / fn
        if not p.exists():
            continue
        parts = PM.model_span_trials(p)
        out[f"M2_{task}_typology"] = PM.span_typology(parts, reverse=reverse)
        out[f"M2_{task}_typology"]["all_spans_not_comparable"] = PM.span_typology(
            PM.all_span_trials(p), reverse=reverse)
        out[f"M2_{task}_serial_position"] = PM.serial_position(parts)
    if (run_dir / "tasks/wm_nback.jsonl").exists():
        out["M3_nback_error_structure"] = ES.nback_error_profile(
            ES.nback_model_trials(run_dir))
    if (run_dir / "tasks/wm_word_recognition.jsonl").exists():
        out["M4_word_recognition_lag"] = ES.wr_lag_summary(ES.wr_lag_model(run_dir))
    for task in ("narrative_qa", "factual_qa"):
        if (run_dir / f"tasks/wm_{task}.jsonl").exists():
            counts = ES.distractor_counts_model(run_dir, task)
            out[f"M5_{task}_distractor"] = ES.distractor_profile(counts)
            out[f"M5_{task}_distractor"]["per_question_counts"] = {
                k: dict(v) for k, v in counts.items()}
    for task in ("craft_task", "map_task"):
        if (run_dir / f"tasks/wm_{task}.jsonl").exists():
            out[f"M5b_{task}_error_index"] = ES.error_index_summary(
                ES.error_index_model(run_dir, task))
    if (run_dir / "tasks/wm_semantic_story_recall.jsonl").exists():
        out["M6_story_recall_gist"] = ES.gist_summary(ES.gist_model(run_dir),
                                                      "recomputed")
    return out


# ---------------------------------------------------------------------- table
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.4f}"


Row = collections.namedtuple(
    "Row", "measure quantity unit human model distance n_human n_model w1")


def _rows_for(key: str, href: dict, mres: dict | None) -> list[Row]:
    """One row per scalar quantity in a measure, with its own unit and n."""
    rows: list[Row] = []
    unit = href.get("unit")
    unit_s = unit if isinstance(unit, str) else "(per quantity, see the doc)"

    def add(quantity, h, m, nh, nm, w1=None, unit_override=None):
        d = None if (h is None or m is None) else abs(float(h) - float(m))
        rows.append(Row(key, quantity, unit_override or unit_s, h, m, d, nh, nm, w1))

    if key == "M1_variable_mapping_intrusion":
        hv, mv = href.get("values") or {}, (mres or {}).get("shares") or {}
        for cls in IF.INTRUSION_CLASSES:
            add(cls, hv.get(cls), mv.get(cls) if mv else None,
                href.get("n_errors"), (mres or {}).get("n_errors"))
        return rows

    if key.endswith("_typology"):
        hv, mv = href.get("values") or {}, (mres or {}).get("shares") or {}
        for cls in PM.SPAN_ERROR_CLASSES:
            add(cls, hv.get(cls), mv.get(cls) if mv else None,
                href.get("n_errors"), (mres or {}).get("n_errors"))
        add("reversal|nonpalindromic", href.get("reversal_share_nonpalindromic"),
            (mres or {}).get("reversal_share_nonpalindromic"),
            href.get("n_errors_nonpalindromic"),
            (mres or {}).get("n_errors_nonpalindromic"))
        return rows

    if key.endswith("_serial_position"):
        hv = href.get("values") or {}
        mb = (mres or {}).get("bin_means")
        names = ("primacy_third", "middle_third", "recency_third")
        hp = href.get("per_participant") or []
        mp = (mres or {}).get("per_participant") or []
        for i, nm_ in enumerate(names):
            w1 = None
            if hp and mp:
                w1 = ES.one_minus_w1([r[i] for r in hp], [r[i] for r in mp])
            add(nm_, hv.get(nm_), mb[i] if mb else None,
                href.get("n_participants"), (mres or {}).get("n_participants"), w1)
        return rows

    if key == "M3_nback_error_structure":
        hv = href.get("values") or {}
        mv = ((mres or {}).get("overall") or {})
        for q in ("miss_rate", "fa_rate", "ratio", "unanswered_rate",
                  "lure_fa_rate", "nonlure_fa_rate"):
            # Each quantity's n is the denominator IT was computed over, not the
            # measure's trial total: miss over target trials, fa over non-target,
            # lure over lure trials, nonlure over the rest.
            if q == "miss_rate":
                nh, nm = href.get("n_target"), (mv.get("n_target") if mv else None)
            elif q in ("fa_rate",):
                nh, nm = href.get("n_nontarget"), (mv.get("n_nontarget") if mv else None)
            elif q == "lure_fa_rate":
                nh, nm = href.get("n_lure_trials"), (mv.get("n_lure_trials") if mv else None)
            elif q == "nonlure_fa_rate":
                nh = (href.get("n_nontarget") or 0) - (href.get("n_lure_trials") or 0)
                nm = (((mv.get("n_nontarget") or 0) - (mv.get("n_lure_trials") or 0))
                      if mv else None)
            else:
                nh, nm = href.get("n_trials"), (mv.get("n_trials") if mv else None)
            w1 = None
            if q in ("miss_rate", "fa_rate") and mres:
                col = "miss" if q == "miss_rate" else "fa"
                w1 = ES.one_minus_w1(href.get(f"per_cell_{col}") or [],
                                     [v[col] for v in (mres.get("per_cell") or {}).values()])
            add(q, hv.get(q), mv.get(q) if mv else None, nh, nm, w1)
        for i, nm_ in enumerate(("position_third_1", "position_third_2",
                                 "position_third_3")):
            hp = [r[i] for r in (href.get("per_cell_thirds") or [])]
            mp = ([v["thirds"][i] for v in (mres.get("per_cell") or {}).values()]
                  if mres else [])
            add(nm_, ES._mean_or_none(hp) if hp else None,
                ES._mean_or_none(mp) if mp else None,
                href.get("n_cells"), (mres or {}).get("n_cells"),
                ES.one_minus_w1(hp, mp) if (hp and mp) else None,
                "accuracy over answered trials in that third of the block")
        return rows

    if key == "M4_word_recognition_lag":
        hv = href.get("values") or {}
        hn = href.get("n_participants_per_bin") or {}
        hp = href.get("per_participant") or {}
        mbins = {b["lag"]: b for b in ((mres or {}).get("bins") or [])}
        for label in ES.LAG_BIN_LABELS:
            mb = mbins.get(label)
            w1 = (ES.one_minus_w1(hp.get(label) or [], mb["per_participant"])
                  if mb and hp.get(label) else None)
            add(f"lag {label}", hv.get(label), mb["accuracy"] if mb else None,
                hn.get(label), mb["n_participants"] if mb else None, w1)
        return rows

    if key.endswith("_distractor"):
        add("within_side_agreement",
            (href.get("values") or {}).get("within_side_agreement"),
            (mres or {}).get("within_side_agreement"),
            href.get("n_ordered_pairs"), (mres or {}).get("n_ordered_pairs"))
        if mres:
            task = href.get("task")
            hc = {k: collections.Counter(v)
                  for k, v in (href.get("per_question_counts") or {}).items()}
            mc = {k: collections.Counter(v)
                  for k, v in (mres.get("per_question_counts") or {}).items()}
            cross = ES.cross_side_distractor_agreement(hc, mc)
            # There is no "human value" for a cross-side statistic, so the human
            # column carries CHANCE (1/3) as the reference point and the distance
            # column is left empty rather than printing a difference of two
            # different things. n_human is shared questions, n_model is pairs.
            rows.append(Row(key, "cross_side_agreement [human col = chance]",
                            cross["unit"], cross["chance"],
                            cross["cross_side_agreement"], None,
                            cross["n_shared_questions"], cross["n_pairs"], None))
            del task
        return rows

    if key.endswith("_error_index"):
        hv = href.get("values") or {}
        hp = href.get("per_participant") or []
        mb = {str(b["index"]): b for b in ((mres or {}).get("by_index") or [])}
        for i, k in enumerate(sorted(hv, key=int)):
            m = mb.get(k)
            w1 = (ES.one_minus_w1(hp[i], m["per_participant"])
                  if (mres and i < len(hp) and m) else None)
            add(f"q{k}_error_rate", hv[k], m["error_rate"] if m else None,
                (href.get("n_participants_per_index") or {}).get(k),
                m["n_participants"] if m else None, w1)
        return rows

    if key == "M6_story_recall_gist":
        hv = href.get("values") or {}
        w1 = (ES.one_minus_w1(href.get("per_participant") or [],
                              (mres or {}).get("per_participant") or [])
              if mres else None)
        add("gist_similarity", hv.get("gist_similarity_recomputed"),
            (mres or {}).get("mean"), href.get("n_recomputed"),
            (mres or {}).get("n"), w1)
        add("gist_similarity (human stored, web app -- NOT comparable)",
            hv.get("gist_similarity_stored_web_app"), None,
            href.get("n_stored"), None)
        return rows
    return rows


def print_table(run_dir: Path, ref: dict) -> None:
    mres = model_measures(run_dir)
    measures = ref["measures"]
    print(f"\nERROR-SHAPE REPORT  (report only -- nothing here gates)")
    print(f"run        : {run_dir}")
    print(f"human ref  : {CACHE.relative_to(ROOT)}")
    print("distance is |human - model| IN THE MEASURE'S OWN UNIT -- it is a "
          "DIFFERENCE,\nnot a score, and it is not humanlikeness. "
          "`1-W1` is 1 - Wasserstein-1 over\nper-participant values in that same "
          "unit, reported only where both sides have\nthem at matched granularity.")

    for group, title in ((SEARCH_TASKS, "SEARCH-SET TASKS"),
                         (HELDOUT_TASKS, "HELD-OUT TASKS (reference only, "
                                         "never part of the search table)")):
        keys = [k for k, v in sorted(measures.items()) if v.get("task") in group]
        if not keys:
            continue
        print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")
        for key in keys:
            href = measures[key]
            rows = _rows_for(key, href, mres.get(key))
            flags = []
            if href.get("task_is_broken"):
                flags.append("TASK IS BROKEN -- see caveat")
            if href.get("pooled_only"):
                flags.append("pooled only")
            if href.get("not_a_distractor_measure"):
                flags.append("not a distractor measure")
            print(f"\n{key}   [task={href.get('task')}]"
                  + (f"   ({'; '.join(flags)})" if flags else ""))
            u = href.get("unit")
            if isinstance(u, str):
                print(f"    unit: {u}")
            if href.get("caveat"):
                print(f"    CAVEAT: {href['caveat']}")
            mm = mres.get(key)
            if mm is None:
                print("    model: task absent from this run dir")
            elif mm.get("note"):
                print(f"    model: {mm['note']}")
            if href.get("note"):
                print(f"    human: {href['note']}")
            print(f"    {'quantity':<44}{'human':>10}{'model':>10}{'distance':>10}"
                  f"{'n_human':>10}{'n_model':>10}{'1-W1':>9}")
            for r in rows:
                print(f"    {r.quantity:<44}{_fmt(r.human):>10}{_fmt(r.model):>10}"
                      f"{_fmt(r.distance):>10}{_fmt(r.n_human):>10}"
                      f"{_fmt(r.n_model):>10}{_fmt(r.w1):>9}")
            if key == "M3_nback_error_structure":
                hci = href.get("ratio_ci_over_cells")
                mci = (mm or {}).get("ratio_ci_over_cells")
                print("    miss/fa ratio 95% bootstrap CI over (participant, level) "
                      f"cells: human {hci}, model {mci}")
            if key.endswith("_typology") and mm and mm.get("all_spans_not_comparable"):
                a = mm["all_spans_not_comparable"]
                print("    model, all 19 spans, staircase NOT applied -- NOT "
                      f"human-comparable (n_errors={a['n_errors']}): {a['shares']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--rebuild-human-cache", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="dump the raw measure dicts instead of the table")
    args = ap.parse_args()

    if args.rebuild_human_cache:
        build_cache()
        print(f"wrote {CACHE}")
        if not args.run_dirs:
            return 0

    ref = load_cache()
    if not args.run_dirs:
        print(json.dumps(ref, indent=2))
        return 0
    for rd in args.run_dirs:
        if args.json:
            print(json.dumps({str(rd): model_measures(rd)}, indent=2, default=str))
        else:
            print_table(rd, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
