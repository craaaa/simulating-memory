#!/usr/bin/env python3
"""reconcile — decisive correctness check for the option-level long table.

The §2 structural gates prove SHAPE, not that the join keys are RIGHT. The blind
tagging pass flagged that the question banks were edited after some runs, so the
current bank's option text/order/answers might disagree with what the model runs
(`parsed_answers`, `metrics`) and the human export (`longdata_strict.csv`) actually
recorded. If so, `endorsed` / `option_is_true` would be silently wrong while every
structural gate stays green.

This reconciles the derived option-level table against BOTH independent run-time
ground truths and hard-fails on any mismatch:

  MODELS — for each source JSONL row, the set of endorsed option numbers we derived
           must equal `parsed_answers[q]`, and per-question accuracy recomputed from
           `endorsed` + `option_is_true` must equal the row's stored `metrics`.
  HUMANS — per (pid, topic, question_id, condition), `correct` (exact match) and
           `partial` recomputed from our table must equal longdata_strict.csv.

Run:  python analysis/tests/reconcile.py   (exit 0 = reconciled, 1 = mismatch)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # analysis/tests
ANALYSIS = HERE.parent
REPO = ANALYSIS.parents[2]
PROC = ANALYSIS / "data" / "processed"

# import the ingest module's bank loader (filename starts with a digit → importlib)
spec = importlib.util.spec_from_file_location("ingest01", ANALYSIS / "py" / "01_ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

import yaml  # noqa: E402

BANK = ingest.load_question_bank()          # topic -> [{q_id, options, answer:set}]
TOPICS = ingest.TOPICS
LEVELS = ingest.LEVELS
fails: list[str] = []


def load_table() -> list[dict]:
    def b(x):
        return str(x).strip().lower() in ("true", "1", "yes", "t")
    rows = []
    with (PROC / "responses.csv").open() as f:
        for r in csv.DictReader(f):
            r["option_is_true"] = b(r["option_is_true"])
            r["endorsed"] = b(r["endorsed"])
            r["option_n"] = int(r["option_n"])
            rows.append(r)
    return rows


# ── MODELS ──
def reconcile_models(tbl):
    # derived: respondent_id -> question_id -> {opt_n: (endorsed, is_true)}
    der: dict = {}
    for r in tbl:
        if r["agent"] != "model":
            continue
        der.setdefault(r["respondent_id"], {}).setdefault(r["question_id"], {})[r["option_n"]] = (
            r["endorsed"], r["option_is_true"])

    cfg = yaml.safe_load((ANALYSIS / "config" / "models.yaml").open())
    prompt_cond = cfg["prompting_condition"]
    compactor_cond = cfg["compactor_condition"]
    checked = 0
    for m in cfg["models"]:
        name = m["name"]
        for side, path, cond in (
            ("prompting", REPO / m["prompting_path"], prompt_cond),
            ("wm", REPO / m["compactor_path"], compactor_cond),
        ):
            model_name = name if side == "prompting" else f"{name}__wm"
            if not path.exists():
                continue
            for line in path.open():
                row = json.loads(line)
                if row.get("condition_id") != cond:
                    continue
                topic = str(row.get("topic_id") or "").strip()
                level = row.get("level")
                if topic not in TOPICS or level not in LEVELS:
                    continue
                rid = f"{model_name}:{cond}:{topic}:{level}:r{row.get('repeat_index')}"
                parsed = row.get("parsed_answers") or {}
                bank = BANK[topic]
                ps_correct = ps_total = 0
                exact_correct = 0
                for qi, q in enumerate(bank, start=1):
                    q_id = q["q_id"]
                    stored = set(int(x) for x in (parsed.get(str(qi)) or []))
                    dopts = der.get(rid, {}).get(q_id, {})
                    derived_sel = {n for n, (e, _t) in dopts.items() if e}
                    if derived_sel != stored:
                        fails.append(f"MODEL {rid} {q_id}: endorsed {sorted(derived_sel)} != parsed_answers {sorted(stored)}")
                    # per-statement + exact-match recomputation
                    for n, (e, t) in dopts.items():
                        ps_total += 1
                        if e == t:
                            ps_correct += 1
                    truth = {n for n, (_e, t) in dopts.items() if t}
                    if derived_sel == truth:
                        exact_correct += 1
                    checked += 1
                mt = row.get("metrics") or {}
                if "per_statement_correct" in mt and ps_correct != mt["per_statement_correct"]:
                    fails.append(f"MODEL {rid}: per_statement_correct {ps_correct} != stored {mt['per_statement_correct']}")
                if "exact_match_correct" in mt and exact_correct != mt["exact_match_correct"]:
                    fails.append(f"MODEL {rid}: exact_match_correct {exact_correct} != stored {mt['exact_match_correct']}")
    return checked


# ── HUMANS ──
LONGDATA = REPO / "application" / "listening_qa" / "analysis" / "data" / "raw" / "human_longdata_strict_multi_v6.csv"
LONG_TOPIC = {"martial_arts": "martial_arts", "fruits_v2": "fruits",
              "astronomy": "astronomy", "fabrics": "fabrics"}


def reconcile_humans(tbl):
    # derived per (pid, topic, q_id): correct (exact), partial
    der: dict = {}
    for r in tbl:
        if r["agent"] != "human":
            continue
        der.setdefault((r["respondent_id"], r["topic"], r["question_id"]), []).append(
            (r["option_n"], r["endorsed"], r["option_is_true"]))

    def derive(cells):
        given = {n for n, e, _t in cells if e}
        truth = {n for n, _e, t in cells if t}
        exact = 1 if given == truth else 0
        n_correct_opts = len(truth)
        n_given_correct = len(given & truth)
        n_given_wrong = len(given - truth)
        partial = max(0.0, (n_given_correct - n_given_wrong) / n_correct_opts) if n_correct_opts else 0.0
        return exact, partial

    if not LONGDATA.exists():
        fails.append(f"HUMAN: longdata not found at {LONGDATA}")
        return 0
    checked = matched = 0
    with LONGDATA.open() as f:
        for r in csv.DictReader(f):
            topic = LONG_TOPIC.get(r["topic"])
            q_id = r["question_id"]
            pid = r["pid"]
            level = r["condition"]
            if topic is None or level not in LEVELS:
                continue
            checked += 1
            cells = der.get((pid, topic, q_id))
            if cells is None:
                # pid may be excluded on our side only if attention filtering differs; longdata_strict
                # already applied whole-participant AT exclusion, so absence is a real mismatch.
                fails.append(f"HUMAN {pid} {topic} {q_id}: present in longdata, absent in table")
                continue
            matched += 1
            exact, partial = derive(cells)
            if exact != int(r["correct"]):
                fails.append(f"HUMAN {pid} {topic} {q_id}: correct {exact} != longdata {r['correct']}")
            if abs(partial - float(r["partial"])) > 1e-6:
                fails.append(f"HUMAN {pid} {topic} {q_id}: partial {partial:.4f} != longdata {r['partial']}")
    return matched


def main():
    tbl = load_table()
    nm = reconcile_models(tbl)
    nh = reconcile_humans(tbl)
    print(f"Reconciled {nm} model (respondent×question) records and {nh} human (pid×topic×question) records.")
    if fails:
        print(f"\nRECONCILIATION FAILED — {len(fails)} mismatch(es); first 25:")
        for f in fails[:25]:
            print("  -", f)
        sys.exit(1)
    print("RECONCILIATION PASSED — derived endorsed/option_is_true match both run-time ground truths.")


if __name__ == "__main__":
    main()
