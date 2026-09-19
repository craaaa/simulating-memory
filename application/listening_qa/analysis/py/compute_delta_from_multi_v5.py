#!/usr/bin/env python3
"""Compute the equivalence bound Δ (D4) from the EARLIER multi_v5 pilot — never multi_v6.

Δ = 0.5 × smallest human `level` effect (log-odds), where the level effect is estimated
from a human-only binomial GLM of per-option correctness on `level` (ref=control), pooled
over topics/options. Uses multi_v5 (structurally identical to v6; all 4 banks verified
IDENTICAL). Whole-participant attention exclusion, matching the v6 analysis set rule.

This script emits the option-level pilot table; the GLM + Δ is fit in R
(compute_delta.R) which needs no package install. Writes:
  data/interim/multi_v5_human_options.csv

Run:  python analysis/py/compute_delta_from_multi_v5.py  (then Rscript R/compute_delta.R)
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
RAWV5 = ANALYSIS / "data" / "raw" / "multi_v5"
INTERIM = ANALYSIS / "data" / "interim"

spec = importlib.util.spec_from_file_location("ingest01", HERE / "01_ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

import yaml  # noqa: E402

TOPICS, LEVELS = ingest.TOPICS, ingest.LEVELS
HUMAN_TSV_TOKEN, FL_TOPIC = ingest.HUMAN_TSV_TOKEN, ingest.FL_TOPIC
parse_response = ingest.parse_response


def load_v5_bank():
    bank = {}
    for topic in TOPICS:
        qdata = yaml.safe_load((RAWV5 / "question_banks" / topic / "questions.yaml").open())
        qs = []
        for q in qdata["questions"]:
            if q["metadata"].get("type") != "content":
                continue
            ans = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
            qs.append({"q_id": q["q_id"],
                       "options": {int(k): str(v) for k, v in q["options"].items()},
                       "answer": set(int(a) for a in ans)})
        bank[topic] = qs
    return bank


def main():
    bank = load_v5_bank()
    # dedupe across both export files by PID (longest duration)
    best = {}
    for tsv in sorted(RAWV5.glob("multi_v5_*.tsv")):
        with tsv.open(encoding="utf-16") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))[2:]
        for r in rows:
            pid = (r.get("PROLIFIC_PID") or "").strip()
            if not pid:
                continue
            dur = float(r.get("Duration (in seconds)", 0) or 0)
            if pid not in best or dur > float(best[pid].get("Duration (in seconds)", 0) or 0):
                best[pid] = r

    out = []
    n_ok = 0
    for pid, r in best.items():
        if (r.get("Finished") or "").strip() != "True":
            continue
        if not (r.get("martial_arts_cond") or "").strip():
            continue
        # whole-participant attention pass
        ok = True
        for topic in TOPICS:
            tok = HUMAN_TSV_TOKEN[topic]
            g = (r.get(f"{tok}_AT_{tok}") or "").strip()
            c = (r.get(f"{tok}_AT_correct") or "").strip()
            if (frozenset(g.split(",")) if g else frozenset()) != (frozenset(c.split(",")) if c else frozenset()):
                ok = False
                break
        if not ok:
            continue
        n_ok += 1
        for topic in TOPICS:
            tok = HUMAN_TSV_TOKEN[topic]
            level = (r.get(f"{tok}_cond") or "").strip()
            if level not in LEVELS:
                continue
            for q in bank[topic]:
                texts = list(q["options"].values())
                given = parse_response((r.get(f"{tok}_{q['q_id']}") or "").strip(), texts)
                for opt_n, opt_text in q["options"].items():
                    endorsed = opt_text in given
                    is_true = opt_n in q["answer"]
                    out.append({"topic": topic, "level": level, "q_id": q["q_id"],
                                "option_n": opt_n, "endorsed": int(endorsed),
                                "option_is_true": int(is_true),
                                "correct": int(endorsed == is_true)})

    INTERIM.mkdir(parents=True, exist_ok=True)
    outp = INTERIM / "multi_v5_human_options.csv"
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topic", "level", "q_id", "option_n",
                                          "endorsed", "option_is_true", "correct"])
        w.writeheader()
        w.writerows(out)
    print(f"multi_v5 whole-participant-pass humans: {n_ok}; option rows: {len(out)}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
