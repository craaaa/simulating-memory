#!/usr/bin/env python3
"""01_ingest — build the canonical option-level long table (ANALYSIS_PLAN.md §2).

One row per (respondent, text, question, option). Humans come from the raw
Qualtrics TSV (option-level endorsement recovered with the same combinatorial
`parse_response` matcher used by the released pipeline); models come from the
full-grid JSONL runs (`parsed_answers` = selected option numbers per question).

Writes:
  data/processed/responses.parquet   (canonical; R reads this via arrow)
  data/processed/responses.csv       (mirror; base-R readable, no packages)
  outputs/logs/01_ingest.log.json    (seed-free provenance: counts, inputs, git)

Run:  python analysis/py/01_ingest.py
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent          # analysis/py
ANALYSIS = HERE.parent                            # analysis/
RAW = ANALYSIS / "data" / "raw"
PROC = ANALYSIS / "data" / "processed"
LOGS = ANALYSIS / "outputs" / "logs"
CONFIG = ANALYSIS / "config"

# ── canonical topic / level / condition vocab ──
TOPICS = ["astronomy", "fruits", "martial_arts", "fabrics"]          # canonical
LEVELS = ["control", "repeat_short", "repeat_long", "distractor"]
# Qualtrics TSV uses fruits_v2 for the fruits topic; map both ways.
HUMAN_TSV_TOKEN = {"astronomy": "astronomy", "fruits": "fruits_v2",
                   "martial_arts": "martial_arts", "fabrics": "fabrics"}
FL_TOPIC = {"FL_32": "martial_arts", "FL_33": "fruits", "FL_34": "astronomy", "FL_35": "fabrics"}


# ── question bank: per-topic ordered content questions, option texts, truth ──
def load_question_bank() -> dict:
    """topic -> list of dicts: {q_id, options:{n:text}, answer:set[int]}, content only, in file order."""
    bank: dict[str, list[dict]] = {}
    for topic in TOPICS:
        qfile = RAW / "question_banks" / topic / "questions.yaml"
        qdata = yaml.safe_load(qfile.open())
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


def load_option_labels() -> dict:
    """(topic, q_id, opt_n) -> {option_type, cue_match}, from the blind sidecar if present."""
    f = ANALYSIS / "stimuli" / "option_labels.yaml"
    if not f.exists():
        return {}
    doc = yaml.safe_load(f.open()) or {}
    out = {}
    for topic, qs in (doc.get("topics") or {}).items():
        for q_id, opts in (qs or {}).items():
            for opt_n, lab in (opts or {}).items():
                ot = lab.get("option_type")
                if ot is True:      # YAML parsed unquoted `true` as boolean
                    ot = "true"
                elif ot is False:
                    ot = "false"
                out[(topic, str(q_id), int(opt_n))] = {
                    "option_type": ot,
                    "cue_match": lab.get("cue_match"),
                }
    return out


def parse_response(raw: str, all_opts: list[str]) -> frozenset:
    """Recover the set of selected option TEXTS from a Qualtrics multi-select cell.

    Qualtrics joins selected choice texts with ', ' (or ',') and choice texts can
    themselves contain commas, so a plain split is wrong — try every subset whose
    join reproduces the cell (same approach as multi_v6/data_prep.parse_response)."""
    if not raw:
        return frozenset()
    for r in range(len(all_opts) + 1):
        for combo in combinations(all_opts, r):
            if ",".join(combo) == raw:
                return frozenset(combo)
    for r in range(len(all_opts) + 1):
        for combo in combinations(all_opts, r):
            if ", ".join(combo) == raw:
                return frozenset(combo)
    return frozenset(s.strip() for s in raw.split(","))


# ── humans ──
def ingest_humans(bank: dict) -> list[dict]:
    tsv = RAW / "human_results_multi_v6.tsv"
    with tsv.open(encoding="utf-16") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = rows[2:]  # Qualtrics: two header/meta rows below the field names

    # dedupe by PROLIFIC_PID, keep the longest-duration finished submission
    best: dict[str, dict] = {}
    for r in rows:
        pid = (r.get("PROLIFIC_PID") or "").strip()
        if not pid:
            continue
        dur = float(r.get("Duration (in seconds)", 0) or 0)
        if pid not in best or dur > float(best[pid].get("Duration (in seconds)", 0) or 0):
            best[pid] = r

    out: list[dict] = []
    for pid, r in best.items():
        if (r.get("Finished") or "").strip() != "True":
            continue
        if not (r.get("martial_arts_cond") or "").strip():
            continue
        group = (r.get("group") or "").strip()
        fl_raw = (r.get("FL_27_DO") or "").strip()
        fl_order = [FL_TOPIC.get(c) for c in fl_raw.split("|") if c in FL_TOPIC]

        # WHOLE-PARTICIPANT attention exclusion (user directive 2026-07-30): keep a
        # respondent only if they passed the attention check in ALL 4 topics. This is
        # the `longdata_strict` rule — reconcile.py matches that file 1:1.
        attn_pass_all = True
        for topic in TOPICS:
            tok = HUMAN_TSV_TOKEN[topic]
            g = (r.get(f"{tok}_AT_{tok}") or "").strip()
            c = (r.get(f"{tok}_AT_correct") or "").strip()
            given = frozenset(g.split(",")) if g else frozenset()
            corr = frozenset(c.split(",")) if c else frozenset()
            if given != corr:
                attn_pass_all = False
                break
        if not attn_pass_all:
            continue

        for topic in TOPICS:
            tok = HUMAN_TSV_TOKEN[topic]
            level = (r.get(f"{tok}_cond") or "").strip()
            if level not in LEVELS:
                continue
            position = (fl_order.index(topic) + 1) if topic in fl_order else None
            t_qs = float(r.get(f"{tok}_timing_qs_Page Submit", 0) or 0)
            rt_ms = int(round(t_qs * 1000)) if t_qs else None

            for q in bank[topic]:
                q_id = q["q_id"]
                opts = q["options"]
                all_texts = list(opts.values())
                raw_cell = (r.get(f"{tok}_{q_id}") or "").strip()
                given_texts = parse_response(raw_cell, all_texts)
                for opt_n, opt_text in opts.items():
                    out.append({
                        "respondent_id": pid,
                        "agent": "human",
                        "model_name": None,
                        "sample_idx": None,
                        "group_id": int(group) if group.isdigit() else None,
                        "position": position,
                        "modality": "audio",
                        "topic": topic,
                        "level": level,
                        "question_id": q_id,
                        "option_id": f"{q_id}_o{opt_n}",
                        "option_n": opt_n,
                        "option_is_true": bool(opt_n in q["answer"]),
                        "endorsed": bool(opt_text in given_texts),
                        "rt_ms": rt_ms,
                        "attn_pass_all": True,
                        "system": "human",
                        "encode_mode": None,
                    })
    return out


# ── models (shared row emitter for a single JSONL, filtered to one condition) ──
def _emit_model_rows(jsonl: Path, bank: dict, *, keep_cond, model_name: str,
                     system: str, encode_mode) -> list[dict]:
    """keep_cond: a condition_id string, or None to accept every row in the file."""
    out: list[dict] = []
    for line in jsonl.open():
        r = json.loads(line)
        cond = r.get("condition_id")
        if keep_cond is not None and cond != keep_cond:
            continue
        topic = str(r.get("topic_id") or "").strip()
        level = r.get("level")
        if topic not in TOPICS or level not in LEVELS:
            continue
        sample_idx = r.get("repeat_index")
        parsed = r.get("parsed_answers") or {}
        for qi, q in enumerate(bank[topic], start=1):
            q_id = q["q_id"]
            selected = set(int(x) for x in (parsed.get(str(qi)) or []))
            for opt_n in q["options"]:
                out.append({
                    # unique per trial: a model draw is independent per (topic, level) cell,
                    # so the identity must carry topic+level (repeat_index resets per cell).
                    "respondent_id": f"{model_name}:{cond}:{topic}:{level}:r{sample_idx}",
                    "agent": "model",
                    "model_name": model_name,
                    "sample_idx": int(sample_idx) if sample_idx is not None else None,
                    "group_id": None,
                    "position": None,
                    "modality": "text",
                    "topic": topic,
                    "level": level,
                    "question_id": q_id,
                    "option_id": f"{q_id}_o{opt_n}",
                    "option_n": opt_n,
                    "option_is_true": bool(opt_n in q["answer"]),
                    "endorsed": bool(opt_n in selected),
                    "rt_ms": None,
                    "attn_pass_all": True,
                    "system": system,
                    "encode_mode": encode_mode,
                })
    return out


def ingest_models(bank: dict) -> list[dict]:
    models_cfg = yaml.safe_load((CONFIG / "models.yaml").open())
    prompt_cond = models_cfg["prompting_condition"]        # C2
    compactor_cond = models_cfg["compactor_condition"]     # C2-stream
    repo_root = ANALYSIS.parents[2]  # simulating-memory/
    out: list[dict] = []

    for m in models_cfg["models"]:
        name = m["name"]
        # prompting side (condition C2)
        pj = repo_root / m["prompting_path"]
        if pj.exists():
            out += _emit_model_rows(pj, bank, keep_cond=prompt_cond, model_name=name,
                                    system="prompting", encode_mode=None)
        else:
            print(f"  WARN prompting jsonl missing: {pj}")
        # compactor side (condition C2-stream)
        cj = repo_root / m["compactor_path"]
        if cj.exists():
            out += _emit_model_rows(cj, bank, keep_cond=compactor_cond,
                                    model_name=f"{name}__wm",
                                    system="compactor", encode_mode=compactor_cond)
        else:
            print(f"  WARN compactor jsonl missing: {cj}")
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ANALYSIS).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    bank = load_question_bank()
    labels = load_option_labels()

    rows = ingest_humans(bank) + ingest_models(bank)
    df = pd.DataFrame(rows)

    # attach blind option_type / cue_match labels (null if sidecar not yet built)
    def _lab(row, key):
        return (labels.get((row["topic"], row["question_id"], row["option_n"])) or {}).get(key)
    df["option_type"] = df.apply(lambda r: _lab(r, "option_type"), axis=1)
    df["cue_match"] = df.apply(lambda r: _lab(r, "cue_match"), axis=1)

    # column order per schema §2 (+ helper columns)
    cols = ["respondent_id", "agent", "system", "encode_mode", "model_name",
            "sample_idx", "group_id", "position", "modality", "topic", "level",
            "question_id", "option_id", "option_n", "option_is_true",
            "option_type", "cue_match", "endorsed", "rt_ms", "attn_pass_all"]
    df = df[cols]

    PROC.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "responses.csv", index=False)
    parquet_ok = True
    try:
        df.to_parquet(PROC / "responses.parquet", index=False)
    except Exception as e:      # pyarrow/numpy ABI issues — CSV is the R-readable fallback
        parquet_ok = False
        print(f"  WARN parquet write failed ({e}); responses.csv written as canonical fallback.")

    n_lab = int(df["option_type"].notna().sum())
    log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_rev(),
        "inputs": {
            "human_tsv": "data/raw/human_results_multi_v6.tsv",
            "option_labels_present": bool(labels),
        },
        "n_rows": len(df),
        "n_human_rows": int((df.agent == "human").sum()),
        "n_model_rows": int((df.agent == "model").sum()),
        "n_humans": int(df.loc[df.agent == "human", "respondent_id"].nunique()),
        "models": sorted(df.loc[df.agent == "model", "model_name"].dropna().unique().tolist()),
        "option_type_labeled_rows": n_lab,
        "option_type_labeled": bool(n_lab == len(df)),
    }
    (LOGS / "01_ingest.log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    if not log["option_type_labeled"]:
        print("\n  NOTE: option_type/cue_match not fully labeled yet "
              "(stimuli/option_labels.yaml missing or partial). Re-run 01_ingest after tagging completes.")


if __name__ == "__main__":
    main()
