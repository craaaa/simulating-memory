"""Vanilla (V1) capability-screener driver.

Runs the V1 (plain reading-comprehension) condition of the listening-QA full grid
against every candidate model *via OpenRouter*, into a SEPARATE tree
(runs/screener_vanilla/<slug>/tasks) so released runs/prompting/ is untouched.

Why V1 and not C1-C4: the C1-C4 prompts frame the model as a limited-memory human,
so they measure simulated behavior, not whether the model can answer the multiselect
questions at all. V1 hands the model the full transcript in-context and just asks it to
answer -> true ceiling.

Two-step, cost-gated:
    # free: hit OpenRouter /models, print which candidate ids are runnable
    python -m application.listening_qa.screener_vanilla_driver --validate-only
    # paid: run V1 n=3 for every available id, then print the screening table
    python -m application.listening_qa.screener_vanilla_driver --run
    # re-print the table from existing outputs (no calls)
    python -m application.listening_qa.screener_vanilla_driver --report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data import load_topics
from .prompting import content_questions

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "runs" / "screener_vanilla"
DOCUMENTS_DIR = REPO / "application" / "listening_qa" / "data"
OR_BASE = "https://openrouter.ai/api/v1"
PASS_THRESHOLD = 0.95  # vanilla ceiling: full text in-context -> capable model near 1.0
# A (topic, question) is "wrong at a level" if the majority of repeats at that level
# got it wrong. Systematic := wrong at >=2 of the 4 levels for the SAME question.
# That pattern means the model can't answer the question itself (comprehension/format
# failure), not that a harder text (repeat/distractor) confused it -- a single-level
# error is far more likely to be text-difficulty noise.
LEVEL_WRONG_MAJORITY = 0.5
SYSTEMATIC_MIN_LEVELS = 2

# slug (matches runs/prompting/ dir + analysis entries) -> OpenRouter model id.
# Deduped: id-duplicate slugs are dropped so we never pay twice for one model.
# extra_body carries per-model toggles (qwen3-8b thinking on/off).
ROSTER: List[Dict[str, object]] = [
    # -- provider-prefixed ids (already OpenRouter form) --
    {"slug": "cohere_command-a", "id": "cohere/command-a"},
    {"slug": "deepseek_deepseek-r1-distill-llama-70b", "id": "deepseek/deepseek-r1-distill-llama-70b"},
    {"slug": "deepseek_deepseek-v3.2", "id": "deepseek/deepseek-v3.2-exp"},
    {"slug": "google_gemma-3-27b-it", "id": "google/gemma-3-27b-it"},
    {"slug": "google_gemma-4-26b-a4b-it", "id": "google/gemma-4-26b-a4b-it"},
    {"slug": "google_gemma-4-31b-it", "id": "google/gemma-4-31b-it"},
    {"slug": "meta-llama_llama-3-8b-instruct", "id": "meta-llama/llama-3-8b-instruct"},
    {"slug": "meta-llama_llama-3.1-70b-instruct", "id": "meta-llama/llama-3.1-70b-instruct"},
    {"slug": "meta-llama_llama-3.1-8b-instruct", "id": "meta-llama/llama-3.1-8b-instruct"},
    {"slug": "meta-llama_llama-3.3-70b-instruct", "id": "meta-llama/llama-3.3-70b-instruct"},
    {"slug": "meta-llama_llama-4-maverick", "id": "meta-llama/llama-4-maverick"},
    {"slug": "meta-llama_llama-4-scout", "id": "meta-llama/llama-4-scout"},
    {"slug": "microsoft_phi-4", "id": "microsoft/phi-4"},
    {"slug": "microsoft_wizardlm-2-8x22b", "id": "microsoft/wizardlm-2-8x22b"},
    {"slug": "mistralai_mistral-small-24b-instruct-2501", "id": "mistralai/mistral-small-24b-instruct-2501"},
    {"slug": "moonshotai_kimi-k2-0905", "id": "moonshotai/kimi-k2-0905"},
    {"slug": "nousresearch_hermes-4-70b", "id": "nousresearch/hermes-4-70b"},
    {"slug": "nvidia_nemotron-3-nano-30b-a3b", "id": "nvidia/nemotron-3-nano-30b-a3b"},
    {"slug": "nvidia_nemotron-3-super-120b-a12b", "id": "nvidia/nemotron-3-super-120b-a12b"},
    {"slug": "nvidia_nemotron-3-ultra-550b-a55b", "id": "nvidia/nemotron-3-ultra-550b-a55b"},
    {"slug": "Qwen_Qwen2.5-32B-Instruct", "id": "qwen/qwen-2.5-32b-instruct"},
    {"slug": "qwen_qwen-2.5-72b-instruct", "id": "qwen/qwen-2.5-72b-instruct"},
    {"slug": "qwen_qwen3-235b-a22b-2507", "id": "qwen/qwen3-235b-a22b-2507"},
    {"slug": "qwen_qwen3-30b-a3b-instruct-2507", "id": "qwen/qwen3-30b-a3b-instruct-2507"},
    {"slug": "qwen_qwen3-30b-a3b-thinking-2507", "id": "qwen/qwen3-30b-a3b-thinking-2507"},
    {"slug": "qwen_qwen3-32b", "id": "qwen/qwen3-32b"},
    {"slug": "qwen_qwen3-next-80b-a3b-instruct", "id": "qwen/qwen3-next-80b-a3b-instruct"},
    {"slug": "qwen_qwen3.5-122b-a10b", "id": "qwen/qwen3.5-122b-a10b"},
    {"slug": "qwen_qwen3.5-397b-a17b", "id": "qwen/qwen3.5-397b-a17b"},
    {"slug": "qwen_qwen3.6-27b", "id": "qwen/qwen3.6-27b"},
    {"slug": "z-ai_glm-4.6", "id": "z-ai/glm-4.6"},
    # -- bare/blank ids: normalized to OpenRouter provider-prefixed form --
    {"slug": "gpt-4.1", "id": "openai/gpt-4.1"},
    {"slug": "openai_gpt-4.1-mini", "id": "openai/gpt-4.1-mini"},
    {"slug": "openai_gpt-5.4", "id": "openai/gpt-5.4"},
    {"slug": "gemini-3.1-pro-preview", "id": "google/gemini-3.1-pro-preview"},
    {"slug": "gemini-3.5-flash", "id": "google/gemini-3.5-flash"},
    {"slug": "gemini-3.6-flash", "id": "google/gemini-3.6-flash"},
    {"slug": "allenai_olmo-3-32b-think", "id": "allenai/olmo-3-32b-think"},
    {"slug": "anthropic_claude-opus-4.6", "id": "anthropic/claude-opus-4.6"},
    {"slug": "qwen_qwen3-8b_true", "id": "qwen/qwen3-8b", "extra_body": {"enable_thinking": True}},
    {"slug": "qwen_qwen3-8b_false", "id": "qwen/qwen3-8b", "extra_body": {"enable_thinking": False}},
]
# Intentionally dropped id-duplicates (same OpenRouter id as a kept slug):
#   qwen_qwen3-30b-a3b-instruct  == qwen_qwen3-30b-a3b-instruct-2507
#   qwen_qwen3-30b-a3b-thinking  == qwen_qwen3-30b-a3b-thinking-2507
#   meta-llama_Meta-Llama-3-8B-Instruct == meta-llama_llama-3-8b-instruct


def load_env_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found (env or .env).")


def fetch_available_ids(key: str) -> set:
    req = urllib.request.Request(
        f"{OR_BASE}/models", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return {m["id"] for m in data.get("data", [])}


def partition(available: set) -> Tuple[List[Dict], List[Dict]]:
    ok, missing = [], []
    for entry in ROSTER:
        (ok if entry["id"] in available else missing).append(entry)
    return ok, missing


def out_dir_for(slug: str) -> Path:
    return OUT_ROOT / slug / "tasks"


_TOPIC_QUESTIONS: Optional[Dict[str, List]] = None


def topic_questions() -> Dict[str, List]:
    """topic_id -> content questions, in the same order build_prompt numbered them."""
    global _TOPIC_QUESTIONS
    if _TOPIC_QUESTIONS is None:
        _TOPIC_QUESTIONS = {
            t.topic_id: content_questions(t.questions) for t in load_topics(DOCUMENTS_DIR)
        }
    return _TOPIC_QUESTIONS


def systematic_errors(slug: str) -> List[Dict]:
    """Flag (topic, question) pairs wrong at >=SYSTEMATIC_MIN_LEVELS levels.

    Returns a list of dicts: topic_id, question_index, question text, whether the
    gold answer is "None of the above", and which levels it was wrong at.
    """
    grid = out_dir_for(slug) / "application_listening_qa_full_grid.jsonl"
    if not grid.exists():
        return []
    rows = [json.loads(l) for l in grid.read_text().splitlines() if l.strip()]

    per_level_correct: Dict[Tuple[str, str, int], List[bool]] = defaultdict(list)
    for r in rows:
        if r.get("parsed_answers") is None:
            continue
        qs = topic_questions().get(r["topic_id"])
        if not qs:
            continue
        for i, q in enumerate(qs, start=1):
            pred = set(r["parsed_answers"].get(str(i), []))
            true = set(q.answer)
            per_level_correct[(r["topic_id"], r["level"], i)].append(pred == true)

    wrong_levels: Dict[Tuple[str, int], set] = defaultdict(set)
    for (topic_id, level, i), vals in per_level_correct.items():
        frac_correct = sum(vals) / len(vals) if vals else 0.0
        if frac_correct <= LEVEL_WRONG_MAJORITY:
            wrong_levels[(topic_id, i)].add(level)

    flagged = []
    for (topic_id, i), levels in wrong_levels.items():
        if len(levels) < SYSTEMATIC_MIN_LEVELS:
            continue
        q = topic_questions()[topic_id][i - 1]
        # gold answer text(s), to spot the "bad at None of the above" pattern directly
        gold_texts = [q.options.get(a, "?") for a in q.answer]
        flagged.append({
            "topic_id": topic_id,
            "question_index": i,
            "question": q.question,
            "gold_texts": gold_texts,
            "gold_is_none_of_above": any(t.strip().lower() == "none of the above" for t in gold_texts),
            "levels_wrong": sorted(levels),
        })
    return flagged


def run_one(entry: Dict, key: str, n_repeats: int, max_tokens: int) -> Tuple[str, int]:
    slug, model_id = entry["slug"], entry["id"]
    out_dir = out_dir_for(slug)
    cmd = [
        sys.executable, "-m", "application.listening_qa.full_grid_cli", "run",
        "--model", str(model_id),
        "--backend", "openai",
        "--base-url", OR_BASE,
        "--conditions", "V1",
        "--n-repeats-per-cell", str(n_repeats),
        "--max-tokens", str(max_tokens),
        "--out-dir", str(out_dir),
        "--max-parallel", "16",
    ]
    if entry.get("extra_body"):
        cmd += ["--extra-body-json", json.dumps(entry["extra_body"])]
    env = dict(os.environ, OPENROUTER_API_KEY=key)
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    print(f"  [{slug}] rc={proc.returncode} :: {tail[-1] if tail else ''}")
    return slug, proc.returncode


def summarize(slug: str) -> Optional[Dict]:
    """Per-cell V1 exact-match from the run summary, plus a low-score / parse flag."""
    summ = out_dir_for(slug) / "application_listening_qa_full_grid_summary.json"
    if not summ.exists():
        return None
    data = json.loads(summ.read_text())
    cells = data.get("cells", {})
    exacts = [c["exact_match_mean"] for c in cells.values() if c.get("exact_match_mean") is not None]
    parse_err = sum(c.get("parse_error_count", 0) for c in cells.values())
    overall = sum(exacts) / len(exacts) if exacts else None
    worst = min(exacts) if exacts else None
    sys_errs = systematic_errors(slug)
    return {
        "overall": overall,
        "worst_cell": worst,
        "n_cells": len(cells),
        "parse_errors": parse_err,
        "model": data.get("model"),
        "systematic_errors": sys_errs,
    }


def report() -> None:
    print(
        f"\n=== Vanilla V1 screening table "
        f"(PASS if overall exact-match >= {PASS_THRESHOLD:.2f} AND no systematic errors) ==="
    )
    print(f"{'slug':45s} {'overall':>8s} {'worst':>7s} {'parseErr':>9s} {'sysErr':>7s}  verdict")
    print("-" * 90)
    rows = []
    for entry in ROSTER:
        s = summarize(entry["slug"])
        if s is None:
            continue
        rows.append((entry["slug"], s))
    rows.sort(key=lambda r: (r[1]["overall"] is None, -(r[1]["overall"] or 0)))
    for slug, s in rows:
        ov = s["overall"]
        wc = s["worst_cell"]
        n_sys = len(s["systematic_errors"])
        if ov is None:
            verdict = "NO DATA"
        elif n_sys > 0:
            verdict = "FAIL systematic"
        elif ov >= PASS_THRESHOLD:
            verdict = "PASS"
        else:
            # low score, errors confined to <2 levels/topic: could be genuine
            # incapacity OR reasoning-truncation -- eyeball before FAILing.
            verdict = "REVIEW raw_response" if s["parse_errors"] else "FAIL"
        ovs = f"{ov:.3f}" if ov is not None else "  --"
        wcs = f"{wc:.3f}" if wc is not None else "  --"
        print(f"{slug:45s} {ovs:>8s} {wcs:>7s} {s['parse_errors']:>9d} {n_sys:>7d}  {verdict}")
    print("-" * 90)
    print("sysErr = # (topic,question) pairs wrong at >=2 of 4 levels -> comprehension/format")
    print("failure on that question, not text-difficulty noise. Any sysErr count -> FAIL")
    print("regardless of overall accuracy (a high mean can hide one persistently-missed question).")
    print("REVIEW = low score, no systematic pattern, WITH parse errors -> possible truncation;")
    print("eyeball runs/screener_vanilla/<slug>/tasks/*_full_grid.jsonl raw_response before FAILing.")
    print()
    any_sys = [(slug, s) for slug, s in rows if s["systematic_errors"]]
    if any_sys:
        print("=== Systematic error detail ===")
        for slug, s in any_sys:
            print(f"\n{slug}:")
            for e in s["systematic_errors"]:
                tag = " [gold=None of the above]" if e["gold_is_none_of_above"] else ""
                print(
                    f"  {e['topic_id']} Q{e['question_index']} wrong at {e['levels_wrong']}{tag}: "
                    f"{e['question'][:90]}"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    if args.report and not args.run:
        report()
        return

    key = load_env_key()
    available = fetch_available_ids(key)
    ok, missing = partition(available)

    print(f"OpenRouter serves {len(available)} models. Candidates: {len(ROSTER)}.")
    print(f"\nAVAILABLE ({len(ok)}):")
    for e in ok:
        print(f"  {e['slug']:45s} -> {e['id']}")
    print(f"\nNOT ON OPENROUTER ({len(missing)}):")
    for e in missing:
        print(f"  {e['slug']:45s} -> {e['id']}")

    if args.validate_only:
        return
    if not args.run:
        print("\n(pass --run to execute the paid screener on the AVAILABLE set)")
        return

    print(f"\nRunning V1 n={args.n_repeats} max_tokens={args.max_tokens} for {len(ok)} models...")
    for e in ok:
        run_one(e, key, args.n_repeats, args.max_tokens)
    report()


if __name__ == "__main__":
    main()
