#!/usr/bin/env python3
"""Align LLM vs human on which *cell* individual samples tend to score higher on (listening QA).

Adapted from ``application/level_pair_preference_alignment.py`` (reading QA) for the
listening_qa task: ``doc_id`` -> ``topic_id`` (4 topics, not ~10 biographies),
single-select ``accuracy`` -> multi-select ``exact_match_accuracy``, and conditions
C1-C4 (standalone prompting) plus a synthetic ``WM`` condition (the working-memory
compactor, which only ever runs as C2).

Each **base comparison** is two cells within the same topic: **A** = (topic ``t``,
level ``l1``) vs **B** = (``t``, ``l2``), ``l1 != l2``. There are only
C(4,2)=6 level pairs x 4 topics = 24 of these — comparisons are always within a
topic, never across topics, so the level effect isn't confounded by one topic
being generally easier/harder than another.

**Preference is decided per individual sample, not per cell mean.** For each of
the 24 base comparisons, draw one random individual accuracy value from cell A's
raw draws (one participant, for human; one repeat, for a model condition) and one
from cell B's, and compare those two values directly. Repeat ``--n-samples-per-pair``
times per base comparison (default 2000) — this uses the real per-participant /
per-repeat variance already in the data instead of collapsing each cell to a single
fixed mean, and gives 24 * n_samples_per_pair total draws rather than a fixed n=24.
Ties (equal values) broken by coin flip.

Agreement = the condition's per-draw preferred side (A/B) matches the human's
per-draw preferred side, for that same draw.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

LEVELS: tuple[str, ...] = ("control", "repeat_short", "repeat_long", "distractor")
CONDITIONS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "WM")

DEFAULT_MODEL_SLUG = "meta-llama_llama-3.1-8b-instruct"
DEFAULT_STANDALONE_JSONL = (
    REPO_ROOT
    / "runs"
    / "prompting"
    / DEFAULT_MODEL_SLUG
    / "tasks"
    / "application_listening_qa_full_grid.jsonl"
)
DEFAULT_WM_JSONL = (
    REPO_ROOT
    / "runs"
    / "compactor"
    / DEFAULT_MODEL_SLUG
    / "tasks"
    / "wm_application_listening_qa_full_grid.jsonl"
)
DEFAULT_HUMAN_CSV = (
    REPO_ROOT
    / ".claude"
    / "worktrees"
    / "compare-task"
    / "application"
    / "prolific_study"
    / "multi_v6"
    / "longdata_strict.csv"
)
HUMAN_TOPIC_MAP = {
    "martial_arts": "martial_arts",
    "fruits_v2": "fruits",
    "astronomy": "astronomy",
    "fabrics": "fabrics",
}


def load_human_topic_level_accuracies(
    human_csv: Path,
) -> dict[str, dict[str, list[float]]]:
    """topic_id -> level -> list of per-participant exact-match accuracies (fraction of 5 questions correct)."""
    raw = list(csv.DictReader(human_csv.open()))
    by_trial: dict[tuple[str, str, str], list[int]] = {}
    for r in raw:
        topic = HUMAN_TOPIC_MAP.get(r["topic"])
        if topic is None:
            continue
        key = (r["pid"], topic, r["condition"])
        by_trial.setdefault(key, []).append(int(r["correct"]))

    out: dict[str, dict[str, list[float]]] = {}
    for (_pid, topic, level), corrects in by_trial.items():
        if level not in LEVELS:
            continue
        exact = sum(corrects) / len(corrects)
        out.setdefault(topic, {}).setdefault(level, []).append(exact)
    return out


def load_llm_topic_level_condition(
    standalone_jsonl: Path,
    wm_jsonl: Path,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """condition_id -> topic_id -> level -> list of exact_match_accuracy (one entry per trial)."""
    out: dict[str, dict[str, dict[str, list[float]]]] = {c: {} for c in CONDITIONS}

    for line in standalone_jsonl.open():
        r = json.loads(line)
        cond = r.get("condition_id")
        if cond not in out:
            continue
        topic = str(r.get("topic_id") or "").strip()
        level = r.get("level")
        if not topic or level not in LEVELS:
            continue
        acc = (r.get("metrics") or {}).get("exact_match_accuracy")
        if acc is None:
            continue
        out[cond].setdefault(topic, {}).setdefault(str(level), []).append(float(acc))

    for line in wm_jsonl.open():
        r = json.loads(line)
        if r.get("condition_id") not in ("C2", "C2-stream"):
            continue
        topic = str(r.get("topic_id") or "").strip()
        level = r.get("level")
        if not topic or level not in LEVELS:
            continue
        acc = (r.get("metrics") or {}).get("exact_match_accuracy")
        if acc is None:
            continue
        out["WM"].setdefault(topic, {}).setdefault(str(level), []).append(float(acc))

    return out


def enumerate_base_pairs(topics: list[str]) -> list[tuple[str, str, str]]:
    """All C(4,2)=6 level pairs per topic: (topic, l1, l2). 24 total across 4 topics."""
    pairs: list[tuple[str, str, str]] = []
    for t in topics:
        for l1, l2 in combinations(LEVELS, 2):
            pairs.append((t, l1, l2))
    return pairs


def _individual_preference(
    stats: dict[str, list[float]],
    level_a: str,
    level_b: str,
    rng: np.random.Generator,
) -> str | None:
    """Draw ONE individual sample from level_a's draws and ONE from level_b's draws
    (not the means) and return which is higher: ``"A"``, ``"B"``, coin-flip on tie."""
    a = stats.get(level_a) or []
    b = stats.get(level_b) or []
    if not a or not b:
        return None
    va = a[int(rng.integers(0, len(a)))]
    vb = b[int(rng.integers(0, len(b)))]
    if va > vb:
        return "A"
    if vb > va:
        return "B"
    return "A" if int(rng.integers(0, 2)) == 0 else "B"


def _rng_for_seed(seed: int, salt_str: str) -> np.random.Generator:
    salt = zlib.adler32(salt_str.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, salt]))


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Analytic 95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return None, None
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(center - half), float(center + half)


def run_individual_samples(
    *,
    llm: dict[str, dict[str, dict[str, list[float]]]],
    human: dict[str, dict[str, list[float]]],
    base_pairs: list[tuple[str, str, str]],
    eval_conditions: list[str],
    seed: int,
    n_samples_per_pair: int,
) -> dict[str, Any]:
    hits = {c: 0 for c in eval_conditions}
    denom = {c: 0 for c in eval_conditions}
    n_h = 0
    # per-rep hit record per condition (None = that condition had no data for this rep),
    # aligned by rep index so conditions can be paired for a paired-difference test.
    rep_hits: dict[str, list[bool | None]] = {c: [] for c in eval_conditions}

    level_pairs = sorted({(l1, l2) for (_t, l1, l2) in base_pairs})
    hits_by_level_pair = {c: {lp: 0 for lp in level_pairs} for c in eval_conditions}
    denom_by_level_pair = {c: {lp: 0 for lp in level_pairs} for c in eval_conditions}

    for pair_idx, (topic, l1, l2) in enumerate(base_pairs):
        h_stats = human.get(topic) or {}
        lp = (l1, l2)
        for rep in range(n_samples_per_pair):
            rng = _rng_for_seed(seed, f"listening_indiv_{pair_idx}_{rep}")
            h_side = _individual_preference(h_stats, l1, l2, rng)
            if h_side is None:
                continue
            n_h += 1
            for c in eval_conditions:
                m_stats = llm.get(c, {}).get(topic) or {}
                m_side = _individual_preference(m_stats, l1, l2, rng)
                if m_side is None:
                    rep_hits[c].append(None)
                    continue
                denom[c] += 1
                denom_by_level_pair[c][lp] += 1
                is_hit = m_side == h_side
                rep_hits[c].append(is_hit)
                if is_hit:
                    hits[c] += 1
                    hits_by_level_pair[c][lp] += 1

    acc: dict[str, dict[str, Any]] = {}
    for c in eval_conditions:
        lo, hi = wilson_ci(hits[c], denom[c]) if denom[c] else (None, None)
        by_lp: dict[str, dict[str, Any]] = {}
        for lp in level_pairs:
            h_ = hits_by_level_pair[c][lp]
            d_ = denom_by_level_pair[c][lp]
            lp_lo, lp_hi = wilson_ci(h_, d_) if d_ else (None, None)
            by_lp[f"{lp[0]}_vs_{lp[1]}"] = {
                "agreement": (float(h_) / float(d_)) if d_ else None,
                "ci_lo": lp_lo,
                "ci_hi": lp_hi,
                "n": d_,
            }
        acc[c] = {
            "agreement": (float(hits[c]) / float(denom[c])) if denom[c] else None,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": denom[c],
            "by_level_pair": by_lp,
        }

    return {
        "n_base_pairs": len(base_pairs),
        "n_samples_per_pair": n_samples_per_pair,
        "n_draws_human_defined": n_h,
        "hits": hits,
        "denom": denom,
        "accuracy_by_condition": acc,
        "rep_hits": rep_hits,
    }


def _exact_binomial_two_sided_p(k: int, n: int) -> float:
    """Two-sided p-value for McNemar's test on ``n`` discordant pairs (``k`` = smaller count).
    Uses the standard normal approximation with continuity correction — exact binomial
    coefficients overflow at the pair counts here (thousands), and the normal approximation
    is standard practice for McNemar's test at this sample size anyway (no scipy needed)."""
    import math

    if n == 0:
        return 1.0
    b_minus_c = n - 2 * k  # = |b - c|
    numerator = max(abs(b_minus_c) - 1, 0)  # continuity correction
    chi2_stat = (numerator**2) / n
    z = math.sqrt(chi2_stat)
    phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return min(1.0, 2 * (1 - phi))


def paired_difference_test(
    rep_hits: dict[str, list[bool | None]],
    cond_a: str,
    cond_b: str,
    *,
    seed: int,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Paired comparison of cond_a vs cond_b's agreement rate, using only reps where both
    have a defined hit (same human draw underlies both, so this is a proper paired test —
    not two independent samples).

    - McNemar's test on the discordant pairs (b = a-hit/b-miss, c = a-miss/b-hit).
    - Bootstrap 95% CI on the paired difference (mean(hit_a) - mean(hit_b)), resampling
      rep indices (not the two conditions separately) so the pairing is preserved.
    """
    a_list = rep_hits[cond_a]
    b_list = rep_hits[cond_b]
    n_common = min(len(a_list), len(b_list))
    a_arr: list[bool] = []
    b_arr: list[bool] = []
    for i in range(n_common):
        av, bv = a_list[i], b_list[i]
        if av is None or bv is None:
            continue
        a_arr.append(av)
        b_arr.append(bv)

    a_np = np.asarray(a_arr, dtype=bool)
    b_np = np.asarray(b_arr, dtype=bool)
    n_paired = len(a_np)

    b_discordant = int(np.sum(a_np & ~b_np))  # a hit, b miss
    c_discordant = int(np.sum(~a_np & b_np))  # a miss, b hit

    if b_discordant + c_discordant > 0:
        p_value = _exact_binomial_two_sided_p(
            min(b_discordant, c_discordant), b_discordant + c_discordant
        )
    else:
        p_value = 1.0

    observed_diff = float(np.mean(a_np)) - float(np.mean(b_np)) if n_paired else None

    boot_diffs = []
    if n_paired:
        rng = _rng_for_seed(seed, f"listening_paired_boot_{cond_a}_{cond_b}")
        for _ in range(n_boot):
            idx = rng.integers(0, n_paired, size=n_paired)
            boot_diffs.append(float(np.mean(a_np[idx])) - float(np.mean(b_np[idx])))
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    else:
        ci_lo, ci_hi = None, None

    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_paired": n_paired,
        "agreement_a": float(np.mean(a_np)) if n_paired else None,
        "agreement_b": float(np.mean(b_np)) if n_paired else None,
        "observed_diff_a_minus_b": observed_diff,
        "bootstrap_ci_lo": float(ci_lo) if ci_lo is not None else None,
        "bootstrap_ci_hi": float(ci_hi) if ci_hi is not None else None,
        "mcnemar_b_discordant": b_discordant,
        "mcnemar_c_discordant": c_discordant,
        "mcnemar_p_value": p_value,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-jsonl", type=Path, default=DEFAULT_STANDALONE_JSONL)
    ap.add_argument("--wm-jsonl", type=Path, default=DEFAULT_WM_JSONL)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-samples-per-pair",
        type=int,
        default=2000,
        help="Individual-sample draws per (topic, level1, level2) base comparison. "
        "Total draws = 24 * this. Default 2000 -> 48000 draws.",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Bootstrap resamples for the paired WM-vs-prompting difference CI.",
    )
    args = ap.parse_args()

    human = load_human_topic_level_accuracies(args.human_csv)
    llm = load_llm_topic_level_condition(args.standalone_jsonl, args.wm_jsonl)

    topics = sorted(
        set(human.keys()) | {t for c in CONDITIONS for t in llm.get(c, {}).keys()}
    )
    base_pairs = enumerate_base_pairs(topics)

    result = run_individual_samples(
        llm=llm,
        human=human,
        base_pairs=base_pairs,
        eval_conditions=list(CONDITIONS),
        seed=args.seed,
        n_samples_per_pair=args.n_samples_per_pair,
    )
    rep_hits = result.pop("rep_hits")

    paired_tests = [
        paired_difference_test(rep_hits, "WM", c, seed=args.seed, n_boot=args.n_boot)
        for c in ["C1", "C2", "C3", "C4"]
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_json or (
        REPO_ROOT
        / "application"
        / "comparisons"
        / f"listening_level_pair_alignment_{stamp}.json"
    )

    summary = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "method": "individual_sample_draws_within_topic",
            "n_samples_per_pair": args.n_samples_per_pair,
            "levels": list(LEVELS),
            "eval_conditions": list(CONDITIONS),
            "topics": topics,
            "standalone_jsonl": str(args.standalone_jsonl),
            "wm_jsonl": str(args.wm_jsonl),
            "human_csv": str(args.human_csv),
        },
        "result": result,
        "paired_wm_vs_prompting": paired_tests,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")

    def _fmt(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "na"

    acc = result["accuracy_by_condition"]
    print(
        f"human draws defined: {result['n_draws_human_defined']} / {result['n_base_pairs'] * result['n_samples_per_pair']}"
    )
    for c in CONDITIONS:
        row = acc[c]
        ci = (
            f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}]"
            if row["ci_lo"] is not None
            else "na"
        )
        print(
            f"  {c}: agreement_with_human={_fmt(row['agreement'])}  95% CI={ci}  (n={row['n']})"
        )
        for lp_key, lp_row in row["by_level_pair"].items():
            lp_ci = (
                f"[{_fmt(lp_row['ci_lo'])}, {_fmt(lp_row['ci_hi'])}]"
                if lp_row["ci_lo"] is not None
                else "na"
            )
            print(
                f"      {lp_key}: agreement={_fmt(lp_row['agreement'])}  95% CI={lp_ci}  (n={lp_row['n']})"
            )

    print(
        "\npaired WM vs. prompting-method conditions (McNemar + bootstrap CI on the difference):"
    )
    for t in paired_tests:
        diff = t["observed_diff_a_minus_b"]
        ci = f"[{_fmt(t['bootstrap_ci_lo'])}, {_fmt(t['bootstrap_ci_hi'])}]"
        sig = "*" if t["mcnemar_p_value"] < 0.05 else " "
        print(
            f"  WM vs {t['cond_b']}: WM={_fmt(t['agreement_a'])} {t['cond_b']}={_fmt(t['agreement_b'])} "
            f"diff(WM-{t['cond_b']})={_fmt(diff)} 95% CI={ci} "
            f"McNemar p={t['mcnemar_p_value']:.4g}{sig} (n_paired={t['n_paired']})"
        )


if __name__ == "__main__":
    main()
