"""Axis A4: does variable-mapping fail the way humans fail?

Why this axis had to exist before any proposer iteration was spent.

`variable_mapping` is the most attractive cell in the whole table and was, until
this module, completely unguarded. It holds the largest headroom (humanlikeness
0.355, so ~0.645 available, which is 0.081 of the 8-task mean), it is the most
precisely measured task (min credible delta 0.017, see `metric_noise.py`), and
none of A1 (digit span), A2 (word recognition) or A3 (story recall) touch it.

That combination is exactly the shape of a metric that gets gamed. The model sits
at 0.992 accuracy against a human 0.731-among-answered, and the model's score
distribution is a point mass -- 99% of participants at exactly 1.0, two unique
values, sd 0.069. There is no partial-credit structure to exploit, so the *only*
way to move this task is to manufacture failures. A candidate that injects
uniform noise into retrieval would move it a long way, and nothing in the
evaluation would have objected.

Two things about the human data make a real guard possible.

1. **Humans make interference errors, not random ones.** Classifying each human
   error by what city they picked: 23.0% chose a city previously assigned to *that
   same name* (a stale binding), 46.1% chose a city belonging to *another* name,
   and only 30.9% chose a city never assigned to anyone. So 69.1% of errors are
   intrusions. The chance rate is not 0 -- the option generator puts previously
   assigned cities among the distractors, and on human error trials 57.7% of wrong
   options were already-assigned -- so the human bias toward intrusions is
   +0.114 over chance, about 3 standard errors at n=152. Real, but modest, which
   is why it is the secondary statistic here.

2. **Human errors concentrate on high-interference items.** `relationCount` --
   how many assignments that name has accumulated -- averages **6.18 on errors
   against 4.46 on correct answers**. This is the primary statistic, because it is
   the one a noise injector provably cannot fake: dropping keys at random, or
   corrupting retrieval at a fixed rate, produces errors independent of
   `relationCount`, so the ratio goes to 1.0. Humans are at 1.39.

So A4 reports `rc_ratio = mean(relationCount | error) / mean(relationCount |
correct)`. Human 1.386. A uniform-noise candidate predicts 1.0. A candidate whose
mechanism is genuinely interference-driven predicts > 1.

Caveat, stated rather than hidden: at baseline the model makes only ~12 errors in
1500 questions, so its own A4 is almost unmeasurable. That is the point. A4 is a
guard that activates precisely when a candidate starts failing -- it has nothing
to say about the baseline and everything to say about any candidate that buys
variable_mapping improvement. `n_errors` is reported so a ratio computed from a
handful of errors is visibly untrustworthy rather than quietly believed.

Usage:
    python meta_harness/interference.py                     # human reference
    python meta_harness/interference.py --run <run_dir>     # a candidate
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "runs/human/working-memory-variable-mapping"
LETTERS = "ABCDEFGH"

# Measured on all 154 human participants by this module; see the docstring.
# Read from the task rather than hard-coded, so the interleaving assumption in
# model_trials cannot silently drift out of step with the stimulus generator.
try:
    from bench.tasks.variable_mapping import TURNS_PER_QUESTION
except Exception:                                    # analysis must not hard-fail
    TURNS_PER_QUESTION = 2

HUMAN_RC_RATIO = 1.386
# Where the humans sit between pure noise (1.0) and the most interference-shaped
# result their own error count and relation_count distribution permit: rc_ratio
# 1.3867 against a ceiling of 2.0372 over 152 errors in 759 trials. So humans are
# at 37% of their attainable interference structure, NOT at the ceiling -- which
# is what makes this the comparable target. The raw 1.386 is not reachable by a
# model run on the 1500-trial grid below several hundred errors, because the
# model's relation_count saturates at 10 (its ceiling is only ~1.26 at 35 errors).
#
# Caveat, and a comparability gap worth recording: the two sides' relation_count
# distributions differ in shape, not just scale. The human records span 1..10
# roughly evenly, while the model grid is [2,4,6,8,10,10,10,10,10,10] because
# variable_mapping.py asks one question per TURNS_PER_QUESTION=2 assignments. The
# normalisation makes the two comparable in position-between-noise-and-ceiling,
# which is the best available, but it does not make the underlying distributions
# the same.
HUMAN_RC_RATIO_NORMALIZED = 0.3728
HUMAN_INTRUSION_SHARE = 0.691
HUMAN_INTRUSION_CHANCE = 0.577
HUMAN_STALE_SHARE = 0.230


def _classify(selected: str, correct: str, name: str,
              assigned_before: dict[str, list[str]]) -> str:
    """intrusion from the same name, from another name, or a novel guess."""
    own_previous = [c for c in assigned_before.get(name, []) if c != correct]
    if selected in own_previous:
        return "stale_same_name"
    if any(selected in cities for cities in assigned_before.values()):
        return "intrusion_other_name"
    return "novel_guess"


def _summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    errs = [t for t in trials if not t["correct"]]
    oks = [t for t in trials if t["correct"]]
    kinds = {k: 0 for k in ("stale_same_name", "intrusion_other_name", "novel_guess")}
    for t in errs:
        kinds[t["kind"]] += 1
    n_err = len(errs)

    rc_err = [t["rc"] for t in errs if t["rc"] is not None]
    rc_ok = [t["rc"] for t in oks if t["rc"] is not None]
    rc_ratio = (float(np.mean(rc_err)) / float(np.mean(rc_ok))
                if rc_err and rc_ok and np.mean(rc_ok) else float("nan"))

    chance = [t["chance"] for t in errs if t["chance"] is not None]
    intr = kinds["stale_same_name"] + kinds["intrusion_other_name"]
    return {
        "n_trials": len(trials),
        "n_errors": n_err,
        "error_rate": round(n_err / len(trials), 4) if trials else None,
        "rc_mean_error": round(float(np.mean(rc_err)), 3) if rc_err else None,
        "rc_mean_correct": round(float(np.mean(rc_ok)), 3) if rc_ok else None,
        "rc_ratio": round(rc_ratio, 4) if rc_ratio == rc_ratio else None,
        "intrusion_share": round(intr / n_err, 4) if n_err else None,
        "intrusion_chance": round(float(np.mean(chance)), 4) if chance else None,
        "stale_share": round(kinds["stale_same_name"] / n_err, 4) if n_err else None,
        "kinds": kinds,
    }


def human_trials() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in sorted(glob.glob(str(HUMAN_DIR / "*.json"))):
        payload = json.load(open(f)).get("payload") or {}
        asg = sorted(payload.get("assignments") or [], key=lambda a: a.get("turn", 0))
        for q in payload.get("questions") or []:
            sel, cor = q.get("selectedCity"), q.get("correctCity")
            if sel is None:
                continue
            turn = q.get("turn", 0)
            before: dict[str, list[str]] = {}
            for a in asg:
                if a.get("turn", 0) >= turn:
                    break
                before.setdefault(a["name"], []).append(a.get("city"))
            wrong = [o for o in (q.get("options") or []) if o != cor]
            assigned = {c for cities in before.values() for c in cities}
            out.append({
                "correct": bool(q.get("correct")),
                "rc": q.get("relationCount"),
                "kind": _classify(sel, cor, q.get("name"), before),
                "chance": (float(np.mean([o in assigned for o in wrong]))
                           if wrong else None),
            })
    return out


def model_trials(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "tasks/wm_variable_mapping.jsonl"
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("condition_id") or r.get("condition")) != "C2":
            continue
        asg = sorted(r.get("assignments") or [], key=lambda a: a.get("turn", 0))
        answers = r.get("parsed_answers") or {}
        for q in r.get("questions") or []:
            idx = q.get("question_index")
            letter = answers.get(str(idx), answers.get(idx))
            cor = q.get("correct_city")
            options = q.get("options") or []
            # A malformed or missing answer is a real failure, but its *content*
            # is unknown, so it counts as an error with no classifiable kind --
            # recorded as a novel guess rather than dropped, which would flatter
            # a candidate that degrades by emitting garbage.
            selected = None
            if isinstance(letter, str) and letter in LETTERS:
                j = LETTERS.index(letter)
                if j < len(options):
                    selected = options[j]

            # Assignments are interleaved with questions, not front-loaded:
            # variable_mapping.py emits a question every TURNS_PER_QUESTION
            # assignments (`if turn % TURNS_PER_QUESTION == 0`), so question k is
            # asked immediately after assignment turn TURNS_PER_QUESTION * k and
            # the participant has seen ALL of turns 1..TURNS_PER_QUESTION*k.
            #
            # This previously read `q.get("turn", q.get("question_index", 0))` and
            # compared it against assignment turns. Model question records carry no
            # `turn` field, so it always fell back to question_index (1..10) while
            # assignment turns run 1..20 -- keeping k-1 assignments where 2k had
            # been presented, an under-count of roughly 2x. Real intrusions from
            # the unseen half were misclassified as `novel_guess`, biasing
            # intrusion_share down. `relation_count` is len(mapping), the number of
            # DISTINCT people known, which caps at TARGET_RELATIONS, so it cannot
            # be used to recover the window either.
            idx_i = int(idx) if idx is not None else 0
            cutoff = q.get("turn")
            if cutoff is None:
                cutoff = TURNS_PER_QUESTION * idx_i
            before: dict[str, list[str]] = {}
            for a in asg:
                if a.get("turn", 0) > cutoff:
                    break
                before.setdefault(a["name"], []).append(a.get("city"))
            assigned = {c for cities in before.values() for c in cities}
            wrong = [o for o in options if o != cor]
            is_ok = selected is not None and selected == cor
            out.append({
                "correct": is_ok,
                "rc": q.get("relation_count"),
                "kind": ("novel_guess" if selected is None
                         else _classify(selected, cor, q.get("name"), before)),
                "chance": (float(np.mean([o in assigned for o in wrong]))
                           if wrong else None),
            })
    return out


def rc_ratio_ceiling(rcs: list[float], n_errors: int) -> float | None:
    """The largest rc_ratio attainable at this error count, given this run's own
    relation_count distribution: put the n_errors highest-rc trials in the error
    group and everything else in the correct group.

    This exists because rc_ratio is NOT scale-free, and comparing it across runs
    with different error counts is invalid. relation_count is len(mapping), which
    saturates at TARGET_RELATIONS, so the distribution is bunched at the top and
    the attainable maximum rises with the error count. Measured on the real
    1500-trial model grid: 1.2525 at 12 errors, 1.2575 at 35, 1.2857 at 150,
    1.3750 at 400. Iteration 1's displacement scored rc_ratio 1.2575 at exactly 35
    errors -- its arithmetic maximum, with rc_mean_error 10.000 -- and full_context
    scored 1.2525 at exactly 12, also its maximum. So a bare rc_ratio near 1.25
    means "every error was late", not "errors are interference-shaped", and the
    human 1.386 is unreachable below several hundred errors.
    """
    if n_errors <= 0 or n_errors >= len(rcs):
        return None
    s = sorted((float(r) for r in rcs if r is not None), reverse=True)
    if n_errors >= len(s):
        return None
    err, cor = s[:n_errors], s[n_errors:]
    if not err or not cor or not sum(cor):
        return None
    denom = sum(cor) / len(cor)
    return (sum(err) / len(err)) / denom if denom else None


def a4(run_dir: Path) -> dict[str, Any]:
    """A4 for one candidate run, with its distance from the human reference."""
    trials = model_trials(run_dir)
    res = _summarize(trials)
    res["human_rc_ratio"] = HUMAN_RC_RATIO
    res["human_intrusion_share"] = HUMAN_INTRUSION_SHARE
    if res.get("rc_ratio") is not None:
        res["rc_ratio_distance"] = round(abs(res["rc_ratio"] - HUMAN_RC_RATIO), 4)

    # Raw intrusion_share is not comparable between the two sides: by late
    # questions nearly every city has been assigned to someone, so a wrong option
    # is "an intrusion" by construction. The model's chance level is 0.889 against
    # the humans' 0.577, and once the assignment window was fixed the model's raw
    # share saturated at 1.000 against a human 0.691 -- which reads as the model
    # being MORE intrusion-prone when it is only more exposed. Above chance the two
    # sides agree closely (model +0.111, human +0.114), so that is the comparable
    # quantity and the raw share is kept only for continuity.
    if res.get("intrusion_share") is not None and res.get("intrusion_chance") is not None:
        res["intrusion_above_chance"] = round(
            res["intrusion_share"] - res["intrusion_chance"], 4)
        res["human_intrusion_above_chance"] = round(
            HUMAN_INTRUSION_SHARE - HUMAN_INTRUSION_CHANCE, 4)

    # Scale-free version: where does this run sit between pure noise (rc_ratio 1.0,
    # errors independent of load) and the most interference-shaped result its own
    # error count permits? 0.0 is noise, 1.0 is the ceiling. This is what the guard
    # should compare, because the raw ratio's ceiling moves with n_errors.
    rcs = [t["rc"] for t in trials if t.get("rc") is not None]
    n_err = int(res.get("n_errors") or 0)
    ceil = rc_ratio_ceiling(rcs, n_err)
    res["rc_ratio_ceiling"] = round(ceil, 4) if ceil is not None else None
    if ceil is not None and res.get("rc_ratio") is not None and ceil > 1.0:
        res["rc_ratio_normalized"] = round(
            (res["rc_ratio"] - 1.0) / (ceil - 1.0), 4)
    else:
        res["rc_ratio_normalized"] = None
    res["human_rc_ratio_normalized"] = HUMAN_RC_RATIO_NORMALIZED
    if res.get("rc_ratio_normalized") is not None:
        res["rc_ratio_normalized_distance"] = round(
            abs(res["rc_ratio_normalized"] - HUMAN_RC_RATIO_NORMALIZED), 4)

    # Below this, the ratio is dominated by sampling noise and must not be
    # treated as evidence in either direction.
    res["trustworthy"] = bool(n_err >= 30)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="candidate run dir (per-model)")
    args = ap.parse_args()

    if args.run:
        print(json.dumps(a4(Path(args.run)), indent=2))
    else:
        h = _summarize(human_trials())
        print(json.dumps(h, indent=2))
        print("\nintrusion share above chance: "
              f"{h['intrusion_share'] - h['intrusion_chance']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
