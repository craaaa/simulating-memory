"""Do the guards actually fire? Synthetic tests for the ones real runs never triggered.

Motivation: a guard that has never fired is a guard that has never been tested. After
wave 0, A1 and A3 had both rejected real candidates, so they are known to work. A4
had not -- every candidate so far produces ~12 errors on variable_mapping out of
1500 questions, below A4's own trustworthiness threshold of 30, so the branch that
rejects an unstructured gain has never executed.

Rather than wait for a candidate to exercise it on a GPU, fabricate the two cases A4
exists to separate and check it separates them:

  noise-like        errors independent of interference load    -> rc_ratio ~ 1.0
  interference-like errors concentrated on high relationCount  -> rc_ratio > 1.0

If A4 cannot tell those apart on synthetic data where the ground truth is known by
construction, it cannot tell them apart on a candidate either.

Run:
    python meta_harness/test_guards.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from meta_harness import interference as IF  # noqa: E402
from meta_harness import score_candidate as SC  # noqa: E402

LETTERS = "ABCDEFGH"
CITIES = ["Chicago", "Phoenix", "Dallas", "Houston", "San Jose", "San Diego",
          "New York", "Philadelphia", "Los Angeles", "Austin"]
NAMES = ["Jennifer", "James", "Linda", "Robert", "Mary", "William", "John",
         "Patricia", "Michael", "Elizabeth"]


def _make_record(rng: random.Random, pid: int, mode: str,
                 error_rate: float) -> dict:
    """One variable_mapping-shaped record with errors injected by `mode`.

    mode="noise"        error probability is constant, independent of relation_count
    mode="interference" error probability rises with relation_count
    """
    assignments, questions, answers = [], [], {}
    owner: dict[str, str] = {}
    turn = 0
    rel: dict[str, int] = {}

    for q_idx in range(1, 11):
        # Two assignments per question, as the real task does.
        for _ in range(2):
            turn += 1
            name = rng.choice(NAMES)
            city = rng.choice(CITIES)
            assignments.append({"turn": turn, "name": name, "city": city,
                                "statement": f"{name} lives in {city}."})
            owner[name] = city
            rel[name] = rel.get(name, 0) + 1

        name = rng.choice(list(owner))
        correct = owner[name]
        rc = rel[name]
        distractors = rng.sample([c for c in CITIES if c != correct], 3)
        options = distractors + [correct]
        rng.shuffle(options)

        if mode == "noise":
            p_err = error_rate
        else:
            # Errors scale with interference load. The constant is arbitrary --
            # this is a synthetic fixture, not a psychological claim.
            p_err = min(0.95, error_rate * rc / 2.0)

        if rng.random() < p_err:
            # Choose a wrong option. Bias toward already-assigned cities in the
            # interference case, which is what a real interference mechanism does.
            wrong = [o for o in options if o != correct]
            if mode == "interference":
                assigned = [o for o in wrong if o in owner.values()]
                pick = rng.choice(assigned or wrong)
            else:
                pick = rng.choice(wrong)
        else:
            pick = correct

        answers[str(q_idx)] = LETTERS[options.index(pick)]
        questions.append({"question_index": q_idx, "turn": turn, "name": name,
                          "correct_city": correct, "options": options,
                          "relation_count": rc})

    return {"id": f"synthetic:p{pid}", "condition_id": "C2", "participant_id": pid,
            "assignments": assignments, "questions": questions,
            "parsed_answers": answers, "final_kv": {}, "step_logs": [],
            "metrics": {"n_questions": 10}}


def _write_run(tmp: Path, mode: str, error_rate: float, n: int = 150) -> Path:
    run = tmp / mode
    (run / "tasks").mkdir(parents=True, exist_ok=True)
    rng = random.Random(hash(mode) & 0xFFFF)
    rows = [_make_record(rng, i, mode, error_rate) for i in range(n)]
    (run / "tasks/wm_variable_mapping.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Matched error RATE, different error STRUCTURE. That is the whole point:
        # A4 must not be a disguised error-rate detector.
        noise = IF.a4(_write_run(tmp, "noise", 0.20))
        interf = IF.a4(_write_run(tmp, "interference", 0.20))

        print(f"{'case':16s} {'n_err':>6s} {'err_rate':>9s} {'rc_ratio':>9s} "
              f"{'intrusion':>10s} {'trust':>6s}")
        for name, r in (("noise", noise), ("interference", interf)):
            print(f"{name:16s} {r['n_errors']:6d} {r['error_rate']:9.3f} "
                  f"{r['rc_ratio']:9.3f} {r['intrusion_share']:10.3f} "
                  f"{str(r['trustworthy']):>6s}")

        if not noise["trustworthy"] or not interf["trustworthy"]:
            failures.append("fixtures did not produce >=30 errors; raise error_rate")
        if not (noise["rc_ratio"] < 1.15):
            failures.append(
                f"A4 does not flag noise: rc_ratio {noise['rc_ratio']:.3f} "
                f"should be < 1.15 when errors are rc-independent")
        if not (interf["rc_ratio"] > 1.15):
            failures.append(
                f"A4 does not clear interference: rc_ratio "
                f"{interf['rc_ratio']:.3f} should be > 1.15")

        # Now the guard itself, not just the statistic. Both cases are given the
        # same fabricated variable_mapping improvement, so only structure differs.
        vm_gain = SC.NOISE_FLOOR["variable_mapping"] + 0.05
        for name, ax in (("noise", noise), ("interference", interf)):
            rec = {"axes": {"A4": ax}, "delta_vs_baseline": {"variable_mapping": vm_gain}}
            guards: list[str] = []
            a4 = rec["axes"]["A4"]
            d = rec["delta_vs_baseline"]["variable_mapping"]
            if d > SC.NOISE_FLOOR["variable_mapping"]:
                if not a4.get("trustworthy"):
                    guards.append("too few errors")
                elif a4.get("rc_ratio") is not None and a4["rc_ratio"] < 1.15:
                    guards.append("unstructured")
            fired = bool(guards)
            want = name == "noise"
            print(f"guard on {name:13s} fired={fired}  expected={want}")
            if fired != want:
                failures.append(
                    f"guard on {name}: fired={fired}, expected={want}")

        # And a candidate that leaves variable_mapping alone owes nothing, even
        # with a noise-shaped A4 -- the guard must be conditional, not blanket.
        rec = {"axes": {"A4": noise}, "delta_vs_baseline": {"variable_mapping": 0.001}}
        d = rec["delta_vs_baseline"]["variable_mapping"]
        fired = d > SC.NOISE_FLOOR["variable_mapping"]
        print(f"guard when task untouched fired={fired}  expected=False")
        if fired:
            failures.append("guard fired on a candidate that did not move the task")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all guard tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
