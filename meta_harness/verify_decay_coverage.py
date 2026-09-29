"""Prove, per task, that a candidate's decay actually fires -- and fires per step.

HYPOTHESIS UNDER TEST
    H: `random_decay_v3`'s decay reaches all eight search tasks, once per tool-enabled
       turn, with exactly one decay rate per scored unit.

    Evidence that SUPPORTS H, for every one of the eight tasks:
      * decay rolls >= 1                        (the hook is reached at all)
      * keys actually dropped >= 1              (a roll can change the store, so the
                                                 roll count is not a no-op in disguise)
      * distinct decay rates == scored units    (one "ability" per observation in the
                                                 score distribution, so the population
                                                 has a spread rather than 3 rates)
      * rolls == tool-enabled step() calls      (per-STEP, not per-episode)
    Evidence that REJECTS H: any task with 0 rolls, or 0 keys dropped, or a
    distinct-rate count that differs from the scored-unit count.

    The baseline arm is run alongside as the attribution control: it must show 0 rolls
    and 0 keys dropped on every task. Without it, a roll count could be coming from the
    task harness rather than from the candidate.

WHY THIS EXISTS
    The predecessors `random_decay` and `random_decay_v2` hook `encode()` (v1 also
    `recall()`). Three of the eight search tasks call neither -- they drive `step()`.
    Measured before this script existed: nback 96 step / 0 encode / 0 recall,
    word_recognition 12 / 0 / 0, variable_mapping 40 / 0 / 0. And the rate was seeded
    from stimulus content, which is shared across participants on three tasks
    (semantic_story_recall drew 4 distinct rates across 200 rows, craft_task 3 across
    150). Neither defect was visible in any score; only counting made them visible.

HOW IT AVOIDS TWO TRAPS THE AUDIT HIT
    1. `inject.apply()` is not idempotent and `verify_interface.check()` applies
       internally, so a verify-then-run sequence in ONE process stacks the candidate
       class and doubles every roll. `apply()` now raises on a second apply of the same
       file (`meta_harness/inject.py`), which is confirmed by
       `meta_harness/test_inject_idempotence.py`. This script still runs each arm in its
       own SUBPROCESS, and never calls `verify_interface.check()`.
    2. An earlier instrumentation reported "0 rolls changed the store" everywhere. That
       was an artifact: its stub gated tool-call writes on a PROCESS-WIDE counter, so
       after two calls every store was empty and there was nothing left to drop. The
       stub here keeps its counters per INSTANCE and writes on every tool-enabled turn,
       and `keys_written` is reported per task so an empty-store artifact is visible
       rather than silent.

UNITS
    Decay rates are proportions in [0, DECAY_MAX]. Rolls, steps, keys, rows and scored
    units are integer counts. Nothing here is a humanlikeness value.

WHAT IT IS NOT
    Not a score. The stub LLM answers nonsense, so accuracies and humanlikeness from
    these runs are meaningless and are not reported. Only counts are.

    Not the real roll counts either, on one task. `run_recognition_stream` stops at the
    model's third parse-able error; the stub answers unparse-ably on purpose (a reply
    naming both "old" and "new" parses to None, which `score_game` records as neither
    correct nor an error), so word_recognition runs to its CEILING of
    `max_trials_per_game` here. The real run stops earlier -- the human anchor is a mean
    survival of 34.49 of 100 trials. Read that task's roll count as an upper bound.

Run (driver; spawns one subprocess per arm):
    python meta_harness/verify_decay_coverage.py

One arm only, in this process (used by the driver, not normally by hand):
    python meta_harness/verify_decay_coverage.py --arm random_decay_v3 --json <path>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# The eight tasks a candidate is optimized against, in the order
# meta_harness/cluster/candidate.sbatch runs them. `SIZES` are cut down so the whole
# sweep runs offline in seconds, but every task keeps >= 12 scored units so that "at
# least one key was dropped somewhere" is not a coin flip: at the mean rate of 0.175 a
# single-roll task drops nothing in a given unit with probability ~0.68, which over 12
# units is ~0.01.
#
# Digit span keeps `n_participants=1` and `sequences_per_span > 2` deliberately: that is
# the branch `src/score.py::_digit_span_participant_scores` takes under
# `cluster/search_set.yaml` (`split_by_seq = len(pid_set) <= 1 and len(seq_set) > 2`), so
# the scored unit here is a sequence_index, as it is in production. The expected count is
# never computed from these numbers -- it is read back out of score.py itself.
SIZES: dict[str, dict[str, Any]] = {
    "wm_digit_span_forward": dict(min_span=2, max_span=4, sequences_per_span=12,
                                  n_participants=1, stimuli_seed=42),
    "wm_digit_span_reverse": dict(min_span=2, max_span=4, sequences_per_span=12,
                                  n_participants=1, stimuli_seed=42),
    "wm_nback": dict(n_repeat=4, n_repeats_per_participant=1, stimuli_seed=123),
    "wm_word_recognition": dict(words_json_path="data/words.json", n_repeat=12,
                                max_trials_per_game=100, stimuli_seed=42),
    "wm_variable_mapping": dict(names_json_path="data/names.json",
                                city_json_path="data/city.json",
                                n_repeat=4, n_runs_per_participant=3, stimuli_seed=42),
    "wm_narrative_qa": dict(data_json_path="data/narrative_QA.json",
                            n_participants=12, stimuli_seed=42),
    "wm_semantic_story_recall": dict(repeats_per_stimulus=3),
    "wm_craft_task": dict(data_json_path="data/craft_task.json",
                          n_participants=4, n_tasks=3, stimuli_seed=42),
}

ARMS = {
    "baseline": "meta_harness/candidates/baseline/harness.py",
    "random_decay_v3": "meta_harness/candidates/random_decay_v3/harness.py",
}


# ---------------------------------------------------------------------------
# stub LLM
# ---------------------------------------------------------------------------

def _make_stub_class():
    from bench.core.llm import LLM, LLMResponse, LLMToolResponse

    class WritingStub(LLM):
        """Writes a fresh key on every tool-enabled turn. Per-INSTANCE state only.

        Two keys on the first tool-enabled turn, one on each turn after. Not arbitrary:
        `WorkingMemoryAgent._tool_call_cap()` is `max(6, tool_interactions * 1.5)`, so a
        stub writing two keys per turn would exhaust the budget on a long task (2t > 1.5t)
        and its later writes would be dropped, which would look like a decay effect. With
        1 + t calls against max(6, 1.5t) the budget is never the binding constraint.

        The first-turn pair matters for the five batch tasks: they get exactly one
        tool-enabled turn, so a single key would make "was anything dropped" a coin flip
        per unit rather than a measurement.
        """

        def __init__(self) -> None:
            self.tool_turns = 0
            self.writes = 0
            self.plain_calls = 0

        def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512, **kw):
            self.plain_calls += 1
            # Names both "old" and "new" so wm_word_recognition's `_parse_old_new`
            # returns None. An unparsed reply is neither correct nor an error, so the
            # third-error stop never fires and the task runs to its trial ceiling.
            return LLMResponse(text="not old, not new: different. 1: A")

        def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                                temperature=0.0, max_tokens=256, **kw):
            last_is_user = bool(messages) and messages[-1].get("role") == "user"
            if tools and last_is_user:
                self.tool_turns += 1
                n = 2 if self.tool_turns == 1 else 1
                calls = []
                for _ in range(n):
                    self.writes += 1
                    calls.append({
                        "id": f"stub-{self.writes}",
                        "type": "function",
                        "function": {
                            "name": "write_memory",
                            "arguments": json.dumps({"key": f"k{self.writes}",
                                                     "value": f"v{self.writes}"}),
                        },
                    })
                return LLMToolResponse(tool_calls=calls)
            return LLMToolResponse(
                tool_calls=[], content="not old, not new: different. 1: A",
                finish_reason="stop")

    return WritingStub


# ---------------------------------------------------------------------------
# one arm, in this process
# ---------------------------------------------------------------------------

def _install_counting_class(registry: list, stub_factory) -> None:
    """Wrap the CURRENTLY injected agent class in a counting subclass and rebind it.

    This is a second rebinding, not a second `inject.apply()` of the candidate file: the
    candidate class is subclassed once, so its `step()` override runs once per turn. Any
    additional `load_candidate` + `apply` of the same file would stack the override and
    double every roll, which is the hazard `inject.apply()` now raises on.
    """
    import bench.core.wm_agent as WA
    from meta_harness.inject import apply as inject_apply

    Injected = WA.WorkingMemoryAgent

    class Counting(Injected):  # type: ignore[misc,valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            # Each agent gets its OWN stub. `evaluate()` takes a single llm and shares
            # it across every agent in the task, which would make the stub's per-turn
            # write schedule process-wide -- precisely the artifact that produced a
            # bogus "0 rolls changed the store" in an earlier instrumentation.
            self.llm = stub_factory()
            self.n_step = 0
            self.n_step_tools = 0
            self.n_encode = 0
            self.n_recall = 0
            registry.append(self)

        def step(self, user_message: str, *, allow_tools: bool = True,
                 max_tokens: int = 1024) -> str:
            self.n_step += 1
            if allow_tools:
                self.n_step_tools += 1
            return super().step(user_message, allow_tools=allow_tools,
                                max_tokens=max_tokens)

        def encode(self, content: Any) -> Any:
            # `encode()` calls `self.step()`, so on a batch task n_step == n_encode.
            self.n_encode += 1
            return super().encode(content)

        def recall(self, recall_prompt: str | None = None, max_tokens: int = 512) -> str:
            self.n_recall += 1
            return super().recall(recall_prompt, max_tokens=max_tokens)

    shim = ModuleType("mh_counting_shim")
    shim.WorkingMemoryAgent = Counting  # type: ignore[attr-defined]
    inject_apply(shim, strict=True)


def run_arm(arm: str, out_json: Path) -> int:
    import bench.cli  # noqa: F401  -- binds the names in every task module first

    from meta_harness.inject import apply, load_candidate

    cand = load_candidate(ROOT / ARMS[arm])
    apply(cand)
    decay_max = getattr(cand, "DECAY_MAX", None)

    registry: list = []
    Stub = _make_stub_class()
    _install_counting_class(registry, Stub)

    import importlib

    import score as S

    results: dict[str, Any] = {"arm": arm, "decay_max": decay_max, "tasks": {}}
    tmp_root = Path(tempfile.mkdtemp(prefix=f"mh_decay_{arm}_"))

    for task in SIZES:
        mod = importlib.import_module(f"bench.tasks.{task}")
        out_dir = tmp_root / task
        (out_dir / "tasks").mkdir(parents=True, exist_ok=True)
        registry.clear()
        mod.evaluate(
            llm=Stub(),
            out_dir=out_dir,
            model_cfg={"name": "stub", "backend": "stub"},
            temperature=0.0,
            debug=False,
            max_parallel_participants=1,   # single-threaded: the registry is a plain list
            **SIZES[task],
        )
        agents = list(registry)
        bare = task[len("wm_"):]
        # Scored units come from score.py itself, applied to the rows this arm actually
        # wrote -- never from arithmetic over SIZES. `_digit_span_participant_scores`
        # switches grouping on a predicate over the rows, so a config change must not be
        # able to move the expected count without moving the measured one.
        scored_units = int(S.llm_scores(bare, out_dir, "compactor").size)
        rates = sorted({round(getattr(a, "_decay_rate", 0.0), 12) for a in agents}) \
            if any(hasattr(a, "_decay_rate") for a in agents) else []
        results["tasks"][bare] = {
            "agents": len(agents),
            "rows": sum(1 for _ in (out_dir / "tasks" / f"{task}.jsonl").open()),
            "scored_units": scored_units,
            "step_calls": sum(a.n_step for a in agents),
            "step_calls_tools_on": sum(a.n_step_tools for a in agents),
            "encode_calls": sum(a.n_encode for a in agents),
            "recall_calls": sum(a.n_recall for a in agents),
            "decay_rolls": sum(getattr(a, "decay_rolls", 0) for a in agents),
            "rolls_that_dropped": sum(
                getattr(a, "decay_rolls_that_dropped", 0) for a in agents),
            "keys_dropped": sum(getattr(a, "decay_keys_dropped", 0) for a in agents),
            # One stub instance is shared by every agent in a task (each `evaluate()`
            # gets a single llm), so summing per agent would multiply by the agent
            # count. De-duplicate by identity.
            "keys_written": sum(
                {id(a.llm): a.llm.writes for a in agents
                 if hasattr(getattr(a, "llm", None), "writes")}.values()),
            "distinct_rates": len(rates),
            "rate_min": rates[0] if rates else None,
            "rate_max": rates[-1] if rates else None,
        }
        print(f"  {arm:<16} {bare:<22} "
              f"steps {results['tasks'][bare]['step_calls']:>5} "
              f"rolls {results['tasks'][bare]['decay_rolls']:>5} "
              f"dropped {results['tasks'][bare]['keys_dropped']:>5}", flush=True)

    out_json.write_text(json.dumps(results, indent=2))
    return 0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

COLS = [
    ("step_calls", "step()", 7),
    ("encode_calls", "encode", 7),
    ("recall_calls", "recall", 7),
    ("decay_rolls", "rolls", 7),
    ("rolls_that_dropped", "rolls>0", 8),
    ("keys_dropped", "keys-", 7),
    ("keys_written", "keys+", 7),
    ("distinct_rates", "rates", 6),
    ("scored_units", "units", 6),
]


def _print_table(res: dict[str, Any]) -> None:
    arm = res["arm"]
    print(f"\n=== arm: {arm}   (DECAY_MAX = {res['decay_max']}, a proportion)")
    head = f"{'task':<22}" + "".join(f"{lab:>{w}}" for _, lab, w in COLS) + f"{'rate range':>22}"
    print(head)
    print("-" * len(head))
    for task, t in res["tasks"].items():
        rng = ("-" if t["rate_min"] is None
               else f"{t['rate_min']:.4f}..{t['rate_max']:.4f}")
        print(f"{task:<22}"
              + "".join(f"{t[k]:>{w}}" for k, _, w in COLS)
              + f"{rng:>22}")


def _check(v3: dict[str, Any], base: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    for task, t in v3["tasks"].items():
        if t["decay_rolls"] < 1:
            fails.append(f"{task}: 0 decay rolls -- the hook is never reached")
        if t["keys_dropped"] < 1:
            fails.append(f"{task}: 0 keys dropped over {t['decay_rolls']} rolls "
                         f"({t['keys_written']} keys written)")
        if t["distinct_rates"] != t["scored_units"]:
            fails.append(f"{task}: {t['distinct_rates']} distinct rates but "
                         f"{t['scored_units']} scored units")
        if t["decay_rolls"] != t["step_calls_tools_on"]:
            fails.append(f"{task}: {t['decay_rolls']} rolls over "
                         f"{t['step_calls_tools_on']} tool-enabled steps -- not per-step")
    for task, t in base["tasks"].items():
        if t["decay_rolls"] or t["keys_dropped"]:
            fails.append(f"baseline/{task}: {t['decay_rolls']} rolls, "
                         f"{t['keys_dropped']} keys dropped -- the counts are not "
                         f"attributable to the candidate")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.arm:
        if args.json is None:
            ap.error("--arm requires --json")
        return run_arm(args.arm, args.json)

    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="mh_decay_reports_") as td:
        for arm in ("baseline", "random_decay_v3"):
            path = Path(td) / f"{arm}.json"
            print(f"=== running arm {arm} in a subprocess", flush=True)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--arm", arm, "--json", str(path)],
                cwd=str(ROOT))
            if proc.returncode != 0 or not path.exists():
                print(f"!!! arm {arm} failed (exit {proc.returncode})")
                return 1
            reports[arm] = json.loads(path.read_text())

    for arm in ("baseline", "random_decay_v3"):
        _print_table(reports[arm])

    print("\nColumns: step()/encode/recall = LLM-agent method calls (integer counts); "
          "rolls = decay rolls executed;\nrolls>0 = rolls that removed at least one key; "
          "keys- = keys dropped; keys+ = keys the stub wrote;\nrates = distinct decay "
          "rates drawn (proportions); units = scored units per src/score.py.")
    print("\nword_recognition roll counts are an UPPER BOUND: the stub answers "
          "unparse-ably so the\nthird-error stop never fires and the task runs to its "
          "trial ceiling. Human mean survival\nis 34.49 of 100 trials.")

    fails = _check(reports["random_decay_v3"], reports["baseline"])
    print()
    if fails:
        print("HYPOTHESIS REJECTED:")
        for f in fails:
            print(f"  !!! {f}")
        return 1
    print("HYPOTHESIS SUPPORTED: every task shows >= 1 roll, >= 1 key dropped, "
          "one rate per scored unit,\nand exactly one roll per tool-enabled step; the "
          "baseline arm shows 0 rolls throughout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
