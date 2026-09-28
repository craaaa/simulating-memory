"""`respond_first` MINUS eviction — iteration 6 ablation, written by the coordinator.

Contributes no new mechanism and pre-registers no mechanism of its own. It exists to
answer one question that `respond_first` cannot answer about itself: **is eviction still
doing any work once the response is ordered before the store update?**

WHY THIS ARM
------------
`respond_first` passes 10 of 14 pre-registered rows with the mechanism fully confirmed —
unparsed replies 0.00 per block at every n-back level, answered 14/14/14, nback 0.9596,
leak closed at 1.000, variable_mapping 0.695 — and is blocked by exactly one task:

    craft_task   0.8907 -> 0.8534   delta -0.0373 against a 0.030 floor

That loss is caused by EVICTION, not by response-ordering, and it was registered as a
named hazard before the run by both `evicting_reset` and `respond_first`. The reason is
structural to an inverted objective: a refusing store rejects the fifth write, and the
agent's repair attempts get truncated by the tool-call cap, so it ends holding FEWER
chunks. An evicting store admits those writes, so the agent ends holding more retained
material — and on `craft_task` more retained material means a MORE capable agent, which
is further from humans, which costs humanlikeness. Every unconditional evicting store
pays it: measured mean keys held 3.33 -> 3.67 under either eviction rule.

Response-ordering has nothing to do with the store. So this arm removes eviction and
keeps the ordering.

THE PREDICTION, WHICH CUTS BOTH WAYS
------------------------------------
* `craft_task` should return to the baseline, because nothing here touches the store.
* n-back should STILL be fixed. If the answer is emitted before any tool use, it no
  longer matters whether the budget is later exhausted by refusals — which is route 1,
  the refusal loop, not merely route 3. Response-ordering would then subsume route 1 as
  well, and eviction would be unnecessary rather than merely costly.

If n-back instead collapses toward the baseline's 6.82 at n=3, eviction is doing real
work and `craft_task`'s cost has to be paid some other way. Either outcome is
informative, which is why the arm is worth one 15-minute process.

WHAT IT INHERITS, AND THE ONE LINE THAT DIFFERS
-----------------------------------------------
Everything from `respond_first` except the store. Its `step()` — the three-path dispatch
that emits ACT 1 under `tool_choice="none"` and then ACT 2 with tools on the same
cumulative budget — the control-state block, the episode bookkeeping, `encode()`,
`recall()`, `reset_messages()`, `_tool_call_cap()`, the prompts and the tools are all
inherited verbatim. `__init__` is overridden to NOT install
`EpisodicDisplacementMemory`, leaving the baseline `WorkingMemory` with its refusal
intact. That is the whole difference, and `_assert_one_line_difference()` checks at
import that nothing else diverged.

The leak stays closed: closure lives in `step()` rebuilding the message list, which is
inherited untouched. That matters because the leak closure is the project's one
cross-substrate result (+0.32 to +0.34 on variable_mapping on two models) and an arm
that lost it would be worthless.

NOT IDEMPOTENT — read this before editing
-----------------------------------------
`inject.apply()` is not idempotent, and a double apply is what doubled every episodic
candidate's prompt delta on 2400 of 2400 n-back turns before it was caught. The parent
is imported exactly ONCE here, at module import time, which is before `run_candidate.py`
calls `apply()`. `apply()` now raises on a second apply of the same file, so the failure
mode is loud rather than silent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PARENT_PATH = _HERE.parent / "respond_first" / "harness.py"


def _load_parent_once() -> Any:
    """Import `respond_first` exactly once, under its own module name."""
    name = "mh_respond_only_parent"
    if name in sys.modules:
        return sys.modules[name]
    if not _PARENT_PATH.exists():
        raise FileNotFoundError(f"ablation needs {_PARENT_PATH}")
    spec = importlib.util.spec_from_file_location(name, _PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_PARENT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_RF = _load_parent_once()

# Re-exported unchanged, so the injector rebinds exactly what the parent rebinds.
MAX_KEYS = _RF.MAX_KEYS
TOOLS = _RF.TOOLS
CONDITION_PROMPTS = _RF.CONDITION_PROMPTS
SummarizerAgent = _RF.SummarizerAgent

_RFAgent = _RF.WorkingMemoryAgent


def _assert_one_line_difference() -> None:
    """Fail loudly if this stops being a one-line ablation.

    The arm is only interpretable because `__init__` is the sole difference. If a later
    edit overrides anything else here, the comparison against `respond_first` stops
    isolating eviction and starts measuring two things.
    """
    mine = {k for k in vars(WorkingMemoryAgent) if not k.startswith("__")}
    if mine:
        raise ValueError(
            f"respond_only overrides more than __init__: {sorted(mine)}. This arm is "
            f"only interpretable as respond_first MINUS eviction; anything else makes "
            f"it a second mechanism."
        )
    if "__init__" not in vars(_RFAgent):
        raise ValueError(
            "respond_first no longer overrides __init__, so there is nothing to "
            "subtract and this ablation is meaningless as written."
        )


class WorkingMemoryAgent(_RFAgent):                     # type: ignore[misc]
    """`respond_first` with the baseline store: the refusal is kept, the ordering is not.

    MRO: this class -> respond_first -> the live baseline agent. `step()` and everything
    else resolve to the parent's; only the store differs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Deliberately calls the BASELINE agent's __init__, skipping respond_first's,
        # whose only added effect is installing EpisodicDisplacementMemory. `self.wm`
        # therefore stays the baseline `WorkingMemory`, which REFUSES on a full store.
        super(_RFAgent, self).__init__(*args, **kwargs)


_assert_one_line_difference()


def _selftest_store_is_baseline() -> None:
    """Confirm the store actually refuses, which is the entire point of the arm.

    `respond_first`'s store would return a "displaced" message here. If this arm
    silently inherited it, the comparison would be against itself.
    """
    from bench.core.working_memory import WorkingMemory as _Base

    agent = WorkingMemoryAgent.__new__(WorkingMemoryAgent)
    store = _Base()
    cap = int(getattr(_RF, "MAX_KEYS", 4))
    for i in range(cap):
        store.write_key(f"k{i}", "v")
    res = store.write_key("overflow", "v")
    if "memory is full" not in res:
        raise ValueError(
            f"respond_only expects the BASELINE refusing store; a full-store write "
            f"returned {res!r}. If this says 'displaced', the ablation inherited the "
            f"parent's evicting store and measures nothing."
        )
    # And the class must not install a custom store at construction.
    if type(agent).__mro__[1] is not _RFAgent:
        raise ValueError("unexpected MRO; the ablation is not subclassing respond_first")


_selftest_store_is_baseline()


MANIFEST: dict[str, Any] = {
    "id": "respond_only",
    "parent": "respond_first",
    "capacity": MAX_KEYS,
    "decay": (
        "None. The baseline store is used unchanged, including its refusal on a full "
        "store. This arm subtracts eviction; it adds nothing."
    ),
    "role": (
        "ablation -- isolates whether eviction still does any work once the response "
        "is ordered before the store update. Contributes no new mechanism."
    ),
    "summary": (
        "respond_first minus eviction. Its step() reordering, control-state block, "
        "episode bookkeeping and every prompt are inherited verbatim; __init__ is "
        "overridden so the baseline refusing WorkingMemory stays in place. Written "
        "because craft_task's -0.0373 under respond_first is caused by eviction "
        "admitting writes a refusing store would reject -- more retained material "
        "means a more capable agent, which costs humanlikeness on an inverted "
        "objective -- while response-ordering does not touch the store at all. If "
        "n-back stays fixed here, response-ordering subsumes the refusal loop (route 1) "
        "as well as route 3 and eviction is unnecessary rather than merely costly; if "
        "n-back collapses, eviction is load-bearing and craft's cost must be paid "
        "another way."
    ),
    "injection": sorted({"WorkingMemoryAgent", "SummarizerAgent", "MAX_KEYS",
                         "CONDITION_PROMPTS", "TOOLS"}),
    "ablation_of": {
        "parent": "respond_first",
        "removed": "EpisodicDisplacementMemory (the evicting store)",
        "kept": "the ACT-1/ACT-2 response ordering and everything else, verbatim",
        "asserted_at_import": [
            "this class overrides __init__ and nothing else",
            "respond_first still overrides __init__, so there is something to subtract",
            "a full-store write REFUSES rather than displacing",
        ],
    },
}
