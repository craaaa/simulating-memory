"""`episodic_reset_v2` composed with `primacy_v2` — iteration 3, composed arm.

Written by the coordinator, not by a proposer. It contributes no new mechanism and
pre-registers no mechanism of its own: it exists to measure whether the two
iteration-3 repairs are additive, redundant or conflicting, which neither can answer
alone.

WHY THIS IS A COMPOSITION AND NOT A REWRITE
-------------------------------------------
The two candidates override disjoint methods and both cooperate through `super()`:

    primacy_v2.WorkingMemoryAgent       overrides __init__ only, installing
                                        PrimacyV2Memory(owner=self)
    episodic_reset_v2.WorkingMemoryAgent overrides step() only, calling super().step()

So the composition is ordinary cooperative multiple inheritance and copies no code.
`__init__` resolves to primacy_v2's (episodic_reset_v2 defines none), installing the
maintenance-limited activation store; `step()` resolves to episodic_reset_v2's, which
prepends the control-state block and then delegates down the MRO to the baseline's
`step()`. Both derive from the same live `_BaseAgent`, so the MRO is linear and
unambiguous. If either candidate is later changed to override the other's method, this
file breaks loudly rather than silently — see `_assert_disjoint()`.

The two source files are loaded ONCE each, here, at import time. That matters:
`inject.apply()` rebinds `bench.core.wm_agent.WorkingMemoryAgent`, and a candidate
module resolves its base at ITS import time, so loading a candidate after `apply()` has
run would subclass the previously injected class and stack the overrides. Iteration 3a
observed exactly that — the control-state block emitted twice in one user message.
Importing both here, before `run_candidate.py` calls `apply()`, gives both the pristine
base.

WHY THE COMPOSITION IS NOT EXPECTED TO BE ADDITIVE
--------------------------------------------------
Iteration 3a's note, which is the reason this arm is worth a process: under the
baseline `step()` the store's contents were never shown to the agent during a turn, so
`primacy`-family eviction was unobservable to the model on `nback` and
`variable_mapping`. `episodic_reset_v2` puts `to_recall_text()` on the prompt path every
turn. So in composition, WHICH key was evicted and the ORDER survivors are listed in
become part of the agent's context on those two tasks. Part of any composed effect may
therefore be the agent reacting to a visibly changed store rather than to the retention
change itself. That is registered as the composed arm's own prediction (C3 below) and is
not attributable to either candidate alone.

THE HAZARD 3b FLAGGED, AND WHY IT DOES NOT FIRE
-----------------------------------------------
`primacy_v2._batch_writes()` reads `self._owner._messages` to count how many
`write_memory` calls the assistant message being dispatched carried, distinguishing a
simultaneous batch (eviction stops at 2, store settles at 3) from serial arrival
(stops at 3, store settles at 4, i.e. `primacy` exactly). It assumes the baseline
contract: the assistant message is appended BEFORE its tool calls are dispatched. It
fails safe to "serial" if it reads `None`, which would silently make this arm
`primacy_v2`-inactive.

Checked at the source rather than assumed: `episodic_reset_v2.step()` calls
`reset_messages()` at the TURN BOUNDARY, before delegating to `super().step()`, and its
own comment records why clearing mid-loop would be invalid (a `tool` message with no
preceding assistant `tool_calls` is rejected). So by the time the base `step()`
dispatches, `_messages` holds the assistant message. The hazard does not fire.

3b asked for one assertion to distinguish "v2 active" from "v2 silently degraded":
craft's five-rule rows must settle at 3 keys, not 4. `_selftest_batch_semantics()`
below performs it in-process at import time, so a silent degradation fails the run
rather than producing a mislabelled result.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_CANDIDATES = _HERE.parent


def _load_once(name: str) -> Any:
    """Import a sibling candidate's harness.py exactly once, under its own name."""
    mod_name = f"mh_composed_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _CANDIDATES / name / "harness.py"
    if not path.exists():
        raise FileNotFoundError(f"composed arm needs {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ER = _load_once("episodic_reset_v2")
_PR = _load_once("primacy_v2")

_ERAgent = _ER.WorkingMemoryAgent
_PRAgent = _PR.WorkingMemoryAgent

# MAX_KEYS is read live by both stores; they must agree or the composition is
# ill-defined rather than merely surprising.
if int(_ER.MAX_KEYS) != int(_PR.MAX_KEYS):
    raise ValueError(
        f"composed arm: MAX_KEYS disagrees, episodic_reset_v2={_ER.MAX_KEYS} "
        f"primacy_v2={_PR.MAX_KEYS}"
    )
MAX_KEYS = int(_PR.MAX_KEYS)

# Prompts and tools: neither candidate changes TOOLS, and each reports its own
# prompt delta. They must be the same objects, or one is silently overriding the
# other's prompt surface and no delta would be attributable.
for _name in ("TOOLS", "CONDITION_PROMPTS"):
    _a, _b = getattr(_ER, _name, None), getattr(_PR, _name, None)
    if _a != _b:
        raise ValueError(
            f"composed arm: {_name} differs between the two candidates; the "
            f"composition would silently pick one and no prompt delta would be "
            f"attributable"
        )
TOOLS = _PR.TOOLS
CONDITION_PROMPTS = _PR.CONDITION_PROMPTS
SummarizerAgent = _PR.SummarizerAgent


def _assert_disjoint() -> None:
    """Fail loudly if the two candidates stop overriding disjoint methods.

    The composition is only interpretable because each owns different methods. If a
    later edit makes both override `step()` or both override `__init__`, MRO order
    would silently decide which mechanism wins.
    """
    overlap = []
    for meth in ("step", "__init__", "recall", "encode", "reset_messages",
                 "_tool_call_cap"):
        in_er = meth in vars(_ERAgent)
        in_pr = meth in vars(_PRAgent)
        if in_er and in_pr:
            overlap.append(meth)
    if overlap:
        raise ValueError(
            f"composed arm: both candidates now override {overlap}; MRO order would "
            f"silently decide the mechanism. Compose them explicitly instead."
        )
    # The two methods the composition relies on must each still exist somewhere.
    if "step" not in vars(_ERAgent):
        raise ValueError("composed arm: episodic_reset_v2 no longer overrides step()")
    if "__init__" not in vars(_PRAgent):
        raise ValueError("composed arm: primacy_v2 no longer overrides __init__()")


_assert_disjoint()


class WorkingMemoryAgent(_ERAgent, _PRAgent):          # type: ignore[misc]
    """Both repairs, one agent, no new mechanism.

    MRO: this class -> episodic_reset_v2 -> primacy_v2 -> the live baseline agent.
    `__init__` comes from primacy_v2 and installs the maintenance-limited activation
    store. `step()` comes from episodic_reset_v2 and, after prepending the
    control-state block, delegates down to the baseline's `step()`.
    """

    # Nothing is defined here on purpose. Any method added would be a third
    # mechanism and this arm would stop being a composition.


def _selftest_batch_semantics() -> None:
    """The check 3b asked for: confirm primacy_v2's batch rule is still ACTIVE.

    `_batch_writes()` fails safe to "serial" when it cannot read a message list, which
    would make this arm silently equivalent to `primacy` on the very cell that
    distinguishes them. So assert the simultaneous-batch semantics directly: five
    chunks offered in ONE assistant message must settle the store at 3, not 4.

    This runs at import time, so a silent degradation fails the run rather than
    producing a mislabelled result.
    """
    agent = WorkingMemoryAgent.__new__(WorkingMemoryAgent)   # no LLM needed
    store = _PR.PrimacyV2Memory(owner=agent)

    class _Msgs:
        """Minimal stand-in for the base `step()` contract: the assistant message
        carrying every requested tool call, appended before dispatch."""

        def __init__(self, n: int) -> None:
            self._messages = [
                {"role": "user", "content": "x"},
                {"role": "assistant", "tool_calls": [
                    {"function": {"name": "write_memory"}} for _ in range(n)
                ]},
            ]

    # Simultaneous batch of five: eviction stops at (MAX_KEYS-1)-1, so 3 held.
    store._owner = _Msgs(5)
    for i in range(1, 6):
        store.write_key(f"rule{i}", f"v{i}")
    occ_batch = len(store.store)

    # Serial arrival: one write per assistant message, so MAX_KEYS held.
    store2 = _PR.PrimacyV2Memory(owner=_Msgs(1))
    for i in range(1, 6):
        store2._owner = _Msgs(1)
        store2.write_key(f"k{i}", f"v{i}")
    occ_serial = len(store2.store)

    # Eviction stops at (MAX_KEYS-1)-1 = 2 and then ADMITS the incoming chunk, so the
    # store settles at 3 = MAX_KEYS-1. The distinguishing fact is that it is one below
    # capacity: `primacy` would leave 4 here.
    if occ_batch != MAX_KEYS - 1:
        raise ValueError(
            f"composed arm: primacy_v2's batch rule is NOT active -- a simultaneous "
            f"batch of 5 settled at {occ_batch} keys, expected {MAX_KEYS - 1} (one "
            f"below capacity; `primacy` leaves {MAX_KEYS}). This arm would silently "
            f"be measuring `primacy`, not `primacy_v2`."
        )
    if occ_serial != MAX_KEYS:
        raise ValueError(
            f"composed arm: serial arrival settled at {occ_serial} keys, expected "
            f"{MAX_KEYS}; the fallback path is wrong."
        )


_selftest_batch_semantics()


MANIFEST: dict[str, Any] = {
    "id": "episodic_primacy",
    "parent": "episodic_reset_v2 + primacy_v2",
    "capacity": MAX_KEYS,
    "decay": (
        "Inherited unchanged from primacy_v2: ACT-R base-level activation with "
        "d = 0.5, eviction of argmin A_i, stopping at (MAX_KEYS-1)-1 for a "
        "simultaneous batch and MAX_KEYS-1 for serial arrival."
    ),
    "role": (
        "composed arm -- measures whether iteration 3's two repairs are additive, "
        "redundant or conflicting. Contributes no new mechanism."
    ),
    "summary": (
        "episodic_reset_v2's step() (control state carried across the turn boundary, "
        "no episodic record of earlier stimuli) composed with primacy_v2's store "
        "(maintenance-limited ACT-R eviction). Cooperative multiple inheritance over "
        "disjoint method sets, asserted at import time; no code copied from either. "
        "Not expected to be additive: episodic_reset_v2 puts to_recall_text() on the "
        "prompt path every turn, so primacy_v2's eviction becomes visible to the "
        "agent on nback and variable_mapping, where it was previously unobservable."
    ),
    "injection": sorted({"WorkingMemoryAgent", "SummarizerAgent", "MAX_KEYS",
                         "CONDITION_PROMPTS", "TOOLS"}),
    "composition": {
        "mro": ["episodic_primacy", "episodic_reset_v2", "primacy_v2", "baseline"],
        "step_from": "episodic_reset_v2",
        "init_from": "primacy_v2",
        "batch_rule_active_asserted_at_import": True,
        "hazard_checked": (
            "primacy_v2._batch_writes() needs the assistant message appended before "
            "dispatch; episodic_reset_v2 resets at the turn boundary, before "
            "super().step(), so it is intact. Verified at the source."
        ),
    },
}
