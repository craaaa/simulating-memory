"""Swap a candidate harness into `bench` without editing the task files.

Why this is not just `bench.core.wm_agent.WorkingMemoryAgent = Candidate`:
the seven `bench/tasks/wm_*.py` modules each do

    from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
    from ..core.working_memory import MAX_KEYS
    from .wm_prompt_parts import WM_SYSTEM_PROMPTS

so the names are bound into each task module's own namespace at import time.
Rebinding only the defining module leaves every task still pointing at the
original object.  This module rebinds the name everywhere it is already bound.

That also defines the real search surface.  A candidate may override any of:

    WorkingMemoryAgent   the harness class itself (encode/recall/step/...)
    MAX_KEYS             capacity -- searchable per the user's decision, but it
                         is the Cowan-2001 commitment, so a candidate that
                         moves it has to argue for that
    WM_SYSTEM_PROMPTS    the per-condition system prompts

Usage (inside a runner, before bench.cli dispatches any task):

    from meta_harness.inject import load_candidate, apply
    cand = load_candidate("meta_harness/candidates/baseline/harness.py")
    report = apply(cand)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

# Names a candidate is allowed to override, and where they originate.
OVERRIDABLE = {
    "WorkingMemoryAgent": "bench.core.wm_agent",
    "SummarizerAgent": "bench.core.wm_agent",
    "MAX_KEYS": "bench.core.working_memory",
    "NUM_SLOTS": "bench.core.working_memory",
    "WM_SYSTEM_PROMPTS": "bench.tasks.wm_prompt_parts",
    "CONDITION_PROMPTS": "bench.core.wm_agent",
    "TOOLS": "bench.core.wm_agent",
}


def load_candidate(path: str | Path, name: str = "mh_candidate") -> ModuleType:
    """Import a candidate harness file as a standalone module."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bench_modules() -> list[ModuleType]:
    """Every already-imported bench module, which is where stale bindings live."""
    return [m for n, m in list(sys.modules.items())
            if n.startswith("bench") and m is not None]


def apply(candidate: ModuleType, *, strict: bool = True) -> dict[str, Any]:
    """Rebind each name the candidate overrides, everywhere it is bound.

    Returns a report of what was replaced and where, so a candidate's record can
    show exactly which surface it touched rather than leaving it implicit.
    """
    overrides = {k: getattr(candidate, k) for k in OVERRIDABLE
                 if hasattr(candidate, k)}
    if not overrides:
        if strict:
            raise ValueError(
                "candidate overrides nothing; expected at least one of "
                + ", ".join(sorted(OVERRIDABLE))
            )
        return {"overrides": {}, "rebound": {}}

    # Import the defining modules so their canonical binding is updated too,
    # and so a candidate that only overrides MAX_KEYS still takes effect in code
    # that reads it lazily.
    for origin in set(OVERRIDABLE[k] for k in overrides):
        try:
            importlib.import_module(origin)
        except ImportError:
            pass

    rebound: dict[str, list[str]] = {k: [] for k in overrides}
    for mod in _bench_modules():
        for key, value in overrides.items():
            if hasattr(mod, key):
                setattr(mod, key, value)
                rebound[key].append(mod.__name__)

    missing = [k for k, v in rebound.items() if not v]
    if missing and strict:
        raise RuntimeError(
            "candidate overrides names that are bound nowhere in the imported "
            f"bench modules: {missing}. Import bench.cli before calling apply()."
        )

    return {
        "overrides": {k: getattr(v, "__name__", repr(v)[:60])
                      for k, v in overrides.items()},
        "rebound": rebound,
    }


def describe(report: dict[str, Any]) -> str:
    lines = []
    for key, mods in sorted(report.get("rebound", {}).items()):
        lines.append(f"  {key}: rebound in {len(mods)} module(s)")
        for m in sorted(mods):
            lines.append(f"      {m}")
    return "\n".join(lines) if lines else "  (nothing rebound)"
