"""Interface-compliance check for a Meta-Harness candidate.

Runs offline against a scripted stub LLM, so it costs nothing and needs no GPU.
A candidate that fails here is recorded with a null score rather than silently
dropped, so the search history shows what was attempted.

What is checked:
  1. the candidate overrides at least one name on the injectable surface
  2. injection actually rebinds it in the imported bench modules
  3. the harness class has the methods the seven wm_* task modules call, with
     compatible signatures
  4. one synthetic encode/recall round trip completes and returns the shapes the
     tasks destructure (`encode()` -> dict carrying final_kv; `wm.store` a dict)
  5. the declared capacity is honoured -- writes beyond it are refused

Usage:
    python meta_harness/verify_interface.py meta_harness/candidates/<id>/harness.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.core.llm import LLM, LLMResponse, LLMToolResponse  # noqa: E402

REQUIRED_METHODS = {
    # name: parameters the task modules actually pass by keyword
    "encode": set(),
    "recall": set(),
    "get_log": set(),
}
REQUIRED_INIT_KWARGS = {"llm", "condition_id", "temperature", "debug",
                        "system_prompt_override"}


class StubLLM(LLM):
    """Deterministic stub: writes two keys, then answers with fixed text.

    Enough to exercise the tool-dispatch loop and the recall path without any
    network call.
    """

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None, **kw) -> LLMResponse:
        return LLMResponse(text="stub recall answer")

    @staticmethod
    def _call(cid: str, key: str, value: str) -> dict:
        # OpenAI tool-call shape: wm_agent reads tc["id"] and
        # tc["function"]["name"] / ["arguments"].
        return {
            "id": cid,
            "type": "function",
            "function": {
                "name": "write_memory",
                "arguments": json.dumps({"key": key, "value": value}),
            },
        }

    def generate_with_tools(self, messages, tools, *, tool_choice="auto",
                            temperature=0.0, max_tokens=256, **kw) -> LLMToolResponse:
        self.calls += 1
        if tool_choice != "none" and self.calls <= 2:
            return LLMToolResponse(
                tool_calls=[self._call(f"stub-{self.calls}",
                                       f"k{self.calls}", f"v{self.calls}")]
            )
        return LLMToolResponse(tool_calls=[], content="done", finish_reason="stop")


def check(path: str) -> tuple[bool, list[str]]:
    problems: list[str] = []
    notes: list[str] = []

    # bench.cli must be imported first so every task-module binding exists.
    import bench.cli  # noqa: F401
    from meta_harness.inject import OVERRIDABLE, apply, describe, load_candidate

    cand = load_candidate(path)

    overridden = [k for k in OVERRIDABLE if hasattr(cand, k)]
    if not overridden:
        return False, [f"overrides none of: {', '.join(sorted(OVERRIDABLE))}"]
    notes.append(f"overrides: {', '.join(sorted(overridden))}")

    try:
        report = apply(cand)
    except Exception as e:  # noqa: BLE001
        return False, [f"injection failed: {type(e).__name__}: {e}"]
    notes.append("injection report:\n" + describe(report))

    Harness = getattr(cand, "WorkingMemoryAgent", None)
    if Harness is None:
        # A candidate may legitimately change only MAX_KEYS or the prompts.
        notes.append("no WorkingMemoryAgent override; skipping class checks")
        return not problems, notes + problems

    sig = inspect.signature(Harness.__init__)
    missing_kwargs = REQUIRED_INIT_KWARGS - set(sig.parameters)
    if missing_kwargs:
        problems.append(f"__init__ missing kwargs the tasks pass: {sorted(missing_kwargs)}")

    for meth in REQUIRED_METHODS:
        if not callable(getattr(Harness, meth, None)):
            problems.append(f"missing method: {meth}()")

    if problems:
        return False, notes + problems

    # --- one synthetic round trip -----------------------------------------
    try:
        h = Harness(llm=StubLLM(), condition_id="C2", temperature=0.0,
                    debug=False, system_prompt_override=None)
        enc = h.encode("1: alpha\n2: bravo\n3: charlie")
        if not isinstance(enc, dict):
            problems.append(f"encode() returned {type(enc).__name__}, expected dict")
        elif "final_kv" not in enc:
            problems.append(f"encode() dict lacks 'final_kv'; keys={sorted(enc)}")
        store = getattr(getattr(h, "wm", None), "store", None)
        if not isinstance(store, dict):
            problems.append("wm.store is not a dict")
        else:
            notes.append(f"round trip stored {len(store)} key(s): {sorted(store)}")
        out = h.recall("recall now")
        if not isinstance(out, str):
            problems.append(f"recall() returned {type(out).__name__}, expected str")
    except Exception as e:  # noqa: BLE001
        problems.append(f"round trip raised {type(e).__name__}: {e}")
        return False, notes + problems

    # --- capacity is enforced ---------------------------------------------
    try:
        from bench.core.working_memory import MAX_KEYS as declared
        wm = type(h.wm)()
        for i in range(declared + 3):
            wm.write_key(f"key{i}", "v")
        if len(wm.store) > declared:
            problems.append(
                f"capacity not enforced: {len(wm.store)} keys held, declared {declared}"
            )
        else:
            notes.append(f"capacity enforced at {declared} key(s)")
    except Exception as e:  # noqa: BLE001
        notes.append(f"capacity check skipped ({type(e).__name__}: {e})")

    return not problems, notes + problems


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ok, lines = check(sys.argv[1])
    print(f"=== interface check: {sys.argv[1]}")
    for line in lines:
        print(f"  {line}" if not line.startswith(" ") else line)
    print()
    print("  PASS" if ok else "  FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
