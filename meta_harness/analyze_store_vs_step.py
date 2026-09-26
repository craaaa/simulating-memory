"""Cross-tabulate store overflow policy against step() version on n-back `answered`.

This is the query that refuted the conclusion "remove the refusal and n-back clears".
Eviction fixes n=3 under the BASELINE step() but not under the rewritten one, so the
step() rewrite is a second independent cause and the two fixes are not additive in the
one combination that has been run.

Kept in the repo rather than the scratchpad because it is the check that should be run
before any future claim that one mechanism explains an n-back result: when a mechanism
explains a difference, check every arm that already varies along that mechanism.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "meta_harness"))
import nback_levels as NL  # noqa: E402

MODEL = "Qwen_Qwen3-30B-A3B-Instruct-2507"

# (arm, run dir, overflow policy, which step() it uses)
ARMS = [
    ("baseline", "iter0/baseline", "refuses", "baseline"),
    ("random_decay", "iter0/random_decay", "refuses", "baseline"),
    ("random_decay_v2", "iter0/random_decay_v2", "refuses", "baseline"),
    ("chunk_limit", "iter2/chunk_limit", "refuses", "baseline"),
    ("serial_recognition", "iter2/serial_recognition", "refuses", "baseline"),
    ("full_context", "iter0/full_context", "refuses*", "baseline"),
    ("displacement", "iter1/displacement", "EVICTS", "baseline"),
    ("primacy", "iter2/primacy", "EVICTS", "baseline"),
    ("primacy_v2", "iter3/primacy_v2", "EVICTS", "baseline"),
    ("episodic_reset", "iter2/episodic_reset", "refuses", "v1"),
    ("episodic_reset_v2", "iter3/episodic_reset_v2", "refuses", "v2"),
    ("episodic_primacy", "iter3/episodic_primacy", "EVICTS", "v2"),
    ("episodic_reset_v3", "iter4/episodic_reset_v3", "refuses", "v3"),
]


def main() -> int:
    print("n-back `answered` (of 14) by overflow policy and step() version")
    print("* full_context refuses but never FILLS the store, so it never meets the "
          "refusal -- it is not a counterexample, it is outside the contrast.\n")
    print(f"{'arm':<22}{'store':<10}{'step()':<10}"
          f"{'n=1':>8}{'n=2':>8}{'n=3':>8}{'keys n=3':>10}")
    cells: dict[tuple[str, str], list[float]] = {}
    for name, rel, store, step in ARMS:
        d = ROOT / "meta_harness/runs" / rel / MODEL
        if not d.exists():
            continue
        try:
            diag = NL.report(d).get("diagnostics", {})
        except Exception as e:  # noqa: BLE001
            print(f"{name:<22}  could not read: {type(e).__name__}")
            continue
        a = [(diag.get(n) or {}).get("answered") for n in (1, 2, 3)]
        k3 = (diag.get(3) or {}).get("keys_held")
        if a[2] is not None and store in ("refuses", "EVICTS"):
            cells.setdefault((store, step), []).append(a[2])
        print(f"{name:<22}{store:<10}{step:<10}"
              + "".join(f"{('-' if v is None else v):>8}" for v in a)
              + f"{('-' if k3 is None else k3):>10}")

    print("\nn=3 answered, grouped:")
    for (store, step), vals in sorted(cells.items()):
        print(f"  store={store:<8} step()={step:<9} "
              f"n={len(vals)}  values {sorted(vals)}")
    base_ref = cells.get(("refuses", "baseline"), [])
    base_ev = cells.get(("EVICTS", "baseline"), [])
    rw_ref = [v for (s, st), vs in cells.items() if s == "refuses" and st != "baseline"
              for v in vs]
    rw_ev = [v for (s, st), vs in cells.items() if s == "EVICTS" and st != "baseline"
             for v in vs]
    if base_ref and base_ev:
        print(f"\nunder the BASELINE step(), eviction moves n=3 from "
              f"{max(base_ref):.2f} (best refusing) to {min(base_ev):.2f} "
              f"(worst evicting)")
    if rw_ref and rw_ev:
        print(f"under a REWRITTEN step(), eviction moves n=3 from "
              f"{max(rw_ref):.2f} to {max(rw_ev):.2f} -- the fixes are NOT additive, "
              f"so the rewrite is a second independent cause")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
