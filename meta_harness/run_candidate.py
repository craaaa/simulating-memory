"""Evaluate one candidate harness by running bench in-process.

In-process is not optional: `inject.apply()` rebinds names inside the already
imported `bench.*` modules, so shelling out to `python -m bench.cli` would run
the ORIGINAL harness and silently produce baseline numbers for every candidate.
That failure would be invisible in the results, which is why the runner writes
the injection report next to the scores.

Usage:
    python meta_harness/run_candidate.py \
        --candidate meta_harness/candidates/baseline/harness.py \
        --config meta_harness/cluster/gate_word_recognition.yaml \
        --out-dir meta_harness/runs/<iteration>/<candidate_id> \
        -t wm_word_recognition
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="path to the candidate harness.py")
    ap.add_argument("--config", required=True, help="bench YAML config")
    ap.add_argument("--out-dir", required=True, help="where bench writes this candidate's run")
    ap.add_argument("-t", "--task", action="append", default=[],
                    help="task name, repeatable; empty means all in the config")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--story", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the offline interface check (not recommended)")
    args = ap.parse_args()

    # bench.cli first, so every task-module binding exists before injection.
    import bench.cli  # noqa: PLC0415

    from meta_harness.inject import apply, describe, load_candidate  # noqa: PLC0415

    cand_path = Path(args.candidate).resolve()

    if not args.skip_verify:
        from meta_harness.verify_interface import check  # noqa: PLC0415
        ok, lines = check(str(cand_path))
        if not ok:
            print(f"!!! interface check FAILED for {cand_path}")
            for line in lines:
                print(line)
            # Record the failure rather than dropping it silently.
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "compliance.json").write_text(json.dumps(
                {"candidate": str(cand_path), "passed": False, "detail": lines},
                indent=2))
            return 2
        print("=== interface check passed")

    cand = load_candidate(cand_path)
    report = apply(cand)
    print("=== injection report")
    print(describe(report))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(getattr(cand, "MANIFEST", {}))
    manifest.update({
        "candidate_path": str(cand_path),
        "injection": report,
        "config": str(Path(args.config).resolve()),
        "tasks": args.task,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "compliance.json").write_text(json.dumps(
        {"candidate": str(cand_path), "passed": True}, indent=2))
    # Keep the exact source that produced these numbers, so a later edit to the
    # candidate cannot silently reinterpret an old result.
    (out / "harness.py").write_text(cand_path.read_text())

    # Every parameter must be passed explicitly: run() is a typer command, so
    # omitted arguments would arrive as OptionInfo objects rather than defaults.
    t0 = time.time()
    bench.cli.run(
        config=args.config,
        task=args.task,
        model=None,
        out_dir=args.out_dir,
        repeat=args.repeat,
        debug=args.debug,
        story=args.story,
        extra_body_json=None,
        qwen_thinking=None,
    )
    elapsed = time.time() - t0

    manifest["elapsed_seconds"] = round(elapsed, 1)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"=== candidate run finished in {elapsed:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
