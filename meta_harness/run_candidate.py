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


def _holds_a_run(out: Path) -> bool:
    """Does this directory already hold run output?"""
    if not out.exists():
        return False
    if (out / "manifest.json").exists():
        return True
    return any(out.rglob("tasks/*.jsonl"))


def _resolve_out_dir(out: Path, policy: str) -> Path:
    """Decide where to write, so a repeat can never land on top of an earlier run.

    The acceptance rule is three repeats per candidate, and until now the only thing
    stopping repeat 2 from overwriting repeat 1 was the caller remembering to vary the
    path -- an ITER value in one sbatch script, an occurrence counter in another, a TAG in
    a third. That discipline failed once already: job 18719683's baseline arm overwrote the
    earlier Hermes baseline, and because runs/ is gitignored the comparison recorded against
    it can no longer be reproduced. The writer is the only place that can enforce this, so
    it enforces it here.

    fail      (default) refuse to touch a directory that already holds a run
    new       write to the next free NAME_rep<N> beside it
    overwrite proceed anyway, for a deliberate re-run
    """
    if policy == "overwrite" or not _holds_a_run(out):
        return out
    if policy == "fail":
        raise SystemExit(
            f"!!! {out} already holds a run.\n"
            f"    Pass --if-exists new to write the next free repeat directory, or\n"
            f"    --if-exists overwrite to replace it deliberately.")
    n = 2
    while True:
        cand = out.parent / f"{out.name}_rep{n}"
        if not _holds_a_run(cand):
            print(f"=== {out} already holds a run; writing repeat {n} -> {cand}")
            return cand
        n += 1


def _git_provenance() -> dict[str, str | bool]:
    """The commit that produced a run, so two generations of runs stay distinguishable.

    Without this, a pre-fix and a post-fix run are told apart only by their directory name,
    which is exactly the thing that has already gone wrong once.
    """
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return {"git_head": "unknown"}
        return {"git_head": head.stdout.strip(),
                "git_dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_head": "unknown"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="path to the candidate harness.py")
    ap.add_argument("--config", required=True, help="bench YAML config")
    ap.add_argument("--out-dir", required=True, help="where bench writes this candidate's run")
    ap.add_argument("-t", "--task", action="append", default=[],
                    help="task name, repeatable; REQUIRED (see --all-tasks)")
    ap.add_argument("--all-tasks", action="store_true",
                    help="deliberately run every registered task (31 of them, "
                         "including the non-compactor prompting and sum_ variants)")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--story", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the offline interface check (not recommended)")
    ap.add_argument("--if-exists", choices=("fail", "new", "overwrite"), default="fail",
                    help="what to do when --out-dir already holds a run: fail (default), "
                         "write the next free NAME_rep<N> (new), or replace it (overwrite)")
    args = ap.parse_args()

    # An empty task list makes bench run ALL 31 registered tasks, including the
    # prompting and sum_ variants that are not the compactor at all.  Naming the
    # tasks in the YAML's task_config does NOT restrict what runs.  This cost a
    # cancelled 17-minute GPU job before the guard existed.
    if not args.task and not args.all_tasks:
        print("refusing to run: no -t/--task given, which would run all 31 "
              "registered tasks (prompting and sum_ variants included).\n"
              "Name the tasks explicitly, or pass --all-tasks if that is really "
              "what you want.")
        return 2

    # Settle the output directory BEFORE loading bench or a 70B server's worth of work:
    # a refusal here should cost a second, not a job.
    out = _resolve_out_dir(Path(args.out_dir), args.if_exists)
    args.out_dir = str(out)

    # bench.cli first, so every task-module binding exists before injection.
    import bench.cli  # noqa: PLC0415

    from meta_harness.inject import apply, describe, load_candidate  # noqa: PLC0415

    cand_path = Path(args.candidate).resolve()

    if not args.skip_verify:
        # In a SUBPROCESS, not in-process. check() calls inject.apply() internally,
        # and apply() is not idempotent: the second load subclasses the
        # already-injected class, so both step() overrides run and every prompt delta
        # is emitted twice. That is exactly what happened -- 2400 of 2400 non-first
        # n-back turns in episodic_reset_v3, evicting_reset and both held-out arms
        # carried the control-state block twice. Verifying out-of-process keeps the
        # run process clean, and inject.apply() now raises on a second apply so this
        # cannot recur silently.
        import subprocess  # noqa: PLC0415
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "verify_interface.py"),
             str(cand_path)],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        ok = proc.returncode == 0
        lines = (proc.stdout + proc.stderr).splitlines()
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

    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(getattr(cand, "MANIFEST", {}))
    manifest.update({
        "candidate_path": str(cand_path),
        "injection": report,
        "config": str(Path(args.config).resolve()),
        "tasks": args.task,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_git_provenance(),
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
