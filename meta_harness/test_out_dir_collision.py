"""A repeat must never be able to land on top of an earlier run.

The acceptance rule is three repeats per candidate, but until now the only thing keeping
repeat 2 off repeat 1 was the caller varying the path by hand -- an ITER value in one sbatch
script, an occurrence counter in another, a TAG in a third -- while the writer itself did
`mkdir(parents=True, exist_ok=True)` and wrote regardless. That discipline failed once: job
18719683's baseline arm overwrote the earlier Hermes baseline, and because runs/ is
gitignored, the evicting_reset held-out comparison recorded against it cannot be reproduced.

Run: python meta_harness/test_out_dir_collision.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_harness.run_candidate import (  # noqa: E402
    _git_provenance, _holds_a_run, _resolve_out_dir,
)


def _make_run(d: Path, rows: int = 3) -> None:
    (d / "model" / "tasks").mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"candidate_path": "x"}))
    (d / "model" / "tasks" / "wm_nback.jsonl").write_text(
        "\n".join(json.dumps({"i": i}) for i in range(rows)) + "\n")


def test_empty_dir_is_used_as_is() -> None:
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "iter9" / "cand"
        assert _resolve_out_dir(out, "fail") == out
        out.mkdir(parents=True)
        assert _resolve_out_dir(out, "fail") == out, "an empty dir is not a run"
        print("ok  a missing or empty directory is used as given")


def test_existing_run_fails_by_default() -> None:
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "cand"
        _make_run(out)
        assert _holds_a_run(out)
        try:
            _resolve_out_dir(out, "fail")
        except SystemExit as e:
            assert "already holds a run" in str(e), str(e)
            print("ok  an existing run is refused by default, with the two ways out named")
            return
        raise AssertionError("writing over an existing run was allowed")


def test_new_allocates_successive_repeats() -> None:
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "cand"
        _make_run(out)
        second = _resolve_out_dir(out, "new")
        assert second.name == "cand_rep2", second
        _make_run(second)
        third = _resolve_out_dir(out, "new")
        assert third.name == "cand_rep3", third
        # and the first two are untouched
        assert _holds_a_run(out) and _holds_a_run(second)
        print("ok  repeats allocate cand_rep2, cand_rep3 without touching the earlier runs")


def test_overwrite_is_explicit_only() -> None:
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "cand"
        _make_run(out)
        assert _resolve_out_dir(out, "overwrite") == out
        print("ok  overwriting requires asking for it by name")


def test_provenance_records_the_commit() -> None:
    p = _git_provenance()
    assert "git_head" in p, p
    assert p["git_head"] != "unknown", p
    assert len(p["git_head"]) == 40, p
    print(f"ok  provenance records the commit ({p['git_head'][:8]}, "
          f"dirty={p.get('git_dirty')})")


if __name__ == "__main__":
    test_empty_dir_is_used_as_is()
    test_existing_run_fails_by_default()
    test_new_allocates_successive_repeats()
    test_overwrite_is_explicit_only()
    test_provenance_records_the_commit()
    print("\nall passed")
