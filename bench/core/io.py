from __future__ import annotations
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


class JsonlSink:
    """Thread-safe incremental JSONL writer. Truncates the file on construction."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def append(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_timestamp() -> str:
    """UTC timestamp for stamping run output directories, e.g. 20260714T202413Z.

    Respects RUN_TIMESTAMP env var so callers can pin the stamp across multiple
    CLIs in the same logical run.
    """
    return os.getenv("RUN_TIMESTAMP", "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# Files whose diffs are captured on every run so prompt changes are recoverable
# even from a dirty working tree (no commit required).
_PROMPT_FILES = [
    "bench/tasks/wm_prompt_parts.py",
    "bench/tasks/wm_mcq_common.py",
    "application/listening_qa/prompting.py",
    "application/reading_qa/prompting.py",
]


def _run_git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def git_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return git commit hash, dirty status, and diff of prompt files.

    Captured at run start so the exact prompts used are always recoverable,
    even without committing before each experiment.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    tags = _run_git(["tag", "--points-at", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    is_dirty = bool(status)

    diffs: Dict[str, Optional[str]] = {}
    for rel_path in _PROMPT_FILES:
        diff = _run_git(["diff", "HEAD", "--", rel_path], root)
        diffs[rel_path] = diff if diff else None

    return {
        "commit": commit,
        "branch": branch,
        "tags": [t for t in (tags or "").splitlines() if t],
        "dirty": is_dirty,
        "prompt_diffs": diffs,
    }
