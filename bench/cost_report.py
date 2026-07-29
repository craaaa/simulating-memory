"""Aggregate LLM spend across every full-grid run under runs/prompting and
runs/compactor, broken down by provider (openrouter / gemini-direct /
openai-direct / anthropic-direct / local-vllm) and by model.

Reads each run's config_snapshot.json (for backend/base_url/model) paired
with its *_summary.json (for llm_usage). Local vLLM runs have no llm_usage
(free/GPU-hours, not billed) and are reported separately with $0.

Cost per run is usage["actual_cost_usd"] when present (OpenRouter's real
billed amount, from resp.usage.cost) else usage["estimated_cost_usd"]
(token-count x list-price, from bench.core.io.estimate_cost_usd — None for
models not in that table, never guessed).

Usage:
    python -m bench.cost_report
    python -m bench.cost_report --gemini-budget 1000
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(add_completion=False)


def _classify_provider(backend: Optional[str], base_url: Optional[str]) -> str:
    if backend and backend.lower() == "anthropic":
        return "anthropic-direct"
    bu = (base_url or "").lower()
    if "openrouter" in bu:
        return "openrouter"
    if "generativelanguage.googleapis" in bu:
        return "gemini-direct"
    if not bu:
        return "openai-direct"
    return f"other ({base_url})"


def _find_runs() -> list[dict]:
    """One entry per tasks/ dir containing both config_snapshot.json and a
    *_full_grid_summary.json (the two full-grid CLIs write both; ad-hoc
    scripts that skip config_snapshot.json are reported as unclassified)."""
    runs = []
    for root_name in ("prompting", "compactor"):
        root = REPO_ROOT / "runs" / root_name
        if not root.exists():
            continue
        for summary_path in sorted(root.glob("**/tasks/*_full_grid_summary.json")):
            tasks_dir = summary_path.parent
            config_path = tasks_dir / "config_snapshot.json"
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            config = {}
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            model = summary.get("model") or config.get("model") or "?"
            backend = config.get("backend")
            base_url = config.get("base_url")
            usage = summary.get("llm_usage") or {}
            runs.append(
                {
                    "run_type": root_name,
                    "path": str(summary_path.relative_to(REPO_ROOT)),
                    "model": model,
                    "provider": _classify_provider(backend, base_url),
                    "usage": usage,
                }
            )
    return runs


def _run_cost(usage: Dict[str, Any]) -> tuple[Optional[float], str]:
    if not usage:
        return None, "no usage recorded (pre-cost-tracking run, or local/free)"
    actual = usage.get("actual_cost_usd")
    if actual is not None:
        return float(actual), "actual"
    est = usage.get("estimated_cost_usd")
    if est is not None:
        return float(est), "estimated"
    return None, "unpriced (model not in pricing table)"


@app.command()
def report(gemini_budget: Optional[float] = typer.Option(1000.0, "--gemini-budget", help="Report remaining budget against this cap for gemini-direct spend.")) -> None:
    runs = _find_runs()

    by_provider: Dict[str, float] = defaultdict(float)
    by_provider_model: Dict[tuple, float] = defaultdict(float)
    unpriced: list[dict] = []
    total_actual = 0.0
    total_estimated = 0.0

    for r in runs:
        cost, source = _run_cost(r["usage"])
        if cost is None:
            unpriced.append(r)
            continue
        by_provider[r["provider"]] += cost
        by_provider_model[(r["provider"], r["model"])] += cost
        if source == "actual":
            total_actual += cost
        else:
            total_estimated += cost

    typer.echo("=== Spend by provider ===")
    for provider, cost in sorted(by_provider.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {provider:20s} ${cost:,.4f}")
    typer.echo(f"  {'TOTAL':20s} ${sum(by_provider.values()):,.4f}  (${total_actual:,.4f} actual + ${total_estimated:,.4f} estimated)")

    typer.echo("\n=== Spend by provider x model ===")
    for (provider, model), cost in sorted(by_provider_model.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {provider:16s} {model:45s} ${cost:,.4f}")

    if gemini_budget is not None:
        gemini_spend = by_provider.get("gemini-direct", 0.0)
        remaining = gemini_budget - gemini_spend
        typer.echo(f"\n=== Gemini budget ===")
        typer.echo(f"  spent:     ${gemini_spend:,.4f}")
        typer.echo(f"  budget:    ${gemini_budget:,.4f}")
        typer.echo(f"  remaining: ${remaining:,.4f}")

    if unpriced:
        typer.echo(f"\n=== {len(unpriced)} run(s) with no cost figure ===")
        for r in unpriced:
            _, reason = _run_cost(r["usage"])
            typer.echo(f"  {r['path']}  ({r['model']}, {reason})")


if __name__ == "__main__":
    app()
