"""STaR rationale bootstrapping for forward/reverse digit span.

    python -m rationales.cli select-data
    python -m rationales.cli dry-run
    python -m rationales.cli show-prompt --direction reverse --kind rationalize
    python -m rationales.cli star --rounds 1

Sampling and training both run through Tinker (see rationales/tinker_client.py).
There is no OpenRouter path: no Qwen3-8B / Qwen3.5-4B is served there, and sampling a
different model than the one being trained would make the corpus off-policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from bench.core.io import run_timestamp, write_json

from . import prompting
from .config import StarConfig
from .data import by_length, load_all
from .select import select
from .star import dry_run as plan_dry_run
from .star import prepare_run, run_star

app = typer.Typer(add_completion=False, help=__doc__)


def _cfg(**overrides) -> StarConfig:
    clean = {k: v for k, v in overrides.items() if v is not None}
    return StarConfig(**clean)


@app.command("select-data")
def select_data(
    seed: int = typer.Option(42, help="RNG seed for the 1:1 subsample and the split."),
    eval_frac: float = typer.Option(0.15, help="Held-out fraction, stratified."),
    out: Optional[Path] = typer.Option(None, help="Write selection.json here."),
) -> None:
    """Build D: all fail trials + an equal number of successes, then split."""
    cfg = _cfg(seed=seed, eval_frac=eval_frac)
    trials, reports = load_all(cfg.directions)
    for r in reports:
        typer.echo(json.dumps(r))

    sel = select(trials, seed=cfg.seed, eval_frac=cfg.eval_frac)
    rep = sel.report
    typer.echo("")
    typer.echo(f"pool: {rep['pool_total']} trials ({rep['pool_fail']} fail / {rep['pool_success']} success)")
    typer.echo(f"train: {rep['n_train']} ({rep['train_fail']} fail)   eval: {rep['n_eval']} ({rep['eval_fail']} fail)")
    if rep["stimulus_overlap_train_eval"]:
        typer.echo(
            f"dropped {len(rep['eval_dropped_for_stimulus_overlap'])} eval trials "
            f"whose digit sequence also appears in train"
        )
    typer.echo("")
    typer.echo("fail-by-length in train (global 1:1 concentrates fails at long spans --")
    typer.echo("that shortcut is what the per-length eval curve exists to detect):")
    for key, cell in rep["train_by_length"].items():
        typer.echo(f"  {key:>12}  n={cell['n']:>3}  fail={cell['n_fail']:>3}")

    if out:
        write_json(out, rep)
        typer.echo(f"\nwrote {out}")


@app.command("show-prompt")
def show_prompt(
    direction: str = typer.Option("forward", help="forward | reverse"),
    kind: str = typer.Option("sample", help="sample | rationalize | training"),
    fewshot: bool = typer.Option(True, help="Include the few-shot rationale demos."),
) -> None:
    """Render one fully-assembled prompt for eyeball review."""
    trials, _ = load_all([direction])
    trial = next(t for t in trials if not t.correct)
    if kind == "rationalize":
        text = prompting.build_rationalize_prompt(trial, fewshot=fewshot)
    elif kind == "training":
        text = prompting.build_sample_prompt(trial, fewshot=False)
    else:
        text = prompting.build_sample_prompt(trial, fewshot=fewshot)
    typer.echo(f"--- {direction} / {kind} / trial {trial.trial_id} ---")
    typer.echo(text)


@app.command("dry-run")
def dry_run_cmd(
    base_model: str = typer.Option("Qwen/Qwen3-8B"),
    rounds: int = typer.Option(1),
    k: int = typer.Option(1, help="Samples per problem. STaR decodes greedily with 1."),
    temperature: float = typer.Option(0.0),
    fewshot: bool = typer.Option(True),
) -> None:
    """Plan a round: request counts, token estimate, cost estimate. No API calls."""
    cfg = _cfg(
        base_model=base_model, rounds=rounds, k=k, temperature=temperature, use_fewshot=fewshot
    )
    plan = plan_dry_run(cfg)
    for key in [
        "base_model",
        "n_train_trials",
        "n_fail_trials",
        "n_eval_trials",
        "n_generations",
        "train_steps",
        "estimated_prefill_tokens",
        "estimated_sample_tokens",
        "estimated_train_tokens",
        "estimated_cost_usd",
    ]:
        typer.echo(f"{key}: {plan[key]}")
    typer.echo(f"basis: {plan['estimate_basis']}")
    cost = plan["estimated_cost_usd"]
    if cost is None:
        typer.echo("\nNO PRICE ON FILE for this model -- check the live models.json before running.")
    elif cost > 5:
        typer.echo(f"\nESTIMATE EXCEEDS $5 (${cost:.2f}) -- confirm before running for real.")


@app.command("star")
def star_cmd(
    base_model: str = typer.Option("Qwen/Qwen3-8B"),
    rounds: int = typer.Option(1, help="Outer loops. Round 1 first; inspect, then extend."),
    k: int = typer.Option(1),
    temperature: float = typer.Option(0.0),
    lora_rank: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-4),
    steps_1: int = typer.Option(60),
    batch_size: int = typer.Option(8),
    seed: int = typer.Option(42),
    fewshot: bool = typer.Option(True),
    leak_filter: bool = typer.Option(True),
    checkpoint_every: int = typer.Option(20, help="save_state every N training steps."),
    max_usd: Optional[float] = typer.Option(
        None, help="Spend ceiling; the run PAUSES when live cost crosses it (resumable)."
    ),
    resume: bool = typer.Option(False, "--resume", help="Continue a paused run."),
    run_dir: Optional[Path] = typer.Option(
        None, help="Run directory to resume. Defaults to the most recent paused run."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation."),
) -> None:
    """Run the STaR loop end to end (sample -> filter -> train -> eval)."""
    cfg = _cfg(
        base_model=base_model,
        rounds=rounds,
        k=k,
        temperature=temperature,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        steps_1=steps_1,
        batch_size=batch_size,
        seed=seed,
        use_fewshot=fewshot,
        leak_filter=leak_filter,
        checkpoint_every=checkpoint_every,
    )

    if resume and run_dir is None:
        from .config import PKG_DIR

        paused = sorted(PKG_DIR.glob("out/*/*/progress.json"))
        if not paused:
            raise typer.BadParameter("no run to resume; drop --resume to start one")
        run_dir = paused[-1].parent
    if resume:
        state = json.loads((run_dir / "progress.json").read_text())
        typer.echo(
            f"resuming {run_dir.name}: round {state['round']}, phase {state['phase']}"
            + (f", train step {state['train_step']}" if state["train_step"] else "")
        )
        if state.get("spend_usd_at_pause"):
            typer.echo(f"spend before pause: ${state['spend_usd_at_pause']:.4f}")
    else:
        plan = plan_dry_run(cfg)
        cost = plan["estimated_cost_usd"]
        typer.echo(f"estimated cost for round 1: {cost if cost is None else f'${cost:.2f}'}")
        if not yes:
            typer.confirm("Proceed with paid Tinker calls?", abort=True)

    summary = run_star(cfg, max_usd=max_usd, resume=resume, run_dir=run_dir)
    if summary.get("paused"):
        typer.echo(f"\nPAUSED: {summary['reason']}")
        typer.echo(f"resume: {summary['resume_with']} --max-usd <higher>")
        raise typer.Exit(code=2)
    typer.echo(json.dumps(summary["bootstrap_signal"], indent=2))
    typer.echo(f"spend: {summary['usage']}")
    typer.echo(f"out: {summary['out_dir']}")


@app.command("probe")
def probe_cmd(
    model: str = typer.Option(
        "qwen/qwen3.8-flash", help="OpenRouter route id, e.g. qwen/qwen3.8-flash."
    ),
    n_per_cell: int = typer.Option(2, help="Trials per (direction x success/fail) cell."),
    fewshot: bool = typer.Option(True),
    temperature: float = typer.Option(0.0),
    max_tokens: int = typer.Option(1024, help="Completion budget per call."),
    thinking: bool = typer.Option(
        False,
        help="Allow provider-side thinking. Off by default: hidden reasoning tokens "
        "eat the completion budget and can return an empty completion.",
    ),
    seed: int = typer.Option(42),
) -> None:
    """Check that the sampling prompts elicit well-formed rationales (OpenRouter).

    Format check only -- off-policy, cents of spend, never written to a corpus. Needs
    OPENROUTER_API_KEY. Use it before paying for a real Tinker round.
    """
    import os

    from .probe import probe, write_report

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        raise typer.BadParameter(
            "OPENROUTER_API_KEY is not set. Export it, or run the command inline: "
            "OPENROUTER_API_KEY=... python -m rationales.cli probe"
        )

    n_calls = 2 * 4 * n_per_cell
    typer.echo(f"{n_calls} calls to {model} via OpenRouter (format check, ~cents)")

    report = probe(
        model=model,
        n_per_cell=n_per_cell,
        fewshot=fewshot,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        seed=seed,
    )
    s = report["summary"]
    for key in [
        "empty_response_rate",
        "well_formed_rate",
        "well_formed_rate_sample",
        "well_formed_rate_rationalize",
        "right_length_rate",
        "median_reasoning_chars",
        "sample_human_match_success_trials",
        "sample_human_match_fail_trials",
        "sample_ground_truth_rate",
        "rationalize_hit_rate",
        "rationalize_hit_rate_fail_trials",
        "rationalize_hint_leak_rate",
    ]:
        typer.echo(f"{key}: {s[key]}")
    if s["parse_error_counts"]:
        typer.echo(f"parse errors: {s['parse_error_counts']}")
    typer.echo(f"\n{s['reading']}")

    path = write_report(report)
    typer.echo(f"\nwrote {path}")
    typer.echo(report["caveat"])


@app.command("cost")
def cost_cmd(
    run_dir: Optional[Path] = typer.Option(
        None, help="Run directory. Defaults to the most recent one under out/."
    ),
    watch: bool = typer.Option(False, "--watch", help="Refresh until the run finishes."),
    interval: float = typer.Option(5.0, help="Seconds between refreshes with --watch."),
) -> None:
    """Read the live cost ledger a running (or finished) round is writing.

    Tinker has no live spend endpoint -- `tinker billing usage` lags real time by hours
    and reports tokens rather than USD -- so this prices the calls client-side from the
    tokens actually sent and returned.
    """
    import json
    import time

    from .config import PKG_DIR, TINKER_PRICING_VERIFIED_ON

    if run_dir is None:
        candidates = sorted(PKG_DIR.glob("out/*/*/cost_ledger.jsonl"))
        if not candidates:
            raise typer.BadParameter(f"no cost ledger found under {PKG_DIR / 'out'}")
        run_dir = candidates[-1].parent
    ledger = run_dir / "cost_ledger.jsonl"
    if not ledger.is_file():
        raise typer.BadParameter(f"no cost ledger at {ledger}")

    def render() -> None:
        rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
        if not rows:
            typer.echo("ledger is empty")
            return
        cum = rows[-1]["cumulative"]
        by_kind: dict = {}
        for r in rows:
            k = by_kind.setdefault(r["kind"], {"calls": 0, "prefill": 0, "sample": 0, "train": 0})
            k["calls"] += 1
            k["prefill"] += r["prefill_tokens"]
            k["sample"] += r["sample_tokens"]
            k["train"] += r["train_tokens"]

        typer.echo(f"{run_dir.name}  ({len(rows)} calls)")
        for kind, k in sorted(by_kind.items()):
            typer.echo(
                f"  {kind:<8} calls={k['calls']:<5} prefill={k['prefill']:>9,} "
                f"sample={k['sample']:>8,} train={k['train']:>9,}"
            )
        cost = cum["cost_usd"]
        typer.echo(
            f"  TOTAL    prefill={cum['prefill_tokens']:>9,} "
            f"sample={cum['sample_tokens']:>8,} train={cum['train_tokens']:>9,}"
        )
        typer.echo(f"  spend: {'unpriced model' if cost is None else f'${cost:0.4f}'}")

    if not watch:
        render()
        typer.echo(
            f"\nList prices verified {TINKER_PRICING_VERIFIED_ON}. Reconcile after the "
            "run with: tinker billing usage <start> <end> --sessions-csv -"
        )
        return

    done = run_dir / "summary.json"
    while True:
        typer.echo("\033[2J\033[H", nl=False)
        render()
        if done.is_file():
            typer.echo("\nrun finished (summary.json written)")
            return
        time.sleep(interval)


@app.command("prepare")
def prepare(
    base_model: str = typer.Option("Qwen/Qwen3-8B"),
    seed: int = typer.Option(42),
) -> None:
    """Create the run dir with selection.json + provenance, without calling the API."""
    cfg = _cfg(base_model=base_model, seed=seed)
    root, sel = prepare_run(cfg, timestamp=run_timestamp())
    typer.echo(f"{root}  train={len(sel.train)} eval={len(sel.eval)}")


if __name__ == "__main__":
    app()
