"""The STaR outer loop.

Per round: generate -> rationalize -> filter -> train (from base) -> eval.
Round N+1 samples from round N's sampler checkpoint; it does NOT continue training
from round N's adapter, and it does NOT reuse round N's corpus.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bench.core.io import ensure_dir, git_provenance, run_timestamp, write_json

from . import evaluate as ev
from . import prompting
from .config import StarConfig, out_root
from .data import load_all
from .filter import build_corpus
from .sample import sample_round, training_pairs, SampleRow, GENERATION, RATIONALIZATION
from .select import Selection, restore, select
from .resume import RunState
from .tinker_client import BudgetExceeded, CostTracker, TinkerSession
from .train import estimate_train_tokens, read_checkpoint, train_round


def prepare_run(cfg: StarConfig, *, timestamp: Optional[str] = None) -> tuple[Path, Selection]:
    """Create the run dir, load + select D once, and stamp provenance."""
    ts = timestamp or run_timestamp()
    root = ensure_dir(out_root(cfg, ts))

    trials, reports = load_all(cfg.directions)
    sel_path = root / "selection.json"
    if sel_path.is_file():
        import json

        sel = restore(trials, json.loads(sel_path.read_text(encoding="utf-8")))
    else:
        sel = select(trials, seed=cfg.seed, eval_frac=cfg.eval_frac)
        write_json(sel_path, sel.report)

    write_json(
        root / "run_config.json",
        {
            "config": cfg.to_dict(),
            "git_provenance": git_provenance(),
            "timestamp": ts,
            "human_data_reports": reports,
            "prompt_additions": prompting.prompt_additions(),
            "sampler_backend": "tinker",
            "star_reference": "Zelikman et al. 2022, arXiv:2203.14465, Algorithm 1",
        },
    )
    return root, sel


def _reload_pool(path: Path, trials_by_id: Dict[str, Any]) -> List[SampleRow]:
    """Rebuild SampleRow objects from a written sample_pool.jsonl."""
    import json

    rows: List[SampleRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        trial = trials_by_id.get(d["trial_id"])
        if trial is None:
            continue
        rows.append(
            SampleRow(
                trial=trial,
                via=d["via"],
                sample_index=d["sample_index"],
                prompt=d["prompt"],
                raw=d["raw"],
                reasoning=d.get("reasoning"),
                digits=d.get("pred_digits") or [],
                accepted=bool(d.get("accepted")),
                parse_errors=d.get("parse_errors") or [],
            )
        )
    return rows


def run_round(
    cfg: StarConfig,
    session: TinkerSession,
    root: Path,
    sel: Selection,
    *,
    round_n: int,
    sampler_path: Optional[str],
    state: Optional[RunState] = None,
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    round_dir = ensure_dir(root / "rounds" / f"round_{round_n}")
    state = state or RunState(round=round_n)

    def checkpoint_state(phase: str) -> None:
        state.advance(phase)
        if state_path:
            state.save(state_path)

    # --- lines 3-4: generate, then rationalize failures --------------------
    client = session.sampling_client(sampler_path)

    def generate(prompt: str, n: int) -> List[str]:
        return session.sample(
            client,
            prompt,
            num_samples=n,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    pool_path = round_dir / "sample_pool.jsonl"
    if not state.is_done_with("sample"):
        # resume=True keeps rationales a paused run already paid for.
        sample_report = sample_round(
            cfg, sel.train, generate=generate, pool_path=pool_path,
            resume=pool_path.is_file(),
        )
        write_json(round_dir / "sample_report.json", sample_report)
        checkpoint_state("filter")
    else:
        sample_report = _read_json(round_dir / "sample_report.json")

    # --- lines 5-6: D_n u D^rat_n -----------------------------------------
    trials_by_id = {t.trial_id: t for t in sel.train}
    rows = _reload_pool(pool_path, trials_by_id)
    pairs = training_pairs(rows, cfg)
    corpus, filter_stats = build_corpus(pairs, cfg, out_dir=round_dir)
    checkpoint_state("train")

    # --- line 7: train from the ORIGINAL base model ------------------------
    if state.checkpoint:
        train_log = _read_json(round_dir / "train_log.json")
        checkpoint = state.checkpoint
    else:
        def on_checkpoint(step: int, path: str) -> None:
            state.train_step = step
            state.train_state_path = path
            if state_path:
                state.save(state_path)

        train_log = train_round(
            cfg, session, corpus, round_n=round_n, out_dir=round_dir,
            start_step=state.train_step,
            resume_state_path=state.train_state_path,
            on_checkpoint=on_checkpoint,
        )
        checkpoint = train_log["checkpoint"]
        state.checkpoint = checkpoint
    checkpoint_state("eval_base")

    # --- eval: tuned vs base on the held-out split -------------------------
    tuned_client = session.sampling_client(checkpoint)
    base_client = session.sampling_client(None)

    def _gen(client_obj):
        def inner(prompt: str, n: int) -> List[str]:
            return session.sample(
                client_obj, prompt, num_samples=n, temperature=0.0, max_tokens=cfg.max_tokens
            )

        return inner

    # Rows are appended as they are scored, so a pause mid-eval resumes instead of
    # re-scoring what has already been paid for.
    base_eval = ev.run_eval(
        cfg, sel.eval, generate=_gen(base_client), label="base",
        rows_path=round_dir / "eval_rows_base.jsonl",
    )
    checkpoint_state("eval_tuned")
    tuned_eval = ev.run_eval(
        cfg, sel.eval, generate=_gen(tuned_client), label=f"round{round_n}",
        rows_path=round_dir / "eval_rows_tuned.jsonl",
    )

    payload = {
        "round": round_n,
        "sampled_from": sampler_path or f"{cfg.base_model} (base)",
        "checkpoint": checkpoint,
        "sample_report": sample_report,
        "filter_stats": filter_stats,
        "train_log": {k: v for k, v in train_log.items() if k != "steps"},
        "fail_side_generation_yield": sample_report["stats"]["fail_side_generation_yield"],
        "base": base_eval["metrics"],
        "tuned": tuned_eval["metrics"],
        "comparison": ev.compare(base_eval, tuned_eval),
        "usage": session.tracker.summary(),
    }
    try:
        from . import plotting

        payload["figures"] = [
            str(p) for p in plotting.plot_round(payload, ensure_dir(root / "figures"), round_n=round_n)
        ]
    except Exception as exc:  # plotting must never lose a completed round
        payload["figures_error"] = repr(exc)

    # Written after plotting so eval.json actually records the figure paths -- the
    # first live round wrote it first and shipped an empty figures list.
    write_json(round_dir / "eval.json", payload)
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    import json

    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_star(
    cfg: StarConfig,
    *,
    timestamp: Optional[str] = None,
    max_usd: Optional[float] = None,
    resume: bool = False,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ts = timestamp or (Path(run_dir).name if run_dir else run_timestamp())
    root, sel = prepare_run(cfg, timestamp=ts)

    # Live cost ledger: one row per API call, with the running USD total. Tail it from
    # another terminal while the round runs -- Tinker has no live spend endpoint.
    tracker = CostTracker(
        cfg.base_model,
        ledger_path=root / "cost_ledger.jsonl",
        max_usd=max_usd,
        show_progress=True,
        run_label=ts,
    )
    session = TinkerSession(
        cfg,
        tracker=tracker,
        # Lets `tinker billing usage --sessions-csv` be joined back to this run.
        user_metadata={"pipeline": "rationales-star", "run": ts, "base_model": cfg.base_model},
    )

    # Build the tokenizer before any thread pool starts: concurrent first-use across
    # workers means concurrent HuggingFace downloads, and a rate-limit there kills the
    # run mid-sampling.
    session.warm_up()

    state_path = root / "progress.json"
    state = RunState.load(state_path) if resume else RunState()
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"nothing to resume: {state_path} does not exist")
    state.paused = False
    state.pause_reason = None
    state.save(state_path)

    per_round: List[Dict[str, Any]] = []
    start_round = state.round if resume else 1
    sampler_path = state.sampler_path if resume else None

    for n in range(start_round, cfg.rounds + 1):
        # A round already finished in an earlier invocation is loaded from disk, not
        # re-run: extending --rounds must not re-pay for completed rounds.
        if n in state.completed_rounds:
            prior = _read_json(root / "rounds" / f"round_{n}" / "eval.json")
            if prior:
                per_round.append(prior)
                sampler_path = prior.get("checkpoint") or sampler_path
            print(f"[round {n}] already complete, skipping")
            continue

        if n != state.round or not resume:
            state.start_round(n)
            state.save(state_path)
        try:
            report = run_round(
                cfg, session, root, sel,
                round_n=n, sampler_path=sampler_path, state=state, state_path=state_path,
            )
        except BudgetExceeded as exc:
            # A pause, not a crash: every phase has already flushed its work, and
            # progress.json says exactly where to pick up.
            state.paused = True
            state.pause_reason = str(exc)
            state.spend_usd_at_pause = tracker.cost_usd
            state.save(state_path)
            paused = {
                "paused": True,
                "reason": str(exc),
                "resume_with": f"python -m rationales.cli star --resume --run-dir {root}",
                "state": asdict(state),
                "usage": tracker.summary(),
                "out_dir": str(root),
                "rounds": per_round,
            }
            write_json(root / "paused.json", paused)
            print(f"\nPAUSED: {exc}\nresume: {paused['resume_with']} --max-usd <higher>")
            return paused

        per_round.append(report)
        state.finish_round()
        state.save(state_path)
        # Sample the next round from this round's adapter; train() still starts from
        # the base model.
        sampler_path = report["checkpoint"]

    (root / "paused.json").unlink(missing_ok=True)
    summary = {
        "rounds": per_round,
        "usage": session.tracker.summary(),
        "out_dir": str(root),
        "bootstrap_signal": [
            {"round": r["round"], "fail_side_generation_yield": r["fail_side_generation_yield"]}
            for r in per_round
        ],
    }
    try:
        from . import plotting

        summary["bootstrap_figure"] = str(
            plotting.plot_bootstrap_signal(summary, ensure_dir(root / "figures"))
        )
    except Exception as exc:
        summary["figures_error"] = repr(exc)

    write_json(root / "summary.json", summary)
    return summary


def dry_run(cfg: StarConfig, *, round_n: int = 1) -> Dict[str, Any]:
    """Plan a round without touching the API: request counts, token and cost estimate.

    Token counts here come from a crude ~4-chars-per-token heuristic (no tokenizer
    available offline), so they are labelled as estimates everywhere they appear.
    """
    from .config import COMPLETION_TOKENS, SUCCESS_MISS_RATE, tinker_cost_usd
    from .tinker_client import offline_token_estimate

    # Prefer the model's real tokenizer (local, no API call, no spend) over the
    # chars/token heuristic -- and render through the chat template, since that is what
    # actually gets billed as prefill.
    est = offline_token_estimate
    basis = "chars/token heuristic"
    try:
        _session = TinkerSession(cfg)
        _tok = _session.tokenizer
        est = lambda text: len(_session.render_prompt_ids(text))  # noqa: E731
        basis = f"exact tokens from {type(_tok).__name__} via the chat template"
    except Exception:
        pass

    trials, _ = load_all(cfg.directions)
    sel = select(trials, seed=cfg.seed, eval_frac=cfg.eval_frac)

    gen_prompts = [prompting.build_sample_prompt(t, fewshot=cfg.use_fewshot) for t in sel.train]
    fails = [t for t in sel.train if not t.correct]
    successes = [t for t in sel.train if t.correct]

    # Every fail trial is assumed to need rationalization, plus the share of success
    # trials the model misses unhinted -- probing showed that is not zero (the few-shot
    # demos induce errors, which is the point, but it costs a second call).
    n_success_missed = int(round(len(successes) * SUCCESS_MISS_RATE))
    rat_trials = fails + successes[:n_success_missed]
    rat_prompts = [
        prompting.build_rationalize_prompt(t, fewshot=cfg.use_fewshot) for t in rat_trials
    ]
    eval_prompts = [prompting.build_sample_prompt(t, fewshot=False) for t in sel.eval]

    prefill = cfg.k * sum(est(p) for p in gen_prompts)
    prefill += cfg.k * sum(est(p) for p in rat_prompts)
    prefill += 2 * sum(est(p) for p in eval_prompts)  # base + tuned

    n_gens = cfg.k * (len(gen_prompts) + len(rat_prompts)) + 2 * len(eval_prompts)
    sample_tokens = n_gens * COMPLETION_TOKENS

    # Training sees the zero-shot prompt plus a completion at the brevity cap.
    approx_corpus = [
        {
            "prompt": prompting.build_sample_prompt(t, fewshot=False),
            "completion": "x" * int(COMPLETION_TOKENS * 3.72),
        }
        for t in sel.train
    ]
    train_tokens = estimate_train_tokens(approx_corpus, cfg, round_n, est)

    cost = tinker_cost_usd(
        cfg.base_model,
        prefill_tokens=prefill,
        sample_tokens=sample_tokens,
        train_tokens=train_tokens,
    )
    return {
        "base_model": cfg.base_model,
        "n_train_trials": len(sel.train),
        "n_fail_trials": len(fails),
        "n_eval_trials": len(sel.eval),
        "n_generations": n_gens,
        "train_steps": cfg.steps_for_round(round_n),
        "estimated_prefill_tokens": prefill,
        "estimated_sample_tokens": sample_tokens,
        "estimated_train_tokens": train_tokens,
        "estimated_cost_usd": cost,
        "n_rationalize_calls": len(rat_prompts),
        "estimate_basis": (
            f"prompt tokens: {basis}; {COMPLETION_TOKENS} completion tokens/generation "
            f"(observed 68-168 live); rationalization assumed for every fail trial plus "
            f"{SUCCESS_MISS_RATE:.0%} of success trials; Tinker list prices verified "
            f"2026-09-10"
        ),
        "example_generation_prompt": gen_prompts[0] if gen_prompts else None,
        "example_rationalize_prompt": rat_prompts[0] if rat_prompts else None,
        "example_training_prompt": (
            prompting.build_sample_prompt(sel.train[0], fewshot=False) if sel.train else None
        ),
    }
