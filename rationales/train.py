"""STaR Algorithm 1 line 7: M_n <- train(M, D_n u D^rat_n).

Two things the paper is explicit about, both honored here:
  * the client is created fresh from the BASE model every round, not resumed from
    M_{n-1} -- "we train from the original pre-trained model M instead of continually
    training one model to avoid overfitting";
  * the dataset is that round's D_n u D^rat_n only, never an accumulation across
    rounds. The full train split is regenerated each round instead.

Schedule follows the paper: LR warmup then constant LR, a fixed step budget per round
growing 20% per outer loop (cfg.steps_for_round).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from bench.core.io import write_json

from .config import StarConfig
from .tinker_client import TinkerSession


def _batches(rows: Sequence[Dict[str, Any]], cfg: StarConfig, n_steps: int, seed: int):
    """Cycle the corpus in shuffled epochs until the step budget is spent."""
    rng = random.Random(seed)
    order: List[int] = []
    for _ in range(n_steps):
        batch: List[Dict[str, Any]] = []
        while len(batch) < cfg.batch_size:
            if not order:
                order = list(range(len(rows)))
                rng.shuffle(order)
            batch.append(rows[order.pop()])
        yield batch


def _lr_at(step: int, cfg: StarConfig) -> float:
    """100-step linear warmup then constant, as in the paper's setup."""
    if cfg.warmup_steps <= 0:
        return cfg.learning_rate
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    return cfg.learning_rate


def train_round(
    cfg: StarConfig,
    session: TinkerSession,
    corpus: Sequence[Dict[str, Any]],
    *,
    round_n: int,
    out_dir: Path,
    start_step: int = 0,
    resume_state_path: Optional[str] = None,
    on_checkpoint: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Algorithm 1 line 7, with mid-training checkpoints.

    ``save_state`` (weights + optimizer) is written every ``cfg.checkpoint_every``
    steps. On resume, ``resume_state_path`` restores optimizer momentum and
    ``start_step`` skips the steps already applied -- restarting from step 0 would both
    re-pay for them and change the schedule the model actually saw.
    """
    if not corpus:
        raise RuntimeError(
            "empty corpus: no rationale was accepted this round, so there is nothing "
            "to train on. Inspect sample_pool.jsonl before rerunning."
        )

    import tinker  # noqa: PLC0415  (only reached on a real run)

    n_steps = cfg.steps_for_round(round_n)
    if resume_state_path:
        training_client = session.training_client_from_state(resume_state_path)
        print(f"[round {round_n}] resumed from {resume_state_path} at step {start_step}")
    else:
        training_client = session.training_client()

    log: List[Dict[str, Any]] = []
    train_tokens = 0
    n_truncated = 0

    for step, batch in enumerate(_batches(corpus, cfg, n_steps, cfg.seed + round_n)):
        if step < start_step:
            continue   # already applied before the pause
        datums = []
        batch_meta = []
        for row in batch:
            datum, meta = session.build_datum(row["prompt"], row["completion"])
            datums.append(datum)
            batch_meta.append(meta)
            train_tokens += meta["n_tokens"]
            if meta["truncated"]:
                n_truncated += 1
                if n_truncated == 1:
                    print(
                        f"WARNING: example exceeds max_seq_length={cfg.max_seq_length} "
                        f"by {meta['dropped_tokens']} tokens; truncation cuts the "
                        "completion, not the prompt. Raise max_seq_length."
                    )

        lr = _lr_at(step, cfg)
        fwd = training_client.forward_backward(datums, loss_fn="cross_entropy")
        opt = training_client.optim_step(tinker.AdamParams(learning_rate=lr))
        fwd_result = fwd.result()
        opt.result()

        # Priced per step so spend is live, not only at the end of the round.
        batch_tokens = sum(m["n_tokens"] for m in batch_meta)
        session.tracker.record("train", train=batch_tokens, round=round_n, step=step)

        # ForwardBackwardOutput carries (loss_fn_output_type, loss_fn_outputs, metrics);
        # there is no .loss attribute, so reading one silently logged None for every
        # step of the first live round and left no loss curve to set steps_1 from.
        metrics = dict(getattr(fwd_result, "metrics", None) or {})
        loss = next(
            (metrics[k] for k in ("loss", "train_loss", "loss:sum", "loss:mean") if k in metrics),
            None,
        )
        if loss is None:
            loss = getattr(fwd_result, "loss", None)
        log.append(
            {"step": step, "lr": lr, "loss": loss, "metrics": metrics, "batch": len(datums)}
        )
        if step % 10 == 0:
            print(f"[round {round_n}] step {step}/{n_steps} lr={lr:.2e} loss={loss}")

        # Periodic full state (weights + optimizer) so a pause costs at most
        # checkpoint_every steps of re-training rather than the whole round.
        is_last = step == n_steps - 1
        if cfg.checkpoint_every and ((step + 1) % cfg.checkpoint_every == 0) and not is_last:
            state_path = training_client.save_state(name=f"r{round_n}-step{step + 1}").result().path
            if on_checkpoint:
                on_checkpoint(step + 1, state_path)

    save = training_client.save_weights_for_sampler(name=f"star-round{round_n}")
    checkpoint = save.result().path

    train_log = {
        "round": round_n,
        "n_steps": n_steps,
        "resumed_from_step": start_step,
        "resume_state_path": resume_state_path,
        "checkpoint_every": cfg.checkpoint_every,
        "corpus_size": len(corpus),
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "trained_from": cfg.base_model,   # NOT the previous round's adapter
        "dataset_scope": "this round only (D_n u D^rat_n), not accumulated",
        "train_tokens": train_tokens,
        "max_seq_length": cfg.max_seq_length,
        "n_truncated_examples": n_truncated,
        "checkpoint": checkpoint,
        "steps": log,
    }
    write_json(out_dir / "train_log.json", train_log)
    (out_dir / "checkpoint.txt").write_text(checkpoint + "\n", encoding="utf-8")
    return train_log


def read_checkpoint(round_dir: Path) -> str:
    path = round_dir / "checkpoint.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint for {round_dir.name}: {path} missing")
    return path.read_text(encoding="utf-8").strip()


def estimate_train_tokens(
    corpus: Sequence[Dict[str, Any]], cfg: StarConfig, round_n: int, token_counter
) -> int:
    """Dry-run estimate: tokens billed as `train` for one round."""
    if not corpus:
        return 0
    sizes = [
        token_counter(r["prompt"]) + token_counter(r["completion"]) for r in corpus[:32]
    ]
    mean = sum(sizes) / len(sizes)
    return int(cfg.steps_for_round(round_n) * cfg.batch_size * mean)
