"""Tinker SDK surface, isolated in one module.

Nothing else in this package imports `tinker`, so the rest of the pipeline (data
selection, prompting, parsing, filtering, metrics) runs and is testable without the
SDK installed or an API key present.

API shapes pinned against https://tinker-docs.thinkingmachines.ai (2026-09-10):
  service_client.create_lora_training_client_async(base_model=..., rank=...)
  training_client.forward_backward_async(datums, loss_fn="cross_entropy")
  training_client.optim_step_async(tinker.AdamParams(learning_rate=...))
  training_client.save_weights_for_sampler(name=...).result().path
  service_client.create_sampling_client(model_path=..., base_model=...)
  sampling_client.sample(prompt=..., num_samples=..., sampling_params=SamplingParams(...))
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .config import StarConfig, tinker_cost_usd


def _as_token_ids(out: Any) -> List[int]:
    """Normalize whatever apply_chat_template returned into a flat list of ids.

    transformers 5.x returns a BatchEncoding (dict-like) rather than a list, and
    batched calls nest one level deeper. Iterating a BatchEncoding yields its KEYS,
    so passing it straight through sends the strings "input_ids"/"attention_mask" to
    the API as tokens.
    """
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return [int(t) for t in out]


def require_api_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is not set. Export it before any paid call; "
            "use --dry-run to plan a run without one."
        )


def _tinker():
    try:
        import tinker  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "the tinker SDK is not installed. `uv pip install tinker tinker-cookbook` "
            "(see rationales/README.md)."
        ) from exc
    return tinker


class BudgetExceeded(RuntimeError):
    """Raised when live spend crosses the configured ceiling."""


@dataclass
class Usage:
    """Token counters for cost reporting. Tinker bills prefill/sample/train separately.

    Counted client-side, from the tokens actually sent and returned, because Tinker has
    no live spend API: `tinker billing usage` lags real time by "up to a few hours" and
    reports tokens/GB-hours rather than USD, and session metrics (console panels,
    Perfetto trace) expose live token totals but no price. Sampling counts come from
    `len(seq.tokens)` on the response, so they are exact rather than estimated.
    """

    prefill_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0
    requests: int = 0

    def add(self, *, prefill: int = 0, sample: int = 0, train: int = 0, requests: int = 1) -> None:
        self.prefill_tokens += prefill
        self.sample_tokens += sample
        self.train_tokens += train
        self.requests += requests

    def summary(self, base_model: str) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "prefill_tokens": self.prefill_tokens,
            "sample_tokens": self.sample_tokens,
            "train_tokens": self.train_tokens,
            "estimated_cost_usd": tinker_cost_usd(
                base_model,
                prefill_tokens=self.prefill_tokens,
                sample_tokens=self.sample_tokens,
                train_tokens=self.train_tokens,
            ),
        }


class CostTracker:
    """Live USD accounting: prices every call as it happens.

    Writes one JSONL row per call to ``ledger_path`` with the running total, so spend
    can be watched from another terminal (`python -m rationales.cli cost --watch`, or
    plain `tail -f`) while a round is in flight. Optionally aborts the run at a ceiling.
    """

    def __init__(
        self,
        base_model: str,
        *,
        ledger_path: Optional[Any] = None,
        max_usd: Optional[float] = None,
        show_progress: bool = True,
        run_label: str = "",
        global_ledger: bool = True,
    ) -> None:
        self.base_model = base_model
        self.usage = Usage()
        self.ledger_path = ledger_path
        self.max_usd = max_usd
        self.show_progress = show_progress
        self.run_label = run_label
        self.global_ledger = global_ledger
        self._sink = None
        self._global_sink = None
        self._lock = __import__("threading").Lock()

    @property
    def cost_usd(self) -> Optional[float]:
        return tinker_cost_usd(
            self.base_model,
            prefill_tokens=self.usage.prefill_tokens,
            sample_tokens=self.usage.sample_tokens,
            train_tokens=self.usage.train_tokens,
        )

    def record(
        self,
        kind: str,
        *,
        prefill: int = 0,
        sample: int = 0,
        train: int = 0,
        requests: int = 1,
        **fields: Any,
    ) -> None:
        with self._lock:
            self.usage.add(prefill=prefill, sample=sample, train=train, requests=requests)
            total = self.cost_usd
            row = {
                "kind": kind,
                "prefill_tokens": prefill,
                "sample_tokens": sample,
                "train_tokens": train,
                "cumulative": {
                    "requests": self.usage.requests,
                    "prefill_tokens": self.usage.prefill_tokens,
                    "sample_tokens": self.usage.sample_tokens,
                    "train_tokens": self.usage.train_tokens,
                    "cost_usd": total,
                },
                **fields,
            }
            if self.ledger_path is not None:
                if self._sink is None:
                    from .resume import AppendSink

                    self._sink = AppendSink(self.ledger_path)
                self._sink.append(row)

            # Lifetime tally, appended to regardless of which run produced the call.
            if self.global_ledger:
                if self._global_sink is None:
                    from .config import GLOBAL_LEDGER
                    from .resume import AppendSink

                    self._global_sink = AppendSink(GLOBAL_LEDGER)
                self._global_sink.append(
                    {
                        "model": self.base_model,
                        "run": self.run_label,
                        "kind": kind,
                        "prefill_tokens": prefill,
                        "sample_tokens": sample,
                        "train_tokens": train,
                        "cost_usd": tinker_cost_usd(
                            self.base_model,
                            prefill_tokens=prefill,
                            sample_tokens=sample,
                            train_tokens=train,
                        ),
                    }
                )
            if self.show_progress and total is not None:
                import sys

                sys.stderr.write(
                    f"\rspend ${total:0.4f} | {self.usage.requests} reqs | "
                    f"prefill {self.usage.prefill_tokens:,} "
                    f"sample {self.usage.sample_tokens:,} "
                    f"train {self.usage.train_tokens:,}\033[K"
                )
                sys.stderr.flush()

        if self.max_usd is not None and total is not None and total > self.max_usd:
            raise BudgetExceeded(
                f"spend ${total:0.4f} exceeded the ${self.max_usd:0.2f} ceiling after "
                f"{self.usage.requests} requests. Work completed so far is on disk; "
                "raise --max-usd to continue."
            )

    def summary(self) -> Dict[str, Any]:
        out = self.usage.summary(self.base_model)
        out["max_usd"] = self.max_usd
        out["ledger"] = str(self.ledger_path) if self.ledger_path else None
        out["pricing_note"] = (
            "Client-side estimate at Tinker list prices. Tinker exposes no live spend "
            "API: `tinker billing usage` lags a few hours and reports tokens, not USD. "
            "Reconcile afterwards by joining that export on user_metadata."
        )
        return out


class TinkerSession:
    """Lazily-built ServiceClient plus tokenizer, shared by sampling and training."""

    def __init__(
        self,
        cfg: StarConfig,
        *,
        tracker: Optional[CostTracker] = None,
        user_metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker or CostTracker(cfg.base_model, show_progress=False)
        # Attached to every training run so billing rows can be traced back to this run
        # (`tinker billing usage --sessions-csv` exports session_id + user_metadata).
        self.user_metadata = dict(user_metadata or {})
        self._service = None
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()

    @property
    def usage(self) -> Usage:
        return self.tracker.usage

    @property
    def service(self):
        if self._service is None:
            require_api_key()
            self._service = _tinker().ServiceClient()
        return self._service

    @property
    def tokenizer(self):
        """Cached tokenizer, built at most once across all worker threads.

        The lock is not optional. Sampling runs in a thread pool, and every thread's
        first call hits this property; without serialization each one launches its own
        HuggingFace download. Worse, a single failure leaves ``_tokenizer`` None, so
        every later call retries -- which is how round 2 died at ~841 calls with
        "429 Too Many Requests" followed by "Unable to load vocabulary from file".
        """
        if self._tokenizer is not None:
            return self._tokenizer
        with self._tokenizer_lock:
            if self._tokenizer is None:
                self._tokenizer = self._build_tokenizer()
        return self._tokenizer

    def _build_tokenizer(self):
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: PLC0415

                return get_tokenizer(self.cfg.base_model)
            except ImportError:
                break
            except Exception as exc:  # transient HF errors (429, network)
                last = exc
                time.sleep(2**attempt)

        # Local cache only -- avoids another network round trip when HF is rate-limiting.
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            return AutoTokenizer.from_pretrained(self.cfg.base_model, local_files_only=True)
        except Exception as exc:
            last = exc

        # Last resort: the training client's tokenizer (costs one API handshake).
        try:
            return self.training_client().get_tokenizer()
        except Exception as exc:
            raise RuntimeError(f"could not load a tokenizer for {self.cfg.base_model}") from (
                last or exc
            )

    def warm_up(self) -> None:
        """Force tokenizer construction before any thread pool starts."""
        _ = self.tokenizer

    # --- sampling ----------------------------------------------------------
    def sampling_client(self, model_path: Optional[str] = None):
        """model_path=None samples the base model (round 1); later rounds pass the
        sampler checkpoint saved by train.py."""
        if model_path:
            return self.service.create_sampling_client(
                model_path=model_path, base_model=self.cfg.base_model
            )
        return self.service.create_sampling_client(base_model=self.cfg.base_model)

    def sample(
        self,
        sampling_client,
        prompt_text: str,
        *,
        num_samples: int,
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        tinker = _tinker()
        tok = self.tokenizer
        prompt_ids = self.render_prompt_ids(prompt_text)
        model_input = tinker.ModelInput.from_ints(prompt_ids)
        response = sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=temperature
            ),
        )
        # The SDK returns an APIFuture here; resolve it. Handled defensively because
        # older/newer versions differ on whether sample() is eager.
        if hasattr(response, "result") and not hasattr(response, "sequences"):
            response = response.result()
        texts: List[str] = []
        sample_tokens = 0
        for seq in response.sequences:
            ids = list(getattr(seq, "tokens", []) or [])
            sample_tokens += len(ids)
            texts.append(tok.decode(ids))
        # Exact counts: prompt tokens we sent, completion tokens the API returned.
        self.tracker.record(
            "sample",
            prefill=len(prompt_ids) * num_samples,
            sample=sample_tokens,
            num_samples=num_samples,
        )
        return texts

    # --- training ----------------------------------------------------------
    def training_client(self):
        """A FRESH LoRA client off the base model.

        Algorithm 1 line 7 trains M, not M_{n-1} -- "we train from the original
        pre-trained model M instead of continually training one model to avoid
        overfitting". So every round calls this again rather than resuming state.
        """
        require_api_key()
        kwargs: Dict[str, Any] = {
            "base_model": self.cfg.base_model,
            "rank": self.cfg.lora_rank,
        }
        if self.user_metadata:
            kwargs["user_metadata"] = self.user_metadata
        try:
            return self.service.create_lora_training_client(**kwargs)
        except TypeError:
            # Older SDKs without user_metadata on this constructor.
            kwargs.pop("user_metadata", None)
            return self.service.create_lora_training_client(**kwargs)

    def training_client_from_state(self, state_path: str):
        """Resume training from a save_state path -- weights AND optimizer momentum.

        Only used to continue an interrupted round. A new round still starts from the
        base model (Algorithm 1 line 7 trains M, not M_{n-1}).
        """
        require_api_key()
        kwargs: Dict[str, Any] = {"path": state_path}
        if self.user_metadata:
            kwargs["user_metadata"] = self.user_metadata
        try:
            return self.service.create_training_client_from_state_with_optimizer(**kwargs)
        except TypeError:
            kwargs.pop("user_metadata", None)
            return self.service.create_training_client_from_state_with_optimizer(**kwargs)

    def build_datum(self, prompt_text: str, completion_text: str):
        """SFT datum with loss weights 0 over the prompt and 1 over the completion.

        This masking is the whole correctness of the SFT step: without it the model is
        trained to reproduce the prompt as well as the rationale.
        """
        tinker = _tinker()
        import torch  # noqa: PLC0415
        from tinker import types  # noqa: PLC0415

        tok = self.tokenizer
        # Identical rendering to sample(): if training saw a different prompt format
        # than sampling, the fine-tune would be optimizing a prompt the model never
        # actually receives at inference time.
        prompt_ids = self.render_prompt_ids(prompt_text)
        completion_ids = tok.encode(completion_text)
        eos = getattr(tok, "eos_token_id", None)
        if eos is not None:
            completion_ids = completion_ids + [eos]

        full_len = len(prompt_ids) + len(completion_ids)
        all_ids = (prompt_ids + completion_ids)[: self.cfg.max_seq_length]
        n_prompt = min(len(prompt_ids), len(all_ids))
        # Truncation cuts from the right, i.e. off the completion -- the part the loss
        # is supposed to cover. Surfaced so a too-small max_seq_length cannot quietly
        # train on half an answer.
        truncated = full_len > len(all_ids)

        input_ids = all_ids[:-1]
        target_ids = all_ids[1:]
        # target_ids[i] is predicted from input_ids[i]; the first completion token sits
        # at target index n_prompt-1.
        weights = [0.0] * len(target_ids)
        for i in range(max(n_prompt - 1, 0), len(target_ids)):
            weights[i] = 1.0

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_ids),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_ids)),
                "weights": tinker.TensorData.from_torch(torch.tensor(weights)),
            },
        )
        return datum, {
            "n_tokens": len(all_ids),
            "n_prompt_tokens": n_prompt,
            "truncated": truncated,
            "dropped_tokens": full_len - len(all_ids),
        }

    def render_prompt_ids(self, prompt_text: str) -> List[int]:
        """Token ids for a user turn, through the model's chat template.

        Instruct models (Qwen3-8B included) expect ChatML. Sending raw prompt text
        instead makes the model *continue the document* rather than answer it -- the
        first live smoke test came back with "No trailing spaces. No empty lines."
        repeated to the token limit and no <reasoning> block at all.

        Qwen3's template emits a thinking block by default; disabled here for the same
        reason it is disabled in the OpenRouter probe (hidden reasoning eats the
        completion budget and can return empty content).

        Falls back to a plain encode when the tokenizer has no template, which keeps
        base (non-instruct) models and the test stub working.
        """
        tok = self.tokenizer
        apply = getattr(tok, "apply_chat_template", None)
        if apply is None or getattr(tok, "chat_template", None) is None:
            return list(tok.encode(prompt_text))
        messages = [{"role": "user", "content": prompt_text}]
        try:
            out = apply(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Templates without an enable_thinking switch.
            out = apply(messages, tokenize=True, add_generation_prompt=True)
        return _as_token_ids(out)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


def offline_token_estimate(text: str) -> int:
    """Token count for --dry-run when no SDK/tokenizer is available.

    Uses CHARS_PER_TOKEN, measured against the Qwen tokenizer via OpenRouter rather
    than assumed (see config.py). Still an estimate -- labelled as such wherever it is
    reported -- but it no longer under-counts few-shot prompts by ~13%.
    """
    from .config import CHARS_PER_TOKEN

    return max(1, int(len(text) / CHARS_PER_TOKEN))
