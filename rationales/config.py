"""Configuration for STaR rationale bootstrapping on digit span.

All tunables live here. Values marked PLACEHOLDER need a decision before a real run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tinker pricing, USD per 1M tokens.
# Verified 2026-09-10 against https://tinker-docs.thinkingmachines.ai/tinker/models.json
# Prices change (prefill/sample rose ~50% and train ~10% on 2026-07-17) -- re-check
# the live models.json before trusting an estimate. Unknown models return None.
# ---------------------------------------------------------------------------
TINKER_PRICING_PER_MILLION: Dict[str, Dict[str, float]] = {
    "Qwen/Qwen3-8B": {"prefill": 0.195, "sample": 0.60, "train": 0.44},
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "sample": 1.005, "train": 0.737},
    "Qwen/Qwen3.5-9B": {"prefill": 0.66, "sample": 1.995, "train": 1.463},
    "openai/gpt-oss-20b": {"prefill": 0.18, "sample": 0.45, "train": 0.396},
}
TINKER_PRICING_VERIFIED_ON = "2026-09-10"

# ---------------------------------------------------------------------------
# Token-accounting constants, MEASURED rather than guessed (2026-09-11).
# Taken from OpenRouter `usage.prompt_tokens` on qwen/qwen3.8-flash, which shares the
# Qwen tokenizer family with the Tinker base Qwen3-8B:
#   generation prompt (few-shot)        3337 chars -> 930 tokens  (3.59 chars/token)
#   rationalization prompt (+ hint)     3577 chars -> 1011 tokens (3.54)
#   training / eval prompt (zero-shot)  1505 chars -> 374 tokens  (4.02)
# The flat 4-chars/token rule under-counts few-shot prompts by ~13%, which is exactly
# where most of the prefill spend sits.
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 3.72
# A completion at the three-sentence cap: ~320 chars of reasoning + answer lines.
COMPLETION_TOKENS = 114
# Share of human-SUCCESS trials the model is expected to miss on the unhinted path,
# sending them to rationalization too. Probed at 1/4; the success half is not free.
SUCCESS_MISS_RATE = 0.25


def tinker_cost_usd(
    base_model: str,
    prefill_tokens: int = 0,
    sample_tokens: int = 0,
    train_tokens: int = 0,
) -> Optional[float]:
    """Cost estimate from Tinker list pricing. None for unrecognized models."""
    rates = TINKER_PRICING_PER_MILLION.get(base_model)
    if rates is None:
        key = base_model.split("/", 1)[-1].lower()
        for name, r in TINKER_PRICING_PER_MILLION.items():
            if name.split("/", 1)[-1].lower() == key:
                rates = r
                break
    if rates is None:
        return None
    return (
        prefill_tokens / 1_000_000 * rates["prefill"]
        + sample_tokens / 1_000_000 * rates["sample"]
        + train_tokens / 1_000_000 * rates["train"]
    )


@dataclass(frozen=True)
class StarConfig:
    """Everything that defines a STaR run. Serialized into run_config.json."""

    # --- task --------------------------------------------------------------
    # Which RationaleTask the loop is pointed at (see rationales/task.py). Everything
    # below is task-independent except `directions`, which only digit span reads.
    task: str = "digit_span"

    # --- model / API -------------------------------------------------------
    # PLACEHOLDER: confirm against Tinker's supported-model list. Note Qwen3-8B is
    # cheaper per token than Qwen3.5-4B (see TINKER_PRICING_PER_MILLION).
    base_model: str = "Qwen/Qwen3-8B"
    # PLACEHOLDER: cookbook renderer for this model family. Must match base_model.
    renderer_name: str = "qwen3_disable_thinking"
    lora_rank: int = 16

    # --- STaR loop ---------------------------------------------------------
    # Algorithm 1 decodes greedily, one sample per problem. k>1 / temperature>0 is a
    # deviation (RFT/ReST territory) and is recorded as such in run_config.json.
    k: int = 1
    temperature: float = 0.0
    max_tokens: int = 512
    rounds: int = 1
    use_fewshot: bool = True
    leak_filter: bool = True
    # listening_qa only: show the participant's answers to the passage's other four
    # questions. A prompt-shaping knob, same class as use_fewshot, and on StarConfig
    # rather than baked into the task so that it is serialized with the run and can
    # actually be turned off. See ListeningQATask for why it might need to be.
    sibling_context: bool = True

    # --- training ----------------------------------------------------------
    # Paper: 100-step LR warmup then constant LR; 40 steps at the first outer loop,
    # growing 20% per round. PLACEHOLDER: 40 was tuned for GPT-J on 10k arithmetic
    # examples; with ~560 examples set steps_1 from the round-1 loss curve.
    learning_rate: float = 1e-4
    steps_1: int = 60
    step_growth: float = 1.2
    warmup_steps: int = 100
    batch_size: int = 8
    # save_state (weights + optimizer) every N steps, so a pause costs at most this
    # many steps of re-training. 0 disables mid-round checkpoints.
    checkpoint_every: int = 20
    # Must hold the zero-shot training prompt (~900 tokens) plus the completion.
    # Probing showed rationales up to ~3400 characters (~850 tokens), so 1024 would
    # silently truncate training targets; build_datum cuts from the right, which would
    # drop the answer lines the loss is supposed to cover.
    max_seq_length: int = 2048

    # --- data --------------------------------------------------------------
    seed: int = 42
    eval_frac: float = 0.15
    directions: tuple = ("forward", "reverse")

    # --- parallelism -------------------------------------------------------
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        # k>1 at temperature 0 draws the same greedy completion k times: k-fold cost,
        # no extra coverage. Rejection sampling only buys anything with a temperature.
        if self.k > 1 and self.temperature == 0.0:
            raise ValueError(
                f"k={self.k} with temperature=0.0 samples the same greedy completion "
                f"{self.k} times -- {self.k}x the cost for no extra coverage. Either "
                "keep k=1 (STaR's greedy decoding) or raise --temperature."
            )
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")

    def steps_for_round(self, round_n: int) -> int:
        return int(round(self.steps_1 * (self.step_growth ** (round_n - 1))))

    @property
    def deviations(self) -> Dict[str, Any]:
        """Non-STaR-faithful settings, recorded with every run."""
        out: Dict[str, Any] = {
            "filter_target": "human_response_not_ground_truth",
            "dataset_balance": "1:1 success/fail subsample",
            "hint_leak_filter": self.leak_filter,
        }
        if self.k != 1 or self.temperature != 0.0:
            out["decoding"] = (
                f"k={self.k}, temperature={self.temperature} "
                "(STaR decodes greedily with one sample per problem)"
            )
        if not self.use_fewshot:
            out["fewshot"] = "disabled (STaR prompts with few-shot rationales)"
        if self.task == "listening_qa" and not self.sibling_context:
            out["sibling_context"] = (
                "disabled (the prompt normally carries the participant's answers to "
                "the passage's other four questions)"
            )
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["directions"] = list(self.directions)
        d["deviations"] = self.deviations
        d["tinker_pricing_verified_on"] = TINKER_PRICING_VERIFIED_ON
        return d


# Every Tinker call ever made by this package appends here, on top of any per-run
# ledger, so the lifetime tally survives across runs, smoke tests and pauses.
GLOBAL_LEDGER = PKG_DIR / "out" / "spend_ledger.jsonl"


def model_slug(base_model: str) -> str:
    return base_model.replace("/", "_")


def out_root(cfg: StarConfig, timestamp: str) -> Path:
    return PKG_DIR / "out" / model_slug(cfg.base_model) / timestamp
