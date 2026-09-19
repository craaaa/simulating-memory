"""Shared color palettes, display names, and style constants for the
application/ plotting scripts (listening_qa and reading_qa comparison plots).

Single source of truth -- other plot_*.py files import from here instead of
redefining these locally, so a palette change only has to happen in one place.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Neutral UI colors, shared across nearly every comparison plot in application/.
# ---------------------------------------------------------------------------
INK = "#2b2b2b"
GRID = "#d9d9d9"
MUTED = "#6b6b6b"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# Chance-level reference line for exact-match / preference-agreement plots.
CHANCE = 0.5

# ---------------------------------------------------------------------------
# Per-reading-level palette.
# ---------------------------------------------------------------------------
# Canonical palette, sourced from the human study's own analysis script
# (application/prolific_study/multi_v6/analyze_v6.py: colors = [...], CONDS
# order from data_prep.py). Used by plot_open_source_compactor_vs_ceiling.py
# and the newer listening_qa comparison plots.
LEVEL_COLORS: dict[str, str] = {
    "control": "#94a3b8",
    "repeat_short": "#fb923c",
    "repeat_long": "#f97316",
    "distractor": "#a78bfa",
}

# Distinct palette for LEVEL-PAIR plots (a pair spans two levels, so it isn't
# just "one of the four LEVEL_COLORS" -- 6 pairs, tableau10-ish categorical).
PAIR_COLORS: dict[str, str] = {
    "control_vs_repeat_short": "#4e79a7",
    "control_vs_repeat_long": "#f28e2b",
    "control_vs_distractor": "#e15759",
    "repeat_short_vs_repeat_long": "#76b7b2",
    "repeat_short_vs_distractor": "#59a14f",
    "repeat_long_vs_distractor": "#af7aa1",
}

# ---------------------------------------------------------------------------
# Condition/series palette: prompting (C1-C3) + compactor (WM) + human baseline.
# ---------------------------------------------------------------------------
SERIES_COLORS: dict[str, str] = {
    "human": "#2b2b2b",
    "C1": "#c6dbef",
    "C2": "#6baed6",
    "C3": "#2171b5",
    "WM": "#238b45",
}

# ---------------------------------------------------------------------------
# Per-model palette, for cross-model comparison plots (slopeplots etc).
# ---------------------------------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "GPT-4.1": "#6b6b6b",
    "Qwen2.5-72B-Instruct": "#eb6834",
    "Gemma-4-31B-it": "#1baf7a",
    "Kimi-K2-0905": "#4a3aa7",
    "Qwen2.5-32B-Instruct": "#d62728",
    "Gemini-3.1-Pro-Preview": "#1f77b4",
    "Command-A": "#bcbd22",
}

# ---------------------------------------------------------------------------
# Friendly display names for known model-slug/run-dir names; anything not
# listed falls back to an auto-formatted version of its slug (see
# plot_open_source_compactor_vs_ceiling._display_name).
# ---------------------------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    "gpt-4.1": "GPT-4.1",
    "openai_gpt-4.1-mini": "GPT-4.1-mini",
    "google_gemma-4-31b-it": "Gemma-4-31B-it",
    "google_gemma-4-26b-a4b-it": "Gemma-4-26B-A4B-it",
    "google_gemma-3-27b-it": "Gemma-3-27B-it",
    "qwen_qwen-2.5-72b-instruct": "Qwen2.5-72B-Instruct",
    "Qwen_Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "mistralai_mistral-small-24b-instruct-2501": "Mistral-Small-24B-2501",
    "microsoft_wizardlm-2-8x22b": "WizardLM-2-8x22B",
    "microsoft_phi-4": "Phi-4",
    "deepseek_deepseek-r1-distill-llama-70b": "R1-Distill-Llama-70B",
    "deepseek_deepseek-v3.2": "DeepSeek-V3.2",
    "meta-llama_llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
    "meta-llama_llama-3.1-70b-instruct": "Llama-3.1-70B-Instruct",
    "meta-llama_llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "meta-llama_Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "meta-llama_llama-4-maverick": "Llama-4-Maverick",
    "meta-llama_llama-4-scout": "Llama-4-Scout",
    "nousresearch_hermes-4-70b": "Hermes-4-70B",
    "nvidia_nemotron-3-super-120b-a12b": "Nemotron-3-Super-120B",
    "nvidia_nemotron-3-nano-30b-a3b": "Nemotron-3-Nano-30B",
    "nvidia_nemotron-3-ultra-550b-a55b": "Nemotron-3-Ultra-550B",
    "cohere_command-a": "Command-A",
    "moonshotai_kimi-k2-0905": "Kimi-K2-0905",
    "z-ai_glm-4.6": "GLM-4.6",
    "qwen_qwen3-32b": "Qwen3-32B",
    "qwen_qwen3-235b-a22b-2507": "Qwen3-235B-A22B-2507",
    "qwen_qwen3-30b-a3b-instruct": "Qwen3-30B-A3B-Instruct",
    "qwen_qwen3-30b-a3b-thinking": "Qwen3-30B-A3B-Thinking",
    "qwen_qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B-Instruct",
    "qwen_qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
    "qwen_qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
    "qwen_qwen3.6-27b": "Qwen3.6-27B",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro-Preview",
    "gemini-3.6-flash": "Gemini-3.6-Flash",
    "gemini-3.5-flash": "Gemini-3.5-Flash",
}


def display_name(slug: str) -> str:
    """Friendly name for a known model slug; auto-formats unknown ones."""
    if slug in DISPLAY_NAMES:
        return DISPLAY_NAMES[slug]
    return slug.split("_", 1)[-1].replace("-", " ").title().replace(" ", "-")
