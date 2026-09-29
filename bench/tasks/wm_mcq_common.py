"""Shared helper for WM-agent tasks that follow the encode-material → answer-MCQs pattern."""
from __future__ import annotations
from typing import Any, Dict

from ..core.llm import LLM
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS


MCQ_RECALL_TEMPLATE = """\
The study phase is now over.
Your working memory currently contains:
{{wm_contents}}

{preamble}

{questions_text}

{format_rules}"""


SUMMARIZER_MCQ_RECALL_TEMPLATE = """\
The study phase is now over.
Your summary of the material is:
{{summary}}

{preamble}

{questions_text}

{format_rules}"""


def run_wm_mcq_trial(
    llm: LLM,
    condition_id: str,
    temperature: float,
    debug: bool,
    encode_content: str,
    questions_text: str,
    recall_preamble: str,
    format_rules: str,
    system_prompt_override: str | None = None,
    recall_max_tokens: int = 512,
    participant_id: Any = None,
) -> Dict[str, Any]:
    """Run one WM-agent trial: encode material, then answer MCQs from memory.

    ``participant_id`` identifies the scored unit this trial produces and is forwarded
    to the agent unchanged. Callers that run several items per participant (craft_task,
    map_task: one row per item) must pass a key that distinguishes the items, because
    ``src/score.py::llm_scores`` scores each of those rows as its own observation.

    Returns
    -------
    dict with keys: ``agent``, ``encoding_log``, ``recall_raw``, ``final_kv``.
    """
    agent = WorkingMemoryAgent(
        llm=llm,
        condition_id=condition_id,
        temperature=temperature,
        debug=debug,
        system_prompt_override=system_prompt_override,
        participant_id=participant_id,
    )

    encoding_log = agent.encode(encode_content)

    # Build recall prompt — the template uses {{ }} for the wm_contents placeholder
    # so we first format everything else, then the result has {wm_contents} for agent.recall()
    recall_prompt = MCQ_RECALL_TEMPLATE.format(
        preamble=recall_preamble,
        questions_text=questions_text,
        format_rules=format_rules,
    )

    recall_raw = agent.recall(recall_prompt=recall_prompt, max_tokens=recall_max_tokens)
    final_kv = agent.wm.store

    return {
        "agent": agent,
        "encoding_log": encoding_log,
        "recall_raw": recall_raw,
        "final_kv": final_kv,
        "slot_utilization": len(final_kv) / MAX_KEYS,
    }


def run_summarizer_mcq_trial(
    llm: LLM,
    condition_id: str,
    temperature: float,
    debug: bool,
    encode_content: str,
    questions_text: str,
    recall_preamble: str,
    format_rules: str,
    system_prompt_override: str | None = None,
    recall_max_tokens: int = 512,
    encode_max_tokens: int = 4096,
    participant_id: Any = None,
) -> Dict[str, Any]:
    """Run one summarizer-agent trial: summarize material, then answer MCQs from summary."""
    agent = SummarizerAgent(
        llm=llm,
        condition_id=condition_id,
        temperature=temperature,
        debug=debug,
        system_prompt_override=system_prompt_override,
        participant_id=participant_id,
    )

    encoding_log = agent.encode(encode_content, max_tokens=encode_max_tokens)

    recall_prompt = SUMMARIZER_MCQ_RECALL_TEMPLATE.format(
        preamble=recall_preamble,
        questions_text=questions_text,
        format_rules=format_rules,
    )

    recall_raw = agent.recall(recall_prompt=recall_prompt, max_tokens=recall_max_tokens)

    return {
        "agent": agent,
        "encoding_log": encoding_log,
        "recall_raw": recall_raw,
        "final_summary": agent.summary,
        "summary_length_words": agent.summary_length_words(),
    }
