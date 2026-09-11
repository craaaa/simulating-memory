"""Prompts for STaR rationale generation on digit span.

Base prompt is the repo's C3 condition (human simulation + limited-memory emphasis),
imported verbatim from bench so prompts stay comparable to existing runs, with a
<reasoning> block added ahead of the existing `press <<D>>.` answer format.

Three prompt variants, and which one is used where matters:
  * build_sample_prompt(trial, fewshot=True)   -- Algorithm 1 line 3 (generation)
  * build_rationalize_prompt(trial)            -- Algorithm 1 line 4 (hint given)
  * build_sample_prompt(trial, fewshot=False)  -- the TRAINING prompt: no hint, no
    few-shot, "as if the model had come up with the rationale without the hint"
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bench.tasks.digit_span_forward import (
    FORMAT_RULES as FORMAT_RULES_FORWARD,
    parse_pressed_digits,
)
from bench.tasks.digit_span_reverse import FORMAT_RULES as FORMAT_RULES_REVERSE
from bench.tasks.digit_span_forward import HUMAN_PROMPT as HUMAN_PROMPT_FORWARD
from bench.tasks.digit_span_reverse import HUMAN_PROMPT as HUMAN_PROMPT_REVERSE
from bench.tasks.human_simulation_prefixes import HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from bench.tasks.stimulus_prompt_shell import (
    DIGIT_SPAN_SEP_BEFORE_RULES,
    wrap_stimulus_prompt,
)

from .data import HumanTrial

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

HUMAN_PROMPT = {"forward": HUMAN_PROMPT_FORWARD, "reverse": HUMAN_PROMPT_REVERSE}
BASE_FORMAT_RULES = {"forward": FORMAT_RULES_FORWARD, "reverse": FORMAT_RULES_REVERSE}

REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"

# The length cap is not cosmetic. Probing qwen3.8-flash produced rationales up to 3400
# characters -- one of them never closed the tag before hitting the token limit, and at
# that size prompt+completion overruns max_seq_length, which would silently truncate
# training targets.
_REASONING_PREAMBLE = f"""First think about how this particular human would encode and recall the sequence, inside a single block:
{REASONING_OPEN}
... your reasoning about what they would hold on to and what they would lose ...
{REASONING_CLOSE}
Keep the reasoning to at most three sentences. Do not work through the sequence step by step, and do not second-guess yourself -- state what this human would encode and what would slip, then stop.
Then, on the lines after {REASONING_CLOSE}:
"""

_HINT_TEMPLATE = """The human actually responded:
{human_response}

Explain in {open_tag}...{close_tag} why a human doing this task would produce exactly that response, then reproduce it in the answer lines.
"""

PLACEHOLDER_MARKER = "PLACEHOLDER"


def format_rules(direction: str) -> str:
    return _REASONING_PREAMBLE + BASE_FORMAT_RULES[direction]


def render_response(digits: List[int]) -> str:
    """Digits in the repo's answer format."""
    return "\n".join(f"press <<{d}>>." for d in digits)


def load_fewshot(direction: str) -> str:
    path = PROMPTS_DIR / f"fewshot_{direction}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing few-shot rationale demos: {path}")
    return path.read_text(encoding="utf-8")


def fewshot_has_placeholders(direction: str) -> bool:
    return PLACEHOLDER_MARKER in load_fewshot(direction)


# bench's C3 text is already direction-specific (HUMAN_PROMPT differs for forward vs
# reverse, and the reverse one carries a worked reversal example), but it never names
# the task. These lines state it explicitly so the direction cannot be inferred wrong
# from an otherwise near-identical prompt. This is an addition to bench's C3 wording
# and is recorded in run_config.json under prompt_additions.
TASK_LINE = {
    "forward": (
        "Task: FORWARD digit span.\n"
        "The human must recall the digits in the SAME order they appeared "
        "(first digit first).\n"
    ),
    "reverse": (
        "Task: REVERSE digit span.\n"
        "The human must recall the digits in REVERSE order, last digit first. "
        "This is NOT the forward task -- the response is the presented sequence "
        "read backwards.\n"
    ),
}


def _prefix(direction: str, *, fewshot: bool) -> str:
    """Task line + C3 prefix + direction-specific task description, with few-shot
    demos appended the way C4 does."""
    prefix = (
        TASK_LINE[direction]
        + HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
        + HUMAN_PROMPT[direction]
    )
    if fewshot:
        prefix = prefix + "\n" + load_fewshot(direction)
    return prefix


def prompt_additions() -> Dict[str, str]:
    """Every string this module adds on top of bench's C3 condition, for provenance."""
    return {
        "task_line_forward": TASK_LINE["forward"],
        "task_line_reverse": TASK_LINE["reverse"],
        "reasoning_preamble": _REASONING_PREAMBLE,
        "hint_template": _HINT_TEMPLATE,
    }


def build_sample_prompt(trial: HumanTrial, *, fewshot: bool = True) -> str:
    """Algorithm 1 line 3, and (with fewshot=False) the training-time prompt."""
    return wrap_stimulus_prompt(
        _prefix(trial.direction, fewshot=fewshot),
        "C3",
        trial.stimulus_text,
        format_rules(trial.direction),
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def build_rationalize_prompt(trial: HumanTrial, *, fewshot: bool = True) -> str:
    """Algorithm 1 line 4: add_hint(x_i, y_i), where y_i is the human's response.

    The hint is never part of a training datum -- see build_sample_prompt(fewshot=False).
    """
    hint = _HINT_TEMPLATE.format(
        human_response=render_response(trial.user_digits),
        open_tag=REASONING_OPEN,
        close_tag=REASONING_CLOSE,
    )
    return wrap_stimulus_prompt(
        _prefix(trial.direction, fewshot=fewshot),
        "C3",
        trial.stimulus_text + "\n\n" + hint,
        format_rules(trial.direction),
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def parse_rationale(text: str) -> Tuple[Optional[str], List[int], List[str]]:
    """Split reasoning from answer, then parse digits from the ANSWER ONLY.

    Deliberately differs from bench's whole-string parse: bench applies PRESS_RE to
    the entire completion, which is fine when there is no reasoning block, but here a
    sentence like "they would press 7 first, then lose the rest" would inject digits
    into the answer. So the text is split on </reasoning> and only the tail is parsed.
    The regex itself is bench's PRESS_RE, unchanged, for comparability.
    """
    errors: List[str] = []
    if text is None:
        return None, [], ["empty_response"]

    if REASONING_CLOSE not in text:
        return None, [], ["no_reasoning_block"]

    head, _, tail = text.partition(REASONING_CLOSE)
    reasoning = head.replace(REASONING_OPEN, "", 1).strip()
    if not reasoning:
        errors.append("empty_reasoning")

    digits = parse_pressed_digits(tail)
    if not digits:
        errors.append("no_digits_parsed")
    return (reasoning or None), digits, errors


def build_completion(reasoning: str, digits: List[int]) -> str:
    """Canonical training target: reasoning block then answer lines."""
    return f"{REASONING_OPEN}\n{reasoning.strip()}\n{REASONING_CLOSE}\n{render_response(digits)}"


_HINT_LEAK_PATTERNS = [
    re.compile(r"actually responded", re.I),
    re.compile(r"\bthe (?:given|provided|stated) (?:answer|response)\b", re.I),
    re.compile(r"\b(?:as|since) (?:we are|I am|you are) told\b", re.I),
    re.compile(r"\bthe hint\b", re.I),
    re.compile(r"\bgiven (?:that )?the (?:human's )?response (?:is|was)\b", re.I),
]


def hint_leak(reasoning: str) -> Optional[str]:
    """Detect a rationale that gives away that it was handed the answer.

    Not part of STaR -- the paper only removes the hint from the prompt. Kept on by
    default because our hint is a short digit string that is trivially quotable, but
    it can be disabled to restore paper behavior.
    """
    for pat in _HINT_LEAK_PATTERNS:
        m = pat.search(reasoning or "")
        if m:
            return m.group(0)
    return None
