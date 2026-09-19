"""Prompts for STaR rationale generation on listening QA.

The base prompt is the repo's C3 condition, imported verbatim from the listening task
so prompts stay comparable to the existing full-grid runs, with a <reasoning> block
added ahead of the answer line.

One deliberate departure from the listening task's own prompt: it presents a passage
with all five questions at once and asks for five answer lines. Here the unit is a
single question, because the STaR filter is exact match against one human's answer --
requiring an exact match on all five questions simultaneously would drive D_n to
near-empty. The other four questions are still present, as that participant answered
them, but as CONTEXT for who this person is rather than as things to answer. Both the
single-question framing and the sibling context are recorded in run_config.json under
prompt_additions.

Three prompt variants, and which one is used where matters:
  * build_sample_prompt(item, fewshot=True)   -- Algorithm 1 line 3 (generation)
  * build_rationalize_prompt(item)            -- Algorithm 1 line 4 (hint given)
  * build_sample_prompt(item, fewshot=False)  -- the TRAINING prompt: no hint, no
    few-shot, "as if the model had come up with the rationale without the hint"
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from application.listening_qa.prompting import CONDITIONS
from bench.tasks.stimulus_prompt_shell import (
    DIGIT_SPAN_SEP_BEFORE_RULES,
    wrap_stimulus_prompt,
)

from ..prompting import REASONING_CLOSE, REASONING_OPEN, hint_leak as _shared_hint_leak
from .data import ListeningItem, SiblingAnswer

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
FEWSHOT_PATH = PROMPTS_DIR / "fewshot_listening.txt"
PLACEHOLDER_MARKER = "PLACEHOLDER"

CONDITION_ID = "C3"

# Every question in all four banks has exactly five options, the fifth being "None of
# the above". An answer number outside that range is a parse failure, not an answer.
MAX_OPTION = 5

ANSWER_RE = re.compile(
    r"^\s*answer:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*$", re.IGNORECASE | re.MULTILINE
)
ANSWER_ANY_RE = re.compile(r"^\s*answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# The listening task's own prompt never says the model is answering one question; with
# the sibling block present, an unstated unit is genuinely ambiguous.
TASK_LINE = (
    "Task: LISTENING comprehension, one multiple-select question.\n"
    "The human heard the passage once, as audio, and cannot replay it. You are shown "
    "the transcript of what they heard.\n"
    "Answer ONLY the question marked 'Question to predict'. The other questions are "
    "shown so you know how this particular person answered them -- do not answer them "
    "again.\n"
)

_REASONING_PREAMBLE = f"""First think about what this particular human would still be holding on to from the passage, inside a single block:
{REASONING_OPEN}
... your reasoning about what they would remember, what would blur, and which options that makes tempting ...
{REASONING_CLOSE}
Keep the reasoning to at most three sentences. Do not work through the options one by one, and do not second-guess yourself -- state what this human would retain and what would slip, then stop.
Then, on the line after {REASONING_CLOSE}:
"""

FORMAT_RULES_ANSWER = """Output ONLY a line in this exact format:
Answer: 1,3
(List every option number this human would select, separated by commas. If they would select none of the content options, use the "None of the above" option number.)
"""

_HINT_TEMPLATE = """The human actually selected:
{human_selection}

Explain in {open_tag}...{close_tag} why a human who had heard this passage once would select exactly those options, then reproduce them in the answer line.
"""

_SIBLING_HEADER = (
    "The same human answered these other questions about the same passage:\n"
)
_TARGET_HEADER = "Question to predict:\n"

# The hint here is a short list of option numbers, trivially quotable, so the shared
# leak patterns are extended with the listening wording.
_LISTENING_LEAK_PATTERNS = [
    re.compile(r"actually selected", re.I),
    re.compile(r"\bthe (?:given|provided|stated) (?:selection|options?|choices?)\b", re.I),
    re.compile(r"\bwe (?:are|were) (?:given|shown) (?:their|the human's) (?:answer|selection)\b", re.I),
]


def format_rules() -> str:
    return _REASONING_PREAMBLE + FORMAT_RULES_ANSWER


def render_options(options: Dict[int, str]) -> str:
    return " ".join(f"{n}) {options[n]}" for n in sorted(options))


def render_answer(answer: Sequence[int]) -> str:
    """The answer line, in the format the format rules ask for."""
    return "Answer: " + ",".join(str(n) for n in sorted(set(answer)))


def render_selection(options: Dict[int, str], selected: Sequence[int]) -> str:
    """A human's selection as numbers AND text.

    Numbers alone would make the hint a string to copy rather than something to
    explain, and the whole point of line 4 is to get a rationale for the content.
    """
    chosen = sorted(set(selected))
    if not chosen:
        return "(nothing)"
    return "\n".join(f"{n}) {options.get(n, '?')}" for n in chosen)


def load_fewshot() -> str:
    if not FEWSHOT_PATH.is_file():
        raise FileNotFoundError(f"missing few-shot rationale demos: {FEWSHOT_PATH}")
    return FEWSHOT_PATH.read_text(encoding="utf-8")


def fewshot_has_placeholders() -> bool:
    return PLACEHOLDER_MARKER in load_fewshot()


def _prefix(*, fewshot: bool) -> str:
    prefix = TASK_LINE + CONDITIONS[CONDITION_ID]["prompt_prefix"]
    if fewshot:
        prefix = prefix + "\n" + load_fewshot()
    return prefix


def _sibling_block(siblings: Sequence[SiblingAnswer]) -> str:
    lines: List[str] = [_SIBLING_HEADER.rstrip("\n")]
    for s in siblings:
        lines.append(f"- {s.question}")
        selected = s.endorsed_text()
        if selected:
            lines.append("  They selected: " + "; ".join(selected))
        else:
            lines.append("  They selected: (nothing)")
    return "\n".join(lines)


def build_stimulus(
    item: ListeningItem, *, sibling_context: bool = True, hint: Optional[str] = None
) -> str:
    parts: List[str] = ["Transcript:", item.passage, ""]
    if sibling_context and item.other_answers:
        parts += [_sibling_block(item.other_answers), ""]
    parts += [
        _TARGET_HEADER.rstrip("\n"),
        f"{item.question} {render_options(item.options)}",
    ]
    if hint:
        parts += ["", hint.rstrip("\n")]
    return "\n".join(parts)


def build_sample_prompt(
    item: ListeningItem, *, fewshot: bool = True, sibling_context: bool = True
) -> str:
    """Algorithm 1 line 3, and (with fewshot=False) the training-time prompt."""
    return wrap_stimulus_prompt(
        _prefix(fewshot=fewshot),
        CONDITION_ID,
        build_stimulus(item, sibling_context=sibling_context),
        format_rules(),
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def build_rationalize_prompt(
    item: ListeningItem, *, fewshot: bool = True, sibling_context: bool = True
) -> str:
    """Algorithm 1 line 4: add_hint(x_i, y_i), where y_i is the human's own selection.

    The hint is never part of a training datum -- see build_sample_prompt(fewshot=False).
    """
    hint = _HINT_TEMPLATE.format(
        human_selection=render_selection(item.options, item.endorsed),
        open_tag=REASONING_OPEN,
        close_tag=REASONING_CLOSE,
    )
    return wrap_stimulus_prompt(
        _prefix(fewshot=fewshot),
        CONDITION_ID,
        build_stimulus(item, sibling_context=sibling_context, hint=hint),
        format_rules(),
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def build_completion(reasoning: str, answer: Sequence[int]) -> str:
    """Canonical training target: reasoning block then the answer line."""
    return (
        f"{REASONING_OPEN}\n{reasoning.strip()}\n{REASONING_CLOSE}\n{render_answer(answer)}"
    )


def parse_rationale(text: str) -> Tuple[Optional[str], List[int], List[str]]:
    """Split reasoning from answer, then parse option numbers from the ANSWER ONLY.

    The split is not cosmetic: rationales talk about options by number ("she would take
    2 and 3 but miss 1"), and parsing the whole completion would pull those into the
    answer.
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

    match = ANSWER_RE.search(tail)
    if not match:
        # Distinguish "answered in prose" from "no answer line at all": the first is a
        # format problem worth fixing in the prompt, the second usually means the
        # completion was cut off.
        errors.append(
            "answer_not_numeric" if ANSWER_ANY_RE.search(tail) else "no_answer_line"
        )
        return (reasoning or None), [], errors

    options = sorted({int(o.strip()) for o in match.group(1).split(",")})
    out_of_range = [o for o in options if o < 1 or o > MAX_OPTION]
    if out_of_range:
        errors.append("option_out_of_range")
        options = [o for o in options if 1 <= o <= MAX_OPTION]
    if not options:
        errors.append("no_options_parsed")
    return (reasoning or None), options, errors


def hint_leak(reasoning: str) -> Optional[str]:
    """Detect a rationale that gives away that it was handed the answer.

    Not part of STaR -- the paper only removes the hint from the prompt.
    """
    shared = _shared_hint_leak(reasoning)
    if shared:
        return shared
    for pat in _LISTENING_LEAK_PATTERNS:
        m = pat.search(reasoning or "")
        if m:
            return m.group(0)
    return None


def prompt_additions() -> Dict[str, str]:
    """Every string this module adds on top of the listening task's C3 condition."""
    return {
        "task_line": TASK_LINE,
        "reasoning_preamble": _REASONING_PREAMBLE,
        "format_rules_answer": FORMAT_RULES_ANSWER,
        "hint_template": _HINT_TEMPLATE,
        "sibling_header": _SIBLING_HEADER,
        "target_header": _TARGET_HEADER,
        "unit_of_prediction": (
            "one (participant, topic, question); the listening task's own prompt asks "
            "for all five questions of a passage at once"
        ),
    }
