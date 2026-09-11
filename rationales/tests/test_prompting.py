"""Prompt assembly, parsing, and the acceptance comparison.

The round-trip test here is the guard against a silent type bug: parsed digits are
list[int] while the human response is a str, so comparing against the raw string is
always False and would empty the generation path for every trial in every round.
"""
from __future__ import annotations

import pytest

from rationales import prompting as P


# --- direction is explicit and different -----------------------------------
def test_forward_and_reverse_prompts_are_not_the_same(fwd_fail, rev_fail):
    fwd = P.build_sample_prompt(fwd_fail, fewshot=False)
    rev = P.build_sample_prompt(rev_fail, fewshot=False)
    assert fwd != rev


@pytest.mark.parametrize(
    "direction,must_contain",
    [
        ("forward", ["FORWARD digit span", "SAME order"]),
        ("reverse", ["REVERSE digit span", "last digit first", "NOT the forward task"]),
    ],
)
def test_task_name_is_stated_explicitly(direction, must_contain, fwd_fail, rev_fail):
    trial = fwd_fail if direction == "forward" else rev_fail
    text = P.build_sample_prompt(trial, fewshot=False)
    for frag in must_contain:
        assert frag in text
    # and the other task's name never appears
    other = "REVERSE digit span" if direction == "forward" else "FORWARD digit span"
    assert other not in text


def test_prompt_reuses_bench_c3_wording(fwd_fail):
    from bench.tasks.digit_span_forward import FORMAT_RULES, HUMAN_PROMPT
    from bench.tasks.human_simulation_prefixes import HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN

    text = P.build_sample_prompt(fwd_fail, fewshot=False)
    assert HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN in text
    assert HUMAN_PROMPT in text
    assert FORMAT_RULES in text
    assert "This is the stimuli that will be presented to the human:" in text
    assert fwd_fail.stimulus_text in text


def test_stimulus_matches_bench_formatting(fwd_success):
    assert fwd_success.stimulus_text == "The digits are the following: [8, 4, 1]"


# --- few-shot and hint are sampling-time only ------------------------------
def test_training_prompt_has_no_fewshot_and_no_hint(rev_fail, monkeypatch):
    monkeypatch.setattr(P, "load_fewshot", lambda d: "FEWSHOT-DEMO-MARKER\n")
    sampled = P.build_sample_prompt(rev_fail, fewshot=True)
    rationalized = P.build_rationalize_prompt(rev_fail, fewshot=True)
    training = P.build_sample_prompt(rev_fail, fewshot=False)

    assert "FEWSHOT-DEMO-MARKER" in sampled
    assert "FEWSHOT-DEMO-MARKER" in rationalized
    assert "FEWSHOT-DEMO-MARKER" not in training
    assert "actually responded" in rationalized
    assert "actually responded" not in training
    assert "actually responded" not in sampled


def test_hint_contains_the_humans_response_not_the_correct_answer(rev_fail):
    text = P.build_rationalize_prompt(rev_fail, fewshot=False)
    assert P.render_response(rev_fail.user_digits) in text
    # The correct answer (5 4 7 2) must NOT be handed over.
    assert P.render_response(rev_fail.expected_digits) not in text


def test_fewshot_files_keep_their_structure():
    """Structure only -- the rationale text itself is the user's to write, and these
    files are edited over time, so nothing here asserts their contents."""
    for d in ("forward", "reverse"):
        text = P.load_fewshot(d)
        assert P.REASONING_OPEN in text and P.REASONING_CLOSE in text
        assert "press <<" in text
        assert "The digits are the following:" in text
        # every opened reasoning block is closed
        assert text.count(P.REASONING_OPEN) == text.count(P.REASONING_CLOSE)
    assert "FORWARD digit span" in P.load_fewshot("forward")
    assert "REVERSE digit span" in P.load_fewshot("reverse")


def test_reverse_fewshot_demos_show_reversed_responses():
    """A reverse demo whose response repeats the presented order would teach the
    wrong task. Check the first answer digit is the last presented digit."""
    import re

    text = P.load_fewshot("reverse")
    blocks = re.findall(
        r"The digits are the following: \[([0-9, ]+)\].*?Human response:\s*\n((?:press <<\d>>\.\n?)+)",
        text,
        flags=re.S,
    )
    assert blocks, "no demo blocks found in fewshot_reverse.txt"
    for presented, response in blocks:
        digits = [int(x) for x in presented.split(",")]
        answer = [int(m) for m in re.findall(r"press <<(\d)>>\.", response)]
        assert answer[0] == digits[-1], (
            f"reverse demo starts with {answer[0]}, expected last presented digit "
            f"{digits[-1]} -- demo shows forward order"
        )


def _demo_blocks(direction):
    """(presented digits, response digits, outcome) for each demo block."""
    import re

    text = P.load_fewshot(direction)
    return [
        (
            [int(x) for x in presented.split(",")],
            [int(m) for m in re.findall(r"press <<(\d)>>\.", response)],
            outcome,
        )
        for presented, response, outcome in re.findall(
            r"The digits are the following: \[([0-9, ]+)\].*?Human response:\s*\n"
            r"((?:press <<\d>>\.\n?)+)\s*Outcome: (Correct|Wrong)",
            text,
            flags=re.S,
        )
    ]


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_demo_outcome_labels_match_the_demo_responses(direction):
    """A block labelled Correct must actually be correct, and Wrong actually wrong.

    A mislabelled demo teaches the model that an error is a success (or the reverse),
    which is exactly the signal this fine-tune is supposed to learn.
    """
    blocks = _demo_blocks(direction)
    assert blocks, f"no demo blocks parsed from fewshot_{direction}.txt"
    for presented, response, outcome in blocks:
        correct = presented if direction == "forward" else list(reversed(presented))
        if outcome == "Correct":
            assert response == correct, f"labelled Correct but {response} != {correct}"
        else:
            assert response != correct, f"labelled Wrong but response is correct: {response}"


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_demos_include_both_outcomes(direction):
    """Half the training corpus is human errors; the demos must show both cases."""
    outcomes = {o for _, _, o in _demo_blocks(direction)}
    assert outcomes == {"Correct", "Wrong"}


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_every_demo_block_carries_a_reasoning_block(direction):
    import re

    text = P.load_fewshot(direction)
    n_trials = len(re.findall(r"^Trial \d+$", text, flags=re.M))
    assert n_trials == text.count(P.REASONING_OPEN) == text.count(P.REASONING_CLOSE)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_demos_are_ready_to_sample(direction):
    """No leftover scaffold text, and no leaked hint phrasing in the demo rationales."""
    import re

    text = P.load_fewshot(direction)
    assert "PLACEHOLDER" not in text
    for reasoning in re.findall(r"<reasoning>(.*?)</reasoning>", text, flags=re.S):
        assert reasoning.strip()
        assert P.hint_leak(reasoning) is None, f"demo rationale leaks the hint: {reasoning[:60]}"


def test_placeholder_gate_uses_the_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "load_fewshot", lambda d: "PLACEHOLDER — write me")
    assert P.fewshot_has_placeholders("forward") is True
    monkeypatch.setattr(P, "load_fewshot", lambda d: "a real rationale")
    assert P.fewshot_has_placeholders("forward") is False


# --- parsing ---------------------------------------------------------------
def test_digits_inside_reasoning_do_not_leak_into_the_answer():
    raw = (
        "<reasoning>they would press 7 first, then 3, then lose the rest</reasoning>\n"
        "press <<5>>.\npress <<4>>."
    )
    reasoning, digits, errs = P.parse_rationale(raw)
    assert digits == [5, 4]
    assert "7" in reasoning and "3" in reasoning
    assert errs == []


def test_missing_reasoning_block_is_rejected():
    reasoning, digits, errs = P.parse_rationale("press <<3>>.")
    assert reasoning is None and digits == [] and errs == ["no_reasoning_block"]


def test_empty_reasoning_and_missing_digits_are_flagged():
    _, digits, errs = P.parse_rationale("<reasoning></reasoning>\nno answer here")
    assert digits == []
    assert "empty_reasoning" in errs and "no_digits_parsed" in errs


def test_parse_handles_none_and_empty():
    assert P.parse_rationale(None)[2] == ["empty_response"]
    assert P.parse_rationale("")[2] == ["no_reasoning_block"]


def test_bare_press_format_still_parses():
    # bench's PRESS_RE is permissive about << >>; we reuse it unchanged.
    _, digits, _ = P.parse_rationale("<reasoning>x</reasoning>\npress 5.\npress 4.")
    assert digits == [5, 4]


# --- the acceptance comparison (int list vs int list) ----------------------
@pytest.mark.parametrize("fixture_name", ["fwd_success", "fwd_fail", "rev_success", "rev_fail"])
def test_human_response_rendered_back_is_accepted(fixture_name, request):
    """A human's own response, rendered as a completion, must parse back to y_i.

    Both directions, success and fail. If this fails, nothing is ever accepted.
    """
    trial = request.getfixturevalue(fixture_name)
    completion = P.build_completion("because.", trial.user_digits)
    reasoning, digits, errs = P.parse_rationale(completion)
    assert errs == []
    assert reasoning == "because."
    assert digits == trial.user_digits
    assert digits is not None and digits == trial.user_digits


def test_reverse_target_is_not_re_reversed(rev_fail):
    # Presented 2 7 4 5; correct 5 4 7 2; human typed 5 4 7.
    assert rev_fail.digits == [2, 7, 4, 5]
    assert rev_fail.expected_digits == [5, 4, 7, 2]
    assert rev_fail.user_digits == [5, 4, 7]
    assert P.render_response(rev_fail.user_digits).splitlines()[0] == "press <<5>>."


def test_completion_round_trip_is_canonical():
    c = P.build_completion("  spaced  ", [1, 2])
    assert c == "<reasoning>\nspaced\n</reasoning>\npress <<1>>.\npress <<2>>."


# --- hint leak -------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "The human actually responded with 5 4 7 so they must have...",
        "Given the human's response is 5 4 7, they clearly dropped the last digit.",
        "Since we are told the answer, the middle blurred.",
        "Following the hint, they lost the tail.",
        "The provided answer shows a truncation.",
    ],
)
def test_hint_leaks_are_detected(text):
    assert P.hint_leak(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Four digits is past this person's comfortable span, so the first item decays.",
        "They rehearse the opening pair and lose the tail under the reversal load.",
    ],
)
def test_clean_rationales_are_not_flagged(text):
    assert P.hint_leak(text) is None


def test_prompt_additions_are_recorded_for_provenance():
    adds = P.prompt_additions()
    assert set(adds) == {
        "task_line_forward",
        "task_line_reverse",
        "reasoning_preamble",
        "hint_template",
    }
    assert all(isinstance(v, str) and v for v in adds.values())
