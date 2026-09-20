"""Listening prompts: what each variant may and may not contain, and the parser.

The load-bearing test here is that the TRAINING prompt carries neither the hint nor
the few-shot demos. STaR trains "as if the model had come up with the rationale without
the hint", and a training prompt that differs from the eval prompt optimizes something
the model is never asked at inference.
"""
from __future__ import annotations

import re

import pytest

from application.listening_qa.prompting import CONDITIONS
from rationales.listening import prompting as lp
from rationales.listening import select as lsel
from rationales.listening.data import RESPONSES_CSV, load_items
from rationales.prompting import REASONING_CLOSE, REASONING_OPEN
from rationales.tasks.listening_qa import ListeningQATask

from listening_fixtures import make_item

needs_real_data = pytest.mark.skipif(
    not RESPONSES_CSV.is_file(),
    reason=f"{RESPONSES_CSV} is a gitignored analysis artifact; run the ingest step",
)


@pytest.fixture
def item():
    # A human who missed one true option: the case the rationale has to explain.
    return make_item(gold=(1, 2, 3), endorsed=(1, 3))


# ---------------------------------------------------------------------------
# What each variant contains
# ---------------------------------------------------------------------------
def test_training_prompt_has_neither_hint_nor_fewshot(item):
    training = lp.build_sample_prompt(item, fewshot=False)
    assert "actually selected" not in training
    assert "Bracklin" not in training  # a few-shot demo passage
    assert REASONING_OPEN in training and REASONING_CLOSE in training


def test_generation_prompt_carries_the_demos_but_never_the_hint(item):
    gen = lp.build_sample_prompt(item, fewshot=True)
    assert "Bracklin" in gen
    assert "actually selected" not in gen


def test_rationalize_prompt_states_the_humans_selection_as_text(item):
    rat = lp.build_rationalize_prompt(item, fewshot=False)
    assert "The human actually selected:" in rat
    # Numbers alone would be a string to copy rather than something to explain.
    assert "1) It stands on a granite stack" in rat
    assert "3) Its lamp turns on a bath of mercury" in rat
    assert "2) It was lit in 1904" not in rat.split("actually selected:")[1]


def test_prompt_reuses_the_listening_tasks_c3_prefix_verbatim(item):
    """Comparability with the existing full-grid runs depends on this exact string."""
    assert CONDITIONS["C3"]["prompt_prefix"] in lp.build_sample_prompt(item, fewshot=False)


def test_task_line_names_the_single_question_unit(item):
    prompt = lp.build_sample_prompt(item, fewshot=False)
    assert "Question to predict" in prompt
    assert "do not answer them again" in prompt


def _sib(endorsed, gold):
    from rationales.listening.data import SiblingAnswer

    return SiblingAnswer(
        "QV02", "q", {1: "a", 2: "b", 3: "c"}, endorsed=list(endorsed), gold=list(gold)
    )


def test_verdict_separates_missing_a_true_option_from_taking_a_false_one():
    """"wrong" collapses two different memory failures. Someone who consistently
    under-selects is not the same listener as someone who reaches for foils, and
    telling them apart is the whole job of the sibling block."""
    assert _sib([1], [1]).verdict == "ok"
    assert _sib([1], [1, 2]).verdict == "-1"          # omission only
    assert _sib([1, 2], [1]).verdict == "+1"          # commission only
    assert _sib([1, 3], [1, 2]).verdict == "-1+1"     # both
    assert _sib([], [1, 2]).verdict == "-2"
    assert _sib([3], [1, 2]).verdict == "-2+1"


def test_verdict_counts_are_the_actual_option_sets():
    s = _sib([1, 3], [1, 2])
    assert s.missed == [2] and s.false_positives == [3]
    assert not s.correct


def test_sibling_correctness_is_exact_set_match():
    """Same definition as ListeningItem.correct, so the two readings of "wrong" cannot
    drift apart. Selecting one of two true options is wrong, not partially right."""
    assert _sib([1], [1]).correct
    assert not _sib([1], [1, 2]).correct


def test_the_legend_explaining_the_codes_is_in_the_prompt(item):
    prompt = lp.build_sample_prompt(item, fewshot=False)
    assert "-n = missed n true options" in prompt
    assert "+n = took n false ones" in prompt


def test_sibling_questions_drop_the_boilerplate_but_the_target_keeps_it(item):
    """The trim is what pays for the verdict codes. It applies only to the summary of
    questions already answered -- the question actually being asked stays verbatim."""
    prompt = lp.build_sample_prompt(item, fewshot=False)
    head, _, target = prompt.partition(lp._TARGET_HEADER.rstrip("\n"))
    assert "Select all that apply." not in head.split(lp._SIBLING_HEADER.rstrip("\n"))[1]
    assert "Select all that apply." in target
    assert lp._trim_boilerplate("Which are true? Select all that apply.") == "Which are true?"
    assert lp._trim_boilerplate("Which describes it?") == "Which describes it?"


def test_the_targets_own_verdict_is_never_stated(item):
    """The codes cover the siblings only. Saying how the target answer scored would
    hand over most of y_i -- "-0+0" on a single-answer question hands over all of it."""
    for prompt in (
        lp.build_sample_prompt(item, fewshot=False),
        lp.build_rationalize_prompt(item, fewshot=False),
    ):
        target = prompt.split(lp._TARGET_HEADER.rstrip("\n"))[1]
        assert "[ok]" not in target
        assert not re.search(r"\[-\d|\[\+\d", target)


def test_sibling_answers_appear_as_context_in_every_variant(item):
    for prompt in (
        lp.build_sample_prompt(item, fewshot=False),
        lp.build_sample_prompt(item, fewshot=True),
        lp.build_rationalize_prompt(item, fewshot=False),
    ):
        assert "The same human answered these other questions" in prompt
        assert "By helicopter" in prompt


def test_sibling_context_can_be_turned_off(item):
    """The control for the 2x2 combination questions, where the siblings can hand the
    model the answer outright."""
    with_ctx = lp.build_sample_prompt(item, fewshot=False, sibling_context=True)
    without = lp.build_sample_prompt(item, fewshot=False, sibling_context=False)
    assert "By helicopter" in with_ctx
    assert "By helicopter" not in without
    # Only the sibling block differs; the question and the rules are untouched.
    assert item.question in without and "Answer: 1,3" in without


def test_every_option_is_rendered_with_its_number(item):
    prompt = lp.build_sample_prompt(item, fewshot=False)
    for n, text in item.options.items():
        assert f"{n}) {text}" in prompt


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parses_reasoning_and_the_answer_line():
    raw = f"{REASONING_OPEN}\nthey kept the gist.\n{REASONING_CLOSE}\nAnswer: 1,3"
    reasoning, options, errors = lp.parse_rationale(raw)
    assert reasoning == "they kept the gist."
    assert options == [1, 3]
    assert errors == []


def test_option_numbers_inside_the_reasoning_do_not_reach_the_answer():
    """The reason the completion is split on </reasoning> at all: rationales discuss
    options by number, and parsing the whole string would pull those in."""
    raw = (
        f"{REASONING_OPEN}\nshe would take 2 and 4 but miss 5 entirely.\n"
        f"{REASONING_CLOSE}\nAnswer: 1"
    )
    _, options, _ = lp.parse_rationale(raw)
    assert options == [1]


def test_answer_order_and_duplicates_are_normalized():
    raw = f"{REASONING_OPEN}\nr\n{REASONING_CLOSE}\nAnswer: 3,1,3"
    _, options, errors = lp.parse_rationale(raw)
    assert options == [1, 3]
    assert errors == []


def test_missing_reasoning_block_is_rejected_outright():
    reasoning, options, errors = lp.parse_rationale("Answer: 1,3")
    assert reasoning is None and options == []
    assert errors == ["no_reasoning_block"]


def test_prose_answer_is_distinguished_from_a_missing_answer():
    prose = f"{REASONING_OPEN}\nr\n{REASONING_CLOSE}\nAnswer: none of the above"
    assert "answer_not_numeric" in lp.parse_rationale(prose)[2]
    cut_off = f"{REASONING_OPEN}\nr\n{REASONING_CLOSE}\n"
    assert "no_answer_line" in lp.parse_rationale(cut_off)[2]


def test_out_of_range_options_are_dropped_and_flagged():
    raw = f"{REASONING_OPEN}\nr\n{REASONING_CLOSE}\nAnswer: 2,9"
    _, options, errors = lp.parse_rationale(raw)
    assert options == [2]
    assert "option_out_of_range" in errors


@pytest.mark.parametrize(
    "raw",
    [
        # Exactly what Tinker returned, verbatim from a live sample_pool.jsonl.
        "<reasoning>\nthey kept the gist.\n</reasoning>\nAnswer: 1,3<|im_end|>",
        "<reasoning>\nthey kept the gist.\n</reasoning>\nAnswer: 1,3<|endoftext|>",
        "<reasoning>\nthey kept the gist.\n</reasoning>\nAnswer: 1,3<|im_end|>\n",
    ],
)
def test_chat_template_control_tokens_do_not_break_the_answer_line(raw):
    """Regression for a full round of wasted sampling. Tinker returns control tokens
    inside the completion with no separating newline, and the answer pattern is anchored
    to end of line, so "Answer: 1,3<|im_end|>" parsed as answer_not_numeric -- 6680
    well-formed completions, zero accepted. OpenRouter strips these server-side, so the
    probe could not have caught it."""
    reasoning, options, errors = lp.parse_rationale(raw)
    assert reasoning == "they kept the gist."
    assert options == [1, 3]
    assert errors == []


def test_control_tokens_are_stripped_from_the_reasoning_too():
    raw = "<reasoning>\nthey kept the gist.<|im_end|>\n</reasoning>\nAnswer: 1"
    reasoning, options, _ = lp.parse_rationale(raw)
    assert "<|" not in reasoning
    assert options == [1]


def test_empty_response_is_not_a_parse_success():
    assert lp.parse_rationale("")[2] == ["no_reasoning_block"]
    assert lp.parse_rationale(None)[2] == ["empty_response"]


def test_build_completion_round_trips_through_the_parser():
    completion = lp.build_completion("they kept the gist.", [3, 1])
    reasoning, options, errors = lp.parse_rationale(completion)
    assert reasoning == "they kept the gist."
    assert options == [1, 3]
    assert errors == []


# ---------------------------------------------------------------------------
# Accept and leak
# ---------------------------------------------------------------------------
def test_accepts_compares_sets_not_lists(item):
    task = ListeningQATask()
    assert task.accepts(item, [3, 1])
    assert not task.accepts(item, [1])
    assert not task.accepts(item, [1, 2, 3])


def test_accepts_targets_the_human_not_the_gold_answer(item):
    """The whole deviation from STaR in one assertion."""
    task = ListeningQATask()
    assert item.gold == [1, 2, 3] and item.endorsed == [1, 3]
    assert task.accepts(item, item.endorsed)
    assert not task.accepts(item, item.gold)


@pytest.mark.parametrize(
    "reasoning",
    [
        "The human actually selected 1 and 3, so they must have remembered those.",
        "Given the response was 1 and 3, they clearly kept the opening.",
        "We are given their answer, so the gist survived.",
    ],
)
def test_hint_leak_catches_a_rationale_that_quotes_being_told(reasoning):
    assert lp.hint_leak(reasoning)


def test_hint_leak_passes_an_honest_rationale():
    assert (
        lp.hint_leak("The opening fact was vivid and survived; the date blurred.") is None
    )


def test_shipped_fewshot_demos_are_written():
    """The reasoning is the one hand-written part of these demos. It was left as
    PLACEHOLDER until the researcher wrote it, because it is a claim about why a
    specific person forgot a specific thing and nobody recorded that."""
    assert not lp.fewshot_has_placeholders()


def test_the_placeholder_gate_blocks_a_run_when_reasoning_is_missing(tmp_path, monkeypatch):
    """The gate that held until the demos were written. Kept live against a temp file
    so it still guards the next task's demos."""
    from rationales.config import StarConfig
    from rationales.tasks.listening_qa import ListeningQATask

    path = tmp_path / "fewshot_listening.txt"
    path.write_text(lp.load_fewshot().replace("they", "PLACEHOLDER", 1), encoding="utf-8")
    monkeypatch.setattr(lp, "FEWSHOT_PATH", path)
    with pytest.raises(RuntimeError, match="PLACEHOLDER"):
        ListeningQATask().check_ready(StarConfig(task="listening_qa", use_fewshot=True))


def test_written_demos_clear_the_gate():
    from rationales.config import StarConfig
    from rationales.tasks.listening_qa import ListeningQATask

    ListeningQATask().check_ready(StarConfig(task="listening_qa", use_fewshot=True))


def test_demo_reasoning_obeys_the_sentence_cap_it_sits_next_to():
    """Models imitate the demos over the instruction when the two disagree, so a demo
    that runs longer than the stated cap silently raises it."""
    cap = 4
    assert f"at most {['one','two','three','four','five'][cap-1]} sentences" in lp.format_rules()
    for n, block in enumerate(_reasoning_blocks(), start=1):
        sentences = [s for s in re.split(r"(?<=[.!?]) +", block.strip()) if s]
        assert len(sentences) <= cap, f"demo {n} runs to {len(sentences)} sentences"


def test_demo_reasoning_never_leaks_being_handed_the_answer():
    for n, block in enumerate(_reasoning_blocks(), start=1):
        assert lp.hint_leak(block) is None, f"demo {n} matches a leak pattern"


def _reasoning_blocks():
    return re.findall(r"<reasoning>\n(.*?)\n</reasoning>", lp.load_fewshot(), re.S)


def test_the_gate_is_skipped_when_fewshot_is_off():
    """--no-fewshot is the documented way past it, and is recorded as a deviation."""
    from rationales.config import StarConfig
    from rationales.tasks.listening_qa import ListeningQATask

    ListeningQATask().check_ready(StarConfig(task="listening_qa", use_fewshot=False))


def test_written_reasoning_clears_the_gate(tmp_path, monkeypatch):
    """The other direction, so the gate is not just permanently red."""
    written = lp.load_fewshot().replace("PLACEHOLDER", "they kept the vivid detail")
    path = tmp_path / "fewshot_listening.txt"
    path.write_text(written, encoding="utf-8")
    monkeypatch.setattr(lp, "FEWSHOT_PATH", path)
    assert not lp.fewshot_has_placeholders()


def test_fewshot_demos_do_not_trip_the_hint_leak_filter():
    """The demos are in the GENERATION prompt, which has no hint. If their wording
    matched a leak pattern, the model would copy that phrasing into its rationales and
    the filter would then drop the corpus it had just paid to produce."""
    assert lp.hint_leak(lp.load_fewshot()) is None


def test_fewshot_demos_show_both_correct_and_incorrect_humans():
    demos = lp.load_fewshot()
    assert "Outcome: Correct" in demos
    assert "Outcome: Wrong" in demos


def test_fewshot_demos_cover_the_failure_modes_the_corpus_has_to_learn():
    """A demo set of only correct answers would teach the model to answer well, which
    is the opposite of the objective."""
    demos = lp.load_fewshot()
    assert "a correct option (1) was missed" in demos
    assert "appeared nowhere in the passage" in demos
    assert "Answer: 5" in demos  # none-of-the-above as a real human strategy


@needs_real_data
def test_every_fewshot_demo_item_is_in_the_train_split():
    """The demos quote four real participants' actual selections. If one of them were
    held out, the generation prompt would be showing the model a person it is about to
    be evaluated on."""
    items, _ = load_items()
    sel = lsel.select(items, seed=42, eval_frac=0.15)
    eval_ids = {it.item_id for it in sel.eval}
    train_ids = {it.item_id for it in sel.train}
    for item_id in lp.FEWSHOT_SOURCE_ITEMS:
        assert item_id in train_ids
        assert item_id not in eval_ids


@needs_real_data
def test_fewshot_demo_passages_are_verbatim_what_those_humans_heard():
    """Each topic has four passages, one per encoding level, and the level is the
    manipulation the whole study turns on. A demo that paraphrases or abridges the
    passage is demonstrating recall from a stimulus nobody was given -- and the
    abridgement bites hardest exactly where the demo matters most, since shortening a
    distractor passage removes the interference the rationale is appealing to.
    """
    items, _ = load_items()
    index = {it.item_id: it for it in items}
    demos = lp.load_fewshot()
    for item_id in lp.FEWSHOT_SOURCE_ITEMS:
        item = index[item_id]
        assert item.passage in demos, (
            f"{item_id}: the demo passage is not the verbatim {item.level} passage"
        )


@needs_real_data
def test_fewshot_demo_questions_and_options_are_verbatim_too():
    items, _ = load_items()
    index = {it.item_id: it for it in items}
    demos = lp.load_fewshot()
    for item_id in lp.FEWSHOT_SOURCE_ITEMS:
        item = index[item_id]
        assert item.question in demos, f"{item_id}: question text altered"
        assert lp.render_options(item.options) in demos, f"{item_id}: options altered"


@needs_real_data
def test_every_demo_shows_that_participants_sibling_answers():
    """The sibling block is not decoration, it is the only thing that individuates a
    participant. Two demos share a passage and a question and differ only in their
    siblings; strip those and the pair becomes one input with two different outputs,
    which teaches the model that the answer is random rather than person-dependent."""
    items, _ = load_items()
    index = {it.item_id: it for it in items}
    demos = lp.load_fewshot()
    for item_id in lp.FEWSHOT_SOURCE_ITEMS:
        item = index[item_id]
        assert len(item.other_answers) == 4
        block = lp._sibling_block(item.other_answers)
        assert block in demos, f"{item_id}: sibling answers missing or altered"


@needs_real_data
def test_the_two_demos_that_share_a_question_are_told_apart_by_their_siblings():
    items, _ = load_items()
    index = {it.item_id: it for it in items}
    a, b = (index[i] for i in lp.FEWSHOT_SOURCE_ITEMS[1:3])
    assert a.respondent_id != b.respondent_id
    assert (a.topic, a.level, a.question_id) == (b.topic, b.level, b.question_id)
    assert a.endorsed != b.endorsed
    # The inputs are identical except for the sibling block, so it has to differ.
    assert lp._sibling_block(a.other_answers) != lp._sibling_block(b.other_answers)


def test_demos_are_laid_out_like_a_real_prompt():
    """Few-shot transfer works by shape. A demo that omits the sibling block or the
    target header is showing the model a different task than it is about to be given."""
    demos = lp.load_fewshot()
    assert demos.count("Transcript:") == len(lp.FEWSHOT_SOURCE_ITEMS)
    assert demos.count(lp._TARGET_HEADER.rstrip("\n")) == len(lp.FEWSHOT_SOURCE_ITEMS)
    assert demos.count(lp._SIBLING_HEADER.rstrip("\n")) == len(lp.FEWSHOT_SOURCE_ITEMS)


@needs_real_data
def test_fewshot_demo_answers_match_what_those_humans_actually_selected():
    """Guards against the demos drifting into invented behavior during an edit."""
    items, _ = load_items()
    index = {it.item_id: it for it in items}
    demos = lp.load_fewshot()
    for item_id in lp.FEWSHOT_SOURCE_ITEMS:
        expected = lp.render_answer(index[item_id].endorsed)
        assert expected in demos, f"{item_id} should be demoed as {expected!r}"


def test_prompt_additions_record_every_string_this_module_adds():
    additions = lp.prompt_additions()
    assert additions["task_line"] == lp.TASK_LINE
    assert "one (participant, topic, question)" in additions["unit_of_prediction"]
