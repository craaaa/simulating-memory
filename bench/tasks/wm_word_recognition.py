from __future__ import annotations
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS
from .word_recognition import (
    FORMAT_RULES,
    HUMAN_PROMPT,
    MAX_ERRORS_BEFORE_STOP,
    TASK_DESC,
    load_words,
    generate_one_game,
    parse_responses,
    save_score_by_condition_figure,
    score_game,
    summarize_game_rows,
)
from .wm_prompt_parts import (
    CONDITIONS,
    SUMMARIZER_CONDITIONS,
    summarizer_recall_prompt,
    summarizer_system_prompt,
    wm_system_prompts,
)

TASK_NAME = "wm_word_recognition"
TASK_NAME_SUM = "sum_word_recognition"

WM_SYSTEM_PROMPTS = wm_system_prompts(
    task_prompt=TASK_DESC,
    human_task_prompt=HUMAN_PROMPT,
)

SUM_SYSTEM_PROMPT = summarizer_system_prompt(task_prompt=TASK_DESC)

SUM_RECALL_PROMPT = summarizer_recall_prompt(
    task_prompt=TASK_DESC,
    recall_instructions=(
        'classify each word below as "Old" (appeared earlier in the original list) or '
        '"New" (first appearance at that position in the list).\n\n{trials_text}'
    ),
    format_rules=FORMAT_RULES,
)


# Turn prompts. Continuous recognition has no study phase: a word is "Old" if it appeared
# earlier in the SAME sequence, so study and test are the same stream. The task therefore
# runs as one turn pair per word -- an ANSWER turn (store + this word, tools OFF) then an
# ENCODE turn (store + this word, tools ON) -- and there is no `encode()` call at all.
#
# What this replaces: `encode()` was handed all 100 trial lines at once, and then the recall
# prompt was handed the same 100 lines again, so Old/New was decidable from the prompt text
# without consulting the store. Measured under that presentation: 36 of 50 participants
# scored >= 0.98 while humans average 0.315 proportion correct, and the 3-strike stop never
# fired for 38 of 50 (mean 1.2 errors in 100 trials; median third-error position 101, i.e.
# censored, against a human median of 32 trials completed).
#
# Turn order is ANSWER-then-ENCODE, which is deliberately the OPPOSITE of `wm_nback`'s
# reason for the same order and the opposite of `wm_variable_mapping`'s order. Here the
# judgement is about the word in front of the participant: hiding it would test writing, not
# recognition, and there is nothing in the store that answering could destroy. Answering
# first also means a tool call can never crowd out the reply, the failure mode that cost the
# stage-1 n-back baseline 3.15 of 14 trials. `wm_variable_mapping` asks about STORED content
# ("Where does X live?"), so it must encode first. Do not "fix" these three into agreement.
#
# The trial index is stated explicitly because `step()` clears the transcript at every turn
# boundary, so the agent has no other cue to where it is in the stream. A human watching
# words appear has that cue for free.
ANSWER_PROMPT = """\
Your working memory currently contains:
{wm_contents}

Original task instructions:
{task_prompt}

Words presented so far in this list, including this one: {pos}.

trial {pos}: {word}

Has this word appeared earlier in this list?

Output ONLY one of: old, new.
No extra text."""

ENCODE_PROMPT = """\
Your working memory currently contains:
{wm_contents}

New word (trial {pos}):
{word}

Update your memory as needed."""

OLD_NEW_RE = re.compile(r"\b(old|new)\b", re.IGNORECASE)


def _parse_old_new(text: str) -> Optional[str]:
    """Extract Old / New from a single-turn reply, or None if the reply is not readable.

    A reply naming BOTH words returns None rather than the first one. Taking the first match
    would read "not old, it is new" as Old, and a negated form is the failure mode a
    bare-first-match regex gets wrong. An unreadable reply is scored as neither correct nor an
    error by `score_game`, and `check_predictions`' coverage guard is what surfaces a run
    where this happens often.

    `word_recognition.parse_responses` parses the bulk `trial N: old` form and is kept for
    the summarizer arm, which still answers all trials in one reply.
    """
    found = {m.group(1).capitalize() for m in OLD_NEW_RE.finditer(text or "")}
    return found.pop() if len(found) == 1 else None


def run_recognition_stream(
    llm: LLM,
    trials: List[Dict[str, Any]],
    cond_id: str,
    temperature: float,
    debug: bool,
    participant_id: Any = None,
) -> Dict[str, Any]:
    """Present one word per turn, answer before storing, stop at the third error.

    The stop replicates the human protocol, which ends a participant's session on their
    third mistake: `trialsCompleted - correctResponses == 3` for 53 of 53 human records, so
    the human score is the number of words survived rather than an accuracy. `score_game`
    already applies the same rule analysis-side; stopping the loop here means the trials a
    human would never have seen are also never presented to the model.

    Returns resp_map (trial_index -> Old/New), the step log, and the turn-kind indices.
    """
    agent = WorkingMemoryAgent(
        llm=llm,
        condition_id=cond_id,
        temperature=temperature,
        debug=debug,
        system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
        participant_id=participant_id,
    )

    resp_map: Dict[int, str] = {}
    answer_steps: List[int] = []
    encode_steps: List[int] = []
    answer_step_by_position: Dict[int, int] = {}
    errors = 0
    stopped_at: Optional[int] = None

    for t in trials:
        pos = t["trial_index"]
        word = t["word"]

        reply = agent.step(
            ANSWER_PROMPT.format(
                wm_contents=agent.wm.to_recall_text(),
                task_prompt=TASK_DESC.strip(),
                pos=pos,
                word=word,
            ),
            allow_tools=False,
            max_tokens=512,
        )
        answer_steps.append(len(agent.get_step_log()) - 1)
        answer_step_by_position[pos] = answer_steps[-1]

        parsed = _parse_old_new(reply)
        if parsed is not None:
            resp_map[pos] = parsed
            expected = "Old" if t["is_old"] else "New"
            if parsed != expected:
                errors += 1

        agent.step(
            ENCODE_PROMPT.format(
                wm_contents=agent.wm.to_recall_text(),
                pos=pos,
                word=word,
            ),
            allow_tools=True,
            max_tokens=512,
        )
        encode_steps.append(len(agent.get_step_log()) - 1)

        if debug:
            print(f"  trial {pos}: {word} -> {parsed} (errors {errors})")

        # An unparsed reply is not an error, matching `score_game`, which records it as
        # `correct: None` and neither counts it nor stops on it.
        if errors >= MAX_ERRORS_BEFORE_STOP:
            stopped_at = pos
            break

    return {
        "resp_map": resp_map,
        "step_log": agent.get_step_log(),
        "final_kv": agent.wm.store,
        "answer_steps": answer_steps,
        "encode_steps": encode_steps,
        "answer_step_by_position": answer_step_by_position,
        "trials_presented": len(answer_steps),
        "stopped_at_third_error": stopped_at,
        "slot_utilization": len(agent.wm.store) / MAX_KEYS,
    }


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    words_json_path: str = "data/words.json",
    n_repeat: int = 1,
    max_trials_per_game: int = 100,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    words = load_words(words_json_path)
    n_participants = max(1, int(n_repeat))
    rng = random.Random(stimuli_seed)

    # Generate one game per participant (shared across conditions)
    games = []
    for _ in range(n_participants):
        trials = generate_one_game(words, rng, max_trials_per_game)
        games.append(trials)

    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in CONDITIONS:
        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            trials = games[pid - 1]

            if debug:
                print(f"\n{'='*50}")
                print(f"{TASK_NAME} | {cond_id} | p{pid} | {len(trials)} trials")
                print(f"{'='*50}")

            # One row per participant, so pid alone is the scored unit.
            stream = run_recognition_stream(
                llm, trials, cond_id, temperature, debug, participant_id=f"p{pid}"
            )

            resp_map = stream["resp_map"]
            scored = score_game(trials, resp_map)
            scored["slot_utilization"] = stream["slot_utilization"]

            if debug:
                print(f"  score: {scored['score']}")
                print()

            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{pid}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "final_kv": stream["final_kv"],
                "gold_trials": trials,
                # One turn pair per word since the encode/answer split, so a positional
                # rule over `step_log` is wrong. These carry the mapping explicitly:
                # `answer_step_by_position[i]` is the turn whose reply scored trial i.
                "step_log": stream["step_log"],
                "answer_steps": stream["answer_steps"],
                "encode_steps": stream["encode_steps"],
                "answer_step_by_position": stream["answer_step_by_position"],
                "trials_presented": stream["trials_presented"],
                "stopped_at_third_error": stream["stopped_at_third_error"],
                "metrics": {
                    "score": scored["score"],
                    "first_error_at": scored["first_error_at"],
                    "n_trials": scored["n_trials"],
                    "slot_utilization": scored["slot_utilization"],
                },
                "per_trial": scored["per_trial"],
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = summarize_game_rows(rows, cond_id, CONDITIONS[cond_id]["name"])
        cond_summary["metrics"]["slot_utilization"] = (
            sum(r["metrics"]["slot_utilization"] for r in rows) / len(rows) if rows else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_score_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {"task": TASK_NAME, "n_participants": n_participants, "conditions": cond_summaries}
    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    words_json_path: str = "data/words.json",
    n_repeat: int = 1,
    max_trials_per_game: int = 100,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    words = load_words(words_json_path)
    n_participants = max(1, int(n_repeat))
    rng = random.Random(stimuli_seed)

    games = []
    for _ in range(n_participants):
        trials = generate_one_game(words, rng, max_trials_per_game)
        games.append(trials)

    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in SUMMARIZER_CONDITIONS:
        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            trials = games[pid - 1]
            word_list_text = "\n".join(f"{t['trial_index']}: {t['word']}" for t in trials)
            agent = SummarizerAgent(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                system_prompt_override=summarizer_system_prompt(
                    TASK_DESC, condition_id=cond_id
                ),
                participant_id=f"p{pid}",
            )

            if debug:
                print(f"\n{'='*50}")
                print(f"{TASK_NAME_SUM} | {cond_id} | p{pid} | {len(trials)} trials")
                print(f"{'='*50}")

            encoding_log = agent.encode(word_list_text)

            trials_text = "\n".join(f"trial {t['trial_index']}: {t['word']}" for t in trials)
            recall_prompt = SUM_RECALL_PROMPT.format(
                trials_text=trials_text,
                summary="{summary}",
            )
            recall_raw = agent.recall(recall_prompt=recall_prompt, max_tokens=1024)

            resp_map = parse_responses(recall_raw)
            scored = score_game(trials, resp_map)
            scored["summary_length_words"] = agent.summary_length_words()

            if debug:
                print(f"  score: {scored['score']}")
                print()

            row = {
                "id": f"{TASK_NAME_SUM}:{cond_id}:p{pid}",
                "condition_id": cond_id,
                "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "encoding_log": encoding_log,
                "final_summary": agent.summary,
                "recall_raw": recall_raw,
                "gold_trials": trials,
                "metrics": {
                    "score": scored["score"],
                    "first_error_at": scored["first_error_at"],
                    "n_trials": scored["n_trials"],
                    "summary_length_words": scored["summary_length_words"],
                },
                "per_trial": scored["per_trial"],
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = summarize_game_rows(
            rows, cond_id, SUMMARIZER_CONDITIONS[cond_id]["name"]
        )
        cond_summary["metrics"]["summary_length_words"] = (
            sum(r["metrics"]["summary_length_words"] for r in rows) / len(rows) if rows else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_score_by_condition_figure(out_dir, TASK_NAME_SUM, cond_summaries)

    summary = {"task": TASK_NAME_SUM, "n_participants": n_participants, "conditions": cond_summaries}
    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
