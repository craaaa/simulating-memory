"""Candidate `serial_recognition`: the recognition test is presented one trial
at a time, so the studied list is no longer visible while it is being judged.

This candidate does not try to improve a score. It converts a task that does not
measure the memory module into one that does. See MANIFEST.md beside this file
for the full argument, the cost estimate, the A2 re-interpretation and the
pre-registered predictions; this docstring states the mechanism and the contract.

THE DEFECT
----------
`bench/tasks/wm_word_recognition.py:110` builds

    trials_text = "\\n".join(f"trial {t['trial_index']}: {t['word']}" for t in trials)

and formats it into `RECALL_PROMPT`. All ~100 recognition trials are therefore
present verbatim in the single recall prompt, in order. Old/New is decidable by
reading the prompt: a word is "old" iff it appears earlier in the block that the
prompt itself displays. The prompt says "Based ONLY on the above contents",
meaning the four-slot store, and 36 of 50 participants ignore that and score
>= 0.98 while holding a mean of 3.4 keys.

In the human protocol the words appear ONE AT A TIME and each judgement is
committed before the next word exists. There is no list to consult. That is the
defining property of a continuous-recognition test (Shepard & Teghtsoonian 1961;
Hockley 1982), and the bench's recall prompt removes it.

THE MECHANISM
-------------
`recall()` is a harness method, so the harness owns the context the model has at
test time. This candidate overrides `recall()` and nothing else. When the recall
prompt carries a trial-list block -- detected structurally, see `_find_trial_run`
-- the harness issues **one LLM call per trial**, each showing the four-slot store
and exactly one trial line, and stitches the per-trial replies back into the same
`trial N: old/new` text the task's parser already expects.

Because the base `recall()` answers with `self.llm.generate(...)` -- a single
fresh prompt, no `self._messages`, no tools -- each of these calls is
**stateless**. Nothing carries between them. So the mechanism is stronger than
serial presentation: each trial is judged independently from the store alone,
with no look-back at trials already judged and no look-ahead to trials not yet
reached. In particular this does not depend on, and cannot be undone by, the
`step()` conversation-history leak that a separate proposer owns: `step()` is
never called here, and `reset_messages()` is never called either, so that surface
is untouched.

Block size is 1 by design, not by convenience. Measured on this run's own
stimuli, the fraction of "old" trials whose first occurrence falls inside the
same visible block is

    k = 1   0.000        k = 5    0.101        k = 20   0.352
    k = 2   0.027        k = 10   0.202        k = 100  1.000  (the baseline)

so any k > 1 leaves a leak in proportion to k. Since the leak is the entire
object of the intervention, k = 1 is the only defensible choice.

WHAT IS NOT CHANGED
-------------------
`MAX_KEYS` stays 4. `WorkingMemory` is untouched -- no change to `write_key`, to
the overflow rule, to primacy, to value formatting or to read-out order, so this
candidate does not collide with the overflow/primacy proposal. `TOOLS`,
`CONDITION_PROMPTS` and `WM_SYSTEM_PROMPTS` are the baseline's. `encode()`,
`step()` and the tool-call cap are the baseline's. The parent is `baseline`, not
`displacement`: `displacement` failed its per-task floor on semantic_story_recall
and its word_recognition gain was a measurement artifact, and building on it
would put two changed method bodies between this run and the control.

THE ONE PROMPT CHANGE, AND WHY IT IS FORCED
-------------------------------------------
A single line is inserted immediately before the trial block:

    You are being asked about one trial at a time. Judge ONLY trial {i} and
    output exactly one line for it.

Everything else -- the system prompt, the store rendering, the task
instructions, the "Based ONLY on the above contents" sentence, `FORMAT_RULES`
and its worked example -- is byte-identical to the baseline's.

It is forced by the ablation, which is the only comparison that makes the result
interpretable. The open arm (`MH_SERIAL_RECOGNITION_MASK=0`) runs the identical
machinery -- same call count, same stitching, same index repair, same
one-judgement-per-call framing -- and differs in exactly one respect: the trial
lines outside the current trial are still visible. Without a directive naming the
target trial the open arm is not well posed, because the model would not know
which of the 100 visible words it is being asked about. Putting the same line in
both arms is what reduces the difference between them to the single bit of
information the candidate is about. It also removes an index-attribution
ambiguity: with the target trial named, the reply is expected to carry the
absolute trial number, so the harness's repair path is a fallback rather than the
normal case.

THE CONTRACT
------------
* Non-recognition recall prompts (digit span, story recall, the MCQ tasks)
  contain no trial-list block and are delegated unchanged to
  `super().recall()`, so those tasks run the baseline code path exactly.
* The returned string is normalised to the format the task's `LINE_RE` already
  matches, one line per trial, all 100 trials, so `parse_responses` and
  `score_game` see what they saw before.
* Diagnostics are appended as `#`-prefixed lines. `parse_responses` skips any
  line that does not match `^trial \\d+: (old|new)$`, so they are inert to
  scoring and survive into `recall_raw` in the JSONL, where the mode, the call
  count and every anomalous reply can be audited after the fact.
* A trial with no parseable judgement emits NO line. `score_game` then records
  `model_response: None` and `correct: None` for it, which counts as neither a
  correct answer nor an error -- so a candidate that degraded by emitting garbage
  would show up as low coverage rather than as an improved error structure.
  `response_coverage` is a pre-registered prediction for this reason.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from bench.core import working_memory as _wm_mod  # noqa: F401  (live MAX_KEYS)
from bench.core.wm_agent import (  # noqa: F401
    CONDITION_PROMPTS,
    SummarizerAgent,
    TOOLS,
)
from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent
from bench.core.working_memory import MAX_KEYS  # noqa: F401

# --------------------------------------------------------------------------- #
# Mode. Default is the candidate (masked); the open arm is the ablation.
# Recorded in MANIFEST["id"] as well as in the per-row diagnostics, because
# `run_candidate.py` copies harness.py into the run dir precisely so that a
# result cannot be reinterpreted later, and an environment variable would
# otherwise defeat that.
# --------------------------------------------------------------------------- #
MASK_OTHER_TRIALS = os.environ.get("MH_SERIAL_RECOGNITION_MASK", "1").strip() != "0"
MODE = "masked" if MASK_OTHER_TRIALS else "open"

BLOCK_SIZE = 1                    # one trial per call; see the leak table above
MIN_TRIALS_TO_SERIALISE = 5       # below this, not a recognition trial list
PER_CALL_MAX_TOKENS = 128         # one formatted line is ~8 tokens
MAX_ANOMALIES_LOGGED = 25

DIRECTIVE = (
    "You are being asked about one trial at a time. Judge ONLY trial {i} and "
    "output exactly one line for it."
)

# A stimulus line is `trial <int>: <word>`. The FORMAT_RULES worked example in
# the same prompt has exactly that shape with "old"/"new" as its value, so the
# value is what separates stimulus from example. No word in data/words.json is
# "old" or "new" (checked: 173 words, none of them).
_TRIAL_LINE = re.compile(r"^trial\s+(\d+):\s*(\S.*?)\s*$", re.IGNORECASE)
_JUDGEMENT = re.compile(r"trial\s+(\d+)\s*:\s*(old|new)\b", re.IGNORECASE)
_BARE_JUDGEMENT = re.compile(r"\b(old|new)\b", re.IGNORECASE)


def _find_trial_run(lines: List[str]) -> Optional[Tuple[int, int, List[Tuple[int, str]]]]:
    """Locate the stimulus trial block: the longest contiguous run of
    `trial <int>: <word>` lines with strictly increasing indices whose values
    are not "old"/"new".

    Returns ``(start, end, [(index, word), ...])`` with ``lines[start:end]``
    being the run, or ``None`` if there is no such run of at least
    ``MIN_TRIALS_TO_SERIALISE`` lines. Structural detection, so no task needs to
    declare anything and no other task's recall prompt can be caught by
    accident.

    Two guards keep the `FORMAT_RULES` worked example out of the result: its
    values are "old"/"new", and it is only three lines long. Unit-checked
    offline on crafted line lists, including non-increasing, duplicated and
    descending indices (a broken run is split rather than spinning, because the
    inner loop always consumes at least its first line).
    """
    best: Optional[Tuple[int, int, List[Tuple[int, str]]]] = None
    i, n = 0, len(lines)
    while i < n:
        m = _TRIAL_LINE.match(lines[i].strip())
        if not m or m.group(2).lower() in ("old", "new"):
            i += 1
            continue
        run: List[Tuple[int, str]] = []
        start = i
        last = -1
        while i < n:
            m = _TRIAL_LINE.match(lines[i].strip())
            if not m or m.group(2).lower() in ("old", "new"):
                break
            idx = int(m.group(1))
            if idx <= last:
                break
            run.append((idx, m.group(2)))
            last = idx
            i += 1
        if len(run) >= MIN_TRIALS_TO_SERIALISE and (best is None or len(run) > len(best[2])):
            best = (start, i, run)
    return best


class WorkingMemoryAgent(_BaseAgent):
    """Baseline harness whose recognition recall is serialised.

    Exactly one method body differs from `baseline`: `recall()`. Everything the
    store does, everything `encode()` and `step()` do, and every prompt and tool
    description are the baseline's.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.serial_recognition_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def recall(self, recall_prompt: str | None = None, max_tokens: int = 512) -> str:
        if recall_prompt is None:
            return super().recall(recall_prompt, max_tokens)

        lines = recall_prompt.split("\n")
        found = _find_trial_run(lines)
        if found is None:
            # Not a recognition trial list -- digit span, story recall, MCQ.
            # Baseline code path, untouched.
            return super().recall(recall_prompt, max_tokens)

        start, end, trials = found
        head, tail = lines[:start], lines[end:]
        # Build per-call prompts from the UNSUBSTITUTED template and substitute
        # {wm_contents} per call, so a store value containing newlines or a
        # `trial N:` line cannot disturb the block detection.
        wm_contents = self.wm.to_recall_text()
        budget = min(int(max_tokens), PER_CALL_MAX_TOKENS)

        out_lines: List[str] = []
        anomalies: List[str] = []
        n_no_judgement = n_index_repaired = n_bare = n_extra_dropped = 0

        for idx, word in trials:
            if MASK_OTHER_TRIALS:
                block = [f"trial {idx}: {word}"]
            else:
                block = [f"trial {i}: {w}" for i, w in trials]
            template = "\n".join(head + [DIRECTIVE.format(i=idx), ""] + block + tail)
            prompt = template.format(wm_contents=wm_contents)

            resp = self.llm.generate(
                prompt,
                system=self._build_system(),
                temperature=self.temperature,
                max_tokens=budget,
            )
            raw = resp.text or ""

            hits = _JUDGEMENT.findall(raw)
            verdict: Optional[str] = None
            if hits:
                exact = [v for (i_str, v) in hits if int(i_str) == idx]
                if exact:
                    verdict = exact[0]
                else:
                    verdict = hits[0][1]
                    n_index_repaired += 1
                    anomalies.append(
                        f"# serial_recognition anomaly trial={idx} kind=index_mismatch "
                        f"raw={raw.strip()[:120]!r}"
                    )
                if len(hits) > 1:
                    n_extra_dropped += len(hits) - 1
            else:
                bare = _BARE_JUDGEMENT.search(raw)
                if bare:
                    verdict = bare.group(1)
                    n_bare += 1
                    anomalies.append(
                        f"# serial_recognition anomaly trial={idx} kind=bare_judgement "
                        f"raw={raw.strip()[:120]!r}"
                    )
                else:
                    n_no_judgement += 1
                    anomalies.append(
                        f"# serial_recognition anomaly trial={idx} kind=no_judgement "
                        f"raw={raw.strip()[:120]!r}"
                    )

            if verdict is not None:
                out_lines.append(f"trial {idx}: {verdict.lower()}")
            self.serial_recognition_log.append(
                {"trial": idx, "word": word, "raw": raw, "verdict": verdict}
            )

        n = len(trials)
        header = (
            f"# serial_recognition mode={MODE} block_size={BLOCK_SIZE} trials={n} "
            f"calls={n} answered={len(out_lines)} "
            f"coverage={len(out_lines) / n if n else 0:.4f} "
            f"no_judgement={n_no_judgement} index_repaired={n_index_repaired} "
            f"bare_judgement={n_bare} extra_judgements_dropped={n_extra_dropped}"
        )
        text = "\n".join(out_lines + [header] + anomalies[:MAX_ANOMALIES_LOGGED])

        if self.debug:
            print("  === SERIAL RECALL ===")
            print(f"  {header}")

        return text


MANIFEST = {
    "id": "serial_recognition" if MASK_OTHER_TRIALS else "serial_recognition_open",
    "parent": "baseline",
    "capacity": MAX_KEYS,
    "decay": (
        "none. No change to WorkingMemory at all -- same capacity, same refuse-on-full "
        "overflow rule, same value formatting, same read-out order. The only changed "
        "method body is WorkingMemoryAgent.recall()."
    ),
    "role": (
        "instrument fix -- converts word_recognition from a task that does not "
        "measure the memory module into one that does"
        if MASK_OTHER_TRIALS else
        "ABLATION for serial_recognition: identical serialisation machinery with the "
        "rest of the trial list still visible. Establishes that any collapse in the "
        "masked arm is caused by removing information, not by the serial framing, the "
        "call count or the response stitching."
    ),
    "summary": (
        "Word recognition is presented ONE TRIAL AT A TIME at recall. DEFECT: "
        "wm_word_recognition.py:110 formats every trial, in order, into the single "
        "recall prompt, so Old/New is decidable by reading the prompt -- 36 of 50 "
        "baseline participants score >=0.98 while holding 3.4 keys, against 1 of 53 "
        "humans above 0.98. Humans judged each word alone, before the next appeared "
        "(continuous recognition; Shepard & Teghtsoonian 1961). MECHANISM: recall() "
        "issues one stateless llm.generate call per trial showing the 4-slot store "
        "and that trial's word only, and stitches the replies into the same "
        "`trial N: old/new` text the task parser expects. Stateless means no "
        "look-back and no look-ahead, so the mechanism is independent of the "
        "step()-history leak another candidate owns; step() and reset_messages() are "
        "not touched. Block size 1 because a block of k re-opens the leak for k>1: "
        "measured on these stimuli, the share of old trials whose first occurrence is "
        "visible in the same block is 0.000 at k=1, 0.101 at k=5, 0.202 at k=10, "
        "1.000 at k=100 (the baseline). COST: word_recognition LLM calls 100 -> 5050; "
        "whole 8-task suite ~5,312 -> ~10,262, x1.93, latency-bound not compute-bound, "
        "each call ~600 prompt tokens with an identical cacheable prefix. PROMPT DELTA: "
        "one inserted line naming the trial under judgement, byte-identical in the "
        "masked and open arms, forced because the open ablation is ill-posed without "
        "it; nothing else differs from baseline. MAX_KEYS stays 4; WorkingMemory "
        "untouched, so no collision with the overflow/primacy proposal. INTERPRETATION: "
        "accuracy should FALL sharply and humanlikeness should RISE, but the rise is "
        "arithmetic given any large drop from 0.816 toward the human 0.315 and is not "
        "independent evidence. The real tests are the ceiling group (36/50 -> <=3/50), "
        "response coverage (>=0.95, so degradation-by-garbage is not flattered) and the "
        "open ablation reproducing the baseline. A2 changes meaning: pre-closure it is a "
        "mixture of 36 participants reading the answer (miss 0.001, fa 0.011) and 7 "
        "consulting the store (miss 0.400, fa 0.767); post-closure it measures the "
        "second population only, so both miss and fa should rise and the baseline's "
        "5.754 distance is not a valid reference. trials_attempted (82.9 vs a human "
        "34.5) becomes a comparability gate rather than a covariate: if it collapses to "
        "~5, A2's per-participant estimate rests on ~5 trials and its distance is "
        "uninterpretable in either direction."
    ),
}
