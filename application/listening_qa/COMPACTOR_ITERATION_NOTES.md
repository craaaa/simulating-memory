# WM compactor iteration notes: replace_key redesign (v12-v18)

Context: `bench/tasks/wm_prompt_parts.py` (`wm_system_prompt`) and `bench/core/wm_agent.py` /
`bench/core/working_memory.py` define the streaming, no-lookback working-memory compactor
used by `application/listening_qa`. This tracks the `replace_key`-tool redesign that replaced
the original `write_memory`/`delete_key` two-tool design (v10 and earlier).

All pilots below: gpt-4.1, condition `C2-stream`, n=5/cell unless noted (80 trials total,
4 topics x 4 levels x 5 repeats), default `trial_tool_call_cap=12`.

## Why replace_key exists

v10 (`write_memory`/`delete_key`, sequential only) was the best-performing version on every
metric tried, but required `parallel_tool_calls=False` to avoid a "write when slots full" bug
class — this multiplies LLM request count (~4.3x vs batch) since every tool call is a separate
round trip. `replace_key(old_key, new_key, value)` was introduced so the store always has
exactly `MAX_KEYS` slots (empty placeholders at start) — "memory is full" becomes structurally
impossible, removing the motivation for `parallel_tool_calls=False` and letting the model batch
several replace_key calls in one turn, cutting request count and cost.

## Metric definitions used for tuning (NOT alignment)

Per user instruction, human-alignment was intentionally *not* used to judge these micro-pilots
(re-measuring it after every small prompt edit risks p-hacking a noisy small-n metric). Instead:
- **accuracy**: mean `exact_match_accuracy` across the 80 trials.
- **not-found rate**: % of `replace_key` calls that failed with "key not found" (the agent
  targeted an `old_key` that doesn't currently exist in the store).
- **merged-key rate**: % of non-empty final slots whose key name contains "and" (a proxy for
  cramming two atomic facts into one slot — the key names openly admit it, e.g. "uses and color").

## Iteration log

| tag | change | not-found | merged-key | accuracy | notes |
|---|---|---|---|---|---|
| v12 | `replace_key` + parallel calls, turn-prompt uses filtered `to_recall_text()` | 32% | n/a | 0.53 | Model reflexively guesses `empty_1`/`empty_2` even after consumed — turn prompt never showed live empty-slot keys. |
| v12b | Turn prompt switched to unfiltered `to_turn_text()` (shows all 4 slots incl. empty) | 7.7% | n/a | 0.58 | Big improvement but not zero — model *sees* correct state but still sometimes guesses. |
| v12c | Removed literal `"empty_1"` example string from system prompt AND tool-schema description (both resent every segment regardless of real state — model was copying the literal example text, not "remembering") | **0.4%** | 7.8% | 0.545 | Root cause was literal-string copying, not visibility. This is the cleanest pre-tuning baseline. |
| v13 | Added abstract `"and"`-in-key red flag rule | (bug: only applied to C2 branch, never C2-stream — invalid A/B test) | 7.8% (unchanged, rule never actually shipped to the tested condition) | n/a | Discovered via user's "updating wm_agent doesn't do anything" correction — `replace_all` matched only C2's divergent wording after an earlier external edit. |
| v14 | Concrete WRONG/RIGHT worked example (reverted — user: "NO CONCRETE EXAMPLE") | — | — | — | Also caught leaking a real eval-topic entity name ("Mirelon") into the prompt; regenericized before this was reverted anyway. |
| v15 | Transplanted 15-word value cap from tool-schema into system prompt; collapsed "Three ways" to "Two ways" (dropped explicit FILL AN EMPTY SLOT case) | 13.5% | 5.3% | 0.45 | Shorter values made `"key: value"` turn-prompt lines ambiguous — model started passing the whole line as `old_key` (e.g. `"topic: astronomy"`). |
| v16 | Restored explicit "Three ways" (FILL AN EMPTY SLOT / AMEND / EVICT) | 7.4% | 3.1% | 0.46 | Helped but didn't fully fix; merged-key rate kept improving with each iteration. |
| v17 | Tightened key <4 words, value <10 words; added least-useful eviction hints | 44.0% | **0.0%** | 0.51 | Length caps achieved perfect atomicity, but model compensated by stuffing full descriptive phrases into `old_key`/`new_key` fields, which then didn't match any real stored key. |
| **v18** | Fixed root cause: `to_turn_text()` changed from bare `"key: value"` lines to explicit `old_key="..."  value="..."` — structurally impossible to confuse key and value | **0.0%** | 0.0% (w/ v17 caps) | 0.50 | This was the actual bug behind v15 and v17's failures — a display-format ambiguity, not a prompt-wording problem. Confirmed by testing whether v17's word-caps prompt, run with the v18 display fix, hits zero errors. |

## Key finding

**The `"key: value"` turn-prompt display format was the root cause of every not-found spike
in this thread (v15's "topic: astronomy", v17's "Pictor Nebula study: shows star formation
process").** No amount of prompt-wording change (explicit "read the current key" instructions,
worked examples, word caps) fixed it, because the ambiguity was structural — nothing marked
where the key ended and the value began in what was shown to the model each turn. Explicitly
quoting and labeling the key (`old_key="..."`) in `working_memory.py`'s `to_turn_text()` fixed
it outright (v18: 0% not-found across two consecutive pilots).

## Where this leaves accuracy vs. v10

Even with v18's mechanics fully clean (0% errors, 0% merged keys), accuracy (0.50) still trails
v10's 0.76 (`write_memory`/`delete_key`, sequential, no parallel calls). The remaining gap is
not explained by call failures or atomicity violations — it appears to be about how the
`replace_key` mental model (single unified action, parallel batching) leads the model to
different, less effective memory-management choices than the old two-tool sequential design,
even when every call succeeds. This is an open question, not yet root-caused.

## Sentence-segmentation cross-check (v12c-cap25 vs v10, both sentence-segmented)

To isolate whether the accuracy gap was about the `replace_key` tool design itself (not
paragraph-vs-sentence segmentation, which was confounded in the v12-v18 table above — those
were all paragraph-segmented except where noted), two pilots were run with sentence-level
segments and a raised `trial_tool_call_cap=25` (added as a CLI flag: `--trial-tool-call-cap`,
`--per-segment-tool-call-cap`; previously hardcoded at 12/4):

| version | n/cell | not-found / write-full | accuracy | cost | notes |
|---|---|---|---|---|---|
| v12c wording (replace_key, parallel) | 3 | 0.0% (0/505) | 0.542 | $3.23 (48 trials) | No trial hit the 25-call cap (max 15 used) — the fixed 12-call budget was never the real bottleneck once the display-format bug was fixed. |
| **v10 wording (write_memory/delete_key, sequential)** | 2 | 22.4% (135/603) | **0.581** | $2.46 (32 trials) | 6/32 trials hit the 25-call cap; mean 18.8 calls/trial. Best sentence-segmented accuracy of the whole session, despite the classic write-full error class reappearing (structurally impossible in replace_key). |

**v10's design still wins on accuracy even under sentence segmentation, and even with the old
"write when full" error class present.** This rules out "maybe replace_key just needs sentence
segmentation to catch up" — it doesn't. The accuracy gap is intrinsic to the tool/prompt design
difference (single unified replace vs. separate write/delete, parallel-batched vs. sequential
with result-before-next-decision), not an artifact of segmentation granularity.

Bug caught during this cross-check: `wm_mcq_common.py`'s `slot_utilization` computation called
`agent.wm.slot_utilization`, a property that only exists on the post-v12 `WorkingMemory` — it
now checks `hasattr` and falls back to counting non-empty `final_kv` values for the old design,
so both tool families can share the same trial-running code path.

Also caught: the OpenAI client had no request timeout configured. A `--trial-tool-call-cap=25`
run hung indefinitely (0% CPU, same 4 ESTABLISHED connections, zero progress for 4+ minutes)
with no error and no way to recover short of killing the process. Fixed with `timeout=120.0,
max_retries=2` on the `OpenAI(...)` client construction in `llm_openai.py`.

## Standing recommendation

v10 remains the best-performing version on every metric measured, under both paragraph AND
sentence segmentation. The `replace_key` line was motivated primarily by cost (cutting
streaming's ~4x request-count multiplier vs batch), and v18/v12c-cap25 achieve clean mechanics
at that lower cost, but neither has matched v10's accuracy — and the sentence-segmentation
cross-check above suggests this isn't a fixable artifact of tuning, but an intrinsic property
of the tool/prompt design. Treat v10 as the standing default for accuracy-sensitive work;
`replace_key` variants remain the better choice only when minimizing cost matters more than
closing this ~0.2 accuracy gap.
