# `serial_recognition` — the recognition test is presented one trial at a time

Parent: `baseline`. Iteration 2b. One changed method body: `WorkingMemoryAgent.recall()`.

This candidate is different in kind from the others in this search. It does not
try to move a score. It converts a task that does not measure the memory module
into one that does, and then reports what the module actually does. Read section
5 before reading any number in section 6 as an improvement.

---

## 1. The defect, with numbers

`bench/tasks/wm_word_recognition.py:110-114`:

```python
trials_text = "\n".join(f"trial {t['trial_index']}: {t['word']}" for t in trials)
recall_prompt = RECALL_PROMPT.format(trials_text=trials_text, wm_contents="{wm_contents}")
recall_raw = agent.recall(recall_prompt=recall_prompt, max_tokens=1024)
```

All 100 recognition trials are in the recall prompt, in order, verbatim. "Old"
means *appeared earlier in the list*, and the list the model must compare against
is the list the prompt just printed. The prompt's own instruction — "Based ONLY
on the above contents", meaning the four-slot store — is unenforceable.

Measured on `meta_harness/runs/iter0/baseline` (50 participants, 100 trials each):

| | mean score | ceiling (≥0.98) | floor (≤0.04) | miss | false alarm | trials attempted | keys held |
|---|---|---|---|---|---|---|---|
| baseline, all 50 | **0.816** | **36/50** | 7/50 | 0.043 | 0.127 | 82.9 | 3.4 |
| baseline, the 36 at ceiling | 1.00 | — | — | **0.001** | **0.011** | 100.0 | — |
| baseline, the 7 at floor | 0.02 | — | — | **0.400** | **0.767** | **4.9** | — |
| *human* | **0.315** | **1/53** | — | 0.272 | 0.045 | 34.5 | — |

A four-slot store holding 3.4 keys does not support miss 0.001 over 2 099 old
trials. 36 of 50 participants are reading the answer off the prompt; 7 are
consulting the store; the remaining 7 are in between. word_recognition's 0.495
humanlikeness is the largest single term in the eight-task mean and it is
measuring prompt legibility, not memory.

**Could the store alone have produced the ceiling scores?** This is the one
reading that would sink the whole diagnosis, and it is checkable offline, so I
checked it rather than assuming. Some stores really do carry indexed
first-occurrence lists — e.g. `first_occurrence: Evening (1), crystal (6),
letter (10), velvet (12), …` — and a trial is decidable from such a store
(word `w` with recorded first index `i` is "old" iff the current trial index
exceeds `i`). Per participant, over all 50:

| group | n | words carrying an index in the store | fraction of the 100 trials decidable from the store | index of the 3rd undecidable trial |
|---|---|---|---|---|
| ceiling (≥0.98) | 36 | mean 6.9, max 42 | mean **0.19**, max 0.89 | mean 8.4, max 51 |
| middle | 7 | mean 7.4 | mean 0.22 | mean 5.4 |
| floor (≤0.04) | 7 | mean 7.6 | mean 0.25 | mean 9.6 |

Scoring stops at the third error, so the best score a store-only reader could
reach is bounded above by `(index of the 3rd undecidable trial − 1)/100`, and
that bound is **0.50 at its maximum across all 50 participants and 0.071 on
average. Zero participants have a store that permits 0.98.** The ceiling group is
reading the prompt. That is what sets the threshold in prediction P1.

**Corroborating evidence from iteration 1.** Under `displacement`,
word_recognition rose +0.1388 — its largest single gain — and the error axes
showed the gain was not recognition improving: misses 0.043 → 0.037 while false
alarms 0.127 → **0.211**, so A2's distance *worsened* 5.754 → 5.920. The ceiling
group fell only 36 → 30. Roughly 43 participants were still reading the list off
the prompt under a change to the overflow rule, which is exactly what a
presentation-level leak predicts and what a store-level change cannot fix.

---

## 2. The mechanism, and why serial is the principled fix

**The tension, stated plainly:** the trial list cannot be deleted, because the
trial list *is* the question. To ask whether `apple` is old you must show
`apple`. What can be removed is the *simultaneity*.

In the human protocol the words appear one at a time and each judgement is
committed before the next word exists. That is the definition of a continuous /
running recognition test (Shepard & Teghtsoonian 1961; Hockley 1982; Nickerson
1972 for the recognition-memory paradigm generally), and it is what makes the
task a memory task: the comparison set is *in the participant*, never on the
page. Any presentation that shows trial *j* while trial *i < j* is still visible
converts recognition into visual search over the display. Serialisation is
therefore not a degradation trick; it restores the only condition under which
"recognition" names anything.

`recall()` is a harness method and it owns the model's test-time context, so the
fix is expressible entirely as an override. Implementation:

* one `self.llm.generate(...)` call per trial, each showing the four-slot store
  and exactly one trial line;
* replies stitched back into the `trial N: old/new` text the task's
  `parse_responses` already consumes, so `score_game`, `per_trial`,
  `error_structure.a2_model` and `score.py` all see the shape they saw before;
* nothing else in `bench` is touched — not the task, not the scorer, not the
  stimuli, not `WorkingMemory`.

**Why not the alternatives.**

| alternative | why not |
|---|---|
| mask trials already judged, keep future ones visible | backwards. The leak is that a repeat's *earlier* occurrence is visible; masking the future removes nothing and leaves every "old" decidable by look-back. |
| mask not-yet-reached trials, keep judged ones visible | this is the leak, restated. Look-back over judged trials is exactly how "old" is read off the page. |
| show the whole list but forbid consulting it | already tried by the benchmark: the prompt says "Based ONLY on the above contents" and 36 of 50 participants ignore it. An instruction is not an information barrier. |
| shorten the studied list, or cap trials | changes the stimuli, i.e. the benchmark, not the harness — and out of scope. |
| force a commitment before the next trial is visible | this *is* serialisation; here it is stronger than a commitment protocol, because the calls are stateless (below), so there is nothing to revise. |

**Stronger than serial, and this matters for the concurrent proposals.** The base
`recall()` answers with `llm.generate`, a single fresh prompt with no
`self._messages` and no tools. Each of the 100 calls is therefore **stateless**:
no look-back at trials already judged, no look-ahead to trials not yet reached,
nothing accumulated in conversation history. So the mechanism is *not* entangled
with the `step()`-history leak that iteration 2c owns — `step()` is never invoked
on this path, `reset_messages()` is never called, and `_messages` is neither read
nor written by `recall()`. Injection is orthogonal: 2c changes `step()`, this
changes `recall()`, and the two touch disjoint method bodies. Likewise no
collision with iteration 2a: `WorkingMemory.write_key`, the overflow rule, value
formatting and read-out order are all the baseline's, byte for byte.

Being stateless is also, honestly, *less* memory than a human has. A human's
memory is updated by the test items themselves — trial 40's word is encoded and
is available when it recurs at trial 80. Here the store is frozen at the end of
`encode()`. What partly compensates is the opposite mismatch: `encode()` is
handed the whole list, so the store is built with foreknowledge of repeats that a
human at trial 40 does not have. Both mismatches live in `encode()`, not
`recall()`, and closing them means serialising *study* as well — see section 7.

**Block size is 1, measured not asserted.** With a block of `k` trials visible, a
repeat whose first occurrence falls inside the same block is still readable.
Computed over this run's own 50 games (5 000 trials, 2 508 of them old):

| k | share of *old* trials readable within the block | share of all trials |
|---|---|---|
| **1** | **0.000** | **0.000** |
| 2 | 0.027 | 0.014 |
| 5 | 0.101 | 0.051 |
| 10 | 0.202 | 0.101 |
| 20 | 0.352 | 0.177 |
| 100 (the baseline) | 1.000 | 0.502 |

At k = 10 a fifth of all "old" judgements are still free. Since the leak is the
entire object of the exercise, and k = 1 is also the protocol-exact choice, k = 1
is the only defensible setting. The batched variant the brief allows for is
implemented (`BLOCK_SIZE`) but set to 1, because the cost estimate in section 4
says batching is not needed.

---

## 3. What changed in the prompt, and why it was forced

Exactly one line is inserted immediately before the trial block:

```
You are being asked about one trial at a time. Judge ONLY trial {i} and output exactly one line for it.
```

Everything else is byte-identical to the baseline: the system prompt, the store
rendering, the "Original task instructions", the "Based ONLY on the above
contents" sentence, `FORMAT_RULES` and its worked example. `TOOLS` and
`CONDITION_PROMPTS` are re-exported unmodified.

It is forced by the ablation, which is the only comparison that makes this result
interpretable (section 5). The open arm shows all 100 trial lines and runs the
same machinery; without a directive naming the target trial it is not a
well-posed question — the model cannot know which of 100 visible words it is
being asked about. Since the line must exist in the open arm and must be
identical in both arms for the arms to differ in one thing only, it exists in
both. Secondarily it removes an index-attribution ambiguity: with the trial named,
the reply is expected to carry the absolute index, so the harness's index-repair
path is a fallback rather than the normal case.

Two non-prompt parameter changes, both stated for the record:

* per-call `max_tokens` is `min(caller_max_tokens, 128)`. One formatted line is
  ~8 tokens; the cap exists so a call that ignores the format cannot emit 100
  hallucinated lines 100 times over. `LLMResponse` carries no `finish_reason`,
  so truncation is not directly observable — which is why any reply without a
  parseable judgement is logged verbatim into `recall_raw` rather than silently
  dropped.
* judgements for trials other than the one under judgement are discarded. In the
  open arm the model can see other words; crediting a judgement produced by a
  call that was not asked about that trial would smuggle the leak back in
  through the stitching.

---

## 4. Call-count multiplier and cost

Counted from the baseline run's own logs (`step_logs` / `turn_logs` /
`model_parsed` lengths, plus encode+recall for the rest):

| task | rows | LLM calls, baseline | LLM calls, this candidate |
|---|---|---|---|
| wm_word_recognition | 50 | 100 | **5 050** |
| wm_nback | 150 | 2 002 | 2 002 |
| wm_variable_mapping | 150 | 1 650 | 1 650 |
| wm_semantic_story_recall | 200 | 400 | 400 |
| wm_digit_span_forward / reverse | 190 each | 380 each | 380 each |
| wm_craft_task | 150 | 300 | 300 |
| wm_narrative_qa | 50 | 100 | 100 |
| **total** | | **≈5 312** | **≈10 262** |

Multiplier: **×50.5 on word_recognition's calls, ×1.93 on the whole search
set.** It is latency-bound, not compute-bound:

* each call is ~600 prompt tokens (system prompt + four-slot store + task text +
  one trial line + format rules) against the baseline's single ~1 400-token call,
  so word_recognition's prompt tokens go from ~70 k to ~3 M — trivial for an
  H200 whose measured KV cache is 733 488 tokens at 89.5× concurrency;
* output tokens *fall*: 100 short replies of ~8 tokens each against one ~500-token
  reply;
* the prompt prefix is identical across all 100 calls of a participant (the store
  is frozen during recall and the trial line sits after it), so automatic prefix
  caching applies to nearly the whole prompt;
* the 100 calls per participant are sequential, but the 50 participants run
  concurrently under `max_parallel_participants: 50`, so the added wall clock is
  ~100 sequential rounds at 50-way concurrency — minutes, against `candidate.sbatch`'s
  8 h limit and the 18 m 18 s that iteration 1 took.

No batching is therefore needed, and no money is spent: local vLLM, GPU hours only.

---

## 5. How A2 must be re-interpreted, and what comparison is valid

### A2 changes meaning, not just value

Pre-closure, A2 is a **mixture over two populations doing different tasks**:

| population | n | miss | false alarm | what it is doing |
|---|---|---|---|---|
| ceiling | 36 | 0.001 | 0.011 | reading the studied list off the prompt |
| floor | 7 | 0.400 | 0.767 | judging from a four-slot store |
| middle | 7 | — | — | partly each |
| **reported A2** | 50 | **0.043** | **0.127** | ratio 0.340, distance 5.754 |

The reported 0.340 is 36 participants' near-zero error rates diluting 7
participants' real ones. Post-closure, A2 measures the second population only —
genuine recognition from four chunks. **So the baseline's 5.754 is not a valid
reference for this candidate's A2 distance, in either direction.** The human
target (miss/fa = 6.09, humans conservative) remains the right target, because it
is a fact about humans and does not depend on the model's instrument; what must
be re-established from scratch is the model-side reference, and that is what this
run produces.

Expect the *direction*: both miss and false alarm should rise, because the 43
previously-leaking participants will now behave like the 7. The ratio may end up
near 0.5 and the distance near the baseline's 5.754 **by coincidence of a
mixture, not by continuity** — which is precisely why the number cannot be read
as "A2 unchanged".

### A2 also acquires a comparability problem, which is the more important point

`score_game` stops at the third error, so each participant contributes only the
trials they attempted. Baseline: 82.9. Humans: 34.5. The 7 store-consulting
participants: **4.9**. If the whole sample lands near 5, then A2's
per-participant miss and false-alarm rates rest on ~5 trials and its bootstrap CI
will be enormous. `trials_attempted` stops being a covariate and becomes a
**comparability gate**: below ~15 attempted, A2's distance from 6.09 is
uninterpretable in either direction, for the same reason the digit-span staircase
was uninterpretable before `protocol_match.py`.

Recommended follow-up, which I have *not* implemented because
`error_structure.py` is outside my deliverables: an **untruncated model-side A2**
over all 100 trials, recoverable from `recall_raw` + `gold_trials` without any
new run. On the baseline it reads acc 0.934, miss 0.042, fa 0.088, ratio 0.479 —
against the truncated 0.043 / 0.127 / 0.340. This is a *model-side diagnostic
only*: the human protocol also stopped at three strikes, so there is no
untruncated human reference and it cannot replace the axis. Its use is to tell a
real response bias from a five-trial artifact.

### The valid comparison, and the ablation I want run alongside

**For word_recognition, the valid contrast is masked vs open, not masked vs
iter0/baseline.** The existing baseline measured that task with the leak open, so
the delta against it is real as an arithmetic fact but is not a statement about
memory. For the other seven tasks `iter0/baseline` remains the correct control,
because they run the baseline code path unchanged (section 6, P7–P9).

The ablation — **`serial_recognition_open`** — is the same file with
`MH_SERIAL_RECOGNITION_MASK=0`:

* same serialisation machinery, same 100 calls per participant, same directive
  line, same one-judgement-per-call framing, same stitching, same index repair,
  same `max_tokens`;
* the **only** difference: the other 99 trial lines are still visible.

It answers the one objection that would otherwise sink the result — *"you asked
it to judge a single word with no context; of course it got worse, that is
framing, not memory."* If the open arm reproduces the baseline (high accuracy,
ceiling group intact) and the masked arm collapses, the collapse is caused by
removing information. If **both** collapse, the collapse is caused by the serial
framing or the stitching and this candidate's claim is void — which is
pre-registered as P12.

How to run it in the same job (one extra process, not one extra job):

```bash
# candidate arm (default; MASK unset or =1)
python meta_harness/run_candidate.py \
  --candidate meta_harness/candidates/serial_recognition/harness.py \
  --config meta_harness/cluster/search_set.yaml \
  --out-dir meta_harness/runs/iter2/serial_recognition "${TASK_ARGS[@]}"

# ablation arm, same file, same process pattern
MH_SERIAL_RECOGNITION_MASK=0 python meta_harness/run_candidate.py \
  --candidate meta_harness/candidates/serial_recognition/harness.py \
  --config meta_harness/cluster/search_set.yaml \
  --out-dir meta_harness/runs/iter2/serial_recognition_open \
  -t wm_word_recognition        # word_recognition alone is enough for the ablation
```

The ablation only needs `wm_word_recognition` (50 participants, ~5 050 calls), so
it costs a fraction of a full arm. `MANIFEST["id"]` switches to
`serial_recognition_open` with the environment variable, so the two rows cannot
collide under `score_candidate.py --record` (the failure mode the worklog already
recorded once), and the mode is written into every row's `recall_raw` header as
well as into the copied `harness.py`.

---

## 6. Pre-registered predictions

Machine-readable in `meta_harness/logs/pending_serial_recognition.json`. All
baselines are `meta_harness/runs/iter0/baseline`. `NOISE_FLOOR` for
word_recognition is 0.121 and thresholds below respect it.

| # | prediction | threshold | baseline | disconfirmed if |
|---|---|---|---|---|
| **P1** | **ceiling group on word_recognition collapses** | **≤ 3 of 50 at score ≥ 0.98** | **36/50** | **≥ 15/50 survive → the leak is not what closes the ceiling, and I am wrong** |
| P2 | mean word_recognition score falls | < 0.35 | 0.816 | > 0.60 |
| P3 | A2 miss rate **rises** | > 0.10 | 0.043 | ≤ 0.043 (falls) |
| P4 | A2 false-alarm rate **rises** | > 0.20 | 0.127 | ≤ 0.127 (falls) |
| P5 | A2 trials attempted **falls** | < 40 | 82.9 | > 60. Gate: below 15, A2's distance is not interpretable |
| P6 | A2 distance does **not** approach the human ratio | > 4.0 | 5.754 | < 2.0 would mean closing the leak fixed the asymmetry, which I do not expect |
| P7 | digit span untouched | A1 leak within ±0.01 of 0.1298 **and** best_span within ±0.5 of 18.4 | 0.1298 / 18.4 | either moves further |
| P8 | n-back untouched | \|Δ HL\| < 0.060 **and** n=3 `answered` within ±1.0 of 6.82 | 0.791 / 6.82 | either moves |
| P9 | the other four tasks untouched | \|Δ\| < noise floor: variable_mapping 0.017, craft 0.025, narrative_qa 0.030; story recall < 0.03 (the enforced floor; its 0.011 noise floor is too tight for a pass-through path) | — | any floor violation |
| P10 | response coverage stays high | ≥ 0.95 of the 100 trials carry a parseable judgement | 1.000 | < 0.90 → this is a compliance/parsing failure, not a leak closure |
| P11 | responses are not degenerate | "old"-response rate over 100 trials in [0.15, 0.95] | 0.528 (per-participant range 0.23–0.79) | outside → constant responding |
| **P12** | **the open ablation reproduces the baseline** | ceiling ≥ 25/50 **and** mean score ≥ 0.70 in the open arm | 36/50, 0.816 | **the open arm also collapses → the effect is the framing, not the information, and this candidate is void** |
| P13 | word_recognition humanlikeness rises | Δ ≥ +0.121 (its noise floor) | 0.495 | falls, or moves < 0.121 |
| P14 | the eight-task mean rises | Δ ≥ +0.026 | 0.7861 | — reported, not claimed as a capability gain |

**P13 and P14 are entailed, not evidence, and I am flagging that rather than
banking it.** Given *any* large accuracy drop from 0.816 toward or past the human
0.315, W₁ shrinks and humanlikeness rises arithmetically — a point mass at zero
would still score W₁ ≈ 0.315, i.e. humanlikeness ≈ 0.685 against the baseline's
0.495. So a rise here is guaranteed by the direction of the change and confirms
nothing about mechanism. The load-bearing tests are **P1 (ceiling collapse),
P10 (coverage), and P12 (the open ablation)**.

**Reported, deliberately not scored:** miss + fa is expected to land in
[0.9, 1.4], i.e. at or slightly below chance discrimination (the 7
store-consulting participants sum to 1.167). That is *not* a garbage signature —
P10 and P12 are the garbage tests. It is familiarity without recollection: the
store typically holds an unindexed "repeated words" list built with foreknowledge
of the whole list, so the model answers "old" on a word's *first* occurrence too.
Humans show the opposite bias (miss 0.272 vs fa 0.045, i.e. conservative). If it
lands there, the named iteration-3 ingredient is a **serial encode** plus a
conservative response criterion — not more forgetting.

**The overshoot this predicts, stated before the run.** Serialisation will very
likely take the model from far *better* than human (0.816 vs 0.315) to far
*worse* (plausibly ~0.05, given the store-consulting subgroup's 4.9 attempted
trials). word_recognition then moves out of the "calibrated degradation" column
and into the "capability" column: the remaining headroom would be *getting the
model up to human level* at four-chunk recognition. That is a real finding about
the search's structure, and it is not visible while the leak is open.

---

## 7. Residual mismatches this does **not** close

Stated so that nobody reads the post-closure number as "word_recognition is now
comparable, full stop".

1. **`encode()` still sees the whole list at once.** The human study phase *is*
   the test phase; here the model reads all 100 trials before judging any. So its
   store is built with foreknowledge of which words repeat, which is why
   false-alarms on first occurrences are the expected failure. Fixing it means
   serialising study as well — turning `encode()` into a per-word stream — which
   is a second, larger intervention and would confound this one. **Iteration 3.**
2. **`MAX_KEYS` bounds slots, not content.** Nothing caps the length of a value.
   In the baseline the longest single value holds 99 comma-separated items, mean
   19.8 across participants, and one store lists 42 words with their first-occurrence
   indices. A "4-chunk limit" that admits a 99-item chunk is not Cowan's limit.
   A3 catches verbatim regurgitation on story recall only; no axis caps value
   length anywhere else. This is a capacity defect independent of any leak, it is
   inside the injectable surface, and it is the strongest remaining candidate
   after this one.
3. **Three strikes truncates the A2 denominator** (section 5). A protocol-matched
   or untruncated A2 is needed before the axis can be trusted post-closure.

## 8. What the check function should assert

`check_predictions.py` is not edited here (three proposers would collide in its
`CHECKS` dict). `serial_recognition_checks(run_dir, baseline)` should read
`pending_serial_recognition.json` and, for each entry, resolve `metric` —
either a dotted path into `score_candidate.evaluate()`'s record
(`axes.A2.fa_rate`, `humanlikeness_by_task.word_recognition`,
`axes.A1.best_span`, `axes.nback_levels.diagnostics.3.answered`) or the explicit
one-line computation given in `compute` for the four quantities no existing
tooling reports (ceiling-group size, mean score, response coverage, old-rate) —
and compare against `threshold` with `comparator`. Missing metric ⇒
`INCONCLUSIVE`, never PASS. P12 must be evaluated against the *open* run dir, not
the candidate's.
