# Domain Spec: Human-Memory Simulation via Compactor Harness Search

Produced by the Meta-Harness onboarding conversation
(`https://github.com/stanford-iris-lab/meta-harness/blob/main/ONBOARDING.md`).
All numbers here are measured from released data in this repo; see
`meta_harness/NOTES.md` for derivations and `baseline_humanlikeness.txt` /
`error_structure.txt` for the raw tables.

## Domain Summary

**Task.** Ten classic psychology memory tasks (digit span forward/reverse,
n-back, word recognition, variable mapping, factual QA, narrative QA, semantic
story recall, map task, craft task) run against a frozen LLM wrapped in the
compactor: a 4-slot key-value working-memory agent
(`bench/core/working_memory.py`, `bench/core/wm_agent.py`).

**What we are improving.** Not accuracy. The target is *humanlikeness*,
`1 - W_1` between the model's and the humans' per-participant score
distributions (`src/score.py`), against the ~50 human participants per task in
`runs/human/`. The `claude-opus-4-6` compactor currently **exceeds** human
performance on 8/10 tasks and falls short on N-Back, so this is a
**calibrated-degradation search**, not a capability search, and the direction
of the needed change is not uniform across tasks. A harness that simply
forgets harder improves 8 tasks and worsens N-Back.

**Unit of evaluation.** One simulated participant on one task: one compactor
episode (encode the stimulus stream under the 4-slot budget, then recall /
answer), scored to a single normalized number in [0,1] by `src/score.py`. One
candidate harness is evaluated by running many such participants per task and
comparing the resulting score *distribution* to the human distribution.

**Fixed.** Task stimuli (`data/`), the task protocols and scorers
(`bench/tasks/`, `src/score.py`), the human reference data (`runs/human/`),
the run/JSONL output contract, and the base model weights.

**Allowed to change (search space).** Everything inside the compactor:

- tool schemas and their docstrings (`TOOLS`)
- the condition/system prompt wrapping the agent (`CONDITION_PROMPTS`,
  `SYSTEM_PROMPT_HUMAN`)
- encode-loop control: segmentation, lookback, tool-call caps, retry policy
- write/overwrite/delete policy, key naming, value formatting
- recall-time context construction (`recall()`, `to_recall_text()`)
- **capacity and decay**: `MAX_KEYS` is searchable, and time- or
  interference-based decay, noisy readout, and rehearsal may be added
  (user's explicit choice)

**Out of scope.** Base model or weights, task stimuli, scoring functions,
the human data, and the prompting (non-compactor) conditions C1/C3.

**Base model.** Search: `qwen/qwen3-30b-a3b-instruct-2507`, served locally
under vLLM on the NYU Torch cluster (free in dollars; GPU hours only). Chosen
on measured headroom, not price — see the substrate table below. Confirmation:
`claude-opus-4-6` primary, with `gpt-5.4` and `llama-3.3-70b-instruct` as
additional held-out models.

**Budget.**

- Search: 20 iterations x 2 candidates = 40 candidates on the reduced search
  set, ~10-30 min single-GPU each, so **~15-25 GPU-hours, $0 API**.
- Proposer: Claude Code (Opus) for ~20 iterations, on the existing
  subscription.
- Confirmation: top 3-5 Pareto finalists x full task set x 3 held-out models.
  On `claude-opus-4-6` a full evaluation is **~$370** (proxied from 19,940
  logged LLM turns / 6.6M chars per evaluation), so **~$1,850 for 5 finalists**
  on opus alone, ~$2-3k including gpt-5.4. **This requires explicit approval
  before any paid call** and is currently `unknown` / unapproved.

### Why not the cheapest model

Measured mean humanlikeness and headroom above the split-half human noise
floor (attainable ceiling 0.955):

| base model | humanlikeness | headroom |
|---|---|---|
| claude-opus-4-6 | 0.730 | **0.225** |
| qwen3-30b-a3b-instruct-2507 | 0.788 | **0.167** |
| llama-3.3-70b-instruct | 0.791 | 0.164 |
| llama-3-8b-instruct | 0.798 | 0.157 |
| gpt-5.4 | 0.806 | 0.148 |
| qwen3-30b-a3b-thinking-2507 | 0.825 | 0.130 |
| qwen3-next-80b-a3b-instruct | 0.832 | 0.122 |
| qwen3-8b | 0.883 | **0.071** |

Weak models are already human-like *by incapacity*. qwen3-8b's headroom sits
at or inside the noise floor on 4/10 tasks, so it has nothing to find.
qwen3-30b-a3b is the best free substrate.

## Harness and Search Plan

**Interface.** Every candidate is a module under
`meta_harness/candidates/<id>/harness.py` exporting a class that satisfies the
surface `bench/tasks/*` already calls on `WorkingMemoryAgent`:

```python
class CandidateHarness:
    def __init__(self, llm, condition_id="C2", temperature=0.0,
                 debug=False, system_prompt_override=None) -> None: ...
    def encode(self, content: str | list[str]) -> dict: ...
    def recall(self, recall_prompt: str | None = None,
               max_tokens: int = 512) -> str: ...
    def step(self, user_message: str, max_tokens: int = 1024) -> str: ...
    def get_log(self) -> list[dict]: ...
    def get_step_log(self) -> list[dict]: ...
    @property
    def wm(self): ...          # must expose .store and .snapshot()
```

**Compliance test.** A `verify_interface.py` that instantiates the candidate
against a stub LLM and asserts: all six methods present with the above
signatures; `encode()` returns a dict carrying `final_kv`; `wm.store` is a
`dict[str, str]`; the declared capacity is reported in the candidate's
`manifest.json`; and one smoke episode per task family completes without
raising. A candidate failing compliance is recorded with score `null` rather
than silently dropped.

**Baselines seeded into the population.**

1. The released compactor, unmodified (`MAX_KEYS=4`, batch `encode`) —
   the number to beat: 0.788 on qwen3-30b.
2. The summarizer baseline (`SummarizerAgent`) — the alternative compression
   family already in the repo.
3. A no-memory control: full stimulus in context, no compactor. Establishes
   how much of current humanlikeness the memory module is responsible for.
4. A deliberate random-decay control: per-participant decay rate drawn from a
   distribution. **This is the adversary, not a contender.** It exists to test
   whether the error-structure axes have teeth, with an explicit pass
   condition: the control must reach mean humanlikeness **>= baseline + 0.05**
   while its A2 distance stays **>= 2x the baseline's**. If it cannot match the
   distribution, it is a weak adversary and the axes are untested rather than
   validated; if it matches the distribution *and* the axes, the axes do not
   discriminate and the search is invalid. This is one candidate evaluation
   (~20 GPU-min) and it **runs first**, immediately after the loop works and
   before the remaining 38 candidates are spent.

**Reusable helpers built before iteration 1.**

- `score_candidate(candidate, model, tasks, n) -> dict` — runs the candidate
  through `bench.cli` and returns per-task humanlikeness plus every axis below.
- `noise_floor.py`, `error_structure.py` (already written, in `meta_harness/`).
- `protocol_match.py` — subsamples model digit-span trials onto the human
  adaptive schedule (needed before axis A1 is usable; see Evaluation).
- `compare.py <candidate_a> <candidate_b>` — per-task delta with bootstrap CIs.

**First search loop.** Seed with the four baselines, then 20 iterations x 2
candidates. The proposer reads prior candidates' source, scores, per-task
vectors, axis values, and raw encode/recall traces off the filesystem.

## Evaluation Plan

**Primary metric.** Mean humanlikeness over the 8 search tasks,
**subject to a per-task floor**: no task may regress more than 0.03 below the
seeded baseline. Without the floor the proposer can buy the entire headline
improvement from Variable Mapping, which alone holds 0.569 of the 2.25 total
task headroom on opus. The per-task vector is stored with every candidate so
trades are visible.

**Error-structure axes (hard Pareto axes, not secondary).** `1 - W_1` on
scores is gameable: a per-participant random decay rate matches the human
distribution while being scientifically empty, and decay is *inside* the
search space. Each axis is scored as |model - human| on a statistic both sides
expose:

| axis | statistic | humans | opus-4-6 compactor |
|---|---|---|---|
| A2 word recognition | miss rate / false-alarm rate | 0.272 / 0.045 = **6.09** | 0.000 / 0.533 = **0.00** |
| A3 story recall | BLEU vs transcript @ recall words | 0.002 @ 137 | **0.199 @ 366** |
| A1 digit span | sub-span failure / supra-span hit | 0.087 / 0.000 | 0.000 / **0.574** |

A2 is the existence proof: opus scores 0.760 humanlikeness on word recognition
with a *fully inverted* error structure — humans are conservative, opus
false-alarms on half of all new words and never misses an old one. A3 shows
opus regurgitating verbatim at 2.7x human recall length.

**A2's denominator is itself a search target.** Word recognition terminates at
3 strikes, so a participant contributes only the trials they attempted — mean
34.5 for humans, 40.2 for opus, and **87.5 for qwen3-30b**, which survives 2.5x
longer than any human and is a non-humanlike signature in its own right.
Bootstrapped, the miss/FA ratio separates cleanly (humans [3.79, 11.39] vs opus
[0.00, 0.00]), so the contrast is not a small-sample artifact. But **any harness
change that shifts `first_error_at` changes the denominator and so moves A2
without fixing the asymmetry.** So A2 is scored jointly with trials-attempted,
which is recorded as a covariate on every candidate, and the proposer cannot
bank an A2 gain that came from surviving longer.

**A1 is not yet usable.** The human digit-span protocol is adaptive and
terminates (~12 trials, ~2 per span) while the model protocol runs all 19 span
lengths, inflating the model's supra-threshold opportunities. `protocol_match.py`
must land before A1 becomes an axis; until then A1 is reported but not
optimized against.

**Search set.** 8 tasks: digit span forward, digit span reverse, n-back,
**word recognition**, variable mapping, narrative QA, semantic story recall,
craft task. The three error-structure axes must live on search tasks — an axis
computed on a held-out task cannot be optimized without leaking it — so word
recognition (A2), story recall (A3) and digit span forward (A1) are all in the
search set by construction, and the held-out pair is chosen from what remains.
Reduced repeats: digit span forward and reverse cut from 1900 rows to ~200 each
(together they are 85% of all LLM turns — 16,975 of 19,940 — for 0.307 + 0.106
headroom), everything else at released repeats. ~4,700 turns per candidate.
Human reference: a fixed random **half** of each task's participants.

**Held-out test.** Both axes, models primary:

- *Models (primary).* The winning harness is evaluated on `claude-opus-4-6`,
  `gpt-5.4`, and `llama-3.3-70b-instruct`, none of which the search touches.
  All three already have full 10-task baselines, so the comparison is direct.
- *Tasks (secondary).* **Map task** and **factual QA** are held out entirely —
  one procedural task and one long-term-memory QA task, headroom 0.221 and
  0.276 on opus. n=2 is weak, so this is a directional check, not a statistical
  claim.
- *Participants (always on).* The other half of each task's human
  participants, never used during search. Guards against fitting human
  sampling noise.

**Noise.** Split-half W_1 within the human participants gives the irreducible
floor, per task: digit span 0.036, reverse 0.026, n-back 0.025, word
recognition 0.075, variable mapping 0.041, factual QA 0.061, narrative QA
0.053, free recall 0.042, map task 0.054, craft task 0.041. Mean attainable
ceiling **0.955**. Model-side bootstrap CIs on the observed W_1 are ~±0.045
at released n. **Any candidate delta under 0.05 on a single task is noise.**

**Runtime.** One reduced-set candidate on qwen3-30b-a3b under vLLM: ~10-30 min
on one GPU at `max_parallel_participants: 50`. One full-set evaluation on a
paid API: ~$370 on opus, ~$46 on gpt-4.1, ~$9 on gpt-4.1-mini.

**Contamination / leakage risks.**

1. *The real one.* Human data is fixed and committed in this repo, and the
   proposer could hard-code human-matching constants instead of discovering a
   memory mechanism. **Blinding the proposer is not achievable and the spec
   does not claim it.** `runs/human/` is committed and reachable via
   `git show` or any prior commit regardless of working-tree permissions; the
   ~28 model dirs under `runs/compactor/` carry human-relative numbers in their
   `metrics_*.txt`; the data is in the published paper; and
   `meta_harness/NOTES.md` plus this spec state the human statistics verbatim
   *by design*, because "humans are conservative on recognition" is exactly the
   mechanism hint the proposer should have.

   So the approach is **detect, not prevent**, which is what the paper's own
   protocol relies on:
   - the **held-out models** are the primary guard — a hard-coded constant
     tuned on qwen3-30b will not transfer to opus, gpt-5.4 and llama-3.3-70b;
   - the **held-out tasks and participant half** are never scored back to the
     proposer;
   - a **source audit** rejects any candidate containing a numeric literal
     within 10% of a human statistic, unless it is a named psychological
     constant with a citation.

   The alternative — running the proposer in a stripped worktree with no
   `runs/` at all — buys real isolation but costs the offline trace warm start,
   which is the highest-signal material available. Rejected for that reason.
2. Story/map/craft stimuli may be in pretraining, which inflates recall
   independent of the harness. Already true of the baselines, so it biases
   absolute humanlikeness but not candidate-to-candidate comparisons.
3. Only 8 search tasks and 20 iterations — overfitting the search set is
   likely. The held-out models are the guard.

## Experience and Logging

**Offline warm start.** Available and worth encoding into the proposer's
initial context:

- `meta_harness/NOTES.md` — the inverted objective, the headroom table, the
  noise floor, the error-structure findings.
- The released compactor traces under `runs/compactor/*/tasks/*.jsonl`:
  full `encoding_log` with tool calls, `final_kv`, and `recall_raw` for every
  participant on every task, across 9 models with complete baselines. This is
  the highest-signal offline material — it shows exactly *how* the current
  harness over-performs (e.g. word recognition writes one key per word and
  then answers "old" by default; story recall stores near-verbatim text).
- The paper (arXiv:2605.25680) for the task protocols and the human study.
- Cowan (2001) for the 4-slot grounding that `MAX_KEYS` encodes.

**Per-candidate artifacts.** Under
`meta_harness/runs/<iteration>/<candidate_id>/`:

```
manifest.json      # parent candidate id, declared capacity/decay, diff summary
harness.py         # the candidate source
scores.json        # per-task humanlikeness + mean + floor check + axis values
tasks/*.jsonl      # full per-participant rows, incl. encoding_log and final_kv
proposer.log       # the proposer transcript that produced this candidate
compliance.json    # interface verification result
```

**Highest-signal debugging artifacts,** in order: `tasks/*.jsonl`
`encoding_log` and `final_kv` (what the harness actually chose to store),
`recall_raw` (how it reconstructed), then `scores.json` axis values.

**Metadata to preserve.** Candidate id and parent id, base model and serving
config, git SHA of `bench/`, search-set composition and repeat counts, the
participant-split seed, vLLM version and sampling params, wall-clock and GPU
hours, and the declared capacity/decay parameters.

**CLI worth building** (`python -m meta_harness.history`):

- `list [--sort mean|axis|iteration]` — one line per candidate
- `show <id>` — manifest, scores, per-task vector, axis values, diff vs parent
- `diff <id_a> <id_b>` — per-task delta with bootstrap CIs
- `trace <id> --task <t> --participant <n>` — one episode's encode/recall trace
- `frontier` — the current Pareto frontier over (humanlikeness, A2, A3)
- `regressions <id>` — tasks violating the per-task floor

## Open Questions and Unknowns

- **Base branch.** This worktree branches from `origin/main` (431474a) as
  requested, but `feat-rationales` is **189 commits ahead** and main lacks
  `encode_streaming` / the `C2-stream` condition, `CLAUDE.md`, and the
  `application/listening_qa` work. The harness interface above is written
  against main's `WorkingMemoryAgent`. **Recommend rebasing onto the newer
  compactor before iteration 1**, so the streaming encode path is inside the
  search space rather than absent from it. Unresolved.
- **Torch allocation.** No vLLM env exists yet; setup is the next action.
  GPU partition, queue wait, and hours available: `unknown`. Proposed default:
  one A100/H100 for an 8-hour interactive allocation to validate the loop on
  2 candidates, then batch `sbatch` jobs for the remaining 38.
- **Confirmation spend.** ~$1,850-3,000 for the finalist evaluations.
  `unknown` / **not approved**. No paid call until approved.
- **`MAX_KEYS != 4` is in scope by the user's choice**, but it breaks the
  Cowan-2001 grounding the compactor is built on. A winning candidate that
  moves capacity needs that argued explicitly in the paper, not silently
  accepted. Same for any decay function: it must be a named psychological
  mechanism, not a fitted noise term.
- **A1 axis** is blocked on `protocol_match.py` (human adaptive schedule vs
  the model's exhaustive 19 spans).
- **N-Back direction conflict.** N-Back is the one task where the model is
  *worse* than humans (0.699 vs 0.866), so it opposes every other task's
  gradient. Whether the search can satisfy both with one harness, or whether
  this exposes a real limit of a single 4-slot mechanism, is an open empirical
  question — and arguably the most interesting result the search could produce.
