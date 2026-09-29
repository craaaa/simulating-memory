# CONTINUITY — session state, 2026-09-28 / 29

## 2026-09-29T02:00Z [USER][CODE] — EXECUTION ORDER, everything blocked on job 18774002

`18774002` (Qwen, 3 baseline repeats, all 8 tasks, `ITER=11postfix` → `runs/iter11postfix/`)
was PENDING at write time. It runs from `/scratch/cl5625/mh-postfix` at commit **410ec2a**.
~70 min. Do these in order; each later item assumes the earlier ones.

**1. Read 18774002 in the registered order, not n-back first.**
`rsync -az -e "ssh -o BatchMode=yes" torch:/scratch/cl5625/mh-postfix/meta_harness/runs/iter11postfix/ meta_harness/runs/iter11postfix/`
Then: (a) the five untouched batch tasks inside their noise bands — this is the precondition
and voids the rest if it fails; (b) `variable_mapping`, expected ≈0.68 humanlikeness; (c)
n-back last. **The specific check that decides whether the turn-order fix worked: n=1
accuracy.** It was 0.9943 pre-fix and 0.4786 in the defective run; it must recover. Also
confirm n=1 blocks no longer end with ≤1 key (149 of 150 did before).

**2. Record it as the new reference generation.** Append to
`logs/postfix_baseline_outcome.md` (which currently documents the *defective* n-back run and
says so). Record in `logs/evolution_summary.jsonl` with an id prefixed so the three
measurement generations never merge.

**3. Re-measure run-to-run noise per task from the 3 repeats.** Resolves a live
disagreement: `run_to_run_floor.json` gives craft_task 0.0000–0.0031 while
`score_repeats.SAME_FAMILY_SD` gives 0.0158, and the post-fix craft delta of −0.0078 sits
between them. Every floor verdict depends on which is right.

**4. Apply the error-shape measures to the post-fix baseline.**
`python meta_harness/report_error_shape.py <run_dir>` — the first model-side numbers for
them, deliberately not computed before now. Their run-to-run spread from step 3 is what
decides which are promotable from report-only to guard or objective; that decision is the
user's.

**5. Land the bundled A4 + A1 documentation edits.** Held deliberately until a baseline
existed: strike "humans sit at 0.3728" from the A4 guard's message text in
`score_candidate.py`; re-justify its 0.15 cutoff from model-side evidence or the step-3
spread rather than from the void human value; apply `logs/doc_audit.md` group 2 (the A4 rows
in `HANDOFF.md`, `WORKLOG.md`, `domain_spec.md`, plus the A1 demotion and the new measures in
`domain_spec.md`'s axis table and `NOTES.md`).

**6. `word_recognition` stage 2** — `logs/instrument_fix_stage2_plan.md`. First the
prerequisite: confirm `src/score.py` divides the human side by trials *attempted* (humans
stop at 3 strikes, ~20 trials; the model runs 100). That check decides whether to replicate
the strike rule in bench or truncate analysis-side. Then implement one word per turn,
answer-then-store, `encode()` deleted — and re-baseline again.

**7. The protocol-mismatch audit** — a subagent was producing
`logs/protocol_mismatch_audit.md` when this was written; check whether it landed. Then decide
per item between changing bench, correcting analysis-side, or documenting. It contains the
deepest open question: `search_set.yaml` sets `temperature: 0.0` and the 50 "participants"
per task are 50 seeded *stimulus sets* run by one deterministic model, so the model's
between-participant spread is item difficulty while the humans' is between-person ability —
and the headline metric is a Wasserstein distance between those two. That may be a question
about whether the objective measures what the project claims, not a protocol detail.

**8. ~~Hermes job 18757709~~ — KILLED 2026-09-29 at 2h05 on user instruction.** It was
running **pre-instrument-fix bench** from the main checkout, so nothing it produced is
comparable to anything measured after `exp/compactor-prefix-v1`, and it was holding 2×H200
that `18774002` needed.

It left **partial output** at `/scratch/cl5625/meta-harness-compactor/meta_harness/runs/heldout/postfix/`
— 4 of 6 arms started (`baseline`, `baseline_rep2`, `respond_first_v2`,
`respond_first_v2_rep2`), 35 `.jsonl` files against 40 for four complete arms, so the last arm
is truncated mid-write. **Do not score it.** It is pre-fix, incomplete, and its per-arm
completeness is unverified. Not pulled down. Delete it or leave it; either way it is not
evidence. The question it was meant to answer — does hiding the tool schemas stop Hermes
emitting `"no response"` on n-back — is still open and would need a fresh run on post-fix
bench, which also means `respond_first_v2` would first have to be checked against the new
turn structure (candidates carry their own `step()` copies, so the bench fix does not reach
them).

**9. DECISION 2026-09-29 [USER]: the candidate search is not the work right now.** "Candidates
don't matter now that we have changed the instrument itself. Most of the candidates have just
been folded into the instruments." Correct, and the twelve split into two groups with different
futures — keep them distinct.

**Group A, folded into the instrument.** Five mechanisms across seven candidates, all now
baseline behaviour, all no-ops as candidates:

| mechanism | candidates | where it lives now |
|---|---|---|
| clear conversation history each turn | `episodic_reset`, `_v2`, `_v3`, `evicting_reset`, `respond_first`, `_v2`, `respond_only` | `step()` |
| let the model read its own store | (n-back had no read channel at all) | both turn-based tasks inject `to_recall_text()` |
| **answer before writing** | `respond_first` — its defining move | n-back answer-then-encode, 410ec2a |
| no tool schemas when calls are forbidden | `respond_first_v2` — its only change | eb3e96f |
| present test items one at a time | `serial_recognition` | planned, `logs/instrument_fix_stage2_plan.md` |

Note what this means about `respond_first`, the project's best candidate: its gain was the
benchmark being wrong in a way it happened to route around, and the corrected instrument was
first built doing the *opposite* until measurement forced the flip.

**Group B, never actually tested — NOT refuted.** `random_decay`, `full_context`,
`displacement`, `primacy`, `primacy_v2`, `chunk_limit`, `episodic_primacy` are genuine
psychological hypotheses (decay, chunking, primacy weighting, capacity), not instrument fixes.
Their near-zero verdicts are worthless because the tasks that would show their effects were
inert: `digit_span_reverse` exactly **+0.0000** for every candidate, six of eight tasks
contributing nothing outside 2 SE. `primacy` was judged without anyone measuring a
serial-position curve — the thing it predicts — which only got built 2026-09-29. Some become
testable for the first time once the instrument is right. `evicting_reset`'s eviction policy is
the borderline case: a store that evicts rather than refusing when full is arguably a design
choice, and it sits in `working_memory.py`, which no instrument fix has touched.

**Before any search resumes, settle whether the objective measures what the project claims** —
the temperature-0 / seeded-stimulus-set pseudo-participant problem in item 7. Resuming now
would repeat today's mistake one level up: optimising an artifact of the measurement instead of
a property of the harness. Then: instrument finished (n-back confirmed, word recognition
fixed), participant question settled, run-to-run noise measured so floors mean something, and
only then ask whether a Group B hypothesis is worth a GPU hour — starting with the ones whose
predicted signature the new error-shape measures can actually see.

**Commits today:** 70befa7 (instrument fix), 050afa1 (sbatch REPO), 38a89a6 (lead-in prompt),
71ea97f (nback_steps + five analyses), 574298c (--baseline bug), aad06c1 + d55ee04 (doc
group 1), d5d2497 (A4 invalid), 06c81c8 (A1 demoted), 3ecc47a (defective baseline outcome),
410ec2a (n-back answer-then-encode). Error-shape agent: 0c9cefa, e02e86a, 5f5c013, 41473f5,
407f710. All pushed to `myfork`.

---

# Earlier entries, 2026-09-28

## 2026-09-28T23:40Z [USER][CODE] — the instrument is being fixed; everything below is history

**Decision (user):** stop searching, fix the benchmark. Reason, measured in `dbd2ea3`: 99–106%
of every frontier candidate's mean-over-8 delta came from `nback` + `variable_mapping`, the
other six tasks contributed nothing outside 2 SE, and those two are exactly the tasks whose
stimuli never left the conversation context. The search was optimising plumbing.

**Landed (commit `70befa7`, pre-fix state tagged `exp/compactor-prefix-v1`):** `step()` clears
the transcript at the turn boundary, so only the KV store crosses a turn; `wm_nback` split into
encode + answer turns per letter with the store injected on both and `TASK_DESC_BY_N` restated
on the answer turn (it previously reached the model nowhere — `wm_system_prompt()` accepts
`task_prompt` and never uses it); `wm_variable_mapping` gains `WM_ENCODE_PROMPT` carrying the
store; `llm_openai.generate_with_tools` now sanitizes message content. Six offline tests in
`meta_harness/test_turn_boundary_reset.py`. Full change list and **pre-registered predictions**:
`meta_harness/logs/instrument_fix.md`. `_tool_call_cap()` untouched.

**n-back was write-only before this.** `TOOLS` has only `write_memory`/`delete_key`, no read
tool, and the turn was the bare string `"Next letter: X"` — the agent wrote keys it could never
read and answered from the transcript.

**No number from before `exp/compactor-prefix-v1` is comparable to one after it.** The twelve
candidate verdicts, the frontier, and all three baseline generations belong to the tag. Every
candidate overrides `step()` with its own copy, so the fix does not reach them; several exist
mainly to call `reset_messages()`, which the baseline now does. Do not run a candidate against
the post-fix baseline before checking it against the new turn structure.

**Running:** job `18767306`, Qwen, 3 post-fix baseline repeats, `ITER=10postfix` →
`runs/iter10postfix/baseline{,_rep2,_rep3}`. It runs from a **second cluster checkout**,
`/scratch/cl5625/mh-postfix` (git worktree at `050afa1`, `.venv` and `runs/` symlinked back),
because job `18757709` is still reading the main checkout and `candidate.sbatch` now takes
`REPO=<path>` for exactly this. Job `18757709` (Hermes, `respond_first_v2`, pre-fix bench) was
left running; it now only answers a Hermes behavioural question under an instrument since
declared broken.

**Deferred on purpose:** migrating the six batch tasks from `recall()` to
`step(..., allow_tools=False)` (instrumentation only, no candidate needs it yet); the
empty-content answer turn (may vanish now that n-back's answer turn is `allow_tools=False`).

## 2026-09-29T00:30Z [USER][CODE] — error-shape work, doc audit, word_recognition planned

**`word_recognition` fix planned, not landed** — `logs/instrument_fix_stage2_plan.md`. Lands
after `18767306` is measured. One word per turn, answer-then-store (deliberately the opposite
order from n-back, because the judged word must be visible), `encode()` deleted. Prerequisite:
confirm `src/score.py` divides the human side by trials *attempted*, since humans stop at 3
strikes (~20 trials) and the model runs 100. Correction recorded there: my mid-session claim
that `HANDOFF.md` described this leak wrongly was itself wrong — `word_list_text` and
`trials_text` are built from the same 100 lines, so the list really is printed twice.

**Error-shape measures for all 8 tasks, report-only** — user decision: compute, gate nothing,
decide promotion after one run gives a run-to-run spread per measure. A subagent is building
six measures into `error_structure.py`, `interference.py`, `protocol_match.py`, plus
`logs/human_error_shape.json`, `report_error_shape.py`, `test_error_shape.py`,
`logs/error_shape_measures.md`. **Do not edit those files while it runs.** Human per-item data
supports measures on 7 of 8 tasks; only `semantic_story_recall` lacks item structure.

**Doc audit** — `logs/doc_audit.md`. Group 1 fixed in `aad06c1`: `PROPOSER.md`'s
"`reset_messages()` is never called by any task" (false since 70befa7, and the reason four
candidates closed the same leak), `HANDOFF.md`'s route-3 "one thing to do next" (dead: 6 of 965
unanswered trials), the frontier marked as history, and `logs/bench_collection_fix.md` amended
to withdraw its claims. Groups 2 and 3 are pending the error-shape work and the
`word_recognition` fix respectively. `WORKLOG.md` and the 17 candidate manifests are
deliberately not rewritten — they record what each proposer knew at the time.

**Also fixed today:** `score_repeats.py --baseline` kept only its last occurrence
(`nargs="+"` without `action="extend"`), so multi-baseline scorings used one baseline.
`respond_first_v2` re-scored at 3 candidate vs 3 baseline repeats: n-back delta −0.0806, still
a FLOOR failure against −0.06 and still outside 2 SE (0.0091); recorded as
`postfix_respond_first_v2_3rep`. And every n-back analysis read the step log positionally,
which breaks on split rows — `meta_harness/nback_steps.py` now owns that mapping and five
scripts use it; `check_predictions.py`'s two n-back decompositions refuse a split row rather
than report a doubled count.


Written to survive a compaction. Read this, then `meta_harness/HANDOFF.md` (whose "one thing to
do next" is superseded by its own 2026-09-28 amendment), then
`meta_harness/logs/postfix_prediction.md`.

Branch `worktree-meta-harness-compactor`, HEAD **e46d6a1**, pushed to `myfork`. Working tree
clean apart from what this file adds.

---

## Units, stated once and to be restated in every message

Humanlikeness = **1 − Wasserstein-1** between the model's and the humans' per-participant score
distributions. Range 0–1, higher is more human-like, measured **in units of task
proportion-correct** (0.03 = three percentage points of distributional distance). A **delta** is
candidate score minus baseline score in those same units, and is the only quantity that can be
negative. The **floor is −0.03** per task: a candidate may lose at most three points on any
single task. Candidate and baseline are each averaged over **3 runs**.

The user asked explicitly for this: name the unit and the reference point for every number,
give baseline / new value / delta as separate columns, never assume the metric is recognised.
Saved as memory `feedback-state-units-explicitly`. `score_repeats.py` now prints all three
columns with the unit in its own header (commit 1633e2f) — that failure was mine, twice.

## Where the project stands

Goal: search harness code so an LLM's working-memory behaviour matches human score
distributions on 8 search tasks (+2 held out). The model is *more* accurate than humans on most
tasks, so "more human-like" usually means making it err as people do.

**Pre-fix Qwen results, all five candidates, 3-run candidate vs 3-run baseline.** Baseline mean
over the 8 search tasks = 0.7861.

| candidate | mean score | delta vs baseline | floors |
|---|---|---|---|
| respond_first | 0.8479 | +0.0618 | all pass |
| respond_only | 0.8428 | +0.0567 | all pass |
| evicting_reset | 0.8358 | +0.0497 | all pass |
| primacy | 0.7931 | +0.0070 | all pass |
| chunk_limit | 0.7805 | −0.0056 | all pass |

**Held-out (Hermes-4-70B) is where everything fails.** `respond_first` n-back 0.1417 against a
0.874 baseline, delta −0.7326: not degradation but cessation — `answered` 0.00 of 14 at all
three levels in all 150 blocks, `"no response"` on 99.1% of the 2100 turns where an answer was
due. `evicting_reset` −0.1377, `episodic_reset_v3` −0.2454. Nothing passes held-out.
Counterweight: `respond_first` takes Hermes variable_mapping from 0.3524 to 0.9533 (+0.6009),
with A4 error structure at normalized rc_ratio 0.3751 against the human 0.3728 — distance
0.0023 over 845 errors, the closest structural match in the project.

## ~~The single mechanism everything reduces to~~ SUPERSEDED 2026-09-28T04:10Z [TOOL]

> **Superseded, do not build on the struck text below.** The post-fix Qwen runs
> (`iter9postA`/`B`) falsified it: spoken tool calls on n-back answer turns went 553 of 2100 →
> **0 of 2100** while trials answered went **10.86 → 10.85 of 14**, and the baseline's own
> n-back humanlikeness did not move (0.7848 → 0.7838). Evidence and the replacement account:
> the OUTCOME section of `meta_harness/logs/postfix_prediction.md` and
> `meta_harness/logs/unanswered_cause.md`.
>
> **Replacement account.** An n-back trial goes unanswered because the model spends that turn
> on memory bookkeeping and emits no label — 418 of 472 unanswered turns returned **empty
> assistant content**, the rest returned prose about the write. Every unanswered trial's own
> turn had `tool_call_cap_hit` set (964 of 965) and no turn without it was ever unanswered
> (0 of 2339); the *cumulative* budget explains 6 of 965. And the sign of the old story was
> wrong: model n-back accuracy is 0.694 against humans' 0.8657, so an unanswered trial pushes
> the model **below** the human distribution. **Route 3 in `HANDOFF.md` is therefore dead**;
> routes 1 and 2 stand.

~~Tools get denied → the model types the tool call out as text instead of answering → that text
carries no classification → the trial is recorded unanswered → its score drops → since the model
is too accurate, the score *moves toward* the human distribution. An uncollected trial looks
like humanlikeness.~~

~~Cause: the harness sent `tools=TOOLS` with `tool_choice="none"` on turns that forbid calls. The
schemas render into the chat template regardless, so the model speaks the call.~~ (The
`tools`/`tool_choice` defect was real and is fixed in eb3e96f; it simply changes no score.)

## Work done today, with commits

- **eb3e96f `fix(bench)`** — the collection fix. `llm_openai`/`llm_anthropic` treat an empty
  `tools` list as "no tools this turn" and omit both `tools` and `tool_choice`;
  `wm_agent.step()`'s two duplicate no-tools branches merged into one passing `tools=[]`;
  `_strip_spoken_tool_calls()` keeps a spoken call out of the ANSWER while leaving it in the
  transcript the model sees; cap-dropped calls recorded in `refused_tool_calls`. Pinned by
  `meta_harness/test_no_tools_when_forbidden.py` (5 tests, offline, fake LLM).
- **378abd3 `feat`** — `candidates/respond_first_v2/`: the parent with ACT 1 sending `tools=[]`.
  13 non-comment lines differ; `test_respond_first_v2_equivalence.py` pins every one to the
  tool-offering path. The candidates build their own requests, so the `bench/` fix cannot reach
  them — `evicting_reset`, `respond_only`, `episodic_reset_v3` still carry the defect internally
  and have no v2.
- **c595a74 `fix`** — `heldout.sbatch` takes `TAG`; `test_heldout_paths.sh` checks both forms.
- **345cf36 `fix`** — `run_candidate.py --if-exists {fail,new,overwrite}`, default `fail`. It
  used to `mkdir(exist_ok=True)` and write regardless. Resolved before bench is imported.
  Manifests now record `git_head`/`git_dirty`. `test_out_dir_collision.py`, 5 tests. Both sbatch
  scripts pass `--if-exists new` and their own counters are deleted.
- **f8aba1e `data`** — three-run Qwen baseline (jobs 18754201/18754204, ~18 min each,
  `iter8repA`/`iter8repB`, deliberately **pre-fix** so they control for pre-fix candidates).
  Re-scoring cleared **every** floor violation in the project: primacy craft −0.0466 → −0.0233
  and narrative −0.0424 → −0.0278, so primacy now passes — the third rejection overturned on
  measurement grounds. Cause: `iter0/baseline` sat at the top of the craft range (0.8907 vs a
  3-run mean of 0.8674), pushing every candidate's craft delta down ~0.023.
- **1633e2f `docs`** — baseline / candidate / delta as separate columns, unit in the header.
- **e46d6a1 `analysis`** — `analyze_artifact_exposure.py` and the registered prediction.

Also added: `analyze_spoken_calls.py`, `analyze_frozen_tasks.py`, `rescore_three_baselines.py`,
`logs/bench_collection_fix.md`, `logs/postfix_prediction.md`.

## Queued jobs — check these first

| job | what | out dir | est |
|---|---|---|---|
| 18757709 | Hermes: `respond_first_v2` ×3 + `baseline` ×3, `TAG=postfix` | `runs/heldout/postfix/` | ~3h15 |
| 18757710/11/13 | Qwen: `respond_first_v2` + `baseline`, one server each | `runs/iter9postA/B/C/` | ~18 min each |

All were PENDING on `QOSMaxGRESPerUser` at last poll. Cluster repo
`/scratch/cl5625/meta-harness-compactor` is at **345cf36**; pull to e46d6a1 before the next
submission (fetch works from a compute node with remote name **`craaaa`**, not `myfork`). SLURM
snapshots the sbatch script at submit time, so these four use the old script text with the new
`run_candidate.py`; their target dirs are fresh, so the default `fail` policy does not trip.

**The one number that decides the whole account: the post-fix Qwen baseline's own n-back score.**
Predicted to rise from 0.7848 toward ~0.87. If it does not move, the artifact explanation is
wrong and the analysis below must be discarded.

Other registered predictions (`logs/postfix_prediction.md`, written before any data): respond_first
n-back delta halves +0.1755 → ~+0.09; its variable_mapping delta holds ~+0.31; its mean-over-8
advantage falls +0.0618 → ~+0.035; `evicting_reset`'s n-back advantage nearly vanishes
+0.0888 → ~+0.02. Hermes (`candidates/respond_first_v2/MANIFEST.md`): n-back delta ≥ −0.06,
`answered` ≥ 10 of 14 at n=1, variable_mapping ≥ +0.30. **If Hermes n-back does not recover, the
artifact account for the collapse is dead and no further ordering candidate is worth running
there.**

## What to distrust in my own conclusions

- **I got the direction of the collection defect wrong twice.** First I claimed the baseline lost
  ~549 of 1500 variable_mapping answers (it loses **0**; that figure was `evicting_reset`'s). Then
  I claimed the losses were inside candidates generally — backwards for n-back, where the
  **baseline** loses 1091 of 4200 answer turns (26.0%) and `respond_first`/`respond_only` lose
  **none**. The error came from counting `maintenance_text` as an answer turn. Count answer turns
  only: step *n+k* of an n-back block is scored trial *k*.
- Three rejections in this project have been overturned by fixing the measurement, not by new
  evidence about the candidate (3-repeat averaging twice, the 3-run baseline once). Treat any
  single-run or single-baseline verdict as provisional.
- On Hermes, three tasks are **inert**: craft 0.9340, digit_span_forward 0.9479,
  digit_span_reverse 0.9498, identical across all seven arms (`analyze_frozen_tasks.py` shows the
  score vectors are byte-identical, so this is a no-op, not a blind metric). The mean over 8 is
  partly a constant. Also, on Hermes the model is *worse* than humans on both digit spans
  (0.2925 vs 0.3442; 0.2435 vs 0.2949), so the project's "model too good on 7 of 8" framing does
  not hold there.
- The collected-only counterfactual assumes an affected trial would have been answered at the
  rate of the unaffected ones in its block. It bounds the artifact; it does not forecast a run.

## Open decisions, deliberately not taken

1. Does held-out passage gate frontier membership? Today's frontier (`respond_first`,
   `respond_only`, `evicting_reset`, `primacy`) is search-only; `respond_first` is measured
   catastrophic on Hermes. My inclination: keep the frontier search-only and record held-out as a
   separate gate, because emptying it discards a real Qwen result.
2. Whether `evicting_reset` / `respond_only` / `episodic_reset_v3` get their own `_v2` with the
   internal fix. Only `respond_first_v2` exists.
3. Whether the −0.03 floor is right at all, given craft's same-family run-to-run sd is 0.0158.

## Standing constraints

No money spend. Torch work on **compute nodes only** (CPU-only node is fine; never the login
node — reach it as `srun --account=torch_pr_287_cds -c 2 --mem=4G -t 5 bash -c '…'`). Never add
`--mail-type`/`--mail-user`. Author files with Write/Edit, never shell heredocs or `sed -i`.
Commit incrementally by path, never `git add -A`; backticks in `-m` get shell-substituted, so
use several `-m` flags. Keep `origin/main` as head; push to `myfork`
(git@github.com:craaaa/simulating-memory.git). Jobs export a dummy `OPENAI_API_KEY` and unset
`OPENROUTER_API_KEY`/`ANTHROPIC_API_KEY` so they cannot reach a paid endpoint. Proposers must not
read `runs/human/` and must not edit `bench/`, `data/`, `src/`, or `runs/`. `runs/` is gitignored,
which is why an overwritten run is unrecoverable. The session scratchpad directory announced
earlier is **gone**; write throwaway scripts into `meta_harness/` and commit them, or skip them.
Local venv: `/Users/cl5625/simulating-memory/.venv/bin/python`. Copy runs down with
`rsync -az -e "ssh -o BatchMode=yes" torch:…` — the `dtn` host has no usable key from here.

## Immediate next actions

1. Poll `squeue --me`. When the Qwen jobs land: `rsync` the run dirs, then check the **baseline's
   n-back score** against the 0.7848 → ~0.87 prediction before looking at anything else.
2. Score `respond_first_v2` against the post-fix baseline with `score_repeats.py` (3 candidate
   runs, 3 baseline runs) and compare each registered number in `logs/postfix_prediction.md` —
   state the prediction and the outcome side by side.
3. When the Hermes job lands, check `answered` per level first, then n-back humanlikeness, against
   the manifest's registered thresholds.
4. Record whatever comes back in `logs/evolution_summary.jsonl` with ids prefixed `postfix_`, so
   the two measurement generations never merge.
