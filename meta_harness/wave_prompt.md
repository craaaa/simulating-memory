# Wave prompt template

Given to a fresh proposer each wave. Keep it short: the point of Meta-Harness is
that the proposer goes and reads the filesystem itself rather than being handed a
compressed summary, so this says where to look and what the rules are, not what
to conclude.

Substitute `{ITER}`, `{N_REMAINING}`, and `{RESULTS_TABLE}`.

---

You are proposing iteration {ITER} of a Meta-Harness search over the compactor in
this repo. **{N_REMAINING} proposer iterations remain in total** — they are
budgeted, so make this one count rather than trying something safe.

## Read this first

`meta_harness/PROPOSER.md` — the objective, the constraints, the axes and their
numbers, and the traps. It is short and all of it matters. In particular: the
objective is *calibrated degradation*, not capability; matching the score
distribution with per-participant noise is worthless and already implemented as a
control; and you must not read `runs/human/`.

## Where the evidence is

    python meta_harness/history.py list          # every candidate so far
    python meta_harness/history.py show <id>     # full record for one
    python meta_harness/history.py diff <a> <b>  # per-task delta, noise marked
    python meta_harness/history.py regressions <id>
    python meta_harness/history.py trace <id> --task word_recognition --n 0

Read actual traces before proposing. `meta_harness/runs/iter*/<id>/tasks/*.jsonl`
carries, per participant, the `encoding_log` with every tool call the harness
made, the `final_kv` it ended with, and the `recall_raw` it produced. That is
where the mechanism is visible; the scores only tell you that something is wrong,
not what.

## Results so far

{RESULTS_TABLE}

## What to produce

1. A hypothesis about a **specific** failure you can point to in the traces.
   Name the participant and task you saw it in.
2. `meta_harness/candidates/<id>/harness.py`, overriding any of
   `WorkingMemoryAgent`, `MAX_KEYS`, `TOOLS`, `CONDITION_PROMPTS`,
   `WM_SYSTEM_PROMPTS` (see `meta_harness/inject.py`). Include a `MANIFEST` dict
   with `id`, `parent`, `capacity`, `decay`, and a `summary` stating what you
   changed and why. If you touch capacity or add decay, the manifest must carry
   the psychological argument — a score is not an argument.
3. Verify it offline, which is free and needs no GPU:

       python meta_harness/verify_interface.py meta_harness/candidates/<id>/harness.py

4. Write `meta_harness/logs/pending_eval.json` as
   `{"iteration": {ITER}, "candidates": ["<id>"]}`.

Do not run anything on the GPU yourself and do not edit anything under `bench/`,
`data/`, `src/`, or `runs/`. Evaluation is handled outside this session.

## Things already known, so you do not spend the iteration rediscovering them

- The released `runs/compactor/*` numbers came from OpenRouter and do **not**
  reproduce under local bf16 serving, so compare only against the locally
  measured baseline in the history.
- A2 (word-recognition miss/false-alarm asymmetry) is the only axis with real
  headroom on this model. A1 and A3 are already near-human and act as guards.
- A single-task delta under ~0.05 is inside the human noise floor. `diff` marks
  these. Do not build a story on one.
- N-Back is the one task where the model is *worse* than humans, so it opposes
  every other task's gradient.
