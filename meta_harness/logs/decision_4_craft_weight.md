# Decision 4 RESOLVED — craft_task keeps 1/8 weight in the objective

**2026-09-29 [USER]: "keep 1/8 weight."** No change to `mean_humanlikeness_search`, no reweighting,
no demotion to pass/fail, no reduction in participant count. `craft_task` stays one of the eight
tasks averaged into the primary objective, exactly as before.

Written to its own file because `logs/open_decisions_brief.md` and `CONTINUITY.md` were being
edited concurrently by another agent. **Fold this into the brief's Decision 4 section and delete
this file.**

## What was proposed and rejected

Four options were on the table (`logs/stimulus_variation_ceiling.md`, `logs/open_decisions_brief.md`):
reweight the objective by resolution or report craft separately; cut its participant count; write
more stimuli; treat it as pass/fail. **All declined.**

My own earlier recommendation — drop craft_task from the primary mean — was **withdrawn before the
decision**, on the user's reframing, and the withdrawal is the substance of this entry.

## Why keeping it is right, in the user's framing

The user's point: *"The whole point of this meta harness is to come up with ways that would
generate more variation in the model responses, so that the model doesn't give a point estimate."*

`meta_harness/domain_spec.md` agrees by construction — a candidate is evaluated by *"comparing the
resulting score distribution to the human distribution."* Matching **dispersion** is therefore part
of the objective, not a side condition. So:

- A model emitting a point mass is failing the objective in the way this project most cares about.
- `craft_task` is where that failure is most visible: all 50 pseudo-participants produced the
  **identical** signature C2001 = 1.0, C2002 = 1.0, C2003 = 0.8 in `iter11postfix/baseline`. Human
  sd at the 15-question unit is **0.1180** over 8 distinct values; the model's is **0.0000** over 1.
- Removing it from the mean would have deleted the cleanest instance of the target phenomenon.

## The ceiling claim I got wrong, corrected

I reported craft_task's humanlikeness as "capped at ~0.9105, one bit of resolution." **That is the
ceiling for a point-mass model, not for the task.** A candidate producing human-like spread could
reach 1.0. Real headroom is **0.8679 → 1.0 ≈ 0.109**, second only to `word_recognition` — the
opposite of a reason to demote it.

## What remains true, and must travel with the number

None of the measured limitations are retracted; they are now documented limitations of a task that
stays in the objective:

- **3 stimuli, 2 distinct score values, 150 rows.** All 50 participants per item see byte-identical
  questions (verified by hashing). The banks are exhausted: `data/craft_task.json` has exactly 3
  items.
- **Its entire run-to-run variability is one question flipping.** C2003 Q04 ("Which pair can
  eventually produce D?") was wrong for 50/50 participants in repeat 1, 29/50 in repeat 2, 27/50 in
  repeat 3 — the whole of its 0.0342–0.0357 spread. The mechanism is visible in the store: the
  losing encoding keeps 3 rules and drops `A+B→D`, which is exactly what Q04 asks about; the winning
  one keys each rule by its own content and fits 4.
- **Accuracy improvement is penalised at the current level.** Perfect (15/15) scores 0.8130, below
  the actual 0.8690; repeat 3 was *more* accurate and scored *lower*. The optimum for a point-mass
  model is 13/15 at 0.9105. Any candidate improving craft accuracy without adding dispersion loses.
- **Audit M12 is rejected on measurement.** Pooling the model to the human's 15-question unit gives
  a single value with sd exactly 0.0000 in `baseline` — the state the audit itself called
  pathological. The granularity mismatch is real; pooling is not the remedy.

## The related finding this decision makes actionable

The adversary built to generate exactly this dispersion (`random_decay`) was measured at sd
**0.0329** against the human 0.1180, and only 2 distinct values. The audit
(`a03bec8a`, 2026-09-29) found why, and it is not that dispersion is unreachable:

- `_seed_from()` hashes the **stimulus content**, so on craft_task all 50 participants share a
  stimulus, share a hash, and therefore **share a decay rate** — 3 rates for 150 rows.
- The decay is a **complete no-op on 3 of 8 search tasks** (`nback`, `word_recognition`,
  `variable_mapping`), which call `step()` only and never `encode()`/`recall()`.

So "can a mechanism give craft_task human-like spread?" has never actually been tested. That is
what `random_decay_v3` is being built to answer. Keeping craft_task at 1/8 weight is what makes
that test worth running.
