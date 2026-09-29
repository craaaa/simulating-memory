# Predictions for job 18804444 — `random_decay_v3`, registered while the job was PENDING

Submitted 2026-09-29 from `/scratch/cl5625/mh-postfix` at commit **91cc76f**.
`CAND="random_decay_v3 random_decay_v3 random_decay_v3"`, `ITER=13` →
`runs/iter13/random_decay_v3{,_rep2,_rep3}`. Written and committed while the job was
`PD (Priority)`, before any output existed.

## Units

**Humanlikeness** = 1 − Wasserstein-1 between the model's and the humans' per-participant score
distributions, in [0,1]. **Decay rate** is a per-key per-turn drop probability, a proportion in
[0, 0.35]. **Error-class shares** sum to 1.000 over their classes. **Rolls** and **keys dropped**
are integer counts. None of these is humanlikeness except the first.

## What this run is, and what it is not

`random_decay_v3` is the project's **adversary**, not a contender. Its job is to match the human
score distribution through psychologically empty per-participant noise. `domain_spec.md`:

> If it cannot match the distribution, it is a weak adversary and the axes are untested rather
> than validated; if it matches the distribution *and* the axes, the axes do not discriminate and
> the search is invalid.

**Both prior attempts were void as tests.** Measured in the audit: the decay was a **complete
no-op on `nback`, `word_recognition` and `variable_mapping`** (those tasks call `step()` only,
never `encode()`/`recall()`), and had **3 rates for 150 rows** on craft, 4 for 200 on story
recall, 10 for 50 on narrative — because the RNG was seeded from *stimulus content* and those
tasks share stimuli across participants. So the Wave 0 verdict *"the axes discriminate, search is
meaningful"* rests on an adversary that was absent from three tasks and nearly constant on three
more.

v3 fixes both: hooks `step()`, rolls once per tool-enabled turn, seeds from a threaded
`participant_id`. Verified offline and **again on the cluster** before submission — every task
≥1 roll, ≥1 key dropped, distinct rates == scored units, baseline arm 0 rolls throughout.

## The pass condition, as amended

The registered condition had two legs. **The second is withdrawn** — see
`logs/a2_threshold_resolution.md`:

- **Leg 1, unchanged:** mean humanlikeness ≥ baseline + 0.05.
- ~~**Leg 2:** A2 distance ≥ 2× baseline's.~~ **WITHDRAWN.** It required ≥ 11.57, which is 1.9× the
  entire distance from human (6.094) to the ratio's floor (0), and **noise moves A2 toward human,
  not away** — dropping keys raises misses and lowers false alarms. Measured: `random_decay`
  5.4288 and `random_decay_v2` 5.4344 against baseline 5.754, both ~0.32 *closer*. A2 cannot be
  this adversary's guard.
- **Leg 2 replacement:** the adversary must be caught by a named error-shape measure by **more
  than that measure's own measured run-to-run spread**. No such spread exists yet, which is why
  this run is **3 repeats** — the repeats are what establish it.

## P1 — does the noise now reach the three previously dead tasks?

**Hypothesis.** The v1/v2 verdicts were uninformative on `nback`, `word_recognition` and
`variable_mapping` because the decay never fired there, not because those tasks are insensitive.

- **Supports:** each of the three moves from its `iter12stage2` baseline by more than
  `max(NOISE_FLOOR, RUN_TO_RUN_SPREAD)` for that task — nback 0.060, word_recognition 0.121,
  variable_mapping 0.017.
- **Rejects:** all three stay inside those bands with the decay demonstrably live.

A rejection is the **more interesting** outcome: it would mean the tasks are insensitive to store
contents, which is a separate and more serious finding. `score_candidate.py` already classifies
`nback` and `variable_mapping` as LEAKY — answerable without the store — so a null there is
plausible and would corroborate that classification rather than contradict it.

Baseline references (`iter12stage2`, 3 arms, current scoring incl. Option D for n-back):
nback **0.9633**, word_recognition **0.8278**, variable_mapping **0.9662**.

## P2 — leg 1: can it match the distribution at all?

Baseline mean over 8 = **0.9167** (`iter12stage2`, post-Option-D). Leg 1 needs **≥ 0.9667**.

- **Supports leg 1:** mean over 8 ≥ 0.9667.
- **Rejects:** mean over 8 below baseline, i.e. the adversary overshoots as v1 did (0.6977 against
  a 0.7861 baseline) and is again too weak to test the axes.

**I expect rejection**, and the reason is arithmetic rather than pessimism: the baseline is
already at 0.9167, so leg 1 demands a mean of 0.9667 when the *attainable ceiling* across tasks is
about 0.955 (`NOTES.md`). **Leg 1 may be unsatisfiable on the current instrument.** If so, that is
itself a finding — the pass condition was written when the baseline sat at 0.7861, and fixing the
instrument moved the goalposts out of reach. Record it rather than quietly re-fitting the bar.

## P3 — the replacement leg: does M1 catch it?

**M1** is the `variable_mapping` intrusion-class measure: of the model's errors, what share name
the person's own **stale** city, another person's city, or a city never mentioned. Human:
`stale_same_name` **0.2303**, `intrusion_other_name` 0.4605, `novel_guess` 0.3092.
`iter12stage2` baseline: **0.1718 / 0.6651 / 0.1631** over 1275 errors.

**Hypothesis.** Naming a person's own superseded city requires that stale binding still to be
represented. Random key-dropping destroys it outright, so noise should drive `stale_same_name`
toward 0 and `novel_guess` up — a signature noise cannot fake.

- **Supports:** `stale_same_name` falls below the baseline's 0.1718 by more than its 3-repeat
  spread, measured from this run's own arms.
- **Rejects:** it stays at or above baseline.
- **INCONCLUSIVE, and read this first:** M1 returns `shares={}` when a run has fewer than 30
  errors. **Read `n_errors` before the shares.** v1 and v2 had 12, 12 and 14 errors — M1 was never
  computable on the runs the adversary was previously judged on. Under v3 the count could move
  either way: more wrong answers (usable) or more *unparse-able* ones (which `score_game` scores
  as neither correct nor an error, so they do not raise `n_errors`).

**M2 is NOT a candidate and must not be quoted as one.** I proposed it — digit-span typology,
reasoning that random dropping yields truncation/omission where humans substitute. **The baseline
already has that signature**: human substitution 0.4125 against the baseline model's 0.0808, with
baseline truncation 0.3131 and omission 0.3737. The pattern I attributed to noise is the
baseline's own, so M2 cannot discriminate v3 from baseline in the hypothesised direction.

## P4 — severity: is `DECAY_MAX = 0.35` calibrated for a per-turn regime?

0.35 is v2's value, chosen so the five batch tasks stay **numerically identical to v2** (one
`encode()` → one tool-enabled step → one roll), isolating the change to the defect. The
turn-based tasks have no v2 behaviour to preserve because v2 was a no-op there.

Per-key survival `(1−p)^R` at p=0.35, R = tool-enabled rolls:

| task | R | survival of a never-refreshed key |
|---|---|---|
| batch tasks (5) | 1 | 0.650 |
| nback | 15 / 16 / 17 | 0.0013 / 0.0009 / 0.0006 |
| variable_mapping | 20 | 0.0002 |
| word_recognition | up to 100 | ~0 |

`(1−p)^R` overstates severity on turn-based tasks, because the store is rewritten each turn — the
operative quantity is the refresh horizon **1/p = 2.9 turns**, i.e. a short recency window rather
than an empty store. And on `word_recognition` **R is endogenous to p**: higher p → earlier errors
→ earlier third-error stop → fewer rolls. So R=100 is an upper bound, not a prediction.

- **Rejection criterion, pre-registered:** if `word_recognition` survival falls below the human
  **minimum** of 4 of 100 trials for most scored units, 0.35 is too high for the per-turn regime,
  and a v4 should hold *total* expected loss constant rather than the per-turn rate.

## P5 — cost

`iter12stage2` ran 0.50 h per arm at these sizes. Three arms ≈ **1.5 h** plus ~10 min vLLM
startup, inside `--time=08:00:00`. One job, one vLLM server. Local serving, no paid API.
If it runs past ~4 h, suspect the `word_recognition` R-endogeneity above.

## What a null result means here

The most likely single outcome is **"untested rather than validated"**: leg 1 unsatisfiable
against a 0.9167 baseline, and M1 either uncomputable (n_errors < 30) or not moving. That is not a
wasted run — it would establish, for the first time on a working adversary, that **the current
axis set cannot be shown to catch psychologically empty noise.** Given A1 retired, A4's human
reference void and A2 withdrawn as this adversary's guard, that conclusion is already half-written;
this run is what would make it evidence rather than inference.
