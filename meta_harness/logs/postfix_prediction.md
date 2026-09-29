# Pre-registered prediction for the post-fix runs

Written 2026-09-28, before jobs 18757709 / 18757710 / 18757711 / 18757713 produced any data.

**Units.** Every score is humanlikeness = 1 − Wasserstein-1 between the model's and the
humans' per-participant score distributions. Range 0–1, higher is more human-like, in units of
task proportion-correct (0.03 = three percentage points of distributional distance). A `delta`
is candidate score minus baseline score in those same units and is the only quantity that can
be negative. Candidate and baseline are each averaged over 3 runs.

## Where the uncollected trials actually are

Measured with `meta_harness/analyze_artifact_exposure.py`, counting **answer turns only** —
step *n+k* of an n-back block is scored trial *k*, and `maintenance_text` is not an answer:

| arm | n-back answer turns with a spoken call | variable_mapping answers with a spoken call |
|---|---|---|
| **baseline** | **1091 / 4200 (26.0%)** | 0 / 4500 |
| respond_first | 0 / 6300 | 6 / 4500 |
| respond_only | 0 / 6300 | 6 / 4500 |
| evicting_reset | 120 / 6300 (1.9%) | 10 / 4500 |
| primacy | 203 / 4200 (4.8%) | 0 / 4500 |
| chunk_limit | 1032 / 4200 (24.6%) | 0 / 4500 |

This **corrects what I reported earlier today.** I said the losses were inside the candidates
and that the baseline lost almost nothing. That holds for variable_mapping — the baseline loses
0 of 4500 there — but on n-back it is the other way round: the baseline loses a quarter of its
answer turns, and `respond_first` and `respond_only` lose none, because their answer turn is
issued separately from their memory-update turn. The earlier count that produced my wrong claim
included `maintenance_text`, which is not an answer turn, and so attributed the candidates'
memory-update chatter to their answers.

## What that implies, and the prediction

Scoring only the trials that were actually collected (`correct / (trials − affected)`):

| arm | n-back delta as scored | n-back delta collected-only | of the gain, artifact |
|---|---|---|---|
| respond_first | +0.1755 | **+0.0891** | +0.0864 |
| primacy | +0.1620 | **+0.0764** | +0.0856 |
| respond_only | +0.1437 | **+0.0573** | +0.0864 |
| evicting_reset | +0.0888 | **+0.0191** | +0.0697 |
| chunk_limit | −0.0090 | −0.0041 | −0.0049 |

| arm | vm delta as scored | vm delta collected-only | of the gain, artifact |
|---|---|---|---|
| respond_first | +0.3402 | **+0.3052** | +0.0351 |
| evicting_reset | +0.3417 | +0.3052 | +0.0365 |
| respond_only | +0.3137 | +0.3003 | +0.0134 |

**Predictions for the post-fix Qwen runs (18757710/11/13), stated before the data exists:**

1. The baseline's own n-back humanlikeness **rises**, from 0.7848 toward ~0.87, because its
   answer turns are now collected. This is the load-bearing prediction: if the baseline does
   not move, the artifact account is wrong.
2. `respond_first`'s n-back delta **falls by roughly half**, from +0.1755 to about +0.09.
3. Its variable_mapping delta **mostly survives**, from +0.3402 to about +0.31.
4. Its mean over the 8 search tasks therefore falls from +0.0618 over baseline to about
   **+0.035**, with the baseline mean rising from 0.7861 to about 0.797.
5. `evicting_reset`'s n-back advantage nearly disappears (+0.0888 → about +0.02).

**Prediction for the post-fix Hermes run (18757709)**, already registered in
`candidates/respond_first_v2/MANIFEST.md`: n-back delta ≥ −0.06 against the 0.874 baseline,
`answered` ≥ 10 of 14 at n=1, variable_mapping ≥ +0.30.

## Why this is an estimate and not a forecast

The counterfactual assumes an affected trial would have been answered at the same rate as the
unaffected ones in the same block. The fix changes what the model sees, so behaviour can shift
for other reasons — the budget is spent differently, the store ends up in a different state.
Read the collected-only column as *how much of the gain cannot survive proper collection*, not
as the number the new run will print.

Note also that both quantities move the same way for the frontier's ordering: `respond_first`
stays ahead of `respond_only` and `evicting_reset` on both columns. The artifact inflates the
size of the gains, not their rank.

---

# OUTCOME, same day: the load-bearing prediction failed and the artifact account is dead

Jobs 18757710 and 18757711 (Qwen, post-fix, `iter9postA` / `iter9postB`, each running
`respond_first_v2` and `baseline` against one server). Two repeats per arm; 18757713 still
running.

| prediction | predicted | measured | verdict |
|---|---|---|---|
| 1. baseline's own n-back score rises | 0.7848 → ~0.87 | **0.7838** (0.7857, 0.7820) | **FAILED** |
| 3. variable_mapping delta mostly survives | ~+0.31 | **+0.3394** | held |
| 4. mean-over-8 advantage | ~+0.035 | +0.0285 | roughly held, wrong reason |
| 2. n-back delta halves to ~+0.09 | +0.09 | **−0.0784** for v2 | failed, worse than predicted |

Prediction 1 was the one I said would decide the account, and it is unambiguous: the baseline
did not move. The mean over the 8 search tasks is **0.7861 post-fix, identical to 0.7861
pre-fix.**

**Why it failed, and it is not subtle.** The fix worked mechanically — spoken tool calls on
n-back answer turns went from 553 of 2100 to **0 of 2100** — but `mean_answered` went from
**10.86 of 14 to 10.85 of 14.** The spoken calls were never the unanswered trials. The model
was typing a tool call *alongside* an answer that the parser found anyway, so nothing was ever
being lost. My whole "an uncollected trial looks like humanlikeness" account rested on assuming
those two sets coincided, and they do not.

**What the run did reveal, which is more interesting.** `respond_first_v2` differs from its
parent in one respect — the answer act no longer shows the model tool schemas — and its n-back
collapses:

| arm | n-back humanlikeness | n-back model mean accuracy | trials answered |
|---|---|---|---|
| baseline (post-fix) | 0.7838 | 0.694 | 10.85 of 14 |
| respond_first (pre-fix parent) | 0.9603 | ~0.86 | 14 of 14 |
| **respond_first_v2 (post-fix)** | **0.7054** | **0.588** | **14 of 14** |

Human mean accuracy is 0.8657, so the parent lands almost exactly on the human distribution and
v2 undershoots it badly. v2 answers *every* trial and gets far more of them wrong. So the
parent's +0.1755 n-back gain is **real, not an artifact — and it depends on the tool schemas
being visible during the answer act.** The plausible mechanism: the schemas are where the model
learns what its key-value store is for, so removing them mid-episode leaves the store contents
in the prompt without the interface that explains them.

`respond_first_v2` therefore **fails**: n-back −0.0784 against a −0.06 floor, outside 2 SE
(0.0112), plus craft −0.0303 against −0.03 which is inside the noise. Its variable_mapping gain
of +0.3394 is intact, which is consistent with variable_mapping never having had spoken calls in
the baseline at all.

**Consequences.**
- The `bench/` collection fix (eb3e96f) is still correct as a measurement — an answer should not
  contain a typed-out tool call — but it changes no score, and I oversold it.
- Withdraw the claim that the top candidates' gains are "roughly twice too optimistic". They are
  not inflated by uncollected trials. The pre-fix Qwen numbers stand as measured.
- The Hermes job 18757709 is still worth running, for a different reason than it was queued: it
  now asks whether removing the schemas stops Hermes emitting `"no response"` — a question about
  Hermes' behaviour, not about a scoring artifact.
- Open question worth its own candidate: if visible-but-forbidden schemas are load-bearing for
  the parent's n-back accuracy, what exactly does the model take from them? A candidate that
  keeps the schemas visible and states the store's purpose in the prompt text would separate
  "the model needs the interface description" from "the model needs to see tools it cannot use".
