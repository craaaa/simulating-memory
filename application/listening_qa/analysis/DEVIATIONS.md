# DEVIATIONS.md

Dated, justified departures from ANALYSIS_PLAN.md / PREREG.md. Any result affected by a
post-hoc deviation is reported as **exploratory** (plan §9.2). Pre-fit deviations that are
themselves pre-registered (frozen before the first multi_v6 fit) do not taint confirmatory
status — they are recorded here for transparency.

---

## 2026-07-30 — Equivalence bound Δ: "smallest reliable" instead of literal "smallest"

**Plan wording (D4):** Δ = 0.5 × *smallest* human condition (level) effect in log-odds.

**Deviation:** used the smallest **statistically reliable** level effect (repeat_long,
0.4426, p=0.005) rather than the literal smallest (distractor, 0.11, p=0.43).

**Why:** in the multi_v5 pilot (n=21) the distractor manipulation's effect on per-option
correctness is indistinguishable from zero (p=0.43). Anchoring Δ to a null effect gives
Δ≈0.056, a degenerate band under which no compactor could ever be declared equivalent to
humans regardless of true similarity — defeating the purpose of the equivalence test. The
plan's rule tacitly assumes each manipulation has a real effect; distractor does not here.

**Status:** this is a **pre-fit, pre-registered** choice — frozen in `config/analysis.yaml`
and `PREREG.md` before any multi_v6 fit. It does not make the confirmatory result
exploratory. User-approved 2026-07-30.

**Robustness to report at P4:** also report the equivalence verdict under the literal
Δ=0.056 as a sensitivity row, so the sensitivity of the compactor≈human claim to the bound
is visible.
