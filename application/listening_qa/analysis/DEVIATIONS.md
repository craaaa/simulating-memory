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

---

## 2026-07-30 — GLMM structure: 2-level split fits + per-option_type contrasts (post-data)

**Frozen spec (PREREG):** one 3-level `system` {human, prompting, compactor} GLMM per model,
contrasts marginal over option_type.

**Deviation (EXPLORATORY for affected contrasts):**
1. **Prompting models are completely separated** — they endorse 99.8–100% of true options
   and 0.0% of false options (per-model rates verified). ML log-odds are non-identified there
   (contrast CIs ±thousands). The 3-level joint fit propagated this into the compactor contrasts.
   → Fit **two 2-level `system` models per underlying model**: (human+prompting) and
   (human+compactor). Prompting's separation is then isolated and reported as a **finding**
   ("categorically non-human: ceiling accuracy, zero false endorsement"), not a number to test.
2. **Do not marginalize the contrast over option_type** — a single ceiling stratum drives the
   marginal to ±Inf and destroys the SDT decomposition. Extract system×level contrasts **within
   each option_type** (`system_level_contrasts_by_ot`), flag separated strata (|est|>10 or SE>10).
   The equivalence (compactor−human) evidence is carried by the estimable strata (false options
   are non-degenerate: 0.03–0.09).
3. **No shrinkage prior on the compactor fit.** A Gaussian prior shrinks the contrast toward 0
   = toward "equivalent" = anti-conservative for the compactor≈human hypothesis under test.
   command-a's compactor is itself near-ceiling on true (0.967) → its true-stratum contrast is
   reported as ceiling-limited rather than penalized into estimability.

**Recovery re-validated (plan §9.6):** `03_simulate.R` extended with a ceiling/separation
scenario (one system ~99% true / ~0% false, known interaction on the false stratum). Gate asserts
the true stratum is flagged separated (≥90%) and the false-stratum interaction is still recovered
within CI (≥85%). The refit is trusted only after this passes.

**Status:** the confirmatory *axis* (effect alignment) stands; the specific numeric spec changed
post-data, so the GLMM contrasts are reported **exploratory**. The substantive conclusion is led
by the descriptive endorsement rates (separation-free), with the GLMM as rigor on top.
