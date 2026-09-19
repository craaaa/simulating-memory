# DECISIONS.md — answers to ANALYSIS_PLAN.md §0 (D1–D6)

Dated decisions. Do not change silently; superseding entries append with a new date.

---

## Project root

This analysis lives at `application/listening_qa/analysis/` (a subproject of the
`simulating-memory` repo), **not** at the repo root. The plan's layout (`R/`,
`config/`, `data/`, `outputs/`, `stimuli/`) is reproduced under that directory.
`make all` is run from there.

---

## D1 — Language for the confirmatory GLMM

**2026-07-30: R.** `lme4`/`glmmTMB` + `emmeans` + `TOSTER`.

- R 4.6.1 is installed; the modeling packages are **not** yet installed (installed
  at P4 / P3, logged in `outputs/logs/`).
- **Deviation from a pure-R pipeline, recorded here:** ingest (`01_ingest`) is done
  in **Python** (`py/01_ingest.py`), not R. Reason: the human data is a UTF-16
  Qualtrics TSV whose multi-select cells require the exact combinatorial
  `parse_response` matcher already implemented and validated in
  `application/prolific_study/multi_v6/data_prep.py`. Re-implementing that in R is a
  needless correctness risk. Python writes the canonical `data/processed/responses.parquet`
  (+ a `.csv` mirror); **all statistical modeling from `02_validate` onward is R**, reading
  the parquet/csv as the interchange. This is exactly the escape hatch D1 authorizes
  ("If the repo is Python, put R under … and use Parquet/CSV as the interchange").

## D2 — Confirmatory alignment axis

**2026-07-30: Effect alignment (§4).** The paper is about the passage
manipulation (control / repeat_short / repeat_long / distractor), so the
`agent × level` (and `agent × level × option_type`) GLMM interaction with
TOST equivalence testing is the pre-registered confirmatory analysis.

- Error alignment (§5) and difficulty alignment (§6) are **exploratory**.

## D3 — Model-sample framing

**2026-07-30: plan default adopted.**
- **Framing A** (model = one respondent; endorsement proportion per
  `(model_name, topic, level, option_id)` across its samples, binomial with
  `weights`, or disaggregated with `(1 | model_name/sample_idx)`) for **effect** (§4)
  and **difficulty** (§6) alignment.
- **Framing B** (each sample = a pseudo-participant, compared against each human)
  for **error consistency** (§5a) only.
- Model-side reliability reported separately (§7), never as individual differences.

## Agent / system factor (3-way; user directive 2026-07-30)

The data has three systems: `human`, prompting-`model`, compactor-`model__wm`. Replace the
binary `agent` factor with a 3-level **`system`** factor, **reference = human**. Per model,
fit on rows where system ∈ {human, prompting(M), compactor(M)} (humans shared across all
per-model fits). Two pre-registered confirmatory contrasts:

- **human vs prompting** — *expected different* → ordinary difference test.
- **human vs compactor** — *expected similar* → **equivalence test (TOST)**, uses D4 bound.

So the compactor is the "human-like" hypothesis; the equivalence bound (D4) gates that claim.
GLMM formula updated in `config/analysis.yaml`: `system` in place of `agent`.

## D4 — Equivalence bounds for TOST/ROPE

**2026-07-30: bounds set before any real-data fit, and NOT derived from the
confirmatory dataset (multi_v6).** Because multi_v6 is the confirmatory human set
(see D5), the equivalence bound is estimated from an **earlier human pilot**
(`application/prolific_study/multi_v5` / `pilot_v4`, whichever has a usable
condition effect) or from literature — never from multi_v6.

- Rule (plan default): bound = ±(0.5 × smallest human condition effect in log-odds),
  where the "smallest human condition effect" is taken from the pilot fit.
- The exact numeric bound is computed and **frozen in `PREREG.md` (P3), before the
  first multi_v6 fit.** Recorded in `config/analysis.yaml` as `equivalence_bound_logodds`
  with its provenance. Placeholder until then: `null` (must be non-null before P4 runs).

## D5 — Text-modality human control batch

**2026-07-30: No. multi_v6 (`results.tsv`) is the FINAL confirmatory human set;
audio only.**
- `modality` column exists in the schema from day one but is constant `audio` for
  all humans and `text` for all models.
- Consequence: every human↔model divergence is (modality + agent), unseparated.
  The write-up scopes the claim to **"text-model vs audio-human"** (§7 modality
  confound is a caveat, not a measurement). The prosody-cue-leakage diagnostic (§7)
  is still run within the human-audio data.

## Human attention exclusion

**2026-07-30 (user directive):** use only humans who passed the attention check in
**all 4 topics** (whole-participant exclusion — the `longdata_strict` rule), applied at
ingest. Result: **201 humans** retained. `reconcile.py` matches `longdata_strict.csv`
`correct`/`partial` 1:1 (4020 records) as a correctness proof of the option-level join.

## D6 — Model set (never averaged; one fit per model)

**2026-07-30 (revised, per user):** the model set = the **paired prompting +
streaming-compactor** models from the released comparison plots
(`application/comparisons/listening_level_pair_alignment_*.json` meta). Every model
uses the **C2 condition family**: prompting side = **C2 (HumPr)**; compactor side =
**C2-stream** (streaming segment-by-segment encode). Each model appears twice in the
parquet as distinct `model_name`s — `<name>` (system=`prompting`) and `<name>__wm`
(system=`compactor`) — and is fit separately. No cross-model averaging.

Superseded earlier same-day defaults: model condition C1 → **C2**; the C1 batch-encode
compactor runs (gpt-4.1-mini, llama-3.1-8b) and prompting-only models (gemini-3.6-flash,
gemini-3.1-pro) are **dropped** — they are not streaming.

Confirmatory set — 5 clean pairs, n=20 repeats/cell both sides, full C1–C4 prompting grid:

| name | prompting slug | compactor dir (C2-stream) |
|---|---|---|
| Command-A | cohere_command-a | CohereLabs_c4ai-command-a-03-2025 |
| Qwen2.5-32B-Instruct | Qwen_Qwen2.5-32B-Instruct | Qwen_Qwen2.5-32B-Instruct |
| Qwen2.5-72B-Instruct | qwen_qwen-2.5-72b-instruct | Qwen_Qwen2.5-72B-Instruct |
| Gemma-4-31B-it | google_gemma-4-31b-it | google_gemma-4-31B-it |
| Kimi-K2-0905 | moonshotai_kimi-k2-0905 | moonshotai_kimi-k2-0905 |

Exploratory — **GPT-4.1** (`confirmatory: false`). Its compactor tree has ~20
colliding-id run dirs that must never be merged (project memory). The released headline
JSON pointed at `20260720T211617Z` (only n≈3/cell — a pilot); we instead name the n=20
streaming run `n20_stream_sentence_cap30` explicitly, marked exploratory pending user
confirmation of the intended dir.

All paths are explicit in `config/models.yaml`; nothing downstream hard-codes the set.
