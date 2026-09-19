# Gemini integration: bugs found, screening results, methodology notes

Companion to `COMPACTOR_ITERATION_NOTES.md` (gpt-4.1 prompt iteration) and
`OPEN_SOURCE_MODEL_COMPARISON.md` (open-weight model sweep). This covers what changed
to get Gemini models working through the same pipeline, and what was found once they did.

## Two real bugs in the OpenAI-compat path, both Gemini-specific

Both were found running `gemini-3.1-pro-preview` through the WM compactor
(`application.listening_qa.wm_full_grid_cli`), which — unlike prompting — requires
multi-turn tool use.

### 1. `thought_signature` not round-tripped (broke every Gemini compactor run)

Gemini's OpenAI-compatible endpoint attaches an opaque `extra_content.google.thought_signature`
blob to each tool call. The API requires that blob be echoed back verbatim on the assistant
message in the next turn — if it's missing, every tool call past the first fails with
`400: Function call is missing a thought_signature in functionCall parts`.
`bench/core/llm_openai.py`'s `generate_with_tools()` was rebuilding tool_calls into a plain
`{id, type, function}` dict, silently dropping `extra_content`. Fixed by round-tripping
`extra_content` unconditionally when present (no-op for every other backend, verified this
does not break OpenAI/OpenRouter/Anthropic tool calls).

### 2. `recall_max_tokens=512` truncated every Gemini recall (silent, no error)

Gemini 3.x are reasoning models — hidden "thinking" tokens are drawn from the same
`max_completion_tokens` budget as the visible answer, but are NOT reflected in the
`completion_tokens` usage field (only `total_tokens` reflects them; confirmed by cross-checking
against the native Gemini API's explicit `usageMetadata.thoughtsTokenCount`, which matches
`total_tokens - prompt_tokens - completion_tokens` exactly). With `wm_mcq_common.py`'s old
hardcoded `recall_max_tokens=512`, the model's hidden thinking ate most of the budget, and
recall answers were silently truncated mid-question (every trial answered only Q1-Q2 of 5,
with **zero parse errors** — the truncated output was still valid syntax, just short). This is
what produced an initial "18.8% exact-match" compactor result that looked like a real
comprehension failure but was actually the whole downstream 3-5 questions never being
attempted. Fixed by adding `--recall-max-tokens` (default 2048) to `wm_full_grid_cli.py`,
replacing the hardcoded 512. **Any reasoning-model compactor run should sanity-check that
`len(parsed_answers) == len(questions)` for every row before trusting the accuracy number** —
truncation doesn't show up as a parse error.

## Cost-tracking gap this exposed (fixed at the same time)

Gemini's OpenAI-compat `usage.completion_tokens` excludes those same hidden reasoning tokens,
but they ARE billed (folded into `total_tokens`). `bench/core/llm_openai.py` now tracks
`total_tokens_reported` separately and exposes `billable_completion_tokens = completion_tokens
+ hidden_reasoning_tokens` (where `hidden_reasoning_tokens = total_tokens_reported -
prompt_tokens - completion_tokens`, floored at 0). Cost estimates now bill off this instead of
raw `completion_tokens`. Checked: OpenAI/OpenRouter reasoning models (tested via OpenRouter's
`qwen/qwen3-30b-a3b-thinking-2507`) don't have this gap — their `completion_tokens` already
includes reasoning tokens (o1-style accounting), so the fix is a no-op there. Real-world impact:
a `gemini-3.1-pro-preview` compactor pilot's cost estimate went from $2.35 (wrong, using
`completion_tokens` alone) to $5.08 (right, same run) once this was fixed.

Also added: OpenRouter's actual per-response billed `$` (`resp.usage.cost`, requested via
`extra_body: {"usage": {"include": true}}`) is now preferred over any list-price estimate when
available — exact, not computed. See `bench/cost_report.py` for the cross-run spend aggregator
and the `$1000` Gemini budget tracker (memory: `reference_gemini_budget`).

## thinking_level: can't disable reasoning, but LOW helps

Gemini 3.1 Pro cannot fully disable thinking (`thinking_budget=0` → `400: This model only works
in thinking mode`). The correct current param is `thinking_level` (`low`/`medium`/`high`) via
`extra_body: {"google": {"thinking_config": {"thinking_level": "low"}}}` — **not** the older
`thinking_budget`, and the two can't be combined. API defaults to HIGH if unset.

On an n=2 pilot, `low` vs default HIGH: **cheaper ($3.60 vs $5.08) AND slightly higher accuracy
(66.9% vs 64.4%)** — strictly better, not just a cost/quality tradeoff. Used `low` for the full
n=20 run. No exploration yet of `medium` — HIGH and LOW are the only two points measured.

## Screening results (bar: `feedback_model_screening_bar` — errors confined to ONE
(topic,level) cell, not recurring across topics/levels)

| model | prompting ceiling (n=5 or n=20) | verdict | notes |
|---|---|---|---|
| **gemini-3.1-pro-preview** | 1.0 (n=5), no persistent errors | **PASS** | Full compactor run: n=20, 62.3% exact-match — beats gpt-4.1's compactor (~54%). |
| gemini-3.6-flash | 0.98 (n=20) | FAIL | Errors recur across `martial_arts` (control+distractor) — same question both levels. |
| gemini-3.5-flash | 0.965 (n=5) | FAIL | Same failure mode, same question (`martial_arts` Q3), plus `fruits` — spans 2 topics. |

### The `martial_arts` Q3 bug, shared across both failing Gemini models

Both `gemini-3.5-flash` and `gemini-3.6-flash` fail the exact same question in the exact same
way: Q3 asks about velthrak's governing body, correct answer is `{1, 3}` (the Dalviri Arts
Council governs certification **and** it was founded in 1947 — two co-occurring facts in one
passage). Both models almost always answer `{3}` only — they recall the founding year but
consistently drop the governing-body-name fact, across dozens of trials. This is a systematic,
reproducible weakness in how these two Gemini models handle a specific two-clause fact
structure, not model-specific noise — worth a note if testing further Gemini models, since it
may recur.

## Bradley-Terry cross-comparability: betas are NOT comparable across conditions

`bayesian_bt_level_strength.fit_bt_strengths_bayesian()` fits each condition (human, C1, C2,
C3, WM) **independently** — pairwise level comparisons are only ever drawn from within one
condition's own data (`combinations(levels, 2)` inside one `cell` dict); human and LLM never
appear in the same comparison, and each series pins its own `control=0` separately. So a
grouped bar chart with all five series sharing one y-axis (the original
`plot_listening_qa_bt_level_strength_bar_bayesian.py`) visually invites comparing e.g.
human-control's beta against WM-repeat_short's beta as if on one scale — they're not; there's
no bridging observation connecting the scales. Added faceted variants (independent y-axis per
condition/model) to make this explicit:
`plot_listening_qa_bt_level_strength_facets_bayesian.py` (facet by condition) and
`plot_listening_qa_bt_level_strength_bayesian_slopeplot_faceted.py` (facet by model, WM only).

Concretely, WM's fit has ~7x human's `tau` (topic-to-topic heterogeneity) and ~14x human's
`sigma_obs` (per-comparison noise) — not a fitting artifact: humans apply roughly the same
skill across all 4 topics (low tau), while the compactor's accuracy swings hard by topic (e.g.
the `martial_arts` bug above hammers one topic specifically) — genuine topic-inconsistency, not
sampling noise, and partly why the hierarchical (partial-pooling) fit has visibly wider
credible intervals than the older full-pooling WLS+bootstrap version.

## Plotting infra: application/plot_style.py

Consolidated colors/names that were duplicated (and in one case, inconsistent) across
`plot_*.py` files into a single `application/plot_style.py`. Found two different
`LEVEL_COLORS` palettes in the codebase — the canonical one (sourced from the human study's
own `analyze_v6.py`) and an older one used by two files that predated the "canonical" label;
both now use the canonical palette.
