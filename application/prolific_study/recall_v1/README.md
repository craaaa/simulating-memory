# recall_v1 — free recall after a filled delay

`recall_v1` is `multi_v6` with the recognition QA task replaced by a **production**
measure. The study phase is untouched; only what happens after each passage changes.

Per topic:

```
audio passage  ->  60 s math task  ->  free recall
```

The math task is a Brown–Peterson style **filled delay**: self-paced arithmetic that
occupies working memory and blocks rehearsal of the passage just heard. Its accuracy is
not a dependent variable of interest — it is a manipulation check that the participant
actually engaged with the delay rather than rehearsing through it.

## What carries over from multi_v6 unchanged

- Four topics: `martial_arts`, `fruits_v2`, `astronomy`, `fabrics`.
- Four passage conditions: `control`, `repeat_short`, `repeat_long`, `distractor`.
- The same hosted audio, played once with no replay:
  `https://craaaa.github.io/simulating-memory/multi_v5/<topic>/audio/<cond>.mp3`
- `group` arrives as a URL parameter; one of 24 flow branches sets `<topic>_cond` for all
  four topics.
- `BlockRandomizer` `FL_27` (`SubSet: 4`, `EvenPresentation: true`) randomises topic
  **order**; every participant still sees all four topics. Order is recoverable from the
  export column `FL_27_DO`.
- Consent block and its non-consent screen-out, the audio check and its screen-out, the
  instructions, the per-topic break, the debrief, and the Prolific completion redirects.

### Naming: "math", not "distractor"

`multi_v6` already uses `distractor` for a *passage condition* — a passage padded with
topic-irrelevant prose (`texts/distractor.md`, `audio/distractor.mp3`, a value of
`<topic>_cond`). The arithmetic task is therefore called the **math task** everywhere
here, so `distractor` keeps its existing meaning and nothing downstream is ambiguous.

## What changed

| | multi_v6 | recall_v1 |
|---|---|---|
| After each passage | 5 multi-select QA questions + 1 attention check, one page | 60 s math task, then free recall, two pages |
| Memory measure | recognition (exact + partial credit) | production (free recall text) |
| Attention checks | one per topic, pinned to visible position 3 | **none** |
| In-survey quality screen-out | `FL_160` → PoorQuality redirect | removed |
| Feedback slider | rates "Passage" / "Questions" | rates "Passage" / "Recall task" |

### Removed: the attention checks and the PoorQuality branch

Deleting the QA blocks deletes the four attention-check questions. `multi_v6`'s
`FL_160` branch referenced those four QIDs, so it had to go as well — otherwise the QSF
ships with branch logic pointing at questions that no longer exist. Prolific completion
code `C1ICGI9P` is consequently unused.

This costs little. `FL_160`'s four conditions were joined by `And`, so it only ever fired
for participants who failed **all four** attention checks, while `data_prep.py` excluded
on **any** failure. The real exclusion rule always lived in the analysis code, not the
survey.

If `recall_v1` needs a quality gate, it has to be a new one — the recall task itself has
no correct answer to check against.

## Files

| File | Role |
|---|---|
| `build_recall_v1.py` | Builds `recall_v1.qsf` from `../multi_v6/multi_v6.qsf`. The only entry point. |
| `qualtrics_math_question.js` | The 60 s math task. Inlined into each `<topic>_math` question by the builder. |
| `qualtrics_recall_question.js` | Free-recall dwell gate and paste blocking. Inlined into each `<topic>_recall` question. |
| `recall_v1.qsf` | Generated artifact, committed so it is importable without running the builder. |
| `<topic>/texts/*.md` | TTS source scripts for the audio, copied from `multi_v6` unchanged. Kept for reference and for coding recall protocols against the studied content. |

There is no `questions.yaml` — `recall_v1` has no question bank.

Note the inherited directory/tag mismatch: the topic key is `fruits_v2` but its content
directory is `fruits`. `build_recall_v1.py` carries an explicit `TOPIC_CONTENT_DIRNAME`
map, as `multi_v6/data_prep.py` does.

### Rebuilding

```sh
python build_recall_v1.py                # validate, then write recall_v1.qsf
python build_recall_v1.py --check-only   # validate only
```

The builder edits a deep copy of the template: it removes the four `_qs` blocks and their
28 questions, drops the stale attention-check answer keys and the `FL_160`/`FL_142` flow
nodes, adds two blocks and four questions per topic, rewires each topic group inside
`FL_27`, declares the new embedded data, and rewords the feedback slider. It refuses to
write if any internal-consistency check fails.

Two checks are measured **relative to the template**, because `multi_v6` already violates
them and inherited defects are out of scope: duplicate `DataExportTag`s (the template
reuses `Q1`/`Q2` across the consent, audio-check, instructions and debrief questions) and
undeclared piped fields (the orphaned `musical_instruments` trial JS pipes
`musical_instruments_cond`, which no surviving branch declares). Only newly introduced
violations fail the build.

The orphaned `BL_musinstr_*` blocks and the `Trash / Unused Questions` block are
flow-unreachable in the template and are deliberately left alone.

## Task details

**Math task** (`<topic>_math`, Text/Graphic shell driven entirely by JS)

- Operator uniform over `+`, `−`, `×`. Operands: two 2-digit numbers for `+`/`−`,
  single-digit × 2-digit for `×`. Subtraction is ordered so the answer is never negative.
- Self-paced: one problem on screen, type the answer, Enter (or Submit, for mobile)
  commits, brief ✓/✗, next problem appears immediately.
- 60 s wall clock with a visible countdown, then auto-advance. Paste into the answer field
  is blocked.

**Free recall** (`<topic>_recall`, Text Entry / Essay Text Box)

- Prompt names only the topic, never passage content: "You just heard a passage about
  \<blurb\>. Write down everything you can remember about it. Include as much detail as
  you can, in any order." Blurbs follow the existing per-topic phrasing in the trial JS
  ("a type of combat sport", "a type of fruit", "an astronomical object", "a type of
  textile").
- Qualtrics owns the textarea, so the response exports natively. Paste and drop are
  blocked; a live character count is shown.
- **Untimed above a floor**: Next is hidden for 60 s ("You may continue in Ns"), then
  appears. There is no upper limit.
- **Force Response is OFF by design.** Requiring text would pressure participants into
  confabulating when they remember nothing. An empty recall is a real datum and becomes an
  exclusion decision at analysis time.

## Export columns

**Expected**, not yet confirmed against a real export. Qualtrics derives these names
from each question's `DataExportTag`, which the builder sets — but the template shows tags
surviving into column names with mangling (`martial_arts_qs-Jun29,2026_DO`), so treat the
names below as the intent and verify them in step 4 of the checklist before coding against
them. Note also that the export is UTF-16 with two header rows above the data.

New, per topic:

| Column | Meaning |
|---|---|
| `<topic>_recall` | The free-recall text. |
| `<topic>_math_n_attempted` | Problems committed during the 60 s. |
| `<topic>_math_n_correct` | How many were right. |
| `<topic>_math_trials` | JSON array of `{problem, op, a, b, given, correct, rt_ms}` per trial. Parse each cell with `json.loads`; the blob contains commas and quotes, so never split it naively. |
| `<topic>_timing_math_*` | Page timer (First Click / Last Click / Page Submit / Click Count). |
| `<topic>_timing_recall_*` | Page timer; `Page Submit` is the recall duration. |

Gone: `<topic>_QMA01`…`QFB05` and the four `<topic>_AT_<topic>` / `<topic>_AT_correct`
columns.

Still present and still meaningful: `group`, `<topic>_cond`, `FL_27_DO`,
`<topic>_timing_passage_*`, `<topic>_difficulty_1` (passage) and `_difficulty_2` (now the
recall task), `<topic>_comments`, `passage_count`, `Duration (in seconds)`, `Finished`,
`PROLIFIC_PID`.

## Analysis: not yet implemented

`multi_v6`'s pipeline (`data_prep.py`, `build_longdata.py`, `analyze_v6.py`,
`bayesian_model.py`, `report_model.py`, `power_analysis.py`) scores multi-select
recognition responses — `col_correct`, `AT_COL`, `content_cols`, and `parse_response`'s
set reconstruction. **None of it applies to free-recall text, and none of it is ported
here.**

Scoring free recall needs an idea-unit rubric (segment each passage into scoreable
propositions, then code each protocol for which propositions it contains) plus a coding
pass, whether human or model-assisted. That does not exist yet. The `<topic>/texts/*.md`
files are here because they are the source content to build that rubric from.

Once recall is scored, the `multi_v6` model structure transfers directly —
`FL_27_DO` → serial position still works, and the design is otherwise identical.

## Before running: what the user must do

1. Import: Qualtrics → Projects → Create → **Import a QSF** → `recall_v1.qsf`. The survey
   imports **Inactive**; activate it deliberately.
2. Preview with `?group=1` appended. `group` arrives as a URL parameter — without it no
   branch fires, `<topic>_cond` stays empty, and the trial JS shows its "Configuration
   error" message.
3. Walk one topic end to end: audio plays and Next appears when it ends; the math task
   accepts typed answers and auto-advances at 60 s; recall Next unlocks at 60 s.
4. Check a preview export for the new columns above.
5. **Revise the Prolific duration estimate and pay.** Per topic this adds 60 s of math
   plus a ≥60 s recall floor; across four topics that is **≥8 minutes** more than
   `multi_v6` before any typing time. Actual recall typing will add more.
6. Decide whether a replacement quality gate is needed, given there are no attention
   checks (see above).

Note that the builder's checks prove the QSF is internally consistent, not that Qualtrics
accepts it. The import is the real test.
