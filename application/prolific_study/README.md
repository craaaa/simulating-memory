# Prolific Reading-QA Pilot

Human-subjects parallel to `application/reading_qa/`. Pilot: 1 doc, single reading condition, 10 MCQs. Qualtrics handles consent, MCQ storage, completion redirect; jsPsych runs the timed non-selectable reading trial inside one Qualtrics question.

## Files

- `birds.md` — passage source (Markdown; inline emphasis preserved as HTML on build).
- `birds_q.yaml` — 10 MCQs with answers/metadata, plus the `title` and `passage` keys.
- `build_stimuli.py` — reads `birds.md` + `birds_q.yaml`, validates them, emits `stimuli.js` (`var STIMULI = {...};`) for pasting into Qualtrics.
- `build_qualtrics_import.py` — reads `birds_q.yaml`, emits `qualtrics_mcq_import.txt` (Advanced Format) for bulk-importing the 10 MCQs into Qualtrics.
- `qualtrics_reading_question.js` — JS for the reading-trial Qualtrics question.
- `qualtrics_survey_spec.md` — survey-build instructions + Taskflow CSV.
- `consent_raw.html` — IRB consent form FY2020-4512 (Word-export source).
- `clean_consent.py` — strips Word/MSO cruft → `consent.html` for Qualtrics paste.
- `consent.html` — cleaned consent (generated; safe to paste into Qualtrics rich-text Source view).
- `export_results.py` — score Qualtrics CSV export against `birds_q.yaml`. *(TODO — implement after first pilot export.)*

## Launch sequence

1. IRB approval in hand (protocol covers stimuli, design, no-copy attestation, debrief).
2. Fill `birds.md` with the real passage and `birds_q.yaml` with 10 MCQs + correct answers.
3. `python application/prolific_study/build_stimuli.py [--timer-seconds N]` → produces `stimuli.js`.
4. Build Qualtrics survey per `qualtrics_survey_spec.md`. Paste `stimuli.js` into the reading question.
5. Create Prolific study; set the Qualtrics `assigned_doc` Embedded Data to the stimulus id (default `birds`).
6. Test draft-preview link end-to-end in Chrome/Safari/Firefox.
7. Run 5-participant pilot, inspect Qualtrics export, then scale.

## Stimulus assignment

Three reading conditions: `birds`, `birds_three`, `birds_repeat`. Prolific Taskflow assigns each participant one `assigned_doc` via URL param and balances allocation. Qualtrics reads `assigned_doc` from Embedded Data; the reading-trial JS looks up `STIMULI[assigned_doc]`. No Qualtrics-side randomization.
