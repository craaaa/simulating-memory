# multi_v2 — Memory Study Stimuli

Reading comprehension study testing incidental memory for factual information about novel concepts. Participants read one text per topic and answer multi-select questions.

## Topics

- astronomy
- birds_v2
- fruits_v2
- martial_arts
- musical_instruments

---

## Design

**One tested concept per topic.** Control text introduces the concept with 6–8 atomic facts. Three other conditions repeat or pad the same facts.

| Condition      | Structure                                          | Word count  | Audio target |
|----------------|----------------------------------------------------|-------------|--------------|
| control        | Base facts, stated once each                       | 50–100 w    | 15–30 s      |
| repeat_short   | Each fact stated twice (base + elaboration)        | 100–150 w   | 30–45 s      |
| repeat_long    | Each fact stated three times                       | ~180–220 w  | ~60 s        |
| distractor     | Control verbatim + untested filler paragraph       | ~180–220 w  | ~60 s        |

`repeat_long` and `distractor` must be approximately the same length (within ~10%).

### Key design principles

- **One tested concept** per topic — pick whichever has cleaner, more testable facts
- **6–8 atomic facts** in control — all tested, no padding, no cross-concept questions
- **Conversational tone** — informative but spoken-register friendly; tone should be consistent across all topics, but exact wording (e.g. "Let me introduce you to...") must NOT be reused across different topics
- **Questions paraphrase** — do not copy wording from the text verbatim into answer options
- **All answer options must be answerable** from control text alone

---

## Conditions in Detail

**control** — Conversational paragraph introducing the concept. Every sentence tests a fact; nothing is padding. Register is informative but spoken-friendly (no bullet lists, natural sentence variety). Negative facts (e.g., "doesn't eat nuts") are fine — they add variety and are memorable.

**repeat_short** — Start with control base sentences (verbatim or near-verbatim). After each base sentence, add one elaborating sentence that restates the key fact value AND adds a small new detail (context, implication, comparison) — not a mere reword. Approximately 2× control word count.

**repeat_long** — Each fact stated three times. Build from repeat_short and add a third mention per fact. Third mentions can be more elaborate (a full sentence or added clause). Should read naturally, not mechanically. Approximately 3× control word count.

**distractor** — Paragraph 1 = control text verbatim (or very close; any intentional deviations must be documented). Paragraph 2 = 4–6 sentences of untested filler facts about the **same concept** (e.g., competition format for a martial art, harvest practices for a fruit). Filler facts must NOT be answerable by any question. Total length ≈ repeat_long.

**Do not introduce competing categories in para 2.** Adding a second named concept (even as a foil) risks interfering with recall of the tested concept's attributes. Use the same subject throughout para 2.

---

## Questions

5 content questions + 1 attention check per topic. All answer options must be answerable from the control text alone.

Questions use `answer_type: multi_select` with `scoring: per_statement`. Most questions should have **2 correct options** — that is the point of multi-select. Design questions by splitting a single control sentence into two independently testable sub-claims:

> e.g., "...observed with an optical telescope, tuned to red light." → Option 1: "With an optical telescope" ✓, Option 2: "By picking up red light" ✓

**Answer options must not be mutually exclusive.** If selecting one option logically rules out another, they cannot both be correct. Design each option as an independent testable claim.

**"None of the above" (NOTA) is a valid answer** — and should occasionally be correct (answer=[5]) to prevent participants from dismissing it as a dummy. Do not overuse: most questions should have ≥1 positive correct answer.

Questions with only 1 correct answer are reserved for 2×2 combo questions, NOTA-answer questions, and narrow-fact questions where only one sub-component is testable.

### Question types

| # | Type | Correct answers | Notes |
|---|------|-----------------|-------|
| Q1 | Single feature | **2 correct** | Split one fact into two sub-components as separate options |
| Q2 | Single feature | **2 correct** | Another fact, same structure |
| Q3 | Specific detail | 1–2 correct | Narrow/arbitrary fact — see formats below |
| Q4 | 2×2 combination | **1 correct** | Cross two facts: correct×correct / correct×wrong / wrong×correct / wrong×wrong |
| Q5 | Term-definition / trick | 1 correct or NOTA | See below |
| AT | Attention check | 1 correct | Answerable from option wording alone, no passage knowledge needed |

**Do not use 3-correct multi-select for Q1 or Q2.** Pilot data (astronomy QA01: 28% accuracy) shows three-correct questions are substantially harder. Reserve 3+ correct only if unavoidable; prefer splitting into a separate question.

**Q4 — 2×2 combination.** Options form a 2×2 grid crossing two independent facts (e.g., body shape × sound character): right×right ✓ / right×wrong / wrong×right / wrong×wrong / "None of the above." Only option 1 is correct.

**Q5 — Term-definition matching (recommended).** Options pair a concept label with a definition. Exploits a key memory asymmetry: participants tend to remember the *definition* (what the thing does or looks like) but not the *term* (the correct label). Two variants:

- **All-wrong variant (answer=[5]):** Options 1–4 each pair labels and definitions in ways that are all incorrect — none are fully right. Answer = [5] "None of the above." Example: astronomy QA05.
- **2×2 variant (answer=[1]):** Options form a 2×2 on term × definition correctness: right×right ✓ / right×wrong / wrong×right / wrong×wrong. Answer = [1].
- **Restrictions variant (answer=[1,2] or similar):** "Which of the following are restrictions/requirements that apply?" Tests two independently falsifiable rules from the passage. Use when a passage sentence encodes two separate prohibitions or conditions (e.g., no striking + sparring experience limit). Answer has 2+ correct options.

**Arbitrary tested terms must follow the repetition pattern: once in control, twice in repeat_short, three times in repeat_long.** This applies to any term participants must recall for a question and that has no prior knowledge to anchor it — category labels (e.g. "emission nebula"), technique names (e.g. "sornek"), governing body names, dates, and specific measurements. Floor effect results when these are treated as background context rather than content (astronomy QA05: 9% accuracy in pilot before fix). The main concept name (e.g. "Velthrak", "Frostmere Tanager") does **not** need to be counted — it recurs naturally as the grammatical subject of most sentences.

### Question formats that benefit most from repetition

These formats test arbitrary bindings — facts with no prior knowledge to anchor them — where repetition is the primary encoding mechanism. Prefer these for Q3 and Q5.

**Source attribution** — "Who [invented/classified/named] X, and in what year?" Inventor name + year are maximally arbitrary. Split across two options for 2-correct format, or pair as "[Name], [year]" in a 2×2 with wrong name/wrong year foils.

**Negation/exclusion** — "Which of the following does X NOT [do/have]?" Negative facts are harder to encode and more vulnerable to interference. Include as one option alongside a positive-correct option (not as the sole question focus).

**Exact value binding** — For specific numbers or measurements, split into sub-claims: (a) the magnitude, (b) the unit or referent. Both must be recalled correctly. E.g., "More than thirty" ✓ and "Fewer than fifty" ✓ both correct for a "forty specimens" fact.

**Causal chain** — "What causes X to [property]?" Split into cause option + effect option. Tests whether participants encoded the mechanism, not just the outcome. Best for topics where the passage explains a process.

**Cross-concept binding** — "Which of these is true of X but NOT Y?" Requires participants to bind attributes to the correct concept. Most effective when two concepts share similar features and the tested value is a clear differentiator.

**Attention check.** Answerable purely from option wording — no passage knowledge needed (e.g., "Select the option that contains a color"). Options should include passage-adjacent words so the question blends in.

**Foils** should be drawn from the dropped concept or plausible-but-wrong values. At most one foil per question may be entirely unrelated to the topic.

### YAML metadata

```yaml
metadata:
  type: content | attention_check
  feature: feature_name_or_list
  concepts: [concept_name]
  notes: >
    Explanation of why each option is correct or wrong.
```

---

## Writing Process

### Step 1 — Choose the tested concept

Read both concepts in the existing control.md. Pick whichever has more distinct, memorable features and generates cleaner 2×2 question designs. Drop all text about the other concept.

### Step 2 — Write control.md

Target: **6–8 atomic facts**, all testable, one short paragraph. **50–100 words, 15–30 s audio.**

- Include only facts that will be tested by questions
- One negative fact is OK (e.g. "doesn't eat nuts") — adds variety and is memorable
- Keep conversational: vary sentence structure, no bullet-like strings
- Remove: habitat, origin story, comparisons, any fact not tested
- **If Q5 is a trick question**: the category label used as a foil (e.g. "emission nebula") is an arbitrary term — it must appear **once in control**, **twice in repeat_short**, and **three times in repeat_long**. Pilot data shows floor effect when it is treated as background framing rather than explicitly repeated content.

Count atomic facts before and after. Aim for 6–8.

### Step 3 — Write repeat_short.md

Each tested fact stated **twice** — once in a base sentence (verbatim or near-verbatim from control), once rephrased with a small new detail.

Target: **100–150 words, 30–45 s audio.**

- Second mention should add new information (context, implication, comparison), not merely restate
- One or two short paragraphs
- Write elaborations here first; copy them verbatim into repeat_long

### Step 4 — Write repeat_long.md

Each tested fact stated **three times**. Build from repeat_short, add a third pass per fact.

Target: **~180–220 words, ~60 s audio.** Must ≈ match distractor length.

- Third mentions can be a full additional sentence or clause
- Should read naturally, not mechanical
- Never write independently of repeat_short — use its elaborations verbatim

### Step 5 — Write distractor.md

Control text **verbatim** (paragraph 1) + untested filler facts (paragraph 2).

Target: **~180–220 words, ~60 s audio.**

- Para 1: control text unchanged; document any intentional deviations
- Para 2: 4–6 sentences of filler (habitat, nesting, social behaviour, lifespan, sounds, size, catalog IDs — whatever suits the topic)
- Filler facts must NOT be answerable by any question
- Total length ≈ repeat_long

### Step 6 — Write questions.yaml

Target: **5 content questions + 1 attention check.** See [Questions](#questions) section for design rules.

- All questions answerable from control text alone
- Paraphrase answer options — no verbatim copy from text
- Foils: plausible but clearly wrong (swap correct values, use dropped-concept values)
- Check: would a participant who read only control get the right answer?

Update the YAML header comments to reflect current question IDs and tested features.

### Step 7 — Update visualize.html

Replace all content:

- Participant view: plain text for each condition
- Experimenter view: color-coded sentences showing overlap
  - Yellow `#fde68a`: all 4 conditions
  - Orange `#fdba74`: RS + RL only (elaborations)
  - Light orange `#fed7aa`: RL only
  - Purple `#e9d5ff`: D only
- Questions block: injected via JS (single definition reused across all tabs)
- Parallel view: sentence-by-sentence comparison table

### Step 8 — Generate audio

```bash
cd application/prolific_study
uv run generate_audio_openai.py \
  --texts-dir multi_v2/<topic>/texts \
  --out-dir multi_v2/<topic>/audio
```

Uses OpenAI TTS: marin voice, gpt-4o-mini-tts. `OPENAI_API_KEY` must be in `.env`. Generate only after all texts are approved.

---

## Checklist per topic

- [ ] One concept chosen and documented
- [ ] control.md: 6–8 atomic facts, all tested, conversational, 50–100 w
- [ ] repeat_short.md: each fact ×2, second mention adds new info, 100–150 w
- [ ] repeat_long.md: each fact ×3, natural flow, ~180–220 w, ≈ distractor length
- [ ] distractor.md: control verbatim (para 1) + untested filler (para 2), ~180–220 w
- [ ] questions.yaml: 5 content Qs + 1 AT; Q1+Q2 have **exactly 2** correct each (not 3+); Q4 is 2×2 combo; Q5 is trick or relationship; all answerable from control; options paraphrased
- [ ] Every arbitrary tested term (category labels, technique names, dates, body names) appears once in control, twice in repeat_short, three times in repeat_long
- [ ] After first launch: verify YAML option text against `_DO` columns in export; correct any mismatches
- [ ] visualize.html: updated with current texts and questions
- [ ] audio: generated for all 4 conditions after text approval

---

## File Organization

```
multi_v2/
  <topic>/
    texts/
      control.md
      repeat_short.md
      repeat_long.md
      distractor.md
    audio/
      control.mp3
      repeat_short.mp3
      repeat_long.mp3
      distractor.mp3
    features.md          # concept/feature table (reference)
    questions.yaml       # question bank
    visualize.html       # condition comparison tool
  prompts/
    visualize_prompt.md  # visualizer generation prompt
  visualize.py           # visualizer generator script
  README.md              # this file
```

---

## Process Notes

**Every arbitrary tested term must follow the repetition pattern: once in control, twice in repeat_short, three times in repeat_long.** "Arbitrary" means the participant has no prior knowledge to anchor it — category labels, technique names, governing body names, dates, specific measurements. The main concept name (the subject of the passage) recurs naturally and does not need to be counted. Pilot data shows that arbitrary terms not explicitly repeated produce the same accuracy as control, defeating the manipulation.

**Write one fact per sentence.** Compound sentences (e.g., "grows in X and produces Y") create ambiguity when counting atomic facts and complicate repeat-condition elaborations.

**Verbatim discipline.** Base sentences in repeat_short and repeat_long must match control exactly (or near-exactly). Write repeat_short elaborations first, then copy them verbatim into repeat_long — never write repeat_long independently.

**Propagation.** Every edit to a control sentence → update all 4 conditions immediately. Every edit to a repeat_short elaboration → update repeat_long immediately. Pay particular attention to units and specific values (e.g., meters vs. feet, "900–1,400" vs. "900–4,000") — these are easy to change in control and miss in RS/RL.

**Avoid pure prohibition facts as primary tested content.** Facts stated as blanket negations ("no X of any kind is permitted") have limited elaboration paths — the second and third passes tend to be near-identical restatements. Prefer positive facts (a technique, a measurement, a causal chain) that naturally support additional detail. One prohibition fact per topic is fine; it becomes a Q1/Q2 option rather than the centrepiece of a repeat chain.

**Opening phrases.** All topics should share the same conversational register, but exact phrasing (e.g., "Let me introduce you to...") must not be reused verbatim across different topics. Within a single topic, verbatim repetition of base sentences across conditions is correct by design.

**Audio pronunciation.** Test TTS pronunciation of invented names before committing to spelling. Invented names often need spelling adjustments to sound correct.

**Questions.** Lock questions after texts are approved. Editing questions mid-process while also editing texts causes version confusion.

**YAML option text must match Qualtrics display exactly.** After the first launch, check the `_DO` columns in the export against YAML option text. Qualtrics sometimes reformats options (e.g., prepends "It is …"); a mismatch causes all responses to score as 0. Correct the YAML immediately and document the discrepancy in the question's `notes` field.

**Distractor para 1.** Should be control verbatim; deviations (added or moved sentences) must be intentional and documented. The distractor diverges most from control — audit it separately at the end.

**Visualizer.** Run the visualizer frequently during writing. Near-mismatches (sentences that differ by one word) are invisible without it.
