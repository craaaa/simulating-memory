This folder contains prompts, texts, and audio for the reading comprehension application task.

Prompts are used to generate texts in 4 conditions:
- control
- repeat_short
- repeat_long
- distractor

### Topics
- Birds
- Board games
- Cultural festivals
- Fruits
- Musical instruments
- Minerals

### Length Specifications

|              | Text length | Audio length | Num. concepts | Num. features | Num. mentions |
|--------------|-------------|--------------|---------------|---------------|---------------|
| control      | 150-200     | 1:30-1:45    | 2             | 5             | 1             |
| repeat_short | 150-200     | 1:30-1:45    | 2             | 3             | 2             |
| repeat_long  | 500-550     | 3:45-4:00    | 2             | 5             | 3             |
| distractor   | 500-550     | 3:45-4:00    | 4             | 10            | 1             |

Text length is given in word count
Audio length is given in minutes
Number of concepts: broad feature-containing concept (e.g. bird species, type of fruit) 
Number of features: individual features that each concept differs on (e.g. beak shape, taste)
Number of mentions: number of time each feature value is mentioned in the text

### Generation Process
#### Information generation
1. User will select topic
2. Generate 4 plausible but made-up concepts within that topic (e.g. species of bird)
  - Select 2 concepts to be tested, and 2 concepts to be distractor concepts
3. Generate 10 features that each concept may have, and select a value for each feature for each concept.
  - Values may share overlapping components but must be unique (e.g. two orange birds, but one has black wingtips). 
  - A concept should be relatively coherent, i.e. values may be related (e.g. coastal habitat + eats fish)
  - A concept may contain surprising values
4. Create a `features.md` document that shows, in a table, all features and values for each concept. 
5. Iterate features and values until user approves
6. Split features into:
  - Tested: 3 features that will be the tested features
    - One of the tested features should be an arbitrary one (e.g. Latin name of bird)
  - Padding: 2 features that will pad the control text
  - Distractor: 5 features that will pad the distractor text
7. Create 5 contentful questions out of those features, focusing on concept-feature binding. 
  - Questions may address one concept or both concepts.
  - Answer options should primarily contain relevant in-text distractors. At most 2 questions may use an answer option that is not available in the text.
  - Lightly paraphrase answers where possible
8. Create 2 attention check questions that have answers that are somewhat passage-related. All questions should be multi-select questions. Save all questions in `questions.yaml`

#### Text writing
**Notes**: Ensure all generated texts are within the specified text length boundaries. Participants will read 1 text from each topic, so phrasing should NOT be varied enough that text is not repetitive. Tone of text should be informative and educational but acceptable for spoken register. No disfluencies.

1. Generate control text. This text should contain text describing the tested features and padding features of the tested concepts
2. Generate repeat_short text from control text. Remove text containing padding features, keeping tested feature text verbatim. After each tested feature sentence, add an additional sentence that elaborates on, paraphrases, or extends the tested feature, making sure to repeat that key value of the feature (e.g. Feature: green scales, Elaboration: green scales help with camouflage)
3. Generate repeat_long text from control and repeat_short text. This text should contain EXACTLY all of the text from the control as well as the additional sentences of elaboration from the repeat_short text. In addition, add 1 sentence of elaboration to padding features. Add 2 sentences of elaboration to each tested feature. In total, the text should contain 3 tested features and 2 padding features, each referenced three times.
4. Generate distractor text from control text. This text contains the control text verbatim, but adds 2 paragraphs for 2 new concepts. Order should be: Tested concept 1, Distractor concept 1, Tested concept 2, Distractor concept 2. Also add one sentence for each of the 5 distractor features per concept.

#### Audio file creation
1. After user approval, generate audio using hume.ai
2. Use 'knowledgeable' voice. Ensure that between-sentence pause is 0.6 seconds, between-paragraph pause is 1.2 seconds

### File Organization
- Each topic should get its own directory containing texts/ and audio/ subdir
  - Each topic should have `questions.yaml` and `features.md`
- All texts go in texts/ subdir, labeled by condition
- All audio files are in mp3 format and go in audio/ subdir, labeled by condition

An example is given in the birds/ directory.

### Process Notes (lessons from first full run)

**Structure upfront:**
- Write one fact per sentence from the start. Compound sentences (e.g. "grows in X and produces Y") need to be split later when reorganizing by feature, which is costly.
- Assign T/P/D roles and write base sentences before elaborations. Locking base sentences first prevents cascading propagation issues across conditions.

**Verbatim matching discipline:**
- Write repeat_short elaborations first, then copy them verbatim into repeat_long — do not write repeat_long independently and try to retrofit.
- Run the visualizer frequently during writing. Near-mismatches (sentences that differ by one word) are invisible without it.

**Propagation workflow:**
- Every edit to a base sentence → update all 4 conditions immediately, not in batches.
- Every edit to a repeat_short elaboration → update repeat_long immediately, and vice versa.

**Audio:**
- Test TTS pronunciation of invented concept names with a short snippet before committing. Invented names often need spelling adjustments to sound correct (e.g. "Cliff-pear" vs "Cliff-pair").
- Generate audio snippets for candidate sentences when iterating on wording — faster and cheaper than regenerating full texts.
- Regenerate audio only after texts are fully approved.

**Questions:**
- Lock questions early and do not return to them until final review. Editing questions mid-process while also editing texts creates confusion about which version is canonical.

**Distractor text:**
- The distractor diverges most from the control template (different intros, removed sentences, concept-specific phrasing). Treat it as semi-independent and audit it separately at the end against the other conditions.
