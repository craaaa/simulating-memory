# Prompt: Text Visualizer

## Purpose
Generate `visualize.html` for a given study topic, showing:
- Sentence-level overlap highlighting across all 4 conditions
- Side-by-side parallel view (all conditions in a table)
- Questions and answers panel

## Usage
```bash
cd application/prolific_study/multi
python visualize.py <topic_dir>        # e.g. birds, board_games
python visualize.py birds --output /tmp/birds_review.html
python visualize.py birds --threshold 0.75   # lower = stricter matching
```

## Output
`<topic_dir>/visualize.html` — self-contained single HTML file, no dependencies.

## Overlap color key
| Color | Conditions sharing the sentence |
|-------|--------------------------------|
| Gold | All 4 (C + RS + RL + D) |
| Amber | C + RS + RL |
| Blue (light) | C + RL + D |
| Cyan | C + RL only |
| Orange | RS + RL (elaboration sentences) |
| Light orange | RL only |
| Purple | D only |
| Indigo | C + D only |

Badges on each sentence show which *other* conditions also contain it.

## Input files expected
- `<topic_dir>/texts/{control,repeat_short,repeat_long,distractor}.md`
- `<topic_dir>/questions.yaml` (optional; requires PyYAML)

Missing text files produce a warning but do not crash — useful for partial topics.

## Algorithm
Sentences are split on `./?/!` followed by a capital letter.
Fuzzy similarity via `difflib.SequenceMatcher` with threshold 0.82 (configurable).
Sentences with similarity > threshold are clustered together and share a color.
