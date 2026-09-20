# Few-shot rationale demos

`fewshot_forward.txt` and `fewshot_reverse.txt` are the STaR few-shot prompt set `P`
(Zelikman et al. 2022 use ~10 hand-written rationale examples). They are injected into
the **generation** and **rationalization** prompts and are stripped from the
**training** prompt, so the fine-tuned model internalizes the format instead of
depending on the demos.

Everything in these files is sent to the model verbatim. Keep notes-to-self out of
them — this file is where such notes go.

## Block format

Copy this pattern to add a demo. Structure and the `press <<D>>.` answer format are
fixed; only the `<reasoning>` text is yours to write.

```
Human Participant A

Trial 1
The digits are the following: [8, 4, 1]
<reasoning>
why this human remembered what they remembered
</reasoning>
Human response:
press <<8>>.
press <<4>>.
press <<1>>.
Outcome: Correct
```

The block structure mirrors `bench/tasks/prompts/digit_span_*_c4_human.txt` (the C4
condition's real human transcripts), with the `<reasoning>` block added.

## Rules

- **Reverse demos must show reversed responses.** The first `press` line is the *last*
  presented digit. `test_reverse_fewshot_demos_show_reversed_responses` enforces this —
  a forward-ordered demo in the reverse file would teach the wrong task.
- **Write reasoning that explains the error, not one that asserts it.** The response
  should read as a consequence of what the reasoning says was encoded and lost.
- **Include both correct and incorrect trials.** Half the training corpus is human
  success trials and half is human errors.
- **Leave no `PLACEHOLDER` text.** `sample` refuses to run while any remains (or pass
  `--no-fewshot`).
- Changes here are captured per run in `run_config.json` via `git_provenance`'s
  `prompt_diffs`, so a dirty working tree is still reproducible.

## `fewshot_listening.txt`

Same role, different construction. Digit-span stimuli are generated, so a demo can use
a sequence no participant ever saw. Listening QA has sixteen stimuli in total and the
split is by participant, so every passage appears on both sides of it — there is no
held-out stimulus to demo on, and inventing one would mean demonstrating memory
behavior nobody observed.

So these demos are **real study passages and what four real participants actually
selected**, with only the `<reasoning>` written by hand. The four items are listed in
`rationales/listening/prompting.py:FEWSHOT_SOURCE_ITEMS` and recorded in every run's
`prompt_additions`.

### Status: the four `<reasoning>` blocks are unwritten

The items, the passages, the questions and the `Answer:` lines are real and in place.
Only the `<reasoning>` text is `PLACEHOLDER`, and it is left that way on purpose: it is
a claim about why a specific person forgot a specific thing, and nobody recorded that.
It is the researcher's to write, not something to reconstruct plausibly.

Until it is written, `star`, `sample` and `probe` all refuse to run with `--fewshot`
(`test_the_placeholder_gate_actually_blocks_a_run`). `select-data` and `show-prompt`
still work, so you can render the scaffold and see where the text goes:

```bash
python -m rationales.cli show-prompt --task listening_qa --kind sample
```

Each `PLACEHOLDER` line names what that demo has to explain. What the four are:

Each demo is laid out **exactly like a real prompt** — transcript, the sibling block,
then the target question — because few-shot transfer works by shape, and because the
sibling answers are the only thing that makes one participant distinguishable from
another.

Each sibling answer is marked `[correct]` or `[wrong]` (exact set match, the same
definition `ListeningItem.correct` uses). The target's own correctness is never stated —
that is most of `y_i`, and on a single-answer question it is all of it.

| # | topic / level | siblings right | gold | they chose | what the siblings show, and what the rationale must account for |
|---|---|---|---|---|---|
| 1 | fabrics, control | 0/4 | `[1]` | `[1]` | an under-selector: two siblings are wrong only because they picked one true option where several applied. Why does that habit cost nothing on a single-answer question? |
| 2 | astronomy, distractor | 1/4 | `[1,2]` | `[3,4]` | holds the observation method firmly; every question asking *what the object is* went wrong, twice by importing the dying-star description from the second passage. |
| 3 | astronomy, distractor | 1/4 | `[1,2]` | `[5]` | mirror image of #2: method gone to a radio-pulse foil, but study purpose recalled *correctly* on the combination question. Why decline it when asked directly? |
| 4 | martial arts, control | 2/4 | `[1,2]` | `[2]` | had the sparring rule exactly right, and had already asserted velthrak *allows* limited strikes. Why is no-striking unavailable to them? |

**#2 and #3 are the same passage and the same question, answered by two different
people.** Their sibling blocks are the *only* difference in the input. Write that pair
together and make the two rationales turn on that difference — if they don't, the demo
says the answer is random rather than person-dependent, which is the opposite of the
objective. `test_the_two_demos_that_share_a_question_are_told_apart_by_their_siblings`
holds the setup in place; only the reasoning can undo it.

Cross-passage participant identity is deliberately not represented. Two of these cases
happen to come from the same respondent and nothing is made of it — each demo stands as
one listener on one passage.

When the four are written, flip
`test_shipped_fewshot_demos_still_need_their_reasoning_written` to assert the gate is
clear. An earlier draft of these reasonings (model-written, since replaced) is in git
history at commit `cd3de9d` if it is useful as a foil to write against.

### Rules

- **Every demo item must be in the TRAIN split** (seed 42, `eval_frac` 0.15).
  `test_every_fewshot_demo_item_is_in_the_train_split` enforces it: a demo drawn from a
  held-out participant would show the model a person it is about to be scored on.
- **The `Answer:` line must be what that participant selected**, not what they should
  have. `test_fewshot_demo_answers_match_what_those_humans_actually_selected` checks
  each one against `responses.csv`, so the demos cannot drift into invented behavior
  during an edit.
- **Demo wording must not trip the hint-leak filter.** These sit in the *generation*
  prompt, which has no hint; phrasing that matches a leak pattern gets copied into the
  model's rationales and the filter then drops the corpus the run just paid for. An
  earlier draft said "what those people actually selected" and matched
  `actually selected`. `test_fewshot_demos_do_not_trip_the_hint_leak_filter` guards it.
- **Cover the failure modes**, not just correct answers: a missed true option, an
  endorsed interference foil, and none-of-the-above are all real human strategies, and
  a demo set of correct answers teaches the model to answer well — the opposite of the
  objective. Two demos deliberately share a passage and question across different
  participants, since one stimulus producing two different errors is exactly what the
  fine-tune has to learn to condition on.

What this does **not** avoid: the demos reveal the answer shape for the questions they
cover, and those questions also appear in eval items belonging to other participants.
No choice of demo item avoids that, given sixteen stimuli. It is one more reason the
eval to read is `human_match` — which the demos cannot give away for an unseen person —
rather than ground-truth accuracy.
