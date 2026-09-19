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
