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
