"""Build a Qualtrics Advanced-Format TXT import for the MCQ block.

Reads the question bank (`birds_q.yaml`) and emits `qualtrics_mcq_import.txt`,
which can be imported into a Qualtrics survey via
Survey builder -> Tools -> Import/Export -> Import Questions.

Each question becomes a single-answer multiple-choice question with a stable
`[[ID:QNN]]` export tag and a page break between questions. Choices are emitted
in their YAML order with no letter prefixes (choice order is meant to be
randomized in Qualtrics; scoring keys off the choice text in the CSV export).

Note: Force Response and correct-answer scoring cannot be expressed in the TXT
format and must be set in the Qualtrics editor after import. The correct answers
remain in `birds_q.yaml`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_stimuli import load_questions

HERE = Path(__file__).resolve().parent


def build_import(block_name: str, questions: list[dict]) -> str:
    lines: list[str] = ["[[AdvancedFormat]]", "", f"[[Block:{block_name}]]", ""]
    for i, q in enumerate(questions):
        lines.append("[[Question:MC:SingleAnswer]]")
        lines.append(f"[[ID:{q.get('q_id', f'Q{i + 1:02d}')}]]")
        lines.append(str(q["question"]).strip())
        lines.append("")
        lines.append("[[Choices]]")
        for key in ["A", "B", "C", "D", "E"]:
            lines.append(str(q["options"][key]).strip())
        lines.append("")
        if i < len(questions) - 1:
            lines.append("[[PageBreak]]")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "birds_q.yaml"))
    ap.add_argument("--block-name", default="Birds MCQs")
    ap.add_argument("--out", default=str(HERE / "qualtrics_mcq_import.txt"))
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    out_path = Path(args.out)
    out_path.write_text(build_import(args.block_name, questions), encoding="utf-8")
    print(f"Wrote {out_path} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
