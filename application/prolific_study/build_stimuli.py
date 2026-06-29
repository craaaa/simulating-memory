"""Build the STIMULI JS blob for the Qualtrics reading question.

Reads the passage from a Markdown file (`birds.md`) and the multiple-choice
questions from a YAML file (`birds_q.yaml`), then emits a single-line
`var STIMULI = {...};` that can be pasted into `qualtrics_reading_question.js`
in place of the inlined placeholder.

There is a single reading condition (no per-level variants). Markdown emphasis
(`_italic_`, `*italic*`, `**bold**`, `__bold__`) in the passage is converted to
HTML (`<em>`, `<strong>`) so typographic details survive into Qualtrics.

Validation runs here so config bugs surface at build time rather than mid-study.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDER = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC_STAR = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_ITALIC_UNDER = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", re.DOTALL)


def md_inline_to_html(text: str) -> str:
    """Convert inline Markdown emphasis to HTML, escaping other HTML first."""
    text = html.escape(text, quote=False)
    text = _BOLD_STAR.sub(r"<strong>\1</strong>", text)
    text = _BOLD_UNDER.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_STAR.sub(r"<em>\1</em>", text)
    text = _ITALIC_UNDER.sub(r"<em>\1</em>", text)
    return text


def load_passage(md_path: Path) -> str:
    """Read a Markdown passage and return HTML paragraphs joined by newlines.

    Paragraphs are separated by blank lines in the source; the downstream JS
    wraps each newline-delimited chunk in its own <p>, so we collapse each
    paragraph to a single line while preserving inline emphasis as HTML.
    """
    raw = md_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    rendered = []
    for para in paragraphs:
        joined = " ".join(line.strip() for line in para.splitlines())
        rendered.append(md_inline_to_html(joined))
    if not rendered:
        raise ValueError(f"{md_path}: passage is empty")
    return "\n".join(rendered)


def load_questions(yaml_path: Path) -> list[dict]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions", [])
    if len(questions) != 10:
        raise ValueError(f"{yaml_path}: expected 10 questions, got {len(questions)}")
    for q in questions:
        opts = q.get("options", {})
        if set(opts.keys()) != {"A", "B", "C", "D", "E"}:
            raise ValueError(f"{yaml_path}/{q.get('q_id')}: options must be A-E")
        if any(not str(v).strip() for v in opts.values()):
            raise ValueError(f"{yaml_path}/{q.get('q_id')}: empty option text")
        ans = q.get("answer")
        valid_letters = {"A", "B", "C", "D", "E"}
        if not ((isinstance(ans, list) and ans and all(a in valid_letters for a in ans)) or ans in valid_letters):
            raise ValueError(f"{yaml_path}/{q.get('q_id')}: invalid answer")
    return questions


def build_js(stimuli: dict) -> str:
    # Questions are omitted — Qualtrics renders the MCQs natively.
    return "var STIMULI = " + json.dumps(stimuli, ensure_ascii=False) + ";"


AUDIO_BASE_URL = "https://craaaa.github.io/simulating-memory/birds_texts"
AUDIO_DATE_PREFIX = "20260617"

# Reading-condition docs that share the single question bank.
# timer_seconds: 180 for longer conditions (repeat, distractors); 120 for the rest.
DOCS = [
    {"doc_id": "birds",             "passage": "birds_texts/birds.md",             "timer_seconds": 120},
    {"doc_id": "birds_easier",      "passage": "birds_texts/birds_easier.md",      "timer_seconds": 120},
    {"doc_id": "birds_listed",      "passage": "birds_texts/birds_listed.md",      "timer_seconds": 120},
    {"doc_id": "birds_repeat",      "passage": "birds_texts/birds_repeat.md",      "timer_seconds": 180},
    {"doc_id": "birds_distractors", "passage": "birds_texts/birds_distractors.md", "timer_seconds": 180},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "birds_q.yaml"),
                    help="YAML question bank shared across docs.")
    ap.add_argument("--out", default=str(HERE / "stimuli.js"))
    args = ap.parse_args()

    yaml_path = Path(args.questions)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    title = data.get("title", "")
    load_questions(yaml_path)  # validate only

    stimuli: dict = {}
    for d in DOCS:
        md_path = HERE / d["passage"]
        wav_filename = f"{AUDIO_DATE_PREFIX}_{d['doc_id']}.wav"
        stimuli[d["doc_id"]] = {
            "title": title,
            "text": load_passage(md_path),
            "timer_seconds": d["timer_seconds"],
            "url": f"{AUDIO_BASE_URL}/{wav_filename}",
        }

    out_path = Path(args.out)
    out_path.write_text(build_js(stimuli) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(stimuli)} docs: {list(stimuli)})")


if __name__ == "__main__":
    main()
