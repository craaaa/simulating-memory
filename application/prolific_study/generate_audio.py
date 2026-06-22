#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "hume",
#   "python-dotenv",
# ]
# ///
"""Generate MP3 audio for each text in pilot_v4/texts/ using the Hume TTS API.

Usage:
    uv run generate_audio.py [--texts-dir PATH] [--out-dir PATH]

Reads HUME_API_KEY from .env in the same directory as this script.
Voice used: "Knowledgeable" (custom voice).
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from dotenv import load_dotenv
import os

from hume import HumeClient
from hume.tts import PostedUtterance, FormatMp3, PostedUtteranceVoiceWithName
from hume.core import RequestOptions

HERE = Path(__file__).resolve().parent

load_dotenv(HERE / ".env")


SENTENCE_SILENCE = 0.6  # seconds after each sentence
PARAGRAPH_SILENCE = 1.2  # seconds after last sentence of each paragraph


def utterances_from_text(text: str, voice: PostedUtteranceVoiceWithName) -> list[PostedUtterance]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result = []
    for i, para in enumerate(paragraphs):
        sentences = [s.strip() + "." for s in para.split(".") if s.strip()]
        is_last_para = i == len(paragraphs) - 1
        for j, s in enumerate(sentences):
            is_last_sentence = j == len(sentences) - 1
            silence = PARAGRAPH_SILENCE if (is_last_sentence and not is_last_para) else SENTENCE_SILENCE
            result.append(PostedUtterance(text=s, voice=voice, trailing_silence=silence))
    return result


def synthesize(client: HumeClient, text: str) -> bytes:
    voice = PostedUtteranceVoiceWithName(name="Knowledgeable")
    response = client.tts.synthesize_json(
        utterances=utterances_from_text(text, voice),
        format=FormatMp3(type="mp3"),
        num_generations=1,
        request_options=RequestOptions(timeout_in_seconds=120),
    )
    return base64.b64decode(response.generations[0].audio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts-dir", default=str(HERE / "pilot_v4/texts"))
    ap.add_argument("--out-dir", default=str(HERE / "pilot_v4/audio"))
    args = ap.parse_args()

    texts_dir = Path(args.texts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("HUME_API_KEY")
    if not api_key:
        raise SystemExit("HUME_API_KEY not set in .env or environment")

    client = HumeClient(api_key=api_key)

    md_files = sorted(texts_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {texts_dir}")

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8").strip()
        out_path = out_dir / (md_path.stem + ".mp3")
        print(f"{md_path.name} -> {out_path.relative_to(HERE)} ...", end=" ", flush=True)
        audio = synthesize(client, text)
        out_path.write_bytes(audio)
        print(f"done ({len(audio):,} bytes)")


if __name__ == "__main__":
    main()
