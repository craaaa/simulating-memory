#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "python-dotenv",
# ]
# ///
"""Generate MP3 audio for each text in a texts/ directory using the OpenAI TTS API.

Usage:
    uv run generate_audio_openai.py [--texts-dir PATH] [--out-dir PATH] [--voice VOICE] [--model MODEL]

Reads OPENAI_API_KEY from .env in the same directory as this script.

Default voice: Marin (requires gpt-4o-mini-tts).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import os

from openai import OpenAI

HERE = Path(__file__).resolve().parent

load_dotenv(HERE / ".env")

DEFAULT_VOICE = "marin"
DEFAULT_MODEL = "gpt-4o-mini-tts"

INSTRUCTIONS = """\
Voice: Clear, engaged, and energetic, projecting confidence and genuine enthusiasm for the material.

Pacing: Slow and deliberate, pausing often to allow the listener to follow along comfortably.

Tone: Warm, animated, and informative, maintaining a balance between formality and approachability — sound genuinely interested, not flat or rote.

Punctuation: Structured with commas and long, substantial pauses at the end of sentences for clarity, ensuring information is easily digestible. Give key facts and figures — names, numbers, category labels — a noticeably stronger emphasis and a touch more volume than the surrounding words, so they stand out on a first listen.

Emotion: Cheerful, engaged, and enthusiastic throughout; convey genuine enjoyment and curiosity about the topic, like sharing an interesting fact with a friend.\
"""


def synthesize(client: OpenAI, text: str, voice: str, model: str) -> bytes:
    kwargs = dict(
        input=text,
        model=model,
        voice=voice,
        response_format="mp3",
    )
    if model == "gpt-4o-mini-tts":
        kwargs["instructions"] = INSTRUCTIONS
    response = client.audio.speech.create(**kwargs)
    return response.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts-dir", default=str(HERE / "multi_v2/birds/texts"))
    ap.add_argument("--out-dir", default=str(HERE / "multi_v2/birds/audio"))
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    choices=["tts-1", "tts-1-hd", "gpt-4o-mini-tts"])
    args = ap.parse_args()

    texts_dir = Path(args.texts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in .env or environment")

    client = OpenAI(api_key=api_key)

    md_files = sorted(texts_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {texts_dir}")

    print(f"Voice: {args.voice}  Model: {args.model}")
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8").strip()
        out_path = out_dir / (md_path.stem + ".mp3")
        print(f"{md_path.name} -> {out_path} ...", end=" ", flush=True)
        audio = synthesize(client, text, args.voice, args.model)
        out_path.write_bytes(audio)
        print(f"done ({len(audio):,} bytes)")


if __name__ == "__main__":
    main()
