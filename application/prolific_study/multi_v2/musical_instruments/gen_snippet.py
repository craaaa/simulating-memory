#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "hume",
#   "python-dotenv",
# ]
# ///
import base64, os, sys
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent.parent / ".env")

from hume import HumeClient
from hume.tts import PostedUtterance, FormatMp3, PostedUtteranceVoiceWithName
from hume.core import RequestOptions

TEXT = (
    "The Kasten-zither was first catalogued by the instrument maker Hans Krausel in 1802. "
    "That eighteen oh two catalog entry written by Krausel is the first detailed written account of the instrument we have. "
    "In it, Krausel described it as a fixture of Bavarian inn life, and commonplace among Bavarian musicians as of 1802."
)

SENTENCE_SILENCE = 0.6
PARAGRAPH_SILENCE = 1.2

api_key = os.environ.get("HUME_API_KEY")
if not api_key:
    sys.exit("HUME_API_KEY not set")

client = HumeClient(api_key=api_key)
voice = PostedUtteranceVoiceWithName(name="Knowledgeable")

sentences = [s.strip() for s in TEXT.split(".") if s.strip()]
utterances = []
for i, s in enumerate(sentences):
    silence = SENTENCE_SILENCE if i < len(sentences) - 1 else PARAGRAPH_SILENCE
    utterances.append(PostedUtterance(text=s + ".", voice=voice, trailing_silence=silence))

print(f"Synthesizing {len(utterances)} utterances...")
response = client.tts.synthesize_json(
    utterances=utterances,
    format=FormatMp3(type="mp3"),
    num_generations=1,
    request_options=RequestOptions(timeout_in_seconds=120),
)
audio = base64.b64decode(response.generations[0].audio)
out = HERE / "audio" / "repeat_long.mp3"
out.write_bytes(audio)
print(f"Saved: {out} ({len(audio):,} bytes)")
