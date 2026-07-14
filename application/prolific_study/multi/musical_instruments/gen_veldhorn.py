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

TEXT = "The Tambic Drum is a percussion instrument traced to the Tambali craft guild of Fes, Morocco, around sixteen fifty."

api_key = os.environ.get("HUME_API_KEY")
if not api_key:
    sys.exit("HUME_API_KEY not set")

client = HumeClient(api_key=api_key)
voice = PostedUtteranceVoiceWithName(name="Knowledgeable")

response = client.tts.synthesize_json(
    utterances=[PostedUtterance(text=TEXT, voice=voice, trailing_silence=0.6)],
    format=FormatMp3(type="mp3"),
    num_generations=1,
    request_options=RequestOptions(timeout_in_seconds=120),
)
audio = base64.b64decode(response.generations[0].audio)
out = HERE / "audio" / "tambic.mp3"
out.write_bytes(audio)
print(f"Saved: {out} ({len(audio):,} bytes)")
