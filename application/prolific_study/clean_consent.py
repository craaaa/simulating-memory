"""Strip Word/MSO cruft from consent_raw.html and emit consent.html.

Output is body innerHTML only — safe to paste into Qualtrics rich-text editor.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment

HERE = Path(__file__).resolve().parent
SRC = HERE / "consent_raw.html"
DST = HERE / "consent.html"


def clean(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for tag in soup.find_all(["style", "script", "meta", "link", "title", "xml"]):
        tag.decompose()
    for tag in soup.find_all(["o:p", "v:shape", "v:imagedata"]):
        tag.unwrap()

    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if attr.lower().startswith(("mso-", "v:", "o:", "w:", "xmlns")):
                del tag.attrs[attr]
            if attr in {"class", "lang", "align"}:
                del tag.attrs[attr]
            if attr == "style":
                v = tag.attrs[attr]
                v = re.sub(r"mso-[^;]+;?", "", v)
                v = v.strip().strip(";")
                if v:
                    tag.attrs[attr] = v
                else:
                    del tag.attrs[attr]

    for tag in soup.find_all(["font"]):
        tag.unwrap()

    body = soup.body
    inner = "".join(str(c) for c in body.contents) if body else str(soup)

    inner = re.sub(r"<!\[if[^\]]*\]>.*?<!\[endif\]>", "", inner, flags=re.DOTALL)
    inner = re.sub(r"\n{3,}", "\n\n", inner)
    return inner.strip()


def main() -> None:
    raw = SRC.read_text(encoding="utf-8", errors="replace")
    DST.write_text(clean(raw) + "\n", encoding="utf-8")
    print(f"Wrote {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
