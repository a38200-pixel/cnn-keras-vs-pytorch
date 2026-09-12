"""Replace only generated README marker contents."""
from __future__ import annotations

from pathlib import Path


def replace_marker(path: Path, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
    if start not in text or end not in text:
        raise ValueError(f"README marker missing: {marker}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    path.write_text(f"{before}{start}\n{content.rstrip()}\n{end}{after}", encoding="utf-8")
