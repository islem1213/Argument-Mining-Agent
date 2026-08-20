"""Conservative sentence segmentation with source offsets."""
from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
# This deliberately conservative boundary set avoids turning every comma into an ADU.
CLAUSE_BREAK = re.compile(r"\s*(?:;|—|\b(?:however|therefore|although|but)\b)\s*", re.IGNORECASE)

@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int

def segment(text: str) -> list[Segment]:
    """Create candidate ADUs with sentence boundaries plus high-confidence clause breaks."""
    segments: list[Segment] = []
    for sentence in SENTENCE.finditer(text):
        body, base = sentence.group(), sentence.start()
        last = 0
        for boundary in CLAUSE_BREAK.finditer(body):
            candidate = body[last:boundary.start()].strip()
            if candidate:
                offset = base + last + (len(body[last:]) - len(body[last:].lstrip()))
                segments.append(Segment(candidate, offset, offset + len(candidate)))
            last = boundary.end()
        candidate = body[last:].strip()
        if candidate:
            offset = base + last + (len(body[last:]) - len(body[last:].lstrip()))
            segments.append(Segment(candidate, offset, offset + len(candidate)))
    return segments
