"""Discourse-aware segmentation into candidate ADUs with exact source offsets.

Sentences are split first (abbreviation- and decimal-safe), then a small set of
high-precision discourse markers split sentences into clauses. Unlike v0.1 the
marker is *kept* as metadata, because markers such as "because" or "however"
are the strongest cheap signal for component type and relation direction.
Spans exclude the marker itself, which matches the AAEC annotation convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CONTRAST = ("however", "but", "although", "though", "yet", "nevertheless", "nonetheless",
            "on the other hand", "whereas", "in contrast", "even so")
REASON = ("because", "since", "given that", "due to the fact that", "as shown by",
          "for example", "for instance", "in fact", "studies show", "research shows")
CONCLUSION = ("therefore", "thus", "hence", "so", "consequently", "as a result",
              "it follows that", "which means", "this means", "in conclusion", "overall")
MARKERS = {**{m: "contrast" for m in CONTRAST}, **{m: "reason" for m in REASON},
           **{m: "conclusion" for m in CONCLUSION}}

# Markers that may open a new clause in the middle of a sentence. "so", "since" and
# "yet" are ambiguous, so they only split when preceded by a comma or semicolon.
_STRONG_MID = r"because|therefore|however|although|whereas|consequently|hence|thus|but"
_WEAK_MID = r"so|since|yet|which means|as a result"
CLAUSE_BREAK = re.compile(
    rf"\s*;\s*|\s+(?=(?:{_STRONG_MID})\b)|,\s+(?=(?:{_WEAK_MID})\b)", re.IGNORECASE
)
_LEADING_MARKER = re.compile(
    r"^(?P<marker>" + "|".join(sorted((re.escape(m) for m in MARKERS), key=len, reverse=True))
    + r")\b[\s,:]*",
    re.IGNORECASE,
)
_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "st", "fig", "al", "no"}
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int
    marker: str | None = None
    sentence: int = 0
    sentence_initial: bool = True

    @property
    def marker_type(self) -> str | None:
        return MARKERS.get(self.marker) if self.marker else None


def sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for sentences, skipping common abbreviations."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        before = text[start:match.start()].split()
        last_word = before[-1].lower().rstrip(".") if before else ""
        if match.group().startswith(".") and (last_word in _ABBREVIATIONS or len(last_word) == 1):
            continue
        spans.append((start, match.end()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _make_segment(text: str, start: int, end: int, sentence: int = 0, initial: bool = True) -> Segment | None:
    raw = text[start:end]
    start += len(raw) - len(raw.lstrip())
    raw = raw.strip().rstrip(",")
    marker = None
    found = _LEADING_MARKER.match(raw)
    if found and len(raw) > found.end():
        marker = found.group("marker").lower()
        start += found.end()
        raw = raw[found.end():]
    if not re.search(r"\w", raw):
        return None
    return Segment(raw, start, start + len(raw), marker, sentence, initial)


def segment(text: str) -> list[Segment]:
    """Split ``text`` into candidate ADUs whose offsets index the original string."""
    segments: list[Segment] = []
    for index, (s_start, s_end) in enumerate(sentences(text)):
        body = text[s_start:s_end]
        last = 0
        for boundary in CLAUSE_BREAK.finditer(body):
            if boundary.start() == 0:
                continue
            seg = _make_segment(text, s_start + last, s_start + boundary.start(), index, last == 0)
            # Very short fragments ("Yes but") are not worth a separate ADU.
            if seg and len(seg.text.split()) >= 3:
                segments.append(seg)
                last = boundary.end()
        seg = _make_segment(text, s_start + last, s_end, index, last == 0)
        if seg:
            segments.append(seg)
    return segments
