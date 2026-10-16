"""Fallacy and weakness detection over mined components.

* ``heuristic``: lexical cue spotting. Low precision by design — it flags spans
  worth a second look, with confidence capped at 0.4.
* ``llm``: a structured-output judge restricted to a fixed fallacy taxonomy,
  which must quote no new ids and explain each finding.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from .schema import Component, ComponentLabel, Fallacy, FallacyFinding, Relation

_CUES: list[tuple[Fallacy, re.Pattern[str], str]] = [
    (Fallacy.AD_HOMINEM, re.compile(r"\b(idiots?|stupid|liars?|ignorant|clowns?|hypocrites?)\b", re.I),
     "attacks the person rather than the argument"),
    (Fallacy.SLIPPERY_SLOPE, re.compile(r"\b(will (inevitably|eventually) lead to|next thing you know|"
                                        r"before (long|we know it)|opens the floodgates)\b", re.I),
     "predicts a chain of consequences without support"),
    (Fallacy.FALSE_DILEMMA, re.compile(r"\b(either\b.{1,60}\bor\b|the only (option|way|choice)|"
                                       r"there is no alternative)", re.I),
     "presents only two options"),
    (Fallacy.HASTY_GENERALIZATION, re.compile(r"\b(everyone|nobody|no one|always|never|all (people|students|"
                                              r"teenagers|parents))\b", re.I),
     "generalises from limited evidence"),
    (Fallacy.APPEAL_TO_AUTHORITY, re.compile(r"\b(experts (agree|say)|scientists say|every(body|one) knows|"
                                             r"it is well known)\b", re.I),
     "appeals to popularity or unnamed authority instead of evidence"),
    (Fallacy.APPEAL_TO_EMOTION, re.compile(r"\b(think of the children|heartbreaking|terrifying|outrageous)\b", re.I),
     "relies on emotional language"),
]


def heuristic_fallacies(components: list[Component]) -> list[FallacyFinding]:
    findings = []
    for comp in components:
        if comp.label == ComponentLabel.NON_ARGUMENTATIVE:
            continue
        for fallacy, pattern, why in _CUES:
            match = pattern.search(comp.text)
            if match:
                findings.append(FallacyFinding(component=comp.id, fallacy=fallacy, confidence=0.4,
                                               explanation=f"Cue '{match.group(0)}': {why}."))
    return findings


class _Findings(BaseModel):
    findings: list[FallacyFinding]


def llm_fallacies(components: list[Component], relations: list[Relation], client=None) -> list[FallacyFinding]:
    from .llm import structured

    ids = {c.id for c in components}
    payload = {
        "components": [{"id": c.id, "label": str(c.label), "text": c.text} for c in components
                       if c.label != ComponentLabel.NON_ARGUMENTATIVE],
        "relations": [r.model_dump(mode="json", include={"source", "target", "type"}) for r in relations],
        "taxonomy": [f.value for f in Fallacy],
    }
    result = structured(
        "You are a critical-thinking tutor who identifies informal fallacies conservatively.",
        "Identify fallacies in the argument below. Use only the taxonomy labels and component ids given. "
        "Report a fallacy only when it is clearly present, and explain it in one sentence.\n\n"
        + json.dumps(payload, ensure_ascii=False),
        _Findings, client,
    )
    return [f for f in result.findings if f.component in ids]


def detect(components: list[Component], relations: list[Relation], backend: str = "heuristic",
           client=None) -> list[FallacyFinding]:
    if backend == "llm":
        return llm_fallacies(components, relations, client)
    return heuristic_fallacies(components)
