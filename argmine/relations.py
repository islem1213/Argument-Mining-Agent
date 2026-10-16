"""Typed relation extraction: discourse heuristic, local NLI, or structured LLM judge.

Every backend returns validated :class:`~argmine.schema.Relation` objects, and the
same post-processing applies to all of them: unknown ids and self-loops are
dropped, and (optionally) each component keeps at most one outgoing edge, which
is the tree constraint of the persuasive-essay annotation scheme.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from .schema import Component, ComponentLabel, Relation, RelationType, Stance

CLAIMS = (ComponentLabel.MAJOR_CLAIM, ComponentLabel.CLAIM)


class RelationResponse(BaseModel):
    relations: list[Relation]


def argumentative(components: list[Component]) -> list[Component]:
    return [c for c in components if c.label != ComponentLabel.NON_ARGUMENTATIVE]


def candidate_pairs(components: list[Component], window: int = 2) -> list[tuple[Component, Component]]:
    """Plausible (source, target) pairs: neighbours within ``window`` ADUs in both
    directions, plus every claim → every major claim (long-range by design)."""
    units = argumentative(components)
    pairs: dict[tuple[str, str], tuple[Component, Component]] = {}
    for i, source in enumerate(units):
        if source.label == ComponentLabel.MAJOR_CLAIM:
            continue
        for j in range(max(0, i - window), min(len(units), i + window + 1)):
            target = units[j]
            if j != i and not (source.label in CLAIMS and target.label == ComponentLabel.PREMISE):
                pairs[(source.id, target.id)] = (source, target)
        if source.label == ComponentLabel.CLAIM:
            for target in units:
                if target.label == ComponentLabel.MAJOR_CLAIM:
                    pairs[(source.id, target.id)] = (source, target)
    return list(pairs.values())


def postprocess(relations: list[Relation], components: list[Component], tree: bool = True) -> list[Relation]:
    ids = {c.id for c in components}
    clean = [r for r in relations if r.source in ids and r.target in ids and r.source != r.target]
    if tree:
        best: dict[str, Relation] = {}
        for rel in clean:
            if rel.source not in best or rel.confidence > best[rel.source].confidence:
                best[rel.source] = rel
        clean = list(best.values())
    unique = {(r.source, r.target): r for r in clean}
    return sorted(unique.values(), key=lambda r: (int(r.source[1:]) if r.source[1:].isdigit() else 0, r.target))


# --------------------------------------------------------------------------- heuristic

def _heuristic(components: list[Component], window: int) -> list[Relation]:
    units = argumentative(components)
    majors = [c for c in units if c.label == ComponentLabel.MAJOR_CLAIM]
    relations: list[Relation] = []

    def nearest(i: int, labels: tuple[ComponentLabel, ...], direction: int) -> Component | None:
        rng = range(i - 1, max(-1, i - 1 - window), -1) if direction < 0 else range(i + 1, min(len(units), i + 1 + window))
        return next((units[j] for j in rng if units[j].label in labels), None)

    for i, unit in enumerate(units):
        if unit.label == ComponentLabel.MAJOR_CLAIM:
            continue
        marker_type = _marker_type(unit)
        if unit.label == ComponentLabel.CLAIM:
            target = majors[0] if majors else None
            if target and target.id != unit.id:
                attack = marker_type == "contrast" or unit.stance == Stance.CON
                if marker_type == "contrast" and i > 0 and units[i - 1].sentence == unit.sentence:
                    target = units[i - 1]
                relations.append(Relation(source=unit.id, target=target.id,
                                          type=RelationType.ATTACK if attack else RelationType.SUPPORT,
                                          confidence=0.55, rationale="claim → major claim (stance/marker)"))
            continue
        # Premises.
        if marker_type == "contrast":
            target = units[i - 1] if i > 0 else None
            if target:
                relations.append(Relation(source=unit.id, target=target.id, type=RelationType.ATTACK,
                                          confidence=0.6, rationale=f"contrast marker '{unit.marker}'"))
            continue
        nxt = units[i + 1] if i + 1 < len(units) else None
        if nxt is not None and _marker_type(nxt) == "conclusion":
            relations.append(Relation(source=unit.id, target=nxt.id, type=RelationType.SUPPORT, confidence=0.65,
                                      rationale=f"followed by conclusion marker '{nxt.marker}'"))
            continue
        target = nearest(i, CLAIMS, -1) or nearest(i, CLAIMS, +1) or nearest(i, (ComponentLabel.PREMISE,), -1)
        if target:
            confidence = 0.65 if marker_type == "reason" else 0.5
            why = f"reason marker '{unit.marker}'" if marker_type == "reason" else "nearest claim"
            relations.append(Relation(source=unit.id, target=target.id, type=RelationType.SUPPORT,
                                      confidence=confidence, rationale=why))
    return relations


def _marker_type(component: Component) -> str | None:
    from .segmenter import MARKERS

    return MARKERS.get(component.marker) if component.marker else None


# --------------------------------------------------------------------------- NLI

def _nli(components: list[Component], window: int, threshold: float = 0.55) -> list[Relation]:
    from .nli import get_scorer

    pairs = candidate_pairs(components, window)
    scores = get_scorer().score([(s.text, t.text) for s, t in pairs])
    relations = []
    for (source, target), score in zip(pairs, scores):
        if score.entailment >= threshold and score.entailment >= score.contradiction:
            relations.append(Relation(source=source.id, target=target.id, type=RelationType.SUPPORT,
                                      confidence=round(score.entailment, 4), rationale="NLI entailment"))
        elif score.contradiction >= threshold:
            relations.append(Relation(source=source.id, target=target.id, type=RelationType.ATTACK,
                                      confidence=round(score.contradiction, 4), rationale="NLI contradiction"))
    return relations


# --------------------------------------------------------------------------- LLM

LLM_RELATION_GUIDE = """Decide, for each candidate pair, whether SOURCE supports or attacks TARGET.
- support: SOURCE gives a reason or evidence that makes TARGET more acceptable
- attack: SOURCE rebuts or undercuts TARGET, making it less acceptable
Only return pairs from the candidate list, and omit pairs with no clear relation rather than guessing.
Give a one-sentence rationale and a calibrated confidence in [0, 1]."""


def _llm(components: list[Component], window: int, client=None, batch_size: int = 40) -> list[Relation]:
    from .llm import structured

    pairs = candidate_pairs(components, window)
    units = [{"id": c.id, "label": str(c.label), "marker": c.marker, "text": c.text} for c in argumentative(components)]
    allowed = {(s.id, t.id) for s, t in pairs}
    relations: list[Relation] = []
    for i in range(0, len(pairs), batch_size):
        batch = [{"source": s.id, "target": t.id} for s, t in pairs[i:i + batch_size]]
        result = structured(
            "You are a careful argument-relation annotator.",
            LLM_RELATION_GUIDE + "\n\nComponents:\n" + json.dumps(units, ensure_ascii=False)
            + "\n\nCandidate pairs:\n" + json.dumps(batch),
            RelationResponse, client,
        )
        relations += [r for r in result.relations if (r.source, r.target) in allowed]
    return relations


def extract(components: list[Component], backend: str = "heuristic", window: int = 2, tree: bool = True,
            client=None) -> list[Relation]:
    """Extract relation edges with the chosen backend (``heuristic`` | ``nli`` | ``llm``)."""
    if backend in ("local", "heuristic"):
        raw = _heuristic(components, window)
    elif backend == "nli":
        raw = _nli(components, window)
    elif backend == "llm":
        raw = _llm(components, window, client)
    else:
        raise ValueError(f"unknown relation backend: {backend}")
    return postprocess(raw, components, tree)
