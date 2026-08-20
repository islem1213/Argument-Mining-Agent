"""Typed relation extraction with an LLM-ready, validated response contract."""
from __future__ import annotations

import json
import os
from enum import StrEnum
from pydantic import BaseModel, Field

class RelationType(StrEnum):
    SUPPORT = "support"
    ATTACK = "attack"

class Relation(BaseModel):
    source: str
    target: str
    type: RelationType
    confidence: float = Field(ge=0, le=1)
    rationale: str | None = None

class RelationResponse(BaseModel):
    relations: list[Relation]

def candidate_pairs(components: list[dict], window: int = 2) -> list[tuple[dict, dict]]:
    """Bound the semantic relation search to nearby argumentative ADUs."""
    pairs: list[tuple[dict, dict]] = []
    argumentative = [c for c in components if c["label"] != "non_argumentative"]
    for i, source in enumerate(argumentative):
        for target in argumentative[max(0, i - window):i]:
            pairs.append((source, target))
    return pairs

def _extract_llm(components: list[dict], window: int) -> RelationResponse:
    """Ask an OpenAI-compatible model for relations, then validate every field."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the LLM extra first: pip install -e .[llm]") from exc
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "not-needed"), base_url=os.environ.get("ARGMINE_LLM_BASE_URL") or None)
    pairs = [{"source": source["id"], "target": target["id"]} for source, target in candidate_pairs(components, window)]
    prompt = (
        "Return JSON only, shaped as {relations: [{source, target, type, confidence, rationale}]}. "
        "Use only the supplied component ids. type must be support or attack. "
        "Do not infer relations when the evidence is insufficient. Components:\\n" + json.dumps(components) + "\\nCandidate pairs only:\\n" + json.dumps(pairs)
    )
    response = client.chat.completions.create(
        model=os.environ.get("ARGMINE_LLM_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "system", "content": "You are a careful argument-relation annotator."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0,
    )
    return RelationResponse.model_validate_json(response.choices[0].message.content or '{"relations": []}')

def extract(components: list[dict], backend: str = "local", window: int = 2) -> RelationResponse:
    """Extract relation edges with a deterministic baseline or validated LLM JSON."""
    if backend == "llm":
        return _extract_llm(components, window)
    relations: list[Relation] = []
    for source, target in candidate_pairs(components, window):
        if source["label"] == "premise":
            relations.append(Relation(source=source["id"], target=target["id"], type="support", confidence=0.55, rationale="Local premise-to-nearby-component heuristic."))
            break
    return RelationResponse(relations=relations)
