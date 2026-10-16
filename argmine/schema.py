"""Typed data contract shared by every pipeline stage.

The JSON produced by :class:`ArgumentGraph` is the public, versioned output of
``argmine``. Every backend (heuristic, NLI, LLM) must produce these models, so
downstream code never has to care which backend generated a graph.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.2"


class _StrEnum(str, Enum):
    """``enum.StrEnum`` equivalent that also works on Python 3.10."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ComponentLabel(_StrEnum):
    MAJOR_CLAIM = "major_claim"
    CLAIM = "claim"
    PREMISE = "premise"
    NON_ARGUMENTATIVE = "non_argumentative"


class Stance(_StrEnum):
    PRO = "pro"
    CON = "con"
    NEUTRAL = "neutral"


class RelationType(_StrEnum):
    SUPPORT = "support"
    ATTACK = "attack"


class Fallacy(_StrEnum):
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_DILEMMA = "false_dilemma"
    SLIPPERY_SLOPE = "slippery_slope"
    HASTY_GENERALIZATION = "hasty_generalization"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    APPEAL_TO_EMOTION = "appeal_to_emotion"
    CIRCULAR_REASONING = "circular_reasoning"
    RED_HERRING = "red_herring"
    FALSE_CAUSE = "false_cause"


class Component(BaseModel):
    """An argumentative discourse unit (ADU) anchored to character offsets."""

    id: str
    text: str
    start: int
    end: int
    label: ComponentLabel
    stance: Stance = Stance.NEUTRAL
    marker: str | None = Field(default=None, description="Discourse marker that introduced the span")
    sentence: int | None = Field(default=None, description="Index of the sentence containing the span")
    confidence: float | None = Field(default=None, ge=0, le=1)


class Relation(BaseModel):
    """A directed edge: ``source`` supports or attacks ``target``."""

    source: str
    target: str
    type: RelationType
    confidence: float = Field(default=0.5, ge=0, le=1)
    rationale: str | None = None


class FallacyFinding(BaseModel):
    component: str
    fallacy: Fallacy
    confidence: float = Field(ge=0, le=1)
    explanation: str


class ComponentAnalysis(BaseModel):
    """Output of the formal-argumentation layer for one component."""

    grounded: str = Field(description="in / out / undec under Dung's grounded semantics (attacks only)")
    strength: float = Field(ge=0, le=1, description="DF-QuAD gradual strength over supports and attacks")
    supporters: int = 0
    attackers: int = 0


class Analysis(BaseModel):
    components: dict[str, ComponentAnalysis] = Field(default_factory=dict)
    central_claim: str | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    isolated_components: list[str] = Field(default_factory=list)
    fallacies: list[FallacyFinding] = Field(default_factory=list)


class ArgumentGraph(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str | None = None
    topic: str | None = None
    nodes: list[Component]
    edges: list[Relation]
    analysis: Analysis | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def node(self, node_id: str) -> Component:
        return next(n for n in self.nodes if n.id == node_id)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)
