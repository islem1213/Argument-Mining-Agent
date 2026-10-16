"""Component classifiers: discourse heuristic, fine-tuned Transformer, and LLM.

All backends implement ``classify_all(segments) -> list[(label, confidence)]`` so
the pipeline can swap them freely.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field

from ..schema import ComponentLabel
from ..segmenter import Segment

Prediction = tuple[ComponentLabel, float]

EVIDENCE_CUES = re.compile(
    r"\b(stud(y|ies)|research|data|survey|evidence|according to|percent|statistics?|experts?|"
    r"report(s|ed)?|found that|shows?|for (example|instance))\b|\d+(\.\d+)?\s?%",
    re.IGNORECASE,
)
CLAIM_CUES = re.compile(
    r"\b(should|must|ought to|need to|it is (clear|obvious|important)|i (believe|think|argue)|"
    r"we (should|must)|better|worse|best|worst|essential|necessary|unacceptable)\b",
    re.IGNORECASE,
)
COUNTER_CUES = re.compile(r"\b(critics|opponents|some (people|argue|say|believe)|others (argue|believe|claim))\b",
                          re.IGNORECASE)
NON_ARGUMENTATIVE = re.compile(
    r"^(hello|hi|thank you|thanks|in this essay|this essay|let us|let's|first(ly)?,? i will)\b|\?$",
    re.IGNORECASE,
)


class ComponentClassifier(Protocol):
    name: str

    def classify_all(self, segments: list[Segment]) -> list[Prediction]: ...


class HeuristicClassifier:
    """Transparent, deterministic baseline driven by discourse markers and lexical cues."""

    name = "heuristic"

    def _one(self, seg: Segment, has_claim_before: bool) -> Prediction:
        text = seg.text
        if NON_ARGUMENTATIVE.search(text.strip()) or (len(text.split()) < 3 and not seg.marker):
            return ComponentLabel.NON_ARGUMENTATIVE, 0.6
        if seg.marker in ("in conclusion", "overall"):
            return ComponentLabel.MAJOR_CLAIM, 0.6
        if seg.marker_type == "reason" or EVIDENCE_CUES.search(text):
            return ComponentLabel.PREMISE, 0.65
        if seg.marker_type == "conclusion" or COUNTER_CUES.search(text):
            return ComponentLabel.CLAIM, 0.6
        if seg.marker_type == "contrast":
            # "X, but Y" rebuts X locally; a sentence-initial "However, Y" is a counter-claim.
            return (ComponentLabel.CLAIM, 0.55) if seg.sentence_initial else (ComponentLabel.PREMISE, 0.55)
        if not has_claim_before:
            # The first standpoint in an essay is usually its thesis.
            return ComponentLabel.MAJOR_CLAIM, 0.6 if CLAIM_CUES.search(text) else 0.45
        if CLAIM_CUES.search(text):
            return ComponentLabel.CLAIM, 0.55
        return ComponentLabel.PREMISE, 0.5

    def classify_all(self, segments: list[Segment]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for seg in segments:
            seen_claim = any(label in (ComponentLabel.MAJOR_CLAIM, ComponentLabel.CLAIM) for label, _ in predictions)
            predictions.append(self._one(seg, seen_claim))
        return refine_with_discourse(segments, predictions)


def refine_with_discourse(segments: list[Segment], predictions: list[Prediction]) -> list[Prediction]:
    """Fix labels using the *next* segment's marker ("A, so B" ⇒ A is a premise of B)."""
    refined = list(predictions)
    for i in range(len(segments) - 1):
        nxt = segments[i + 1].marker_type
        label, conf = refined[i]
        if label == ComponentLabel.NON_ARGUMENTATIVE:
            continue
        if nxt == "conclusion" and label != ComponentLabel.MAJOR_CLAIM:
            refined[i] = (ComponentLabel.PREMISE, max(conf, 0.6))
        elif nxt == "reason" and label == ComponentLabel.PREMISE:
            refined[i] = (ComponentLabel.CLAIM, conf)
    return refined


class TransformerComponentClassifier:
    """Load a local or Hugging Face sequence classifier trained with ``argmine.components.train``."""

    name = "transformer"

    def __init__(self, model_path: str):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("Install training/inference support: pip install -e '.[train]'") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()

    def classify_all(self, segments: list[Segment]) -> list[Prediction]:
        if not segments:
            return []
        encoded = self.tokenizer([s.text for s in segments], truncation=True, padding=True, return_tensors="pt")
        with self._torch.no_grad():
            probs = self.model(**encoded).logits.softmax(-1)
        out: list[Prediction] = []
        for row in probs:
            index = int(row.argmax())
            label = str(self.model.config.id2label.get(index, "non_argumentative")).lower().replace("-", "_")
            out.append((ComponentLabel(label), float(row[index])))
        return out


class _LLMLabel(BaseModel):
    id: str
    label: ComponentLabel
    confidence: float = Field(ge=0, le=1)


class _LLMLabels(BaseModel):
    components: list[_LLMLabel]


LLM_COMPONENT_GUIDE = """Label each argumentative discourse unit (ADU) using the persuasive-essay scheme \
(Stab & Gurevych, 2017):
- major_claim: the author's central standpoint on the topic (usually one, sometimes restated in the conclusion)
- claim: a controversial statement that supports or attacks the major claim and itself needs justification
- premise: a reason, piece of evidence or example given to support or attack a claim or another premise
- non_argumentative: greetings, meta-text, questions, or background that takes no argumentative role
Units are given in document order with their preceding discourse marker. Use every id exactly once."""


class LLMComponentClassifier:
    """Zero-shot labelling of all segments in a single structured-output call."""

    name = "llm"

    def __init__(self, client=None, chunk_size: int = 60):
        self.client, self.chunk_size = client, chunk_size

    def classify_all(self, segments: list[Segment]) -> list[Prediction]:
        from ..llm import structured

        if not segments:
            return []
        units = [{"id": f"c{i + 1}", "marker": s.marker, "text": s.text} for i, s in enumerate(segments)]
        by_id: dict[str, Prediction] = {}
        for start in range(0, len(units), self.chunk_size):  # long documents are labelled in chunks
            chunk = units[start:start + self.chunk_size]
            result = structured("You are an expert annotator of argumentative text.",
                                LLM_COMPONENT_GUIDE + "\n\nUnits:\n" + json.dumps(chunk, ensure_ascii=False),
                                _LLMLabels, self.client)
            by_id.update({item.id: (item.label, item.confidence) for item in result.components})
        fallback = HeuristicClassifier().classify_all(segments)
        return [by_id.get(f"c{i + 1}", fallback[i]) for i in range(len(segments))]


def get_classifier(backend: str = "heuristic", model_path: str | None = None, client=None) -> ComponentClassifier:
    if model_path or backend == "transformer":
        if not model_path:
            raise ValueError("the transformer backend needs --classifier-model")
        return TransformerComponentClassifier(model_path)
    if backend == "llm":
        return LLMComponentClassifier(client)
    return HeuristicClassifier()
