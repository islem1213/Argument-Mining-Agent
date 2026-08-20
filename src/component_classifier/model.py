"""Deterministic baseline and optional fine-tuned Transformer inference adapter."""
from __future__ import annotations

from enum import StrEnum

class ComponentLabel(StrEnum):
    MAJOR_CLAIM = "major_claim"
    CLAIM = "claim"
    PREMISE = "premise"
    NON_ARGUMENTATIVE = "non_argumentative"

PREMISE_CUES = ("because", "since", "evidence", "study", "data", "therefore")
CLAIM_CUES = ("should", "must", "need to", "better", "worse")
NON_ARGUMENTATIVE_CUES = ("hello", "thank you", "in conclusion", "for example")

def _heuristic_classify(text: str, index: int = 0) -> ComponentLabel:
    normalized = text.lower()
    if len(normalized.split()) < 4 or any(cue in normalized for cue in NON_ARGUMENTATIVE_CUES):
        return ComponentLabel.NON_ARGUMENTATIVE
    if any(cue in normalized for cue in PREMISE_CUES):
        return ComponentLabel.PREMISE
    if index == 0 or any(cue in normalized for cue in CLAIM_CUES):
        return ComponentLabel.MAJOR_CLAIM if index == 0 else ComponentLabel.CLAIM
    return ComponentLabel.CLAIM

class TransformerComponentClassifier:
    """Load a local/Hugging Face sequence classifier trained on the project label order."""
    def __init__(self, model_path: str):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install training/inference support: pip install -e .[train]") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def classify(self, text: str) -> ComponentLabel:
        encoded = self.tokenizer(text, truncation=True, return_tensors="pt")
        logits = self.model(**encoded).logits[0].detach().tolist()
        index = max(range(len(logits)), key=logits.__getitem__)
        label = self.model.config.id2label.get(index, ComponentLabel.NON_ARGUMENTATIVE)
        return ComponentLabel(str(label).lower().replace("-", "_"))

def classify(text: str, index: int = 0, model: TransformerComponentClassifier | None = None) -> ComponentLabel:
    return model.classify(text) if model else _heuristic_classify(text, index)
