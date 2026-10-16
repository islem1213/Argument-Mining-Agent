"""Local natural-language-inference scorer (no API, runs on CPU).

An NLI cross-encoder gives a cheap, fully self-hosted signal for both relation
typing (entailment ≈ support, contradiction ≈ attack) and topic stance. Any
Hugging Face NLI model whose labels contain "entail" and "contradict" works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_NLI_MODEL = os.environ.get("ARGMINE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")


@dataclass(frozen=True)
class NLIScore:
    entailment: float
    contradiction: float
    neutral: float


class NLIScorer:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("Install local model support: pip install -e '.[nli]'") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {i: str(name).lower() for i, name in self.model.config.id2label.items()}
        self._ent = next(i for i, n in labels.items() if "entail" in n)
        self._con = next(i for i, n in labels.items() if "contradict" in n)
        self._neu = next((i for i, n in labels.items() if "neutral" in n), None)

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[NLIScore]:
        """Score ``(premise, hypothesis)`` pairs."""
        results: list[NLIScore] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            encoded = self.tokenizer([p for p, _ in batch], [h for _, h in batch], truncation=True,
                                     padding=True, return_tensors="pt")
            with self._torch.no_grad():
                probs = self.model(**encoded).logits.softmax(-1).tolist()
            for row in probs:
                neutral = row[self._neu] if self._neu is not None else 1 - row[self._ent] - row[self._con]
                results.append(NLIScore(row[self._ent], row[self._con], neutral))
        return results


@lru_cache(maxsize=2)
def get_scorer(model_name: str = DEFAULT_NLI_MODEL) -> NLIScorer:
    return NLIScorer(model_name)
