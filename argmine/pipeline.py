"""End-to-end pipeline: segment → classify → stance → relations → reasoning → critique."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from . import critique as critique_mod
from .components.model import ComponentClassifier, get_classifier
from .reasoning import analyse
from .relations import extract
from .schema import ArgumentGraph, Component, ComponentLabel, Stance
from .segmenter import segment
from .stance import heuristic_stance, nli_stances


@dataclass
class PipelineConfig:
    component_backend: str = "heuristic"   # heuristic | transformer | llm
    classifier_model: str | None = None    # path or HF id for the transformer backend
    stance_backend: str = "heuristic"      # heuristic | nli
    relation_backend: str = "heuristic"    # heuristic | nli | llm
    relation_window: int = 2
    tree: bool = True                      # at most one outgoing relation per component
    reasoning: bool = True                 # grounded + DF-QuAD analysis
    fallacies: str | None = None           # None | heuristic | llm


class ArgumentMiner:
    def __init__(self, config: PipelineConfig | None = None, client=None,
                 classifier: ComponentClassifier | None = None):
        self.config = config or PipelineConfig()
        self.client = client
        self.classifier = classifier or get_classifier(self.config.component_backend,
                                                       self.config.classifier_model, client)

    def mine(self, text: str, topic: str | None = None, doc_id: str | None = None) -> ArgumentGraph:
        cfg, started = self.config, time.perf_counter()
        segments = segment(text)
        predictions = self.classifier.classify_all(segments)
        components = [
            Component(id=f"c{i + 1}", text=s.text, start=s.start, end=s.end, label=label, marker=s.marker, sentence=s.sentence,
                      confidence=round(conf, 4), stance=heuristic_stance(s.text, s.marker))
            for i, (s, (label, conf)) in enumerate(zip(segments, predictions))
        ]
        if cfg.stance_backend == "nli":
            reference = topic or next((c.text for c in components if c.label == ComponentLabel.MAJOR_CLAIM), None)
            if reference:
                for comp, (stance, _) in zip(components, nli_stances([c.text for c in components], reference)):
                    comp.stance = stance
        for comp in components:
            if comp.label == ComponentLabel.MAJOR_CLAIM and topic is None:
                comp.stance = Stance.PRO  # the thesis is, by definition, pro its own standpoint

        relations = extract(components, cfg.relation_backend, cfg.relation_window, cfg.tree, self.client)
        analysis = analyse(components, relations) if cfg.reasoning else None
        if cfg.fallacies:
            findings = critique_mod.detect(components, relations, cfg.fallacies, self.client)
            if analysis is None:
                from .schema import Analysis
                analysis = Analysis()
            analysis.fallacies = findings
        metadata = {k: str(v) for k, v in asdict(cfg).items() if v is not None}
        metadata["elapsed_ms"] = str(round(1000 * (time.perf_counter() - started)))
        return ArgumentGraph(id=doc_id, topic=topic, nodes=components, edges=relations, analysis=analysis,
                             metadata=metadata)


def run(text: str, relation_backend: str = "heuristic", relation_window: int = 2, topic: str | None = None,
        classifier: ComponentClassifier | None = None, **options) -> dict:
    """Convenience wrapper returning the portable JSON dict (kept for v0.1 callers)."""
    config = PipelineConfig(relation_backend=relation_backend, relation_window=relation_window, **options)
    return ArgumentMiner(config, classifier=classifier).mine(text, topic).to_dict()
