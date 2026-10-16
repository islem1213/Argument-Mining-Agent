"""The NLI backend is exercised with a stub scorer so CI needs no model download."""
import argmine.nli as nli
from argmine import ArgumentMiner, PipelineConfig
from argmine.nli import NLIScore


class StubScorer:
    def score(self, pairs, batch_size=16):
        out = []
        for premise, _hypothesis in pairs:
            if "costs" in premise:
                out.append(NLIScore(0.05, 0.9, 0.05))
            elif "sleep" in premise or "learning" in premise:
                out.append(NLIScore(0.8, 0.05, 0.15))
            else:
                out.append(NLIScore(0.1, 0.1, 0.8))
        return out


def test_nli_relations_and_stance(monkeypatch):
    monkeypatch.setattr(nli, "get_scorer", lambda *a, **k: StubScorer())
    text = "Schools should start later. Teenagers need more sleep. However, critics worry about transport costs."
    g = ArgumentMiner(PipelineConfig(relation_backend="nli", stance_backend="nli")).mine(text)
    types = {(e.source, e.type.value) for e in g.edges}
    assert ("c2", "support") in types and ("c3", "attack") in types
    assert g.node("c3").stance.value == "con"
