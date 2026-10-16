import pytest

from argmine import ArgumentMiner, PipelineConfig
from argmine.llm import structured
from argmine.relations import RelationResponse

from .conftest import FakeClient

TEXT = "Cities should ban cars. Pollution harms children. Shops would lose customers."


def test_llm_components_and_relations_are_validated_and_filtered():
    client = FakeClient({
        "_LLMLabels": [{"components": [
            {"id": "c1", "label": "major_claim", "confidence": 0.9},
            {"id": "c2", "label": "premise", "confidence": 0.8},
            {"id": "c3", "label": "claim", "confidence": 0.7}]}],
        "RelationResponse": [{"relations": [
            {"source": "c2", "target": "c1", "type": "support", "confidence": 0.9, "rationale": "harm"},
            {"source": "c3", "target": "c1", "type": "attack", "confidence": 0.8, "rationale": "cost"},
            {"source": "c9", "target": "c1", "type": "support", "confidence": 0.9, "rationale": "hallucinated"}]}],
    })
    cfg = PipelineConfig(component_backend="llm", relation_backend="llm")
    g = ArgumentMiner(cfg, client=client).mine(TEXT)
    assert [n.label.value for n in g.nodes] == ["major_claim", "premise", "claim"]
    assert {(e.source, e.target, e.type.value) for e in g.edges} == {("c2", "c1", "support"), ("c3", "c1", "attack")}


def test_structured_repairs_invalid_json_once():
    client = FakeClient({"RelationResponse": ['{"relations": [{"source": 1}]}',
                                              {"relations": []}]})
    assert structured("sys", "user", RelationResponse, client).relations == []
    assert "invalid" in client.calls[1][1][-1]["content"]


def test_structured_gives_up_after_retries():
    client = FakeClient({"RelationResponse": ["nope", "still nope"]})
    with pytest.raises(RuntimeError):
        structured("sys", "user", RelationResponse, client, retries=1)


def test_openai_client_falls_back_to_json_object(monkeypatch):
    pytest.importorskip("openai")
    from types import SimpleNamespace

    from argmine.llm import OpenAICompatibleClient

    seen = []

    def create(**kwargs):
        seen.append(kwargs["response_format"]["type"])
        if kwargs["response_format"]["type"] == "json_schema":
            raise ValueError("response_format json_schema is not supported by this server")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"relations": []}'))])

    client = OpenAICompatibleClient(model="local", base_url="http://localhost:11434/v1", api_key="x")
    monkeypatch.setattr(client._client.chat.completions, "create", create)
    assert structured("s", "u", RelationResponse, client).relations == []
    structured("s", "u", RelationResponse, client)
    assert seen == ["json_schema", "json_object", "json_object"]  # capability is remembered
