from argmine import ArgumentMiner, PipelineConfig, run
from argmine.schema import ArgumentGraph, ComponentLabel, RelationType


def test_discourse_structure(sample):
    g = ArgumentMiner().mine(sample)
    labels = {n.text: n.label for n in g.nodes}
    assert labels["Schools should start later in the morning."] == ComponentLabel.MAJOR_CLAIM
    assert labels["Teenagers need more sleep"] == ComponentLabel.PREMISE
    edges = {(g.node(e.source).text[:12], g.node(e.target).text[:12], e.type) for e in g.edges}
    assert ("Teenagers ne", "later starts", RelationType.SUPPORT) in edges  # "A, so B"
    assert ("critics worr", "Schools shou", RelationType.ATTACK) in edges   # "However, ..." counter-claim
    assert ("many distric", "Buses would ", RelationType.ATTACK) in edges   # "X, but Y"


def test_tree_constraint_and_valid_ids(sample):
    g = ArgumentMiner().mine(sample)
    sources = [e.source for e in g.edges]
    assert len(sources) == len(set(sources))
    ids = {n.id for n in g.nodes}
    assert all(e.source in ids and e.target in ids for e in g.edges)


def test_round_trip_and_v01_wrapper(sample):
    data = run(sample)
    assert data["schema_version"] == "0.2"
    assert ArgumentGraph.model_validate(data).nodes


def test_fallacy_flags():
    g = ArgumentMiner(PipelineConfig(fallacies="heuristic")).mine(
        "We must ban phones. Everyone knows they ruin focus.")
    assert any(f.fallacy.value == "hasty_generalization" for f in g.analysis.fallacies)


def test_empty_text():
    g = ArgumentMiner().mine("   ")
    assert g.nodes == [] and g.edges == []
