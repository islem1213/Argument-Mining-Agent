from argmine.data.brat import component_rows, parse_ann
from argmine.evaluate import evaluate

TEXT = "Title\n\nWe should act now. Because delays cost lives, waiting is wrong."
ANN = (
    "T1\tMajorClaim 7 24\tWe should act now\n"
    "T2\tPremise 33 50\tdelays cost lives\n"
    "T3\tClaim 52 68\twaiting is wrong\n"
    "R1\tsupports Arg1:T2 Arg2:T3\n"
    "A1\tStance T3 For\n"
)


def test_parse_brat():
    g = parse_ann(ANN, TEXT)
    assert [n["label"] for n in g["nodes"]] == ["major_claim", "premise", "claim"]
    assert {"source": "T3", "target": "T1", "type": "support"} in g["edges"]
    assert all(TEXT[n["start"]:n["end"]] == n["text"] for n in g["nodes"])
    assert {r["label"] for r in component_rows({"text": TEXT, **g})} >= {"premise", "claim"}


def test_span_alignment_makes_relation_eval_id_independent():
    gold = {"d": {"id": "d", **parse_ann(ANN, TEXT)}}
    pred = {"d": {"id": "d", "nodes": [
        {"id": "c1", "start": 7, "end": 25, "label": "major_claim"},   # includes the full stop: approximate match
        {"id": "c2", "start": 33, "end": 50, "label": "premise"},
        {"id": "c3", "start": 52, "end": 69, "label": "claim"}],
        "edges": [{"source": "c2", "target": "c3", "type": "support"},
                  {"source": "c3", "target": "c1", "type": "attack"}]}}
    report = evaluate(gold, pred)
    assert report["span"]["f1"] == 1.0
    assert report["component_macro_f1"] == 1.0
    assert report["link"]["f1"] == 1.0
    assert report["relation"]["f1"] == 0.5
    assert evaluate(gold, pred, threshold=1.0)["span"]["f1"] < 1.0
