from argmine.reasoning import df_quad, grounded_labelling
from argmine.schema import Relation


def test_grounded_reinstatement():
    # c attacks b, b attacks a  =>  c in, b out, a in (reinstated)
    labels = grounded_labelling(["a", "b", "c"], [("b", "a"), ("c", "b")])
    assert labels == {"a": "in", "b": "out", "c": "in"}


def test_grounded_odd_cycle_is_undecided():
    labels = grounded_labelling(["a", "b"], [("a", "b"), ("b", "a")])
    assert set(labels.values()) == {"undec"}


def test_df_quad():
    rels = [Relation(source="s", target="t", type="support"), Relation(source="a", target="t", type="attack")]
    assert df_quad(["t", "s", "a"], rels)["t"] == 0.5  # equal support and attack cancel out
    assert df_quad(["t", "s"], rels[:1])["t"] == 0.75
    assert df_quad(["t", "a"], rels[1:])["t"] == 0.25
