"""Formal-argumentation layer on top of the mined graph.

Mining tells us *what* the structure is; this module tells us what follows from
it. Two complementary, well-studied semantics are implemented:

* **Grounded semantics** (Dung, 1995) on the attack relation: the most sceptical
  set of arguments that can be defended. Each component is labelled
  ``in`` (accepted), ``out`` (defeated) or ``undec``.
* **DF-QuAD gradual semantics** (Rago et al., 2016) on the bipolar graph: a
  strength in [0, 1] that combines supporters and attackers recursively, which
  is more informative than an in/out label for real text.

It also reports structural diagnostics that are useful for feedback on essays:
unsupported claims, isolated components and the most central claim.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx

from .schema import Analysis, Component, ComponentAnalysis, ComponentLabel, Relation, RelationType


def grounded_labelling(nodes: list[str], attacks: list[tuple[str, str]]) -> dict[str, str]:
    attackers: dict[str, set[str]] = defaultdict(set)
    for source, target in attacks:
        attackers[target].add(source)
    label: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node in label:
                continue
            if all(label.get(a) == "out" for a in attackers[node]):
                label[node], changed = "in", True
            elif any(label.get(a) == "in" for a in attackers[node]):
                label[node], changed = "out", True
    return {n: label.get(n, "undec") for n in nodes}


def _aggregate(values: list[float]) -> float:
    """Probabilistic sum used by DF-QuAD: 1 - ∏(1 - v)."""
    result = 1.0
    for v in values:
        result *= 1 - v
    return 1 - result if values else 0.0


def df_quad(nodes: list[str], relations: list[Relation], base: dict[str, float] | None = None,
            iterations: int = 100, tol: float = 1e-6) -> dict[str, float]:
    base = base or {}
    supporters, attackers = defaultdict(list), defaultdict(list)
    for rel in relations:
        (supporters if rel.type == RelationType.SUPPORT else attackers)[rel.target].append(rel.source)
    strength = {n: base.get(n, 0.5) for n in nodes}
    for _ in range(iterations):
        updated = {}
        for n in nodes:
            v0 = base.get(n, 0.5)
            va = _aggregate([strength[a] for a in attackers[n]])
            vs = _aggregate([strength[s] for s in supporters[n]])
            updated[n] = v0 - v0 * (va - vs) if va >= vs else v0 + (1 - v0) * (vs - va)
        delta = max((abs(updated[n] - strength[n]) for n in nodes), default=0.0)
        strength = updated
        if delta < tol:
            break
    return {n: round(v, 4) for n, v in strength.items()}


def analyse(components: list[Component], relations: list[Relation]) -> Analysis:
    units = [c for c in components if c.label != ComponentLabel.NON_ARGUMENTATIVE]
    ids = [c.id for c in units]
    rels = [r for r in relations if r.source in ids and r.target in ids]
    grounded = grounded_labelling(ids, [(r.source, r.target) for r in rels if r.type == RelationType.ATTACK])
    strength = df_quad(ids, rels)
    sup, att = defaultdict(int), defaultdict(int)
    touched = set()
    for r in rels:
        (sup if r.type == RelationType.SUPPORT else att)[r.target] += 1
        touched |= {r.source, r.target}

    central = None
    claims = [c for c in units if c.label in (ComponentLabel.MAJOR_CLAIM, ComponentLabel.CLAIM)]
    if claims:
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        graph.add_edges_from((r.source, r.target) for r in rels)
        # Centrality = how many components (directly or transitively) argue about it; pure Python, no SciPy.
        rank = {n: len(nx.ancestors(graph, n)) for n in ids}
        majors = [c for c in claims if c.label == ComponentLabel.MAJOR_CLAIM] or claims
        central = max(majors, key=lambda c: (rank.get(c.id, 0.0), -int(c.id[1:]) if c.id[1:].isdigit() else 0)).id

    return Analysis(
        components={n: ComponentAnalysis(grounded=grounded[n], strength=strength[n], supporters=sup[n],
                                         attackers=att[n]) for n in ids},
        central_claim=central,
        unsupported_claims=[c.id for c in claims if sup[c.id] == 0],
        isolated_components=[n for n in ids if n not in touched] if len(ids) > 1 else [],
    )
