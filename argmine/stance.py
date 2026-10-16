"""Stance of each component towards the topic (pro / con / neutral).

* ``heuristic``: transparent lexical proxy (word-boundary matching, so "notably"
  no longer counts as a negation as it did in v0.1).
* ``nli``: zero-shot NLI against the topic — entailment ⇒ pro, contradiction ⇒ con.
  When no topic is given, the major claim is used as the topic.
"""
from __future__ import annotations

import re

from .schema import Stance

_CON = re.compile(r"\b(not|never|no|harm(s|ful)?|worse|critics|opponents|against|risk(s|y)?|danger(ous)?|"
                  r"costly|fail(s|ed)?|problem(s|atic)?|unfair|downside)\b|n't\b", re.IGNORECASE)
_PRO = re.compile(r"\b(should|improve(s|d)?|benefit(s|ial)?|support(s)?|better|help(s|ful)?|advantage(s)?|"
                  r"effective|increase(s)? (learning|safety|health)|positive)\b", re.IGNORECASE)


def heuristic_stance(text: str, marker: str | None = None) -> Stance:
    con, pro = len(_CON.findall(text)), len(_PRO.findall(text))
    if marker in ("however", "but", "although", "though", "nevertheless", "on the other hand"):
        con += 1
    if con > pro:
        return Stance.CON
    if pro > con:
        return Stance.PRO
    return Stance.NEUTRAL


def nli_stances(texts: list[str], topic: str, threshold: float = 0.5) -> list[tuple[Stance, float]]:
    from .nli import get_scorer

    results = []
    for score in get_scorer().score([(t, topic) for t in texts]):
        if score.entailment >= threshold and score.entailment > score.contradiction:
            results.append((Stance.PRO, score.entailment))
        elif score.contradiction >= threshold:
            results.append((Stance.CON, score.contradiction))
        else:
            results.append((Stance.NEUTRAL, score.neutral))
    return results


def estimate_stance(text: str, topic: str | None = None) -> str:
    """Backward-compatible v0.1 entry point."""
    return str(heuristic_stance(text))
