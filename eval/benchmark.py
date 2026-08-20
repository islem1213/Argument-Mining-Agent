"""Exact-match F1 for JSONL graphs keyed by document id."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def load(path: Path) -> dict:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}

def f1(predicted: set, gold: set) -> float:
    precision = len(predicted & gold) / len(predicted) if predicted else 0.0
    recall = len(predicted & gold) / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def node_key(document: str, node: dict) -> tuple:
    """Prefer gold offsets; ids are a fallback for hand-authored demo graphs."""
    return (document, node.get("start", node["id"]), node.get("end", node["id"]), node["label"])

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--gold", required=True, type=Path); p.add_argument("--predictions", required=True, type=Path); a = p.parse_args()
    gold, pred = load(a.gold), load(a.predictions)
    gold_nodes = {node_key(d, n) for d, g in gold.items() for n in g["nodes"]}
    pred_nodes = {node_key(d, n) for d, g in pred.items() for n in g["nodes"]}
    gold_edges = {(d, e["source"], e["target"], e["type"]) for d, g in gold.items() for e in g["edges"]}
    pred_edges = {(d, e["source"], e["target"], e["type"]) for d, g in pred.items() for e in g["edges"]}
    print(json.dumps({"component_f1": round(f1(pred_nodes, gold_nodes), 4), "relation_f1": round(f1(pred_edges, gold_edges), 4)}, indent=2))
if __name__ == "__main__": main()
