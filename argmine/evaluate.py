"""Span-aware evaluation of predicted argument graphs against gold graphs.

v0.1 compared relations by node *id*, which only works when predicted and gold
ids coincide. Here predicted components are first aligned to gold components by
character overlap (the "approximate match" convention of the argument-mining
literature: overlap ≥ 50% of the longer span; use ``--match exact`` for 100%),
and relations are compared through that alignment.

Reports: span F1 (labels ignored), per-label and macro component F1, typed
relation F1 and untyped link F1.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LABELS = ("major_claim", "claim", "premise")


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {str(row["id"]): row for row in rows}


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}


def _overlap(a: dict, b: dict) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / max(a["end"] - a["start"], b["end"] - b["start"], 1)


def align(pred_nodes: list[dict], gold_nodes: list[dict], threshold: float = 0.5) -> dict[str, str]:
    """Greedy one-to-one alignment, best overlap first. Returns {pred_id: gold_id}."""
    candidates = sorted(((_overlap(p, g), p["id"], g["id"]) for p in pred_nodes for g in gold_nodes
                         if _overlap(p, g) >= threshold), reverse=True)
    mapping, used = {}, set()
    for _, pid, gid in candidates:
        if pid not in mapping and gid not in used:
            mapping[pid] = gid
            used.add(gid)
    return mapping


def evaluate(gold: dict[str, dict], pred: dict[str, dict], threshold: float = 0.5) -> dict:
    span = Counter()
    labels = {label: Counter() for label in LABELS}
    rel, link = Counter(), Counter()
    for doc_id, g in gold.items():
        p = pred.get(doc_id, {"nodes": [], "edges": []})
        pnodes = [n for n in p["nodes"] if n["label"] != "non_argumentative"]
        gnodes = [n for n in g["nodes"] if n["label"] != "non_argumentative"]
        mapping = align(pnodes, gnodes, threshold)
        glabel = {n["id"]: n["label"] for n in gnodes}
        span["tp"] += len(mapping)
        span["fp"] += len(pnodes) - len(mapping)
        span["fn"] += len(gnodes) - len(mapping)
        matched_gold = set(mapping.values())
        for n in pnodes:
            gid = mapping.get(n["id"])
            if gid is not None and glabel[gid] == n["label"]:
                labels[n["label"]]["tp"] += 1
            else:
                if n["label"] in labels:
                    labels[n["label"]]["fp"] += 1
                if gid is not None and glabel[gid] in labels:
                    labels[glabel[gid]]["fn"] += 1
        for n in gnodes:
            if n["id"] not in matched_gold and n["label"] in labels:
                labels[n["label"]]["fn"] += 1

        gold_rel = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
        pred_rel = {(mapping.get(e["source"], "?" + e["source"]), mapping.get(e["target"], "?" + e["target"]), e["type"])
                    for e in p["edges"]}
        for gold_set, pred_set, counter in ((gold_rel, pred_rel, rel),
                                            ({x[:2] for x in gold_rel}, {x[:2] for x in pred_rel}, link)):
            counter["tp"] += len(gold_set & pred_set)
            counter["fp"] += len(pred_set - gold_set)
            counter["fn"] += len(gold_set - pred_set)

    per_label = {label: prf(c["tp"], c["fp"], c["fn"]) for label, c in labels.items()}
    present = [v["f1"] for k, v in per_label.items() if sum(labels[k].values())]
    return {
        "documents": len(gold),
        "match_threshold": threshold,
        "span": prf(span["tp"], span["fp"], span["fn"]),
        "components": per_label,
        "component_macro_f1": round(sum(present) / len(present), 4) if present else 0.0,
        "relation": prf(rel["tp"], rel["fp"], rel["fn"]),
        "link": prf(link["tp"], link["fp"], link["fn"]),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate argument graphs (JSONL keyed by `id`).")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", type=Path, help="Predicted graphs JSONL")
    parser.add_argument("--run", action="store_true", help="Run argmine on the gold texts instead of --predictions")
    parser.add_argument("--relation-backend", default="heuristic")
    parser.add_argument("--component-backend", default="heuristic")
    parser.add_argument("--classifier-model")
    parser.add_argument("--match", choices=("approximate", "exact"), default="approximate")
    parser.add_argument("--save-predictions", type=Path)
    args = parser.parse_args(argv)
    gold = load_jsonl(args.gold)
    if args.run:
        from .pipeline import ArgumentMiner, PipelineConfig

        miner = ArgumentMiner(PipelineConfig(
            component_backend="transformer" if args.classifier_model else args.component_backend,
            classifier_model=args.classifier_model, relation_backend=args.relation_backend, reasoning=False))
        pred = {doc_id: miner.mine(doc["text"], doc_id=doc_id).to_dict() for doc_id, doc in gold.items()}
        if args.save_predictions:
            args.save_predictions.write_text("".join(json.dumps(p) + "\n" for p in pred.values()), encoding="utf-8")
    elif args.predictions:
        pred = load_jsonl(args.predictions)
    else:
        parser.error("give --predictions or --run")
    print(json.dumps(evaluate(gold, pred, 1.0 if args.match == "exact" else 0.5), indent=2))


if __name__ == "__main__":
    main()
