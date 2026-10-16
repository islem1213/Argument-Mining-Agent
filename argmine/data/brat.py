"""Convert brat-annotated corpora (e.g. AAEC / "Argument Annotated Essays v2",
Stab & Gurevych 2017) to argmine's JSONL formats.

The corpus is licensed separately and is *not* redistributed here. Download it
from TU Darmstadt, unzip ``brat-project-final`` and run::

    python -m argmine.data.brat data/raw/ArgumentAnnotatedEssays-2.0/brat-project-final \
        --split data/raw/ArgumentAnnotatedEssays-2.0/train-test-split.csv --out data/processed

Outputs, per split:

* ``graphs_<split>.jsonl``     — gold graphs ``{id, text, nodes, edges}`` for ``argmine.eval``
* ``components_<split>.jsonl`` — ``{text, label, context}`` rows for ``argmine.components.train``
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

LABEL_MAP = {"MajorClaim": "major_claim", "Claim": "claim", "Premise": "premise"}
RELATION_MAP = {"supports": "support", "attacks": "attack"}


def parse_ann(ann: str, text: str, stance_edges: bool = True) -> dict:
    nodes, edges, stances = {}, [], {}
    for line in ann.splitlines():
        if not line.strip():
            continue
        key, body = line.split("\t", 1)[0], line.split("\t")[1]
        if key.startswith("T"):
            label, start, end = body.split()[:3]
            if label in LABEL_MAP:
                start, end = int(start), int(end)
                nodes[key] = {"id": key, "text": text[start:end], "start": start, "end": end,
                              "label": LABEL_MAP[label]}
        elif key.startswith("R"):
            rel, arg1, arg2 = body.split()[:3]
            if rel in RELATION_MAP:
                edges.append({"source": arg1.split(":")[1], "target": arg2.split(":")[1],
                              "type": RELATION_MAP[rel]})
        elif key.startswith("A"):
            parts = body.split()
            if parts[0] == "Stance" and len(parts) >= 3:
                stances[parts[1]] = parts[2]
    # In AAEC, claim → major claim links are encoded as a Stance attribute, not a relation.
    majors = [n for n in nodes.values() if n["label"] == "major_claim"]
    for node_id, value in stances.items():
        if node_id in nodes:
            nodes[node_id]["stance"] = "pro" if value == "For" else "con"
            if stance_edges and majors:
                edges.append({"source": node_id, "target": majors[0]["id"],
                              "type": "support" if value == "For" else "attack"})
    ordered = sorted(nodes.values(), key=lambda n: n["start"])
    return {"nodes": ordered, "edges": edges}


def _non_argumentative_spans(text: str, nodes: list[dict], min_words: int = 4) -> list[dict]:
    """Sentences that contain no annotated component become non_argumentative examples."""
    spans = []
    for match in re.finditer(r"[^.!?\n]+[.!?]", text):
        start, end = match.start(), match.end()
        if not any(n["start"] < end and start < n["end"] for n in nodes) and len(match.group().split()) >= min_words:
            spans.append({"text": match.group().strip(), "start": start, "end": end})
    return spans


def convert(corpus: Path, split_file: Path | None = None) -> dict[str, list[dict]]:
    splits: dict[str, str] = {}
    if split_file:
        with split_file.open(encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter=";"):
                if len(row) >= 2 and row[0].strip('"') != "ID":
                    splits[row[0].strip('"')] = row[1].strip('"').lower()
    out: dict[str, list[dict]] = {}
    for ann_path in sorted(corpus.glob("*.ann")):
        text = ann_path.with_suffix(".txt").read_text(encoding="utf-8")
        graph = parse_ann(ann_path.read_text(encoding="utf-8"), text)
        out.setdefault(splits.get(ann_path.stem, "all"), []).append({"id": ann_path.stem, "text": text, **graph})
    return out


def component_rows(doc: dict) -> list[dict]:
    text, rows = doc["text"], []
    for node in doc["nodes"]:
        line_start = text.rfind("\n", 0, node["start"]) + 1
        line_end = text.find("\n", node["end"])
        rows.append({"text": node["text"], "label": node["label"],
                     "context": text[line_start: line_end if line_end != -1 else len(text)].strip()})
    rows += [{"text": s["text"], "label": "non_argumentative", "context": ""}
             for s in _non_argumentative_spans(text, doc["nodes"])]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", type=Path, help="Directory containing .ann/.txt pairs")
    parser.add_argument("--split", type=Path, help="train-test-split.csv from the AAEC release")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, docs in convert(args.corpus, args.split).items():
        with (args.out / f"graphs_{split}.jsonl").open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
        with (args.out / f"components_{split}.jsonl").open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for d in docs for r in component_rows(d))
        print(f"{split}: {len(docs)} essays")


if __name__ == "__main__":
    main()
