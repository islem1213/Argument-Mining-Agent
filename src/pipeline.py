from __future__ import annotations

import argparse
import json
from pathlib import Path

from .component_classifier.model import TransformerComponentClassifier, classify
from .graph_builder import build, write_graphml, write_svg
from .relation_extractor import extract
from .segmenter import segment
from .stance import estimate_stance

def run(text: str, relation_backend: str = "local", relation_window: int = 2, topic: str | None = None, classifier: TransformerComponentClassifier | None = None) -> dict:
    components = [{"id": f"c{i+1}", "text": item.text, "start": item.start, "end": item.end, "label": str(classify(item.text, i, classifier)), "stance": estimate_stance(item.text, topic)} for i, item in enumerate(segment(text))]
    relations = [item.model_dump() for item in extract(components, relation_backend, relation_window).relations]
    return build(components, relations)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an argument graph from text.")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("argument_graph.json"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--graphml", type=Path, help="Optional NetworkX/GraphML export")
    parser.add_argument("--topic", help="Topic/main claim used by the stance step")
    parser.add_argument("--relation-window", type=int, default=2)
    parser.add_argument("--classifier-model", help="Path or Hugging Face id for fine-tuned component weights")
    parser.add_argument("--relation-backend", choices=("local", "llm"), default="local")
    args = parser.parse_args()
    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    if not text: parser.error("provide text or --file")
    classifier = TransformerComponentClassifier(args.classifier_model) if args.classifier_model else None
    graph = run(text, args.relation_backend, args.relation_window, args.topic, classifier)
    args.output.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    if args.image: write_svg(graph, args.image)
    if args.graphml: write_graphml(graph, args.graphml)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
