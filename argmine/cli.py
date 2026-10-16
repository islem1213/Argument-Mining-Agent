"""Command-line interface: ``argmine`` (single document or JSONL batch)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .export import summary, write_graphml, write_html, write_json, write_mermaid, write_svg
from .pipeline import ArgumentMiner, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="argmine", description="Extract and analyse an argument graph from text.")
    p.add_argument("text", nargs="?", help="Text to analyse (or use --file / --batch / stdin)")
    p.add_argument("--file", type=Path, help="Read the document from a text file")
    p.add_argument("--batch", type=Path, help="JSONL with {id, text[, topic]} rows; writes JSONL to --output")
    p.add_argument("--output", type=Path, default=Path("argument_graph.json"))
    p.add_argument("--image", type=Path, help="Write a layered SVG argument map")
    p.add_argument("--html", type=Path, help="Write a self-contained interactive HTML report")
    p.add_argument("--mermaid", type=Path, help="Write a Mermaid flowchart (.mmd)")
    p.add_argument("--graphml", type=Path, help="Write GraphML for Gephi / yEd / NetworkX")
    p.add_argument("--topic", help="Topic or thesis the stance step is measured against")
    g = p.add_argument_group("backends")
    g.add_argument("--components", dest="component_backend", choices=("heuristic", "transformer", "llm"),
                   default="heuristic")
    g.add_argument("--classifier-model", help="Path or Hugging Face id of fine-tuned component weights")
    g.add_argument("--stance", dest="stance_backend", choices=("heuristic", "nli"), default="heuristic")
    g.add_argument("--relation-backend", choices=("heuristic", "local", "nli", "llm"), default="heuristic",
                   help="'local' is a v0.1 alias of 'heuristic'")
    g.add_argument("--relation-window", type=int, default=2)
    g.add_argument("--no-tree", action="store_true", help="Allow several outgoing relations per component")
    g.add_argument("--fallacies", choices=("heuristic", "llm"), help="Also flag possible informal fallacies")
    g.add_argument("--no-reasoning", action="store_true", help="Skip grounded / DF-QuAD analysis")
    p.add_argument("-q", "--quiet", action="store_true", help="Do not print the summary table")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    backend = "transformer" if args.classifier_model else args.component_backend
    return PipelineConfig(
        component_backend=backend, classifier_model=args.classifier_model, stance_backend=args.stance_backend,
        relation_backend="heuristic" if args.relation_backend == "local" else args.relation_backend,
        relation_window=args.relation_window, tree=not args.no_tree, reasoning=not args.no_reasoning,
        fallacies=args.fallacies,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    miner = ArgumentMiner(config_from_args(args))

    if args.batch:
        output = args.output if args.output.suffix == ".jsonl" else args.output.with_suffix(".jsonl")
        count = 0
        with args.batch.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    row = json.loads(line)
                    graph = miner.mine(row["text"], row.get("topic", args.topic), str(row.get("id", count)))
                    dst.write(json.dumps(graph.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
        print(f"Wrote {count} graphs to {output}")
        return

    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text:
        parser.error("provide text, --file, --batch or pipe text on stdin")
    graph = miner.mine(text, args.topic)
    write_json(graph, args.output)
    for path, writer in ((args.image, write_svg), (args.mermaid, write_mermaid), (args.graphml, write_graphml)):
        if path:
            writer(graph, path)
    if args.html:
        write_html(graph, args.html, text)
    if not args.quiet:
        print(summary(graph))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
