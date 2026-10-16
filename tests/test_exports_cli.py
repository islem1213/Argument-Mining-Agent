import json
import xml.etree.ElementTree as ET

from argmine import ArgumentMiner
from argmine.cli import main
from argmine.export import render_html, render_svg, to_mermaid, write_graphml


def test_exports(sample, tmp_path):
    g = ArgumentMiner().mine(sample)
    ET.fromstring(render_svg(g))  # well-formed XML
    assert to_mermaid(g).startswith("flowchart BT")
    page = render_html(g, sample)
    assert "<mark" in page and "const DATA" in page
    write_graphml(g, tmp_path / "g.graphml")
    assert (tmp_path / "g.graphml").stat().st_size > 0


def test_cli_single_and_batch(sample, tmp_path, capsys):
    out = tmp_path / "g.json"
    main([sample, "--output", str(out), "--html", str(tmp_path / "g.html"), "-q"])
    assert json.loads(out.read_text())["nodes"]
    batch = tmp_path / "in.jsonl"
    batch.write_text(json.dumps({"id": "x", "text": sample}) + "\n")
    main(["--batch", str(batch), "--output", str(tmp_path / "out.jsonl")])
    assert json.loads((tmp_path / "out.jsonl").read_text())["id"] == "x"


def test_mcp_tools(sample):
    from argmine.mcp_server import argument_map, critique_argument, mine_arguments

    assert mine_arguments(sample)["nodes"]
    assert "Central claim" in critique_argument(sample)
    assert argument_map(sample).startswith("```mermaid")
