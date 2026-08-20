from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import networkx as nx

def build(components: list[dict], relations: list[dict]) -> dict:
    return {"nodes": components, "edges": relations}

def to_networkx(graph: dict) -> nx.DiGraph:
    """Convert the portable JSON graph to a NetworkX directed graph."""
    result = nx.DiGraph()
    result.add_nodes_from((node["id"], node) for node in graph["nodes"])
    result.add_edges_from((edge["source"], edge["target"], edge) for edge in graph["edges"])
    return result

def write_graphml(graph: dict, path: str | Path) -> None:
    """Write an interoperable graph file for Gephi, yEd, and NetworkX users."""
    network = to_networkx(graph)
    for _, attrs in network.nodes(data=True):
        attrs["text"] = str(attrs.get("text", ""))
    nx.write_graphml(network, path)

def write_svg(graph: dict, path: str | Path) -> None:
    """Create a dependency-free, portable preview of an argument graph."""
    nodes = graph["nodes"]
    height = max(180, 85 * len(nodes) + 60)
    rows = []
    for i, node in enumerate(nodes):
        y = 35 + i * 85
        label = escape(node["text"][:72])
        rows.append(f'<rect x="35" y="{y}" width="550" height="52" rx="8" fill="#eff6ff" stroke="#2563eb"/><text x="50" y="{y+22}" font-family="Arial" font-size="12" fill="#1e3a8a">{escape(node["id"])} · {escape(node["label"])}</text><text x="50" y="{y+40}" font-family="Arial" font-size="12">{label}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{height}" viewBox="0 0 620 {height}"><rect width="100%" height="100%" fill="white"/>{"".join(rows)}</svg>'
    Path(path).write_text(svg, encoding="utf-8")
