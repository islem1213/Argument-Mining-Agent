"""Model Context Protocol server: expose argument mining as tools for any MCP client
(Claude Desktop, Claude Code, Cursor, VS Code, LangGraph or your own agent).

    pip install -e '.[mcp]'
    argmine-mcp                      # stdio transport
    argmine-mcp --http --port 8765   # streamable HTTP transport

Backends are taken from ``ARGMINE_RELATION_BACKEND`` / ``ARGMINE_COMPONENT_BACKEND``
(defaults: heuristic), so the server works fully offline out of the box.
"""
from __future__ import annotations

import argparse
import os

from .export import to_mermaid
from .pipeline import ArgumentMiner, PipelineConfig

INSTRUCTIONS = (
    "Tools for argument mining. `mine_arguments` returns claims, premises, support/attack relations and "
    "an acceptability analysis. `critique_argument` returns a short feedback report. `argument_map` returns a "
    "Mermaid diagram that most chat clients render."
)


def _miner(fallacies: str | None = None) -> ArgumentMiner:
    return ArgumentMiner(PipelineConfig(
        component_backend=os.environ.get("ARGMINE_COMPONENT_BACKEND", "heuristic"),
        relation_backend=os.environ.get("ARGMINE_RELATION_BACKEND", "heuristic"),
        stance_backend=os.environ.get("ARGMINE_STANCE_BACKEND", "heuristic"),
        fallacies=fallacies,
    ))


def mine_arguments(text: str, topic: str | None = None) -> dict:
    """Extract the argument graph of `text`: components (major_claim / claim / premise), stance,
    support/attack edges, grounded acceptability and DF-QuAD strength for every component."""
    return _miner().mine(text, topic).to_dict()


def critique_argument(text: str, topic: str | None = None) -> str:
    """Give structured feedback on an argument: central claim, unsupported claims, defeated claims,
    isolated statements and possible informal fallacies."""
    fallacy_backend = "llm" if os.environ.get("ARGMINE_COMPONENT_BACKEND") == "llm" else "heuristic"
    graph = _miner(fallacy_backend).mine(text, topic)
    a, node = graph.analysis, graph.node
    lines = []
    if a.central_claim:
        info = a.components[a.central_claim]
        lines.append(f"Central claim ({a.central_claim}): \"{node(a.central_claim).text}\" — "
                     f"{'accepted' if info.grounded == 'in' else 'not accepted'} under grounded semantics, "
                     f"strength {info.strength:.2f}, {info.supporters} supporter(s), {info.attackers} attacker(s).")
    defeated = [n for n, i in a.components.items() if i.grounded == "out"]
    for title, ids in (("Claims without support", a.unsupported_claims), ("Defeated components", defeated),
                       ("Isolated components", a.isolated_components)):
        if ids:
            lines.append(f"{title}: " + "; ".join(f"{i} \"{node(i).text}\"" for i in ids))
    lines += [f"Possible {f.fallacy.value.replace('_', ' ')} in {f.component}: {f.explanation}" for f in a.fallacies]
    return "\n".join(lines) or "No structural issues found."


def argument_map(text: str, topic: str | None = None) -> str:
    """Return a Mermaid flowchart of the argument structure of `text`."""
    return "```mermaid\n" + to_mermaid(_miner().mine(text, topic)) + "```"


def create_server():
    try:
        from mcp.server import MCPServer as Server  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise SystemExit("Install MCP support first: pip install -e '.[mcp]'") from exc
    server = Server("argmine", instructions=INSTRUCTIONS)
    for tool in (mine_arguments, critique_argument, argument_map):
        server.tool()(tool)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="argmine-mcp")
    parser.add_argument("--http", action="store_true", help="Serve over streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server()
    if args.http:
        try:  # mcp >= 2 takes transport options as keyword arguments
            server.run("streamable-http", host=args.host, port=args.port)
        except TypeError:  # mcp 1.x reads them from settings
            server.settings.host, server.settings.port = args.host, args.port
            server.run("streamable-http")
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
