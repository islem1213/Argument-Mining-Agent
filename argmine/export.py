"""Exporters: JSON, GraphML, layered SVG, Mermaid, and a self-contained interactive HTML report."""
from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

import networkx as nx

from .schema import ArgumentGraph, ComponentLabel, RelationType

PALETTE = {  # (fill, stroke, text)
    ComponentLabel.MAJOR_CLAIM: ("#dbeafe", "#1d4ed8", "#1e3a8a"),
    ComponentLabel.CLAIM: ("#ede9fe", "#7c3aed", "#4c1d95"),
    ComponentLabel.PREMISE: ("#ccfbf1", "#0f766e", "#134e4a"),
    ComponentLabel.NON_ARGUMENTATIVE: ("#f3f4f6", "#9ca3af", "#4b5563"),
}
EDGE = {RelationType.SUPPORT: "#16a34a", RelationType.ATTACK: "#dc2626"}
BOX_W, BOX_H, GAP_X, GAP_Y, MARGIN = 230, 84, 28, 70, 30


def _coerce(graph: ArgumentGraph | dict) -> ArgumentGraph:
    return graph if isinstance(graph, ArgumentGraph) else ArgumentGraph.model_validate(graph)


def build(components: list[dict], relations: list[dict]) -> dict:
    """v0.1 helper kept for compatibility."""
    return {"nodes": components, "edges": relations}


def write_json(graph: ArgumentGraph | dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_coerce(graph).to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def to_networkx(graph: ArgumentGraph | dict) -> nx.DiGraph:
    g = _coerce(graph)
    result = nx.DiGraph()
    for node in g.nodes:
        attrs = node.model_dump(mode="json", exclude_none=True)
        if g.analysis and node.id in g.analysis.components:
            attrs.update(g.analysis.components[node.id].model_dump(mode="json"))
        result.add_node(node.id, **attrs)
    for edge in g.edges:
        result.add_edge(edge.source, edge.target, **edge.model_dump(mode="json", exclude_none=True))
    return result


def write_graphml(graph: ArgumentGraph | dict, path: str | Path) -> None:
    """Interoperable graph file for Gephi, yEd and NetworkX."""
    nx.write_graphml(to_networkx(graph), path)


# --------------------------------------------------------------------------- layout

def layout(graph: ArgumentGraph) -> dict[str, tuple[float, float]]:
    """Layered layout: major claims on top, each node one row below the node it targets."""
    out = {}
    for e in graph.edges:
        out.setdefault(e.source, e.target)
    default_level = {ComponentLabel.MAJOR_CLAIM: 0, ComponentLabel.CLAIM: 1, ComponentLabel.PREMISE: 2,
                     ComponentLabel.NON_ARGUMENTATIVE: 3}
    levels: dict[str, int] = {}

    def level(node_id: str, seen: frozenset = frozenset()) -> int:
        if node_id in levels:
            return levels[node_id]
        target = out.get(node_id)
        node = graph.node(node_id)
        if target is None or target in seen or node.label == ComponentLabel.MAJOR_CLAIM:
            value = default_level[node.label] if target is None or node.label == ComponentLabel.MAJOR_CLAIM else 1
        else:
            value = level(target, seen | {node_id}) + 1
        levels[node_id] = value
        return value

    for node in graph.nodes:
        level(node.id)
    rows: dict[int, list[str]] = {}
    for node in graph.nodes:
        rows.setdefault(levels[node.id], []).append(node.id)
    positions: dict[str, tuple[float, float]] = {}
    for depth in sorted(rows):
        ids = rows[depth]
        # Order children under their parent (barycentre), then by document order.
        ids.sort(key=lambda n: (positions.get(out.get(n, ""), (1e9, 0))[0], int(n[1:]) if n[1:].isdigit() else 0))
        cursor = MARGIN
        for node_id in ids:
            parent_x = positions.get(out.get(node_id, ""), (MARGIN, 0))[0]
            x = max(cursor, parent_x)  # sit under the parent when there is room
            positions[node_id] = (x, MARGIN + 30 + depth * (BOX_H + GAP_Y))
            cursor = x + BOX_W + GAP_X
    return positions


def render_svg(graph: ArgumentGraph | dict, interactive: bool = False) -> str:
    g = _coerce(graph)
    pos = layout(g)
    width = max((x for x, _ in pos.values()), default=0) + BOX_W + MARGIN
    height = max((y for _, y in pos.values()), default=0) + BOX_H + MARGIN + 30
    width = max(width, 560)
    status = g.analysis.components if g.analysis else {}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="Inter, Segoe UI, Arial, sans-serif" class="argmap">',
        '<defs>' + "".join(
            f'<marker id="arrow-{t.value}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for t, c in EDGE.items())
        + '</defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    legend_x = MARGIN
    for label, (fill, stroke, _) in PALETTE.items():
        parts.append(f'<rect x="{legend_x}" y="12" width="12" height="12" rx="3" fill="{fill}" stroke="{stroke}"/>'
                     f'<text x="{legend_x + 17}" y="22" font-size="11" fill="#374151">{label.value.replace("_", " ")}</text>')
        legend_x += 32 + 6.5 * len(label.value)
    parts.append(f'<line x1="{legend_x}" y1="18" x2="{legend_x + 22}" y2="18" stroke="{EDGE[RelationType.SUPPORT]}" '
                 f'stroke-width="2"/><text x="{legend_x + 27}" y="22" font-size="11" fill="#374151">support</text>'
                 f'<line x1="{legend_x + 85}" y1="18" x2="{legend_x + 107}" y2="18" stroke="{EDGE[RelationType.ATTACK]}" '
                 f'stroke-width="2" stroke-dasharray="5 3"/><text x="{legend_x + 112}" y="22" font-size="11" '
                 'fill="#374151">attack</text>')

    for e in g.edges:
        if e.source not in pos or e.target not in pos:
            continue
        (sx, sy), (tx, ty) = pos[e.source], pos[e.target]
        x1, x2 = sx + BOX_W / 2, tx + BOX_W / 2
        if sy > ty:  # source below target: vertical S-curve
            y1, y2 = sy, ty + BOX_H
            d = f"M{x1},{y1} C{x1},{(y1 + y2) / 2} {x2},{(y1 + y2) / 2} {x2},{y2 + 2}"
        else:  # same row or upward: arc above
            y1, y2 = sy, ty
            lift = 40 + abs(x1 - x2) / 6
            d = f"M{x1},{y1} C{x1},{y1 - lift} {x2},{y2 - lift} {x2},{y2 - 2}"
        dash = ' stroke-dasharray="6 4"' if e.type == RelationType.ATTACK else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{EDGE[e.type]}" stroke-width="{1.2 + 2 * e.confidence:.1f}"'
                     f'{dash} marker-end="url(#arrow-{e.type.value})" class="edge" data-source="{e.source}" '
                     f'data-target="{e.target}"><title>{escape(e.source)} {e.type.value} {escape(e.target)}'
                     f' ({e.confidence:.2f}){" — " + escape(e.rationale) if e.rationale else ""}</title></path>')

    for node in g.nodes:
        x, y = pos[node.id]
        fill, stroke, ink = PALETTE[node.label]
        info = status.get(node.id)
        opacity = "0.55" if info and info.grounded == "out" else "1"
        weight = "2.5" if info and info.grounded == "in" else "1.2"
        lines = textwrap.wrap(node.text, 36)[:3]
        if len(textwrap.wrap(node.text, 36)) > 3:
            lines[-1] = lines[-1][:33] + "…"
        head = f"{node.id} · {node.label.value.replace('_', ' ')} · {node.stance.value}"
        parts.append(f'<g class="node" data-id="{node.id}" opacity="{opacity}"><title>{escape(node.text)}</title>'
                     f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="{weight}"/>'
                     f'<text x="{x + 10}" y="{y + 17}" font-size="10.5" font-weight="600" fill="{ink}">{escape(head)}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x + 10}" y="{y + 35 + i * 14}" font-size="11.5" fill="#111827">{escape(line)}</text>')
        if info:
            bar = (BOX_W - 20) * info.strength
            parts.append(f'<rect x="{x + 10}" y="{y + BOX_H - 9}" width="{BOX_W - 20}" height="3" rx="1.5" fill="#e5e7eb"/>'
                         f'<rect x="{x + 10}" y="{y + BOX_H - 9}" width="{bar:.1f}" height="3" rx="1.5" fill="{stroke}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def write_svg(graph: ArgumentGraph | dict, path: str | Path) -> None:
    Path(path).write_text(render_svg(graph), encoding="utf-8")


# --------------------------------------------------------------------------- Mermaid

def to_mermaid(graph: ArgumentGraph | dict) -> str:
    """Mermaid flowchart — renders natively in GitHub, GitLab, Notion and Obsidian."""
    g = _coerce(graph)
    lines = ["flowchart BT"]
    for node in g.nodes:
        text = textwrap.shorten(node.text, 70, placeholder="…").replace('"', "#quot;")
        lines.append(f'    {node.id}["<b>{node.id} · {node.label.value.replace("_", " ")}</b><br/>{text}"]:::{node.label.value}')
    for e in g.edges:
        arrow = "-->" if e.type == RelationType.SUPPORT else "-.->"
        lines.append(f"    {e.source} {arrow}|{e.type.value}| {e.target}")
    for label, (fill, stroke, ink) in PALETTE.items():
        lines.append(f"    classDef {label.value} fill:{fill},stroke:{stroke},color:{ink}")
    for i, e in enumerate(g.edges):
        lines.append(f"    linkStyle {i} stroke:{EDGE[e.type]}")
    return "\n".join(lines) + "\n"


def write_mermaid(graph: ArgumentGraph | dict, path: str | Path) -> None:
    Path(path).write_text(to_mermaid(graph), encoding="utf-8")


# --------------------------------------------------------------------------- HTML

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Argument map</title>
<style>
:root{{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#f9fafb}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:var(--bg)}}
header{{padding:18px 24px;border-bottom:1px solid var(--line);background:#fff}}h1{{font-size:18px;margin:0}}
header p{{margin:4px 0 0;color:var(--muted)}}main{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;overflow:auto}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}}.stat{{background:var(--bg);border-radius:8px;padding:6px 10px}}
.stat b{{display:block;font-size:16px}}.source{{white-space:pre-wrap}}mark{{border-radius:4px;padding:0 2px;cursor:pointer}}
mark.major_claim{{background:#dbeafe}}mark.claim{{background:#ede9fe}}mark.premise{{background:#ccfbf1}}
mark.non_argumentative{{background:#f3f4f6}}mark.active{{outline:2px solid #f59e0b}}
.node.active rect:first-of-type{{stroke:#f59e0b;stroke-width:3}}
.node{{cursor:pointer}}.edge{{transition:opacity .15s}}.dim{{opacity:.15}}
#detail h3{{margin:.2em 0;font-size:15px}}.pill{{display:inline-block;border-radius:99px;padding:1px 8px;margin:2px;background:var(--bg);font-size:12px}}
.warn{{color:#b45309}}ul{{padding-left:18px}}h2{{font-size:14px;margin:12px 0 6px}}
@media (max-width:900px){{main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Argument map</h1><p>{subtitle}</p></header>
<main><section class="card">{svg}</section>
<aside class="card"><div class="stats">{stats}</div><div id="detail"><p class="muted">Hover or click a component.</p></div>
<h2>Diagnostics</h2>{diagnostics}{source}</aside></main>
<script>
const DATA = {data};
const byId = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));
const analysis = (DATA.analysis || {{}}).components || {{}};
function esc(s){{return String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);}}
function focus(id){{
  document.querySelectorAll('.node,mark').forEach(el => el.classList.toggle('active', el.dataset.id === id));
  document.querySelectorAll('.edge').forEach(el => el.classList.toggle('dim', id && el.dataset.source !== id && el.dataset.target !== id));
  const n = byId[id]; if (!n) return; const a = analysis[id];
  const rel = DATA.edges.filter(e => e.source === id || e.target === id)
    .map(e => `<li>${{e.source}} <b>${{e.type}}</b> ${{e.target}} <span class="pill">${{e.confidence.toFixed(2)}}</span>${{e.rationale ? ' — ' + esc(e.rationale) : ''}}</li>`).join('');
  const fall = ((DATA.analysis||{{}}).fallacies||[]).filter(f => f.component === id)
    .map(f => `<li class="warn"><b>${{f.fallacy.replaceAll('_',' ')}}</b>: ${{esc(f.explanation)}}</li>`).join('');
  document.getElementById('detail').innerHTML = `<h3>${{n.id}} · ${{n.label.replaceAll('_',' ')}}</h3><p>${{esc(n.text)}}</p>
    <span class="pill">stance: ${{n.stance}}</span>${{n.marker ? `<span class="pill">marker: ${{esc(n.marker)}}</span>` : ''}}
    ${{a ? `<span class="pill">grounded: ${{a.grounded}}</span><span class="pill">strength: ${{a.strength.toFixed(2)}}</span>` : ''}}
    ${{rel ? `<ul>${{rel}}</ul>` : ''}}${{fall ? `<ul>${{fall}}</ul>` : ''}}`;
}}
document.querySelectorAll('.node,mark').forEach(el => {{
  el.addEventListener('mouseenter', () => focus(el.dataset.id));
  el.addEventListener('click', () => focus(el.dataset.id));
}});
</script></body></html>
"""


def _highlight(text: str, graph: ArgumentGraph) -> str:
    pieces, cursor = [], 0
    for node in sorted(graph.nodes, key=lambda n: n.start):
        if node.start < cursor or text[node.start:node.end] != node.text:
            continue
        pieces += [html.escape(text[cursor:node.start]),
                   f'<mark class="{node.label.value}" data-id="{node.id}" title="{node.id}">'
                   f'{html.escape(node.text)}</mark>']
        cursor = node.end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def render_html(graph: ArgumentGraph | dict, text: str | None = None) -> str:
    g = _coerce(graph)
    counts = {label: sum(n.label == label for n in g.nodes) for label in ComponentLabel}
    stats = "".join(f'<div class="stat"><b>{v}</b>{k.value.replace("_", " ")}</div>' for k, v in counts.items() if v)
    stats += f'<div class="stat"><b>{len(g.edges)}</b>relations</div>'
    diag = []
    if g.analysis:
        a = g.analysis
        if a.central_claim:
            info = a.components.get(a.central_claim)
            diag.append(f"<li>Central claim <b>{a.central_claim}</b>"
                        + (f" is <b>{info.grounded}</b> under grounded semantics (strength {info.strength:.2f})"
                           if info else "") + ".</li>")
        if a.unsupported_claims:
            diag.append(f'<li class="warn">Unsupported claims: {", ".join(a.unsupported_claims)}.</li>')
        if a.isolated_components:
            diag.append(f'<li class="warn">Isolated components: {", ".join(a.isolated_components)}.</li>')
        for f in a.fallacies:
            diag.append(f'<li class="warn">{f.component}: possible {f.fallacy.value.replace("_", " ")} '
                        f'({f.confidence:.2f})</li>')
    diagnostics = f"<ul>{''.join(diag)}</ul>" if diag else "<p>No issues detected.</p>"
    source = f'<h2>Source text</h2><div class="source">{_highlight(text, g)}</div>' if text else ""
    subtitle = html.escape(g.topic or "Components, relations and their acceptability")
    data = json.dumps(g.to_dict(), ensure_ascii=False).replace("</", "<\\/")
    return _HTML.format(subtitle=subtitle, svg=render_svg(g), stats=stats, diagnostics=diagnostics,
                        source=source, data=data)


def write_html(graph: ArgumentGraph | dict, path: str | Path, text: str | None = None) -> None:
    Path(path).write_text(render_html(graph, text), encoding="utf-8")


# --------------------------------------------------------------------------- terminal

def summary(graph: ArgumentGraph | dict) -> str:
    g = _coerce(graph)
    rows = []
    for n in g.nodes:
        info = g.analysis.components.get(n.id) if g.analysis else None
        extra = f" [{info.grounded}, {info.strength:.2f}]" if info else ""
        rows.append(f"{n.id:>4} {n.label.value:<17} {n.stance.value:<7}{extra} {textwrap.shorten(n.text, 70)}")
    rows += [f"     {e.source} --{e.type.value}--> {e.target} ({e.confidence:.2f})" for e in g.edges]
    if g.analysis and g.analysis.fallacies:
        rows += [f"  !  {f.component}: {f.fallacy.value} — {f.explanation}" for f in g.analysis.fallacies]
    return "\n".join(rows)
