# Copyright 2026 Arkhip A. Dmitriev
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Drawings of ZX diagrams: one layered layout, rendered with matplotlib, Unicode or ASCII.

:func:`layout` places a :class:`~archytaszx.repl.zxview.View`'s nodes in columns by their
distance from the outputs (outputs rightmost), splits wires that skip columns with dummy
points, and orders each column by barycentre sweeps to reduce crossings. A diagram above
:data:`MAX_NODES` nodes is summarised: each summed term becomes one box and chains of
splitters or binders one node.

:func:`grid` draws a layout on a character grid, crossings as ``┼`` (``+`` in ASCII);
:func:`figure` draws it with matplotlib; :func:`draw` picks a renderer with fallbacks.
"""

from __future__ import annotations

import io
import math
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram, Node, NodeId, PortRef
from archytaszx.repl import display as D
from archytaszx.repl.printer import rule_turns
from archytaszx.repl.zxview import View

MAX_NODES = 150
"""The most nodes drawn one by one; above it terms and splitter chains are summarised."""

STYLES = ("matplot", "unicode", "ascii")

_SWEEPS = 8


@dataclass
class Item:
    """A drawn vertex: a node, a summarised term, a boundary end or a wire's dummy point."""

    key: str
    kind: str
    label: str
    col: int = 0
    row: int = 0


@dataclass
class Edge:
    """A wire as the item keys it passes, dummies included, and its dimension."""

    path: list[str]
    dim: str


@dataclass
class Frame:
    """A bang box: the items it holds and its multiplicity."""

    keys: list[str]
    label: str


@dataclass
class Layout:
    """Items placed in columns and rows, wires, bang-box frames, the scalar and phase legend."""

    items: dict[str, Item]
    edges: list[Edge]
    frames: list[Frame]
    scalar: str
    legend: list[str] = field(default_factory=list)
    summary: bool = False

    @property
    def columns(self) -> int:
        return 1 + max((i.col for i in self.items.values()), default=0)


def _scalar(e: sympy.Expr) -> str:
    """``e`` as text, a long number as an approximation."""
    text = _plain(e)
    if len(text) <= 40 or e.free_symbols:
        return text
    return f"≈ {sympy.N(e, 6)}"


def _plain(e: sympy.Expr) -> str:
    return sympy.sstr(sympy.nsimplify(e) if e.has(sympy.Float) else e).replace("**", "^")


class _Labels:
    """Short node labels; phase vectors too long for a label go to a legend."""

    def __init__(self) -> None:
        self.legend: list[str] = []
        self._seen: dict[str, str] = {}

    def phase(self, node: Node) -> str:
        pv = node.phase
        if pv is None or pv.is_zero:
            return ""
        if pv.rule is not None and not pv.entries():
            text = _plain(rule_turns(pv))
            if len(text) <= 14:
                return text
            if text not in self._seen:
                name = f"α{len(self._seen) + 1}"
                self._seen[text] = name
                self.legend.append(f"{name} = {text} turns at index k")
            return self._seen[text]
        value = _basis_value(node)
        if value is not None:
            return str(value)
        entries = pv.entries()
        if pv.dim.is_concrete and pv.dim.to_int() == 2:
            return _plain(pv.get(1).to_sympy_turns())
        if pv.dim.is_concrete:
            r = pv.dim.to_int()
            c = pv.get(1).to_sympy_turns()
            if all(_mod1(pv.get(k).to_sympy_turns() - k * c) for k in range(1, r)):
                return f"{_plain(c)}·k"
        text = ", ".join(f"{k}: {_plain(p.to_sympy_turns())}" for k, p in sorted(entries.items()))
        if len(text) <= 14:
            return "{" + text + "}"
        if text not in self._seen:
            name = f"α{len(self._seen) + 1}"
            self._seen[text] = name
            self.legend.append(f"{name} = {{{text}}} turns")
        return self._seen[text]


def _mod1(e: sympy.Expr) -> bool:
    e = sympy.nsimplify(e) if e.has(sympy.Float) else e
    return bool(e.is_Rational and e.q == 1) or e == 0


def _basis_value(node: Node) -> int | None:
    """v when ``node`` is the X spider ``|v>``."""
    pv = node.phase
    if node.generator_type.name != "X" or node.inputs or pv is None or not pv.dim.is_concrete:
        return None
    r = pv.dim.to_int()
    c = -pv.get(1).to_sympy_turns() * r
    if not (c.is_Integer and all(_mod1(pv.get(k).to_sympy_turns() + k * c / r) for k in range(1, r))):
        return None
    return int(c) % r


def _node_label(node: Node, labels: _Labels) -> tuple[str, str]:
    name = node.generator_type.name
    if name in ("Z", "X"):
        return name, labels.phase(node)
    return name, ""


def _dim_text(node: Node, ref: PortRef) -> str:
    port = node.legs(ref.direction)[ref.index]
    return str(port.dim)


def _items(view: View, max_nodes: int) -> tuple[dict[str, Item], list[Edge], list[Frame], bool, list[str]]:
    g = view.diagram
    summary = len(g.nodes) > max_nodes and bool(view.groups)
    labels = _Labels()
    owner: dict[NodeId, str] = {}
    items: dict[str, Item] = {}
    for nid, node in sorted(g.nodes.items()):
        if summary and nid in view.groups:
            t = view.groups[nid]
            key = f"g{t}"
            if key not in items:
                items[key] = Item(key, "group", f"term {t + 1}: {view.labels.get(t, '')}")
            owner[nid] = key
            continue
        kind, text = _node_label(node, labels)
        key = f"n{int(nid)}"
        items[key] = Item(key, kind, text)
        owner[nid] = key
    if summary:
        _merge_chains(g, owner, items)
    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for wire in sorted(g.wires, key=lambda w: w.sort_key()):
        a, b = owner[wire.a.node_id], owner[wire.b.node_id]
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        if items[a].kind == "group" or items[b].kind == "group":
            if pair in seen:
                continue
            seen.add(pair)
        edges.append(Edge([a, b], _dim_text(g.nodes[wire.a.node_id], wire.a)))
    for k, ref in enumerate(g.boundary_outputs):
        key = f"o{k}"
        name = view.outputs[k] if k < len(view.outputs) else f"out {k}"
        items[key] = Item(key, "out", name)
        edges.append(Edge([owner[ref.node_id], key], _dim_text(g.nodes[ref.node_id], ref)))
    for k, ref in enumerate(g.boundary_inputs):
        key = f"i{k}"
        items[key] = Item(key, "in", f"in {k}")
        edges.append(Edge([key, owner[ref.node_id]], _dim_text(g.nodes[ref.node_id], ref)))
    terminal = {ref: f"o{k}" for k, ref in enumerate(g.boundary_outputs)}
    terminal.update({ref: f"i{k}" for k, ref in enumerate(g.boundary_inputs)})
    frames = []
    for _, box in sorted(g.bang_boxes.items()):
        if box.node_scope:
            keys = {owner[n] for n in box.node_scope}
        else:
            keys = {terminal.get(ref, owner[ref.node_id]) for ref in box.port_scope}
        frames.append(Frame(sorted(keys), str(box.multiplicity)))
    return items, edges, frames, summary, labels.legend + list(view.notes)


def _merge_chains(g: Diagram, owner: dict[NodeId, str], items: dict[str, Item]) -> None:
    """Chains of splitters (or binders) wired output to input merged into their first node."""
    parent: dict[str, str] = {}

    def find(k: str) -> str:
        while parent.get(k, k) != k:
            k = parent[k]
        return k

    for wire in sorted(g.wires, key=lambda w: w.sort_key()):
        a, b = owner[wire.a.node_id], owner[wire.b.node_id]
        ka, kb = items[a].kind, items[b].kind
        if ka == kb and ka in ("S", "B") and a != b:
            ra, rb = find(a), find(b)
            if ra != rb:
                first, second = sorted((ra, rb), key=lambda k: int(k[1:]))
                parent[second] = first
    for nid, key in list(owner.items()):
        root = find(key)
        if root != key:
            owner[nid] = root
            items.pop(key, None)


def layout(view: View, max_nodes: int = MAX_NODES) -> Layout:
    """The layered layout of ``view``; deterministic."""
    items, edges, frames, summary, legend = _items(view, max_nodes)
    adj: dict[str, list[str]] = {k: [] for k in items}
    for e in edges:
        a, b = e.path[0], e.path[-1]
        adj[a].append(b)
        adj[b].append(a)
    order = sorted(items, key=_key_order)
    dist: dict[str, int] = {}
    frontier = [k for k in order if items[k].kind == "out"]
    for k in frontier:
        dist[k] = 0
    while True:
        while frontier:
            nxt: list[str] = []
            for k in frontier:
                for m in adj[k]:
                    if m not in dist:
                        dist[m] = dist[k] + 1
                        nxt.append(m)
            frontier = nxt
        rest = [k for k in order if k not in dist]
        if not rest:
            break
        dist[rest[0]] = 1
        frontier = [rest[0]]
    top = max(dist.values(), default=0)
    for k, it in items.items():
        it.col = top - dist[k]
    inputs = [k for k in items if items[k].kind == "in"]
    for k in inputs:
        items[k].col = 0
    dummy = 0
    routed: list[Edge] = []
    for e in edges:
        a, b = e.path[0], e.path[-1]
        if items[a].col > items[b].col:
            a, b = b, a
        path = [a]
        for c in range(items[a].col + 1, items[b].col):
            key = f"d{dummy}"
            dummy += 1
            items[key] = Item(key, "dummy", "", c)
            path.append(key)
        path.append(b)
        routed.append(Edge(path, e.dim))
    _order(items, routed)
    return Layout(items, routed, frames, _scalar(view.diagram.scalar.to_sympy()), legend, summary)


def _key_order(k: str) -> tuple[int, int]:
    rank = {"o": 0, "i": 1, "g": 2, "n": 3, "d": 4}
    return rank[k[0]], int(k[1:])


def _order(items: dict[str, Item], edges: Sequence[Edge]) -> None:
    """Rows within each column by barycentre sweeps."""
    cols: dict[int, list[str]] = {}
    for k in sorted(items, key=_key_order):
        cols.setdefault(items[k].col, []).append(k)
    left: dict[str, list[str]] = {k: [] for k in items}
    right: dict[str, list[str]] = {k: [] for k in items}
    for e in edges:
        for a, b in zip(e.path, e.path[1:]):
            if items[a].col < items[b].col:
                right[a].append(b)
                left[b].append(a)
            elif items[a].col > items[b].col:
                right[b].append(a)
                left[a].append(b)
    pos = {k: i for col in cols.values() for i, k in enumerate(col)}
    span = sorted(cols)
    for sweep in range(_SWEEPS):
        seq = span[::-1] if sweep % 2 == 0 else span
        for c in seq:
            nbrs = right if sweep % 2 == 0 else left
            col = cols[c]
            def bary(k: str) -> tuple[float, int]:
                ns = nbrs[k]
                return (sum(pos[m] for m in ns) / len(ns) if ns else pos[k], pos[k])
            col.sort(key=bary)
            for i, k in enumerate(col):
                pos[k] = i
    for col in cols.values():
        for i, k in enumerate(col):
            items[k].row = i


def _routes(lay: Layout) -> tuple[dict[int, int], list[tuple[Edge, list[tuple[str, str, int]]]]]:
    """Per column the number of vertical tracks in the channel to its right, and per edge its
    hops ``(left, right, track)``, track -1 for a straight hop; hops fanning out of one item
    or into one item share a track."""
    ncol = lay.columns
    tracks: dict[int, int] = {c: 0 for c in range(ncol)}
    fan_out: dict[str, int] = {}
    fan_in: dict[str, int] = {}
    for e in lay.edges:
        for a, b in zip(e.path, e.path[1:]):
            if lay.items[a].col > lay.items[b].col:
                a, b = b, a
            fan_out[a] = fan_out.get(a, 0) + 1
            fan_in[b] = fan_in.get(b, 0) + 1
    shared: dict[tuple[str, str], int] = {}
    plan: list[tuple[Edge, list[tuple[str, str, int]]]] = []
    for e in lay.edges:
        hops = []
        for a, b in zip(e.path, e.path[1:]):
            ia, ib = lay.items[a], lay.items[b]
            if ia.col > ib.col:
                a, b, ia, ib = b, a, ib, ia
            c = ia.col
            if ia.row == ib.row and ia.col != ib.col:
                hops.append((a, b, -1))
                continue
            if ia.col == ib.col:
                key = ("same", a + "|" + b)
            else:
                key = ("out", a) if fan_out[a] >= fan_in[b] else ("in", b)
            if key not in shared:
                shared[key] = tracks[c]
                tracks[c] += 1
            hops.append((a, b, shared[key]))
        plan.append((e, hops))
    return tracks, plan


def _dim_label(text: str) -> str:
    """A wire dimension, a large integer as a product of prime powers."""
    if not text.isdigit() or len(text) <= 4:
        return text
    parts = sympy.factorint(int(text))
    return "·".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(parts.items()))


# -- character grid --------------------------------------------------------------------------

_L, _R, _U, _D = 1, 2, 4, 8
_BOX = {
    _L | _R: "─", _U | _D: "│", _R | _D: "┌", _L | _D: "┐", _R | _U: "└", _L | _U: "┘",
    _L | _R | _D: "┬", _L | _R | _U: "┴", _U | _D | _R: "├", _U | _D | _L: "┤",
    _L | _R | _U | _D: "┼", _L: "─", _R: "─", _U: "│", _D: "│",
}
_GLYPHS = {
    True: {"Z": "Ⓩ", "X": "Ⓧ", "W": "▷W", "F": "[F]", "T": "[T]", "Ti": "[T⁻¹]", "S": "◁S",
           "B": "▷B"},
    False: {"Z": "(", "X": "(", "W": ">W", "F": "[F]", "T": "[T]", "Ti": "[T^-1]", "S": "<S",
            "B": ">B"},
}


def _cell(it: Item, unicode: bool) -> str:
    if it.kind in ("out", "in", "dummy"):
        return it.label
    if it.kind == "group":
        return f"[{it.label}]" if unicode else f"[{_ascii(it.label)}]"
    glyph = _GLYPHS[unicode].get(it.kind, it.kind)
    if it.kind in ("Z", "X"):
        if unicode:
            return glyph + it.label
        return f"({it.kind}{' ' + _ascii(it.label) if it.label else ''})"
    return glyph


def _ascii(text: str) -> str:
    return text.replace("·", "*").replace("⟩", ">").replace("…", "...").replace("α", "a")


def grid(lay: Layout, unicode: bool = True) -> str:
    """The layout drawn on a character grid."""
    cells = {k: _cell(it, unicode) for k, it in lay.items.items()}
    ncol = lay.columns
    widths = [0] * ncol
    for k, it in lay.items.items():
        widths[it.col] = max(widths[it.col], len(cells[k]), 1)
    for fr in lay.frames:
        cols = sorted({lay.items[k].col for k in fr.keys})
        if len(cols) == 1:
            widths[cols[0]] = max(widths[cols[0]], len(fr.label) + 6)
    tracks, plan = _routes(lay)
    xs = []
    x = 2
    for c in range(ncol):
        xs.append(x)
        x += widths[c] + 3 + tracks[c]
    y0 = 2
    step = 3 if len(lay.frames) > 1 else 2

    def ypos(it: Item) -> int:
        return y0 + step * it.row
    lines: dict[tuple[int, int], int] = {}

    def mark(px: int, py: int, bits: int) -> None:
        lines[(px, py)] = lines.get((px, py), 0) | bits

    def hline(xa: int, xb: int, py: int) -> None:
        if xa > xb:
            xa, xb = xb, xa
        for px in range(xa, xb + 1):
            mark(px, py, (_L if px > xa else 0) | (_R if px < xb else 0))

    def vline(px: int, ya: int, yb: int) -> None:
        if ya > yb:
            ya, yb = yb, ya
        for py in range(ya, yb + 1):
            mark(px, py, (_U if py > ya else 0) | (_D if py < yb else 0))

    def right_end(k: str) -> int:
        it = lay.items[k]
        return xs[it.col] + (len(cells[k]) if it.kind != "dummy" else widths[it.col])

    for _, hops in plan:
        for a, b, track in hops:
            ia, ib = lay.items[a], lay.items[b]
            xa, ya, yb = right_end(a), ypos(ia), ypos(ib)
            if ia.kind == "dummy":
                hline(xs[ia.col], xa, ya)
            if ib.kind == "dummy":
                hline(xs[ib.col], right_end(b), yb)
            if ia.col == ib.col:
                xt = xs[ia.col] + widths[ia.col] + 2 + track
                hline(xa, xt, ya)
                vline(xt, ya, yb)
                hline(right_end(b), xt, yb)
                continue
            xb = xs[ib.col] - 1
            if track < 0:
                hline(xa, xb, ya)
                continue
            xt = xs[ia.col] + widths[ia.col] + 2 + track
            hline(xa, xt, ya)
            vline(xt, ya, yb)
            hline(xt, xb, yb)
    canvas: dict[tuple[int, int], str] = {}
    for (px, py), bits in lines.items():
        canvas[(px, py)] = _BOX[bits] if unicode else ("-" if bits in (_L, _R, _L | _R) else
                                                       "|" if bits in (_U, _D, _U | _D) else "+")
    for fr in lay.frames:
        its = [lay.items[k] for k in fr.keys]
        xa = min(xs[i.col] for i in its) - 1
        xb = max(xs[i.col] + len(cells[i.key]) for i in its)
        ya = min(ypos(i) for i in its) - 1
        yb = max(ypos(i) for i in its) + 1
        last = max(i.col for i in its)
        xb = max(xb, xs[last] + widths[last])
        h, v = ("─", "│") if unicode else ("-", "|")
        corners = "┌┐└┘" if unicode else "++++"
        for px in range(xa, xb + 1):
            canvas[(px, ya)] = h
            canvas[(px, yb)] = h
        for py in range(ya, yb + 1):
            canvas[(xa, py)] = v if (xa, py) not in lines else ("┤" if unicode else "+")
            canvas[(xb, py)] = v if (xb, py) not in lines else ("├" if unicode else "+")
        canvas[(xa, ya)], canvas[(xb, ya)], canvas[(xa, yb)], canvas[(xb, yb)] = corners
        tag = f" {fr.label} "
        start = (xa + xb + 1 - len(tag)) // 2
        for j, ch in enumerate(tag):
            if start + j < xb:
                canvas[(start + j, ya)] = ch
    for k, it in lay.items.items():
        if it.kind == "dummy":
            continue
        for j, ch in enumerate(cells[k]):
            canvas[(xs[it.col] + j, ypos(it))] = ch
    height = max((py for _, py in canvas), default=0) + 1
    width = max((px for px, _ in canvas), default=0) + 1
    rows = []
    for py in range(height):
        rows.append("".join(canvas.get((px, py), " ") for px in range(width)).rstrip())
    while rows and not rows[0].strip():
        rows.pop(0)
    head = f"scalar: {lay.scalar}"
    if lay.summary:
        head += "   (summarised: one box per term)"
    legend = [l if unicode else _ascii(l) for l in lay.legend]
    return "\n".join([head if unicode else _ascii(head), *rows, *legend])


# -- matplotlib ------------------------------------------------------------------------------

_FILL = {"Z": "#8fd18f", "X": "#ef8f8f", "F": "#f2e07a", "T": "#f2e07a", "Ti": "#f2e07a"}


def density(lay: Layout) -> float:
    """Pixels rendered per displayed pixel: 2, growing with the square root of the drawn
    items beyond 40 and with the scale-down a figure wider than the terminal needs."""
    items = sum(1 for it in lay.items.values() if it.kind != "dummy")
    value = 2 * max(1.0, math.sqrt(items / 40))
    cell = D.TERMINAL.cell_width
    if cell:
        room = max(10, shutil.get_terminal_size().columns - 1) * cell
        natural = (lay.columns * 1.6 + 2) * 0.9 * D.CONFIG.size * 100
        value *= max(1.0, natural / room)
    return min(value, 8.0)


def figure(
    lay: Layout, *, dark_theme: bool, opaque: bool = False, density: float = 1
) -> bytes:
    """The layout drawn with matplotlib as a PNG at ``density`` times 100 dpi."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

    fg, bg = D.colours(dark_theme)
    if D.CONFIG.color != "auto":
        fg = D.CONFIG.color
    scale = D.CONFIG.size
    ncol = lay.columns
    nrow = 1 + max((i.row for i in lay.items.values()), default=0)
    dx, dy = 1.6, 0.9
    width = (ncol * dx + 2) * 0.9 * scale
    height = (nrow * dy + 1.5) * 0.9 * scale
    dpi = 100 * min(density, D._MAX_PIXELS / (100 * max(width, height)))
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(-1, ncol * dx + 0.5)
    ax.set_ylim(-(nrow * dy) - 0.3, 1.0)
    font = 9 * scale
    def at(k: str) -> tuple[float, float]:
        it = lay.items[k]
        return it.col * dx, -it.row * dy
    tracks, plan = _routes(lay)
    labelled: set[str] = set()
    for e, hops in plan:
        for a, b, track in hops:
            (xa, ya), (xb, yb) = at(a), at(b)
            if track < 0:
                xs_, ys_ = [xa, xb], [ya, yb]
            else:
                c = lay.items[a].col
                step = (dx - 0.8) / (tracks[c] + 1)
                xm = xa + 0.4 + step * (track + 1)
                if lay.items[a].col == lay.items[b].col:
                    xs_, ys_ = [xa, xm, xm, xb], [ya, ya, yb, yb]
                else:
                    xs_, ys_ = [xa, xm, xm, xb], [ya, ya, yb, yb]
            ax.plot(xs_, ys_, color=fg, lw=0.7, zorder=1)
        first = e.path[0]
        if first not in labelled and lay.items[first].kind != "dummy":
            labelled.add(first)
            (xa, ya) = at(first)
            ax.text(xa + 0.3, ya + 0.04, _dim_label(e.dim), color=fg, fontsize=font * 0.55,
                    ha="left", va="bottom", zorder=4)
    for fr in lay.frames:
        pts = [at(k) for k in fr.keys]
        xa, xb = min(p[0] for p in pts) - 0.5, max(p[0] for p in pts) + 0.5
        ya, yb = min(p[1] for p in pts) - 0.4, max(p[1] for p in pts) + 0.4
        ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ls="--", ec=fg, lw=0.9))
        ax.text(xb, yb, f" {fr.label}", color=fg, fontsize=font, ha="left", va="bottom")
    for k, it in lay.items.items():
        x, y = at(k)
        if it.kind == "dummy":
            continue
        if it.kind in ("Z", "X"):
            ax.add_patch(Circle((x, y), 0.17, fc=_FILL[it.kind], ec=fg, lw=0.8, zorder=2))
            ax.text(x, y, it.kind, color="#111111", fontsize=font * 0.75, ha="center",
                    va="center", zorder=3)
            if it.label:
                ax.text(x, y + 0.22, it.label, color=fg, fontsize=font * 0.8, ha="center",
                        va="bottom", zorder=3)
        elif it.kind == "W":
            tri = [(x - 0.17, y - 0.17), (x - 0.17, y + 0.17), (x + 0.2, y)]
            ax.add_patch(Polygon(tri, closed=True, fc=bg, ec=fg, lw=0.9, zorder=2))
        elif it.kind in ("F", "T", "Ti"):
            ax.add_patch(Rectangle((x - 0.15, y - 0.15), 0.3, 0.3, fc=_FILL[it.kind], ec=fg,
                                   lw=0.8, zorder=2))
            ax.text(x, y, it.kind, color="#111111", fontsize=font * 0.7, ha="center",
                    va="center", zorder=3)
        elif it.kind in ("S", "B"):
            dia = [(x, y - 0.14), (x + 0.12, y), (x, y + 0.14), (x - 0.12, y)]
            ax.add_patch(Polygon(dia, closed=True, fc="#bbbbbb", ec=fg, lw=0.8, zorder=2))
        elif it.kind == "group":
            half = min(dx * 0.48, 0.035 * len(it.label) * scale + 0.1)
            ax.add_patch(FancyBboxPatch((x - half, y - 0.18), 2 * half, 0.36,
                                        boxstyle="round,pad=0.02", fc=bg, ec=fg, lw=0.8,
                                        zorder=2))
            ax.text(x, y, it.label, color=fg, fontsize=font * 0.7, ha="center", va="center",
                    zorder=3)
        else:
            ax.text(x, y, it.label, color=fg, fontsize=font * 0.8, ha="center", va="center",
                    zorder=3, bbox={"fc": bg, "ec": "none", "pad": 1})
    corner = f"scalar: {lay.scalar}"
    if lay.legend:
        corner += "\n" + "\n".join(lay.legend)
    fig.text(0.01, 0.99, corner, color=fg, fontsize=font * 0.8, ha="left", va="top")
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=dpi, transparent=not opaque,
                facecolor=bg if opaque else "none")
    return out.getvalue()


def _utf8() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("-", "") in ("utf8", "utf8sig")


def draw(view: View, style: str = "matplot") -> str:
    """``view`` drawn in ``style``; matplot falls back to Unicode without an image protocol
    or matplotlib, Unicode to ASCII on a terminal that is not UTF-8. With
    :data:`~archytaszx.repl.display.SAVE_DIR` set the matplotlib drawing is also saved there."""
    if style not in STYLES:
        raise ValueError(f"unknown drawing style {style!r}")
    lay = layout(view)
    proto = D.protocol() if style == "matplot" else "none"
    if proto != "none" or D.SAVE_DIR is not None:
        dense = density(lay) if proto == "none" or D.scalable(proto) else 1
        try:
            png = figure(
                lay,
                dark_theme=D.dark(),
                opaque=proto == "sixel" or D.SAVE_DIR is not None,
                density=dense,
            )
        except Exception:  # noqa: BLE001
            png = None
        if png is not None:
            D.keep(png, "zx")
            image = D.encode_image(png, proto, dense) if style == "matplot" else None
            if image is not None:
                return image
    if style == "matplot":
        style = "unicode"
    if style == "unicode" and not _utf8():
        style = "ascii"
    return grid(lay, unicode=style == "unicode")
