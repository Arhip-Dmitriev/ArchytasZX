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

:func:`layout` places a :class:`View`'s items in columns, inputs leftmost and outputs
rightmost, splits every wire into hops through the channels between columns (dummy points
where it passes a column) and orders each column by barycentre sweeps, keeping each bang box's
items together and each connected component in one block. A diagram above :data:`MAX_NODES`
nodes is summarised: each summed term becomes one box and chains of splitters or binders one
node.

:func:`plan` gives each item a band of lanes, one lane per port, by a longest-path solve of the
column and bang-box constraints, then routes each hop through its channel with a maze search in
which no two wires share a segment or a bend, wires cross only at right angles and a wire
crosses a box border only to enter or leave that box. :func:`grid` draws a plan on a character
grid, :func:`figure` with matplotlib; :func:`draw` picks a renderer with fallbacks.
"""

from __future__ import annotations

import heapq
import io
import itertools
import logging
import math
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef
from archytaszx.repl import terminal as T
from archytaszx.repl.printer import rule_turns, rule_weight


@dataclass
class View:
    """A diagram with the nodes of each summed term grouped: ``groups`` maps a node to its
    term, ``labels`` names each term, ``outputs`` names the outputs (``out k`` when empty),
    ``notes`` define dimension names and ``shaded`` nodes (the conjugate sheet of a doubled
    diagram) are drawn shaded."""

    diagram: Diagram
    groups: dict[NodeId, int] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    shaded: frozenset[NodeId] = frozenset()


MAX_NODES = 150
"""The most nodes drawn one by one; above it terms and splitter chains are summarised."""

STYLES = ("matplot", "unicode", "ascii")

_SWEEPS = 8
_ALIGN = 12
_BEND = 3
_CROSS = 2
_K = sympy.Symbol("k", integer=True)


@dataclass
class Port:
    """A leg end on an item: its side (``L`` or ``R``), its wire and the port boxes around it,
    innermost first."""

    item: str
    side: str
    wire: int
    boxes: list[int] = field(default_factory=list)


@dataclass
class Item:
    """A drawn vertex: a node, a summarised term, a boundary end or a wire's dummy point, with
    its ports on each side top to bottom and the node boxes holding it, outermost first."""

    key: str
    kind: str
    label: str = ""
    node: Node | None = None
    col: int = 0
    left: list[str] = field(default_factory=list)
    right: list[str] = field(default_factory=list)
    boxes: list[int] = field(default_factory=list)
    component: int = 0
    shaded: bool = False


@dataclass
class Hop:
    """A wire's passage through the channel right of column ``channel``, from port ``a`` to
    port ``b``."""

    wire: int
    channel: int
    a: str
    b: str


@dataclass
class Wire:
    """A diagram wire: its end ports, its dimension and its hops in order from ``ends[0]``."""

    ends: tuple[str, str]
    dim: sympy.Expr
    hops: list[int] = field(default_factory=list)


@dataclass
class Frame:
    """A bang box: its multiplicity, nesting depth, and the items (node box) or ports (port box)
    it holds; ``span`` is the column range of a node box."""

    key: int
    mult: sympy.Expr
    depth: int
    items: set[str] = field(default_factory=set)
    ports: list[str] = field(default_factory=list)
    span: tuple[int, int] = (0, 0)


Tree = list["str | tuple[int, Tree]"]


@dataclass
class Layout:
    """Items in ordered columns, their ports, wires split into hops, bang boxes, the scalar and
    notes; ``tree`` lists each column top to bottom with every node box's contents nested."""

    items: dict[str, Item]
    ports: dict[str, Port]
    wires: list[Wire]
    hops: list[Hop]
    frames: list[Frame]
    scalar: sympy.Expr
    notes: list[str] = field(default_factory=list)
    summary: bool = False
    tree: dict[int, Tree] = field(default_factory=dict)

    @property
    def columns(self) -> int:
        return 1 + max((i.col for i in self.items.values()), default=0)


# -- labels ----------------------------------------------------------------------------------

_GREEK = dict(
    zip(
        [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "pi",
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
            "Gamma",
            "Delta",
            "Theta",
            "Lambda",
            "Xi",
            "Pi",
            "Sigma",
            "Upsilon",
            "Phi",
            "Psi",
            "Omega",
        ],
        "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ",
        strict=True,
    )
)
_GREEK_RE = re.compile(r"\b(" + "|".join(_GREEK) + r")\b")


def _clean(e: sympy.Expr) -> sympy.Expr:
    e = sympy.sympify(e)
    if isinstance(e, sympy.Integer) and abs(int(e)) > 9999:
        parts = sympy.factorint(int(e))
        factors = [
            sympy.Pow(p, k, evaluate=False) if k > 1 else p for p, k in sorted(parts.items())
        ]
        return sympy.Mul(*factors, evaluate=False)
    return sympy.nsimplify(e) if e.has(sympy.Float) else e


def _text(e: sympy.Expr, unicode: bool) -> str:
    """``e`` as plain text: Greek letters, ``·`` products and ``√`` in Unicode."""
    s = str(sympy.sstr(_clean(e))).replace("**", "^")
    if not unicode:
        return s
    s = _GREEK_RE.sub(lambda m: _GREEK[m.group(1)], s)
    s = re.sub(r"\bsqrt\((\w+)\)", r"√\1", s)
    s = re.sub(r"\bsqrt\(", "√(", s)
    s = re.sub(r"\bI\b", "i", s)
    s = re.sub(r"(?<=[\d)π])\*(?=[A-Za-zπα-ωΑ-Ω√(])", "", s)
    s = re.sub(r"(?<=π)\*", "", s)
    return s.replace("*", "·")


def _tex(e: sympy.Expr) -> str:
    """``e`` as mathtext without dollars: fractions, superscripts, ``{\\cdot}`` products."""
    s = str(sympy.latex(_clean(e), mul_symbol="dot"))
    s = re.sub(r"(\d|\\pi|\bi) \\cdot ", r"\1 ", s)
    s = re.sub(r" \\cdot (\\pi|e\^|i\b|\\sqrt)", r" \1", s)
    return s.replace(r" \cdot ", r"{\cdot}")


def _fmt(e: sympy.Expr, fmt: str) -> str:
    return _tex(e) if fmt == "tex" else _text(e, fmt == "uni")


def _mod1(e: sympy.Expr) -> bool:
    e = sympy.nsimplify(e) if e.has(sympy.Float) else e
    return bool(e.is_Rational and e.q == 1) or e == 0


def _sep(fmt: str) -> str:
    return r",\ " if fmt == "tex" else ", "


def _at(value: str, index: str, fmt: str) -> str:
    if fmt == "tex":
        return f"{value}@{index}" if index != "k" else f"{value}\\,@k"
    return f"{value}@{index}" if index != "k" else f"{value} @k"


def _turns(entries: dict[int, sympy.Expr], rule: sympy.Expr | None, fmt: str) -> str:
    parts = [_at(_fmt(v, fmt), str(k), fmt) for k, v in sorted(entries.items())]
    if rule is not None:
        parts.append(_at(_fmt(rule, fmt), "k", fmt))
    return _sep(fmt).join(parts)


def _phase_entries(node: Node) -> tuple[dict[int, sympy.Expr], sympy.Expr | None]:
    """The phase entries in turns and the family rule in ``k``; a concrete linear family as a
    rule."""
    pv = node.phase
    assert pv is not None
    entries = {k: p.to_sympy_turns() for k, p in pv.entries().items()}
    rule = rule_turns(pv) if pv.rule is not None else None
    if rule is None and pv.dim.is_concrete and len(entries) > 1:
        r = pv.dim.to_int()
        c = pv.get(1).to_sympy_turns()
        if r > 2 and all(_mod1(pv.get(k).to_sympy_turns() - k * c) for k in range(1, r)):
            return {}, c * _K
    return entries, rule


def _engine_label(node: Node, fmt: str) -> str:
    """Phases and weights as typed: ``α@1``, ``k/d @k``, ``w: λ@1``, joined by ``·``."""
    pv = node.phase
    if pv is None or pv.is_zero:
        return ""
    entries, rule = _phase_entries(node)
    phase = _turns(entries, rule, fmt)
    if not pv.is_weighted:
        return phase
    weights = {k: w.to_sympy() for k, w in pv.weights().items()}
    wrule = rule_weight(pv) if pv.weight_rule is not None else None
    weight = ("w{:}\\," if fmt == "tex" else "w: ") + _turns(weights, wrule, fmt)
    if not phase:
        return weight
    return phase + (r"\ \cdot\ " if fmt == "tex" else " · ") + weight


def _radians(turns: sympy.Expr) -> sympy.Expr:
    """``2π`` times ``turns`` with every phase symbol read as an angle already in radians."""
    turns = sympy.sympify(turns)
    swap = {x: x / (2 * sympy.pi) for x in turns.free_symbols if not x.is_integer}
    return sympy.simplify(2 * sympy.pi * turns.xreplace(swap))


def _cis(weight: sympy.Expr, angle: sympy.Expr, fmt: str) -> str:
    """``weight e^{i angle}``, a multiple of π in the exponent written ``2πi p/q``."""
    weight, angle = sympy.simplify(weight), sympy.simplify(angle)
    if angle == 0:
        return _fmt(weight, fmt)
    pi = {"tex": r"\pi ", "uni": "π", "ascii": "pi*"}[fmt]
    i = "i"
    ratio = sympy.simplify(angle / sympy.pi)
    star = "*" if fmt == "ascii" else ""
    if not ratio.has(sympy.pi) and not ratio.is_Add:
        num, den = sympy.fraction(ratio)
        c, rest = num.as_coeff_Mul()
        sign = "-" if c < 0 else ""
        lead = "" if abs(c) == 1 else f"{_fmt(abs(c), fmt)}{star}"
        tail = "" if rest == 1 else (star or " ") + _fmt(rest, fmt)
        power = f"{sign}{lead}{pi}{i}{tail}".replace(" " + " ", " ")
        if den != 1:
            power += "/" + _fmt(den, fmt)
    else:
        inner = _fmt(angle, fmt)
        power = f"{i}{star}({inner})" if angle.is_Add else f"{i}{star}{inner}"
    if fmt == "tex":
        head = "" if weight == 1 else _fmt(weight, fmt) + r"\,"
        return f"{head}e^{{{power}}}"
    head = "" if weight == 1 else _fmt(weight, fmt) + ("·" if fmt == "uni" else "*")
    return f"{head}exp({power})" if fmt == "ascii" else f"{head}e^({power})"


def _vector(
    node: Node,
    at: Callable[[int], str],
    indices: set[int],
    family: str | None,
    fmt: str,
    bare: bool,
    rest: str,
) -> str:
    """A component vector ``(v_1, ..., v_{d-1})``: a family alone as ``(v_k)_k``, one entry at
    index 1 of a symbolic dimension as ``v_1`` (``(v_1)`` unless ``bare``), other entries at a
    symbolic dimension listed up to the last set one and closed by ``rest, …, rest``."""
    pv = node.phase
    assert pv is not None
    dots = r"\dots" if fmt == "tex" else ("…" if fmt == "uni" else "...")
    if family is not None and not indices:
        return f"({family})_k"
    if pv.dim.is_concrete:
        seq = [at(k) for k in range(1, pv.dim.to_int())]
        if len(seq) == 1 and bare:
            return seq[0]
        return "(" + _sep(fmt).join(seq) + ")"
    if indices == {1} and family is None:
        return at(1) if bare else f"({at(1)})"
    last = max(indices, default=1)
    seq = [at(k) for k in range(1, last + 1)]
    tail = [at(last + 1), dots] if family is not None else [rest, dots, rest]
    return "(" + _sep(fmt).join([*seq, *tail]) + ")"


def _paper_label(node: Node, fmt: str) -> str:
    """The paper's notation: phases ``(α_1, …, α_{d-1})`` in radians, a weighted spider's
    coefficients ``w_k e^{iα_k}`` as a vector."""
    pv = node.phase
    if pv is None or pv.is_zero:
        return ""
    rule = rule_turns(pv) if pv.rule is not None else None
    phases = set(pv.entries())
    if not pv.is_weighted:
        family = None if rule is None else _fmt(_radians(rule), fmt)
        return _vector(
            node,
            lambda k: _fmt(_radians(pv.get(k).to_sympy_turns()), fmt),
            phases,
            family,
            fmt,
            bare=True,
            rest="0",
        )
    wrule = rule_weight(pv) if pv.weight_rule is not None else None
    family = None
    if rule is not None or wrule is not None:
        family = _cis(
            sympy.Integer(1) if wrule is None else wrule,
            sympy.Integer(0) if rule is None else _radians(rule),
            fmt,
        )
    return _vector(
        node,
        lambda k: _cis(pv.weight(k).to_sympy(), _radians(pv.get(k).to_sympy_turns()), fmt),
        phases | set(pv.weights()),
        family,
        fmt,
        bare=False,
        rest="1",
    )


def _boxed(it: Item) -> bool:
    """Whether ``it`` is a weighted spider drawn as a Z or X box."""
    node = it.node
    return (
        T.CONFIG.publication
        and it.kind in ("Z", "X")
        and node is not None
        and node.phase is not None
        and node.phase.is_weighted
    )


def _label(it: Item, fmt: str) -> str:
    """The label beside an item: phases of a spider, a group's term, a boundary's name."""
    if it.kind in ("Z", "X") and it.node is not None:
        if T.CONFIG.publication:
            return _paper_label(it.node, fmt)
        return _engine_label(it.node, fmt)
    if it.kind == "Ti" and fmt == "tex":
        return "T^{-1}"
    if it.kind in ("T", "Ti") and fmt == "tex":
        return "T"
    return ""


def _notes(lay: Layout, fmt: str) -> list[str]:
    """The drawing's notes: the phase unit, a dimension shared by every wire, the summary."""
    notes = []
    if not T.CONFIG.publication and any(
        it.node is not None
        and it.kind in ("Z", "X")
        and it.node.phase is not None
        and (it.node.phase.entries() or it.node.phase.rule is not None)
        for it in lay.items.values()
    ):
        notes.append("phases in turns")
    dims = {w.dim for w in lay.wires}
    if len(dims) == 1:
        (only,) = dims
        name = f"${_tex(only)}$" if fmt == "tex" else _text(only, fmt == "uni")
        notes.append(f"every wire has dimension {name}")
    if lay.summary:
        notes.append("summarised: one box per term")
    return notes + list(lay.notes)


def _dims_shown(lay: Layout) -> bool:
    return len({w.dim for w in lay.wires}) > 1


# -- layout ----------------------------------------------------------------------------------


def _key_order(k: str) -> tuple[int, int]:
    rank = {"o": 0, "i": 1, "g": 2, "n": 3, "d": 4}
    return rank[k[0]], int(k[1:])


def _extract(view: View, max_nodes: int) -> Layout:
    """Items, ports, wires and boxes of ``view``; wires inside a merged chain or a summarised
    term are dropped, a node's self-loop kept."""
    g = view.diagram
    summary = len(g.nodes) > max_nodes and bool(view.groups)
    owner: dict[NodeId, str] = {}
    items: dict[str, Item] = {}
    for nid, node in sorted(g.nodes.items()):
        if summary and nid in view.groups:
            t = view.groups[nid]
            key = f"g{t}"
            items.setdefault(key, Item(key, "group", f"term {t + 1}: {view.labels.get(t, '')}"))
            owner[nid] = key
            continue
        key = f"n{int(nid)}"
        items[key] = Item(key, node.generator_type.name, node=node, shaded=nid in view.shaded)
        owner[nid] = key
    if summary:
        _merge_chains(g, owner, items)
    ports: dict[str, Port] = {}
    sides: dict[tuple[str, str], list[tuple[tuple[object, ...], str]]] = {}
    port_of: dict[PortRef, str] = {}
    wires: list[Wire] = []

    def add(item: str, side: str, wire: int, rank: tuple[object, ...]) -> str:
        bucket = sides.setdefault((item, side), [])
        key = f"{item}.{side}{len(bucket)}"
        bucket.append((rank, key))
        ports[key] = Port(item, side, wire)
        return key

    def side(ref: PortRef) -> str:
        return "R" if ref.direction is Direction.OUTPUT else "L"

    def dim(ref: PortRef) -> sympy.Expr:
        return g.nodes[ref.node_id].legs(ref.direction)[ref.index].dim.to_sympy()

    seen: set[tuple[tuple[str, str], ...]] = set()
    for wire in sorted(g.wires, key=lambda w: w.sort_key()):
        a, b = owner[wire.a.node_id], owner[wire.b.node_id]
        grouped = items[a].kind == "group" or items[b].kind == "group"
        if a == b and (wire.a.node_id != wire.b.node_id or grouped):
            continue
        if grouped:
            pair = tuple(sorted(((a, side(wire.a)), (b, side(wire.b)))))
            if pair in seen:
                continue
            seen.add(pair)
        w = len(wires)
        pa = add(a, side(wire.a), w, wire.a.sort_key())
        pb = add(b, side(wire.b), w, wire.b.sort_key())
        port_of[wire.a], port_of[wire.b] = pa, pb
        wires.append(Wire((pa, pb), dim(wire.a)))
    for k, ref in enumerate(g.boundary_inputs):
        key = f"i{k}"
        items[key] = Item(key, "in", f"in {k}")
        w = len(wires)
        pa = add(key, "R", w, (0,))
        pb = add(owner[ref.node_id], side(ref), w, ref.sort_key())
        port_of[ref] = pb
        wires.append(Wire((pa, pb), dim(ref)))
    for k, ref in enumerate(g.boundary_outputs):
        key = f"o{k}"
        items[key] = Item(key, "out", view.outputs[k] if k < len(view.outputs) else f"out {k}")
        w = len(wires)
        pa = add(owner[ref.node_id], side(ref), w, ref.sort_key())
        pb = add(key, "L", w, (0,))
        port_of[ref] = pa
        wires.append(Wire((pa, pb), dim(ref)))
    for (item, s), bucket in sides.items():
        keys = [key for _, key in sorted(bucket, key=lambda e: e[0])]
        if s == "L":
            items[item].left = keys
        else:
            items[item].right = keys
    boxes = sorted(g.bang_boxes.items())
    frames = []
    for i, (_, box) in enumerate(boxes):
        depth, up = 0, box
        while up.parent is not None and up.parent in g.bang_boxes:
            depth, up = depth + 1, g.bang_boxes[up.parent]
        frame = Frame(i, box.multiplicity.to_sympy(), depth)
        if box.node_scope:
            frame.items = {owner[n] for n in box.node_scope if n in owner}
        else:
            refs = sorted(box.port_scope, key=lambda r: r.sort_key())
            frame.ports = [port_of[r] for r in refs if r in port_of]
        frames.append(frame)
        for key in frame.items:
            items[key].boxes.append(i)
        for key in frame.ports:
            ports[key].boxes.append(i)
    for it in items.values():
        it.boxes.sort(key=lambda f: (frames[f].depth, -len(frames[f].items)))
    for p in ports.values():
        p.boxes.sort(key=lambda f: -frames[f].depth)
    return Layout(items, ports, wires, [], frames, g.scalar.to_sympy(), list(view.notes), summary)


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


def _layer(lay: Layout) -> None:
    """Columns: inputs first, outputs last, a node fed from an input as early as its wires
    allow, any other node as late; empty columns removed."""
    items, ports = lay.items, lay.ports
    order = sorted(items, key=_key_order)
    succ: dict[str, list[str]] = {k: [] for k in items}
    loose: dict[str, list[str]] = {k: [] for k in items}
    for w in lay.wires:
        pa, pb = ports[w.ends[0]], ports[w.ends[1]]
        if pa.item == pb.item:
            continue
        if pa.side == "R" and pb.side == "L":
            succ[pa.item].append(pb.item)
        elif pa.side == "L" and pb.side == "R":
            succ[pb.item].append(pa.item)
        else:
            loose[pa.item].append(pb.item)
            loose[pb.item].append(pa.item)
    state: dict[str, int] = {}
    kept: dict[str, list[str]] = {k: [] for k in items}
    for root in order:
        if root in state:
            continue
        state[root] = 1
        stack = [(root, iter(succ[root]))]
        while stack:
            k, rest = stack[-1]
            nxt = next(rest, None)
            if nxt is None:
                state[k] = 2
                stack.pop()
                continue
            if state.get(nxt) == 1:
                continue
            kept[k].append(nxt)
            if nxt not in state:
                state[nxt] = 1
                stack.append((nxt, iter(succ[nxt])))
    pred: dict[str, list[str]] = {k: [] for k in items}
    for k, ns in kept.items():
        for m in ns:
            pred[m].append(k)
    indeg = {k: len(pred[k]) for k in items}
    topo = [k for k in order if indeg[k] == 0]
    for k in topo:
        for m in kept[k]:
            indeg[m] -= 1
            if indeg[m] == 0:
                topo.append(m)
    col: dict[str, int] = {}
    for k in topo:
        if items[k].kind == "in":
            col[k] = 0
        else:
            col[k] = max((col[p] + 1 for p in pred[k]), default=1)
    reach = {k for k in items if items[k].kind == "in"}
    for k in topo:
        if any(p in reach for p in pred[k]):
            reach.add(k)
    inner = [k for k in topo if items[k].kind not in ("in", "out")]
    last = max((col[k] for k in inner), default=0) + 1
    for k in items:
        if items[k].kind == "out":
            col[k] = last
    for k in reversed(topo):
        if k in inner and k not in reach and kept[k]:
            col[k] = min(col[m] for m in kept[k]) - 1
    for _ in range(3):
        for k in inner:
            if not pred[k] and not kept[k] and loose[k]:
                col[k] = min(max(col[loose[k][0]], 1), last - 1) if last > 1 else 1
    used = sorted(set(col.values()))
    rank = {c: i for i, c in enumerate(used)}
    for k, it in items.items():
        it.col = rank[col[k]]


def _components(lay: Layout) -> None:
    """Numbers connected components: those with inputs first, by their first input, then those
    with outputs, then closed ones."""
    parent = {k: k for k in lay.items}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for w in lay.wires:
        a, b = find(lay.ports[w.ends[0]].item), find(lay.ports[w.ends[1]].item)
        if a != b:
            parent[max(a, b)] = min(a, b)
    rank: dict[str, tuple[int, int]] = {}
    for k in sorted(lay.items, key=_key_order):
        r = find(k)
        kind = lay.items[k].kind
        key = (0 if kind == "in" else 1 if kind == "out" else 2, _key_order(k)[1])
        rank[r] = min(rank.get(r, key), key)
    ids = {r: i for i, r in enumerate(sorted(rank, key=lambda r: rank[r]))}
    for k, it in lay.items.items():
        it.component = ids[find(k)]


def _channel(lay: Layout, port: str) -> int:
    p = lay.ports[port]
    c = lay.items[p.item].col
    return c if p.side == "R" else c - 1


def _hops(lay: Layout) -> None:
    """Splits each wire into hops, one per channel, with a dummy item where it crosses a
    column; a dummy is inside the node boxes holding both ends."""
    items, ports = lay.items, lay.ports
    dummy = 0
    for wi, w in enumerate(lay.wires):
        a, b = w.ends
        if ports[a].side == "L" and ports[b].side == "R":
            w.ends = (b, a)
            a, b = b, a
        ca, cb = _channel(lay, a), _channel(lay, b)
        ia, ib = items[ports[a].item], items[ports[b].item]
        common = [f for f in ia.boxes if f in ib.boxes]
        step = 1 if cb > ca else -1
        cur, ch = a, ca
        while ch != cb:
            c = ch + 1 if step > 0 else ch
            key = f"d{dummy}"
            dummy += 1
            d = Item(key, "dummy", col=c, boxes=list(common), component=ia.component)
            d.left, d.right = [f"{key}.L0"], [f"{key}.R0"]
            ports[d.left[0]] = Port(key, "L", wi)
            ports[d.right[0]] = Port(key, "R", wi)
            items[key] = d
            enter, leave = (d.left[0], d.right[0]) if step > 0 else (d.right[0], d.left[0])
            w.hops.append(len(lay.hops))
            lay.hops.append(Hop(wi, ch, cur, enter))
            cur = leave
            ch += step
        w.hops.append(len(lay.hops))
        lay.hops.append(Hop(wi, ch, cur, b))
    for f in lay.frames:
        if f.items:
            cols = [items[k].col for k in f.items]
            f.span = (min(cols), max(cols))


def _inside(lay: Layout, g: int, f: int) -> bool:
    """Whether node box ``g`` lies inside node box ``f``."""
    fg, ff = lay.frames[g], lay.frames[f]
    return g != f and fg.items <= ff.items and (fg.items != ff.items or fg.depth > ff.depth)


def _arrange(
    lay: Layout, c: int, keys: list[str], value: dict[str, float], fvalue: dict[int, float]
) -> Tree:
    """Column ``c`` as a tree: siblings sorted by component then ``value``, each node box
    spanning ``c`` one contiguous block."""
    items = lay.items
    if all(items[k].kind in ("in", "out") for k in keys):
        ends: Tree = [*sorted(keys, key=lambda k: (items[k].component, _key_order(k)[1]))]
        return ends
    spanning = [f.key for f in lay.frames if f.items and f.span[0] <= c <= f.span[1]]
    children: dict[int | None, list[tuple[tuple[float, ...], str | int]]] = {None: []}
    comp = {f: min(items[k].component for k in lay.frames[f].items) for f in spanning}
    for f in spanning:
        children.setdefault(f, [])
    for f in spanning:
        ups = [g for g in spanning if _inside(lay, f, g)]
        up = min(ups, key=lambda g: (len(lay.frames[g].items), -lay.frames[g].depth), default=None)
        children[up].append(((comp[f], fvalue.get(f, 0.0), 1, f), f))
    for k in keys:
        it = items[k]
        home = next((f for f in reversed(it.boxes) if f in children), None)
        children[home].append(((it.component, value[k], 0, *_key_order(k)), k))

    def build(parent: int | None) -> Tree:
        out: Tree = []
        for _, e in sorted(children[parent], key=lambda x: x[0]):
            out.append(e if isinstance(e, str) else (e, build(e)))
        return out

    return build(None)


def _flatten(tree: Tree) -> list[str]:
    out: list[str] = []
    for e in tree:
        if isinstance(e, str):
            out.append(e)
        else:
            out.extend(_flatten(e[1]))
    return out


def _order(lay: Layout) -> None:
    """Rows within each column by barycentre sweeps over port positions; spider legs then
    sorted by the position of their far end."""
    items, ports = lay.items, lay.ports
    cols: dict[int, list[str]] = {}
    for k in sorted(items, key=lambda k: (items[k].component, _key_order(k))):
        cols.setdefault(items[k].col, []).append(k)
    partner: dict[str, str] = {}
    for h in lay.hops:
        partner[h.a], partner[h.b] = h.b, h.a
    pos = {k: float(i) for col in cols.values() for i, k in enumerate(col)}

    def ppos(p: str) -> float:
        it = items[ports[p].item]
        side = it.left if ports[p].side == "L" else it.right
        return pos[it.key] + (side.index(p) + 1) / (len(side) + 1)

    def bary(k: str, side: str) -> float:
        it = items[k]
        ps = it.left if side == "L" else it.right
        far = [partner[p] for p in ps if items[ports[partner[p]].item].col != it.col]
        return sum(ppos(q) for q in far) / len(far) if far else pos[k]

    def fvalues() -> dict[int, float]:
        out = {}
        for f in lay.frames:
            if f.items:
                far = [
                    ppos(partner[p])
                    for k in f.items
                    for p in items[k].left + items[k].right
                    if items[ports[partner[p]].item].col != items[k].col
                ]
                out[f.key] = (
                    sum(far) / len(far) if far else sum(pos[k] for k in f.items) / len(f.items)
                )
        return out

    def legs() -> None:
        for it in items.values():
            if it.kind not in ("Z", "X"):
                continue
            for ps in (it.left, it.right):
                far = {p: partner[p] for p in ps}
                if all(items[ports[q].item].col != it.col for q in far.values()):
                    rank = {p: (ppos(far[p]), i) for i, p in enumerate(ps)}
                    ps.sort(key=rank.__getitem__)

    for sweep in range(_SWEEPS):
        legs()
        forward = sweep % 2 == 1
        fv = fvalues()
        for c in sorted(cols, reverse=not forward):
            value = {k: bary(k, "L" if forward else "R") for k in cols[c]}
            cols[c] = _flatten(_arrange(lay, c, cols[c], value, fv))
            for i, k in enumerate(cols[c]):
                pos[k] = float(i)
    legs()
    fv = fvalues()
    for c in sorted(cols):
        lay.tree[c] = _arrange(lay, c, cols[c], dict(pos), fv)
        for i, k in enumerate(_flatten(lay.tree[c])):
            pos[k] = float(i)
    legs()


def layout(view: View, max_nodes: int = MAX_NODES) -> Layout:
    """The layered layout of ``view``; deterministic."""
    lay = _extract(view, max_nodes)
    _layer(lay)
    _components(lay)
    _hops(lay)
    _order(lay)
    return lay


# -- placement and routing --------------------------------------------------------------------


@dataclass
class Metrics:
    """What a renderer needs reserved: lanes between items, label lanes above an item, the
    width in channel slots of a text, a wire's dimension label, a box's multiplicity and the
    clear lanes around a boxed leg."""

    gap: int
    label_lanes: Callable[[Item], int]
    slots: Callable[[str], int]
    dim: Callable[[Wire], str]
    mult: Callable[[Frame], str]
    clear: int = 0


@dataclass
class Channel:
    """A routed channel: slot types left to right (``E`` edge, ``M`` margin, ``P`` track,
    ``S`` spacer, ``X…`` border), each box border's slot and each hop's corner points
    ``(slot, 2 * lane)``, odd values lying between lanes."""

    types: list[str]
    borders: dict[str, int]
    paths: dict[int, list[tuple[int, int]]]

    @property
    def width(self) -> int:
        return len(self.types)

    def margin(self, left: bool) -> range:
        """The margin slots at the left or right edge."""
        n = len(self.types)
        if left:
            end = 1
            while end < n and self.types[end] == "M":
                end += 1
            return range(1, end)
        start = n - 1
        while start > 0 and self.types[start - 1] == "M":
            start -= 1
        return range(start, n - 1)


@dataclass
class Plan:
    """Lanes and routes: each item's band top, height, glyph lane and centre (offsets from the
    top), each port's lane offset, each node box's interior lanes and each channel."""

    top: dict[str, int]
    height: dict[str, int]
    glyph: dict[str, int]
    centre: dict[str, float]
    offset: dict[str, int]
    lanes: dict[int, tuple[int, int]]
    channels: dict[int, Channel]

    def lane(self, lay: Layout, port: str) -> int:
        return self.top[lay.ports[port].item] + self.offset[port]


def _bands(lay: Layout, metrics: Metrics) -> tuple[dict[str, int], ...]:
    """Each item's height, glyph lane, port offsets (a boxed port gets a label lane above and a
    padding lane below per box) and centre."""
    height: dict[str, int] = {}
    glyph: dict[str, int] = {}
    offset: dict[str, int] = {}
    centre: dict[str, float] = {}
    for k, it in lay.items.items():
        ll = metrics.label_lanes(it)
        spans = []
        for plist in (it.left, it.right):
            cur, offs = 0, []
            for p in plist:
                n = len(lay.ports[p].boxes)
                a = n + metrics.clear if n else 0
                cur += a
                offs.append(cur)
                cur += 1 + a
            spans.append((plist, cur, offs))
        h = max(1, spans[0][1], spans[1][1])
        for plist, cur, offs in spans:
            shift = ll + (h - cur) // 2
            for p, o in zip(plist, offs, strict=True):
                offset[p] = shift + o
        height[k] = ll + h
        glyph[k] = ll + (h - 1) // 2
        centre[k] = ll + (h - 1) / 2
    return height, glyph, offset, centre  # type: ignore[return-value]


def _solve(
    lay: Layout, height: dict[str, int], offset: dict[str, int], gap: int
) -> tuple[dict[str, int], dict[int, tuple[int, int]]]:
    """Band tops and node box lanes: the least positions meeting the column and box
    constraints, iterated towards straight wires."""
    Var = tuple[str, object]
    edges: dict[Var, list[tuple[Var, int]]] = {}

    def add(u: Var, v: Var, w: int) -> None:
        edges.setdefault(u, []).append((v, w))
        edges.setdefault(v, [])

    def walk(entries: Tree, parent: int | None) -> None:
        prev: Var | None = None
        for e in entries:
            head: Var = ("i", e) if isinstance(e, str) else ("T", e[0])
            if prev is None:
                if parent is not None:
                    add(("T", parent), head, 1)
                else:
                    edges.setdefault(head, [])
            elif prev[0] == "i":
                assert isinstance(prev[1], str)
                add(prev, head, height[prev[1]] + (gap if head[0] == "i" else 1))
            else:
                add(prev, head, 2)
            if isinstance(e, str):
                prev = ("i", e)
            else:
                add(("T", e[0]), ("B", e[0]), 1)
                walk(e[1], e[0])
                prev = ("B", e[0])
        if parent is not None and prev is not None:
            if prev[0] == "i":
                assert isinstance(prev[1], str)
                add(prev, ("B", parent), height[prev[1]])
            else:
                add(prev, ("B", parent), 1)

    def comp(e: str | tuple[int, Tree]) -> int:
        if isinstance(e, str):
            return lay.items[e].component
        return min(lay.items[k].component for k in lay.frames[e[0]].items)

    for c in sorted(lay.tree):
        chunks: dict[int, Tree] = {}
        for e in lay.tree[c]:
            chunks.setdefault(comp(e), []).append(e)
        for n in sorted(chunks):
            walk(chunks[n], None)
    preds: dict[Var, list[tuple[Var, int]]] = {v: [] for v in edges}
    for u, outs in edges.items():
        for v, w in outs:
            preds[v].append((u, w))
    indeg = {v: len(preds[v]) for v in edges}
    keyed = sorted(edges, key=lambda v: (v[0], str(v[1])))
    topo = [v for v in keyed if indeg[v] == 0]
    for v in topo:
        for m, _ in edges[v]:
            indeg[m] -= 1
            if indeg[m] == 0:
                topo.append(m)
    done = set(topo)
    topo += [v for v in keyed if v not in done]

    def run(lb: dict[str, int]) -> dict[Var, int]:
        y: dict[Var, int] = {}
        for v in topo:
            low = lb.get(v[1], 0) if v[0] == "i" else 0  # type: ignore[call-overload]
            y[v] = max([low, *(y[u] + w for u, w in preds[v] if u in y)])
        return y

    def run_up(ub: dict[str, int], floor: int) -> dict[Var, int]:
        z: dict[Var, int] = {}
        for v in reversed(topo):
            high = ub.get(v[1], floor) if v[0] == "i" else floor  # type: ignore[call-overload]
            z[v] = min([high, *(z[m] - w for m, w in edges[v] if m in z)])
        return z

    partner: dict[str, str] = {}
    for h in lay.hops:
        partner[h.a], partner[h.b] = h.b, h.a
    y = run({})

    def desired(y: dict[Var, int]) -> dict[str, int]:
        """Per item the median lane that would make its wires to other columns straight."""
        lb: dict[str, int] = {}
        for k, it in lay.items.items():
            cands = []
            for p in it.left + it.right:
                q = partner.get(p)
                if q is None:
                    continue
                qi = lay.ports[q].item
                if lay.items[qi].col != it.col:
                    cands.append(y[("i", qi)] + offset[q] - offset[p])
            if cands:
                cands.sort()
                lb[k] = cands[(len(cands) - 1) // 2]
        return lb

    def swaps(y: dict[Var, int]) -> set[str]:
        """Right-hand items of hop pairs that exchange the same two lanes across a channel."""
        ends: dict[int, list[tuple[int, int, str]]] = {}
        for h in lay.hops:
            pa, pb = lay.ports[h.a], lay.ports[h.b]
            if lay.items[pa.item].col == lay.items[pb.item].col:
                continue
            if lay.items[pa.item].col > lay.items[pb.item].col:
                pa, pb, qa, qb = pb, pa, h.b, h.a
            else:
                qa, qb = h.a, h.b
            la = y[("i", pa.item)] + offset[qa]
            lb = y[("i", pb.item)] + offset[qb]
            if la != lb:
                ends.setdefault(h.channel, []).append((la, lb, pb.item))
        out = set()
        for hops in ends.values():
            seen = {(la, lb) for la, lb, _ in hops}
            out |= {k for la, lb, k in hops if lb > la and (lb, la) in seen}
        return out

    for sweep in range(_ALIGN):
        lb = desired(y)
        down = run(lb)
        if sweep >= _ALIGN - 2:
            y = down
            for _ in range(3):
                bump = swaps(y)
                if not bump:
                    break
                for k in bump:
                    lb[k] = y[("i", k)] + 1
                y = run(lb)
            continue
        ub = {k: lb.get(k, down[("i", k)]) for k in lay.items}
        up = run_up(ub, max(down.values(), default=0))
        y = run({k: (down[("i", k)] + up[("i", k)]) // 2 for k in lay.items})
    shift = _pack(lay, y, height, gap)
    top = {k: y[("i", k)] + shift[lay.items[k].component] for k in lay.items}
    lanes = {
        f.key: (
            y[("T", f.key)] + shift[_frame_comp(lay, f)],
            y[("B", f.key)] + shift[_frame_comp(lay, f)],
        )
        for f in lay.frames
        if ("T", f.key) in y
    }
    return top, lanes


def _frame_comp(lay: Layout, f: Frame) -> int:
    return min(lay.items[k].component for k in f.items)


def _pack(
    lay: Layout, y: dict[tuple[str, object], int], height: dict[str, int], gap: int
) -> dict[int, int]:
    """Per component the lane shift that stacks it just below the components before it in
    each column and channel it occupies; a component without boundary takes the first gap
    that fits."""
    items = lay.items
    group = {it.component: it.component for it in items.values()}

    def find(n: int) -> int:
        while group[n] != n:
            n = group[n]
        return n

    for f in lay.frames:
        if f.items:
            comps = sorted({items[k].component for k in f.items})
            for n in comps[1:]:
                group[find(n)] = find(comps[0])
    extent: dict[int, dict[int, tuple[int, int]]] = {}

    def cover(g: int, where: int, lo: int, hi: int) -> None:
        spans = extent.setdefault(g, {})
        a, b = spans.get(where, (lo, hi))
        spans[where] = (min(a, lo), max(b, hi))

    for k, it in items.items():
        cover(find(it.component), 2 * it.col, y[("i", k)], y[("i", k)] + height[k] - 1)
    for f in lay.frames:
        if f.items and ("T", f.key) in y:
            for c in range(f.span[0], f.span[1] + 1):
                cover(find(_frame_comp(lay, f)), 2 * c, y[("T", f.key)] - 1, y[("B", f.key)] + 1)
    for h in lay.hops:
        lanes = [y[("i", lay.ports[q].item)] for q in (h.a, h.b)]
        cover(find(items[lay.ports[h.a].item].component), 2 * h.channel + 1, min(lanes), max(lanes))
    closed = {find(it.component) for it in items.values()} - {
        find(it.component) for it in items.values() if it.kind in ("in", "out")
    }
    taken: dict[int, list[tuple[int, int]]] = {}
    skyline: dict[int, int] = {}
    moved: dict[int, int] = {}
    sep = 2 + gap
    for g in sorted(extent):
        spans = extent[g]
        base = -min(lo for lo, _ in spans.values())
        off = max(
            [0, *(skyline[w] + sep - (lo + base) for w, (lo, _) in spans.items() if w in skyline)]
        )
        if g in closed:
            off = 0
            while any(
                lo + base + off < b + sep and a < hi + base + off + sep
                for w, (lo, hi) in spans.items()
                for a, b in taken.get(w, [])
            ):
                off += 1
        moved[g] = base + off
        for w, (lo, hi) in spans.items():
            skyline[w] = max(skyline.get(w, -1), hi + base + off)
            taken.setdefault(w, []).append((lo + base + off, hi + base + off))
    return {n: moved[find(n)] for n in group}


@dataclass
class _Box:
    """A box border in a channel: its membership key, frame, port (port box) and lanes."""

    key: str
    frame: int
    port: str | None
    lanes: tuple[int, int]


def _port_boxes(lay: Layout, plan: Plan, port: str) -> list[_Box]:
    """The boxes around ``port``, innermost first, with their interior lanes."""
    lane = plan.lane(lay, port)
    return [
        _Box(f"P{f}:{port}", f, port, (lane - i, lane + i - 1))
        for i, f in enumerate(lay.ports[port].boxes, start=1)
    ]


def _levels(lay: Layout, boxes: list[_Box]) -> dict[str, int]:
    """Border levels from a channel edge: a box one past every box inside it."""

    def within(g: _Box, f: _Box) -> bool:
        if f.port is not None:
            return g.port == f.port and lay.frames[g.frame].depth > lay.frames[f.frame].depth
        if g.port is not None:
            return lay.ports[g.port].item in lay.frames[f.frame].items
        return _inside(lay, g.frame, f.frame)

    level: dict[str, int] = {}
    for f in sorted(
        boxes,
        key=lambda b: (b.port is None, len(lay.frames[b.frame].items), -lay.frames[b.frame].depth),
    ):
        level[f.key] = 1 + max(
            (level[g.key] for g in boxes if g.key in level and within(g, f)), default=-1
        )
    return level


class _Maze:
    """One channel's slots and lanes with the segments, points and box regions already taken."""

    def __init__(
        self,
        types: list[str],
        lanes: tuple[int, int],
        rects: list[tuple[str, int, int, int, int]],
        blocked: set[tuple[int, int]],
    ) -> None:
        self.types = types
        self.lo, self.hi = lanes
        self.rects = rects
        self.blocked = blocked
        self.h: set[tuple[int, int]] = set()
        self.v: set[tuple[int, int]] = set()
        self.use: dict[tuple[int, int], str] = {}
        self._mem: dict[tuple[int, int], frozenset[str]] = {}
        tracks = [x for x, t in enumerate(types) if t == "P"]
        self.mid = (tracks[0] + tracks[-1]) / 2 if tracks else 0.0

    def mem(self, x: int, y: int) -> frozenset[str]:
        """The boxes containing point ``(x, y)``."""
        got = self._mem.get((x, y))
        if got is None:
            got = frozenset(k for k, x0, x1, t, b in self.rects if x0 <= x <= x1 and t <= y <= b)
            self._mem[(x, y)] = got
        return got

    def search(
        self, start: tuple[int, int], goal: tuple[int, int], sdir: int, gdir: int, relaxed: bool
    ) -> list[tuple[int, int]] | None:
        """A* from ``start`` leaving in ``sdir`` to ``goal`` arriving in ``gdir``; bends and
        crossings cost extra; None when no route exists."""
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
        opp = (1, 0, 3, 2)
        width = len(self.types)
        exits = self.mem(*start) - self.mem(*goal)
        enters = self.mem(*goal) - self.mem(*start)
        gx, gy = goal
        s = (start[0], start[1], sdir)
        best = {s: 0.0}
        back: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        heap = [(float(abs(start[0] - gx) + abs(start[1] - gy)), 0.0, 0, s)]
        count = 0
        while heap:
            _, g, _, state = heapq.heappop(heap)
            x, y, d = state
            if (x, y) == goal and state != s:
                path = [(x, y)]
                while state in back:
                    state = back[state]
                    path.append(state[:2])
                return path[::-1]
            if g > best.get(state, g):
                continue
            through = (x, y) in self.use and state != s and not relaxed
            for nd in range(4):
                if nd == opp[d] or (through and nd != d) or (state == s and nd != sdir):
                    continue
                dx, dy = steps[nd]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and self.lo <= ny <= self.hi):
                    continue
                hit = (nx, ny) == goal
                if hit and nd != gdir:
                    continue
                if not hit and self.types[nx] == "E":
                    continue
                if nd >= 2 and self.types[x] != "P":
                    continue
                cost = 1.0 + (_BEND + abs(x - self.mid) * 1e-3 if nd != d else 0)
                if not relaxed:
                    seg = (x, min(y, ny)) if nd >= 2 else (min(x, nx), y)
                    if seg in (self.v if nd >= 2 else self.h):
                        continue
                    if (nx, ny) in self.blocked:
                        continue
                    other = self.use.get((nx, ny))
                    if other is not None:
                        if hit or other != ("v" if nd < 2 else "h"):
                            continue
                        cost += _CROSS
                    if self.rects:
                        m1, m2 = self.mem(x, y), self.mem(nx, ny)
                        if m1 != m2 and ((m1 - m2) - exits or (m2 - m1) - enters):
                            continue
                nxt = (nx, ny, nd)
                ng = g + cost
                if ng < best.get(nxt, ng + 1):
                    best[nxt] = ng
                    back[nxt] = state
                    count += 1
                    heapq.heappush(heap, (ng + abs(nx - gx) + abs(ny - gy), ng, count, nxt))
        return None

    def take(self, path: list[tuple[int, int]]) -> None:
        """Marks ``path``'s segments and points as used."""
        for (x1, y1), (x2, y2) in itertools.pairwise(path):
            if y1 == y2:
                self.h.add((min(x1, x2), y1))
            else:
                self.v.add((x1, min(y1, y2)))
        for i, p in enumerate(path):
            if i == 0 or i == len(path) - 1:
                self.use[p] = "e"
                continue
            a, b = path[i - 1], path[i + 1]
            if a[1] == p[1] == b[1]:
                self.use[p] = "x" if p in self.use else "h"
            elif a[0] == p[0] == b[0]:
                self.use[p] = "x" if p in self.use else "v"
            else:
                self.use[p] = "b"


def _corners(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = [path[0]]
    for i in range(1, len(path) - 1):
        a, p, b = path[i - 1], path[i], path[i + 1]
        if not ((a[0] == p[0] == b[0]) or (a[1] == p[1] == b[1])):
            out.append(p)
    out.append(path[-1])
    return out


def _route(
    lay: Layout, plan: Plan, metrics: Metrics, ch: int, hops: list[int], span: tuple[int, int]
) -> Channel:
    """Routes the hops of channel ``ch`` on their own tracks; unused tracks removed."""
    items, ports = lay.items, lay.ports
    left: list[_Box] = []
    right: list[_Box] = []
    inner: list[_Box] = []
    for f in lay.frames:
        if not f.items or f.key not in plan.lanes:
            continue
        box = _Box(f"F{f.key}", f.key, None, plan.lanes[f.key])
        if f.span[0] <= ch < f.span[1]:
            inner.append(box)
        elif ch == f.span[1]:
            left.append(box)
        elif ch == f.span[0] - 1:
            right.append(box)
    for it in items.values():
        if it.col == ch:
            for p in it.right:
                left += _port_boxes(lay, plan, p)
        elif it.col == ch + 1:
            for p in it.left:
                right += _port_boxes(lay, plan, p)
    texts: dict[bool, list[str]] = {True: [], False: []}
    for b in left + right:
        if b.port is not None:
            texts[b in left].append(metrics.mult(lay.frames[b.frame]))
    for h in hops:
        w = lay.wires[lay.hops[h].wire]
        if w.hops[0] == h and (text := metrics.dim(w)):
            texts[items[ports[lay.hops[h].a].item].col == ch].append(text)
    margin = {s: max((metrics.slots(t) for t in ts), default=-1) + 2 for s, ts in texts.items()}

    def ends(h: int) -> tuple[tuple[int, int], tuple[int, int], int]:
        hop = lay.hops[h]
        pa, pb = ports[hop.a], ports[hop.b]
        ea = 0 if items[pa.item].col == ch else -1
        eb = 0 if items[pb.item].col == ch else -1
        return (ea, plan.lane(lay, hop.a)), (eb, plan.lane(lay, hop.b)), 0

    straight = {
        h for h in hops if ends(h)[0][0] != ends(h)[1][0] and ends(h)[0][1] == ends(h)[1][1]
    }
    pool = 2 * (len(hops) - len(straight)) + 3
    lvl_left, lvl_right = _levels(lay, left), _levels(lay, right)
    nl = 1 + max(lvl_left.values(), default=-1)
    nr = 1 + max(lvl_right.values(), default=-1)
    order = sorted(hops, key=lambda h: (h not in straight, abs(ends(h)[0][1] - ends(h)[1][1]), h))
    tries = 4
    for attempt in range(tries):
        types = ["E"] + ["M"] * margin[True]
        for lvl in range(nl):
            types += ["P"] * pool + ["S", f"XL{lvl}"]
        types += ["P"] * pool
        for lvl in reversed(range(nr)):
            types += [f"XR{lvl}", "S"] + ["P"] * pool
        types += ["M"] * margin[False] + ["E"]
        width = len(types)
        slot = {t: i for i, t in enumerate(types) if t.startswith("X")}
        borders: dict[str, int] = {}
        rects: list[tuple[str, int, int, int, int]] = []
        blocked: set[tuple[int, int]] = set()
        lines: list[tuple[int, int, int, int]] = []
        for b in inner:
            rects.append((b.key, 0, width - 1, 2 * b.lanes[0], 2 * b.lanes[1]))
            lines.append((0, width - 1, *b.lanes))
        for b in left:
            xb = slot[f"XL{lvl_left[b.key]}"]
            borders[b.key] = xb
            rects.append((b.key, 0, xb - 1, 2 * b.lanes[0], 2 * b.lanes[1]))
            lines.append((0, xb, *b.lanes))
            blocked.update((x, 2 * b.lanes[0]) for x in range(1, xb))
        for b in right:
            xb = slot[f"XR{lvl_right[b.key]}"]
            borders[b.key] = xb
            rects.append((b.key, xb + 1, width - 1, 2 * b.lanes[0], 2 * b.lanes[1]))
            lines.append((xb, width - 1, *b.lanes))
            if b.port is not None:
                blocked.update((x, 2 * b.lanes[0]) for x in range(xb + 1, width - 1))
        maze = _Maze(types, (2 * span[0] - 2, 2 * span[1] + 2), rects, blocked)
        for x0, x1, t, bottom in lines:
            for y in (2 * t - 1, 2 * bottom + 1):
                for x in range(x0, x1 + 1):
                    maze.use[(x, y)] = "h"
                    if x < x1:
                        maze.h.add((x, y))
        paths: dict[int, list[tuple[int, int]]] = {}
        failed = None
        for h in order:
            (ea, la), (eb, lb), _ = ends(h)
            xa = 0 if ea == 0 else width - 1
            xb = 0 if eb == 0 else width - 1
            sdir = 0 if xa == 0 else 1
            gdir = 1 if xb == 0 else 0
            path = maze.search((xa, 2 * la), (xb, 2 * lb), sdir, gdir, relaxed=False)
            if path is None and attempt == tries - 1:
                path = maze.search((xa, 2 * la), (xb, 2 * lb), sdir, gdir, relaxed=True)
            if path is None:
                failed = h
                break
            maze.take(path)
            paths[h] = path
        if failed is None:
            break
        order.remove(failed)
        order.insert(0, failed)
        pool *= 2
    used = {x for x, _ in maze.v} | {p[0] for p, u in maze.use.items() if u == "b"}
    keep = [x for x, t in enumerate(types) if t != "P" or x in used]
    remap = {x: i for i, x in enumerate(keep)}
    return Channel(
        [types[x] for x in keep],
        {k: remap[x] for k, x in borders.items()},
        {h: [(remap[x], y) for x, y in _corners(p)] for h, p in paths.items()},
    )


def plan(lay: Layout, metrics: Metrics) -> Plan:
    """Lanes for every item and node box, and every channel routed."""
    height, glyph, offset, centre = _bands(lay, metrics)
    top, lanes = _solve(lay, height, offset, metrics.gap)
    result = Plan(top, height, glyph, centre, offset, lanes, {})  # type: ignore[arg-type]
    span = (0, max((top[k] + height[k] for k in top), default=1))
    by_channel: dict[int, list[int]] = {}
    for i, h in enumerate(lay.hops):
        by_channel.setdefault(h.channel, []).append(i)
    for ch in range(-1, lay.columns):
        result.channels[ch] = _route(lay, result, metrics, ch, by_channel.get(ch, []), span)
    return result


def _main_lane(lay: Layout, pl: Plan) -> float:
    """The middle lane of the first component's items."""
    main = [k for k, it in lay.items.items() if it.component == 0]
    if not main:
        return 0.0
    return (min(pl.top[k] for k in main) + max(pl.top[k] + pl.height[k] - 1 for k in main)) / 2


# -- character grid --------------------------------------------------------------------------

_L, _R, _U, _D = 1, 2, 4, 8
_BOX = {
    _L | _R: "─",
    _U | _D: "│",
    _R | _D: "┌",
    _L | _D: "┐",
    _R | _U: "└",
    _L | _U: "┘",
    _L: "─",
    _R: "─",
    _U: "│",
    _D: "│",
}
_ASCII = {
    _L | _R: "-",
    _U | _D: "|",
    _R | _D: ".",
    _L | _D: ".",
    _R | _U: "'",
    _L | _U: "'",
    _L: "-",
    _R: "-",
    _U: "|",
    _D: "|",
}
_GLYPHS = {
    True: {
        "Z": "Ⓩ",
        "X": "Ⓧ",
        "W": "◀W",
        "F": "[F]",
        "T": "▷T",
        "Ti": "▶T⁻¹",
        "S": "⊳S",
        "B": "⊲B",
        "G": "⏚",
    },
    False: {
        "Z": "Z",
        "X": "X",
        "W": "<W",
        "F": "[F]",
        "T": "|>T",
        "Ti": "|>T^-1",
        "S": ">S",
        "B": "<B",
        "G": "_|_",
    },
}


def _cell(it: Item, unicode: bool) -> str:
    text = _plain_cell(it, unicode)
    return f"{text}*" if it.shaded and text else text


def _plain_cell(it: Item, unicode: bool) -> str:
    fmt = "uni" if unicode else "ascii"
    if it.kind in ("out", "in"):
        return it.label
    if it.kind == "dummy":
        return ""
    if it.kind == "group":
        return f"[{it.label}]"
    label = _label(it, fmt)
    if it.kind in ("Z", "X"):
        if _boxed(it):
            return f"[{it.kind}]{label}"
        if unicode:
            return _GLYPHS[True][it.kind] + label
        return f"({it.kind}{' ' + label if label else ''})"
    return _GLYPHS[unicode].get(it.kind, it.kind)


def grid(lay: Layout, unicode: bool = True) -> str:
    """The layout drawn on a character grid: wires as box-drawing lines crossing as ``┼``
    (ASCII ``+``, corners ``.`` and ``'``), bang boxes in double lines with ``╫``/``╪`` where a
    wire crosses their border."""
    fmt = "uni" if unicode else "ascii"
    items, ports = lay.items, lay.ports
    shown = _dims_shown(lay)
    metrics = Metrics(
        0,
        lambda it: 0,
        len,
        lambda w: _text(w.dim, unicode) if shown else "",
        lambda f: _text(f.mult, unicode),
    )
    pl = plan(lay, metrics)
    cells = {k: _cell(it, unicode) for k, it in items.items()}

    def brackets(k: str) -> tuple[bool, bool]:
        it = items[k]
        if it.kind in ("dummy", "in", "out"):
            return False, False
        g = pl.glyph[k]
        return tuple(  # type: ignore[return-value]
            len(ps) > 1 or (len(ps) == 1 and pl.offset[ps[0]] != g) for ps in (it.left, it.right)
        )

    gw = {k: len(cells[k]) + sum(brackets(k)) for k in items}
    ncol = lay.columns
    widths = [1] * ncol
    for k, it in items.items():
        widths[it.col] = max(widths[it.col], gw[k])
    scalar = "" if lay.scalar == 1 else _text(lay.scalar, unicode)
    pad = len(scalar) + 2 if scalar else 0
    xslot: dict[tuple[int, int], int] = {}
    xcol: dict[int, int] = {}

    def place() -> None:
        x = pad
        for c in range(-1, ncol):
            if c >= 0:
                xcol[c] = x
                x += widths[c]
            for i in range(pl.channels[c].width):
                xslot[(c, i)] = x + i
            x += pl.channels[c].width

    place()
    for _ in range(3):
        grown = False
        for f in lay.frames:
            if not f.items or f.key not in pl.lanes:
                continue
            xr = xslot[(f.span[1], pl.channels[f.span[1]].borders[f"F{f.key}"])]
            xl = xslot[(f.span[0] - 1, pl.channels[f.span[0] - 1].borders[f"F{f.key}"])]
            need = len(metrics.mult(f)) + 1
            short = max(need - (xr - xcol[f.span[1]]), need + 2 - (xr - xl - 1))
            if short > 0:
                widths[f.span[1]] += short
                grown = True
                place()
        if not grown:
            break
    lmin = 0

    def row(lane: float) -> int:
        return int(2 * (lane - lmin)) + 1

    gx: dict[str, int] = {}
    for k, it in items.items():
        gx[k] = xcol[it.col] + (widths[it.col] - gw[k]) // 2

    def attach(p: str) -> tuple[int, int] | None:
        it = items[ports[p].item]
        if it.kind == "dummy":
            return None
        y = row(pl.lane(lay, p))
        return (gx[it.key] + gw[it.key], y) if ports[p].side == "R" else (gx[it.key] - 1, y)

    lines: dict[tuple[int, int], dict[int, int]] = {}

    def mark(w: int, x: int, y: int, bits: int) -> None:
        bucket = lines.setdefault((x, y), {})
        bucket[w] = bucket.get(w, 0) | bits

    for wi, w in enumerate(lay.wires):
        pts: list[tuple[int, int]] = []
        start = attach(w.ends[0])
        if start is not None:
            pts.append(start)
        for h in w.hops:
            ch = lay.hops[h].channel
            pts += [(xslot[(ch, x)], row(y / 2)) for x, y in pl.channels[ch].paths[h]]
        end = attach(w.ends[1])
        if end is not None:
            pts.append(end)
        for (x1, y1), (x2, y2) in itertools.pairwise(pts):
            if y1 == y2:
                a, b = sorted((x1, x2))
                for x in range(a, b + 1):
                    mark(wi, x, y1, (_L if x > a else 0) | (_R if x < b else 0))
            else:
                a, b = sorted((y1, y2))
                for y in range(a, b + 1):
                    mark(wi, x1, y, (_U if y > a else 0) | (_D if y < b else 0))
    canvas: dict[tuple[int, int], str] = {}
    for cell, ws in lines.items():
        if len(ws) > 1:
            canvas[cell] = "┼" if unicode else "+"
        else:
            bits = next(iter(ws.values()))
            canvas[cell] = (_BOX if unicode else _ASCII).get(bits, "─" if unicode else "-")
    h_, v_, hx, vx = ("═", "║", "╪", "╫") if unicode else ("=", ":", "*", "*")
    corners = ("╔", "╗", "╚", "╝") if unicode else ("#", "#", "#", "#")
    texts: list[tuple[int, int, str]] = []

    def frame(xl: int, xr: int, yt: int, yb: int, label: str, ly: int) -> None:
        for x in range(xl + 1, xr):
            for y in (yt, yb):
                canvas[(x, y)] = hx if (x, y) in lines else h_
        for y in range(yt + 1, yb):
            for x in (xl, xr):
                canvas[(x, y)] = vx if (x, y) in lines else v_
        canvas[(xl, yt)], canvas[(xr, yt)], canvas[(xl, yb)], canvas[(xr, yb)] = corners
        texts.append((xl + 1, yt, "!"))
        texts.append((xr - len(label), ly, label))

    for f in lay.frames:
        if f.items and f.key in pl.lanes:
            t, b = pl.lanes[f.key]
            xl = xslot[(f.span[0] - 1, pl.channels[f.span[0] - 1].borders[f"F{f.key}"])]
            xr = xslot[(f.span[1], pl.channels[f.span[1]].borders[f"F{f.key}"])]
            frame(xl, xr, row(t) - 1, row(b) + 1, metrics.mult(f), row(t))
        for p in f.ports:
            box = next(b for b in _port_boxes(lay, pl, p) if b.frame == f.key)
            ax, _ = attach(p)  # type: ignore[misc]
            col = items[ports[p].item].col
            t, b = box.lanes
            if ports[p].side == "R":
                xb = xslot[(col, pl.channels[col].borders[box.key])]
                frame(ax, xb, row(t) - 1, row(b) + 1, metrics.mult(f), row(t))
            else:
                xb = xslot[(col - 1, pl.channels[col - 1].borders[box.key])]
                frame(xb, ax, row(t) - 1, row(b) + 1, metrics.mult(f), row(t))
    for wi, w in enumerate(lay.wires):
        text = metrics.dim(w)
        if not text:
            continue
        h = w.hops[0]
        ch = lay.hops[h].channel
        x0, y0 = pl.channels[ch].paths[h][0]
        room = pl.channels[ch].margin(x0 == 0)
        if len(room) < len(text):
            continue
        first = xslot[(ch, room[0])] + (len(room) - len(text)) // 2
        texts.append((first, row(y0 / 2), text))
    for x, y, s in texts:
        for j, c in enumerate(s):
            canvas[(x + j, y)] = c
    heavy = ("┃", "┎", "┖", "┠", "┒", "┚", "┨") if unicode else ("]", "]", "]", "]", "[", "[", "[")
    for k, it in items.items():
        if it.kind == "dummy":
            continue
        lb, rb = brackets(k)
        top = pl.top[k]
        gy = row(top + pl.glyph[k])
        x = gx[k]
        for side, on, bx in (("L", lb, x), ("R", rb, x + gw[k] - 1)):
            if not on:
                continue
            ps = it.left if side == "L" else it.right
            prow = {row(top + pl.offset[p]) for p in ps}
            ya, yb = min(prow | {gy}), max(prow | {gy})
            for y in range(ya, yb + 1):
                if y not in prow:
                    c = heavy[0]
                elif side == "R":
                    c = heavy[1] if y == ya else heavy[2] if y == yb else heavy[3]
                else:
                    c = heavy[4] if y == ya else heavy[5] if y == yb else heavy[6]
                canvas[(bx, y)] = c
        for j, c in enumerate(cells[k]):
            canvas[(x + lb + j, gy)] = c
    if scalar:
        mid = row(round(_main_lane(lay, pl)))
        for j, c in enumerate(scalar):
            canvas[(j, mid)] = c
    ys = [y for _, y in canvas]
    width = max((x for x, _ in canvas), default=-1) + 1
    rows = [
        "".join(canvas.get((x, y), " ") for x in range(width)).rstrip()
        for y in range(min(ys, default=0), max(ys, default=-1) + 1)
    ]
    while rows and not rows[0]:
        rows.pop(0)
    while rows and not rows[-1]:
        rows.pop()
    return "\n".join([*rows, *_notes(lay, fmt)])


# -- matplotlib ------------------------------------------------------------------------------

_FILL = {
    "Z": "#009e73",
    "X": "#cc3311",
    "F": "#f2e07a",
    "S": "#c8c8c8",
    "B": "#c8c8c8",
    "Ti": "#b4b4b4",
}
_INK = {"Z": "#111111", "X": "#ffffff"}


def density(lay: Layout) -> float:
    """Pixels rendered per displayed pixel: 2, growing with the square root of the drawn
    items beyond 40 and with the scale-down a figure wider than the terminal needs."""
    items = sum(1 for it in lay.items.values() if it.kind != "dummy")
    value = 2 * max(1.0, math.sqrt(items / 40))
    cell = T.TERMINAL.cell_width
    if cell:
        room = max(10, shutil.get_terminal_size().columns - 1) * cell
        natural = (lay.columns * 1.6 + 2) * 0.9 * T.CONFIG.size * 100
        value *= max(1.0, natural / room)
    return min(value, 8.0)


def save_format() -> str:
    """The saved drawing format: the ``format`` setting, or PDF in publication mode and PNG
    otherwise when it is ``auto``."""
    if T.CONFIG.format != "auto":
        return T.CONFIG.format
    return "pdf" if T.CONFIG.publication else "png"


def figure(
    lay: Layout,
    *,
    dark_theme: bool,
    opaque: bool = False,
    density: float = 1,
    palette: tuple[str, str] | None = None,
    fmt: str = "png",
) -> bytes:
    """The layout drawn with matplotlib as ``fmt`` (png, pdf or svg) at ``density`` times
    100 dpi, in ``palette`` ``(fg, bg)`` when given; publication mode uses Computer Modern
    mathtext and serif text."""
    import matplotlib

    rc = {"pdf.fonttype": 42, "svg.fonttype": "path"}
    if T.CONFIG.publication:
        rc.update({"mathtext.fontset": "cm", "font.family": "serif"})
    with matplotlib.rc_context(rc):  # type: ignore[arg-type]
        return _figure(lay, dark_theme, opaque, density, palette, fmt)


def _figure(
    lay: Layout,
    dark_theme: bool,
    opaque: bool,
    density: float,
    palette: tuple[str, str] | None,
    fmt: str,
) -> bytes:
    from matplotlib import patheffects
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
    from matplotlib.textpath import TextToPath

    fg, bg = T.colours(dark_theme)
    if T.CONFIG.color != "auto":
        fg = T.CONFIG.color
    if palette is not None:
        fg, bg = palette
    s = T.CONFIG.size
    fs_label, fs_letter = 8.5 * s, 7.5 * s
    fs_dim = 0.85 * fs_label

    def snap(v: float, step: int = 1) -> float:
        """``v`` inches rounded to a multiple of ``step`` points."""
        return max(step, step * round(v * 72 / step)) / 72

    sx, py, r, hb = snap(0.085 * s, 2), snap(0.22 * s, 2), snap(0.12 * s), snap(0.12 * s)
    lw = 1.0
    ttp = TextToPath()
    sizes: dict[tuple[str, float], tuple[float, float]] = {}

    def measure(text: str, size: float) -> tuple[float, float]:
        key = (text, size)
        if key not in sizes:
            w, h, _ = ttp.get_text_width_height_descent(
                text, FontProperties(size=size), ismath="$" in text
            )
            sizes[key] = (w / 72, h / 72)
        return sizes[key]

    def plain(text: str) -> str:
        return text.replace("$", r"\$")

    lines: dict[str, list[str]] = {}
    shrink: dict[str, float] = {}
    for k, it in lay.items.items():
        text = _label(it, "tex")
        if not text:
            lines[k] = []
            continue
        parts = text.split(r",\ ")
        rows, cur = [], ""
        for part in parts:
            trial = f"{cur},\\ {part}" if cur else part
            if cur and measure(f"${trial}$", fs_label)[0] > 1.8 * s:
                rows.append(cur + ",")
                cur = part
            else:
                cur = trial
        rows.append(cur)
        lines[k] = [f"${x}$" for x in rows]
        widest = max(measure(x, fs_label)[0] for x in lines[k])
        shrink[k] = min(1.0, max(0.6, 2.2 * s / widest))
    line_h = fs_label * 1.3 / 72
    shown = _dims_shown(lay)
    metrics = Metrics(
        1,
        lambda it: math.ceil(len(lines[it.key]) * line_h / py) if lines[it.key] else 0,
        lambda text: math.ceil(measure(text, fs_dim)[0] / sx) + 1,
        lambda w: f"${_tex(w.dim)}$" if shown else "",
        lambda f: f"${_tex(f.mult)}$",
        1,
    )
    pl = plan(lay, metrics)
    items, ports = lay.items, lay.ports
    big = {"F", "T", "Ti", "W", "S", "B", "G"}

    def shape_w(it: Item) -> float:
        if it.kind in ("Z", "X"):
            return 2 * hb if _boxed(it) else 2 * r
        if it.kind in big:
            return 2 * hb
        if it.kind in ("in", "out"):
            return measure(plain(it.label), fs_label)[0]
        if it.kind == "group":
            return measure(plain(it.label), fs_label * 0.8)[0] + 0.12 * s
        return 0.0

    ncol = lay.columns
    widths = [0.0] * ncol
    scalar = "" if lay.scalar == 1 else f"${_tex(lay.scalar)}$"
    scalar = scalar.replace(r"\frac", r"\dfrac")
    fs_scalar = 1.1 * fs_label
    pad = snap((measure(scalar, fs_scalar)[0] + 0.25 * s if scalar else 0.0) + 0.15 * s)
    xslot: dict[tuple[int, int], float] = {}
    xcol: dict[int, float] = {}
    total = [0.0]

    def place() -> None:
        x = pad
        for c in range(-1, ncol):
            if c >= 0:
                xcol[c] = x
                x += widths[c]
            for i in range(pl.channels[c].width):
                xslot[(c, i)] = x + (i + 0.5) * sx
            x += pl.channels[c].width * sx
        total[0] = x

    def Y(lane: float) -> float:
        return -lane * py

    cx: dict[str, float] = {k: 0.0 for k in items}
    cy = {k: Y(pl.top[k] + pl.centre[k]) for k in items}

    def spread(n: int, j: int, half: float) -> float:
        return 0.0 if n == 1 else half * (1 - 2 * j / (n - 1))

    def attach(p: str) -> list[tuple[float, float]]:
        """The leg of port ``p`` from the item's outline out to its lane."""
        port = ports[p]
        it = items[port.item]
        if it.kind == "dummy":
            return []
        right = port.side == "R"
        sgn = 1 if right else -1
        ps = it.right if right else it.left
        j, n = ps.index(p), len(ps)
        x0, y0 = cx[it.key], cy[it.key]
        ly = Y(pl.lane(lay, p))
        kind = "box" if _boxed(it) else it.kind
        if kind in ("Z", "X"):
            t = math.radians(spread(n, j, 55))
            pt = (x0 + sgn * r * math.cos(t), y0 + r * math.sin(t))
            half = r
        elif kind in ("in", "out"):
            half = shape_w(it) / 2 + 0.04 * s
            pt = (x0 + sgn * half, y0)
        elif kind == "W":
            pt = (x0 - hb, y0) if not right else (snap(x0 + hb * 0.8), y0 + spread(n, j, 0.8 * hb))
            half = hb
        elif kind in ("T", "Ti"):
            pt = (x0 + hb, y0) if right else (snap(x0 - hb * 0.8), y0 + spread(n, j, 0.8 * hb))
            half = hb
        elif kind in ("S", "B"):
            narrow = (kind == "S") == right
            pt = (x0 + sgn * hb, y0 + spread(n, j, (0.4 if narrow else 0.7) * hb))
            half = hb
        else:
            half = shape_w(it) / 2 if kind == "group" else hb
            pt = (x0 + sgn * half, y0 + spread(n, j, 0.7 * hb))
        return [pt, (snap(x0 + sgn * (half + 0.05 * s + 0.6 * abs(ly - pt[1]))), ly)]

    for k, it in items.items():
        text_w = max((measure(x, fs_label * shrink.get(k, 1))[0] for x in lines[k]), default=0.0)
        legs = [attach(p) for p in it.left + it.right]
        reach = max((abs(x) for leg in legs for x, _ in leg), default=0.0)
        widths[it.col] = max(
            widths[it.col], shape_w(it) + 0.14 * s, text_w + 0.08 * s, 2 * reach + 0.06 * s
        )
    widths = [snap(w + 1 / 72, 2) for w in widths]
    place()
    for _ in range(3):
        grown = False
        for f in lay.frames:
            if not f.items or f.key not in pl.lanes:
                continue
            xr = xslot[(f.span[1], pl.channels[f.span[1]].borders[f"F{f.key}"])]
            xl = xslot[(f.span[0] - 1, pl.channels[f.span[0] - 1].borders[f"F{f.key}"])]
            need = measure(metrics.mult(f), fs_label)[0] + 0.1 * s
            short = max(need - (xr - xcol[f.span[1]]), need + 0.25 * s - (xr - xl))
            if short > 1e-9:
                widths[f.span[1]] += snap(short + 1 / 72, 2)
                grown = True
                place()
        if not grown:
            break
    for k, it in items.items():
        cx[k] = xcol[it.col] + widths[it.col] / 2
    nlanes = max((pl.top[k] + pl.height[k] for k in items), default=1)
    nlanes = max(
        [nlanes]
        + [pl.lanes[f.key][1] + 1 for f in lay.frames if f.items and f.key in pl.lanes]
        + [box.lanes[1] + 1 for p in ports for box in _port_boxes(lay, pl, p)]
    )
    notes = _notes(lay, "tex")
    notes_h = (len(notes) * fs_dim * 1.5 / 72 + 0.1 * s) if notes else 0.0
    ytop = Y(-1.5)
    ybot = Y(nlanes + 0.5) - snap(notes_h)
    width = snap(total[0] + 0.15 * s)
    height = ytop - ybot
    dpi = 100 * min(density, T._MAX_PIXELS / (100 * max(width, height)))
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    half = 0.5 / 72
    ax.set_xlim(-half, width - half)
    ax.set_ylim(ybot - half, ytop - half)
    rects: list[tuple[float, float, float, float]] = []
    for f in lay.frames:
        boxes: list[tuple[float, float, float, float, float]] = []
        if f.items and f.key in pl.lanes:
            t, b = pl.lanes[f.key]
            xl = xslot[(f.span[0] - 1, pl.channels[f.span[0] - 1].borders[f"F{f.key}"])]
            xr = xslot[(f.span[1], pl.channels[f.span[1]].borders[f"F{f.key}"])]
            boxes.append((xl, xr, Y(t) + py / 2, Y(b) - py / 2, Y(t)))
        for p in f.ports:
            box = next(b for b in _port_boxes(lay, pl, p) if b.frame == f.key)
            col = items[ports[p].item].col
            t, b = box.lanes
            it = items[ports[p].item]
            if ports[p].side == "R":
                xl = max(attach(q)[1][0] for q in it.right)
                xr = xslot[(col, pl.channels[col].borders[box.key])]
            else:
                xl = xslot[(col - 1, pl.channels[col - 1].borders[box.key])]
                xr = min(attach(q)[1][0] for q in it.left)
            boxes.append((xl, xr, Y(t) + py / 2, Y(b) - py / 2, Y(t)))
        for xl, xr, yt, yb, ly in boxes:
            rects.append((xl, xr, yt, yb))
            ax.add_patch(
                Rectangle(
                    (xl, yb), xr - xl, yt - yb, fill=False, ls=(0, (4, 2)), ec=fg, lw=lw, zorder=1
                )
            )
            ax.text(
                xl + 0.05 * s,
                yt - 0.035 * s,
                "!",
                color=fg,
                fontsize=fs_dim,
                ha="left",
                va="top",
                zorder=6,
            )
            ax.text(
                xr - 0.06 * s,
                ly,
                metrics.mult(f),
                color=fg,
                fontsize=fs_label * 0.9,
                ha="right",
                va="center",
                zorder=6,
            )
    halo = [patheffects.withStroke(linewidth=3.4, foreground=bg)]
    ticks: list[tuple[float, float]] = []
    for wi, w in enumerate(lay.wires):
        pts = attach(w.ends[0])
        for h in w.hops:
            ch = lay.hops[h].channel
            pts += [(xslot[(ch, x)], Y(y / 2)) for x, y in pl.channels[ch].paths[h]]
        pts += attach(w.ends[1])[::-1]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=fg,
            lw=lw,
            zorder=2 + wi * 1e-4,
            path_effects=halo,
            solid_joinstyle="miter",
            solid_capstyle="butt",
        )
        for (x1, y1), (x2, y2) in itertools.pairwise(pts):
            for xl, xr, yt, yb in rects:
                if x1 == x2 and xl < x1 < xr:
                    ticks += [(x1, yy) for yy in (yt, yb) if min(y1, y2) < yy < max(y1, y2)]
                elif y1 == y2 and yb < y1 < yt:
                    ticks += [(xx, y1) for xx in (xl, xr) if min(x1, x2) < xx < max(x1, x2)]
    if ticks:
        ax.plot([t[0] for t in ticks], [t[1] for t in ticks], "o", ms=2.4 * s, color=fg, zorder=3)
    for wi, w in enumerate(lay.wires):
        text = metrics.dim(w)
        if not text:
            continue
        h = w.hops[0]
        ch = lay.hops[h].channel
        x0, y0 = pl.channels[ch].paths[h][0]
        span = pl.channels[ch].margin(x0 == 0)
        if not span:
            continue
        xm = (xslot[(ch, span[0])] + xslot[(ch, span[-1])]) / 2
        ax.text(
            xm,
            Y(y0 / 2),
            text,
            color=fg,
            fontsize=fs_dim,
            ha="center",
            va="center",
            zorder=5,
            bbox={"fc": bg, "ec": "none", "pad": 0.3},
        )
    for k, it in items.items():
        if it.kind == "dummy":
            continue
        drawn = (len(ax.patches), len(ax.lines), len(ax.texts))
        x, y = cx[k], cy[k]
        kind = "box" if _boxed(it) else it.kind
        if it.shaded:
            ax.add_patch(Circle((x, y), 1.6 * max(r, hb), fc=fg, ec="none", alpha=0.12, zorder=3))
        if kind in ("Z", "X"):
            ax.add_patch(Circle((x, y), r, fc=_FILL[kind], ec=fg, lw=lw, zorder=4))
            if not T.CONFIG.publication:
                ax.text(
                    x,
                    y,
                    kind,
                    color=_INK[kind],
                    fontsize=fs_letter * 0.85,
                    ha="center",
                    va="center",
                    zorder=5,
                )
        elif kind == "box":
            ax.add_patch(
                Rectangle(
                    (x - hb, y - hb), 2 * hb, 2 * hb, fc=_FILL[it.kind], ec=fg, lw=lw, zorder=4
                )
            )
        elif kind == "W":
            flat = snap(x + hb * 0.8)
            tri = [(x - hb, y), (flat, y + hb), (flat, y - hb)]
            ax.add_patch(Polygon(tri, closed=True, fc=fg, ec=fg, lw=lw, zorder=4))
        elif kind in ("T", "Ti"):
            flat = snap(x - hb * 0.8)
            tri = [(flat, y + hb), (flat, y - hb), (x + hb, y)]
            fill = bg if kind == "T" else _FILL["Ti"]
            ax.add_patch(Polygon(tri, closed=True, fc=fill, ec=fg, lw=lw, zorder=4))
        elif kind == "F":
            ax.add_patch(
                Rectangle((x - hb, y - hb), 2 * hb, 2 * hb, fc=_FILL["F"], ec=fg, lw=lw, zorder=4)
            )
            ax.text(
                x, y, "F", color="#111111", fontsize=fs_letter, ha="center", va="center", zorder=5
            )
        elif kind == "G":
            sgn = -1 if it.left else 1
            ax.plot([x + sgn * hb, x + sgn * 0.4 * hb], [y, y], color=fg, lw=lw, zorder=4)
            for step, half in ((0.4, 1.0), (0.0, 0.62), (-0.4, 0.28)):
                bx = x + sgn * step * hb
                ax.plot([bx, bx], [y - half * hb, y + half * hb], color=fg, lw=1.6 * lw, zorder=4)
        elif kind in ("S", "B"):
            wide, narrow = hb, hb * 0.5
            lh, rh = (wide, narrow) if kind == "S" else (narrow, wide)
            quad = [(x - hb, y + lh), (x + hb, y + rh), (x + hb, y - rh), (x - hb, y - lh)]
            ax.add_patch(Polygon(quad, closed=True, fc=_FILL[kind], ec=fg, lw=lw, zorder=4))
            ax.text(
                x,
                y,
                kind,
                color="#111111",
                fontsize=fs_letter * 0.8,
                ha="center",
                va="center",
                zorder=5,
            )
        elif kind == "group":
            half = shape_w(it) / 2
            ax.add_patch(
                FancyBboxPatch(
                    (x - half, y - 0.6 * hb),
                    2 * half,
                    1.2 * hb,
                    boxstyle="round,pad=0.02",
                    fc=bg,
                    ec=fg,
                    lw=lw,
                    zorder=4,
                )
            )
            ax.text(
                x,
                y,
                plain(it.label),
                color=fg,
                fontsize=fs_label * 0.8,
                ha="center",
                va="center",
                zorder=5,
            )
        else:
            ax.text(
                x,
                y,
                plain(it.label),
                color=fg,
                fontsize=fs_label,
                ha="center",
                va="center",
                zorder=5,
            )
        if it.shaded:
            for artist in (
                *ax.patches[drawn[0] + 1 :],
                *ax.lines[drawn[1] :],
                *ax.texts[drawn[2] :],
            ):
                artist.set_alpha(0.55)
        if lines[k]:
            size = fs_label * shrink[k]
            n = len(lines[k])
            ll = metrics.label_lanes(it)
            mid = Y(pl.top[k] + (ll - 1) / 2)
            for j, text in enumerate(lines[k]):
                ax.text(
                    x,
                    mid + (n - 1 - 2 * j) * line_h / 2,
                    text,
                    color=fg,
                    fontsize=size,
                    ha="center",
                    va="center",
                    zorder=5,
                )
    if scalar:
        ax.text(
            0.15 * s,
            Y(_main_lane(lay, pl)),
            scalar,
            color=fg,
            fontsize=fs_scalar,
            ha="left",
            va="center",
        )
    for j, note in enumerate(notes):
        ax.text(
            0.1 * s,
            Y(nlanes) - 0.05 * s - j * fs_dim * 1.5 / 72,
            note,
            color=fg,
            fontsize=fs_dim,
            ha="left",
            va="top",
        )
    out = io.BytesIO()
    fonts = logging.getLogger("fontTools")
    level = fonts.level
    fonts.setLevel(logging.ERROR)
    try:
        fig.savefig(
            out, format=fmt, dpi=dpi, transparent=not opaque, facecolor=bg if opaque else "none"
        )
    finally:
        fonts.setLevel(level)
    return out.getvalue()


def _utf8() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("-", "") in ("utf8", "utf8sig")


def draw(view: View, style: str = "matplot") -> str:
    """``view`` drawn in ``style``; matplot falls back to Unicode without an image protocol
    or matplotlib, Unicode to ASCII on a terminal that is not UTF-8. With
    :data:`~archytaszx.repl.terminal.SAVE_DIR` set the matplotlib drawing is also saved there
    in :func:`save_format`, black on white with :data:`~archytaszx.repl.terminal.SAVE_WHITE`."""
    if style not in STYLES:
        raise ValueError(f"unknown drawing style {style!r}")
    lay = layout(view)
    proto = T.protocol() if style == "matplot" else "none"
    if proto != "none" or T.SAVE_DIR is not None:
        dense = density(lay) if proto == "none" or T.scalable(proto) else 1
        png = None
        try:
            if T.SAVE_DIR is not None:
                ext = save_format()
                if T.SAVE_WHITE:
                    data = figure(
                        lay,
                        dark_theme=False,
                        opaque=True,
                        density=density(lay),
                        palette=T.WHITE,
                        fmt=ext,
                    )
                else:
                    data = figure(lay, dark_theme=T.dark(), opaque=True, density=dense, fmt=ext)
                T.keep(data, "zx", ext)
                if ext == "png" and not T.SAVE_WHITE:
                    png = data
            if proto != "none" and png is None:
                png = figure(lay, dark_theme=T.dark(), opaque=proto == "sixel", density=dense)
        except Exception:  # noqa: BLE001
            png = None
        if png is not None:
            image = T.encode_image(png, proto, dense) if style == "matplot" else None
            if image is not None:
                return image
    if style == "matplot":
        style = "unicode"
    if style == "unicode" and not _utf8():
        style = "ascii"
    return grid(lay, unicode=style == "unicode")
