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

"""Lean output for kernel peels and identity slides over linked diagrams (ArchytasZX/Linked.lean,
LinkedPeel.lean, LinkedSlide.lean, LinkedForm.lean, LinkedGhost.lean, LinkedSplice.lean). A boxed
diagram is written as
an ``LDiagram``: its nodes, its wires without those at chain link ports, its boundary, its node
boxes and its links, each link's ends the partners of its two ports; a link port on the boundary
is the ghost ``(G, True, i)`` of its slot ``i``, its boundary place and its link end (``evalG``),
or ``(x, True, i + 1)`` held in the box of the identity ``x`` a peel puts behind it. Each export
also writes each side with a phaseless Z(1 -> 1) behind every boundary link port (``Lb``, ``Lk``)
and proves its linked digit sum equal to the ghost form's (``chain_<tag>``: ``evalL_dropAt`` for
an identity wired to a node leg, ``evalG_ghostAt`` for one at a link end, each step's conditions
checked here on a mirror of the Lean definitions, then in Lean by ``decide +kernel``). Both
exports write both sides' box groups and state ``main`` over each side's digits read at the flat
positions of its ``slotData`` (``flatAt``, the engine's boundary splice). ``export_peel`` writes
the copy map ``cp`` from the peeled box's nodes to the copy's, ``PeelOK``, ``PeelFlat`` and
``main`` through ``evalL_peel_flat``. ``export_slide`` checks equal ``slotData`` and proves
``main`` through two chains to one diagram (``C``) and ``evalG_flatAt``. Both exports assume the
phaseless phase index ``zero`` (also that of a zero phase vector) reads the zero phase."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.graph import Diagram, Direction, PortRef

_KINDS = {"Z": "Z", "X": "X", "F": "F", "T": "T", "Ti": "Ti", "W": "W", "B": "B", "S": "S"}

Port = tuple[int, bool, int]
"""A leg: node id, output side or not, position."""


class LinkedUnsupported(Exception):
    """The diagram or step has no linked-diagram export."""


@dataclass(frozen=True)
class LNode:
    """A node: id, kind, dimension index, phase index, output and input dimension indices."""

    id: int
    kind: str
    dim: int
    ph: int
    outs: tuple[int, ...]
    ins: tuple[int, ...]


@dataclass(frozen=True)
class LBox:
    """A node box: id, multiplicity index, parent, nodes."""

    id: int
    mult: int
    parent: int | None
    nodes: tuple[int, ...]


@dataclass(frozen=True)
class LLink:
    """A chain link of the box ``box`` with its ports and ends."""

    box: int
    l1: Port
    l2: Port
    e1: Port
    e2: Port


@dataclass
class LDiag:
    """A linked diagram, mirroring the Lean ``LDiagram``."""

    nodes: list[LNode]
    wires: list[tuple[Port, Port]]
    outs: list[Port]
    ins: list[Port]
    boxes: list[LBox]
    links: list[LLink]

    def walk(self, pred: Callable[[LBox], bool], fuel: int, start: int | None) -> list[LBox]:
        out: list[LBox] = []
        for _ in range(fuel):
            nxt = next((b for b in self.boxes if pred(b) and b.parent == start), None)
            if nxt is None:
                break
            out.append(nxt)
            start = nxt.id
        return out

    def node_boxes(self, x: int) -> list[LBox]:
        return self.walk(lambda b: x in b.nodes, len(self.boxes), None)

    def chain(self, x: int) -> list[int]:
        return [b.mult for b in self.node_boxes(x)]

    def depth(self, y: int, x: int) -> int:
        bs = self.node_boxes(x)
        return next((i for i, b in enumerate(bs) if b.id == y), len(bs))

    def mult_of(self, y: int) -> int:
        return next((b.mult for b in self.boxes if b.id == y), 0)

    def legs(self) -> list[Port]:
        return [
            *(
                p
                for n in self.nodes
                for p in [
                    *((n.id, True, i) for i in range(len(n.outs))),
                    *((n.id, False, i) for i in range(len(n.ins))),
                ]
            )
        ]

    def bnd(self) -> list[Port]:
        return [*self.outs, *self.ins]


def _port(ref: PortRef) -> Port:
    return (int(ref.node_id), ref.direction is Direction.OUTPUT, ref.index)


class _Tables:
    """Dimension, phase and multiplicity indices shared by both sides."""

    def __init__(self) -> None:
        self.dims: dict[str, int] = {}
        self.phases: dict[str, int] = {}
        self.mults: dict[str, int] = {}

    @staticmethod
    def index(table: dict[str, int], key: str) -> int:
        return table.setdefault(key, len(table))


def _phase_key(phase: object) -> str:
    """The phase table key: a zero phase vector reads as no phase."""
    return repr(None) if isinstance(phase, PhaseVector) and phase.is_zero else repr(phase)


def linked(
    d: Diagram,
    tables: _Tables,
    idents: Mapping[Port, int] | None = None,
    ghost: int | None = None,
    held: Mapping[Port, LBox] | None = None,
) -> LDiag:
    """``d`` as a linked diagram, each boundary leg ``p`` of ``idents`` behind a phaseless
    Z(1 -> 1) of id ``idents[p]`` (wired to ``p``, its other leg in ``p``'s boundary place), with
    ``ghost`` every other boundary link port written as the ghost ``(ghost, True, i)`` of its slot
    ``i`` (its boundary place and its link end); a boundary link port ``p`` of ``held`` has its
    identity, or with ``ghost`` the ghost ``(x, True, i + 1)``, held as the node ``x`` of the box
    ``held[p]`` (appended after ``d``'s boxes). Raises :class:`LinkedUnsupported` outside plain
    node boxes with links whose ports are wired."""
    held = held or {}
    nodes = []
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        name = node.generator_type.name
        if name not in _KINDS:
            raise LinkedUnsupported(f"generator {name}")
        outs = tuple(_Tables.index(tables.dims, str(p.dim)) for p in node.outputs)
        ins = tuple(_Tables.index(tables.dims, str(p.dim)) for p in node.inputs)
        legs = [*outs, *ins]
        dim = legs[0] if legs else _Tables.index(tables.dims, "")
        ph = _Tables.index(tables.phases, _phase_key(node.phase))
        if name == "B" and len(ins) == 2:
            dim, ph = ins
        elif name == "S" and len(outs) == 2:
            dim, ph = outs
        nodes.append(LNode(int(nid), _KINDS[name], dim, ph, outs, ins))
    wires = [(_port(w.a), _port(w.b)) for w in sorted(d.wires, key=lambda w: w.sort_key())]
    bouts = [_port(r) for r in d.boundary_outputs]
    bins = [_port(r) for r in d.boundary_inputs]
    for bp, iid in sorted((idents or {}).items()):
        if bp not in bouts and bp not in bins:
            raise LinkedUnsupported("an identity end off the boundary")
        side = bouts if bp in bouts else bins
        owner = d.nodes[bp[0]]  # type: ignore[index]
        leg = (owner.outputs if bp[1] else owner.inputs)[bp[2]]
        k = _Tables.index(tables.dims, str(leg.dim))
        nodes.append(LNode(iid, "Z", k, _Tables.index(tables.phases, repr(None)), (k,), (k,)))
        side[side.index(bp)] = (iid, bp[1], 0)
        wires.append((bp, (iid, not bp[1], 0)))
    if ghost is not None:
        ports = {_port(r) for b in d.bang_boxes.values() for pq in b.links for r in pq}
        for j, bp in enumerate([*bouts, *bins]):
            if bp in ports:
                side, pos = (bouts, j) if j < len(bouts) else (bins, j - len(bouts))
                g = (held[bp].nodes[0], True, j + 1) if bp in held else (ghost, True, j)
                side[pos] = g
                wires.append((bp, g))
    if any(len(b.nodes) != 1 for b in held.values()):
        raise LinkedUnsupported("a held identity box of more than one node")
    boxes, links, used = [], [], set[int]()
    for bid, b in sorted(d.bang_boxes.items()):
        plain = (
            not b.port_scope
            and b.node_scope
            and not b.cyclic
            and b.index is None
            and b.copy_scalar is None
            and b.pair_of is None
            and not b.tree
            and b.feed is None
            and not b.sheets
            and b.align is None
        )
        if not plain:
            raise LinkedUnsupported(f"box {bid} is no plain node box")
        mult = _Tables.index(tables.mults, str(sp.expand(b.multiplicity.to_sympy())))
        par = None if b.parent is None else int(b.parent)
        boxes.append(LBox(int(bid), mult, par, tuple(sorted(int(x) for x in b.node_scope))))
        for p, q in b.links:
            ends = []
            for ref in (p, q):
                at = [i for i, w in enumerate(wires) if _port(ref) in w]
                if len(at) != 1 or at[0] in used:
                    raise LinkedUnsupported("a link port without one wire of its own")
                used.add(at[0])
                w = wires[at[0]]
                ends.append(w[1] if w[0] == _port(ref) else w[0])
            links.append(LLink(int(bid), _port(p), _port(q), ends[0], ends[1]))
    boxes += [held[bp] for bp in sorted(held)]
    kept = [w for i, w in enumerate(wires) if i not in used]
    out = LDiag(nodes, kept, bouts, bins, boxes, links)
    for w in kept:
        a, c = ([b.id for b in out.node_boxes(e[0])] for e in w)
        if a[: len(c)] != c and c[: len(a)] != a:
            raise LinkedUnsupported("a wire between boxes neither of which holds the other")
    return out


def _boundary_ends(
    before: Diagram, after: Diagram, bid: int, cp: dict[int, int]
) -> tuple[dict[Port, int], dict[Port, int]]:
    """The :func:`linked` identity ends of both sides of a peel of the box ``bid``: one fresh id
    per link port on ``before``'s boundary, on ``after`` at the same leg, or at its copy's for a
    first port of a link of ``bid``."""
    bnd = {_port(r) for r in [*before.boundary_outputs, *before.boundary_inputs]}
    nid = 1 + max((int(x) for x in [*before.nodes, *after.nodes]), default=0)
    ends_l: dict[Port, int] = {}
    ends_r: dict[Port, int] = {}
    for b_id, b in sorted(before.bang_boxes.items()):
        for p, q in b.links:
            for ref in (p, q):
                port = _port(ref)
                if port not in bnd or port in ends_l:
                    continue
                ends_l[port] = nid
                ends_r[_cpp(cp, port) if int(b_id) == bid and ref is p else port] = nid
                nid += 1
    return ends_l, ends_r


def _engine_ends(after: Diagram) -> dict[Port, int]:
    """The phaseless Z(1 -> 1) nodes of ``after`` with one leg on its boundary and the other
    wired, by the port they are wired to."""
    bnd = {_port(r) for r in [*after.boundary_outputs, *after.boundary_inputs]}
    out: dict[Port, int] = {}
    for nid, node in after.nodes.items():
        if node.generator_type.name != "Z" or len(node.inputs) != 1 or len(node.outputs) != 1:
            continue
        if _phase_key(node.phase) != repr(None):
            continue
        legs = [(int(nid), True, 0), (int(nid), False, 0)]
        on = [p for p in legs if p in bnd]
        wired = [
            (_port(w.b) if _port(w.a) in legs else _port(w.a))
            for w in after.wires
            if _port(w.a) in legs or _port(w.b) in legs
        ]
        if len(on) == 1 and len(wired) == 1:
            out[wired[0]] = int(nid)
    return out


def _held(
    after: Diagram, ends_r: dict[Port, int], tables: _Tables, ids: set[int]
) -> tuple[dict[Port, int], dict[Port, LBox]]:
    """The identity ends of ``ends_r`` that ``after`` already holds as phaseless Z(1 -> 1) nodes
    of ``ids``, each with the outermost box of ``after`` holding it, as that box with only the
    node."""
    found: dict[Port, int] = {}
    boxes: dict[Port, LBox] = {}
    bnd = {_port(r) for r in [*after.boundary_outputs, *after.boundary_inputs]}
    engine = {p: y for p, y in _engine_ends(after).items() if y in ids}
    for port in ends_r:
        if port in bnd or port not in engine:
            continue
        y = engine[port]
        owners = [
            (bid, b) for bid, b in sorted(after.bang_boxes.items()) if y in map(int, b.node_scope)
        ]
        if len(owners) != 1 or owners[0][1].parent is not None:
            raise LinkedUnsupported("an engine identity end in no single outermost box")
        bid, b = owners[0]
        mult = _Tables.index(tables.mults, str(sp.expand(b.multiplicity.to_sympy())))
        found[port] = y
        boxes[port] = LBox(int(bid), mult, None, (y,))
    return found, boxes


def _cpp(cp: dict[int, int], p: Port) -> Port:
    return (cp.get(p[0], 0), p[1], p[2])


def _wire_in(ws: Sequence[tuple[Port, Port]], w: tuple[Port, Port]) -> bool:
    return w in ws or (w[1], w[0]) in ws


def _same(L: LDiag, R: LDiag, lk: LLink, x: LLink) -> bool:
    """``x`` of ``R`` has ``lk``'s ports, copy coordinate and multiplicity."""
    return (
        (x.l1, x.l2, x.e1, x.e2) == (lk.l1, lk.l2, lk.e1, lk.e2)
        and R.depth(x.box, lk.l1[0]) == L.depth(lk.box, lk.l1[0])
        and R.mult_of(x.box) == L.mult_of(lk.box)
    )


def _kept(L: LDiag, R: LDiag, lk: LLink) -> bool:
    """Some link of ``R`` is :func:`_same` as ``lk``."""
    return any(_same(L, R, lk, x) for x in R.links)


def peel_ok(
    L: LDiag, R: LDiag, bid: int, mb: int, c: int, B: Sequence[int], cp: dict[int, int]
) -> bool:
    """The Lean ``PeelOK L R bid mb c B cp``, field by field."""
    inB = set(B)

    def cpn(n: LNode) -> LNode:
        return LNode(cp.get(n.id, 0), n.kind, n.dim, n.ph, n.outs, n.ins)

    def cpp(p: Port) -> Port:
        return _cpp(cp, p)

    legs, legs_r = L.legs(), R.legs()
    if any(not all(b.nodes for b in D.boxes) for D in (L, R)):
        return False
    for D in (L, R):
        if any(D.chain(w[0][0]) != D.chain(w[1][0]) for w in D.wires):
            return False
    if R.nodes != L.nodes + [cpn(n) for n in L.nodes if n.id in inB]:
        return False
    ids = [n.id for n in R.nodes]
    if len(set(ids)) != len(ids) or len(set(legs_r)) != len(legs_r):
        return False
    if len(set(R.bnd())) != len(R.bnd()):
        return False
    for n in L.nodes:
        if n.id in inB:
            r = R.chain(cp.get(n.id, 0))
            if L.chain(n.id) != [mb, *r] or R.chain(n.id) != [c, *r]:
                return False
        elif R.chain(n.id) != L.chain(n.id):
            return False
    for w in L.wires:
        a, b = w[0][0] in inB, w[1][0] in inB
        if w[0] not in legs or w[1] not in legs or a != b or not _wire_in(R.wires, w):
            return False
        if a and not _wire_in(R.wires, (cpp(w[0]), cpp(w[1]))):
            return False
    b_links = [lk for lk in L.links if lk.box == bid]
    for w in R.wires:
        ok = (
            w in L.wires
            or any(w0[0][0] in inB and w == (cpp(w0[0]), cpp(w0[1])) for w0 in L.wires)
            or any(w in ((cpp(lk.l1), lk.e1), (lk.e1, cpp(lk.l1))) for lk in b_links)
        )
        if not ok:
            return False
    for lk in L.links:
        ch = L.chain(lk.l1[0])
        dp = L.depth(lk.box, lk.l1[0])
        rest = ch[:dp] + ch[dp + 1 :]
        if not (
            all(p in legs for p in (lk.l1, lk.l2, lk.e1, lk.e2))
            and L.chain(lk.l2[0]) == ch
            and dp < len(ch)
            and ch[dp] == L.mult_of(lk.box)
            and L.chain(lk.e1[0]) == rest
            and L.chain(lk.e2[0]) == rest
        ):
            return False
        ins = [p[0] in inB for p in (lk.l1, lk.l2, lk.e1, lk.e2)]
        if lk.box == bid:
            moved = LLink(lk.box, lk.l1, lk.l2, cpp(lk.l2), lk.e2)
            if not (
                ins == [True, True, False, False]
                and dp == 0
                and L.mult_of(bid) == mb
                and moved in R.links
                and R.depth(bid, lk.l1[0]) == 0
                and R.mult_of(bid) == c
                and _wire_in(R.wires, (cpp(lk.l1), lk.e1))
            ):
                return False
        elif ins[0]:
            if not (
                all(ins)
                and dp >= 1
                and _kept(L, R, lk)
                and any(
                    (x.l1, x.l2, x.e1, x.e2) == tuple(map(cpp, (lk.l1, lk.l2, lk.e1, lk.e2)))
                    and R.depth(x.box, cp.get(lk.l1[0], 0)) + 1 == dp
                    and R.mult_of(x.box) == L.mult_of(lk.box)
                    for x in R.links
                )
            ):
                return False
        elif not (not any(ins) and _kept(L, R, lk)):
            return False
    for x in R.links:
        ok = (
            any(
                lk.box == bid and x == LLink(lk.box, lk.l1, lk.l2, cpp(lk.l2), lk.e2)
                for lk in L.links
            )
            or any(lk.box != bid and _same(L, R, lk, x) for lk in L.links)
            or any(
                lk.box != bid
                and lk.l1[0] in inB
                and (x.l1, x.l2, x.e1, x.e2) == tuple(map(cpp, (lk.l1, lk.l2, lk.e1, lk.e2)))
                and R.depth(x.box, cp.get(lk.l1[0], 0)) + 1 == L.depth(lk.box, lk.l1[0])
                and R.mult_of(x.box) == L.mult_of(lk.box)
                for lk in L.links
            )
        )
        if not ok:
            return False
    for p in L.bnd():
        if p not in legs or p not in R.bnd() or (p[0] in inB and cpp(p) not in R.bnd()):
            return False
    return all(q in L.bnd() or any(p[0] in inB and q == cpp(p) for p in L.bnd()) for q in R.bnd())


def _lport(p: Port) -> str:
    return f"⟨{p[0]}, {'true' if p[1] else 'false'}, {p[2]}⟩"


def _lnat(xs: Sequence[int]) -> str:
    return "[" + ", ".join(map(str, xs)) + "]"


def literal(D: LDiag) -> str:
    """``D`` as a Lean ``LDiagram`` term."""
    nodes = ", ".join(
        f"⟨{n.id}, .{n.kind}, {n.dim}, {n.ph}, {_lnat(n.outs)}, {_lnat(n.ins)}⟩" for n in D.nodes
    )
    wires = ", ".join(f"({_lport(a)}, {_lport(b)})" for a, b in D.wires)
    boxes = ", ".join(
        f"⟨{b.id}, {b.mult}, {'none' if b.parent is None else f'some {b.parent}'}, "
        f"{_lnat(b.nodes)}, []⟩"
        for b in D.boxes
    )
    links = ", ".join(
        f"⟨{x.box}, {_lport(x.l1)}, {_lport(x.l2)}, {_lport(x.e1)}, {_lport(x.e2)}⟩"
        for x in D.links
    )
    return (
        "{ base :=\n      { nodes := [" + nodes + "],\n        wires := [" + wires + "],\n"
        "        outs := [" + ", ".join(map(_lport, D.outs)) + "],\n"
        "        ins := [" + ", ".join(map(_lport, D.ins)) + "],\n"
        "        boxes := [" + boxes + "] },\n    links := [" + links + "] }"
    )


_FIELDS = 21
"""The fields of ``PeelOK``."""

_FLAT_FIELDS = 10
"""The fields of ``PeelFlat``."""


@dataclass
class PeelEnds:
    """A peel's box, its nodes ``B``, the copy map, both sides' identity ends and the boundary
    link ports of ``before`` whose identity ``after`` holds in a box of its own."""

    box: int
    B: list[int]
    fresh: list[int]
    cp: dict[int, int]
    ends_l: dict[Port, int]
    ends_r: dict[Port, int]
    held: dict[Port, LBox]


def peel_ends(before: Diagram, after: Diagram, tables: _Tables) -> PeelEnds | None:
    """The :class:`PeelEnds` of ``after`` peeling the last copy of one outermost node box of
    ``before``: the new nodes of ``after`` are the copies of the box's nodes and the phaseless
    Z(1 -> 1) nodes the peel puts behind boundary link ports; None outside that form."""
    shifted = [
        bid
        for bid, b in before.bang_boxes.items()
        if b.parent is None
        and bid in after.bang_boxes
        and sp.expand(b.multiplicity.to_sympy() - after.bang_boxes[bid].multiplicity.to_sympy())
        == 1
    ]
    if len(shifted) != 1:
        return None
    B = sorted(int(x) for x in before.bang_boxes[shifted[0]].node_scope)
    new = sorted({int(x) for x in after.nodes} - {int(x) for x in before.nodes})
    engine = set(_engine_ends(after).values()) & set(new)
    if len(new) == len(B):
        engine = set()
    elif len(engine) != len(new) - len(B):
        return None
    fresh = [x for x in new if x not in engine]
    cp = dict(zip(B, fresh, strict=True))
    ends_l, ends_r = _boundary_ends(before, after, int(shifted[0]), cp)
    try:
        found, boxes = _held(after, ends_r, tables, engine)
    except LinkedUnsupported:
        return None
    held: dict[Port, LBox] = {}
    for pb, nid in list(ends_l.items()):
        pa = next(p for p, v in ends_r.items() if v == nid)
        if pa in found:
            ends_l[pb] = found[pa]
            held[pb] = boxes[pa]
            del ends_r[pa]
    if set(new) != set(fresh) | {b.nodes[0] for b in held.values()}:
        return None
    return PeelEnds(int(shifted[0]), B, fresh, cp, ends_l, ends_r, held)


def export_peel(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the linked digit sums of ``before`` and ``after`` equal,
    ``after`` peeling the last copy of one outermost node box of ``before``; None outside that
    form."""
    tables = _Tables()
    pe = peel_ends(before, after, tables)
    if pe is None:
        return None
    shifted, B, fresh, cp = [pe.box], pe.B, pe.fresh, pe.cp
    ends_l, ends_r, held = pe.ends_l, pe.ends_r, pe.held
    try:
        L = linked(before, tables, ends_l, held=held)
        R = linked(after, tables, ends_r)
    except LinkedUnsupported:
        return None
    box = next(b for b in L.boxes if b.id == int(shifted[0]))
    c = next(x.mult for x in R.boxes if x.id == box.id)
    order = {n.id: n for n in R.nodes}
    if set(order) != {n.id for n in L.nodes} | set(fresh):
        return None
    R.nodes = [order[n.id] for n in L.nodes] + [
        order[cp[x]] for x in (n.id for n in L.nodes) if x in cp
    ]
    if not peel_ok(L, R, box.id, box.mult, c, B, cp):
        return None
    G = 1 + max(n.id for n in [*L.nodes, *R.nodes])
    ids_l, ids_r = set(ends_l.values()), set(ends_r.values())
    Dl, gl, sl = _reduce(L, ids_l, set(), G)
    Dr, gr, sr = _reduce(R, ids_r, ids_r, G)
    if gl is None or gr is None:
        return None
    try:
        Tl = linked(before, tables, ghost=G, held=held)
        Tr = linked(after, tables, ghost=G)
    except LinkedUnsupported:
        return None
    if not (_same_up_to_order(Tl, Dl) and _same_up_to_order(Tr, Dr)):
        return None
    Gl, Gr = groups(before), groups(after)
    sl_d, sr_d = slot_data(L, Gl), slot_data(R, Gr)
    if (slot_data(Dl, Gl), slot_data(Dr, Gr)) != (sl_d, sr_d):
        return None
    if not (nested(sl_d) and nested(sr_d)):
        return None
    if not peel_flat_ok(L, R, B, cp, box.mult, c, sl_d, sr_d):
        return None
    zero = _Tables.index(tables.phases, repr(None))
    table = ", ".join(f"({x}, {y})" for x, y in cp.items())
    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    flat = ",\n    ".join(["by decide +kernel"] * _FLAT_FIELDS)
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        "/-- The diagram before the peel, as a linked diagram, each boundary link port the\n"
        "ghost of its slot. -/\n"
        f"def L : LDiagram :=\n  {literal(Dl)}\n\n"
        "/-- The diagram after the peel, as a linked diagram, each boundary link port the\n"
        "ghost of its slot. -/\n"
        f"def L' : LDiagram :=\n  {literal(Dr)}\n\n"
        f"/-- The phase index of the phaseless nodes. -/\ndef zero : ℕ := {zero}\n\n"
        "/-- `L` with a phaseless Z(1 -> 1) behind each boundary link port. -/\n"
        + _alias("Lb", L, sl, "L")
        + "/-- `L'` with a phaseless Z(1 -> 1) behind each boundary link port and its copy. -/\n"
        + _alias("Lk", R, sr, "L'")
        + _chain("b", "Lb", "L", "gsL", sl, gl)
        + _chain("k", "Lk", "L'", "gsL'", sr, gr)
        + "/-- Each node of the peeled box to its node in the copy. -/\n"
        f"def cp (x : ℕ) : ℕ := (([{table}] : List (ℕ × ℕ)).lookup x).getD 0\n\n"
        f"theorem ok : PeelOK Lb Lk {box.id} {box.mult} {c} {_lnat(B)} cp :=\n"
        f"  ⟨{proof}⟩\n\n"
        "/-- The box groups before the peel, node lists by box. -/\n"
        f"def G : List (ℕ × List (List ℕ)) := {_lgroups(Gl)}\n\n"
        "/-- The box groups after the peel, node lists by box. -/\n"
        f"def G' : List (ℕ × List (List ℕ)) := {_lgroups(Gr)}\n\n"
        "theorem sdb : Lb.slotData G = L.slotData G := by\n  decide +kernel\n\n"
        "theorem sdk : Lk.slotData G' = L'.slotData G' := by\n  decide +kernel\n\n"
        f"theorem flat : PeelFlat Lb Lk {_lnat(B)} cp {box.mult} {c} (Lb.slotData G)\n"
        f"    (Lk.slotData G') :=\n  ⟨{flat}⟩\n\n"
        "/-- The digit sums read at the flat positions of the two sides' slot data agree. -/\n"
        "theorem main : ∀ (E : Env), (∀ k, E.phase zero k = 0) →\n"
        f"    ∀ (ks : ℕ → ℕ), ks {box.mult} = ks {c} + 1 → ∀ y : ℕ → ℕ,\n"
        "      L'.evalG gsL' E ks (flatAt (L'.slotData G') ks y) =\n"
        "        L.evalG gsL E ks (flatAt (L.slotData G) ks y) := by\n"
        "  intro E hE ks hk y\n"
        "  rw [← sdb, ← sdk]\n"
        "  exact (chain_k E hE ks _).symm.trans\n"
        "    ((evalL_peel_flat ok flat E ks hk y).trans (chain_b E hE ks _))\n\n"
        "end ArchytasZX.Export\n"
    )


Ghost = tuple[Port, int, int]
"""A ghost: its port, its boundary slot and its dimension index."""

Step = tuple[str, LNode, tuple[Port, ...], int]
"""A step: ``drop`` with the wire's ends, the wired leg, the other leg and the leg it meets, or
``ghost`` with the link-end leg, the boundary leg and the ghost port, and its slot."""


def _move_last(D: LDiag, x: LNode) -> LDiag:
    return LDiag(
        [n for n in D.nodes if n.id != x.id] + [x], D.wires, D.outs, D.ins, D.boxes, D.links
    )


def ghost_ok(D: LDiag, gs: Sequence[Ghost], x: LNode, a: Port, c: Port, g: Port, i: int) -> bool:
    """The Lean ``GhostOK D gs x a c g i``, field by field."""
    bnd = D.bnd()

    def rest(lk: LLink) -> list[int]:
        ch, dp = D.chain(lk.l1[0]), D.depth(lk.box, lk.l1[0])
        return ch[:dp] + ch[dp + 1 :] if dp < len(ch) else ch

    return (
        all(D.chain(p[0]) == D.chain(q[0]) for p, q in D.wires)
        and len({n.id for n in D.nodes}) == len(D.nodes)
        and D.nodes[-1] == x
        and x.kind == "Z"
        and x.outs == (x.dim,)
        and x.ins == (x.dim,)
        and {a, c} == {(x.id, True, 0), (x.id, False, 0)}
        and all(x.id not in (p[0], q[0]) for p, q in D.wires)
        and len(set(bnd)) == len(bnd)
        and 0 <= i < len(bnd)
        and bnd[i] == c
        and a not in bnd
        and g not in bnd
        and all(g not in w for w in D.wires)
        and g not in (a, c)
        and D.chain(g[0]) == D.chain(x.id)
        and all(c not in (lk.l1, lk.l2, lk.e1, lk.e2) for lk in D.links)
        and all(a not in (lk.l1, lk.l2) for lk in D.links)
        and all(g not in (lk.l1, lk.l2, lk.e1, lk.e2) for lk in D.links)
        and all(rest(lk) == D.chain(x.id) for lk in D.links if a in (lk.e1, lk.e2))
        and all(q[0] not in (a, c, g) for q in gs)
    )


def ghost_drop(D: LDiag, x: int, a: Port, c: Port, g: Port) -> LDiag:
    """The Lean ``ghostDrop D x a c g``."""

    def s(p: Port) -> Port:
        return g if p in (a, c) else p

    return LDiag(
        [n for n in D.nodes if n.id != x],
        list(D.wires),
        [s(p) for p in D.outs],
        [s(p) for p in D.ins],
        D.boxes,
        [LLink(lk.box, s(lk.l1), s(lk.l2), s(lk.e1), s(lk.e2)) for lk in D.links],
    )


def _reduce(
    D: LDiag, ids: set[int], drops: set[int], G: int
) -> tuple[LDiag, list[Ghost] | None, list[Step]]:
    """``D`` with each node of ``drops`` meeting :func:`drop_ok` dropped (moved last first),
    then each node of ``ids`` meeting :func:`ghost_ok` dropped for the ghost ``(G, True, i)`` of
    its slot ``i``; the ghosts (most recent first; None when a node of ``ids`` stays) and the
    steps."""
    steps: list[Step] = []
    left = set(ids)
    for x in [n for n in D.nodes if n.id in drops]:
        found = _drops(D, x)
        if found:
            w, u, v, P, (nodes, wires, outs, ins, links) = found[0]
            steps.append(("drop", x, (*w, u, v, P), -1))
            D = LDiag(nodes, wires, outs, ins, D.boxes, links)
            left.discard(x.id)
    gs: list[Ghost] = []
    for x in [n for n in D.nodes if n.id in left]:
        M = _move_last(D, x)
        legs = [(x.id, True, 0), (x.id, False, 0)]
        for c in legs:
            if c not in M.bnd():
                continue
            a, i = legs[1 - legs.index(c)], M.bnd().index(c)
            g = (x.id, True, i + 1) if M.chain(x.id) else (G, True, i)
            if ghost_ok(M, gs, x, a, c, g, i):
                steps.append(("ghost", x, (a, c, g), i))
                D = ghost_drop(M, x.id, a, c, g)
                gs.insert(0, (g, i, x.dim))
                left.discard(x.id)
                break
    return D, (None if left else gs), steps


def _alias(name: str, D: LDiag, steps: Sequence[Step], same: str) -> str:
    """``name`` as ``D``'s literal, or as ``same`` with no steps."""
    body = literal(D) if steps else same
    return f"def {name} : LDiagram :=\n  {body}\n\n"


def _lghosts(gs: Sequence[str]) -> str:
    return "[" + ", ".join(gs) + "]"


def _chain(
    tag: str,
    start: str,
    end: str,
    gname: str,
    steps: Sequence[Step],
    gs: Sequence[Ghost],
    define: bool = True,
) -> str:
    """Lean source: the ghosts ``gname`` (with ``define``), the ``DropOK`` or ``GhostOK`` of each
    step (``x<tag><j>``, ``ok<tag><j>``) from ``start``, the last diagram equal to ``end``
    (``eq_<tag>``) and ``chain_<tag>``: ``start``'s linked digit sum equals ``end``'s with the
    ghosts when ``zero`` reads the zero phase."""
    out, at, terms = "", start, list[str]()
    held: list[str] = []
    drop_proof = ",\n    ".join(["by decide +kernel"] * 18)
    ghost_proof = ",\n    ".join(["by decide +kernel"] * 23)
    switched = False
    for j, (kind, x, ports, i) in enumerate(steps, 1):
        xn = f"x{tag}{j}"
        out += f"/-- The identity dropped at step {j}. -/\nabbrev {xn} : DNode := {_lnode(x)}\n\n"
        if kind == "drop":
            w0, w1, u, v, P = ports
            wl = f"({_lport(w0)}, {_lport(w1)})"
            out += (
                f"theorem ok{tag}{j} : DropOK ({at}.moveLast {xn}.id) {xn} {wl} {_lport(u)} "
                f"{_lport(v)} {_lport(P)} :=\n  ⟨{drop_proof}⟩\n\n"
            )
            terms.append(f"(evalL_dropAt ok{tag}{j} E hE ks σ)")
            at = f"({at}.dropAt {xn} {wl} {_lport(v)} {_lport(P)})"
            continue
        if not switched:
            terms.append(f"(evalG_nil {at} E ks σ).symm")
            switched = True
        a, c, g = ports
        out += (
            f"theorem ok{tag}{j} : GhostOK ({at}.moveLast {xn}.id) {_lghosts(held)} {xn} "
            f"{_lport(a)} {_lport(c)} {_lport(g)} {i} :=\n  ⟨{ghost_proof}⟩\n\n"
        )
        terms.append(f"(evalG_ghostAt ok{tag}{j} E hE ks σ)")
        held.insert(0, f"({_lport(g)}, {i}, {x.dim})")
        at = f"({at}.ghostAt {xn} {_lport(a)} {_lport(c)} {_lport(g)})"
    if not switched:
        terms.append(f"(evalG_nil {at} E ks σ).symm")
    body = terms[-1]
    for t in reversed(terms[:-1]):
        body = f"{t}.trans ({body})"
    hE = "hE" if steps else "_"
    tail = f".trans (by rw [eq_{tag}])" if steps else ""
    listed = _lghosts([f"({_lport(g)}, {i}, {k})" for g, i, k in gs])
    defined = f"/-- The ghosts of `{end}`. -/\nabbrev {gname} : Ghosts := {listed}\n\n"
    return (
        out
        + (defined if define else "")
        + (f"theorem eq_{tag} : {at} = {end} := by\n  decide +kernel\n\n" if steps else "")
        + f"/-- `{start}`'s digit sum is `{end}`'s with its ghosts. -/\n"
        f"theorem chain_{tag} : ∀ (E : Env), (∀ k, E.phase zero k = 0) →\n"
        "    ∀ (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ),\n"
        f"      {start}.evalL E ks σ = {end}.evalG {gname} E ks σ :=\n"
        f"  fun E {hE} ks σ => ({body}){tail}\n\n"
    )


def _same_up_to_order(A: LDiag, B: LDiag) -> bool:
    """``A`` and ``B`` alike but for the order of their nodes."""
    key = sorted(A.nodes, key=lambda n: n.id) == sorted(B.nodes, key=lambda n: n.id)
    return key and (A.wires, A.outs, A.ins, A.boxes, A.links) == (
        B.wires,
        B.outs,
        B.ins,
        B.boxes,
        B.links,
    )


def _lnode(n: LNode) -> str:
    return f"⟨{n.id}, .{n.kind}, {n.dim}, {n.ph}, {_lnat(n.outs)}, {_lnat(n.ins)}⟩"


def _subst(v: Port, P: Port, p: Port) -> Port:
    return P if p == v else p


Dropped = tuple[list[LNode], list[tuple[Port, Port]], list[Port], list[Port], list[LLink]]
"""A dropped diagram's nodes, wires, outputs, inputs and links."""


def drop_id(D: LDiag, x: int, w: tuple[Port, Port], v: Port, P: Port) -> Dropped:
    """The Lean ``dropId`` of ``D``: nodes, wires, boundary and links (the boxes kept)."""
    wires = list(D.wires)
    wires.remove(w)
    s: Callable[[Port], Port] = lambda p: _subst(v, P, p)
    return (
        [n for n in D.nodes if n.id != x],
        [(s(a), s(b)) for a, b in wires],
        [s(p) for p in D.outs],
        [s(p) for p in D.ins],
        [LLink(lk.box, s(lk.l1), s(lk.l2), s(lk.e1), s(lk.e2)) for lk in D.links],
    )


def drop_ok(D: LDiag, x: LNode, w: tuple[Port, Port], u: Port, v: Port, P: Port) -> bool:
    """The Lean ``DropOK D x w u v P``, field by field."""
    rest = list(D.wires)
    rest.remove(w)
    return (
        all(D.chain(a[0]) == D.chain(b[0]) for a, b in D.wires)
        and len({n.id for n in D.nodes}) == len(D.nodes)
        and D.nodes[-1] == x
        and x.kind == "Z"
        and x.outs == (x.dim,)
        and x.ins == (x.dim,)
        and {u, v} == {(x.id, True, 0), (x.id, False, 0)}
        and w in ((u, P), (P, u))
        and P[0] != x.id
        and P in D.legs()
        and all(u not in e for e in rest)
        and u not in D.bnd()
        and all(u not in (lk.l1, lk.l2, lk.e1, lk.e2) for lk in D.links)
        and all(v not in (lk.l1, lk.l2) for lk in D.links)
    )


def _drops(D: LDiag, x: LNode) -> list[tuple[tuple[Port, Port], Port, Port, Port, Dropped]]:
    """Each ``(w, u, v, P, dropped)`` of ``x`` in ``D`` meeting :func:`drop_ok`."""
    D = LDiag([n for n in D.nodes if n.id != x.id] + [x], D.wires, D.outs, D.ins, D.boxes, D.links)
    out: list[tuple[tuple[Port, Port], Port, Port, Port, Dropped]] = []
    legs = [(x.id, True, 0), (x.id, False, 0)]
    for u in legs:
        v = legs[1 - legs.index(u)]
        for w in D.wires:
            if u in w and w[0] != w[1]:
                P = w[1] if w[0] == u else w[0]
                if drop_ok(D, x, w, u, v, P):
                    out.append((w, u, v, P, drop_id(D, x.id, w, v, P)))
    return out


def _link_ports_on_boundary(d: Diagram) -> list[Port]:
    """The chain link ports of ``d`` on its boundary, in boundary order."""
    ports = {_port(r) for b in d.bang_boxes.values() for pq in b.links for r in pq}
    return [p for p in map(_port, [*d.boundary_outputs, *d.boundary_inputs]) if p in ports]


def export_slide(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the digit sums of ``before`` and ``after`` equal, each a
    linked diagram with each boundary link port the ghost of its slot, read through any layout of
    equal slot data: both, with a phaseless Z(1 -> 1) behind each boundary link port, drop one
    phaseless Z(1 -> 1) and those identities to one diagram with ghosts; None outside that
    form."""
    nid = 1 + max((int(x) for x in [*before.nodes, *after.nodes]), default=0)
    pl, pr = _link_ports_on_boundary(before), _link_ports_on_boundary(after)
    idl = {p: nid + j for j, p in enumerate(pl)}
    idr = {p: nid + len(pl) + j for j, p in enumerate(pr)}
    G = nid + len(pl) + len(pr)
    tables = _Tables()
    try:
        L = linked(before, tables, idl)
        R = linked(after, tables, idr)
        Tl, Tr = linked(before, tables, ghost=G), linked(after, tables, ghost=G)
    except LinkedUnsupported:
        return None
    if L.boxes != R.boxes:
        return None
    zero = _Tables.index(tables.phases, repr(None))
    Dl, gl, sl = _reduce(L, set(idl.values()), set(), G)
    Dr, gr, sr = _reduce(R, set(idr.values()), set(), G)
    if gl is None or gr is None or not (_same_up_to_order(Tl, Dl) and _same_up_to_order(Tr, Dr)):
        return None
    found = None
    for x in L.nodes:
        if x not in R.nodes or x.kind != "Z" or x.ph != zero or x.id in idl.values():
            continue
        Cl, hl, ml = _reduce(L, {x.id, *idl.values()}, {x.id}, G)
        Cr, hr, mr = _reduce(R, {x.id, *idr.values()}, {x.id}, G)
        if hl is not None and (Cl, hl) == (Cr, hr):
            found = (Cl, hl, ml, mr)
            break
    if found is None:
        return None
    C, hc, ml, mr = found
    Gl, Gr = groups(before), groups(after)
    if slot_data(Dl, Gl) != slot_data(Dr, Gr) or not nested(slot_data(Dl, Gl)):
        return None
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        "/-- The diagram before the slide, as a linked diagram, each boundary link port the\n"
        "ghost of its slot. -/\n"
        f"def L : LDiagram :=\n  {literal(Dl)}\n\n"
        "/-- The diagram after the slide, as a linked diagram, each boundary link port the\n"
        "ghost of its slot. -/\n"
        f"def L' : LDiagram :=\n  {literal(Dr)}\n\n"
        "/-- The box groups before the slide, node lists by box. -/\n"
        f"def G : List (ℕ × List (List ℕ)) := {_lgroups(Gl)}\n\n"
        "/-- The box groups after the slide, node lists by box. -/\n"
        f"def G' : List (ℕ × List (List ℕ)) := {_lgroups(Gr)}\n\n"
        f"/-- The phase index of the phaseless nodes. -/\ndef zero : ℕ := {zero}\n\n"
        "/-- `L` with a phaseless Z(1 -> 1) behind each boundary link port. -/\n"
        + _alias("Lb", L, sl, "L")
        + "/-- `L'` with a phaseless Z(1 -> 1) behind each boundary link port. -/\n"
        + _alias("Lk", R, sr, "L'")
        + "/-- The diagram both sides drop to, with ghosts. -/\n"
        f"def C : LDiagram :=\n  {literal(C)}\n\n"
        + _chain("b", "Lb", "L", "gsL", sl, gl)
        + _chain("k", "Lk", "L'", "gsL'", sr, gr)
        + _chain("sb", "Lb", "C", "gsC", ml, hc)
        + _chain("sk", "Lk", "C", "gsC", mr, hc, define=False)
        + "theorem layout : L'.slotData G' = L.slotData G := by\n  decide +kernel\n\n"
        "/-- The digit sums agree when the phaseless index reads zero. -/\n"
        "theorem sums : ∀ (E : Env), (∀ k, E.phase zero k = 0) →\n"
        "    ∀ (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ), L'.evalG gsL' E ks σ = L.evalG gsL E ks σ :=\n"
        "  fun E hE ks σ => (chain_k E hE ks σ).symm.trans ((chain_sk E hE ks σ).trans\n"
        "    ((chain_sb E hE ks σ).symm.trans (chain_b E hE ks σ)))\n\n"
        "/-- The digit sums read at the flat positions of the two sides' slot data agree. -/\n"
        "theorem main : ∀ (E : Env), (∀ k, E.phase zero k = 0) →\n"
        "    ∀ (ks : ℕ → ℕ) (y : ℕ → ℕ),\n"
        "      L'.evalG gsL' E ks (flatAt (L'.slotData G') ks y) =\n"
        "        L.evalG gsL E ks (flatAt (L.slotData G) ks y) :=\n"
        "  fun E hE ks y => evalG_flatAt layout E ks (sums E hE ks) y\n\n"
        "end ArchytasZX.Export\n"
    )


Groups = list[tuple[int, list[list[int]]]]
"""Box groups: per box with groups, its id and each group's nodes."""


def groups(d: Diagram) -> Groups:
    """The box groups of ``d`` in box order."""
    return [
        (int(bid), [sorted(int(n) for n in g) for g in b.groups])
        for bid, b in sorted(d.bang_boxes.items())
        if b.groups
    ]


def _lgroups(gs: Groups) -> str:
    return "[" + ", ".join(f"({b}, [{', '.join(map(_lnat, g))}])" for b, g in gs) + "]"


SlotData = tuple[int, int, list[tuple[int, int, int | None, list[list[int]]]]]
"""Output and input counts, then per box its id, multiplicity index, parent and slot blocks."""


def slot_data(D: LDiag, gs: Groups) -> SlotData:
    """The Lean ``slotData D gs``."""
    bnd = D.bnd()
    table = dict(reversed(gs))
    out = []
    for b in D.boxes:
        g = table.get(b.id, [])
        rest = [n for n in b.nodes if all(n not in h for h in g)]
        parts = [
            [i for i, p in enumerate(bnd) if p[0] in h and p[0] in b.nodes] for h in [*g, rest]
        ]
        out.append((b.id, b.mult, b.parent, [s for s in parts if s]))
    return (len(D.outs), len(D.ins), out)


def nested(sd: SlotData) -> bool:
    """Whether every slot block of a box lies inside one slot block of its parent."""
    blocks = {b[0]: b[3] for b in sd[2]}
    return all(
        b[2] is None or any(set(P) <= set(Q) for Q in blocks.get(b[2], []))
        for b in sd[2]
        for P in b[3]
    )


def anchors(sd: SlotData, i: int) -> tuple[list[int], list[int]]:
    """The Lean ``(anch sd i, chainS sd i)``: per box holding slot ``i`` (outermost first) the
    least slot of its block on ``i``'s side, then ``i``; and the boxes' multiplicity indices."""
    no, _, bs = sd
    anch, chain, parent = [], [], None
    for _ in range(len(bs)):
        box = next((b for b in bs if b[2] == parent and any(i in P for P in b[3])), None)
        if box is None:
            break
        block = next(P for P in box[3] if i in P)
        anch.append(min([i, *(j for j in block if (j < no) == (i < no))]))
        chain.append(box[1])
        parent = box[0]
    return [*anch, i], chain


def _leg_chain(D: LDiag, q: Port) -> list[int]:
    """The Lean ``legChain`` of ``q`` in a diagram without port boxes."""
    own = D.chain(q[0])
    partner = next((w[1] if w[0] == q else w[0] for w in D.wires if q in w), None)
    if partner is None:
        return own
    other = D.chain(partner[0])
    return own + (other[len(own) :] if len(own) < len(other) else [])


def _agree(A: Sequence[int], A2: Sequence[int], B: Sequence[int], B2: Sequence[int]) -> bool:
    """The Lean ``agree A A2 B B2``."""
    if not (A and A2 and B and B2):
        return True
    a, a2, b, b2 = A[0], A2[0], B[0], B2[0]
    return (
        (a < a2) == (b < b2)
        and (a == a2) == (b == b2)
        and (a != a2 or _agree(A[1:], A2[1:], B[1:], B2[1:]))
    )


def _head(A: Sequence[int]) -> int:
    return A[0] if A else 0


def _copy_vs(
    A: Sequence[int], A2: Sequence[int], E: Sequence[int], F: Sequence[int], b: bool
) -> bool:
    """The Lean ``copyVs A A2 E F b``."""
    if _head(A) == _head(A2):
        return b and _head(F) < _head(E)
    return (_head(A) < _head(A2)) == (_head(E) < _head(F)) and _head(E) != _head(F)


def _copy_copy(A: Sequence[int], A2: Sequence[int], E: Sequence[int], E2: Sequence[int]) -> bool:
    """The Lean ``copyCopy A A2 E E2``."""
    if _head(A) == _head(A2):
        return _agree(A[1:], A2[1:], E, E2)
    return (_head(A) < _head(A2)) == (_head(E) < _head(E2)) and _head(E) != _head(E2)


def peel_flat_ok(
    L: LDiag,
    R: LDiag,
    B: Sequence[int],
    cp: dict[int, int],
    mb: int,
    c: int,
    sd: SlotData,
    sd2: SlotData,
) -> bool:
    """The Lean ``PeelFlat L R B cp mb c sd sd2``, field by field."""
    bl, br = L.bnd(), R.bnd()
    n, n2 = len(bl), len(br)
    if sd[0] + sd[1] != n or sd2[0] + sd2[1] != n2:
        return False
    inb = [p[0] in B for p in bl]
    m = [br.index(p) if p in br else n2 for p in bl]
    ci = [br.index(q) if (q := _cpp(cp, p)) in br else n2 for p in bl]
    A = [anchors(sd, i) for i in range(n)]
    A2 = [anchors(sd2, j) for j in range(n2 + 1)]
    for i in range(n):
        if A[i][1] != _leg_chain(L, bl[i]) or m[i] >= n2:
            return False
        if not inb[i] and A2[m[i]][1] != A[i][1]:
            return False
        if inb[i] and (
            ci[i] >= n2 or A[i][1] != [mb, *A2[ci[i]][1]] or A2[m[i]][1] != [c, *A2[ci[i]][1]]
        ):
            return False
    imgs = m + [ci[i] for i in range(n) if inb[i]]
    if len(set(imgs)) != len(imgs) or not set(range(n2)) <= set(imgs):
        return False
    for i in range(n):
        for j in range(n):
            if not _agree(A[i][0], A[j][0], A2[m[i]][0], A2[m[j]][0]):
                return False
            if inb[i] and not _copy_vs(A[i][0], A[j][0], A2[ci[i]][0], A2[m[j]][0], inb[j]):
                return False
            if inb[i] and inb[j] and not _copy_copy(A[i][0], A[j][0], A2[ci[i]][0], A2[ci[j]][0]):
                return False
    return True
