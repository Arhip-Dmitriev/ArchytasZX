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

"""Lean output for a kernel peel over linked diagrams (ArchytasZX/Linked.lean,
ArchytasZX/LinkedPeel.lean). A boxed diagram is written as an ``LDiagram``: its nodes, its wires
without those at chain link ports, its boundary, its node boxes and its links, each link's ends the
partners of its two ports (a link port on the boundary meets a phaseless Z(1 -> 1) holding its
boundary place). ``export_peel`` writes both sides of a peel of one outermost node box,
the copy map ``cp`` from the box's nodes to the peeled copy's, the ``PeelOK`` conditions (checked
here first on a mirror of the Lean definitions, then in Lean by ``decide +kernel``) and ``main``,
the two linked digit sums equal through ``evalL_peel``; None when the peel is outside that form.
``export_slide`` writes both sides of an identity slide, each dropping the identity spider
(``DropOK``, ``dropId``) to one diagram, and ``main`` through ``evalL_slide``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

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


def linked(d: Diagram, tables: _Tables, idents: Mapping[Port, int] | None = None) -> LDiag:
    """``d`` as a linked diagram, each boundary leg ``p`` of ``idents`` behind a phaseless
    Z(1 -> 1) of id ``idents[p]`` (wired to ``p``, its other leg in ``p``'s boundary place); raises
    :class:`LinkedUnsupported` outside plain node boxes with links whose ports are wired."""
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
        ph = _Tables.index(tables.phases, repr(node.phase))
        nodes.append(LNode(int(nid), _KINDS[name], dim, ph, outs, ins))
    wires = [(_port(w.a), _port(w.b)) for w in sorted(d.wires, key=lambda w: w.sort_key())]
    bouts = [_port(r) for r in d.boundary_outputs]
    bins = [_port(r) for r in d.boundary_inputs]
    for bp, iid in sorted((idents or {}).items()):
        if bp not in bouts and bp not in bins:
            raise LinkedUnsupported("an identity end off the boundary")
        side = bouts if bp in bouts else bins
        held = d.nodes[bp[0]]  # type: ignore[index]
        leg = (held.outputs if bp[1] else held.inputs)[bp[2]]
        k = _Tables.index(tables.dims, str(leg.dim))
        nodes.append(LNode(iid, "Z", k, _Tables.index(tables.phases, repr(None)), (k,), (k,)))
        side[side.index(bp)] = (iid, bp[1], 0)
        wires.append((bp, (iid, not bp[1], 0)))
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
    kept = [w for i, w in enumerate(wires) if i not in used]
    return LDiag(nodes, kept, bouts, bins, boxes, links)


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


def export_peel(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the linked digit sums of ``before`` and ``after`` equal,
    ``after`` peeling the last copy of one outermost node box of ``before``; None outside that
    form."""
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
    fresh = sorted({int(x) for x in after.nodes} - {int(x) for x in before.nodes})
    if len(fresh) != len(B):
        return None
    cp = dict(zip(B, fresh, strict=True))
    ends_l, ends_r = _boundary_ends(before, after, int(shifted[0]), cp)
    tables = _Tables()
    try:
        L = linked(before, tables, ends_l)
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
    table = ", ".join(f"({x}, {y})" for x, y in cp.items())
    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    phi = f"peelφ {_lnat(B)} cp (ks {c})"
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        "/-- The diagram before the peel, as a linked diagram. -/\n"
        f"def L : LDiagram :=\n  {literal(L)}\n\n"
        "/-- The diagram after the peel, as a linked diagram. -/\n"
        f"def L' : LDiagram :=\n  {literal(R)}\n\n"
        "/-- Each node of the peeled box to its node in the copy. -/\n"
        f"def cp (x : ℕ) : ℕ := (([{table}] : List (ℕ × ℕ)).lookup x).getD 0\n\n"
        f"theorem ok : PeelOK L L' {box.id} {box.mult} {c} {_lnat(B)} cp :=\n"
        f"  ⟨{proof}⟩\n\n"
        "/-- The linked digit sums agree, the box's last copy read at the copy's digits. -/\n"
        f"theorem main : ∀ (E : Env) (ks : ℕ → ℕ), ks {box.mult} = ks {c} + 1 →\n"
        "    ∀ σ' : ℕ → List ℕ → ℕ,\n"
        f"      L'.evalL E ks σ' = L.evalL E ks (bndMap L.base L'.base ({phi}) σ') :=\n"
        "  fun E ks hk σ' => evalL_peel ok E ks hk σ'\n\n"
        "end ArchytasZX.Export\n"
    )


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


def export_slide(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the linked digit sums of ``before`` and ``after`` equal
    (``evalL_slide``), both dropping one phaseless Z(1 -> 1) to one diagram; None outside that
    form."""
    tables = _Tables()
    try:
        L = linked(before, tables)
        R = linked(after, tables)
    except LinkedUnsupported:
        return None
    if L.boxes != R.boxes:
        return None
    zero = _Tables.index(tables.phases, repr(None))
    found = None
    for x in L.nodes:
        if x not in R.nodes or x.kind != "Z" or x.ph != zero:
            continue
        for wl, ul, vl, Pl, dl in _drops(L, x):
            for wr, ur, vr, Pr, dr in _drops(R, x):
                if dl == dr and found is None:
                    found = (x, (wl, ul, vl, Pl), (wr, ur, vr, Pr))
    if found is None:
        return None
    x, sl, sr = found
    L.nodes = [n for n in L.nodes if n.id != x.id] + [x]
    R.nodes = [n for n in R.nodes if n.id != x.id] + [x]
    proof = ",\n    ".join(["by decide +kernel"] * 18)

    def args(s: tuple[tuple[Port, Port], Port, Port, Port]) -> str:
        w, u, v, P = s
        return f"({_lport(w[0])}, {_lport(w[1])}) {_lport(u)} {_lport(v)} {_lport(P)}"

    def drop(name: str, s: tuple[tuple[Port, Port], Port, Port, Port]) -> str:
        w, _u, v, P = s
        return f"{name}.dropId x.id ({_lport(w[0])}, {_lport(w[1])}) {_lport(v)} {_lport(P)}"

    rhs = drop("L'", sr)
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        "/-- The diagram before the slide, as a linked diagram. -/\n"
        f"def L : LDiagram :=\n  {literal(L)}\n\n"
        "/-- The diagram after the slide, as a linked diagram. -/\n"
        f"def L' : LDiagram :=\n  {literal(R)}\n\n"
        "/-- The identity spider that slides. -/\n"
        f"def x : DNode := ⟨{x.id}, .Z, {x.dim}, {x.ph}, {_lnat(x.outs)}, {_lnat(x.ins)}⟩\n\n"
        f"theorem ok : DropOK L x {args(sl)} :=\n  ⟨{proof}⟩\n\n"
        f"theorem ok' : DropOK L' x {args(sr)} :=\n  ⟨{proof}⟩\n\n"
        f"theorem same : {drop('L', sl)} = {rhs} := by\n"
        "  decide +kernel\n\n"
        "/-- The linked digit sums agree when the identity's phase index reads zero. -/\n"
        f"theorem main : ∀ (E : Env), (∀ k, E.phase {x.ph} k = 0) →\n"
        "    ∀ (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ), L'.evalL E ks σ = L.evalL E ks σ :=\n"
        "  fun E hE ks σ => evalL_slide ok ok' same E hE ks σ\n\n"
        "end ArchytasZX.Export\n"
    )
