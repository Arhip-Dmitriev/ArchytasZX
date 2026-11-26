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
LinkedPeel.lean, LinkedSlide.lean, LinkedForm.lean, LinkedGhost.lean, LinkedSplice.lean,
LinkedFirst.lean, LinkedDropX.lean, LinkedRead.lean). A boxed diagram is written as
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
phaseless phase index ``zero`` (also that of a zero phase vector) reads the zero phase. A peel of a
box with an index is written over ``evalX`` (:func:`linked` with ``readings``): ``PeelX``,
``GhostOKX``, ``evalX_peel_flat`` and ``main`` over ``evalGX``, the copies the engine writes with
literal dimensions written with their template's indices read at the constant copy
(:func:`_read_copies`); its first copy (a reversed chain, ``LinkedFirst.lean``) through
``FirstOK``, ``FirstX`` and ``evalX_peel_first``, the kept copies written with their template's
indices read one copy higher (:func:`_shift_kept`); an identity wired to a node leg there drops
through ``DropOKX`` and ``evalX_dropAt`` (``LinkedDropX.lean``), and one wired to a read leg,
read as that leg (:func:`_read_ends`), over ``evalN`` (``PeelN``, ``DropN``, ``GhostN``,
``LinkedRead.lean``)."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef

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


Cph = tuple[int, int, int, int]
"""A copy-indexed phase: node, box, coefficient, multiplicity index."""

CDim = tuple[Port, int, int, int]
"""A copy-indexed leg dimension: leg, box, coefficient, multiplicity index."""


@dataclass
class LDiag:
    """A linked diagram, mirroring the Lean ``LDiagram`` without aligned wires."""

    nodes: list[LNode]
    wires: list[tuple[Port, Port]]
    outs: list[Port]
    ins: list[Port]
    boxes: list[LBox]
    links: list[LLink]
    cph: list[Cph] = field(default_factory=list)
    cdim: list[CDim] = field(default_factory=list)
    cnd: list[Cph] = field(default_factory=list)

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


_NAME = re.compile(r"[A-Za-z_]\w*")


def _reads(text: str, sym: str) -> bool:
    return sym in _NAME.findall(text)


def _sym(text: str) -> sp.Expr:
    """``text`` as a sympy expression, every name a plain symbol."""
    return sp.sympify(text, locals={n: sp.Symbol(n) for n in _NAME.findall(text)})


def _index_names(d: Diagram) -> set[str]:
    return {b.index for b in d.bang_boxes.values() if b.index is not None}


def _seed(ds: Sequence[Diagram], tables: _Tables) -> None:
    """Number every dimension and phase of ``ds`` reading no box index first."""
    names = set().union(*map(_index_names, ds))
    for d in ds:
        for x in sorted(d.nodes):
            node = d.nodes[x]
            for leg in [*node.outputs, *node.inputs]:
                if not any(_reads(str(leg.dim), s) for s in names):
                    _Tables.index(tables.dims, str(leg.dim))
            if node.phase is None or not (set(map(str, node.phase.free_symbols)) & names):
                _Tables.index(tables.phases, _phase_key(node.phase))
    _Tables.index(tables.dims, "")


def _readings(d: Diagram) -> tuple[list[Cph], list[CDim]]:
    """The box index readings of ``d``, each ``(_, box, 0, 0)``: per node a phase or second
    dimension of a binder or splitter reading one index (``cph``), per leg a dimension reading one
    (``cdim``). Raises :class:`LinkedUnsupported` when a node or leg reads two, a node's first
    dimension reads one, or a node reads the index of a box off its node chain."""
    cph: list[Cph] = []
    cdim: list[CDim] = []

    def depth(b: BangBoxId) -> int:
        n, q = 0, d.bang_boxes[b].parent
        while q is not None:
            n, q = n + 1, d.bang_boxes[q].parent
        return n

    names = _index_names(d)
    for x in sorted(d.nodes):
        node = d.nodes[x]
        kind = node.generator_type.name
        legs = [((int(x), True, i), str(p.dim)) for i, p in enumerate(node.outputs)]
        legs += [((int(x), False, i), str(p.dim)) for i, p in enumerate(node.inputs)]
        first = legs[0][1] if legs else ""
        second = None
        if kind == "B" and len(node.inputs) == 2:
            first, second = str(node.inputs[0].dim), str(node.inputs[1].dim)
        elif kind == "S" and len(node.outputs) == 2:
            second = str(node.outputs[1].dim)
        held = sorted((b for b, box in d.bang_boxes.items() if x in box.node_scope), key=depth)
        for sym in sorted(names):
            phase = node.phase is not None and sym in set(map(str, node.phase.free_symbols))
            sec = second is not None and _reads(second, sym)
            dims = [p for p, text in legs if _reads(text, sym)]
            if not (phase or sec or dims):
                continue
            binders = [b for b in held if d.bang_boxes[b].index == sym]
            if not binders or _reads(first, sym):
                raise LinkedUnsupported("a box index read outside the linked form")
            bid = int(binders[-1])
            if phase or sec:
                if any(q[0] == int(x) for q in cph) or (phase and sec):
                    raise LinkedUnsupported("a node reading two box indices")
                cph.append((int(x), bid, 0, 0))
            if any(q[0] == p for q in cdim for p in dims):
                raise LinkedUnsupported("a leg reading two box indices")
            cdim += [(p, bid, 0, 0) for p in dims]
    return cph, cdim


def _equal(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify(a - b) == 0)


def _value_key(tables: _Tables, value: sp.Expr, names: set[str]) -> str:
    """The dimension key of ``value``: a numbered key reading none of ``names`` of equal value,
    else its text."""
    for key in tables.dims:
        if key and not any(_reads(key, s) for s in names) and _equal(_sym(key), value):
            return key
    return str(value)


def _end_dim(d: Diagram, bp: Port, text: str, tables: _Tables) -> str:
    """The dimension key of the boundary link port ``bp`` of dimension ``text`` at the copy the
    boundary meets: the index of the link's box at its last copy for a first port, at 0 for a
    second. Raises :class:`LinkedUnsupported` when ``text`` reads another box index."""
    read = sorted(s for s in _index_names(d) if _reads(text, s))
    if not read:
        return text
    for b in d.bang_boxes.values():
        for p, q in b.links:
            if bp not in (_port(p), _port(q)):
                continue
            if read != [b.index]:
                raise LinkedUnsupported("a link port reading another box index")
            at = _sym(str(b.multiplicity.to_sympy())) - 1 if bp == _port(p) else sp.Integer(0)
            value = _sym(text).subs(sp.Symbol(read[0]), at)
            return _value_key(tables, value, _index_names(d))
    raise LinkedUnsupported("an identity end reading a box index off a link")


def linked(
    d: Diagram,
    tables: _Tables,
    idents: Mapping[Port, int] | None = None,
    ghost: int | None = None,
    held: Mapping[Port, LBox] | None = None,
    readings: bool = False,
) -> LDiag:
    """``d`` as a linked diagram, each boundary leg ``p`` of ``idents`` behind a phaseless
    Z(1 -> 1) of id ``idents[p]`` (wired to ``p``, its other leg in ``p``'s boundary place), with
    ``ghost`` every other boundary link port written as the ghost ``(ghost, True, i)`` of its slot
    ``i`` (its boundary place and its link end); a boundary link port ``p`` of ``held`` has its
    identity, or with ``ghost`` the ghost ``(x, True, i + 1)``, held as the node ``x`` of the box
    ``held[p]`` (appended after ``d``'s boxes). With ``readings`` a box may have an index, read
    through ``cph`` and ``cdim`` (:func:`_readings`), and an identity takes its leg's dimension at
    the copy the boundary meets (:func:`_end_dim`). Raises :class:`LinkedUnsupported` outside
    plain node boxes with links whose ports are wired."""
    held = held or {}
    cph, cdim = _readings(d) if readings else ([], [])
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
        text = _end_dim(d, bp, str(leg.dim), tables) if readings else str(leg.dim)
        k = _Tables.index(tables.dims, text)
        nodes.append(LNode(iid, "Z", k, _Tables.index(tables.phases, repr(None)), (k,), (k,)))
        side[side.index(bp)] = (iid, bp[1], 0)
        wires.append((bp, (iid, not bp[1], 0)))
    if any(len(b.nodes) != 1 for b in held.values()):
        raise LinkedUnsupported("a held identity box of other than one node")
    if ghost is not None:
        ports = {_port(r) for b in d.bang_boxes.values() for pq in b.links for r in pq}
        for j, bp in enumerate([*bouts, *bins]):
            if bp in ports:
                side, pos = (bouts, j) if j < len(bouts) else (bins, j - len(bouts))
                g = (held[bp].nodes[0], True, j + 1) if bp in held else (ghost, True, j)
                side[pos] = g
                wires.append((bp, g))
    boxes, links, used = [], [], set[int]()
    for bid, b in sorted(d.bang_boxes.items()):
        plain = (
            not b.port_scope
            and b.node_scope
            and not b.cyclic
            and (b.index is None or readings)
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
    out = LDiag(nodes, kept, bouts, bins, boxes, links, cph, cdim)
    for w in kept:
        a, c = ([b.id for b in out.node_boxes(e[0])] for e in w)
        if a[: len(c)] != c and c[: len(a)] != a:
            raise LinkedUnsupported("a wire between boxes neither of which holds the other")
    return out


def _boundary_ends(
    before: Diagram, after: Diagram, bid: int, cp: dict[int, int], first: bool = False
) -> tuple[dict[Port, int], dict[Port, int]]:
    """The :func:`linked` identity ends of both sides of a peel of the box ``bid``: one fresh id
    per link port on ``before``'s boundary, on ``after`` at the same leg, or at its copy's for a
    first port of a link of ``bid`` (with ``first``, a second port)."""
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
                moved = int(b_id) == bid and ref is (q if first else p)
                ends_r[_cpp(cp, port) if moved else port] = nid
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


def flip(D: LDiag, bid: int) -> LDiag:
    """The Lean ``flipLinks``: ``D`` with the links of the box ``bid`` flipped."""
    links = [LLink(lk.box, lk.l2, lk.l1, lk.e2, lk.e1) if lk.box == bid else lk for lk in D.links]
    return LDiag(
        list(D.nodes), list(D.wires), list(D.outs), list(D.ins), list(D.boxes), links, D.cph, D.cdim
    )


def _links_ok(L: LDiag) -> bool:
    """The ``lwf`` field of ``PeelOK`` and ``RevOK``."""
    legs = L.legs()
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
    return True


def rev_ok(L: LDiag, R: LDiag, bid: int, mb: int, B: Sequence[int]) -> bool:
    """The Lean ``RevOK L R bid mb B``, field by field."""
    inB = set(B)
    legs = L.legs()
    ids = [n.id for n in L.nodes]
    if (R.nodes, R.wires, R.outs, R.ins, R.boxes) != (L.nodes, L.wires, L.outs, L.ins, L.boxes):
        return False
    if R.links != flip(L, bid).links or L.cph or R.cph:
        return False
    if any(len(set(xs)) != len(xs) for xs in (ids, legs, L.bnd())):
        return False
    if any(p not in legs for p in L.bnd()):
        return False
    if any(L.chain(w[0][0]) != L.chain(w[1][0]) for w in L.wires):
        return False
    if any(n.id in inB and L.chain(n.id)[:1] != [mb] for n in L.nodes):
        return False
    for w in L.wires:
        if w[0] not in legs or w[1] not in legs or (w[0][0] in inB) != (w[1][0] in inB):
            return False
    if not _links_ok(L):
        return False
    for lk in L.links:
        ins = [p[0] in inB for p in (lk.l1, lk.l2, lk.e1, lk.e2)]
        if lk.box == bid:
            if not (
                ins == [True, True, False, False]
                and L.depth(bid, lk.l1[0]) == 0
                and L.depth(bid, lk.l2[0]) == 0
                and L.mult_of(bid) == mb
            ):
                return False
        elif ins[0]:
            if not (all(ins) and L.depth(lk.box, lk.l1[0]) >= 1):
                return False
        elif any(ins):
            return False
    return True


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


def _find(
    entries: Sequence[tuple[object, int, int, int]], key: object
) -> tuple[int, int, int] | None:
    """The Lean ``(entries.find? (·.1 = key)).map (·.2)``."""
    return next(((y, c, m) for k, y, c, m in entries if k == key), None)


def _rdc(
    L: LDiag,
    R: LDiag,
    bid: int,
    c: int,
    x: int,
    x2: int,
    o: tuple[int, int, int] | None,
    o2: tuple[int, int, int] | None,
) -> bool:
    """The Lean ``rdC L R bid c x x2 o o2``."""
    if o is None or o2 is None:
        return o is None and o2 is None
    if o[0] == bid:
        return o[1] == 0 and o2[1:] == (1, c) and R.depth(o2[0], x2) == len(R.chain(x2))
    return o2[1:] == o[1:] and R.depth(o2[0], x2) + 1 == L.depth(o[0], x)


def peel_x_ok(
    L: LDiag, R: LDiag, bid: int, mb: int, c: int, B: Sequence[int], cp: dict[int, int]
) -> bool:
    """The Lean ``PeelX L R bid mb c B cp``, field by field."""
    if not peel_ok(L, R, bid, mb, c, B, cp):
        return False
    inB = set(B)
    for n in L.nodes:
        o = _find(L.cph, n.id)
        if n.id in inB and L.depth(bid, n.id) != 0:
            return False
        if _find(R.cph, n.id) != o or (
            o is not None and R.depth(o[0], n.id) != L.depth(o[0], n.id)
        ):
            return False
        if n.id in inB and not _rdc(L, R, bid, c, n.id, cp[n.id], o, _find(R.cph, cp[n.id])):
            return False
    for p in L.legs():
        o = _find(L.cdim, p)
        if _find(R.cdim, p) != o or (o is not None and R.depth(o[0], p[0]) != L.depth(o[0], p[0])):
            return False
        cq = _cpp(cp, p)
        if p[0] in inB and not _rdc(L, R, bid, c, p[0], cq[0], o, _find(R.cdim, cq)):
            return False
    return True


def _read_copies(
    L: LDiag,
    R: LDiag,
    tables: _Tables,
    sym: str,
    value: sp.Expr,
    bid: int,
    cp: dict[int, int],
    reading: tuple[int, int],
    phases: Callable[[int, int, sp.Expr], bool] | None = None,
) -> LDiag | None:
    """``R`` with each copy ``cp[x]`` of a node ``x`` of ``L`` reading the index ``sym`` of the
    peeled box ``bid`` written with ``x``'s dimension and phase indices there and the readings
    ``(_, bid, *reading)``, each literal of the copy checked equal to ``x``'s with ``sym`` at
    ``value`` (a phase of a node other than a binder or splitter by ``phases``); None when one
    differs."""
    keys = {v: k for k, v in tables.dims.items()}

    def same(template: int, copy: int) -> bool:
        return _equal(_sym(keys[template]).subs(sp.Symbol(sym), value), _sym(keys[copy]))

    nodes, cph, cdim = list(R.nodes), list(R.cph), list(R.cdim)
    for n in L.nodes:
        if n.id not in cp:
            continue
        j = next(i for i, m in enumerate(nodes) if m.id == cp[n.id])
        m = nodes[j]
        outs, ins, ph = list(m.outs), list(m.ins), m.ph
        for q, y, _, _ in L.cdim:
            if q[0] != n.id or y != bid:
                continue
            side, own = (outs, n.outs) if q[1] else (ins, n.ins)
            if not same(own[q[2]], side[q[2]]):
                return None
            side[q[2]] = own[q[2]]
            cdim.append(((m.id, q[1], q[2]), bid, *reading))
        if any(x == n.id and y == bid for x, y, _, _ in L.cph):
            if not (
                same(n.ph, m.ph)
                if n.kind in ("B", "S")
                else phases is not None and phases(n.id, m.id, value)
            ):
                return None
            ph = n.ph
            cph.append((m.id, bid, *reading))
        nodes[j] = LNode(m.id, m.kind, m.dim, ph, tuple(outs), tuple(ins))
    return LDiag(
        nodes, list(R.wires), list(R.outs), list(R.ins), list(R.boxes), list(R.links), cph, cdim
    )


def _shift_kept(
    L: LDiag,
    R: LDiag,
    tables: _Tables,
    sym: str,
    bid: int,
    B: Sequence[int],
    m1: int,
    phases: Callable[[int, int, sp.Expr], bool] | None = None,
) -> LDiag | None:
    """``R`` with each node ``x`` of ``B`` reading the index ``sym`` of the box ``bid`` written
    with ``x``'s dimension and phase indices in ``L`` and the readings ``(_, bid, 1, m1)``, each
    literal of ``R`` checked equal to ``x``'s with ``sym`` at ``sym + 1`` (a phase of a node other
    than a binder or splitter by ``phases``); None when one differs."""
    keys = {v: k for k, v in tables.dims.items()}
    s = sp.Symbol(sym)

    def same(template: int, kept: int) -> bool:
        return _equal(_sym(keys[template]).subs(s, s + 1), _sym(keys[kept]))

    inB = set(B)
    nodes = list(R.nodes)
    cph = [q for q in R.cph if not (q[0] in inB and q[1] == bid)]
    cdim = [q for q in R.cdim if not (q[0][0] in inB and q[1] == bid)]
    for n in L.nodes:
        if n.id not in inB:
            continue
        j = next(i for i, m in enumerate(nodes) if m.id == n.id)
        m = nodes[j]
        outs, ins, ph = list(m.outs), list(m.ins), m.ph
        for q, y, _, _ in L.cdim:
            if q[0] != n.id or y != bid:
                continue
            side, own = (outs, n.outs) if q[1] else (ins, n.ins)
            if not same(own[q[2]], side[q[2]]):
                return None
            side[q[2]] = own[q[2]]
            cdim.append((q, bid, 1, m1))
        if any(x == n.id and y == bid for x, y, _, _ in L.cph):
            if not (
                same(n.ph, m.ph)
                if n.kind in ("B", "S")
                else phases is not None and phases(n.id, m.id, s + 1)
            ):
                return None
            ph = n.ph
            cph.append((n.id, bid, 1, m1))
        nodes[j] = LNode(m.id, m.kind, m.dim, ph, tuple(outs), tuple(ins))
    return LDiag(
        nodes, list(R.wires), list(R.outs), list(R.ins), list(R.boxes), list(R.links), cph, cdim
    )


def _rds(
    L: LDiag,
    R: LDiag,
    bid: int,
    m1: int,
    x: int,
    o: tuple[int, int, int] | None,
    o2: tuple[int, int, int] | None,
) -> bool:
    """The Lean ``rdS L R bid m1 x o o2``."""
    if o is None or o2 is None:
        return o is None and o2 is None
    if o[0] == bid:
        return L.depth(o[0], x) == 0 and o[1] == 0 and o2[1:] == (1, m1) and R.depth(o2[0], x) == 0
    return o2[1:] == o[1:] and R.depth(o2[0], x) == L.depth(o[0], x) and L.depth(o[0], x) >= 1


def _rdc0(
    L: LDiag,
    R: LDiag,
    bid: int,
    x: int,
    x2: int,
    o: tuple[int, int, int] | None,
    o2: tuple[int, int, int] | None,
) -> bool:
    """The Lean ``rdC0 L R bid x x2 o o2``."""
    if o is None or o2 is None:
        return o is None and o2 is None
    if o[0] == bid:
        return (
            L.depth(o[0], x) == 0
            and o[1] == 0
            and o2[1] == 0
            and R.depth(o2[0], x2) == len(R.chain(x2))
        )
    return o2[1:] == o[1:] and R.depth(o2[0], x2) + 1 == L.depth(o[0], x)


def first_x_ok(L: LDiag, R: LDiag, bid: int, m1: int, B: Sequence[int], cp: dict[int, int]) -> bool:
    """The Lean ``FirstX L R bid m1 B cp``, field by field."""
    inB = set(B)
    for n in L.nodes:
        o, o2 = _find(L.cph, n.id), _find(R.cph, n.id)
        if n.id not in inB:
            if o2 != o or (o is not None and R.depth(o[0], n.id) != L.depth(o[0], n.id)):
                return False
            continue
        x2 = cp.get(n.id, 0)
        if x2 in inB or not _rds(L, R, bid, m1, n.id, o, o2):
            return False
        if not _rdc0(L, R, bid, n.id, x2, o, _find(R.cph, x2)):
            return False
    for p in L.legs():
        o, o2 = _find(L.cdim, p), _find(R.cdim, p)
        if p[0] not in inB:
            if o2 != o or (o is not None and R.depth(o[0], p[0]) != L.depth(o[0], p[0])):
                return False
            continue
        cq = _cpp(cp, p)
        if not _rds(L, R, bid, m1, p[0], o, o2) or not _rdc0(
            L, R, bid, p[0], cq[0], o, _find(R.cdim, cq)
        ):
            return False
    return True


def strip(D: LDiag) -> LDiag:
    """The Lean ``strip``: ``D`` without its readings."""
    return LDiag(
        list(D.nodes), list(D.wires), list(D.outs), list(D.ins), list(D.boxes), list(D.links)
    )


def _apart(*Ds: LDiag, loose: bool = False) -> bool:
    """Whether every dimension and phase index read at a copy lies above every one read plain;
    with ``loose``, whether every value ``Nat.pair k v`` of a read index ``k`` does."""
    read: dict[bool, set[int]] = {True: set(), False: set()}
    plain: dict[bool, set[int]] = {True: set(), False: set()}
    for D in Ds:
        legs = {q[0] for q in D.cdim}
        nodes = {q[0] for q in D.cph}
        dims = {q[0] for q in D.cnd}
        for n in D.nodes:
            (read if n.id in dims else plain)[True].add(n.dim)
            for o, side in ((True, n.outs), (False, n.ins)):
                for i, k in enumerate(side):
                    (read if (n.id, o, i) in legs else plain)[True].add(k)
            (read if n.id in nodes else plain)[n.kind in ("B", "S")].add(n.ph)
    if loose:
        return all(
            not read[t] or max(plain[t], default=-1) < min(k * k + k for k in read[t])
            for t in (True, False)
        )
    return all(not read[t] or max(plain[t], default=-1) < min(read[t]) for t in (True, False))


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
    reads = ""
    if D.cph or D.cdim:
        cph = ", ".join(f"({x}, {y}, {c}, {m})" for x, y, c, m in D.cph)
        cdim = ", ".join(f"({_lport(p)}, {y}, {c}, {m})" for p, y, c, m in D.cdim)
        reads = f",\n    cph := [{cph}],\n    cdim := [{cdim}]"
    if D.cnd:
        reads += (
            ",\n    cnd := [" + ", ".join(f"({x}, {y}, {c}, {m})" for x, y, c, m in D.cnd) + "]"
        )
    return (
        "{ base :=\n      { nodes := [" + nodes + "],\n        wires := [" + wires + "],\n"
        "        outs := [" + ", ".join(map(_lport, D.outs)) + "],\n"
        "        ins := [" + ", ".join(map(_lport, D.ins)) + "],\n"
        "        boxes := [" + boxes + "] },\n    links := [" + links + "]" + reads + " }"
    )


_FIELDS = 21
"""The fields of ``PeelOK``."""

_FLAT_FIELDS = 10
"""The fields of ``PeelFlat``."""

_REV_FIELDS = 16
"""The fields of ``RevOK``."""

_PEELX_FIELDS = 7
"""The fields of ``PeelX`` past ``PeelOK``."""

_GHOSTX_FIELDS = 3
"""The fields of ``GhostOKX`` past ``GhostOK``."""

_FIRSTX_FIELDS = 9
"""The fields of ``FirstX``."""

_DROPX_FIELDS = 4
"""The fields of ``DropOKX`` past ``DropOK``."""


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


def peel_ends(
    before: Diagram, after: Diagram, tables: _Tables, first: bool = False
) -> PeelEnds | None:
    """The :class:`PeelEnds` of ``after`` peeling the last (with ``first``, the first) copy of
    one outermost node box of ``before``: the new nodes of ``after`` are the copies of the box's
    nodes and the phaseless Z(1 -> 1) nodes the peel puts behind boundary link ports; None
    outside that form."""
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
    ends_l, ends_r = _boundary_ends(before, after, int(shifted[0]), cp, first)
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
    ``after`` peeling the last copy of one outermost node box of ``before`` (``PeelOK``), or its
    first copy (``PeelOK`` between the two sides with the box's links flipped); None outside
    that form."""
    if any(b.pair_of is not None for b in before.bang_boxes.values()):
        return _export_pair_peel(before, after, False) or _export_pair_peel(before, after, True)
    return (
        _export_peel(before, after, False)
        or _export_peel(before, after, True)
        or _export_peel(before, after, False, True)
    )


PairSpec = tuple[int, Port, Port, str, bool]
"""A pair box: its id, left and right legs, pair phase key, and whether its first leg (outputs
first) meets the right leg."""


def _strip_pairs(
    d: Diagram, owner: BangBoxId | None = None
) -> tuple[Diagram, BangBoxId, list[PairSpec]] | None:
    """``d`` without its pair boxes and their nodes, its one outermost box (or ``owner``) and the
    pair boxes over it; None unless each pair box (multiplicity 1, no parent, no copy scalar)
    holds one Zp node wired once to its left port and once to its right port, both legs of Z
    nodes of the outermost box off the boundary and wired only there."""
    owners = [b for b, x in sorted(d.bang_boxes.items()) if x.pair_of is None]
    pairs = [(b, x) for b, x in sorted(d.bang_boxes.items()) if x.pair_of is not None]
    if owner is None and len(owners) == 1:
        owner = owners[0]
    if owner not in owners or not pairs:
        return None
    scope = d.bang_boxes[owner].node_scope
    bnd = {*d.boundary_inputs, *d.boundary_outputs}
    out = d.copy()
    specs: list[PairSpec] = []
    for bid, b in pairs:
        if (
            b.pair_of != owner
            or b.parent is not None
            or b.copy_scalar is not None
            or len(b.node_scope) != 1
            or len(b.left_ports) != 1
            or len(b.right_ports) != 1
            or str(b.multiplicity.to_sympy()) != "1"
        ):
            return None
        (z,) = b.node_scope
        node = d.nodes[z]
        (lp,) = b.left_ports
        (rp,) = b.right_ports
        if node.generator_type.name != "Zp" or len(node.outputs) + len(node.inputs) != 2:
            return None
        legs = [PortRef(z, Direction.OUTPUT, i) for i in range(len(node.outputs))]
        legs += [PortRef(z, Direction.INPUT, i) for i in range(len(node.inputs))]
        ends = []
        for leg in legs:
            ws = [w for w in d.wires if leg in w.endpoints()]
            if len(ws) != 1:
                return None
            ends.append(ws[0].b if ws[0].a == leg else ws[0].a)
        if lp == rp or {*ends} != {lp, rp}:
            return None
        for ref in (lp, rp):
            if (
                ref.node_id not in scope
                or d.nodes[ref.node_id].generator_type.name != "Z"
                or ref in bnd
                or len([w for w in d.wires if ref in w.endpoints()]) != 1
            ):
                return None
        specs.append((int(bid), _port(lp), _port(rp), _phase_key(node.phase), ends[0] == rp))
        out.remove_bang_box(bid)
        out.remove_node(z)
    return out, owner, specs


def pair_ok(
    L: LDiag,
    R: LDiag,
    bid: int,
    mb: int,
    c: int,
    B: Sequence[int],
    pairs: Sequence[tuple[Port, Port]],
) -> bool:
    """The Lean ``PairFirst`` and ``PairLast`` fields ``multL``, ``multR`` and ``form`` for pairs
    of default ``LPair`` flags over the box ``bid``, each pair its left and right legs."""
    if L.mult_of(bid) != mb or R.mult_of(bid) != c:
        return False
    legs = set(L.legs())
    return all(q in legs and q[0] in B and _leg_chain(L, q) == [mb] for pair in pairs for q in pair)


def _export_pair_peel(before: Diagram, after: Diagram, first: bool) -> str | None:
    """:func:`export_peel` of a box with pair boxes (``LinkedPair.lean``): ``after`` must be the
    engine's peel of ``before`` (``first`` its first copy); the digit sums are ``evalP`` of the
    peel of ``before`` without its pair boxes (:func:`_strip_pairs`), each pair box a pair of
    ``LPair`` (``PairFirst``, ``PairLast``)."""
    from archytaszx.diagram.bangbox import BangBoxError, peel_one
    from archytaszx.semantics.kernel import _same

    got = _strip_pairs(before)
    if got is None:
        return None
    s0, owner, specs = got
    try:
        want = peel_one(before, owner, first=first).diagram
        r0 = peel_one(s0, owner, first=first).diagram
    except BangBoxError:
        return None
    if not _same(want, after):
        return None
    return _export_peel(s0, r0, first, pairs=specs)


def _export_peel(
    before: Diagram,
    after: Diagram,
    first: bool,
    reads: bool = False,
    pairs: Sequence[PairSpec] | None = None,
) -> str | None:
    """:func:`export_peel` of the last copy, or with ``first`` of the first copy; over ``evalX``
    (``PeelX``, ``GhostOKX``; with ``first`` ``FirstOK`` and ``FirstX``) when a box has an
    index; with ``reads`` over ``evalN`` (``PeelN``, ``DropN``, ``GhostN``), each end identity
    wired to a read leg read as that leg (:func:`_read_ends`); with ``pairs`` over ``evalP``
    with those pair boxes, no boundary link ports."""
    if pairs is not None and reads:
        return None
    tables = _Tables()
    xm = bool(_index_names(before) | _index_names(after))
    if xm:
        _seed([before, after], tables)
    pe = peel_ends(before, after, tables, first)
    if pe is None:
        return None
    shifted, B, fresh, cp = [pe.box], pe.B, pe.fresh, pe.cp
    ends_l, ends_r, held = pe.ends_l, pe.ends_r, pe.held
    try:
        L = linked(before, tables, ends_l, held=held, readings=xm)
        R = linked(after, tables, ends_r, readings=xm)
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
    sym = before.bang_boxes[BangBoxId(pe.box)].index
    after_mult = after.bang_boxes[BangBoxId(pe.box)].multiplicity.to_sympy()
    value = _sym(str(after_mult))
    m1 = _Tables.index(tables.mults, "1") if xm and first else 0
    reading = (0, 0) if first else (1, c)

    def phases(x: int, y: int, at: sp.Expr) -> bool:
        a = before.nodes[NodeId(x)].phase
        b = after.nodes[NodeId(y)].phase
        if a is None or b is None or sym is None:
            return False
        own = {s.name: s for s in after_mult.free_symbols}
        at = at.subs(
            {
                s: own.get(s.name, sp.Symbol(s.name, integer=True, nonnegative=True))
                for s in at.free_symbols
            }
        )
        moved = a.substitute({sym: at})
        return moved == b or repr(moved) == repr(b)

    def read(D: LDiag) -> LDiag | None:
        if not (xm and sym is not None):
            return D
        out = _read_copies(
            L, D, tables, sym, sp.Integer(0) if first else value, box.id, cp, reading, phases
        )
        if out is not None and first:
            out = _shift_kept(L, out, tables, sym, box.id, B, m1, phases)
        return out

    got = read(R)
    if got is None:
        return None
    R = got
    if reads:
        if first or not xm:
            return None
        both = _read_ends(L, R, set(ends_l.values()) & set(ends_r.values()))
        if both is None:
            return None
        L, R = both
    if not first and not peel_ok(L, R, box.id, box.mult, c, B, cp):
        return None
    if xm and not first and not peel_x_ok(L, R, box.id, box.mult, c, B, cp):
        return None
    if reads and not peel_n_ok(L, R, box.id, c, B, cp):
        return None
    Ls, Rs = strip(L), strip(R)
    Lr, Rr = flip(Ls, box.id), flip(Rs, box.id)
    if first and not (
        rev_ok(Ls, Lr, box.id, box.mult, B)
        and rev_ok(Lr, Ls, box.id, box.mult, B)
        and rev_ok(Rs, Rr, box.id, c, B)
        and rev_ok(Rr, Rs, box.id, c, B)
        and peel_ok(Lr, Rr, box.id, box.mult, c, B, cp)
        and all(_cpp(cp, q)[0] not in B for q in L.bnd() if q[0] in B)
    ):
        return None
    if xm and first and not first_x_ok(L, R, box.id, m1, B, cp):
        return None
    G = 1 + max(n.id for n in [*L.nodes, *R.nodes])
    ids_l, ids_r = set(ends_l.values()), set(ends_r.values())
    Dl, gl, sl, rl = _reduce(L, ids_l, set(), G, xm, reads)
    Dr, gr, sr, rr = _reduce(R, ids_r, ids_r, G, xm, reads)
    if gl is None or gr is None:
        return None
    if pairs is not None and (not xm or gl or gr or sl or sr):
        return None
    if pairs is not None and not pair_ok(
        L, R, box.id, box.mult, c, B, [(lp, rp) for _, lp, rp, _, _ in pairs]
    ):
        return None
    try:
        Tl = linked(before, tables, ghost=G, held=held, readings=xm)
        Tr = linked(after, tables, ghost=G, readings=xm)
    except LinkedUnsupported:
        return None
    got = read(Tr)
    if got is None:
        return None
    Tr = got
    if not (_same_up_to_order(Tl, Dl) and _same_up_to_order(Tr, Dr)):
        return None
    if xm and not _apart(L, R, Dl, Dr, loose=reads):
        return None
    Gl, Gr = groups(before), groups(after)
    sl_d, sr_d = slot_data(L, Gl), slot_data(R, Gr)
    if (slot_data(Dl, Gl), slot_data(Dr, Gr)) != (sl_d, sr_d):
        return None
    if not (nested(sl_d) and nested(sr_d)):
        return None
    if not peel_flat_ok(L, R, B, cp, box.mult, c, sl_d, sr_d, first):
        return None
    zero = _Tables.index(tables.phases, repr(None))
    table = ", ".join(f"({x}, {y})" for x, y in cp.items())
    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    if xm and not first:
        proof = "⟨" + proof + "⟩,\n    " + ",\n    ".join(["by decide +kernel"] * _PEELX_FIELDS)
    if reads:
        proof = "⟨" + proof + "⟩,\n    by decide +kernel,\n    by decide +kernel"
    flat = ",\n    ".join(["by decide +kernel"] * (_FLAT_FIELDS + first))
    rev = ",\n    ".join(["by decide +kernel"] * _REV_FIELDS)
    pre, post = ("Lb", "Lk") if not first else ("Lbr", "Lkr")
    sb, sk = ("Lb.strip", "Lk.strip") if xm else ("Lb", "Lk")
    revs = (
        (
            "/-- `Lb` with the peeled box's links flipped. -/\n"
            f"def Lbr : LDiagram :=\n  {literal(Lr)}\n\n"
            "/-- `Lk` with the peeled box's links flipped. -/\n"
            f"def Lkr : LDiagram :=\n  {literal(Rr)}\n\n"
            f"theorem rb : RevOK {sb} Lbr {box.id} {box.mult} {_lnat(B)} :=\n  ⟨{rev}⟩\n\n"
            f"theorem rb' : RevOK Lbr {sb} {box.id} {box.mult} {_lnat(B)} :=\n  ⟨{rev}⟩\n\n"
            f"theorem rk : RevOK {sk} Lkr {box.id} {c} {_lnat(B)} :=\n  ⟨{rev}⟩\n\n"
            f"theorem rk' : RevOK Lkr {sk} {box.id} {c} {_lnat(B)} :=\n  ⟨{rev}⟩\n\n"
        )
        if first
        else ""
    )
    firstx = ",\n    ".join(["by decide +kernel"] * _FIRSTX_FIELDS)
    hcp = (
        (
            f"theorem okx : FirstX Lb Lk {box.id} {m1} {_lnat(B)} cp :=\n  ⟨{firstx}⟩\n\n"
            f"theorem fo : FirstOK Lb Lbr Lk Lkr {box.id} {box.mult} {c} {_lnat(B)} cp :=\n"
            "  ⟨rb, rb', rk, rk', ok⟩\n\n"
        )
        if first and xm
        else (
            "theorem hcp : ∀ i < Lb.base.bnd.length, inB Lb "
            f"{_lnat(B)} i = true →\n    (cpP cp (Lb.bndAt i)).node ∉ {_lnat(B)} := by\n"
            "  decide +kernel\n\n"
        )
        if first
        else ""
    )
    step = (
        "(evalX_peel_first fo okx flat E ks hk hm y)"
        if first and xm
        else "(evalL_peel_first rb rb' rk rk' ok flat hcp E ks hk y)"
        if first
        else f"(eval{'X' if xm else 'L'}_peel_flat ok flat E ks hk y)"
    )
    one, hm = (f" ks {m1} = 1 →", " hm") if first and xm else ("", "")
    kind = "PeelFirst" if first else "PeelFlat"
    ev = "evalGX" if xm else "evalG"
    evb, evk = f"{ev} gsL", f"{ev} gsL'"
    pair_text = ""
    if pairs is not None:
        ph = {key: _Tables.index(tables.phases, key) for _, _, _, key, _ in pairs}
        items = ", ".join(
            f"{{ box := {box.id}, left := {_lport(lp)}, right := {_lport(rp)}, "
            f"ph := {ph[key]}, swap := {'true' if sw else 'false'} }}"
            for _, lp, rp, key, sw in pairs
        )
        if first:
            moved = f"[p.resid {m1}, p.fixLeft (cpP cp p.left) {m1}]"
            form = f"PairFirst Lb Lk ps ps' {box.id} {box.mult} {c} {m1} {_lnat(B)} cp"
            step = "(evalP_peel_first_flat fo okx pf flat E ks hk hm y)"
        else:
            moved = f"[p, p.fixRight (cpP cp p.right) {c}]"
            form = f"PairLast Lb Lk ps ps' {box.id} {box.mult} {c} {_lnat(B)} cp"
            step = "(evalP_peel_flat ok pf flat E ks hk y)"
        pair_text = (
            "/-- The pair boxes before the peel. -/\n"
            f"def ps : List LPair := [{items}]\n\n"
            "/-- The pair boxes after the peel. -/\n"
            f"def ps' : List LPair := ps.flatMap fun p => {moved}\n\n"
            f"theorem pf : {form} :=\n"
            "  ⟨by decide +kernel, by decide +kernel, by decide +kernel, rfl⟩\n\n"
        )
        evb, evk = "evalP ps", "evalP ps'"
    if pairs is not None:
        chb = chk = ""
    elif reads:
        step = "(evalN_peel_flat ok flat E ks hk y)"
        chb, evb = _chain_n("b", "Lb", "L", "gsL", "grL", sl, gl, rl, L)
        chk, evk = _chain_n("k", "Lk", "L'", "gsL'", "grL'", sr, gr, rr, R)
    else:
        chb = _chain("b", "Lb", "L", "gsL", sl, gl, x=xm)
        chk = _chain("k", "Lk", "L'", "gsL'", sr, gr, x=xm)
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
        + chb
        + chk
        + "/-- Each node of the peeled box to its node in the copy. -/\n"
        f"def cp (x : ℕ) : ℕ := (([{table}] : List (ℕ × ℕ)).lookup x).getD 0\n\n"
        + revs
        + f"theorem ok : {'PeelN' if reads else 'PeelX' if xm and not first else 'PeelOK'} "
        f"{pre} {post} {box.id} "
        f"{box.mult} {c} "
        f"{_lnat(B)} cp :=\n  ⟨{proof}⟩\n\n"
        "/-- The box groups before the peel, node lists by box. -/\n"
        f"def G : List (ℕ × List (List ℕ)) := {_lgroups(Gl)}\n\n"
        "/-- The box groups after the peel, node lists by box. -/\n"
        f"def G' : List (ℕ × List (List ℕ)) := {_lgroups(Gr)}\n\n"
        "theorem sdb : Lb.slotData G = L.slotData G := by\n  decide +kernel\n\n"
        "theorem sdk : Lk.slotData G' = L'.slotData G' := by\n  decide +kernel\n\n"
        f"theorem flat : {kind} Lb Lk {_lnat(B)} cp {box.mult} {c} (Lb.slotData G)\n"
        f"    (Lk.slotData G') :=\n  ⟨{flat}⟩\n\n"
        + hcp
        + pair_text
        + "/-- The digit sums read at the flat positions of the two sides' slot data agree. -/\n"
        "theorem main : ∀ (E : Env), (∀ k, E.phase zero k = 0) →\n"
        f"    ∀ (ks : ℕ → ℕ), ks {box.mult} = ks {c} + 1 →{one} ∀ y : ℕ → ℕ,\n"
        f"      L'.{evk} E ks (flatAt (L'.slotData G') ks y) =\n"
        f"        L.{evb} E ks (flatAt (L.slotData G) ks y) := by\n"
        f"  intro E hE ks hk{hm} y\n"
        "  rw [← sdb, ← sdk]\n"
        + (
            f"  exact {step}\n\n"
            if pairs is not None
            else "  exact (chain_k E hE ks _).symm.trans\n"
            f"    ({step}.trans (chain_b E hE ks _))\n\n"
        )
        + "end ArchytasZX.Export\n"
    )


Ghost = tuple[Port, int, int]
"""A ghost: its port, its boundary slot and its dimension index."""

Step = tuple[str, LNode, tuple[Port, ...], int]
"""A step: ``drop`` with the wire's ends, the wired leg, the other leg and the leg it meets, or
``ghost`` with the link-end leg, the boundary leg and the ghost port, and its slot."""


def _read_ends(L: LDiag, R: LDiag, ids: set[int]) -> tuple[LDiag, LDiag] | None:
    """``L`` and ``R`` with each unread end identity of ``ids`` wired in ``R`` to a leg whose
    dimension is read given that leg's dimension index, and that leg's reading on its two legs
    (``cdim``) and on itself (``cnd``); None when there is none."""
    changed: dict[int, tuple[LNode, tuple[int, int, int]]] = {}
    for x in R.nodes:
        if x.id not in ids or x.kind != "Z" or x.outs != (x.dim,) or x.ins != (x.dim,):
            continue
        if _find(R.cnd, x.id) is not None or any(q[0][0] == x.id for q in R.cdim):
            continue
        legs = [(x.id, True, 0), (x.id, False, 0)]
        P = next((b if a in legs else a for a, b in R.wires if (a in legs) != (b in legs)), None)
        owner = None if P is None else next((n for n in R.nodes if n.id == P[0]), None)
        o = None if P is None else _find(R.cdim, P)
        if P is None or owner is None or o is None:
            continue
        k = (owner.outs if P[1] else owner.ins)[P[2]]
        changed[x.id] = (LNode(x.id, x.kind, k, x.ph, (k,), (k,)), o)
    if not changed or any(x not in {n.id for n in L.nodes} for x in changed):
        return None

    def apply(D: LDiag) -> LDiag:
        return LDiag(
            [changed[n.id][0] if n.id in changed else n for n in D.nodes],
            D.wires,
            D.outs,
            D.ins,
            D.boxes,
            D.links,
            D.cph,
            [*D.cdim, *(((x, b, 0), *o) for x, (_, o) in changed.items() for b in (True, False))],
            [*D.cnd, *((x, *o) for x, (_, o) in changed.items())],
        )

    return apply(L), apply(R)


def peel_n_ok(L: LDiag, R: LDiag, bid: int, c: int, B: Sequence[int], cp: dict[int, int]) -> bool:
    """The fields of the Lean ``PeelN L R bid mb c B cp`` past ``PeelX``."""
    for n in L.nodes:
        o = _find(L.cnd, n.id)
        if _find(R.cnd, n.id) != o or (
            o is not None and R.depth(o[0], n.id) != L.depth(o[0], n.id)
        ):
            return False
        if n.id in B and not _rdc(L, R, bid, c, n.id, cp[n.id], o, _find(R.cnd, cp[n.id])):
            return False
    return True


def _lreads(gr: Sequence[CDim]) -> str:
    return "[" + ", ".join(f"({_lport(g)}, {y}, {c}, {m})" for g, y, c, m in gr) + "]"


_DROPN_FIELDS = 6
"""The fields of ``DropN`` past ``DropOK``."""

_GHOSTN_FIELDS = 6
"""The fields of ``GhostN`` past ``GhostOK``."""


def _chain_n(
    tag: str,
    start: str,
    end: str,
    gname: str,
    rname: str,
    steps: Sequence[Step],
    gs: Sequence[Ghost],
    gr: Sequence[CDim],
    D0: LDiag,
) -> tuple[str, str]:
    """:func:`_chain` over ``evalN`` from ``start`` (``DropN`` and ``evalN_dropAt``; while node
    readings remain ``GhostN`` and ``evalGXR_ghostAt``, ``evalN_plain`` and the ``GhostOKX``
    steps once none remain) and the end's digit sum, with the read bounds ``rname`` when it ends
    over ``evalGXR``."""
    out, at, terms = "", start, list[str]()
    held: list[str] = []
    reads: list[CDim] = []
    D = D0
    one = "by decide +kernel"
    drop_proof = (
        "⟨⟨" + ",\n    ".join([one] * 18) + "⟩,\n    " + ",\n    ".join([one] * _DROPN_FIELDS)
    )
    ghost_x = (
        "⟨⟨" + ",\n    ".join([one] * 23) + "⟩,\n    " + ",\n    ".join([one] * _GHOSTX_FIELDS)
    )
    ghost_n = (
        "⟨⟨" + ",\n    ".join([one] * 23) + "⟩,\n    " + ",\n    ".join([one] * _GHOSTN_FIELDS)
    )
    mode = "N"
    for j, (kind, node, ports, i) in enumerate(steps, 1):
        xn = f"x{tag}{j}"
        out += (
            f"/-- The identity dropped at step {j}. -/\nabbrev {xn} : DNode := {_lnode(node)}\n\n"
        )
        M = _move_last(D, node)
        if kind == "drop":
            w0, w1, u, v, P = ports
            wl = f"({_lport(w0)}, {_lport(w1)})"
            out += (
                f"theorem ok{tag}{j} : DropN ({at}.moveLast {xn}.id) {xn} {wl} {_lport(u)} "
                f"{_lport(v)} {_lport(P)} :=\n  {drop_proof}⟩\n\n"
            )
            terms.append(f"(evalN_dropAt ok{tag}{j} E hE ks σ)")
            at = f"({at}.dropAtN {xn} {wl} {_lport(v)} {_lport(P)})"
            nodes, wires, outs, ins, links = drop_id(M, node.id, (w0, w1), v, P)
            D = LDiag(
                nodes,
                wires,
                outs,
                ins,
                M.boxes,
                links,
                M.cph,
                [q for q in M.cdim if q[0][0] != node.id],
                [q for q in M.cnd if q[0] != node.id],
            )
            continue
        if mode == "N" and not D.cnd:
            terms.append(f"(evalN_plain {at} (by decide +kernel) E ks σ)")
            terms.append(f"(evalGX_nil {at} E ks σ).symm")
            mode = "X"
        a, c, g = ports
        if mode == "N":
            out += (
                f"theorem ok{tag}{j} : GhostN ({at}.moveLast {xn}.id) {_lghosts(held)} "
                f"{_lreads(reads)} {xn} {_lport(a)} {_lport(c)} {_lport(g)} {i} :=\n"
                f"  {ghost_n}⟩\n\n"
            )
            terms.append(f"(evalGXR_ghostAt ok{tag}{j} E hE ks σ)")
            at = f"({at}.ghostAtN {xn} {_lport(a)} {_lport(c)} {_lport(g)})"
        else:
            out += (
                f"theorem ok{tag}{j} : GhostOKX ({at}.moveLast {xn}.id) {_lghosts(held)} {xn} "
                f"{_lport(a)} {_lport(c)} {_lport(g)} {i} :=\n  {ghost_x}⟩\n\n"
            )
            terms.append(f"(evalGX_ghostAt ok{tag}{j} E hE ks σ)")
            at = f"({at}.ghostAtX {xn} {_lport(a)} {_lport(c)} {_lport(g)})"
        reads = _gr_add(M, node.id, g, reads)
        D = ghost_drop(M, node.id, a, c, g)
        held.insert(0, f"({_lport(g)}, {i}, {node.dim})")
    if mode == "N" and not steps:
        pass
    elif mode == "N" and not D.cnd and all(k == "drop" for k, *_ in steps):
        terms.append(f"(evalN_plain {at} (by decide +kernel) E ks σ)")
        terms.append(f"(evalGX_nil {at} E ks σ).symm")
        mode = "X"
    body = terms[-1] if terms else "rfl"
    for t in reversed(terms[:-1]):
        body = f"{t}.trans ({body})"
    hE = "hE" if steps else "_"
    tail = f".trans (by rw [eq_{tag}])" if steps else ""
    if steps and mode == "N":
        tail = f".trans (by rw [eq_{tag}]; congr 1 <;> decide +kernel)"
    listed = _lghosts([f"({_lport(g)}, {i}, {k})" for g, i, k in gs])
    defined = f"/-- The ghosts of `{end}`. -/\nabbrev {gname} : Ghosts := {listed}\n\n"
    if mode == "N":
        defined += (
            f"/-- The ghost read bounds of `{end}`. -/\n"
            f"abbrev {rname} : GReads := {_lreads(gr)}\n\n"
        )
        ev = f"evalGXR {gname} {rname}"
    else:
        ev = f"evalGX {gname}"
    text = (
        out
        + defined
        + (f"theorem eq_{tag} : {at} = {end} := by\n  decide +kernel\n\n" if steps else "")
        + f"/-- `{start}`'s digit sum is `{end}`'s with its ghosts. -/\n"
        f"theorem chain_{tag} : ∀ (E : Env), (∀ k, E.phase zero k = 0) →\n"
        "    ∀ (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ),\n"
        f"      {start}.evalN E ks σ = {end}.{ev} E ks σ :=\n"
        f"  fun E {hE} ks σ => ({body}){tail}\n\n"
    )
    return text, ev


def _move_last(D: LDiag, x: LNode) -> LDiag:
    return LDiag(
        [n for n in D.nodes if n.id != x.id] + [x],
        D.wires,
        D.outs,
        D.ins,
        D.boxes,
        D.links,
        D.cph,
        D.cdim,
        D.cnd,
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


def ghost_x_ok(D: LDiag, gs: Sequence[Ghost], x: LNode, a: Port, c: Port, g: Port, i: int) -> bool:
    """The Lean ``GhostOKX D gs x a c g i``, field by field."""
    return (
        ghost_ok(D, gs, x, a, c, g, i)
        and all(q[0] != x.id for q in D.cph)
        and all(q[0][0] != x.id for q in D.cdim)
    )


def ghost_drop(D: LDiag, x: int, a: Port, c: Port, g: Port) -> LDiag:
    """The Lean ``ghostDropX D x a c g`` (``ghostDrop`` keeping the readings)."""

    def s(p: Port) -> Port:
        return g if p in (a, c) else p

    return LDiag(
        [n for n in D.nodes if n.id != x],
        list(D.wires),
        [s(p) for p in D.outs],
        [s(p) for p in D.ins],
        D.boxes,
        [LLink(lk.box, s(lk.l1), s(lk.l2), s(lk.e1), s(lk.e2)) for lk in D.links],
        D.cph,
        [q for q in D.cdim if q[0][0] != x],
        [q for q in D.cnd if q[0] != x],
    )


def ghost_n_ok(
    D: LDiag, gs: Sequence[Ghost], gr: Sequence[CDim], x: LNode, a: Port, c: Port, g: Port, i: int
) -> bool:
    """The Lean ``GhostN D gs gr x a c g i``, field by field."""
    o = _find(D.cnd, x.id)
    return (
        ghost_ok(D, gs, x, a, c, g, i)
        and all(q[0] != x.id for q in D.cph)
        and _find(D.cdim, (x.id, True, 0)) == o
        and _find(D.cdim, (x.id, False, 0)) == o
        and (o is None or D.depth(o[0], g[0]) == D.depth(o[0], x.id))
        and all(r[0] != g for r in gr)
    )


def _gr_add(D: LDiag, x: int, g: Port, gr: list[CDim]) -> list[CDim]:
    """The Lean ``grAdd D x g gr``."""
    o = _find(D.cnd, x)
    return gr if o is None else [(g, *o), *gr]


def _reduce(
    D: LDiag, ids: set[int], drops: set[int], G: int, xm: bool = False, rm: bool = False
) -> tuple[LDiag, list[Ghost] | None, list[Step], list[CDim]]:
    """``D`` with each node of ``drops`` meeting :func:`drop_ok` dropped (moved last first),
    then each node of ``ids`` meeting :func:`ghost_ok` dropped for the ghost ``(G, True, i)`` of
    its slot ``i``; the ghosts (most recent first; None when a node of ``ids`` stays), the steps
    and the ghost read bounds. With ``xm`` the ghosts meet :func:`ghost_x_ok`, and nodes of
    ``drops`` meeting :func:`drop_x_ok` are dropped only when the ghosts alone leave a node of
    ``ids``; with ``rm`` the drops meet :func:`drop_n_ok` and, while node readings remain, the
    ghosts :func:`ghost_n_ok`."""
    if xm and drops:
        alone = _reduce(D, ids, set(), G, True, rm)
        if alone[1] is not None:
            return alone
    steps: list[Step] = []
    left = set(ids)
    for x in [n for n in D.nodes if n.id in drops]:
        found = _drops(D, x, xm, rm)
        if found:
            w, u, v, P, (nodes, wires, outs, ins, links) = found[0]
            steps.append(("drop", x, (*w, u, v, P), -1))
            D = LDiag(
                nodes,
                wires,
                outs,
                ins,
                D.boxes,
                links,
                D.cph,
                [q for q in D.cdim if q[0][0] != x.id],
                [q for q in D.cnd if q[0] != x.id],
            )
            left.discard(x.id)
    gs: list[Ghost] = []
    gr: list[CDim] = []
    for x in [n for n in D.nodes if n.id in left]:
        M = _move_last(D, x)
        legs = [(x.id, True, 0), (x.id, False, 0)]
        for c in legs:
            if c not in M.bnd():
                continue
            a, i = legs[1 - legs.index(c)], M.bnd().index(c)
            g = (x.id, True, i + 1) if M.chain(x.id) else (G, True, i)
            if (
                ghost_n_ok(M, gs, gr, x, a, c, g, i)
                if rm and M.cnd
                else (ghost_x_ok if xm else ghost_ok)(M, gs, x, a, c, g, i)
            ):
                steps.append(("ghost", x, (a, c, g), i))
                gr = _gr_add(M, x.id, g, gr)
                D = ghost_drop(M, x.id, a, c, g)
                gs.insert(0, (g, i, x.dim))
                left.discard(x.id)
                break
    return D, (None if left else gs), steps, gr


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
    x: bool = False,
) -> str:
    """Lean source: the ghosts ``gname`` (with ``define``), the ``DropOK`` or ``GhostOK`` of each
    step (``x<tag><j>``, ``ok<tag><j>``) from ``start``, the last diagram equal to ``end``
    (``eq_<tag>``) and ``chain_<tag>``: ``start``'s linked digit sum equals ``end``'s with the
    ghosts when ``zero`` reads the zero phase; with ``x`` the same over ``evalX`` and ``evalGX``
    (``GhostOKX``, ``ghostAtX``)."""
    out, at, terms = "", start, list[str]()
    held: list[str] = []
    drop_proof = ",\n    ".join(["by decide +kernel"] * 18)
    dk, da, de = "DropOK", "dropAt", "evalL_dropAt"
    ghost_proof = ",\n    ".join(["by decide +kernel"] * 23)
    gk, ga, ge, sx = "GhostOK", "ghostAt", "evalG", "evalL"
    if x:
        ghost_proof = "⟨" + ghost_proof + "⟩,\n    "
        ghost_proof += ",\n    ".join(["by decide +kernel"] * _GHOSTX_FIELDS)
        gk, ga, ge, sx = "GhostOKX", "ghostAtX", "evalGX", "evalX"
        drop_proof = "⟨" + drop_proof + "⟩,\n    "
        drop_proof += ",\n    ".join(["by decide +kernel"] * _DROPX_FIELDS)
        dk, da, de = "DropOKX", "dropAtX", "evalX_dropAt"
    switched = False
    for j, (kind, node, ports, i) in enumerate(steps, 1):
        xn = f"x{tag}{j}"
        out += (
            f"/-- The identity dropped at step {j}. -/\nabbrev {xn} : DNode := {_lnode(node)}\n\n"
        )
        if kind == "drop":
            w0, w1, u, v, P = ports
            wl = f"({_lport(w0)}, {_lport(w1)})"
            out += (
                f"theorem ok{tag}{j} : {dk} ({at}.moveLast {xn}.id) {xn} {wl} {_lport(u)} "
                f"{_lport(v)} {_lport(P)} :=\n  ⟨{drop_proof}⟩\n\n"
            )
            terms.append(f"({de} ok{tag}{j} E hE ks σ)")
            at = f"({at}.{da} {xn} {wl} {_lport(v)} {_lport(P)})"
            continue
        if not switched:
            terms.append(f"({ge}_nil {at} E ks σ).symm")
            switched = True
        a, c, g = ports
        out += (
            f"theorem ok{tag}{j} : {gk} ({at}.moveLast {xn}.id) {_lghosts(held)} {xn} "
            f"{_lport(a)} {_lport(c)} {_lport(g)} {i} :=\n  ⟨{ghost_proof}⟩\n\n"
        )
        terms.append(f"({ge}_ghostAt ok{tag}{j} E hE ks σ)")
        held.insert(0, f"({_lport(g)}, {i}, {node.dim})")
        at = f"({at}.{ga} {xn} {_lport(a)} {_lport(c)} {_lport(g)})"
    if not switched:
        terms.append(f"({ge}_nil {at} E ks σ).symm")
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
        f"      {start}.{sx} E ks σ = {end}.{ge} {gname} E ks σ :=\n"
        f"  fun E {hE} ks σ => ({body}){tail}\n\n"
    )


def _same_up_to_order(A: LDiag, B: LDiag) -> bool:
    """``A`` and ``B`` alike but for the order of their nodes."""
    key = sorted(A.nodes, key=lambda n: n.id) == sorted(B.nodes, key=lambda n: n.id)
    reads = (sorted(A.cph), sorted(A.cdim), sorted(A.cnd)) == (
        sorted(B.cph),
        sorted(B.cdim),
        sorted(B.cnd),
    )
    return (
        key
        and reads
        and (A.wires, A.outs, A.ins, A.boxes, A.links)
        == (
            B.wires,
            B.outs,
            B.ins,
            B.boxes,
            B.links,
        )
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
    if w not in D.wires:
        return False
    rest = list(D.wires)
    rest.remove(w)
    owner = next((n for n in D.nodes if n.id == P[0]), None)
    dims = None if owner is None else (owner.outs if P[1] else owner.ins)
    return (
        not D.cph
        and dims is not None
        and P[2] < len(dims)
        and dims[P[2]] == x.dim
        and all(D.chain(a[0]) == D.chain(b[0]) for a, b in D.wires)
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


def drop_x_ok(D: LDiag, x: LNode, w: tuple[Port, Port], u: Port, v: Port, P: Port) -> bool:
    """The Lean ``DropOKX D x w u v P``, field by field, with ``DropOK``'s ``pdim``."""
    owner = next((n for n in D.nodes if n.id == P[0]), None)
    dims = None if owner is None else (owner.outs if P[1] else owner.ins)
    return (
        drop_ok(strip(D), x, w, u, v, P)
        and dims is not None
        and P[2] < len(dims)
        and dims[P[2]] == x.dim
        and all(q[0] != x.id for q in D.cph)
        and all(q[0][0] != x.id and q[0] != P for q in D.cdim)
    )


def drop_n_ok(D: LDiag, x: LNode, w: tuple[Port, Port], u: Port, v: Port, P: Port) -> bool:
    """The Lean ``DropN D x w u v P``, field by field, with ``DropOK``'s ``pdim``."""
    owner = next((n for n in D.nodes if n.id == P[0]), None)
    dims = None if owner is None else (owner.outs if P[1] else owner.ins)
    o = _find(D.cnd, x.id)
    return (
        drop_ok(strip(D), x, w, u, v, P)
        and dims is not None
        and P[2] < len(dims)
        and dims[P[2]] == x.dim
        and all(q[0] != x.id for q in D.cph)
        and _find(D.cdim, (x.id, True, 0)) == o
        and _find(D.cdim, (x.id, False, 0)) == o
        and _find(D.cdim, P) == o
        and (o is None or D.depth(o[0], P[0]) == D.depth(o[0], x.id))
    )


def _drops(
    D: LDiag, x: LNode, xm: bool = False, rm: bool = False
) -> list[tuple[tuple[Port, Port], Port, Port, Port, Dropped]]:
    """Each ``(w, u, v, P, dropped)`` of ``x`` in ``D`` meeting :func:`drop_ok` (with ``xm``
    :func:`drop_x_ok`, with ``rm`` :func:`drop_n_ok`)."""
    D = _move_last(D, x)
    out: list[tuple[tuple[Port, Port], Port, Port, Port, Dropped]] = []
    legs = [(x.id, True, 0), (x.id, False, 0)]
    for u in legs:
        v = legs[1 - legs.index(u)]
        for w in D.wires:
            if u in w and w[0] != w[1]:
                P = w[1] if w[0] == u else w[0]
                ok = drop_n_ok if rm else drop_x_ok if xm else drop_ok
                if ok(D, x, w, u, v, P):
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
    Dl, gl, sl, _ = _reduce(L, set(idl.values()), set(), G)
    Dr, gr, sr, _ = _reduce(R, set(idr.values()), set(), G)
    if gl is None or gr is None or not (_same_up_to_order(Tl, Dl) and _same_up_to_order(Tr, Dr)):
        return None
    found = None
    for x in L.nodes:
        if x not in R.nodes or x.kind != "Z" or x.ph != zero or x.id in idl.values():
            continue
        Cl, hl, ml, _ = _reduce(L, {x.id, *idl.values()}, {x.id}, G)
        Cr, hr, mr, _ = _reduce(R, {x.id, *idr.values()}, {x.id}, G)
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


def _copy_vs0(
    A: Sequence[int], A2: Sequence[int], E: Sequence[int], F: Sequence[int], b: bool
) -> bool:
    """The Lean ``copyVs0 A A2 E F b``."""
    if _head(A) == _head(A2):
        return b and _head(E) < _head(F)
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
    first: bool = False,
) -> bool:
    """The Lean ``PeelFlat L R B cp mb c sd sd2`` (with ``first`` ``PeelFirst``), field by
    field."""
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
            vs = _copy_vs0 if first else _copy_vs
            if inb[i] and not vs(A[i][0], A[j][0], A2[ci[i]][0], A2[m[j]][0], inb[j]):
                return False
            if first and _head(A[i][0]) == _head(A[j][0]) and inb[i] != inb[j]:
                return False
            if inb[i] and inb[j] and not _copy_copy(A[i][0], A[j][0], A2[ci[i]][0], A2[ci[j]][0]):
                return False
    return True
