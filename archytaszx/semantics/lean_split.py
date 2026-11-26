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

"""Lean output for a kernel power split over linked diagrams (ArchytasZX/LinkedSplit.lean).
Both sides are written as ``LDiagram``s (:func:`lean_linked.linked`), the outside nodes the split
renames given back their ids. Every box index is cleared after its readings are written down: a
node of the split box whose phase reads its index gets the copy-indexed phase ``(x, box, 0, m)``;
a node or leg reading the index of another box ``y`` gets ``(x, y, 0, m)`` in ``cph`` (its phase,
or the second dimension of a binder or splitter) or ``(leg, y, 0, m)`` in ``cdim`` (its
dimension). Plain dimension and phase indices are numbered before every index read at a copy, so
no ``Nat.pair`` of a read index meets a plain one. A wire between two boxes of one alignment group
is an aligned wire ``(a, b, perm)``: ``a`` the end whose node holds fewer of the other end's
boxes, those boxes written as port boxes of ``a``, ``perm`` the places of ``a``'s copy index read
by ``b``'s. A split of base ``b`` is written as ``b - 1`` splits in two: step ``k`` leaves the box
at ``(b - k) h`` copies and moves its top ``h`` copies into the ``k``-th new box. Each step's
copy-indexed phases and leg dimensions follow ``BoxSplitOK``'s ``cph'`` and ``cdim'`` from the
first diagram's, and are checked against the diagram after the split. A leg of a Z or X spider off
the box, wired or aligned into it, is a tail entry ``(leg, appended leg)``. ``export_split``
writes every diagram, copy map, box map and tail list, the ``BoxSplitOK`` conditions of each step
(checked here first on a mirror, then in Lean by ``decide +kernel``) and ``main``, the digit sums
``evalX`` of the first and last diagram equal through ``evalX_split`` at every step; None outside
that form."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.bangbox import _symbol_in, aligned_members, aligned_wires
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.lean_linked import (
    LDiag,
    LinkedUnsupported,
    LLink,
    LNode,
    Port,
    _cpp,
    _lnat,
    _lport,
    _phase_key,
    _Tables,
    _wire_in,
    linked,
)

Cph = tuple[int, int, int, int]
"""A copy-indexed phase: node, box, coefficient, multiplicity index."""

CDim = tuple[Port, int, int, int]
"""A copy-indexed leg dimension: leg, box, coefficient, multiplicity index."""

Align = tuple[Port, Port, tuple[int, ...]]
"""An aligned wire: its two legs and the places of the first's copy index the second reads."""

Tail = tuple[Port, Port]
"""A tail entry: a leg bundled over the split box and the leg appended for the new box."""

_FIELDS = 28
"""The fields of ``BoxSplitOK``."""

_PERM_FIELDS = 22
"""The fields of ``LegPermOK``."""

_PBOX = 1000
"""The least first id of a port box."""


@dataclass(frozen=True)
class PBox:
    """A port box: id, multiplicity index, parent, the leg it holds, the node box it mirrors."""

    id: int
    mult: int
    parent: int | None
    port: Port
    src: int


@dataclass
class XDiag:
    """A linked diagram with copy-indexed phases and leg dimensions, aligned wires and port boxes,
    mirroring the Lean ``LDiagram``."""

    d: LDiag
    cph: list[Cph] = field(default_factory=list)
    cdim: list[CDim] = field(default_factory=list)
    aligned: list[Align] = field(default_factory=list)
    pboxes: list[PBox] = field(default_factory=list)

    def partner(self, q: Port) -> Port | None:
        """The Lean ``Diagram.partner``."""
        w = next((w for w in self.d.wires if q in w), None)
        return None if w is None else (w[1] if w[0] == q else w[0])

    def tail(self, q: Port) -> list[int]:
        """The Lean ``Diagram.tail``."""
        r = self.partner(q)
        if r is not None:
            a, b = self.d.chain(q[0]), self.d.chain(r[0])
            return b[len(a) :] if len(a) < len(b) else []
        boxes = self.d.node_boxes(q[0])
        start = boxes[-1].id if boxes else None
        out: list[int] = []
        for _ in range(len(self.d.boxes) + len(self.pboxes)):
            nxt = next((b for b in self.pboxes if b.port == q and b.parent == start), None)
            if nxt is None:
                break
            out.append(nxt.mult)
            start = nxt.id
        return out

    def leg_chain(self, q: Port) -> list[int]:
        return self.d.chain(q[0]) + self.tail(q)

    def port_dim(self, p: Port) -> int:
        node = next((n for n in self.d.nodes if n.id == p[0]), None)
        if node is None:
            return 0
        side = node.outs if p[1] else node.ins
        return side[p[2]] if p[2] < len(side) else 0

    def mates(self, q: Port) -> list[Port]:
        """The other ends of the wires and aligned wires at ``q``."""
        out = [w[1] if w[0] == q else w[0] for w in self.d.wires if q in w]
        out += [b if a == q else a for a, b, _ in self.aligned if q in (a, b)]
        return out


def _pick(t: Sequence[int], perm: Sequence[int]) -> list[int]:
    return [t[i] if i < len(t) else 0 for i in perm]


def _set(t: Sequence[int], i: int, x: int) -> list[int]:
    out = list(t)
    if i < len(out):
        out[i] = x
    return out


def _ext(T: Sequence[Tail], n: LNode) -> LNode:
    """The Lean ``extT T n``."""

    def get(side: tuple[int, ...], i: int) -> int:
        return side[i] if i < len(side) else 0

    outs = tuple(get(n.outs, e[0][2]) for e in T if e[0][0] == n.id and e[0][1])
    ins = tuple(get(n.ins, e[0][2]) for e in T if e[0][0] == n.id and not e[0][1])
    return replace(n, outs=n.outs + outs, ins=n.ins + ins)


def _copied(
    L: XDiag, B: set[int], cp: dict[int, int], cb: dict[int, int]
) -> tuple[list[Cph], list[CDim]]:
    """The Lean ``cph'`` and ``cdim'`` of a split of ``L``."""

    def up(y: int, x: int) -> int:
        return 1 if L.d.depth(y, x) == 0 else 0

    cph = [
        *L.cph,
        *(
            (cp.get(q[0], 0), cb.get(q[1], 0), q[2] + up(q[1], q[0]), q[3])
            for q in L.cph
            if q[0] in B
        ),
    ]
    cdim = [
        *L.cdim,
        *(
            (_cpp(cp, q[0]), cb.get(q[1], 0), q[2] + up(q[1], q[0][0]), q[3])
            for q in L.cdim
            if q[0][0] in B
        ),
    ]
    return cph, cdim


def split_ok(
    L: XDiag,
    R: XDiag,
    boxes: tuple[int, int, int, int, int],
    B: Sequence[int],
    cp: dict[int, int],
    cb: dict[int, int],
    T: Sequence[Tail],
) -> bool:
    """The Lean ``BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T``, field by field."""
    bid, bid2, mb, c1, c2 = boxes
    inB = set(B)
    keys = [e[0] for e in T]
    D, E = L.d, R.d

    def sl(p: Port) -> bool:
        return p[0] in inB or p in keys

    def cq(p: Port) -> Port:
        if p[0] in inB:
            return _cpp(cp, p)
        return next((e[1] for e in T if e[0] == p), p)

    def spos(p: Port) -> int:
        return 0 if p[0] in inB else len(D.chain(p[0]))

    legs, legs_r = D.legs(), E.legs()
    ids_l = {n.id for n in D.nodes}
    for qc in L.cph:
        if qc[0] not in ids_l or E.depth(qc[1], qc[0]) != D.depth(qc[1], qc[0]):
            return False
        dq = D.depth(qc[1], qc[0])
        if qc[0] in inB and (
            E.depth(cb.get(qc[1], 0), cp.get(qc[0], 0)) != dq or (dq == 0 and qc[3] != c1)
        ):
            return False
    for qd in L.cdim:
        leg = qd[0]
        dq = D.depth(qd[1], leg[0])
        if leg not in legs or leg in keys or E.depth(qd[1], leg[0]) != dq:
            return False
        if leg[0] in inB and (
            E.depth(cb.get(qd[1], 0), cp.get(leg[0], 0)) != dq or (dq == 0 and qd[3] != c1)
        ):
            return False
    if (list(R.cph), list(R.cdim)) != _copied(L, inB, cp, cb):
        return False
    copies = [LNode(cp.get(n.id, 0), n.kind, n.dim, n.ph, n.outs, n.ins) for n in D.nodes]
    want = [_ext(T, n) for n in D.nodes]
    if E.nodes != want + [c for n, c in zip(D.nodes, copies, strict=True) if n.id in inB]:
        return False
    ids = [n.id for n in E.nodes]
    for xs in (ids, legs, legs_r, E.bnd()):
        if len(set(xs)) != len(xs):
            return False
    kinds = {n.id: n.kind for n in D.nodes}
    for n in D.nodes:
        ch = D.chain(n.id)
        if n.id in inB:
            x = cp.get(n.id, 0)
            if not (
                ch[:1] == [mb]
                and E.chain(n.id) == [c1, *ch[1:]]
                and E.chain(x) == [c2, *ch[1:]]
                and D.depth(bid, n.id) == 0
                and E.depth(bid, n.id) == 0
                and E.depth(bid2, x) == 0
            ):
                return False
        elif E.chain(n.id) != ch:
            return False
    for tp, tp2 in T:
        if not (
            tp in legs and tp[0] not in inB and tp2[:2] == tp[:2] and kinds.get(tp[0]) in ("Z", "X")
        ):
            return False
    images = []
    for p in legs:
        lc = L.leg_chain(p)
        if p not in legs_r or R.port_dim(p) != L.port_dim(p):
            return False
        if not sl(p):
            if R.leg_chain(p) != lc:
                return False
            continue
        img, s = cq(p), spos(p)
        if not (
            s < len(lc)
            and lc[s] == mb
            and R.leg_chain(p) == _set(lc, s, c1)
            and img in legs_r
            and R.leg_chain(img) == _set(lc, s, c2)
            and R.port_dim(img) == L.port_dim(p)
            and img not in legs
        ):
            return False
        images.append(img)
    if len(set(images)) != len(images) or any(
        img not in legs and img not in images for img in legs_r
    ):
        return False
    for w in D.wires:
        if not (
            w[0] in legs
            and w[1] in legs
            and sl(w[0]) == sl(w[1])
            and L.leg_chain(w[1]) == L.leg_chain(w[0])
            and _wire_in(E.wires, w)
        ):
            return False
        if sl(w[0]) and not (
            spos(w[0]) == 0 and spos(w[1]) == 0 and _wire_in(E.wires, (cq(w[0]), cq(w[1])))
        ):
            return False
    for w in E.wires:
        if not _wire_in(D.wires, w) and not any(
            sl(w0[0]) and _wire_in([(cq(w0[0]), cq(w0[1]))], w) for w0 in D.wires
        ):
            return False

    def same(lk: LLink, x: LLink) -> bool:
        return (
            (x.l1, x.l2, x.e1, x.e2) == (lk.l1, lk.l2, lk.e1, lk.e2)
            and E.depth(x.box, lk.l1[0]) == D.depth(lk.box, lk.l1[0])
            and E.mult_of(x.box) == D.mult_of(lk.box)
        )

    def copied(lk: LLink, x: LLink) -> bool:
        return (
            (x.l1, x.l2, x.e1, x.e2) == tuple(map(cq, (lk.l1, lk.l2, lk.e1, lk.e2)))
            and E.depth(x.box, cq(lk.l1)[0]) == D.depth(lk.box, lk.l1[0])
            and E.mult_of(x.box) == D.mult_of(lk.box)
        )

    for lk in D.links:
        ch = L.leg_chain(lk.l1)
        dp = D.depth(lk.box, lk.l1[0])
        rest = ch[:dp] + ch[dp + 1 :]
        four = (lk.l1, lk.l2, lk.e1, lk.e2)
        if not (
            all(p in legs for p in four)
            and L.leg_chain(lk.l2) == ch
            and dp < len(ch)
            and ch[dp] == D.mult_of(lk.box)
            and L.leg_chain(lk.e1) == rest
            and L.leg_chain(lk.e2) == rest
        ):
            return False
        ins = [sl(p) for p in four]
        if ins[0]:
            if not (
                all(ins)
                and dp >= 1
                and all(spos(p) == 0 for p in four)
                and any(same(lk, x) for x in E.links)
                and any(copied(lk, x) for x in E.links)
            ):
                return False
        elif any(ins) or not any(same(lk, x) for x in E.links):
            return False
    for lx in E.links:
        if not any(same(lk, lx) for lk in D.links) and not any(
            sl(lk.l1) and copied(lk, lx) for lk in D.links
        ):
            return False
    for a, b, perm in L.aligned:
        lc = L.leg_chain(a)
        if not (
            a in legs
            and b in legs
            and sl(a) == sl(b)
            and all(i < len(lc) for i in perm)
            and L.leg_chain(b) == _pick(lc, perm)
            and (a, b, perm) in R.aligned
        ):
            return False
        if sl(a) and not (
            spos(b) < len(perm)
            and all((perm[i] == spos(a)) == (i == spos(b)) for i in range(len(perm)))
            and (cq(a), cq(b), perm) in R.aligned
        ):
            return False
    for wa in R.aligned:
        if wa not in L.aligned and not any(
            sl(a) and wa == (cq(a), cq(b), perm) for a, b, perm in L.aligned
        ):
            return False
    for p in D.bnd():
        if p not in legs or p not in E.bnd() or (sl(p) and cq(p) not in E.bnd()):
            return False
    return all(q in D.bnd() or any(sl(p) and q == cq(p) for p in D.bnd()) for q in E.bnd())


_NAME = re.compile(r"[A-Za-z_]\w*")


def _reads(text: str, sym: str) -> bool:
    return sym in _NAME.findall(text)


def _legs(d: Diagram, x: NodeId) -> list[tuple[Port, str]]:
    node = d.nodes[x]
    return [((int(x), True, i), str(p.dim)) for i, p in enumerate(node.outputs)] + [
        ((int(x), False, i), str(p.dim)) for i, p in enumerate(node.inputs)
    ]


def _second(d: Diagram, x: NodeId) -> str | None:
    """The second dimension of a binder or splitter."""
    node = d.nodes[x]
    name = node.generator_type.name
    if name == "B" and len(node.inputs) == 2:
        return str(node.inputs[1].dim)
    if name == "S" and len(node.outputs) == 2:
        return str(node.outputs[1].dim)
    return None


def _seed(ds: Sequence[Diagram], tables: _Tables, names: set[str]) -> None:
    """Number every dimension and phase reading none of ``names`` first."""
    for d in ds:
        for x in sorted(d.nodes):
            node = d.nodes[x]
            for _, text in _legs(d, x):
                if not any(_reads(text, s) for s in names):
                    _Tables.index(tables.dims, text)
            if node.phase is None or not (set(map(str, node.phase.free_symbols)) & names):
                _Tables.index(tables.phases, _phase_key(node.phase))
    _Tables.index(tables.dims, "")


@dataclass
class _Read:
    """The box index readings of one diagram off the split boxes."""

    cph: list[Cph] = field(default_factory=list)
    cdim: list[CDim] = field(default_factory=list)


def _readings(d: Diagram, skip: set[int], tables: _Tables) -> _Read | None:
    """The readings of every box index but those of ``skip``: per node a phase or second
    dimension reading one index (``cph``), per leg a dimension reading one (``cdim``); None when a
    node reads two, a node's first dimension reads one, or a node reads the index of a box off
    its node chain."""
    out = _Read()
    seen: set[int] = set()

    def depth(b: BangBoxId) -> int:
        n, p = 0, d.bang_boxes[b].parent
        while p is not None:
            n, p = n + 1, d.bang_boxes[p].parent
        return n

    names = {b.index for b in d.bang_boxes.values() if b.index is not None}
    for x in sorted(d.nodes):
        node = d.nodes[x]
        legs = _legs(d, x)
        second = _second(d, x)
        held = sorted((b for b, box in d.bang_boxes.items() if x in box.node_scope), key=depth)
        for sym in sorted(names):
            phase = node.phase is not None and sym in set(map(str, node.phase.free_symbols))
            sec = second is not None and _reads(second, sym)
            dims = [p for p, text in legs if _reads(text, sym)]
            if not (phase or sec or dims):
                continue
            binders = [b for b in held if d.bang_boxes[b].index == sym]
            if not binders:
                return None
            bid = binders[-1]
            if int(bid) in skip:
                continue
            box = d.bang_boxes[bid]
            m = _Tables.index(tables.mults, str(sp.expand(box.multiplicity.to_sympy())))
            first = legs[0][1] if legs else ""
            if node.generator_type.name == "B" and len(node.inputs) == 2:
                first = str(node.inputs[0].dim)
            if _reads(first, sym):
                return None
            if phase or sec:
                if int(x) in seen or (phase and sec):
                    return None
                seen.add(int(x))
                out.cph.append((int(x), int(bid), 0, m))
            out.cdim += [(p, int(bid), 0, m) for p in dims]
    return out


def _strip(d: Diagram) -> tuple[Diagram, list[tuple[Port, Port, list[int]]]]:
    """``d`` with every box index and alignment cleared and every aligned wire removed, and the
    removed wires with the alignment group of each."""
    out = d.copy()
    roots = sorted({b.align for b in d.bang_boxes.values() if b.align is not None})
    groups = [aligned_members(d, r) for r in roots if r in d.bang_boxes]
    gone = []
    for w in sorted(aligned_wires(d), key=lambda w: w.sort_key()):
        held = [
            g
            for g in groups
            if all(
                any(x in d.bang_boxes[m].node_scope for m in g) for x in (w.a.node_id, w.b.node_id)
            )
        ]
        out.remove_wire(w.a, w.b)
        gone.append((_port(w.a), _port(w.b), sorted(int(m) for m in held[0]) if held else []))
    for bid in d.bang_boxes:
        out.set_bang_box_extension(bid, index=None, align=None)
    return out, gone


def _port(ref: PortRef) -> Port:
    return (int(ref.node_id), ref.direction is Direction.OUTPUT, ref.index)


def _aligned(
    D: LDiag, gone: Sequence[tuple[Port, Port, list[int]]], base: int
) -> tuple[list[Align], list[PBox]] | None:
    """Each removed aligned wire as an aligned wire of ``D`` with its port boxes; None outside
    two nodes held directly by member boxes of one count, one end's boxes a part of the other's.
    Port box ids start at ``base`` or past the largest node box id."""
    aligned: list[Align] = []
    pboxes: list[PBox] = []
    nid = max(base, 1 + max((b.id for b in D.boxes), default=-1))
    for p, q, members in gone:
        ends = []
        for e in (p, q):
            bs = D.node_boxes(e[0])
            if not bs or bs[-1].id not in members:
                return None
            ends.append(bs)
        al, be = ends
        if al[-1].id == be[-1].id or al[-1].mult != be[-1].mult:
            return None
        k = 0
        while k < min(len(al), len(be)) - 1 and al[k].id == be[k].id:
            k += 1
        da, db = al[k:-1], be[k:-1]
        if da and db:
            return None
        if da:
            p, q, al, be, da, db = q, p, be, al, db, da
        n = len(be)
        perm = [i if i < k else (i + 1 if i < n - 1 else k) for i in range(n)]
        parent = al[-1].id
        for b in db:
            pboxes.append(PBox(nid, b.mult, parent, p, b.id))
            parent = nid
            nid += 1
        aligned.append((p, q, tuple(perm)))
    return aligned, pboxes


def _xlinked(d: Diagram, tables: _Tables, skip: set[int]) -> XDiag:
    """``d`` as an :class:`XDiag`, the indices of the boxes ``skip`` cleared unread; raises
    :class:`LinkedUnsupported` outside that form."""
    reads = _readings(d, skip, tables)
    if reads is None:
        raise LinkedUnsupported("a box index read outside the linked form")
    stripped, gone = _strip(d)
    D = linked(stripped, tables)
    found = _aligned(D, gone, _PBOX)
    if found is None:
        raise LinkedUnsupported("an aligned wire outside the linked form")
    return XDiag(D, reads.cph, reads.cdim, found[0], found[1])


def _split_reads(d: Diagram, bid: BangBoxId) -> bool:
    """A leg dimension or second dimension of a node of ``bid`` reads its index."""
    box = d.bang_boxes[bid]
    if box.index is None:
        return False
    for x in box.node_scope:
        texts = [t for _, t in _legs(d, x)]
        if any(_reads(t, box.index) for t in texts):
            return True
    return False


def _moved(X: XDiag, f: dict[Port, Port]) -> XDiag:
    """``X`` with every leg reference ``p`` replaced by ``f[p]``."""

    def g(p: Port) -> Port:
        return f.get(p, p)

    D = X.d
    return XDiag(
        LDiag(
            list(D.nodes),
            [(g(a), g(b)) for a, b in D.wires],
            [g(p) for p in D.outs],
            [g(p) for p in D.ins],
            list(D.boxes),
            [LLink(x.box, g(x.l1), g(x.l2), g(x.e1), g(x.e2)) for x in D.links],
        ),
        list(X.cph),
        [(g(q[0]), *q[1:]) for q in X.cdim],
        [(g(a), g(b), perm) for a, b, perm in X.aligned],
        [replace(b, port=g(b.port)) for b in X.pboxes],
    )


def _appended(R: XDiag, fixed: set[int], copies: set[int]) -> tuple[XDiag, dict[Port, Port]]:
    """``R`` with each leg of a node off ``fixed`` wired or aligned to a node of ``copies`` moved
    after the node's other legs on its side, keeping their orders, and the map of its legs to
    ``R``'s."""
    rn: dict[Port, Port] = {}
    for n in R.d.nodes:
        if n.id in fixed:
            continue
        for o, side in ((True, n.outs), (False, n.ins)):
            ports = [(n.id, o, i) for i in range(len(side))]
            new = [p for p in ports if any(m[0] in copies for m in R.mates(p))]
            order = [p for p in ports if p not in new] + new
            rn.update({(n.id, o, i): p for i, p in enumerate(order) if p != (n.id, o, i)})
    back = {q: p for p, q in rn.items()}
    return _moved(R, back), rn


def perm_ok(A: XDiag, R: XDiag, rn: dict[Port, Port]) -> bool:
    """The Lean ``LegPermOK A R rn``, field by field."""
    D, E = A.d, R.d

    def f(p: Port) -> Port:
        return rn.get(p, p)

    legs = D.legs()
    ids = [n.id for n in D.nodes]
    if E.nodes != D.nodes or len(set(ids)) != len(ids) or len(set(legs)) != len(legs):
        return False
    if len(set(E.bnd())) != len(E.bnd()) or sorted(map(f, legs)) != sorted(legs):
        return False
    kinds = {n.id: n.kind for n in D.nodes}
    for p in legs:
        q = f(p)
        if q[:2] != p[:2] or (q != p and kinds.get(p[0]) not in ("Z", "X")):
            return False
        if R.leg_chain(q) != A.leg_chain(p) or R.port_dim(q) != A.port_dim(p):
            return False
    if any(E.chain(x) != D.chain(x) for x in ids):
        return False
    if E.wires != [(f(a), f(b)) for a, b in D.wires]:
        return False
    if E.outs != [f(p) for p in D.outs] or E.ins != [f(p) for p in D.ins]:
        return False
    if E.links != [LLink(x.box, f(x.l1), f(x.l2), f(x.e1), f(x.e2)) for x in D.links]:
        return False
    for lk in D.links:
        ch = A.leg_chain(lk.l1)
        dp = D.depth(lk.box, lk.l1[0])
        rest = ch[:dp] + ch[dp + 1 :]
        if not (
            E.depth(lk.box, f(lk.l1)[0]) == dp
            and E.mult_of(lk.box) == D.mult_of(lk.box)
            and all(p in legs for p in (lk.l1, lk.l2, lk.e1, lk.e2))
            and A.leg_chain(lk.l2) == ch
            and dp < len(ch)
            and ch[dp] == D.mult_of(lk.box)
            and A.leg_chain(lk.e1) == rest
            and A.leg_chain(lk.e2) == rest
        ):
            return False
    for w in D.wires:
        if w[0] not in legs or w[1] not in legs or A.leg_chain(w[1]) != A.leg_chain(w[0]):
            return False
    if R.cph != A.cph or any(E.depth(q[1], q[0]) != D.depth(q[1], q[0]) for q in A.cph):
        return False
    if R.cdim != [(f(q[0]), *q[1:]) for q in A.cdim]:
        return False
    if any(q[0] not in legs or E.depth(q[1], q[0][0]) != D.depth(q[1], q[0][0]) for q in A.cdim):
        return False
    if R.aligned != [(f(a), f(b), perm) for a, b, perm in A.aligned]:
        return False
    for a, b, perm in A.aligned:
        lc = A.leg_chain(a)
        if not (
            a in legs
            and b in legs
            and all(i < len(lc) for i in perm)
            and A.leg_chain(b) == _pick(lc, perm)
        ):
            return False
    return all(p in legs for p in D.bnd())


def _renamed(L: XDiag, R: XDiag, fixed: set[int]) -> tuple[XDiag, dict[int, int]] | None:
    """``R`` with each node off ``fixed`` and off ``L`` given the id of the one node of ``L`` off
    ``R`` with its generator, phase, leading legs, boundary places and wiring, and the map of the
    ids; None when some node has no such match."""
    ids_l = {n.id for n in L.d.nodes}
    ids_r = {n.id for n in R.d.nodes}
    gone = sorted(ids_l - ids_r)
    new = sorted(ids_r - ids_l - fixed)
    if len(gone) != len(new):
        return None
    by_l = {n.id: n for n in L.d.nodes}
    by_r = {n.id: n for n in R.d.nodes}
    match: dict[int, int] = {}

    def fits(x: int, y: int) -> bool:
        nx, ny = by_l[x], by_r[y]
        if (nx.kind, nx.ph, nx.dim) != (ny.kind, ny.ph, ny.dim):
            return False
        if ny.outs[: len(nx.outs)] != nx.outs or ny.ins[: len(nx.ins)] != nx.ins:
            return False
        legs = [(True, i) for i in range(len(nx.outs))] + [(False, i) for i in range(len(nx.ins))]
        for o, i in legs:
            p, q = (x, o, i), (y, o, i)
            if (p in L.d.outs) != (q in R.d.outs) or (p in L.d.ins) != (q in R.d.ins):
                return False
            ml, mr = L.mates(p), R.mates(q)
            if len(ml) != len(mr) or len(ml) > 1:
                return False
            if not ml:
                continue
            pl, pr = ml[0], mr[0]
            if pl[0] not in gone:
                if pr != pl:
                    return False
            elif pl[0] in match:
                if pr != (match[pl[0]], pl[1], pl[2]):
                    return False
            elif pr[0] not in new:
                return False
        return True

    while len(match) < len(gone):
        free = [y for y in new if y not in match.values()]
        progress = False
        for x in gone:
            if x in match:
                continue
            cands = [y for y in free if fits(x, y)]
            if len(cands) == 1:
                match[x] = cands[0]
                free.remove(cands[0])
                progress = True
        if not progress:
            return None
    back = {y: x for x, y in match.items()}

    def f(p: Port) -> Port:
        return (back.get(p[0], p[0]), p[1], p[2])

    D = R.d
    return XDiag(
        LDiag(
            [replace(n, id=back.get(n.id, n.id)) for n in D.nodes],
            [(f(a), f(b)) for a, b in D.wires],
            [f(p) for p in D.outs],
            [f(p) for p in D.ins],
            [replace(b, nodes=tuple(sorted(back.get(x, x) for x in b.nodes))) for b in D.boxes],
            [LLink(x.box, f(x.l1), f(x.l2), f(x.e1), f(x.e2)) for x in D.links],
        ),
        [(back.get(q[0], q[0]), *q[1:]) for q in R.cph],
        [(f(q[0]), *q[1:]) for q in R.cdim],
        [(f(a), f(b), perm) for a, b, perm in R.aligned],
        [replace(b, port=f(b.port)) for b in R.pboxes],
    ), back


def _below(R: LDiag, tops: Sequence[int]) -> set[int]:
    """The boxes of ``R`` in or under the boxes ``tops``."""
    out = set(tops)
    grew = True
    while grew:
        more = {b.id for b in R.boxes if b.parent is not None and b.parent in out} - out
        out |= more
        grew = bool(more)
    return out


def _stage(
    L: XDiag,
    R: XDiag,
    bid: int,
    mult: int,
    keep: set[int],
    drop_boxes: set[int],
    drop_nodes: set[int],
) -> XDiag | None:
    """``R`` with the nodes ``drop_nodes`` and boxes ``drop_boxes`` left out, ``bid`` at the
    multiplicity index ``mult``, and the legs appended to the nodes of ``L`` kept only when wired
    into the nodes ``keep`` (a leading run of the appended legs); None otherwise."""
    lens = {n.id: (len(n.outs), len(n.ins)) for n in L.d.nodes}
    nodes = []
    for n in R.d.nodes:
        if n.id in drop_nodes:
            continue
        if n.id in lens:
            sizes = []
            for o, side in ((True, n.outs), (False, n.ins)):
                base = lens[n.id][0 if o else 1]
                held = [
                    any(m[0] in keep for m in R.mates((n.id, o, i))) for i in range(base, len(side))
                ]
                count = sum(held)
                if held != [True] * count + [False] * (len(held) - count):
                    return None
                sizes.append(base + count)
            n = replace(n, outs=n.outs[: sizes[0]], ins=n.ins[: sizes[1]])
        nodes.append(n)
    size = {n.id: (len(n.outs), len(n.ins)) for n in nodes}

    def live(p: Port) -> bool:
        return p[0] in size and p[2] < size[p[0]][0 if p[1] else 1]

    D = R.d
    return XDiag(
        LDiag(
            nodes,
            [w for w in D.wires if live(w[0]) and live(w[1])],
            [p for p in D.outs if live(p)],
            [p for p in D.ins if live(p)],
            [
                replace(b, mult=mult) if b.id == bid else b
                for b in D.boxes
                if b.id not in drop_boxes
            ],
            [x for x in D.links if x.box not in drop_boxes],
        ),
        aligned=[w for w in R.aligned if live(w[0]) and live(w[1])],
        pboxes=[
            replace(b, mult=mult) if b.src == bid else b
            for b in R.pboxes
            if live(b.port) and b.src not in drop_boxes
        ],
    )


def _tails(prev: XDiag, nxt: XDiag, B: set[int], cp: dict[int, int]) -> list[Tail] | None:
    """The tail entries of a step, sorted by their leg in ``nxt``: each leg off ``B`` wired or
    aligned to a leg of ``B``, with the leg of its node in ``nxt`` wired or aligned to the copy of
    that leg; None when one has none."""
    pairs = [*prev.d.wires, *((a, b) for a, b, _ in prev.aligned)]
    out: list[Tail] = []
    for w in pairs:
        for p, q in (w, (w[1], w[0])):
            if p[0] in B or q[0] not in B:
                continue
            image = _cpp(cp, q)
            p2 = next((m for m in nxt.mates(image) if m[0] == p[0]), None)
            if p2 is None:
                return None
            out.append((p, p2))
    return sorted(out, key=lambda e: e[1])


def _xliteral(X: XDiag) -> str:
    """``X`` as a Lean ``LDiagram`` term."""
    D = X.d
    nodes = ", ".join(
        f"⟨{n.id}, .{n.kind}, {n.dim}, {n.ph}, {_lnat(n.outs)}, {_lnat(n.ins)}⟩" for n in D.nodes
    )
    wires = ", ".join(f"({_lport(a)}, {_lport(b)})" for a, b in D.wires)

    def par(p: int | None) -> str:
        return "none" if p is None else f"some {p}"

    boxes = ", ".join(
        [f"⟨{b.id}, {b.mult}, {par(b.parent)}, {_lnat(b.nodes)}, []⟩" for b in D.boxes]
        + [f"⟨{b.id}, {b.mult}, {par(b.parent)}, [], [{_lport(b.port)}]⟩" for b in X.pboxes]
    )
    links = ", ".join(
        f"⟨{x.box}, {_lport(x.l1)}, {_lport(x.l2)}, {_lport(x.e1)}, {_lport(x.e2)}⟩"
        for x in D.links
    )
    cph = ", ".join(f"({x}, {y}, {c}, {m})" for x, y, c, m in X.cph)
    cdim = ", ".join(f"({_lport(p)}, {y}, {c}, {m})" for p, y, c, m in X.cdim)
    aligned = ", ".join(f"⟨{_lport(a)}, {_lport(b)}, {_lnat(perm)}⟩" for a, b, perm in X.aligned)
    return (
        "{ base :=\n      { nodes := [" + nodes + "],\n        wires := [" + wires + "],\n"
        "        outs := [" + ", ".join(map(_lport, D.outs)) + "],\n"
        "        ins := [" + ", ".join(map(_lport, D.ins)) + "],\n"
        "        boxes := [" + boxes + "] },\n    links := [" + links + "],\n"
        f"    cph := [{cph}],\n    cdim := [{cdim}],\n    aligned := [{aligned}] }}"
    )


def _tails_lean(T: Sequence[Tail]) -> str:
    return "[" + ", ".join(f"({_lport(a)}, {_lport(b)})" for a, b in T) + "]"


def _table(name: str, k: int, pairs: dict[int, int], what: str) -> str:
    rows = ", ".join(f"({x}, {y})" for x, y in pairs.items())
    return (
        f"/-- Step {k}: {what}. -/\n"
        f"def {name}{k} (x : ℕ) : ℕ := (([{rows}] : List (ℕ × ℕ)).lookup x).getD 0\n"
    )


def _plain_apart(Xs: Sequence[XDiag]) -> bool:
    """Every dimension and phase index read plainly lies below every index read at a copy."""
    plain_d: set[int] = set()
    read_d: set[int] = set()
    plain_p: set[int] = set()
    read_p: set[int] = set()
    for X in Xs:
        held = {q[0] for q in X.cph}
        ports = {q[0] for q in X.cdim}
        for n in X.d.nodes:
            plain_d.add(n.dim)
            two = n.kind in ("B", "S")
            (read_d if two and n.id in held else plain_d if two else set()).add(n.ph)
            if not two:
                (read_p if n.id in held else plain_p).add(n.ph)
            for i, k in enumerate(n.outs):
                (read_d if (n.id, True, i) in ports else plain_d).add(k)
            for i, k in enumerate(n.ins):
                (read_d if (n.id, False, i) in ports else plain_d).add(k)
    return (not read_d or max(plain_d, default=-1) < min(read_d)) and (
        not read_p or max(plain_p, default=-1) < min(read_p)
    )


def _box_map(prev: XDiag, nxt: XDiag, src: int, dst: int, cp: dict[int, int]) -> dict[int, int]:
    """Each box of ``prev`` in or under ``src`` to the box of ``nxt`` in or under ``dst`` holding
    the copies of its nodes."""
    out = {src: dst}
    under = _below(nxt.d, [dst])
    for b in prev.d.boxes:
        if b.id == src or b.id not in _below(prev.d, [src]):
            continue
        want = tuple(sorted(cp.get(x, -1) for x in b.nodes))
        y = next((c.id for c in nxt.d.boxes if c.id in under and c.nodes == want), None)
        if y is None:
            return {}
        out[b.id] = y
    return out


def export_split(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the digit sums ``evalX`` of ``before`` and ``after``
    equal, ``after`` splitting one outermost node box of ``before`` into ``b`` boxes, the ``c``-th
    new box reading the box index raised by ``c`` times the copy count; None outside that form."""
    tops = sorted(
        (
            int(b)
            for b in after.bang_boxes
            if b not in before.bang_boxes and after.bang_boxes[b].parent is None
        ),
        reverse=True,
    )
    split = [
        b
        for b, box in before.bang_boxes.items()
        if box.parent is None
        and b in after.bang_boxes
        and box.multiplicity != after.bang_boxes[b].multiplicity
    ]
    if not tops or len(split) != 1 or before.scalar != after.scalar:
        return None
    bid = split[0]
    base = len(tops) + 1
    old = sp.expand(before.bang_boxes[bid].multiplicity.to_sympy())
    half = sp.expand(after.bang_boxes[bid].multiplicity.to_sympy())
    if sp.expand(old - base * half) != 0 or any(
        sp.expand(after.bang_boxes[BangBoxId(y)].multiplicity.to_sympy() - half) != 0 for y in tops
    ):
        return None
    index = before.bang_boxes[bid].index
    if after.bang_boxes[bid].index != index or any(
        (after.bang_boxes[BangBoxId(y)].index is None) != (index is None) for y in tops
    ):
        return None
    if _split_reads(before, bid) or any(_split_reads(after, BangBoxId(y)) for y in [bid, *tops]):
        return None
    tables = _Tables()
    names = {b.index for d in (before, after) for b in d.bang_boxes.values() if b.index}
    _seed([before, after], tables, names)
    try:
        L = _xlinked(before, tables, {int(bid)})
        R0 = _xlinked(after, tables, {int(bid), *tops})
    except LinkedUnsupported:
        return None
    B = sorted(int(x) for x in before.bang_boxes[bid].node_scope)
    scopes = [B] + [sorted(int(x) for x in after.bang_boxes[BangBoxId(y)].node_scope) for y in tops]
    if any(len(s) != len(B) for s in scopes):
        return None
    cps = [dict(zip(B, s, strict=True)) for s in scopes]
    copies = {x for s in scopes[1:] for x in s}
    A0, rn0 = _appended(R0, set(B) | copies, copies)
    found = _renamed(L, A0, set(B) | copies)
    if found is None:
        return None
    R, back = found

    def ren(p: Port) -> Port:
        return (back.get(p[0], p[0]), p[1], p[2])

    rn = {ren(a): ren(b) for a, b in rn0.items()}
    by_id = {n.id: n for n in R.d.nodes}
    if set(by_id) != {n.id for n in L.d.nodes} | copies:
        return None
    mults = {c: _Tables.index(tables.mults, str(sp.expand(c * half))) for c in range(1, base + 1)}
    if _Tables.index(tables.mults, str(old)) != mults[base]:
        return None
    ph = {n.id: n.ph for n in L.d.nodes}
    held = {q[0] for q in L.cph}
    for x in B:
        node = before.nodes[x]  # type: ignore[index]
        if x not in after.nodes or after.nodes[x].phase != node.phase:  # type: ignore[index]
            return None
        reads = index is not None and node.phase is not None and index in node.phase.free_symbols
        if reads:
            if x in held:
                return None
            L.cph.append((x, int(bid), 0, mults[1]))
        for c, y in enumerate(tops, start=1):
            copy = after.nodes[cps[c][x]]  # type: ignore[index]
            want = node.phase
            if reads and index is not None and node.phase is not None:
                index2 = after.bang_boxes[BangBoxId(y)].index
                scope = after.bang_boxes[BangBoxId(y)].node_scope
                sym = None if index2 is None else _symbol_in(after, scope, None, index2)
                if sym is None:
                    return None
                want = node.phase.substitute({index: sym + c * half})
            if copy.phase != want:
                return None
            if reads:
                by_id[cps[c][x]] = replace(by_id[cps[c][x]], ph=ph[x])
    R.d.nodes = [by_id[n.id] for n in L.d.nodes] + [
        by_id[cp[n.id]] for cp in cps[1:] for n in L.d.nodes if n.id in cp
    ]
    owners = [int(bid), *tops]
    stages: list[XDiag] = [L]
    steps = []
    for k in range(1, base):
        if k < base - 1:
            stage = _stage(
                L,
                R,
                tops[k - 1],
                mults[base - k],
                {x for s in scopes[: k + 1] for x in s},
                _below(R.d, tops[k:]),
                {x for s in scopes[k + 1 :] for x in s},
            )
        else:
            stage = replace(R, cph=[], cdim=[])
        if stage is None:
            return None
        prev = stages[-1]
        Bk = scopes[k - 1]
        cp = {cps[k - 1][x]: cps[k][x] for x in B}
        cb = _box_map(prev, stage, owners[k - 1], owners[k], cp)
        if not cb:
            return None
        stage.cph, stage.cdim = _copied(prev, set(Bk), cp, cb)
        stages.append(stage)
        steps.append((Bk, cp, cb))
    final = stages[-1]
    if sorted(final.cdim) != sorted(R.cdim) or sorted(R.cph) != sorted(
        q for q in final.cph if q[1] not in owners
    ):
        return None
    if not _plain_apart(stages):
        return None
    last = "P" if rn else "L'"
    names_l = ["L", *(f"M{k}" for k in range(1, base - 1)), last]
    decls = []
    for k in range(base):
        what = (
            "The diagram before the split"
            if k == 0
            else "The diagram after the split, each new leg last on its side"
            if k == base - 1 and rn
            else "The diagram after the split"
            if k == base - 1
            else f"The diagram after step {k} of the split"
        )
        decls.append(
            f"/-- {what}, as a linked diagram. -/\n"
            f"def {names_l[k]} : LDiagram :=\n  {_xliteral(stages[k])}\n"
        )
    if rn:
        engine = _moved(final, rn)
        if not perm_ok(final, engine, rn):
            return None
        rows = ", ".join(f"({_lport(a)}, {_lport(b)})" for a, b in sorted(rn.items()))
        pproof = ",\n    ".join(["by decide +kernel"] * _PERM_FIELDS)
        decls.append(
            "/-- The diagram after the split, as a linked diagram. -/\n"
            f"def L' : LDiagram :=\n  {_xliteral(engine)}\n\n"
            "/-- Each new leg moved from last on its side to its place in `L'`. -/\n"
            f"def rn (p : Port) : Port := (([{rows}] : List (Port × Port)).lookup p).getD p\n\n"
            "theorem okP : LegPermOK P L' rn :=\n"
            f"  ⟨{pproof}⟩\n"
        )
    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    hyps, sigma = [], "σ'"
    for k in range(1, base):
        Bk, cp, cb = steps[k - 1]
        T = _tails(stages[k - 1], stages[k], set(Bk), cp)
        if T is None:
            return None
        boxes = (owners[k - 1], owners[k], mults[base - k + 1], mults[1], mults[base - k])
        if not split_ok(stages[k - 1], stages[k], boxes, Bk, cp, cb, T):
            return None
        args = " ".join(map(str, boxes))
        decls.append(
            _table("cp", k, cp, "each node of the box split to its node in the new box")
            + "\n"
            + _table("cb", k, cb, "each box of the box split to its box in the new box")
            + "\n"
            f"/-- Step {k}: each leg bundled over the box split and its leg for the new box. -/\n"
            f"def T{k} : List (Port × Port) := {_tails_lean(T)}\n\n"
            f"theorem ok{k} :\n    BoxSplitOK {names_l[k - 1]} {names_l[k]} {args} {_lnat(Bk)} "
            f"cp{k} cb{k} T{k} :=\n"
            f"  ⟨{proof}⟩\n"
        )
        hyps.append(f"ks {boxes[2]} = ks {boxes[3]} + ks {boxes[4]}")
        sigma_k = f"(splitφ {names_l[k - 1]} {_lnat(Bk)} cp{k} T{k} (ks {mults[1]}))"
        if k == 1:
            sigma = f"(bndMap {names_l[0]}.base {names_l[1]}.base {sigma_k} @@)"
        else:
            sigma = sigma.replace(
                "@@", f"(bndMap {names_l[k - 1]}.base {names_l[k]}.base {sigma_k} @@)"
            )
    sigma = sigma.replace("@@", "(bndMap P.base L'.base (rnφ rn) σ')" if rn else "σ'")
    chain = "evalX_split ok1 E ks h1 _"
    for k in range(2, base):
        chain = f"(evalX_split ok{k} E ks h{k} _).trans ({chain})"
    if rn:
        chain = f"(evalX_legPerm okP E ks σ').trans ({chain})"
    head = " →\n    ".join(hyps)
    hs = " ".join(f"h{k}" for k in range(1, base))
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        + "\n".join(decls)
        + "\n/-- The digit sums agree, each step's moved copies read at the new box's digits. -/\n"
        f"theorem main : ∀ (E : Env) (ks : ℕ → ℕ),\n    {head} →\n"
        "    ∀ σ' : ℕ → List ℕ → ℕ,\n"
        f"      L'.evalX E ks σ' = L.evalX E ks {sigma} :=\n"
        f"  fun E ks {hs} σ' => {chain}\n\n"
        "end ArchytasZX.Export\n"
    )
