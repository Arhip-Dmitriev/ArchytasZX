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

"""Lean output for a kernel align merge over linked diagrams (ArchytasZX/LinkedMerge.lean).
Both sides are written as ``LDiagram``s with aligned wires (:func:`lean_split._xlinked`), each
pair box over the merged box as an ``LPair`` (:func:`lean_linked._strip_pairs`), each Zp(1 -> 1)
node of a top-level box between a leg of a node in no box and a leg of its box as an ``LPair``
with ``fixL`` (:func:`_strip_zp`), the merged side's nodes in the first side's order.
``export_merge`` writes both sides, the pairs, their box groups, ``MergeOK`` and ``SameSlots``
(checked here first on a mirror, then in Lean by ``decide +kernel``) and ``main``, the ``evalP``
digit sums of both sides read at the flat positions of their slot data equal through
``evalP_merge_flat``; None outside that form."""

from __future__ import annotations

from collections.abc import Sequence

from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.lean_linked import (
    LinkedUnsupported,
    PairSpec,
    Port,
    _lgroups,
    _lport,
    _phase_key,
    _port,
    _strip_pairs,
    _Tables,
    anchors,
    groups,
    slot_data,
)
from archytaszx.semantics.lean_split import XDiag, _seed, _xlinked, _xliteral

_FIELDS = 16
"""The fields of ``MergeOK``."""

Zp = tuple[int, int, Port, Port, str, bool]
"""A Zp(1 -> 1) node read as a pair: node, box, outside leg, leg in the box, phase key, and
whether its output meets the leg in the box."""


def _strip_zp(d: Diagram) -> tuple[Diagram, list[Zp]] | None:
    """``d`` without its Zp(1 -> 1) nodes of one top-level box, each wired once to a leg of a
    node in no box and once to a leg of a node of its box only, both legs off the boundary and
    wired only there; None when a Zp node is outside that form."""
    bnd = {*d.boundary_inputs, *d.boundary_outputs}
    out = d.copy()
    found: list[Zp] = []
    for z, node in sorted(d.nodes.items()):
        if node.generator_type.name != "Zp":
            continue
        held = [b for b, x in d.bang_boxes.items() if z in x.node_scope]
        if len(node.outputs) != 1 or len(node.inputs) != 1 or len(held) != 1:
            return None
        box = d.bang_boxes[held[0]]
        if box.parent is not None or box.pair_of is not None or box.links:
            return None
        ends = []
        for leg in (PortRef(z, Direction.OUTPUT, 0), PortRef(z, Direction.INPUT, 0)):
            ws = [w for w in d.wires if leg in w.endpoints()]
            if len(ws) != 1:
                return None
            ends.append(ws[0].b if ws[0].a == leg else ws[0].a)
        homes = [[b for b, x in d.bang_boxes.items() if e.node_id in x.node_scope] for e in ends]
        if sorted(map(len, homes)) != [0, 1] or [held[0]] not in homes:
            return None
        inner = ends[homes.index([held[0]])]
        outer = ends[1 - homes.index([held[0]])]
        for ref in ends:
            if ref in bnd or ref.node_id == z:
                return None
            if len([w for w in d.wires if ref in w.endpoints()]) != 1:
                return None
        found.append(
            (
                int(z),
                int(held[0]),
                _port(outer),
                _port(inner),
                _phase_key(node.phase),
                ends[0] == inner,
            )
        )
        out.remove_node(NodeId(z))
    return out, found


def _cph_at(X: XDiag, x: int) -> tuple[int, int, int] | None:
    q = next((q for q in X.cph if q[0] == x), None)
    return None if q is None else (X.d.depth(q[1], x), q[2], q[3])


def _cdim_at(X: XDiag, p: Port) -> tuple[int, int, int] | None:
    q = next((q for q in X.cdim if q[0] == p), None)
    return None if q is None else (X.d.depth(q[1], p[0]), q[2], q[3])


def _id_al(X: XDiag, a: tuple[Port, Port, tuple[int, ...]]) -> bool:
    c = X.leg_chain(a[0])
    return list(a[2]) == list(range(len(c))) and c == X.leg_chain(a[1])


def merge_ok(L: XDiag, R: XDiag, mults: Sequence[tuple[int, int]]) -> bool:
    """The Lean ``MergeOK L R ps ps'``, ``mults`` each pair's copy count on the two sides."""
    legs = L.d.legs()
    read = [*legs, *L.d.bnd(), *(w[0] for w in L.d.wires), *(a[0] for a in L.aligned)]
    return (
        R.d.nodes == L.d.nodes
        and R.d.outs == L.d.outs
        and R.d.ins == L.d.ins
        and all(R.d.chain(n.id) == L.d.chain(n.id) for n in L.d.nodes)
        and all(R.tail(p) == L.tail(p) for p in legs)
        and all(R.leg_chain(p) == L.leg_chain(p) for p in read)
        and all(_cph_at(R, n.id) == _cph_at(L, n.id) for n in L.d.nodes)
        and all(_cdim_at(R, p) == _cdim_at(L, p) for p in legs)
        and not L.d.links
        and not R.d.links
        and all(w in R.d.wires for w in L.d.wires)
        and all(a in R.aligned or (_id_al(L, a) and (a[0], a[1]) in R.d.wires) for a in L.aligned)
        and all(
            w in L.d.wires or any(_id_al(L, a) and (a[0], a[1]) == w for a in L.aligned)
            for w in R.d.wires
        )
        and all(a in L.aligned for a in R.aligned)
        and all(x == y for x, y in mults)
    )


def export_merge(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the ``evalP`` digit sums of ``before`` and ``after``
    equal, ``after`` the kernel's :func:`~archytaszx.diagram.bangbox.align_merged` of one
    alignment group of ``before``; None outside that form."""
    from archytaszx.diagram.bangbox import BangBoxError, align_merged
    from archytaszx.semantics.kernel import KernelError, _align_group, _same

    if before.scalar != after.scalar:
        return None
    roots = sorted({b.align for b in before.bang_boxes.values() if b.align is not None})
    root: BangBoxId | None = None
    for r in roots:
        if r not in before.bang_boxes:
            continue
        try:
            _align_group(before, r)
            if _same(align_merged(before, r), after):
                root = r
                break
        except (KernelError, BangBoxError):
            continue
    if root is None:
        return None
    specs: list[PairSpec] = []
    sides: list[Diagram] = []
    zps: list[list[Zp]] = []
    for d in (before, after):
        if any(b.pair_of is not None for b in d.bang_boxes.values()):
            got = _strip_pairs(d, root)
            if got is None:
                return None
            d, _, found = got
            if sides and found != specs:
                return None
            specs = found
        elif specs:
            return None
        stripped = _strip_zp(d)
        if stripped is None:
            return None
        d, zp = stripped
        sides.append(d)
        zps.append(zp)
    zl, zr = zps
    if [z[:1] + z[2:] for z in zl] != [z[:1] + z[2:] for z in zr]:
        return None
    b0, a0 = sides
    tables = _Tables()
    names = {b.index for d in sides for b in d.bang_boxes.values() if b.index}
    _seed(sides, tables, names)
    try:
        L = _xlinked(b0, tables, set())
        R = _xlinked(a0, tables, set())
    except LinkedUnsupported:
        return None
    if L.pboxes or R.pboxes:
        return None
    order = {n.id: n for n in R.d.nodes}
    if set(order) != {n.id for n in L.d.nodes}:
        return None
    R.d.nodes = [order[n.id] for n in L.d.nodes]
    mults = [(R.d.mult_of(int(root)), L.d.mult_of(int(root))) for _ in specs]
    mults += [(R.d.mult_of(y[1]), L.d.mult_of(x[1])) for x, y in zip(zl, zr, strict=True)]
    if not merge_ok(L, R, mults):
        return None
    Gl, Gr = groups(b0), groups(a0)
    sl, sr = slot_data(L.d, Gl), slot_data(R.d, Gr)
    n = sl[0] + sl[1]
    if sr[:2] != sl[:2] or any(anchors(sr, i) != anchors(sl, i) for i in range(n)):
        return None
    keys = [key for _, _, _, key, _ in specs] + [z[4] for z in zl]
    ph = {key: _Tables.index(tables.phases, key) for key in keys}

    def flag(b: bool) -> str:
        return "true" if b else "false"

    def pairs(zp: Sequence[Zp]) -> str:
        items = [
            f"{{ box := {int(root)}, left := {_lport(lp)}, right := {_lport(rp)}, "
            f"ph := {ph[key]}, swap := {flag(sw)} }}"
            for _, lp, rp, key, sw in specs
        ] + [
            f"{{ box := {b}, left := {_lport(lp)}, right := {_lport(rp)}, ph := {ph[key]}, "
            f"swap := {flag(sw)}, fixL := true }}"
            for _, b, lp, rp, key, sw in zp
        ]
        return "[" + ", ".join(items) + "]"

    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        "/-- The diagram before the merge, as a linked diagram with aligned wires. -/\n"
        f"def L : LDiagram :=\n  {_xliteral(L)}\n\n"
        "/-- The diagram after the merge, as a linked diagram with aligned wires. -/\n"
        f"def L' : LDiagram :=\n  {_xliteral(R)}\n\n"
        "/-- The pairs before the merge: the pair boxes, then each Zp(1 -> 1) node. -/\n"
        f"def ps : List LPair := {pairs(zl)}\n\n"
        "/-- The pairs after the merge. -/\n"
        f"def ps' : List LPair := {pairs(zr)}\n\n"
        "/-- The box groups before the merge, node lists by box. -/\n"
        f"def G : List (ℕ × List (List ℕ)) := {_lgroups(Gl)}\n\n"
        "/-- The box groups after the merge, node lists by box. -/\n"
        f"def G' : List (ℕ × List (List ℕ)) := {_lgroups(Gr)}\n\n"
        f"theorem ok : MergeOK L L' ps ps' :=\n  ⟨{proof}⟩\n\n"
        "theorem ss : SameSlots (L.slotData G) (L'.slotData G') :=\n"
        "  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩\n\n"
        "/-- The digit sums read at the flat positions of the two sides' slot data agree. -/\n"
        "theorem main : ∀ (E : Env) (ks : ℕ → ℕ) (y : ℕ → ℕ),\n"
        "    L'.evalP ps' E ks (flatAt (L'.slotData G') ks y) =\n"
        "      L.evalP ps E ks (flatAt (L.slotData G) ks y) :=\n"
        "  evalP_merge_flat ok ss\n\n"
        "end ArchytasZX.Export\n"
    )
