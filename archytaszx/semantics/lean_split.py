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

"""Lean output for a kernel power split of base 2 over linked diagrams
(ArchytasZX/LinkedSplit.lean). Both sides are written as ``LDiagram``s
(:func:`lean_linked.linked`, the split box's index cleared); each node of the split box whose phase
reads the box index gets the copy-indexed phase ``(x, box, 0, c₁)``, its copy ``(cp x, box₂, 1,
c₁)`` and the phase index of ``x``. ``export_split`` writes both sides, the copy map ``cp``, the
``BoxSplitOK`` conditions (checked here first on a mirror, then in Lean by ``decide +kernel``) and
``main``, the two linked digit sums equal through ``evalL_split``; None outside that form."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.bangbox import _symbol_in
from archytaszx.diagram.graph import BangBoxId, Diagram
from archytaszx.semantics.lean_linked import (
    LDiag,
    LinkedUnsupported,
    LLink,
    LNode,
    Port,
    _cpp,
    _lnat,
    _same,
    _Tables,
    _wire_in,
    linked,
    literal,
)

Cph = tuple[int, int, int, int]
"""A copy-indexed phase: node, box, coefficient, multiplicity index."""

_FIELDS = 20
"""The fields of ``BoxSplitOK``."""


def split_ok(
    L: LDiag,
    R: LDiag,
    cph: Sequence[Cph],
    cph2: Sequence[Cph],
    boxes: tuple[int, int, int, int, int],
    B: Sequence[int],
    cp: dict[int, int],
) -> bool:
    """The Lean ``BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp``, field by field."""
    bid, bid2, mb, c1, c2 = boxes
    inB = set(B)

    def cpp(p: Port) -> Port:
        return _cpp(cp, p)

    legs, legs_r = L.legs(), R.legs()
    ids_l = {n.id for n in L.nodes}
    for D in (L, R):
        if any(D.chain(w[0][0]) != D.chain(w[1][0]) for w in D.wires):
            return False
    if any(not (q[0] in ids_l and q[0] in inB and q[1] == bid and q[2] == 0) for q in cph):
        return False
    if list(cph2) != [*cph, *((cp.get(q[0], 0), bid2, 1, c1) for q in cph)]:
        return False
    copies = [LNode(cp.get(n.id, 0), n.kind, n.dim, n.ph, n.outs, n.ins) for n in L.nodes]
    if R.nodes != L.nodes + [c for n, c in zip(L.nodes, copies, strict=True) if n.id in inB]:
        return False
    ids = [n.id for n in R.nodes]
    if len(set(ids)) != len(ids) or len(set(legs_r)) != len(legs_r):
        return False
    if len(set(R.bnd())) != len(R.bnd()):
        return False
    for n in L.nodes:
        ch = L.chain(n.id)
        if n.id in inB:
            x = cp.get(n.id, 0)
            if not (
                ch[:1] == [mb]
                and R.chain(n.id) == [c1, *ch[1:]]
                and R.chain(x) == [c2, *ch[1:]]
                and L.depth(bid, n.id) == 0
                and R.depth(bid, n.id) == 0
                and R.depth(bid2, x) == 0
            ):
                return False
        elif R.chain(n.id) != ch:
            return False
    for w in L.wires:
        a, b = w[0][0] in inB, w[1][0] in inB
        if w[0] not in legs or w[1] not in legs or a != b or not _wire_in(R.wires, w):
            return False
        if a and not _wire_in(R.wires, (cpp(w[0]), cpp(w[1]))):
            return False
    for w in R.wires:
        if w not in L.wires and not any(
            w0[0][0] in inB and w == (cpp(w0[0]), cpp(w0[1])) for w0 in L.wires
        ):
            return False

    def copied(lk: LLink, x: LLink) -> bool:
        return (
            (x.l1, x.l2, x.e1, x.e2) == tuple(map(cpp, (lk.l1, lk.l2, lk.e1, lk.e2)))
            and R.depth(x.box, cp.get(lk.l1[0], 0)) == L.depth(lk.box, lk.l1[0])
            and R.mult_of(x.box) == L.mult_of(lk.box)
        )

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
        if ins[0]:
            if not (
                all(ins)
                and dp >= 1
                and any(_same(L, R, lk, x) for x in R.links)
                and any(copied(lk, x) for x in R.links)
            ):
                return False
        elif any(ins) or not any(_same(L, R, lk, x) for x in R.links):
            return False
    for lx in R.links:
        if not any(_same(L, R, lk, lx) for lk in L.links) and not any(
            lk.l1[0] in inB and copied(lk, lx) for lk in L.links
        ):
            return False
    for p in L.bnd():
        if p not in legs or p not in R.bnd() or (p[0] in inB and cpp(p) not in R.bnd()):
            return False
    return all(q in L.bnd() or any(p[0] in inB and q == cpp(p) for p in L.bnd()) for q in R.bnd())


def _indexed(d: Diagram, bid: BangBoxId) -> Diagram:
    """``d`` with the index of ``bid`` cleared; raises :class:`LinkedUnsupported` when a leg
    dimension reads it."""
    box = d.bang_boxes[bid]
    if box.index is None:
        return d
    for x in box.node_scope:
        node = d.nodes[x]
        if any(box.index in str(p.dim) for p in [*node.outputs, *node.inputs]):
            raise LinkedUnsupported("a leg dimension reads the box index")
    out = d.copy()
    out.set_bang_box_extension(bid, index=None)
    return out


def _cph_lean(cph: Sequence[Cph]) -> str:
    return "[" + ", ".join(f"({x}, {y}, {c}, {m})" for x, y, c, m in cph) + "]"


def _with_cph(D: LDiag, cph: Sequence[Cph]) -> str:
    text = literal(D)
    assert text.endswith("] }")
    return text[: -len(" }")] + f",\n    cph := {_cph_lean(cph)} }}"


def export_split(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the linked digit sums of ``before`` and ``after`` equal,
    ``after`` splitting one outermost node box of ``before`` into two boxes, the second reading
    the box index raised by the first's copy count; None outside that form."""
    fresh_boxes = [b for b in after.bang_boxes if b not in before.bang_boxes]
    tops = [b for b in fresh_boxes if after.bang_boxes[b].parent is None]
    split = [
        b
        for b, box in before.bang_boxes.items()
        if box.parent is None
        and b in after.bang_boxes
        and box.multiplicity != after.bang_boxes[b].multiplicity
    ]
    if len(tops) != 1 or len(split) != 1:
        return None
    bid, bid2 = split[0], tops[0]
    old, half, half2 = (
        sp.expand(m.to_sympy())
        for m in (
            before.bang_boxes[bid].multiplicity,
            after.bang_boxes[bid].multiplicity,
            after.bang_boxes[bid2].multiplicity,
        )
    )
    if sp.expand(old - half - half2) != 0 or before.scalar != after.scalar:
        return None
    index, index2 = before.bang_boxes[bid].index, after.bang_boxes[bid2].index
    if (index is None) != (index2 is None) or after.bang_boxes[bid].index != index:
        return None
    tables = _Tables()
    try:
        L = linked(_indexed(before, bid), tables)
        R = linked(_indexed(_indexed(after, bid), bid2), tables)
    except LinkedUnsupported:
        return None
    B = sorted(int(x) for x in before.bang_boxes[bid].node_scope)
    fresh = sorted({n.id for n in R.nodes} - {n.id for n in L.nodes})
    if len(fresh) != len(B):
        return None
    cp = dict(zip(B, fresh, strict=True))
    by_id = {n.id: n for n in R.nodes}
    if set(by_id) != {n.id for n in L.nodes} | set(fresh):
        return None
    c1 = _Tables.index(tables.mults, str(half))
    cph: list[Cph] = []
    sym = (
        None
        if index2 is None
        else _symbol_in(after, after.bang_boxes[bid2].node_scope, None, index2)
    )
    ph = {n.id: n.ph for n in L.nodes}
    for x in B:
        node, copy = before.nodes[x], after.nodes[cp[x]]  # type: ignore[index]
        if after.nodes[x].phase != node.phase:  # type: ignore[index]
            return None
        want = node.phase
        if index is not None and node.phase is not None and index in node.phase.free_symbols:
            if sym is None:
                return None
            cph.append((x, int(bid), 0, c1))
            want = node.phase.substitute({index: sym + half})
        if copy.phase != want:
            return None
        by_id[cp[x]] = replace(by_id[cp[x]], ph=ph[x])
    R.nodes = [by_id[n.id] for n in L.nodes] + [by_id[cp[n.id]] for n in L.nodes if n.id in cp]
    cph2 = [*cph, *((cp[q[0]], int(bid2), 1, c1) for q in cph)]
    mb = _Tables.index(tables.mults, str(old))
    c2 = _Tables.index(tables.mults, str(half2))
    boxes = (int(bid), int(bid2), mb, c1, c2)
    if not split_ok(L, R, cph, cph2, boxes, B, cp):
        return None
    table = ", ".join(f"({x}, {y})" for x, y in cp.items())
    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    args = " ".join(map(str, boxes))
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        "/-- The diagram before the split, as a linked diagram. -/\n"
        f"def L : LDiagram :=\n  {_with_cph(L, cph)}\n\n"
        "/-- The diagram after the split, as a linked diagram. -/\n"
        f"def L' : LDiagram :=\n  {_with_cph(R, cph2)}\n\n"
        "/-- Each node of the split box to its node in the copy. -/\n"
        f"def cp (x : ℕ) : ℕ := (([{table}] : List (ℕ × ℕ)).lookup x).getD 0\n\n"
        f"theorem ok : BoxSplitOK L L' {args} {_lnat(B)} cp :=\n"
        f"  ⟨{proof}⟩\n\n"
        "/-- The linked digit sums agree, the box's copies from the first box's copy count on read "
        "at the copy's digits. -/\n"
        f"theorem main : ∀ (E : Env) (ks : ℕ → ℕ), ks {mb} = ks {c1} + ks {c2} →\n"
        "    ∀ σ' : ℕ → List ℕ → ℕ,\n"
        f"      L'.evalL E ks σ' = L.evalL E ks (bndMap L.base L'.base (splitφ {_lnat(B)} cp "
        f"(ks {c1})) σ') :=\n"
        "  fun E ks hk σ' => evalL_split ok E ks hk σ'\n\n"
        "end ArchytasZX.Export\n"
    )
