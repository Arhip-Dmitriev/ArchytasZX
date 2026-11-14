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
Both sides are written as ``LDiagram``s (:func:`lean_linked.linked`, the split boxes' indices
cleared), the outside nodes the split renames given back their ids. A split of base ``b`` is
written as ``b - 1`` splits in two: step ``k`` leaves the box at ``(b - k) h`` copies and moves its
top ``h`` copies into the ``k``-th new box, so the diagram after step ``k`` keeps the new boxes
``1 .. k``, the legs appended for them, and the box at ``(b - k) h``. Each node of the split box
whose phase reads the box index gets the copy-indexed phase ``(x, box, 0, m)``, its copy in the
``k``-th new box ``(cp x, box_k, 1, (b - k) h)`` and the phase index of ``x``. A leg of a top-level
Z or X spider wired into the box is a tail entry ``(leg, appended leg)``. ``export_split`` writes
every diagram, copy map and tail list, the ``BoxSplitOK`` conditions of each step (checked here
first on a mirror, then in Lean by ``decide +kernel``) and ``main``, the linked digit sums of the
first and last diagram equal through ``evalL_split`` at every step; None outside that form."""

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
    _lport,
    _same,
    _Tables,
    _wire_in,
    linked,
    literal,
)

Cph = tuple[int, int, int, int]
"""A copy-indexed phase: node, box, coefficient, multiplicity index."""

Tail = tuple[Port, Port]
"""A tail entry: a leg bundled over the split box and the leg appended for the new box."""

_FIELDS = 25
"""The fields of ``BoxSplitOK``."""


def _tail(D: LDiag, q: Port) -> list[int]:
    """The Lean ``Diagram.tail`` of ``q`` (no port boxes)."""
    w = next((w for w in D.wires if q in w), None)
    if w is None:
        return []
    a, b = D.chain(q[0]), D.chain((w[1] if w[0] == q else w[0])[0])
    return b[len(a) :] if len(a) < len(b) else []


def _leg_chain(D: LDiag, q: Port) -> list[int]:
    return D.chain(q[0]) + _tail(D, q)


def _port_dim(D: LDiag, p: Port) -> int:
    node = next((n for n in D.nodes if n.id == p[0]), None)
    if node is None:
        return 0
    side = node.outs if p[1] else node.ins
    return side[p[2]] if p[2] < len(side) else 0


def _ext(T: Sequence[Tail], n: LNode) -> LNode:
    """The Lean ``extT T n``."""

    def get(side: tuple[int, ...], i: int) -> int:
        return side[i] if i < len(side) else 0

    outs = tuple(get(n.outs, e[0][2]) for e in T if e[0][0] == n.id and e[0][1])
    ins = tuple(get(n.ins, e[0][2]) for e in T if e[0][0] == n.id and not e[0][1])
    return replace(n, outs=n.outs + outs, ins=n.ins + ins)


def split_ok(
    L: LDiag,
    R: LDiag,
    cph: Sequence[Cph],
    cph2: Sequence[Cph],
    boxes: tuple[int, int, int, int, int],
    B: Sequence[int],
    cp: dict[int, int],
    T: Sequence[Tail],
) -> bool:
    """The Lean ``BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T``, field by field."""
    bid, bid2, mb, c1, c2 = boxes
    inB = set(B)
    keys = [e[0] for e in T]

    def sl(p: Port) -> bool:
        return p[0] in inB or p in keys

    def cq(p: Port) -> Port:
        if p[0] in inB:
            return _cpp(cp, p)
        return next((e[1] for e in T if e[0] == p), p)

    legs, legs_r = L.legs(), R.legs()
    ids_l = {n.id for n in L.nodes}
    if any(not (q[0] in ids_l and (q[0] not in inB or (q[1] == bid and q[3] == c1))) for q in cph):
        return False
    if list(cph2) != [*cph, *((cp.get(q[0], 0), bid2, q[2] + 1, c1) for q in cph if q[0] in inB)]:
        return False
    if any(q[0] not in inB and R.depth(q[1], q[0]) != L.depth(q[1], q[0]) for q in cph):
        return False
    copies = [LNode(cp.get(n.id, 0), n.kind, n.dim, n.ph, n.outs, n.ins) for n in L.nodes]
    want = [_ext(T, n) for n in L.nodes]
    if R.nodes != want + [c for n, c in zip(L.nodes, copies, strict=True) if n.id in inB]:
        return False
    ids = [n.id for n in R.nodes]
    for xs in (ids, legs, legs_r, R.bnd()):
        if len(set(xs)) != len(xs):
            return False
    kinds = {n.id: n.kind for n in L.nodes}
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
    for p, p2 in T:
        if not (
            p in legs
            and p[0] not in inB
            and L.chain(p[0]) == []
            and p2[:2] == p[:2]
            and kinds.get(p[0]) in ("Z", "X")
        ):
            return False
    images = []
    for p in legs:
        lc = _leg_chain(L, p)
        if p not in legs_r or _port_dim(R, p) != _port_dim(L, p):
            return False
        if not sl(p):
            if _leg_chain(R, p) != lc:
                return False
            continue
        q = cq(p)
        if not (
            lc[:1] == [mb]
            and _leg_chain(R, p) == [c1, *lc[1:]]
            and q in legs_r
            and _leg_chain(R, q) == [c2, *lc[1:]]
            and _port_dim(R, q) == _port_dim(L, p)
            and q not in legs
        ):
            return False
        images.append(q)
    if len(set(images)) != len(images) or any(q not in legs and q not in images for q in legs_r):
        return False
    for w in L.wires:
        if not (
            w[0] in legs
            and w[1] in legs
            and sl(w[0]) == sl(w[1])
            and _leg_chain(L, w[1]) == _leg_chain(L, w[0])
            and _wire_in(R.wires, w)
        ):
            return False
        if sl(w[0]) and not _wire_in(R.wires, (cq(w[0]), cq(w[1]))):
            return False
    for w in R.wires:
        if not _wire_in(L.wires, w) and not any(
            sl(w0[0]) and _wire_in([(cq(w0[0]), cq(w0[1]))], w) for w0 in L.wires
        ):
            return False

    def copied(lk: LLink, x: LLink) -> bool:
        return (
            (x.l1, x.l2, x.e1, x.e2) == tuple(map(cq, (lk.l1, lk.l2, lk.e1, lk.e2)))
            and R.depth(x.box, cq(lk.l1)[0]) == L.depth(lk.box, lk.l1[0])
            and R.mult_of(x.box) == L.mult_of(lk.box)
        )

    for lk in L.links:
        ch = _leg_chain(L, lk.l1)
        dp = L.depth(lk.box, lk.l1[0])
        rest = ch[:dp] + ch[dp + 1 :]
        if not (
            all(p in legs for p in (lk.l1, lk.l2, lk.e1, lk.e2))
            and _leg_chain(L, lk.l2) == ch
            and dp < len(ch)
            and ch[dp] == L.mult_of(lk.box)
            and _leg_chain(L, lk.e1) == rest
            and _leg_chain(L, lk.e2) == rest
        ):
            return False
        ins = [sl(p) for p in (lk.l1, lk.l2, lk.e1, lk.e2)]
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
            sl(lk.l1) and copied(lk, lx) for lk in L.links
        ):
            return False
    for p in L.bnd():
        if p not in legs or p not in R.bnd() or (sl(p) and cq(p) not in R.bnd()):
            return False
    return all(q in L.bnd() or any(sl(p) and q == cq(p) for p in L.bnd()) for q in R.bnd())


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


def _partner(D: LDiag, q: Port) -> Port | None:
    w = next((w for w in D.wires if q in w), None)
    return None if w is None else (w[1] if w[0] == q else w[0])


def _renamed(L: LDiag, R: LDiag, fixed: set[int]) -> LDiag | None:
    """``R`` with each node off ``fixed`` and off ``L`` given the id of the one node of ``L`` off
    ``R`` with its generator, phase, leading legs, boundary places and wiring; None when some
    node has no such match."""
    ids_l = {n.id for n in L.nodes}
    ids_r = {n.id for n in R.nodes}
    gone = sorted(ids_l - ids_r)
    new = sorted(ids_r - ids_l - fixed)
    if len(gone) != len(new):
        return None
    by_l = {n.id: n for n in L.nodes}
    by_r = {n.id: n for n in R.nodes}
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
            if (p in L.outs) != (q in R.outs) or (p in L.ins) != (q in R.ins):
                return False
            pl, pr = _partner(L, p), _partner(R, q)
            if pl is None or pl[0] not in gone:
                if pr != pl:
                    return False
            elif pl[0] in match:
                if pr != (match[pl[0]], pl[1], pl[2]):
                    return False
            elif pr is None or pr[0] not in new:
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

    return LDiag(
        [replace(n, id=back.get(n.id, n.id)) for n in R.nodes],
        [(f(a), f(b)) for a, b in R.wires],
        [f(p) for p in R.outs],
        [f(p) for p in R.ins],
        [replace(b, nodes=tuple(sorted(back.get(x, x) for x in b.nodes))) for b in R.boxes],
        [LLink(x.box, f(x.l1), f(x.l2), f(x.e1), f(x.e2)) for x in R.links],
    )


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
    L: LDiag,
    R: LDiag,
    bid: int,
    mult: int,
    keep: set[int],
    drop_boxes: set[int],
    drop_nodes: set[int],
) -> LDiag | None:
    """``R`` with the nodes ``drop_nodes`` and boxes ``drop_boxes`` left out, ``bid`` at the
    multiplicity index ``mult``, and the legs appended to the nodes of ``L`` kept only when wired
    into the nodes ``keep`` (a leading run of the appended legs); None otherwise."""
    lens = {n.id: (len(n.outs), len(n.ins)) for n in L.nodes}
    nodes = []
    for n in R.nodes:
        if n.id in drop_nodes:
            continue
        if n.id in lens:
            sizes = []
            for o, side in ((True, n.outs), (False, n.ins)):
                base = lens[n.id][0 if o else 1]
                held = [
                    (p := _partner(R, (n.id, o, i))) is not None and p[0] in keep
                    for i in range(base, len(side))
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

    return LDiag(
        nodes,
        [w for w in R.wires if live(w[0]) and live(w[1])],
        [p for p in R.outs if live(p)],
        [p for p in R.ins if live(p)],
        [replace(b, mult=mult) if b.id == bid else b for b in R.boxes if b.id not in drop_boxes],
        [x for x in R.links if x.box not in drop_boxes],
    )


def _tails(prev: LDiag, nxt: LDiag, B: set[int], cp: dict[int, int]) -> list[Tail] | None:
    """The tail entries of a step: each leg of a top-level node off ``B`` wired to a leg of ``B``,
    with the leg of its node in ``nxt`` wired to the copy of that leg; None when one has none."""
    out: list[Tail] = []
    for w in prev.wires:
        for p, q in (w, (w[1], w[0])):
            if p[0] in B or q[0] not in B or prev.chain(p[0]) != []:
                continue
            image = _cpp(cp, q)
            p2 = next(
                (
                    (a if b == image else b)
                    for a, b in nxt.wires
                    if image in (a, b) and (a if b == image else b)[0] == p[0]
                ),
                None,
            )
            if p2 is None:
                return None
            out.append((p, p2))
    return sorted(out, key=lambda e: e[1])


def _cph_lean(cph: Sequence[Cph]) -> str:
    return "[" + ", ".join(f"({x}, {y}, {c}, {m})" for x, y, c, m in cph) + "]"


def _with_cph(D: LDiag, cph: Sequence[Cph]) -> str:
    text = literal(D)
    assert text.endswith("] }")
    return text[: -len(" }")] + f",\n    cph := {_cph_lean(cph)} }}"


def _tails_lean(T: Sequence[Tail]) -> str:
    return "[" + ", ".join(f"({_lport(a)}, {_lport(b)})" for a, b in T) + "]"


def export_split(before: Diagram, after: Diagram) -> str | None:
    """Lean source whose ``main`` proves the linked digit sums of ``before`` and ``after`` equal,
    ``after`` splitting one outermost node box of ``before`` into ``b`` boxes, the ``c``-th new
    box reading the box index raised by ``c`` times the copy count; None outside that form."""
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
    tables = _Tables()
    try:
        L = linked(_indexed(before, bid), tables)
        cleared = _indexed(after, bid)
        for y in tops:
            cleared = _indexed(cleared, BangBoxId(y))
        R0 = linked(cleared, tables)
    except LinkedUnsupported:
        return None
    B = sorted(int(x) for x in before.bang_boxes[bid].node_scope)
    scopes = [B] + [sorted(int(x) for x in after.bang_boxes[BangBoxId(y)].node_scope) for y in tops]
    if any(len(s) != len(B) for s in scopes):
        return None
    cps = [dict(zip(B, s, strict=True)) for s in scopes]
    copies = {x for s in scopes[1:] for x in s}
    R = _renamed(L, R0, set(B) | copies)
    if R is None:
        return None
    by_id = {n.id: n for n in R.nodes}
    if set(by_id) != {n.id for n in L.nodes} | copies:
        return None
    mults = {c: _Tables.index(tables.mults, str(sp.expand(c * half))) for c in range(1, base + 1)}
    if _Tables.index(tables.mults, str(old)) != mults[base]:
        return None
    ph = {n.id: n.ph for n in L.nodes}
    cph: list[Cph] = []
    for x in B:
        node = before.nodes[x]  # type: ignore[index]
        if after.nodes[x].phase != node.phase:  # type: ignore[index]
            return None
        reads = index is not None and node.phase is not None and index in node.phase.free_symbols
        if reads:
            cph.append((x, int(bid), 0, mults[1]))
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
            by_id[cps[c][x]] = replace(by_id[cps[c][x]], ph=ph[x])
    R.nodes = [by_id[n.id] for n in L.nodes] + [
        by_id[cp[n.id]] for cp in cps[1:] for n in L.nodes if n.id in cp
    ]
    owners = [int(bid), *tops]
    stages: list[LDiag] = [L]
    cphs: list[list[Cph]] = [cph]
    for k in range(1, base):
        if k < base - 1:
            stage = _stage(
                L,
                R,
                tops[k - 1],
                mults[base - k],
                {x for s in scopes[: k + 1] for x in s},
                _below(R, tops[k:]),
                {x for s in scopes[k + 1 :] for x in s},
            )
        else:
            stage = R
        if stage is None:
            return None
        stages.append(stage)
        cphs.append([*cphs[-1], *((cps[k][q[0]], tops[k - 1], k, mults[1]) for q in cph)])
    names = ["L", *(f"M{k}" for k in range(1, base - 1)), "L'"]
    decls = []
    for k in range(base):
        what = (
            "The diagram before the split"
            if k == 0
            else "The diagram after the split"
            if k == base - 1
            else f"The diagram after step {k} of the split"
        )
        decls.append(
            f"/-- {what}, as a linked diagram. -/\n"
            f"def {names[k]} : LDiagram :=\n  {_with_cph(stages[k], cphs[k])}\n"
        )
    proof = ",\n    ".join(["by decide +kernel"] * _FIELDS)
    hyps, sigma = [], "σ'"
    for k in range(1, base):
        Bk = scopes[k - 1]
        cp = {cps[k - 1][x]: cps[k][x] for x in B}
        T = _tails(stages[k - 1], stages[k], set(Bk), cp)
        if T is None:
            return None
        boxes = (owners[k - 1], owners[k], mults[base - k + 1], mults[1], mults[base - k])
        if not split_ok(stages[k - 1], stages[k], cphs[k - 1], cphs[k], boxes, Bk, cp, T):
            return None
        table = ", ".join(f"({x}, {y})" for x, y in cp.items())
        args = " ".join(map(str, boxes))
        decls.append(
            f"/-- Step {k}: each node of the box split to its node in the new box. -/\n"
            f"def cp{k} (x : ℕ) : ℕ := (([{table}] : List (ℕ × ℕ)).lookup x).getD 0\n\n"
            f"/-- Step {k}: each leg bundled over the box split and its leg for the new box. -/\n"
            f"def T{k} : List (Port × Port) := {_tails_lean(T)}\n\n"
            f"theorem ok{k} :\n    BoxSplitOK {names[k - 1]} {names[k]} {args} {_lnat(Bk)} "
            f"cp{k} T{k} :=\n"
            f"  ⟨{proof}⟩\n"
        )
        hyps.append(f"ks {boxes[2]} = ks {boxes[3]} + ks {boxes[4]}")
        sigma_k = f"(splitφ {_lnat(Bk)} cp{k} T{k} (ks {mults[1]}))"
        sigma = sigma.replace("@@", f"(bndMap {names[k - 1]}.base {names[k]}.base {sigma_k} @@)")
        if k == 1:
            sigma = f"(bndMap {names[0]}.base {names[1]}.base {sigma_k} @@)"
    sigma = sigma.replace("@@", "σ'")
    chain = "evalL_split ok1 E ks h1 _"
    for k in range(2, base):
        chain = f"(evalL_split ok{k} E ks h{k} _).trans ({chain})"
    head = " →\n    ".join(hyps)
    hs = " ".join(f"h{k}" for k in range(1, base))
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX ArchytasZX.LDiagram\n\n"
        + "\n".join(decls)
        + "\n/-- The linked digit sums agree, each step's moved copies read at the new box's "
        "digits. -/\n"
        f"theorem main : ∀ (E : Env) (ks : ℕ → ℕ),\n    {head} →\n"
        "    ∀ σ' : ℕ → List ℕ → ℕ,\n"
        f"      L'.evalL E ks σ' = L.evalL E ks {sigma} :=\n"
        f"  fun E ks {hs} σ' => {chain}\n\n"
        "end ArchytasZX.Export\n"
    )
