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

"""The radix window: an induction step over first-copy peels exposed by library lemmas.

One side's successor, every box peeling its first copy (:func:`~archytaszx.semantics.kernel.
repeel`), is rewritten by :data:`~archytaszx.semantics.lemmas.LIBRARY` lemmas, a phased Z
feeding a pair phase split off when no lemma applies, until the chain boxes carrying the step
index and the unboxed nodes wired only to them read as the hypothesis through cut wires. The
replaced diagram has each alignment group merged (:func:`~archytaszx.semantics.kernel.
align_merge`), each boxed Z holding a boundary slot fused into its boxed Z neighbour (reversed
:func:`~archytaszx.semantics.kernel.box_split`) and its unboxed Z spiders fused and identities
removed; the other side's first-copy peel tidied the same way meets it by isomorphism, and its
re-peel reaches the other successor.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    GraphGrammarError,
    NodeId,
    PortRef,
)
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import IDENTITY_REMOVAL
from archytaszx.semantics import kernel
from archytaszx.semantics.induction import (
    CutRegion,
    InductionDomainError,
    boxed_splits,
    find_cut_region,
    replace_cut_region,
)

MAX_ROUNDS = 12


class RadixWindowError(Exception):
    """Why the radix window does not settle a step."""


def _boxed(d: Diagram, node: NodeId) -> bool:
    return any(node in box.node_scope for box in d.bang_boxes.values())


def _partner(d: Diagram) -> dict[PortRef, PortRef]:
    out: dict[PortRef, PortRef] = {}
    for wire in d.wires:
        out[wire.a], out[wire.b] = wire.b, wire.a
    return out


def _legs(d: Diagram, node: NodeId) -> list[PortRef]:
    n = d.nodes[node]
    return [
        PortRef(node, direction, i)
        for direction in (Direction.INPUT, Direction.OUTPUT)
        for i in range(len(n.legs(direction)))
    ]


def _first_peels(source: Diagram, k: str) -> kernel.Theorem:
    """``first = successor``: every non-pair box of ``source`` peeled at its first copy, then
    at its default copy (:func:`~archytaszx.semantics.kernel.repeel`)."""
    boxes = frozenset(b for b, box in source.bang_boxes.items() if box.pair_of is None)
    return kernel.repeel(source, k, a_first=boxes)


def _core(d: Diagram, k: str, joint: bool = False) -> frozenset[NodeId]:
    """The scopes of the chain boxes whose multiplicity reads ``k``, closed under unboxed nodes
    every wired neighbour of which lies in it; with ``joint``, under the connected sets of
    unboxed nodes every wired neighbour of which outside the set lies in it."""
    core = {
        n
        for box in d.bang_boxes.values()
        if box.links and k in box.multiplicity.free_symbols
        for n in box.node_scope
    }
    if not core:
        return frozenset()
    partner = _partner(d)

    def far(node: NodeId) -> list[NodeId]:
        return [partner[r].node_id for r in _legs(d, node) if r in partner]

    grown = True
    while grown:
        grown = False
        seen: set[NodeId] = set()
        for start in sorted(d.nodes):
            if start in core or start in seen or _boxed(d, start):
                continue
            part, todo = {start}, [start]
            while todo:
                for f in far(todo.pop()):
                    if joint and f not in part and f not in core and not _boxed(d, f):
                        part.add(f)
                        todo.append(f)
            seen |= part
            outside = [f for n in part for f in far(n) if f not in part]
            if outside and all(f in core for f in outside):
                core |= part
                grown = True
    return frozenset(core)


def _pair_splits(t: kernel.Theorem) -> Iterator[kernel.Theorem]:
    """``t`` continued by each split of a Z spider (phased, of three or more outputs, or of four
    or more legs) off its output leg into a pair phase, toward its first other output
    (:func:`~archytaszx.semantics.lemmas._split_off`)."""
    from archytaszx.semantics.lemmas import LemmaError, _split_off, _split_theorem

    d = t.rhs
    partner = _partner(d)
    for node_id in sorted(d.nodes):
        node = d.nodes[node_id]
        legs = len(node.inputs) + len(node.outputs)
        if node.generator_type.name != "Z" or (
            len(node.outputs) < 3 and legs < 4 and node.phase is None
        ):
            continue
        outs = [PortRef(node_id, Direction.OUTPUT, i) for i in range(len(node.outputs))]
        into = [
            r
            for r in outs
            if r in partner and d.nodes[partner[r].node_id].generator_type.name == "Zp"
        ]
        others = [r for r in outs if r not in into]
        for leg in into:
            for via in others[:1]:
                try:
                    out, leaf, mid, hub = _split_off(d, node_id, frozenset({leg}), via)
                    split = cast("kernel.Theorem", _split_theorem(d, out, leaf, mid, hub))
                except (LemmaError, kernel.KernelError, InductionDomainError):
                    continue
                yield kernel.trans(t, split)


def _lemma_round(t: kernel.Theorem) -> kernel.Theorem | None:
    """``t`` continued by the first library lemma rewrite of its right side, else None."""
    from archytaszx.semantics.lemmas import _PROVING, LIBRARY, LemmaError, rewrite

    for entry in LIBRARY:
        if entry.name in _PROVING:
            continue
        try:
            step = cast("kernel.Theorem | None", rewrite(t.rhs, entry))
        except (LemmaError, kernel.KernelError, InductionDomainError):
            continue
        if step is not None:
            return kernel.trans(t, step)
    return None


def expose(
    peeled: Diagram | None, first: kernel.Theorem, own_k: Diagram, k: str
) -> tuple[kernel.Theorem, CutRegion]:
    """``peeled = exposed`` and the cut region of ``exposed`` reading as ``own_k``, from the
    first-copy peel ``first`` (``first.rhs`` is ``peeled``) by lemma rounds and pair splits."""
    t = kernel.symm(first)
    if peeled is not None and not kernel._same(t.lhs, peeled):
        raise RadixWindowError("the re-peel does not end at the peeled successor")
    for _ in range(MAX_ROUNDS):
        for joint in (False, True):
            core = _core(t.rhs, k, joint)
            cut = find_cut_region(t.rhs, core, own_k) if core else None
            if cut is not None:
                return t, cut
        nxt = _lemma_round(t)
        if nxt is None:
            nxt = next((r for s in _pair_splits(t) if (r := _lemma_round(s)) is not None), None)
        if nxt is None:
            raise RadixWindowError("no library lemma applies and the core is not the hypothesis")
        t = nxt
    raise RadixWindowError("the core is not the hypothesis after the lemma rounds")


def _fuse_into(d: Diagram, leaf: NodeId, hub: NodeId) -> Diagram:
    """``d`` with ``leaf`` fused into ``hub``: hub's leg to leaf replaced by leaf's boundary
    leg, leaf's other legs appended in their direction, hub's port references kept."""
    partner = _partner(d)
    slots = {*d.boundary_inputs, *d.boundary_outputs}
    con = next(r for r in _legs(d, hub) if r in partner and partner[r].node_id == leaf)
    lslot = next(r for r in _legs(d, leaf) if r in slots)
    lcon = partner[con]
    src = {
        direction: [r for r in _legs(d, hub) if r.direction is direction] for direction in Direction
    }
    src[con.direction][con.index] = lslot
    for r in _legs(d, leaf):
        if r not in (lslot, lcon):
            src[r.direction].append(r)
    out = d.copy()

    def dims(direction: Direction) -> list[Dim]:
        return [d.nodes[r.node_id].legs(r.direction)[r.index].dim for r in src[direction]]

    h = d.nodes[hub]
    v = out.add_node(h.generator_type, dims(Direction.INPUT), dims(Direction.OUTPUT), phase=h.phase)
    new = {r: PortRef(v, direction, j) for direction in src for j, r in enumerate(src[direction])}
    for wire in list(out.wires):
        if {wire.a.node_id, wire.b.node_id} & {leaf, hub}:
            out.remove_wire(wire.a, wire.b)
    for r, far in sorted(partner.items(), key=lambda item: item[0].sort_key()):
        if r in new and far.node_id not in (leaf, hub):
            out.add_wire(new[r], far)
    out.set_boundary_inputs([new.get(r, r) for r in d.boundary_inputs])
    out.set_boundary_outputs([new.get(r, r) for r in d.boundary_outputs])
    for box_id, box in sorted(d.bang_boxes.items()):
        if hub in box.node_scope:
            home = next((g for g in box.groups if hub in g), None)
            home = home or next((g for g in box.groups if leaf in g), None)
            groups = tuple(
                g2 for g in box.groups if (g2 := (g - {leaf, hub}) | ({v} if g == home else set()))
            )
            out.set_bang_box_extension(box_id, groups=())
            out.set_bang_box_node_scope(box_id, (box.node_scope - {leaf, hub}) | {v})
            out.set_bang_box_extension(box_id, groups=groups)
        if box.pair_of is not None:
            out.set_bang_box_extension(
                box_id,
                left_ports=frozenset(new.get(r, r) for r in box.left_ports),
                right_ports=frozenset(new.get(r, r) for r in box.right_ports),
            )
    out.remove_node(leaf)
    out.remove_node(hub)
    return out


def _box_fusions(t: kernel.Theorem) -> kernel.Theorem:
    """``t`` continued by fusing, while one exists, a boxed Z holding a boundary slot into a Z
    wired to it in the same boxes: the fused diagram's
    :func:`~archytaszx.semantics.induction.boxed_splits` isomorphic to ``t.rhs``, reversed."""
    while True:
        d = t.rhs
        partner = _partner(d)
        slots = {*d.boundary_inputs, *d.boundary_outputs}
        done = False
        for leaf in sorted(d.nodes):
            if done or d.nodes[leaf].generator_type.name != "Z" or d.nodes[leaf].phase is not None:
                continue
            home = {b for b, box in d.bang_boxes.items() if leaf in box.node_scope}
            legs = _legs(d, leaf)
            if not home or not any(r in slots for r in legs):
                continue
            for r in legs:
                if r not in partner:
                    continue
                hub = partner[r].node_id
                there = {b for b, box in d.bang_boxes.items() if hub in box.node_scope}
                if hub == leaf or there != home or d.nodes[hub].generator_type.name != "Z":
                    continue
                copy = frozenset(
                    partner[x].node_id for x in legs if x in partner and partner[x].node_id != hub
                )
                try:
                    fused = _fuse_into(d, leaf, hub)
                except (GraphGrammarError, StopIteration, KeyError):
                    continue
                for cand, _ in boxed_splits(fused, copy):
                    try:
                        back = kernel.iso(d, cand)
                        split = kernel.box_split(fused, cand, copy)
                    except kernel.KernelError:
                        continue
                    t = kernel.trans(t, kernel.trans(back, kernel.symm(split)))
                    done = True
                    break
                if done:
                    break
        if not done:
            return t


def _align_merges(t: kernel.Theorem) -> kernel.Theorem:
    """``t`` continued by :func:`~archytaszx.semantics.kernel.align_merge` of every alignment
    group."""
    from archytaszx.diagram.bangbox import BangBoxError, align_merged

    while True:
        roots = sorted({b.align for b in t.rhs.bang_boxes.values() if b.align is not None})
        for root in roots:
            try:
                merged = align_merged(t.rhs, root)
                step = kernel.align_merge(t.rhs, merged)
            except (BangBoxError, kernel.KernelError):
                continue
            t = kernel.trans(t, step)
            break
        else:
            return t


def _tidy(t: kernel.Theorem) -> kernel.Theorem:
    """``t`` continued by fusing wired unboxed Z spiders and removing unboxed identities while
    any is left."""
    from archytaszx.semantics.lemmas import _fusion

    while True:
        d = t.rhs
        nxt: kernel.Theorem | None = None
        for wire in sorted(d.wires, key=lambda w: w.sort_key()):
            a, b = wire.a.node_id, wire.b.node_id
            if a == b or _boxed(d, a) or _boxed(d, b):
                continue
            if d.nodes[a].generator_type.name == "Z" == d.nodes[b].generator_type.name:
                nxt = kernel.trans(t, cast("kernel.Theorem", _fusion(d, a, b)[0]))
                break
        if nxt is None:
            for match in IDENTITY_REMOVAL.pattern.find_matches(d):
                step = apply(d, IDENTITY_REMOVAL, match).step
                if any(_boxed(d, n) for n in step.consumed_node_ids):
                    continue
                nxt = kernel.trans(t, kernel.rule(d, step))
                break
        if nxt is None:
            return t
        t = nxt


def _slot_groups(t: kernel.Theorem) -> kernel.Theorem:
    """``t`` continued by :func:`~archytaszx.semantics.kernel.regroup` to each box group cut
    down to its boundary-slot nodes, empty groups dropped, in the order of their first slots."""
    d = t.rhs
    order = [r.node_id for r in (*d.boundary_inputs, *d.boundary_outputs)]
    slots = set(order)
    new = d.copy()
    changed = False
    for box_id, box in d.bang_boxes.items():
        kept_groups = [kept for g in box.groups if (kept := g & slots)]
        groups = tuple(sorted(kept_groups, key=lambda g: min(order.index(n) for n in g)))
        if groups != box.groups:
            new.set_bang_box_extension(box_id, groups=groups)
            changed = True
    return kernel.regroup(t, t.lhs, new) if changed else t


def close(
    replaced: Diagram, other_source: Diagram, other_peeled: Diagram | None, k: str
) -> kernel.Theorem:
    """``replaced = other_peeled``: ``replaced`` merged, box-fused and tidied, isomorphic to the
    tidied first-copy peel of ``other_source``, then its re-peel."""
    t = _slot_groups(_tidy(_box_fusions(_slot_groups(_align_merges(kernel.refl(replaced))))))
    moved = _first_peels(other_source, k)
    if other_peeled is not None and not kernel._same(moved.rhs, other_peeled):
        raise RadixWindowError("the other re-peel does not end at the other peeled successor")
    u = _slot_groups(_tidy(kernel.refl(moved.lhs)))
    meet = kernel.iso(t.rhs, u.rhs)
    return kernel.trans(kernel.trans(t, kernel.trans(meet, kernel.symm(u))), moved)


def step(
    own_source: Diagram,
    own_peeled: Diagram | None,
    own_k: Diagram,
    other_source: Diagram,
    other_peeled: Diagram | None,
    other_k: Diagram,
    k: str,
) -> tuple[kernel.Theorem, CutRegion, kernel.Theorem]:
    """The exposing theorem, its cut region and the step theorem of the radix window; a peeled
    successor given as None is not checked."""
    try:
        exposing, cut = expose(own_peeled, _first_peels(own_source, k), own_k, k)
        replaced = replace_cut_region(exposing.rhs, cut, own_k, other_k)
        return exposing, cut, close(replaced, other_source, other_peeled, k)
    except (kernel.KernelError, InductionDomainError) as exc:
        raise RadixWindowError(str(exc)) from exc


def _slot_grouped(d: Diagram) -> Diagram:
    """``d`` with each ungrouped top-level box holding two or more boundary-slot nodes grouped
    one slot node per group, in the order of their first slots."""
    order = [r.node_id for r in (*d.boundary_inputs, *d.boundary_outputs)]
    out = d.copy()
    for box_id, box in d.bang_boxes.items():
        if box.parent is not None or box.groups or box.pair_of is not None:
            continue
        held = sorted({n for n in order if n in box.node_scope}, key=order.index)
        if len(held) >= 2:
            out.set_bang_box_extension(box_id, groups=tuple(frozenset({n}) for n in held))
    return out


def _chain_parts(d: Diagram, box_id: BangBoxId) -> list[frozenset[NodeId]]:
    """The connected parts of a chain box's scope, joined by wires inside it and by links, in
    the order of the box's groups."""
    box = d.bang_boxes[box_id]
    scope = box.node_scope
    root = {n: n for n in scope}

    def find(n: NodeId) -> NodeId:
        while root[n] != n:
            n = root[n]
        return n

    pairs = [(w.a.node_id, w.b.node_id) for w in d.wires]
    pairs += [(p.node_id, q.node_id) for p, q in box.links]
    for a, b in pairs:
        if a in scope and b in scope:
            root[find(a)] = find(b)
    parts: dict[NodeId, set[NodeId]] = {}
    for n in scope:
        parts.setdefault(find(n), set()).add(n)
    rank = {n: i for i, g in enumerate(box.groups) for n in g}
    found = [frozenset(p) for p in parts.values()]
    return sorted(found, key=lambda p: min((rank.get(n, len(rank)), n) for n in p))


def _chain_splits(d: Diagram) -> kernel.Theorem:
    """``d = d'``: every chain box over two or more connected parts split into aligned chain
    boxes (:func:`~archytaszx.semantics.kernel.chain_split`) while one splits."""
    from archytaszx.diagram.bangbox import BangBoxError, chain_splitted

    t = kernel.refl(d)
    while True:
        cur = t.rhs
        for box_id, box in sorted(cur.bang_boxes.items()):
            if not box.links or box.parent is not None:
                continue
            parts = _chain_parts(cur, box_id)
            if len(parts) < 2:
                continue
            try:
                split = chain_splitted(cur, box_id, parts)
                t = kernel.trans(t, kernel.chain_split(cur, split))
            except (BangBoxError, kernel.KernelError):
                continue
            break
        else:
            return t


def regrouped_induction(left: Diagram, right: Diagram, index: str, k: str) -> kernel.Theorem:
    """``left = right`` by :func:`~archytaszx.semantics.kernel.regroup` of the kernel's
    induction over ``index`` between one side with its slot nodes grouped
    (:func:`_slot_grouped`) and its chain boxes split (:func:`_chain_splits`), the exposed side
    of :func:`step`, and the other side grouped; the base by
    :func:`~archytaszx.semantics.kernel.contraction`."""
    from archytaszx.semantics.prove import (
        InductionSide,
        induction_base,
        induction_hypothesis,
        induction_renamed,
        induction_successor,
    )

    grouped = (_slot_grouped(left), _slot_grouped(right))
    reasons: list[str] = []
    for own_at in (0, 1):
        try:
            splits = _chain_splits(grouped[own_at])
            own, other = splits.rhs, grouped[1 - own_at]
            if kernel._same(own, grouped[own_at]):
                raise RadixWindowError("no chain box splits")
            exposing, cut, st = step(
                induction_renamed(own, index, k),
                induction_successor(own, index, k),
                induction_hypothesis(own, index, k),
                induction_renamed(other, index, k),
                induction_successor(other, index, k),
                induction_hypothesis(other, index, k),
                k,
            )
            base = kernel.contraction(induction_base(own, index), induction_base(other, index))
            proved = kernel.induction(
                own, other, index, k, base, InductionSide.START, exposing, cut.region, st, cut=cut
            )
            t = kernel.trans(splits, proved)
            if own_at == 1:
                t = kernel.symm(t)
            return kernel.regroup(t, left, right)
        except (RadixWindowError, kernel.KernelError, InductionDomainError) as exc:
            reasons.append(f"side {own_at}: {exc}")
    raise RadixWindowError("; ".join(reasons))
