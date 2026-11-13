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

"""The carry left open after a chain: a pair ``left, right`` read as ``left = T o L'`` and
``right = M o R'``.

``left`` holds a chain box over the index whose tail (the nodes reached from its link-out
ends, apart from those reached from its link-in ends, :func:`carry_tail`) is, per link-out
end, a path ``U`` of Fourier boxes and phase-free one-in one-out spiders, then a finite part
``M`` at the ends of every path, on boundary outputs of ``left`` only. ``L'`` is ``left``
without the tail, the link-out ends appended to its outputs. ``right`` holds, at ``M``'s
output slots, a region ``S`` isomorphic to ``M``; ``R'`` is ``right`` without ``S``, the far
end of ``S``'s input ``i`` continued by the Fourier power inverting path ``i`` and appended
to its outputs. :func:`open_carry` builds both and ``left`` with ``L'`` replaced by ``R'``,
which equals ``right`` when ``M o U o U^-1 = M``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.induction import (
    CutRegion,
    InductionDomainError,
    _chain_owner,
    _cut_extract,
    _cut_wires,
    _partner_of,
    _ref_dim,
    find_cut_region,
    replace_cut_region,
)


@dataclass(frozen=True)
class OpenCarry:
    """``left_open`` (L'), ``right_open`` (R'), ``cut`` (L' as a region of ``left``) and
    ``closed`` (``left`` with that region replaced by R')."""

    left_open: Diagram
    right_open: Diagram
    cut: CutRegion
    closed: Diagram


def _neighbours(diagram: Diagram) -> dict[NodeId, set[NodeId]]:
    out: dict[NodeId, set[NodeId]] = {n: set() for n in diagram.nodes}
    for wire in diagram.wires:
        out[wire.a.node_id].add(wire.b.node_id)
        out[wire.b.node_id].add(wire.a.node_id)
    return out


def carry_tail(diagram: Diagram, index: str) -> tuple[tuple[PortRef, ...], frozenset[NodeId]]:
    """The link-out ends of the chain box over ``index``, sorted, and the nodes off the box
    reached from them without passing a box node; InductionDomainError when there is no such
    box or those nodes also reach a link-in end."""
    owner = _chain_owner(diagram, index)
    if owner is None:
        raise InductionDomainError("no chain box carries the index")
    box = diagram.bang_boxes[owner]
    scope = box.node_scope
    neighbours = _neighbours(diagram)

    def reached(ends: Sequence[PortRef]) -> set[NodeId]:
        seen: set[NodeId] = set()
        todo = [
            far.node_id
            for end in ends
            for far in (_partner_of(diagram, end),)
            if far is not None and far.node_id not in scope
        ]
        while todo:
            node_id = todo.pop()
            if node_id not in seen:
                seen.add(node_id)
                todo += [n for n in neighbours[node_id] if n not in scope and n not in seen]
        return seen

    outs = tuple(sorted((p for p, _q in box.links), key=PortRef.sort_key))
    tail = reached(outs)
    if not tail:
        raise InductionDomainError("nothing follows the chain")
    if tail & reached([q for _p, q in box.links]):
        raise InductionDomainError("the chain's tail meets its head")
    return outs, frozenset(tail)


def _unitary(diagram: Diagram, node_id: NodeId) -> int | None:
    """The Fourier power of a one-in one-out node: 1 for F, 0 for a phase-free Z or X spider,
    else None."""
    node = diagram.nodes[node_id]
    if node.num_inputs != 1 or node.num_outputs != 1:
        return None
    if node.generator_type is FOURIER_BOX:
        return 1
    if node.generator_type in (Z_SPIDER, X_SPIDER) and node.phase is None:
        return 0
    return None


def _paths(
    diagram: Diagram, ends: Sequence[PortRef], tail: frozenset[NodeId]
) -> tuple[list[PortRef], list[int], frozenset[NodeId]]:
    """Per link-out end, the input port its unitary path ``U`` runs into and the path's Fourier
    power mod 4; and every path node."""
    stops: list[PortRef] = []
    powers: list[int] = []
    walked: set[NodeId] = set()
    for end in ends:
        power, at = 0, _partner_of(diagram, end)
        while at is not None and at.direction is Direction.INPUT and at.node_id in tail:
            step = _unitary(diagram, at.node_id)
            if step is None:
                break
            power += step
            walked.add(at.node_id)
            at = _partner_of(diagram, PortRef(at.node_id, Direction.OUTPUT, 0))
        if at is None or at.direction is not Direction.INPUT or at.node_id not in tail:
            raise InductionDomainError("a carry path does not run into the tail's finite part")
        stops.append(at)
        powers.append(power % 4)
    return stops, powers, frozenset(walked)


def open_carry(left: Diagram, right: Diagram, index: str) -> OpenCarry:
    """L', R', L' as a region of ``left`` and ``left`` with it replaced by R' (see the module
    docstring); InductionDomainError when ``left`` or ``right`` does not have that shape."""
    ends, tail = carry_tail(left, index)
    if any(_footprint_hits(left, tail)):
        raise InductionDomainError("the chain's tail lies in a bang box")
    if any(r.node_id in tail for r in left.boundary_inputs):
        raise InductionDomainError("the chain's tail holds a boundary input")
    stops, powers, walked = _paths(left, ends, tail)
    finite = tail - walked
    if len(set(stops)) != len(stops):
        raise InductionDomainError("two carry paths run into one port")
    for inside, far in _cut_wires(left, finite).items():
        if far.node_id not in walked or inside not in stops:
            raise InductionDomainError("the tail's finite part meets more than the carry paths")
    slots = [i for i, r in enumerate(left.boundary_outputs) if r.node_id in tail]
    if any(left.boundary_outputs[i].node_id not in finite for i in slots):
        raise InductionDomainError("a carry path ends at a boundary output")
    if len(right.boundary_outputs) != len(left.boundary_outputs):
        raise InductionDomainError("the two sides have different output counts")
    region = frozenset(left.nodes) - tail
    cut = CutRegion(
        region,
        tuple(left.boundary_inputs),
        tuple([r for r in left.boundary_outputs if r.node_id not in tail] + list(ends)),
        MappingProxyType(_cut_wires(left, region)),
    )
    left_open = _cut_extract(left, cut)
    left_open.set_scalar(left.scalar)
    finite_cut = CutRegion(
        finite,
        tuple(stops),
        tuple(left.boundary_outputs[i] for i in slots),
        MappingProxyType(_cut_wires(left, finite)),
    )
    measured = _cut_extract(left, finite_cut)
    held = frozenset(right.boundary_outputs[i].node_id for i in slots)
    if len(held) != len(finite) or any(r.node_id in held for r in right.boundary_inputs):
        raise InductionDomainError("the right side's output slots hold no copy of the tail")
    if any(_footprint_hits(right, held)):
        raise InductionDomainError("the right side's copy of the tail lies in a bang box")
    match = find_cut_region(right, held, measured)
    if match is None or match.outputs != tuple(right.boundary_outputs[i] for i in slots):
        raise InductionDomainError("the right side's output slots hold no copy of the tail")
    right_open = right.copy()
    for node_id in sorted(held):
        right_open.remove_node(node_id)
    new_ends: list[PortRef] = []
    for inside, power in zip(match.inputs, powers, strict=True):
        at = match.cuts.get(inside)
        if at is None or at.direction is not Direction.OUTPUT:
            raise InductionDomainError("the right side's copy of the tail is not fed by wires")
        dim = _ref_dim(right, at)
        for _ in range((4 - power) % 4):
            node = right_open.add_node(FOURIER_BOX, [dim], [dim])
            right_open.add_wire(at, PortRef(node, Direction.INPUT, 0))
            at = PortRef(node, Direction.OUTPUT, 0)
        new_ends.append(at)
    right_open.set_boundary_outputs(
        [r for r in right.boundary_outputs if r.node_id not in held] + new_ends
    )
    closed = replace_cut_region(left, cut, left_open, right_open, contiguous=False)
    return OpenCarry(left_open, right_open, cut, closed)


def _footprint_hits(diagram: Diagram, nodes: frozenset[NodeId]) -> list[bool]:
    """Per bang box, whether its node scope or port scope touches ``nodes``."""
    return [
        bool(box.node_scope & nodes) or any(r.node_id in nodes for r in box.port_scope)
        for box in diagram.bang_boxes.values()
    ]
