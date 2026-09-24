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

"""Structural comparison of two diagrams: id for id, and up to renaming."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from archytaszx.diagram.bangbox import BangBox
from archytaszx.diagram.graph import BangBoxId, Diagram, NodeId, PortRef, Wire


@dataclass(frozen=True, slots=True)
class StructuralComparison:
    """The outcome of comparing two diagrams id for id: whether they agree, and where they
    first differ."""

    identical: bool
    reason: str


def compare_structure(a: Diagram, b: Diagram) -> StructuralComparison:
    """Compare two diagrams node id for node id, reporting the first difference in a fixed
    check order."""
    if sorted(a.nodes) != sorted(b.nodes):
        only_a = sorted(set(a.nodes) - set(b.nodes))
        only_b = sorted(set(b.nodes) - set(a.nodes))
        return StructuralComparison(
            False, f"node ids differ: only in a {only_a!r}, only in b {only_b!r}"
        )

    for nid in sorted(a.nodes):
        node_a = a.nodes[nid]
        node_b = b.nodes[nid]
        if node_a.generator_type != node_b.generator_type:
            return StructuralComparison(
                False,
                f"node {nid}: generator_type differs: "
                f"{node_a.generator_type.name} vs {node_b.generator_type.name}",
            )
        if node_a.inputs != node_b.inputs:
            left = tuple(p.dim for p in node_a.inputs)
            right = tuple(p.dim for p in node_b.inputs)
            return StructuralComparison(False, f"node {nid}: inputs differs: {left!r} vs {right!r}")
        if node_a.outputs != node_b.outputs:
            left = tuple(p.dim for p in node_a.outputs)
            right = tuple(p.dim for p in node_b.outputs)
            return StructuralComparison(
                False, f"node {nid}: outputs differs: {left!r} vs {right!r}"
            )
        if node_a.phase != node_b.phase:
            left_phase = str(node_a.phase) if node_a.phase is not None else "None"
            right_phase = str(node_b.phase) if node_b.phase is not None else "None"
            return StructuralComparison(
                False, f"node {nid}: phase differs: {left_phase} vs {right_phase}"
            )

    wires_a = sorted(a.wires, key=Wire.sort_key)
    wires_b = sorted(b.wires, key=Wire.sort_key)
    if wires_a != wires_b:
        wire_only_a = sorted(a.wires - b.wires, key=Wire.sort_key)
        wire_only_b = sorted(b.wires - a.wires, key=Wire.sort_key)
        return StructuralComparison(
            False, f"wires differ: only in a {wire_only_a!r}, only in b {wire_only_b!r}"
        )

    if sorted(a.bang_boxes) != sorted(b.bang_boxes):
        box_only_a = sorted(set(a.bang_boxes) - set(b.bang_boxes))
        box_only_b = sorted(set(b.bang_boxes) - set(a.bang_boxes))
        return StructuralComparison(
            False, f"bang box ids differ: only in a {box_only_a!r}, only in b {box_only_b!r}"
        )

    for box_id in sorted(a.bang_boxes):
        box_a = a.bang_boxes[box_id]
        box_b = b.bang_boxes[box_id]
        if box_a.multiplicity != box_b.multiplicity:
            return StructuralComparison(
                False,
                f"bang box {box_id}: multiplicity differs: "
                f"{box_a.multiplicity!r} vs {box_b.multiplicity!r}",
            )
        if sorted(box_a.node_scope) != sorted(box_b.node_scope):
            return StructuralComparison(
                False,
                f"bang box {box_id}: node scope differs: "
                f"{sorted(box_a.node_scope)!r} vs {sorted(box_b.node_scope)!r}",
            )
        scope_a = sorted(box_a.port_scope, key=PortRef.sort_key)
        scope_b = sorted(box_b.port_scope, key=PortRef.sort_key)
        if scope_a != scope_b:
            return StructuralComparison(
                False, f"bang box {box_id}: port scope differs: {scope_a!r} vs {scope_b!r}"
            )
        if box_a.parent != box_b.parent:
            return StructuralComparison(
                False,
                f"bang box {box_id}: parent differs: {box_a.parent!r} vs {box_b.parent!r}",
            )

    if a.boundary_inputs != b.boundary_inputs:
        return StructuralComparison(
            False,
            f"boundary inputs differ: {a.boundary_inputs!r} vs {b.boundary_inputs!r}",
        )

    if a.boundary_outputs != b.boundary_outputs:
        return StructuralComparison(
            False,
            f"boundary outputs differ: {a.boundary_outputs!r} vs {b.boundary_outputs!r}",
        )

    if a.scalar != b.scalar:
        return StructuralComparison(False, f"scalar differs: {a.scalar!r} vs {b.scalar!r}")

    items_a = sorted(a.parameters.items(), key=lambda kv: kv[0])
    items_b = sorted(b.parameters.items(), key=lambda kv: kv[0])
    if items_a != items_b:
        return StructuralComparison(
            False, f"parameter environments differ: {items_a!r} vs {items_b!r}"
        )

    return StructuralComparison(True, "identical")


def _digest(payload: object) -> str:
    """A hash of ``payload``'s ``repr``, stable across processes."""
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:32]


def _box_chain(diagram: Diagram, box: BangBox) -> tuple[str, ...]:
    """The multiplicity reprs of ``box``'s ancestors, outermost last, cut at the first repeat."""
    chain: list[str] = []
    seen: set[BangBoxId] = {box.id}
    parent = box.parent
    while parent is not None and parent in diagram.bang_boxes and parent not in seen:
        seen.add(parent)
        ancestor = diagram.bang_boxes[parent]
        chain.append(repr(ancestor.multiplicity))
        parent = ancestor.parent
    return tuple(chain)


def _initial_colours(diagram: Diagram) -> dict[NodeId, str]:
    """A colour per node from its own id-independent data: generator, port dims, phase,
    boundary positions and bang-box membership."""
    boundary: dict[NodeId, list[tuple[str, int, int]]] = {nid: [] for nid in diagram.nodes}
    for side, refs in (("in", diagram.boundary_inputs), ("out", diagram.boundary_outputs)):
        for position, ref in enumerate(refs):
            if ref.node_id in boundary:
                boundary[ref.node_id].append(
                    (f"{side}:{ref.direction.value}:{ref.index}", position, 0)
                )

    membership: dict[NodeId, list[tuple[str, ...]]] = {nid: [] for nid in diagram.nodes}
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        static = (repr(box.multiplicity), *_box_chain(diagram, box))
        for nid in sorted(box.node_scope):
            if nid in membership:
                membership[nid].append(("node", *static))
        for ref in sorted(box.port_scope, key=PortRef.sort_key):
            if ref.node_id in membership:
                membership[ref.node_id].append(
                    ("port", ref.direction.value, str(ref.index), *static)
                )

    colours: dict[NodeId, str] = {}
    for nid in sorted(diagram.nodes):
        node = diagram.nodes[nid]
        colours[nid] = _digest(
            (
                node.generator_type.name,
                tuple(repr(port.dim) for port in node.inputs),
                tuple(repr(port.dim) for port in node.outputs),
                repr(node.phase),
                tuple(sorted(boundary[nid])),
                tuple(sorted(membership[nid])),
            )
        )
    return colours


def _incidence(diagram: Diagram) -> dict[NodeId, list[tuple[str, int, NodeId, str, int]]]:
    """Each node's wire ends, as ``(own direction, own index, other node, other direction,
    other index)``."""
    incidence: dict[NodeId, list[tuple[str, int, NodeId, str, int]]] = {
        nid: [] for nid in diagram.nodes
    }
    for wire in sorted(diagram.wires, key=Wire.sort_key):
        for near, far in ((wire.a, wire.b), (wire.b, wire.a)):
            if near.node_id in incidence:
                incidence[near.node_id].append(
                    (near.direction.value, near.index, far.node_id, far.direction.value, far.index)
                )
    return incidence


def _refine(diagram: Diagram) -> dict[NodeId, str]:
    """Colours refined by repeated rounds of folding each node's neighbours' colours into
    its own, until the partition stops splitting."""
    colours = _initial_colours(diagram)
    incidence = _incidence(diagram)
    for _round in range(len(colours) + 1):
        updated = {
            nid: _digest(
                (
                    colours[nid],
                    tuple(
                        sorted(
                            (own_dir, own_index, colours[other], other_dir, other_index)
                            for own_dir, own_index, other, other_dir, other_index in incidence[nid]
                        )
                    ),
                )
            )
            for nid in sorted(colours)
        }
        if _classes(updated) == _classes(colours):
            return colours
        colours = updated
    return colours


def _classes(colours: dict[NodeId, str]) -> tuple[tuple[NodeId, ...], ...]:
    """The partition ``colours`` induces, as sorted node-id blocks in sorted colour order."""
    blocks: dict[str, list[NodeId]] = {}
    for nid in sorted(colours):
        blocks.setdefault(colours[nid], []).append(nid)
    return tuple(tuple(blocks[colour]) for colour in sorted(blocks))


def _box_descriptors(diagram: Diagram, label: dict[NodeId, str]) -> tuple[tuple[object, ...], ...]:
    """Every bang box rewritten in terms of ``label`` instead of node ids, sorted."""
    descriptors: list[tuple[object, ...]] = []
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        descriptors.append(
            (
                repr(box.multiplicity),
                _box_chain(diagram, box),
                tuple(sorted(label[nid] for nid in box.node_scope if nid in label)),
                tuple(
                    sorted(
                        (label[ref.node_id], ref.direction.value, ref.index)
                        for ref in box.port_scope
                        if ref.node_id in label
                    )
                ),
            )
        )
    return tuple(sorted(descriptors, key=repr))


def _profile(diagram: Diagram, label: dict[NodeId, str]) -> tuple[object, ...]:
    """The whole diagram rewritten in terms of ``label`` instead of node ids."""
    wires = sorted(
        tuple(
            sorted(
                (
                    (label[wire.a.node_id], wire.a.direction.value, wire.a.index),
                    (label[wire.b.node_id], wire.b.direction.value, wire.b.index),
                )
            )
        )
        for wire in diagram.wires
        if wire.a.node_id in label and wire.b.node_id in label
    )
    return (
        tuple(sorted(label[nid] for nid in label)),
        tuple(wires),
        tuple(
            (label[ref.node_id], ref.direction.value, ref.index) for ref in diagram.boundary_inputs
        ),
        tuple(
            (label[ref.node_id], ref.direction.value, ref.index) for ref in diagram.boundary_outputs
        ),
        _box_descriptors(diagram, label),
        repr(diagram.scalar),
        tuple(sorted(diagram.parameters.items())),
    )


def canonical_key(diagram: Diagram) -> str:
    """A hashable key for ``diagram`` up to node-id and bang-box-id renaming, from colour
    refinement over the graph's invariants.

    Isomorphic diagrams always share a key; distinct diagrams may collide, so a key match is
    a candidate that :func:`isomorphic` decides.
    """
    if not isinstance(diagram, Diagram):
        raise TypeError(f"canonical_key requires a Diagram, got {type(diagram).__name__}")
    return _digest(_profile(diagram, _refine(diagram)))


def _extend(
    order: tuple[NodeId, ...],
    depth: int,
    mapping: dict[NodeId, NodeId],
    used: set[NodeId],
    candidates: dict[NodeId, tuple[NodeId, ...]],
    incidence_a: dict[NodeId, list[tuple[str, int, NodeId, str, int]]],
    wires_b: frozenset[tuple[tuple[int, str, int], tuple[int, str, int]]],
    degree_b: dict[NodeId, int],
) -> bool:
    """Search for a node bijection extending ``mapping``, consistent on every wire whose
    ends are both already assigned."""
    if depth == len(order):
        return True
    source = order[depth]
    for target in candidates[source]:
        if target in used or degree_b[target] != len(incidence_a[source]):
            continue
        mapping[source] = target
        if all(
            (
                min(
                    (int(target), own_dir, own_index),
                    (int(mapping[other]), other_dir, other_index),
                ),
                max(
                    (int(target), own_dir, own_index),
                    (int(mapping[other]), other_dir, other_index),
                ),
            )
            in wires_b
            for own_dir, own_index, other, other_dir, other_index in incidence_a[source]
            if other in mapping
        ):
            used.add(target)
            if _extend(
                order,
                depth + 1,
                mapping,
                used,
                candidates,
                incidence_a,
                wires_b,
                degree_b,
            ):
                return True
            used.discard(target)
        del mapping[source]
    return False


def isomorphic(a: Diagram, b: Diagram) -> bool:
    """Whether a node-id and bang-box-id renaming carries ``a`` onto ``b`` exactly, scalar
    and parameter environment included."""
    if not isinstance(a, Diagram) or not isinstance(b, Diagram):
        raise TypeError(
            f"isomorphic requires two Diagram instances, got {type(a).__name__} "
            f"and {type(b).__name__}"
        )
    if (
        len(a.nodes) != len(b.nodes)
        or len(a.wires) != len(b.wires)
        or len(a.bang_boxes) != len(b.bang_boxes)
        or len(a.boundary_inputs) != len(b.boundary_inputs)
        or len(a.boundary_outputs) != len(b.boundary_outputs)
        or a.scalar != b.scalar
        or sorted(a.parameters.items()) != sorted(b.parameters.items())
    ):
        return False

    colours_a = _refine(a)
    colours_b = _refine(b)
    by_colour: dict[str, list[NodeId]] = {}
    for nid in sorted(colours_b):
        by_colour.setdefault(colours_b[nid], []).append(nid)
    candidates = {nid: tuple(by_colour.get(colours_a[nid], ())) for nid in sorted(colours_a)}
    if any(not choices for choices in candidates.values()):
        return False

    incidence_a = _incidence(a)
    incidence_b = _incidence(b)
    degree_b = {nid: len(ends) for nid, ends in incidence_b.items()}
    wires_b = frozenset(wire.sort_key() for wire in b.wires)
    order = tuple(sorted(colours_a, key=lambda nid: (len(candidates[nid]), colours_a[nid], nid)))

    mapping: dict[NodeId, NodeId] = {}
    if not _extend(order, 0, mapping, set(), candidates, incidence_a, wires_b, degree_b):
        return False
    label = {nid: str(int(mapping[nid])) for nid in mapping}
    identity = {nid: str(int(nid)) for nid in b.nodes}
    return _profile(a, label) == _profile(b, identity)
