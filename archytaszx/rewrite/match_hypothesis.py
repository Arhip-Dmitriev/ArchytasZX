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

"""An induction hypothesis ``L(k) = R(k)`` as a pattern over bang-boxed diagrams.

* :class:`HypothesisPattern` peels ``c`` copies off every owner of ``k`` in the host, ``c``
  the constant ``M' - m_b(k)`` of the host boxes against the hypothesis boxes, then embeds
  ``L(k)`` in the remainder: nodes by generator, dimensions and phase, wires port for port,
  each box of ``L`` onto one host box with the mapped scope, links and fields.
* :func:`apply_hypothesis` swaps the embedded ``L(k)`` for ``R(k)``: cut wires rejoined slot
  for slot, boxed slots kept in their boundary positions, scalar times ``R / L``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.bangbox import BangBox, BangBoxError, add_box_image, peel_owners
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.rule import (
    DimensionConstraint,
    Match,
    Pattern,
    SideCondition,
    SideConditionOutcome,
)

HYPOTHESIS_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("peel_constant", "every host owner of k exceeds its hypothesis box by c"),
    SideCondition("embedding", "L(k) embeds node for node, wire for wire, box for box"),
)


@dataclass(frozen=True, slots=True, eq=False)
class HypothesisMatch:
    """``L(k)`` embedded in ``host`` (the matched diagram with ``peels`` copies peeled) by
    ``node_map`` and ``box_map``."""

    peels: int
    host: Diagram
    node_map: Mapping[NodeId, NodeId]
    box_map: Mapping[BangBoxId, BangBoxId]

    @property
    def side_condition_outcomes(self) -> tuple[SideConditionOutcome, ...]:
        """Both declared conditions, passed."""
        return tuple(
            SideConditionOutcome(c.name, True, c.description) for c in HYPOTHESIS_SIDE_CONDITIONS
        )

    @property
    def dimension_constraints(self) -> tuple[DimensionConstraint, ...]:
        """None."""
        return ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """Always True."""
        return True

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The host images of ``L(k)``'s nodes."""
        return tuple(sorted(set(self.node_map.values())))


def _partners(diagram: Diagram) -> dict[PortRef, PortRef]:
    """Each wired port's far end."""
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    return partner


def _ports(diagram: Diagram, node_id: NodeId) -> Iterator[PortRef]:
    """Every port of one node."""
    node = diagram.nodes[node_id]
    for direction in (Direction.INPUT, Direction.OUTPUT):
        for index in range(len(node.legs(direction))):
            yield PortRef(node_id, direction, index)


def _same_node(pattern: Diagram, p: NodeId, host: Diagram, h: NodeId) -> bool:
    """Same generator, leg dimensions and phase."""
    a, b = pattern.nodes[p], host.nodes[h]
    return (
        a.generator_type == b.generator_type
        and [x.dim for x in a.inputs] == [x.dim for x in b.inputs]
        and [x.dim for x in a.outputs] == [x.dim for x in b.outputs]
        and a.phase == b.phase
    )


def _moved(ref: PortRef, node_map: Mapping[NodeId, NodeId]) -> PortRef:
    return PortRef(node_map[ref.node_id], ref.direction, ref.index)


def slot_owners(diagram: Diagram, ref: PortRef) -> tuple[BangBoxId, ...]:
    """The boxes replicating boundary slot ``ref``: node-scope boxes over its node unless
    ``ref`` is one of their link ports, and port-scope boxes holding it."""
    return tuple(
        box_id
        for box_id, box in sorted(diagram.bang_boxes.items())
        if (ref.node_id in box.node_scope and not any(ref in link for link in box.links))
        or ref in box.port_scope
    )


def _box_image(
    box: BangBox,
    host_id: BangBoxId,
    node_map: Mapping[NodeId, NodeId],
    box_map: Mapping[BangBoxId, BangBoxId],
) -> BangBox | None:
    """``box`` in host coordinates, or None when a box field it names has no image."""

    refs = (box.parent, box.pair_of, box.feed, box.align)
    if any(r is not None and r not in box_map for r in refs):
        return None
    parent, pair_of, feed, align = (None if r is None else box_map[r] for r in refs)
    return replace(
        box,
        id=host_id,
        node_scope=frozenset(node_map[n] for n in box.node_scope),
        port_scope=frozenset(_moved(r, node_map) for r in box.port_scope),
        parent=parent,
        links=tuple((_moved(x, node_map), _moved(y, node_map)) for x, y in box.links),
        pair_of=pair_of,
        left_ports=frozenset(_moved(r, node_map) for r in box.left_ports),
        right_ports=frozenset(_moved(r, node_map) for r in box.right_ports),
        tree=tuple(_moved(r, node_map) for r in box.tree),
        tree_parent=None if box.tree_parent is None else _moved(box.tree_parent, node_map),
        feed=feed,
        feed_port=None if box.feed_port is None else _moved(box.feed_port, node_map),
        groups=tuple(frozenset(node_map[n] for n in g) for g in box.groups),
        align=align,
    )


def _footprint(box: BangBox) -> frozenset[NodeId]:
    return box.node_scope | frozenset(r.node_id for r in box.port_scope)


def _boxes_of(diagram: Diagram) -> dict[NodeId, frozenset[BangBoxId]]:
    """Per node, the boxes whose footprint holds it."""
    out: dict[NodeId, set[BangBoxId]] = {n: set() for n in diagram.nodes}
    for box_id, box in diagram.bang_boxes.items():
        for n in _footprint(box):
            out[n].add(box_id)
    return {n: frozenset(s) for n, s in out.items()}


def _match_boxes(
    pattern: Diagram, host: Diagram, node_map: Mapping[NodeId, NodeId]
) -> dict[BangBoxId, BangBoxId] | None:
    """The host box per pattern box over the mapped footprint with every field mapped, every
    host box meeting the image among them; None otherwise."""
    image = frozenset(node_map.values())
    by_footprint: dict[frozenset[NodeId], list[BangBoxId]] = {}
    for box_id, box in sorted(host.bang_boxes.items()):
        foot = _footprint(box)
        if foot & image:
            if not foot <= image:
                return None
            by_footprint.setdefault(foot, []).append(box_id)
    if sum(len(v) for v in by_footprint.values()) != len(pattern.bang_boxes):
        return None
    box_map: dict[BangBoxId, BangBoxId] = {}
    pending = sorted(pattern.bang_boxes)
    while pending:
        progressed = False
        for box_id in list(pending):
            box = pattern.bang_boxes[box_id]
            refs = [box.parent, box.pair_of, box.feed, box.align]
            if any(r is not None and r not in box_map for r in refs if r != box_id):
                continue
            foot = frozenset(node_map[n] for n in _footprint(box))
            chosen = None
            for host_id in by_footprint.get(foot, []):
                if host_id in box_map.values():
                    continue
                trial = {**box_map, box_id: host_id}
                if _box_image(box, host_id, node_map, trial) == host.bang_boxes[host_id]:
                    chosen = host_id
                    break
            if chosen is None:
                return None
            box_map[box_id] = chosen
            pending.remove(box_id)
            progressed = True
        if not progressed:
            return None
    return box_map


def embeddings(pattern: Diagram, host: Diagram) -> Iterator[dict[NodeId, NodeId]]:
    """Every injective node map from ``pattern`` into ``host`` keeping generators, dimensions,
    phases and every wire port for port, each pattern boundary port landing on a host port
    wired off the image or on the host boundary, a boxed slot on the host boundary."""
    if not pattern.nodes:
        return
    p_partner, h_partner = _partners(pattern), _partners(host)
    p_boundary = frozenset([*pattern.boundary_inputs, *pattern.boundary_outputs])
    h_boundary = frozenset([*host.boundary_inputs, *host.boundary_outputs])
    boxed_slots = frozenset(ref for ref in p_boundary if slot_owners(pattern, ref))
    p_boxed = {n for n, s in _boxes_of(pattern).items() if s}
    h_boxed = {n for n, s in _boxes_of(host).items() if s}
    order: list[NodeId] = []
    seen: set[NodeId] = set()
    for start in sorted(pattern.nodes):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            n = queue.pop(0)
            order.append(n)
            for port in _ports(pattern, n):
                far = p_partner.get(port)
                if far is not None and far.node_id not in seen:
                    seen.add(far.node_id)
                    queue.append(far.node_id)

    def consistent(p: NodeId, h: NodeId, mapping: dict[NodeId, NodeId]) -> bool:
        if (p in p_boxed) != (h in h_boxed) or not _same_node(pattern, p, host, h):
            return False
        for port in _ports(pattern, p):
            hport = PortRef(h, port.direction, port.index)
            far = p_partner.get(port)
            hfar = h_partner.get(hport)
            if far is not None:
                if far.node_id in mapping and hfar != _moved(far, mapping):
                    return False
                if far.node_id not in mapping and (
                    hfar is None or hfar.node_id in mapping.values()
                ):
                    return False
                continue
            if port in boxed_slots and hport not in h_boundary:
                return False
            if hfar is not None and hfar.node_id in mapping.values():
                return False
            if hfar is None and hport not in h_boundary:
                return False
        return True

    def extend(i: int, mapping: dict[NodeId, NodeId]) -> Iterator[dict[NodeId, NodeId]]:
        if i == len(order):
            if all(
                h_partner.get(PortRef(h, port.direction, port.index)) is None
                or h_partner[PortRef(h, port.direction, port.index)].node_id not in mapping.values()
                or PortRef(p, port.direction, port.index) in p_partner
                for p, h in mapping.items()
                for port in _ports(pattern, p)
            ):
                yield dict(mapping)
            return
        p = order[i]
        anchored = [
            (port, p_partner[port])
            for port in _ports(pattern, p)
            if port in p_partner and p_partner[port].node_id in mapping
        ]
        if anchored:
            port, far = anchored[0]
            hfar = h_partner.get(_moved(far, mapping))
            candidates = (
                [hfar.node_id] if hfar is not None and hfar.direction is port.direction else []
            )
        else:
            candidates = sorted(host.nodes)
        used = set(mapping.values())
        for h in candidates:
            if h in used or not consistent(p, h, mapping):
                continue
            mapping[p] = h
            yield from extend(i + 1, mapping)
            del mapping[p]

    yield from extend(0, {})


def _peel_counts(pattern: Diagram, host: Diagram, index: str) -> list[int]:
    """The constants ``c >= 0`` with ``M' - m_b(k) = c`` for some owner ``b`` of ``index`` in
    ``pattern`` and host box multiplicity ``M'``, ascending."""
    counts: set[int] = set()
    for box in pattern.bang_boxes.values():
        if index not in box.multiplicity.free_symbols:
            continue
        mine = box.multiplicity.to_sympy()
        for other in host.bang_boxes.values():
            diff = sp.expand(other.multiplicity.to_sympy() - mine)
            if diff.is_Integer and int(diff) >= 0:
                counts.add(int(diff))
    return sorted(counts)


class HypothesisPattern(Pattern):
    """Occurrences of ``pattern`` (the hypothesis ``L(k)``) after ``c`` copies of every owner
    of ``index`` are peeled off; ``index`` stays bound."""

    locality_radius = 1_000_000

    def __init__(self, pattern: Diagram, index: str) -> None:
        self.pattern = pattern
        self.index = index

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Per peel count ascending, every embedding with its boxes matched."""
        found: list[Match] = []
        for c in _peel_counts(self.pattern, diagram, self.index):
            host = diagram
            try:
                for _ in range(c):
                    host = peel_owners(host, self.index).diagram
            except BangBoxError:
                continue
            for node_map in embeddings(self.pattern, host):
                box_map = _match_boxes(self.pattern, host, node_map)
                if box_map is not None:
                    found.append(HypothesisMatch(c, host, node_map, box_map))
        return tuple(found)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """Peel count, then the support."""
        hm = match if isinstance(match, HypothesisMatch) else None
        assert hm is not None
        return (hm.peels, *(int(n) for n in hm.support_node_ids))


def apply_hypothesis(match: HypothesisMatch, hypothesis: Diagram, replacement: Diagram) -> Diagram:
    """``match.host`` with the image of ``hypothesis`` replaced by ``replacement``, its scalar
    times ``replacement.scalar / hypothesis.scalar``."""
    host = match.host
    working = host.copy()
    partner = _partners(host)
    image = frozenset(match.node_map.values())
    h_in, h_out = list(host.boundary_inputs), list(host.boundary_outputs)
    ends: list[tuple[str, int] | PortRef] = []
    for ref in [*hypothesis.boundary_inputs, *hypothesis.boundary_outputs]:
        hport = _moved(ref, match.node_map)
        if hport in h_in:
            ends.append(("in", h_in.index(hport)))
        elif hport in h_out:
            ends.append(("out", h_out.index(hport)))
        else:
            ends.append(partner[hport])
    for box_id in sorted(match.box_map.values(), reverse=True):
        working.remove_bang_box(box_id)
    for node_id in sorted(image):
        working.remove_node(node_id)
    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(replacement.nodes):
        node = replacement.nodes[old_id]
        id_map[old_id] = working.add_node(
            node.generator_type,
            [port.dim for port in node.inputs],
            [port.dim for port in node.outputs],
            phase=node.phase,
        )
    for wire in sorted(replacement.wires, key=lambda w: w.sort_key()):
        working.add_wire(_moved(wire.a, id_map), _moved(wire.b, id_map))
    box_map: dict[BangBoxId, BangBoxId] = {}
    pending = sorted(replacement.bang_boxes.items())
    while len(box_map) < len(pending):
        for box_id, box in pending:
            needs = [box.parent, box.pair_of, box.feed, box.align]
            if box_id in box_map or any(r is not None and r not in box_map for r in needs):
                continue
            box_map[box_id] = add_box_image(
                working,
                box,
                id_map,
                None if box.parent is None else box_map[box.parent],
                None if box.pair_of is None else box_map[box.pair_of],
                feed=None if box.feed is None else box_map[box.feed],
                align=None if box.align is None else box_map[box.align],
            )
    for ref, end in zip(
        [*replacement.boundary_inputs, *replacement.boundary_outputs], ends, strict=True
    ):
        moved = _moved(ref, id_map)
        if isinstance(end, PortRef):
            working.add_wire(moved, end)
        elif end[0] == "in":
            h_in[end[1]] = moved
        else:
            h_out[end[1]] = moved
    working.set_boundary_inputs(h_in)
    working.set_boundary_outputs(h_out)
    working.multiply_scalar(replacement.scalar * hypothesis.scalar**-1)
    return working
