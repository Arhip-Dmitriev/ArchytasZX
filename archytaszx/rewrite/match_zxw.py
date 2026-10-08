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

"""Matchers for the W node, the triangle and the dimension connectives.

* W fusion: a W output wired into another W's input; the two are one W.
* W identity: a one-output W is the bare wire.
* W zero copy: a phaseless X state into a W copies onto every W output.
* Triangle zero state: a phaseless X state through a T or Ti is unchanged.
* Triangle zero effect: a T into a phaseless X effect is a phaseless Z effect.
* Connective inverse: an S whose outputs feed a B's inputs in order, or a B whose output
  feeds an S, is the identity on its outer legs.
* Connective states: two phaseless same-colour states into a B, or an S into two such
  effects, are one state or effect on the joint leg.

Every pattern requires its legs to carry exactly equal dimensions. Only W fusion fires
inside a node-scope bang box, both nodes sharing their innermost one; every other pattern
requires its nodes to lie outside every node-scope box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    REGISTRY,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef, Wire
from archytaszx.rewrite.match import (
    IdentityMatch,
    _all_passed,
    _claimed_at_most_once,
    _claimed_exactly_once_by_a_wire,
    _exhausts_a_node_scope_box,
    _far_end,
    _in_any_node_scope_box,
    _is_phaseless,
    _neighbour_ids,
    _port_claims,
    _seed_meets_anchors,
    _support_ids,
    _uniform_leg_dim,
    _wire_by_port,
    innermost_node_scope_box,
)
from archytaszx.rewrite.rule import (
    DimensionConstraint,
    Match,
    Pattern,
    SideCondition,
    SideConditionOutcome,
)


def _is(node: Node, generator: GeneratorType) -> bool:
    """True iff ``node`` carries the registered ``generator``."""
    return (
        REGISTRY.is_registered(node.generator_type) and node.generator_type.name == generator.name
    )


def _shared_dim(*nodes: Node) -> Dim | None:
    """The one dimension every leg of ``nodes`` carries, or None."""
    dims = {port.dim for node in nodes for port in (*node.inputs, *node.outputs)}
    return dims.pop() if len(dims) == 1 else None


def _is_phaseless_end(node: Node, colour: GeneratorType, *, state: bool) -> bool:
    """True iff ``node`` is a phaseless ``colour`` spider with one output and no input (a
    state), or one input and no output (an effect)."""
    shape = (0, 1) if state else (1, 0)
    return (
        _is(node, colour) and (node.num_inputs, node.num_outputs) == shape and _is_phaseless(node)
    )


W_FUSION_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("two_w_nodes", "a W output wired into a second W's input"),
    SideCondition("joining_wire_unclaimed", "neither joined port carries a second claim"),
    SideCondition("same_dimension", "every leg of both nodes carries one dimension"),
    SideCondition(
        "bang_box_scope_agreement",
        "both nodes' innermost enclosing node-scope bang box, if any, are identical",
    ),
    SideCondition(
        "leaves_every_bang_box_populated",
        "no node-scope bang box holds the pair and nothing else",
    ),
)


@dataclass(frozen=True, slots=True)
class WFusionMatch:
    """One W output (``position`` of ``first_id``) wired into ``second_id``'s input."""

    first_id: NodeId
    second_id: NodeId
    position: int
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """Both node ids, ascending."""
        return _support_ids(self.first_id, self.second_id)


def find_w_fusion_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[WFusionMatch, ...]:
    """Every W output wired into another W's input, by (first, second) node id."""
    claims, boundary = _port_claims(diagram)
    matches: list[WFusionMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for out_ref, in_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if out_ref.direction is not Direction.OUTPUT or in_ref.direction is not Direction.INPUT:
                continue
            first = diagram.nodes.get(out_ref.node_id)
            second = diagram.nodes.get(in_ref.node_id)
            if first is None or second is None or first.id == second.id:
                continue
            if not _is(first, W_NODE) or not _is(second, W_NODE):
                continue
            if first.num_inputs != 1 or second.num_inputs != 1:
                continue
            if not _claimed_exactly_once_by_a_wire(out_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(in_ref, claims, boundary):
                continue
            dim = _shared_dim(first, second)
            if dim is None:
                continue
            boxes = {
                innermost_node_scope_box(diagram, first.id),
                innermost_node_scope_box(diagram, second.id),
            }
            if len(boxes) != 1 or _exhausts_a_node_scope_box(diagram, (first.id, second.id)):
                continue
            matches.append(
                WFusionMatch(
                    first_id=first.id,
                    second_id=second.id,
                    position=out_ref.index,
                    wire=wire,
                    shared_dim=dim,
                    side_condition_outcomes=_all_passed(W_FUSION_SIDE_CONDITIONS),
                )
            )
    matches.sort(key=lambda m: (int(m.first_id), int(m.second_id), m.position))
    return tuple(matches)


class WFusionPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for W fusion.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_w_fusion_matches`."""
        return find_w_fusion_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_w_fusion_matches` with the candidate wires restricted."""
        return find_w_fusion_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The first then the second node id, then the joined output."""
        fusion = cast(WFusionMatch, match)
        return (int(fusion.first_id), int(fusion.second_id), fusion.position)


W_IDENTITY_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("node_is_a_w_node", "a registered W node"),
    SideCondition("one_in_one_out", "exactly one input leg and one output leg"),
    SideCondition("not_a_self_loop", "no wire joins the node's two legs"),
    SideCondition(
        "legs_singly_claimed",
        "neither leg carries a second wire or a boundary slot alongside one, and at least "
        "one leg carries a wire to splice through",
    ),
    SideCondition(
        "splice_keeps_two_ports",
        "the spliced wire's far port is not the surviving leg's own neighbour",
    ),
    SideCondition("same_dimension", "both legs carry one dimension"),
    SideCondition(
        "leaves_every_bang_box_populated",
        "no node-scope bang box holds the node and nothing else",
    ),
)


def find_w_identity_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[IdentityMatch, ...]:
    """Every one-output W, by node id; the output's wire is spliced whenever it has one,
    otherwise the input's."""
    claims, boundary = _port_claims(diagram)
    by_port = _wire_by_port(diagram)
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    matches: list[IdentityMatch] = []
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        if not _is(node, W_NODE) or (node.num_inputs, node.num_outputs) != (1, 1):
            continue
        if not _seed_meets_anchors(anchors, neighbours, (node_id,), 1):
            continue
        shared_dim = _uniform_leg_dim(node)
        if shared_dim is None:
            continue
        port_in = PortRef(node_id, Direction.INPUT, 0)
        port_out = PortRef(node_id, Direction.OUTPUT, 0)
        if not _claimed_at_most_once(port_in, claims, boundary):
            continue
        if not _claimed_at_most_once(port_out, claims, boundary):
            continue
        wire_in = by_port.get(port_in)
        wire_out = by_port.get(port_out)
        if wire_out is not None and wire_out == wire_in:
            continue
        if wire_out is not None:
            wire, consumed_ref, surviving_ref = wire_out, port_out, port_in
        elif wire_in is not None:
            wire, consumed_ref, surviving_ref = wire_in, port_in, port_out
        else:
            continue
        far_ref = _far_end(wire, consumed_ref)
        if far_ref.node_id == node_id:
            continue
        if anchors is not None and anchors.isdisjoint((node_id, far_ref.node_id)):
            continue
        neighbour = by_port.get(surviving_ref)
        if neighbour is not None and _far_end(neighbour, surviving_ref) == far_ref:
            continue
        if _exhausts_a_node_scope_box(diagram, (node_id,)):
            continue
        matches.append(
            IdentityMatch(
                node_id=node_id,
                wire=wire,
                surviving_ref=surviving_ref,
                far_ref=far_ref,
                shared_dim=shared_dim,
                side_condition_outcomes=_all_passed(W_IDENTITY_SIDE_CONDITIONS),
            )
        )
    return tuple(matches)


class WIdentityPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for W identity removal.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_w_identity_matches`."""
        return find_w_identity_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_w_identity_matches` with the candidate nodes restricted."""
        return find_w_identity_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The removed node's id."""
        return (int(cast(IdentityMatch, match).node_id),)


W_COPY_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "state_is_a_phaseless_x_spider", "an X spider with no input, one output, and no phase"
    ),
    SideCondition("target_is_a_w_node", "a registered W node, fed at its input"),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition("outside_every_bang_box", "neither node lies in any node-scope bang box"),
)

TRIANGLE_STATE_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "state_is_a_phaseless_x_spider", "an X spider with no input, one output, and no phase"
    ),
    SideCondition("target_is_a_triangle", "a registered T or Ti, fed at its input"),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition("outside_every_bang_box", "neither node lies in any node-scope bang box"),
)


@dataclass(frozen=True, slots=True)
class FedStateMatch:
    """One phaseless X state wired into the input of ``target_id``, a W node or a triangle."""

    state_id: NodeId
    target_id: NodeId
    wire: Wire
    output_count: int
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The state's and the target's node ids, ascending."""
        return _support_ids(self.state_id, self.target_id)


def find_fed_state_matches(
    diagram: Diagram,
    *,
    triangle: bool,
    anchors: frozenset[NodeId] | None = None,
) -> tuple[FedStateMatch, ...]:
    """Every phaseless X state wired into a W's input, or a T's or Ti's when ``triangle``, by
    state id."""
    targets = (TRIANGLE, TRIANGLE_INVERSE) if triangle else (W_NODE,)
    conditions = TRIANGLE_STATE_SIDE_CONDITIONS if triangle else W_COPY_SIDE_CONDITIONS
    claims, boundary = _port_claims(diagram)
    matches: list[FedStateMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for state_ref, target_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if state_ref.direction is not Direction.OUTPUT:
                continue
            if target_ref.direction is not Direction.INPUT or target_ref.index != 0:
                continue
            state = diagram.nodes.get(state_ref.node_id)
            target = diagram.nodes.get(target_ref.node_id)
            if state is None or target is None or state.id == target.id:
                continue
            if not _is_phaseless_end(state, X_SPIDER, state=True):
                continue
            if not any(_is(target, generator) for generator in targets):
                continue
            if target.num_inputs != 1 or (triangle and target.num_outputs != 1):
                continue
            if not _claimed_exactly_once_by_a_wire(state_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(target_ref, claims, boundary):
                continue
            dim = _shared_dim(state, target)
            if dim is None or _in_any_node_scope_box(diagram, (state.id, target.id)):
                continue
            matches.append(
                FedStateMatch(
                    state_id=state.id,
                    target_id=target.id,
                    wire=wire,
                    output_count=target.num_outputs,
                    shared_dim=dim,
                    side_condition_outcomes=_all_passed(conditions),
                )
            )
    matches.sort(key=lambda m: (int(m.state_id), int(m.target_id)))
    return tuple(matches)


@dataclass(frozen=True, slots=True)
class FedStatePattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for an X state fed into a
    W node, or into a triangle when ``triangle``.

    Locality radius 1.
    """

    triangle: bool = False

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_fed_state_matches`."""
        return find_fed_state_matches(diagram, triangle=self.triangle)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_fed_state_matches` with the candidate wires restricted."""
        return find_fed_state_matches(diagram, triangle=self.triangle, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The state's then the target's node id."""
        fed = cast(FedStateMatch, match)
        return (int(fed.state_id), int(fed.target_id))


TRIANGLE_EFFECT_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("node_is_a_triangle", "a registered T with one input and one output"),
    SideCondition(
        "effect_is_a_phaseless_x_spider", "an X spider with one input, no output, and no phase"
    ),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition("outside_every_bang_box", "neither node lies in any node-scope bang box"),
)


@dataclass(frozen=True, slots=True)
class TriangleEffectMatch:
    """One T whose output feeds a phaseless X effect."""

    triangle_id: NodeId
    effect_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The triangle's and the effect's node ids, ascending."""
        return _support_ids(self.triangle_id, self.effect_id)


def find_triangle_effect_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[TriangleEffectMatch, ...]:
    """Every T whose output feeds a phaseless X effect, by triangle id."""
    claims, boundary = _port_claims(diagram)
    matches: list[TriangleEffectMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for out_ref, in_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if out_ref.direction is not Direction.OUTPUT or in_ref.direction is not Direction.INPUT:
                continue
            triangle = diagram.nodes.get(out_ref.node_id)
            effect = diagram.nodes.get(in_ref.node_id)
            if triangle is None or effect is None or triangle.id == effect.id:
                continue
            if not _is(triangle, TRIANGLE) or (triangle.num_inputs, triangle.num_outputs) != (1, 1):
                continue
            if not _is_phaseless_end(effect, X_SPIDER, state=False):
                continue
            if not _claimed_exactly_once_by_a_wire(out_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(in_ref, claims, boundary):
                continue
            dim = _shared_dim(triangle, effect)
            if dim is None or _in_any_node_scope_box(diagram, (triangle.id, effect.id)):
                continue
            matches.append(
                TriangleEffectMatch(
                    triangle_id=triangle.id,
                    effect_id=effect.id,
                    wire=wire,
                    shared_dim=dim,
                    side_condition_outcomes=_all_passed(TRIANGLE_EFFECT_SIDE_CONDITIONS),
                )
            )
    matches.sort(key=lambda m: (int(m.triangle_id), int(m.effect_id)))
    return tuple(matches)


class TriangleEffectPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for a T into an X effect.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_triangle_effect_matches`."""
        return find_triangle_effect_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_triangle_effect_matches` with the candidate wires
        restricted."""
        return find_triangle_effect_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The triangle's then the effect's node id."""
        effect = cast(TriangleEffectMatch, match)
        return (int(effect.triangle_id), int(effect.effect_id))


CONNECTIVE_INVERSE_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "inverse_pair",
        "an S whose two outputs feed a B's two inputs in order, or a B whose output feeds an S",
    ),
    SideCondition("joining_wires_unclaimed", "no joined port carries a second claim"),
    SideCondition(
        "matching_dimensions", "each joined pair of legs, and each outer pair, carry one dimension"
    ),
    SideCondition(
        "outer_legs_singly_claimed",
        "no outer leg carries a second claim, and each identity to splice has a wire",
    ),
    SideCondition(
        "splice_keeps_two_ports",
        "no spliced wire's far port is on the pair or is the surviving leg's own neighbour",
    ),
    SideCondition("outside_every_bang_box", "neither node lies in any node-scope bang box"),
)


@dataclass(frozen=True, slots=True)
class Splice:
    """One identity wire left by a rewrite: the wire consumed, the surviving outer port, and
    the far port it is joined to."""

    wire: Wire
    surviving_ref: PortRef
    far_ref: PortRef


@dataclass(frozen=True, slots=True)
class ConnectiveInverseMatch:
    """An S/B pair composing to the identity (``bind_first``: B then S), its joining wires,
    and one :class:`Splice` per resulting identity wire."""

    first_id: NodeId
    second_id: NodeId
    joining: tuple[Wire, ...]
    splices: tuple[Splice, ...]
    bind_first: bool
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The pair's node ids and every spliced far node id, ascending."""
        return _support_ids(
            self.first_id, self.second_id, *(s.far_ref.node_id for s in self.splices)
        )


def _splice(
    pair: tuple[PortRef, PortRef],
    pair_ids: tuple[NodeId, NodeId],
    by_port: dict[PortRef, Wire],
    claims: dict[PortRef, int],
    boundary: frozenset[PortRef],
) -> Splice | None:
    """The splice joining the two outer ports in ``pair``: the second's wire if it has one,
    else the first's; None when neither is wired or the splice would collapse."""
    if not all(_claimed_at_most_once(ref, claims, boundary) for ref in pair):
        return None
    first, second = pair
    if by_port.get(second) is not None:
        consumed, surviving = second, first
    elif by_port.get(first) is not None:
        consumed, surviving = first, second
    else:
        return None
    wire = by_port[consumed]
    far = _far_end(wire, consumed)
    if far.node_id in pair_ids:
        return None
    neighbour = by_port.get(surviving)
    if neighbour is not None and _far_end(neighbour, surviving) == far:
        return None
    return Splice(wire, surviving, far)


def _connective_candidate(
    diagram: Diagram,
    first: Node,
    second: Node,
    by_port: dict[PortRef, Wire],
    claims: dict[PortRef, int],
    boundary: frozenset[PortRef],
) -> ConnectiveInverseMatch | None:
    """The inverse-pair match of ``first`` feeding ``second``, or None."""
    bind_first = _is(first, DIM_BINDER)
    if bind_first and not _is(second, DIM_SPLITTER):
        return None
    if not bind_first and not (_is(first, DIM_SPLITTER) and _is(second, DIM_BINDER)):
        return None
    joins = 1 if bind_first else 2
    joining: list[Wire] = []
    for index in range(joins):
        out_ref = PortRef(first.id, Direction.OUTPUT, index)
        in_ref = PortRef(second.id, Direction.INPUT, index)
        wire = by_port.get(out_ref)
        if wire is None or _far_end(wire, out_ref) != in_ref:
            return None
        if not _claimed_exactly_once_by_a_wire(out_ref, claims, boundary):
            return None
        if not _claimed_exactly_once_by_a_wire(in_ref, claims, boundary):
            return None
        if first.outputs[index].dim != second.inputs[index].dim:
            return None
        joining.append(wire)
    if bind_first:
        pairs = [
            (PortRef(first.id, Direction.INPUT, k), PortRef(second.id, Direction.OUTPUT, k))
            for k in (0, 1)
        ]
        dims_agree = all(first.inputs[k].dim == second.outputs[k].dim for k in (0, 1))
    else:
        pairs = [(PortRef(first.id, Direction.INPUT, 0), PortRef(second.id, Direction.OUTPUT, 0))]
        dims_agree = first.inputs[0].dim == second.outputs[0].dim
    if not dims_agree:
        return None
    splices = []
    for pair in pairs:
        splice = _splice(pair, (first.id, second.id), by_port, claims, boundary)
        if splice is None:
            return None
        splices.append(splice)
    far = [s.far_ref for s in splices]
    if len(set(far)) != len(far):
        return None
    if _in_any_node_scope_box(diagram, (first.id, second.id)):
        return None
    return ConnectiveInverseMatch(
        first_id=first.id,
        second_id=second.id,
        joining=tuple(joining),
        splices=tuple(splices),
        bind_first=bind_first,
        side_condition_outcomes=_all_passed(CONNECTIVE_INVERSE_SIDE_CONDITIONS),
    )


def find_connective_inverse_matches(
    diagram: Diagram, *, bind_first: bool, anchors: frozenset[NodeId] | None = None
) -> tuple[ConnectiveInverseMatch, ...]:
    """Every S feeding a B in order (a B feeding an S when ``bind_first``), by first id."""
    claims, boundary = _port_claims(diagram)
    by_port = _wire_by_port(diagram)
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    head = DIM_BINDER if bind_first else DIM_SPLITTER
    matches: list[ConnectiveInverseMatch] = []
    for node_id in sorted(diagram.nodes):
        first = diagram.nodes[node_id]
        if not _is(first, head) or not _seed_meets_anchors(anchors, neighbours, (node_id,), 1):
            continue
        wire = by_port.get(PortRef(node_id, Direction.OUTPUT, 0))
        if wire is None:
            continue
        other = _far_end(wire, PortRef(node_id, Direction.OUTPUT, 0))
        second = diagram.nodes.get(other.node_id)
        if second is None or second.id == node_id:
            continue
        match = _connective_candidate(diagram, first, second, by_port, claims, boundary)
        if match is not None:
            matches.append(match)
    return tuple(matches)


@dataclass(frozen=True, slots=True)
class ConnectiveInversePattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for an S then B pair, or a
    B then S pair when ``bind_first``.

    Locality radius 2.
    """

    bind_first: bool = False

    locality_radius = 2

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_connective_inverse_matches`."""
        return find_connective_inverse_matches(diagram, bind_first=self.bind_first)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_connective_inverse_matches` with the candidates restricted."""
        return find_connective_inverse_matches(diagram, bind_first=self.bind_first, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The first then the second node id."""
        pair = cast(ConnectiveInverseMatch, match)
        return (int(pair.first_id), int(pair.second_id))


CONNECTIVE_STATES_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "connective_with_two_ends",
        "a B whose two inputs are fed by phaseless Z states, or an S whose two outputs feed "
        "phaseless Z effects",
    ),
    SideCondition("joined_and_unclaimed", "each joining wire's ports carry no second claim"),
    SideCondition(
        "matching_dimensions", "each state or effect carries its connective leg's dimension"
    ),
    SideCondition("outside_every_bang_box", "no node lies in any node-scope bang box"),
)

CONNECTIVE_STATES_SWAPPED_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "connective_with_two_ends",
        "a B whose two inputs are fed by phaseless X states, or an S whose two outputs feed "
        "phaseless X effects",
    ),
    *CONNECTIVE_STATES_SIDE_CONDITIONS[1:],
)


@dataclass(frozen=True, slots=True)
class ConnectiveStatesMatch:
    """A connective whose split legs all end in phaseless states or effects of one colour (X
    when ``swapped``)."""

    connective_id: NodeId
    end_ids: tuple[NodeId, NodeId]
    wires: tuple[Wire, Wire]
    joint_dim: Dim
    effects: bool
    swapped: bool
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The connective's and both ends' node ids, ascending."""
        return _support_ids(self.connective_id, *self.end_ids)


def find_connective_states_matches(
    diagram: Diagram, *, swapped: bool = False, anchors: frozenset[NodeId] | None = None
) -> tuple[ConnectiveStatesMatch, ...]:
    """Every B fed by two phaseless Z states, or S feeding two phaseless Z effects (X when
    ``swapped``), by connective id."""
    colour = X_SPIDER if swapped else Z_SPIDER
    conditions = (
        CONNECTIVE_STATES_SWAPPED_SIDE_CONDITIONS if swapped else CONNECTIVE_STATES_SIDE_CONDITIONS
    )
    claims, boundary = _port_claims(diagram)
    by_port = _wire_by_port(diagram)
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    matches: list[ConnectiveStatesMatch] = []
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        effects = _is(node, DIM_SPLITTER)
        if not effects and not _is(node, DIM_BINDER):
            continue
        if not _seed_meets_anchors(anchors, neighbours, (node_id,), 1):
            continue
        direction = Direction.OUTPUT if effects else Direction.INPUT
        legs = node.legs(direction)
        if len(legs) != 2:
            continue
        ends: list[NodeId] = []
        wires: list[Wire] = []
        for index, port in enumerate(legs):
            ref = PortRef(node_id, direction, index)
            wire = by_port.get(ref)
            if wire is None or not _claimed_exactly_once_by_a_wire(ref, claims, boundary):
                break
            far = _far_end(wire, ref)
            end = diagram.nodes.get(far.node_id)
            if end is None or end.id == node_id or end.id in ends:
                break
            if not _is_phaseless_end(end, colour, state=not effects):
                break
            if not _claimed_exactly_once_by_a_wire(far, claims, boundary):
                break
            if _shared_dim(end) != port.dim:
                break
            ends.append(end.id)
            wires.append(wire)
        if len(ends) != 2:
            continue
        if _in_any_node_scope_box(diagram, (node_id, *ends)):
            continue
        joint = node.legs(Direction.INPUT if effects else Direction.OUTPUT)
        if len(joint) != 1:
            continue
        matches.append(
            ConnectiveStatesMatch(
                connective_id=node_id,
                end_ids=(ends[0], ends[1]),
                wires=(wires[0], wires[1]),
                joint_dim=joint[0].dim,
                effects=effects,
                swapped=swapped,
                side_condition_outcomes=_all_passed(conditions),
            )
        )
    return tuple(matches)


@dataclass(frozen=True, slots=True)
class ConnectiveStatesPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for a connective between
    two Z states or effects, X when ``swapped``.

    Locality radius 1.
    """

    swapped: bool = False

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_connective_states_matches`."""
        return find_connective_states_matches(diagram, swapped=self.swapped)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_connective_states_matches` with the candidates restricted."""
        return find_connective_states_matches(diagram, swapped=self.swapped, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The connective's node id."""
        return (int(cast(ConnectiveStatesMatch, match).connective_id),)
