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

"""Matchers that rearrange a spider around a port-scope bang box.

* Port-box unfusion: a Z or X spider's port-boxed leg moved onto a new phaseless spider of
  its colour, joined to the rest by one wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import BangBoxId, Diagram, NodeId, PortRef
from archytaszx.rewrite.match import (
    _all_passed,
    _in_any_node_scope_box,
    _neighbour_ids,
    _seed_meets_anchors,
    _uniform_leg_dim,
)
from archytaszx.rewrite.match_zxw import _is
from archytaszx.rewrite.rule import (
    DimensionConstraint,
    Match,
    Pattern,
    SideCondition,
    SideConditionOutcome,
)

PORT_BOX_UNFUSION_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("node_is_a_spider", "a registered Z or X spider"),
    SideCondition(
        "one_leg_port_boxed",
        "a top-level port-scope bang box with no children holds exactly this one leg",
    ),
    SideCondition("two_other_legs", "the spider has at least two legs besides the boxed one"),
    SideCondition("same_dimension", "every leg of the spider carries one dimension"),
    SideCondition("outside_every_bang_box", "the spider lies in no node-scope bang box"),
)


@dataclass(frozen=True, slots=True)
class PortBoxUnfusionMatch:
    """The spider ``node_id`` whose leg ``boxed`` is the whole port scope of ``box_id``."""

    node_id: NodeId
    box_id: BangBoxId
    boxed: PortRef
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The spider's node id."""
        return (self.node_id,)


def find_port_box_unfusion_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[PortBoxUnfusionMatch, ...]:
    """Every spider leg that is alone the scope of a port-scope box, by (node, box) id."""
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    parents = {box.parent for box in diagram.bang_boxes.values() if box.parent is not None}
    matches: list[PortBoxUnfusionMatch] = []
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if box.is_node_scope or box.parent is not None or box_id in parents:
            continue
        if len(box.port_scope) != 1:
            continue
        (ref,) = box.port_scope
        node = diagram.nodes.get(ref.node_id)
        if node is None or not (_is(node, Z_SPIDER) or _is(node, X_SPIDER)):
            continue
        if not _seed_meets_anchors(anchors, neighbours, (node.id,), 1):
            continue
        if node.num_inputs + node.num_outputs < 3:
            continue
        if any(
            other_id != box_id and ref in other.port_scope
            for other_id, other in diagram.bang_boxes.items()
        ):
            continue
        dim = _uniform_leg_dim(node)
        if dim is None or _in_any_node_scope_box(diagram, (node.id,)):
            continue
        matches.append(
            PortBoxUnfusionMatch(
                node_id=node.id,
                box_id=box_id,
                boxed=ref,
                shared_dim=dim,
                side_condition_outcomes=_all_passed(PORT_BOX_UNFUSION_SIDE_CONDITIONS),
            )
        )
    matches.sort(key=lambda m: (int(m.node_id), int(m.box_id)))
    return tuple(matches)


class PortBoxUnfusionPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for port-box unfusion.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to :func:`find_port_box_unfusion_matches`."""
        return find_port_box_unfusion_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_port_box_unfusion_matches` with the candidates restricted."""
        return find_port_box_unfusion_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The spider's then the box's id."""
        unfusion = cast(PortBoxUnfusionMatch, match)
        return (int(unfusion.node_id), int(unfusion.box_id))
