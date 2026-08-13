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

"""Tests for qufzx.algebra.dimension.unify_all: F1, Phase 5 post-closing audit round 21."""

from __future__ import annotations

import itertools
import random

import pytest

from qufzx.algebra.dimension import Dim, UnifyAllResult, unify_all
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef, Wire
from qufzx.rewrite.match import resolve_fusion_match

_D = Dim.symbol("d")
_E = Dim.symbol("e")
_PALETTE = (Dim.concrete(2), Dim.concrete(3), Dim.concrete(4), _D, _E, _D * _E, _D**2)


def _agree(a: UnifyAllResult, b: UnifyAllResult) -> bool:
    return a.status is b.status and dict(a.bindings) == dict(b.bindings)


class TestOrderIndependence:
    @pytest.mark.parametrize(
        "dims",
        [
            (_D, Dim.concrete(2), Dim.concrete(3)),  # jointly unsatisfiable
            (Dim.concrete(2), Dim.concrete(2), Dim.concrete(2)),  # success
            (_D, _E, _D * _E),  # deferred residuals
            (_D, _D, Dim.concrete(2)),  # success via one binding, reused
        ],
    )
    def test_every_permutation_agrees(self, dims: tuple[Dim, ...]) -> None:
        results = [unify_all(list(perm)) for perm in itertools.permutations(dims)]
        first = results[0]
        for result in results[1:]:
            assert _agree(result, first)
            assert {repr(p) for p in result.residual_pairs} == {
                repr(p) for p in first.residual_pairs
            }


class TestBasicOutcomes:
    def test_single_dim_succeeds_trivially(self) -> None:
        assert unify_all([Dim.concrete(2)]).is_success

    def test_empty_succeeds_trivially(self) -> None:
        assert unify_all([]).is_success

    def test_all_concrete_equal_succeeds(self) -> None:
        result = unify_all([Dim.concrete(2), Dim.concrete(2), Dim.concrete(2)])
        assert result.is_success
        assert result.bindings == {}

    def test_two_concrete_unequal_fails(self) -> None:
        assert unify_all([Dim.concrete(2), Dim.concrete(3)]).is_failure

    def test_jointly_unsatisfiable_bindings_fail(self) -> None:
        # d unifies with 2 and with 3 independently; jointly unsatisfiable.
        result = unify_all([_D, Dim.concrete(2), Dim.concrete(3)])
        assert result.is_failure

    def test_symbol_binds_and_is_reused(self) -> None:
        result = unify_all([_D, Dim.concrete(2), _D])
        assert result.is_success
        assert result.bindings == {"d": Dim.concrete(2)}

    def test_residual_deferred_pairs_reported(self) -> None:
        # d, e, d*e: every pair is a deferred occurs-check case.
        result = unify_all([_D, _E, _D * _E])
        assert result.is_deferred
        assert len(result.residual_pairs) >= 1


class TestAgreesWithResolveFusionMatch:
    """This module's fixpoint and ``resolve_fusion_match``'s must never silently drift on
    the one question both actually decide identically: whether a *lone connecting pair*,
    with no surviving legs on either side, unifies. With surviving legs present the two are
    not the same question -- ``resolve_fusion_match`` checks each surviving leg only
    against the running ``shared_dim`` (seeded at the A-side leg alone), never every leg
    pairwise against every other, so it is strictly less complete than this module's full
    pairwise closure over the same multiset; that gap is deferred to Phase 10 exactly like
    the cross-node gap below, not something this test asserts away.
    """

    @pytest.mark.parametrize("seed", range(40))
    def test_agrees_on_a_lone_connecting_pair(self, seed: int) -> None:
        rng = random.Random(seed)
        connecting_a = rng.choice(_PALETTE)
        connecting_b = rng.choice(_PALETTE)

        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[connecting_a], output_dims=[])
        b = diagram.add_node(Z_SPIDER, input_dims=[connecting_b], output_dims=[])
        wire = Wire(PortRef(a, Direction.INPUT, 0), PortRef(b, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a, b, wire)
        all_result = unify_all([connecting_a, connecting_b])

        assert resolution.passed is not all_result.is_failure


class TestCrossNodePropagationDeferredToPhase10:
    """A ``d``-vs-``2`` wire and a ``d``-vs-``3`` wire on two *different* nodes (no shared
    port, so no single ``unify_all`` call ever sees both) is not, and per FULL_PLAN.md
    Phase 10 item (i), is not yet meant to be, caught by validate(): diagram-global
    dimension-constraint propagation is explicitly Phase 10's job, not this module's.
    Pinned so the day Phase 10 lands, this test fails and says so.
    """

    def test_unreported_cross_node_contradiction(self) -> None:
        from qufzx.diagram.validate import validate

        d = Dim.symbol("d")
        diagram = Diagram()
        node_x = diagram.add_node(Z_SPIDER, input_dims=[d, Dim.concrete(2)], output_dims=[])
        node_y = diagram.add_node(Z_SPIDER, input_dims=[d, Dim.concrete(3)], output_dims=[])
        diagram.set_boundary_inputs(
            [PortRef(node_x, Direction.INPUT, i) for i in range(2)]
            + [PortRef(node_y, Direction.INPUT, i) for i in range(2)]
        )
        report = validate(diagram)
        # Each node individually binds d to a different concrete value -- jointly
        # unsatisfiable across the diagram -- but neither node's own ALL_LEGS_EQUAL check
        # can see the other node's binding, so both bind silently and validate reports
        # nothing.
        assert report.is_valid
