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

"""Establishes qufzx.rewrite.match: fusion-pattern matching, its side conditions, and ordering."""

from __future__ import annotations

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import (
    X_SPIDER,
    Z_SPIDER,
    DimensionPolicy,
    GeneratorType,
    LegPolicy,
    PhaseSchema,
)
from qufzx.diagram.graph import Diagram, Direction, PortRef, Wire
from qufzx.rewrite.match import (
    FUSION_SIDE_CONDITIONS,
    FusionPattern,
    find_matches,
    resolve_fusion_match,
)
from qufzx.rewrite.rule import (
    ConstraintOutcome,
    ConstraintSource,
    DimensionConstraint,
    RewriteGrammarError,
)
from qufzx.rewrite.rules_library import spider_fusion_builder

from .helpers import build_ghz_with_copy


class TestFindMatchesOnGhzWithCopy:
    def test_finds_exactly_one_match(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        matches = find_matches(diagram)
        assert len(matches) == 1
        match = matches[0]
        assert match.a_id == min(a_id, b_id)
        assert match.b_id == max(a_id, b_id)
        assert match.shared_dim == d
        assert match.all_side_conditions_passed

    def test_all_declared_side_conditions_are_reported(self) -> None:
        d = Dim.concrete(3)
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        reported_names = {outcome.name for outcome in match.side_condition_outcomes}
        declared_names = {condition.name for condition in FUSION_SIDE_CONDITIONS}
        assert reported_names == declared_names
        assert all(outcome.passed for outcome in match.side_condition_outcomes)

    def test_no_dimension_constraints_when_dims_are_syntactically_equal(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        assert match.dimension_constraints == ()


class TestFusionPatternDelegatesToFindMatches:
    def test_returns_the_same_matches(self) -> None:
        d = Dim.concrete(2)
        diagram, _a, _b = build_ghz_with_copy(d)
        pattern_matches = FusionPattern().find_matches(diagram)
        function_matches = find_matches(diagram)
        assert pattern_matches == function_matches


class TestNoMatchBetweenDifferentColors:
    def test_z_and_x_never_match(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()


class TestParallelWiresYieldOneCandidatePerWire:
    """A node pair joined by k wires now yields up to k candidates -- one per wire, each
    fusing across that wire and leaving the other k-1 as self-loops on the merged node.
    See match.py's module docstring, condition 3 (``parallel_wires_become_self_loops``).
    """

    def test_two_wires_between_same_pair_now_matches_twice(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
        matches = find_matches(diagram)
        assert len(matches) == 2
        consumed_wires = {m.wire for m in matches}
        assert len(consumed_wires) == 2
        assert all(m.all_side_conditions_passed for m in matches)

    def test_x_pair_with_one_alternating_and_one_same_direction_wire_yields_one_match(
        self,
    ) -> None:
        """The colour/direction condition applies to the CONSUMED wire only: an X pair
        joined by one OUTPUT->INPUT wire and one OUTPUT->OUTPUT wire yields exactly one
        match (the OUTPUT->INPUT one), never two.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(X_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.OUTPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        ref_a = matches[0].wire.a if matches[0].wire.a.node_id == a_id else matches[0].wire.b
        ref_b = matches[0].wire.b if matches[0].wire.a.node_id == a_id else matches[0].wire.a
        assert ref_a == PortRef(a_id, Direction.OUTPUT, 0)
        assert ref_b == PortRef(b_id, Direction.INPUT, 0)

    def test_leftover_wire_leg_binding_resolves_shared_dim(self) -> None:
        """Fix 1(d): ``_unify_surviving_legs`` needs no change for parallel wires -- the
        leftover wire's endpoints are ordinary surviving legs, already unified against
        ``shared_dim``. Here the leftover leg on A is a concrete dim that only unifies
        with the (symbolic) connecting-pair's shared_dim by binding.
        """
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, three])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
        matches = find_matches(diagram)
        consuming_wire0 = next(
            m
            for m in matches
            if m.wire
            == Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        )
        assert consuming_wire0.shared_dim == three

    def test_stable_order_regardless_of_wire_insertion_order(self) -> None:
        """Fix 1(c): ``(a_id, b_id)`` is no longer a unique key for a candidate, so
        ``find_matches``'s sort must tiebreak deterministically on the consumed wire, not
        on whatever order the parallel wires happened to be added to the diagram.
        """
        d = Dim.concrete(2)

        def _build(reverse_wire_order: bool) -> Diagram:
            diagram = Diagram()
            a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
            b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
            pairs = [
                (PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0)),
                (PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1)),
            ]
            if reverse_wire_order:
                pairs = list(reversed(pairs))
            for ref_a, ref_b in pairs:
                diagram.add_wire(ref_a, ref_b)
            return diagram

        forward = find_matches(_build(False))
        backward = find_matches(_build(True))
        forward_order = [(m.a_id, m.b_id, m.wire) for m in forward]
        backward_order = [(m.a_id, m.b_id, m.wire) for m in backward]
        assert forward_order == backward_order


class TestNoMatchForSelfLoop:
    def test_self_loop_on_single_spider_is_not_fusion(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(
            PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.INPUT, 0)
        )
        assert find_matches(diagram) == ()


class TestDimensionMismatchIsNonMatch:
    def test_different_concrete_dims_refuse_to_match(self) -> None:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()


class TestWireDirectionMustBeOutputToInput:
    """See match.py's module docstring, condition 4, for the Step 4 color-conditioned rule."""

    def test_x_output_to_output_wire_refuses_to_match(self) -> None:
        """X is still direction-strict: a same-direction wire pairs F with F, not F^dagger."""
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(X_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(X_SPIDER, input_dims=[], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.OUTPUT, 0))
        assert find_matches(diagram) == ()

    def test_x_input_to_input_wire_refuses_to_match(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        b_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.INPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()

    def test_z_output_to_output_wire_matches(self) -> None:
        """Z's tensor is diagonal in every axis, so a same-direction wire is valid fusion too."""
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.OUTPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].all_side_conditions_passed

    def test_z_input_to_input_wire_matches(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.INPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].all_side_conditions_passed


class TestMatchSurvivesThirdNodeWiring:
    def test_fused_pair_also_wired_to_a_third_node_still_matches(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        pairs = [{m.a_id, m.b_id} for m in matches]
        assert {a_id, b_id} in pairs


def _phase_at(dim: Dim, index: int) -> PhaseVector:
    return PhaseVector(dim, {index: Phase.turns(1)})


class TestPhaseDimensionMismatchIsNonMatch:
    def test_both_phases_present_with_concrete_phase_dim_binding_the_symbolic_shared_dim(
        self,
    ) -> None:
        # Judgement call 2 (Phase 5 post-closing audit): both legs share the still-symbolic
        # d (shared_dim=d, unbound), A's phase already agrees outright (also over d), and
        # B's phase over the concrete 3 now legitimately binds d := 3 through condition 6's
        # own unify call -- exactly the completeness gap the pre-fix plain-equality version
        # missed (see the module docstring, condition 6). Neither phase's entries reference
        # a dimension symbol, so there is nothing to substitute and reattach_phase succeeds
        # trivially either way.
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d], phase=_phase_at(d, 1)
        )
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[d], output_dims=[], phase=_phase_at(Dim.concrete(3), 1)
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert DimensionConstraint(
            assumed=Dim.concrete(3),
            equal_to=d,
            source=ConstraintSource.node_phase(b_id),
            outcome=ConstraintOutcome.BOUND,
        ) in matches[0].dimension_constraints

    def test_one_sided_phase_binding_the_symbolic_shared_leg_dim_now_matches(self) -> None:
        # Mirror of the above with only one phase present: B has none, so only A's phase
        # (over the concrete 3) needs to unify with shared_dim=d -- binds d := 3, accepted.
        shared = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[shared],
            phase=_phase_at(Dim.concrete(3), 1),
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[shared], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert DimensionConstraint(
            assumed=Dim.concrete(3),
            equal_to=shared,
            source=ConstraintSource.node_phase(a_id),
            outcome=ConstraintOutcome.BOUND,
        ) in matches[0].dimension_constraints

    def test_a_phase_dim_that_fails_to_unify_with_shared_dim_is_still_a_non_match(self) -> None:
        # Two distinct concrete dims can never unify -- this remains a hard non-match,
        # judgement call 2 only widened acceptance for a phase dim that unifies (bare
        # identity or binding), never for one that outright fails to.
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[Dim.concrete(2)],
            phase=_phase_at(Dim.concrete(3), 1),
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()

    def test_matching_phase_dims_over_symbolic_shared_leg_still_match(self) -> None:
        shared = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[shared], phase=_phase_at(shared, 1)
        )
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[shared], output_dims=[], phase=_phase_at(shared, 1)
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert len(find_matches(diagram)) == 1


class TestDimensionConstraintsRecording:
    def test_deferred_leg_dims_are_recorded(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d * e], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        match = matches[0]
        assert match.dimension_constraints == (
            DimensionConstraint(
                assumed=d,
                equal_to=d * e,
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.DEFERRED,
            ),
        )
        agreement = next(
            o for o in match.side_condition_outcomes if o.name == "dimension_agreement"
        )
        assert agreement.deferred

    def test_unify_success_with_binding_records_constraint_not_deferred(self) -> None:
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[three], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        match = matches[0]
        assert match.dimension_constraints == (
            DimensionConstraint(
                assumed=d,
                equal_to=three,
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.BOUND,
            ),
        )
        agreement = next(
            o for o in match.side_condition_outcomes if o.name == "dimension_agreement"
        )
        assert not agreement.deferred

    def test_a_leg_already_bound_by_the_connecting_pair_is_not_recorded_again(self) -> None:
        """Phase 5 post-closing audit, dimension_constraints duplicate-assumption defect.

        A: output dim d (consumed). B: input dim 2 (consumed), output dim 2 (survives). The
        connecting pair binds d := 2 and is recorded once; before the fix, B's surviving leg
        (raw dim 2, unresolved through that binding) was unified against shared_dim again on
        every fixpoint pass that left shared_dim unchanged, re-appending an identical
        ``(d, 2)`` pair each time. Fixed: exactly one entry.
        """
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[two])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].dimension_constraints == (
            DimensionConstraint(
                assumed=d,
                equal_to=two,
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.BOUND,
            ),
        )

    def test_a_surviving_leg_bound_by_an_earlier_surviving_leg_is_not_recorded_again(
        self,
    ) -> None:
        """A: input dims [d, 2], output dim d (consumed). B: input dim d (consumed), output
        dim d (survives). The connecting pair (d, d) is a bare identity, recording nothing.
        A's own surviving leg (2) binds d := 2 and is recorded once, as ``(2, d)``; before
        the fix, every leg checked afterward -- A's own d-leg and B's surviving d-leg, both
        still mentioning the now-bound symbol d, and again on every further fixpoint pass --
        was re-unified against its *raw*, unresolved dim and re-recorded as a duplicate
        ``(d, 2)``. Fixed: exactly one entry total.
        """
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d, two], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].dimension_constraints == (
            DimensionConstraint(
                assumed=two,
                equal_to=d,
                source=ConstraintSource.surviving_leg(PortRef(a_id, Direction.INPUT, 1)),
                outcome=ConstraintOutcome.BOUND,
            ),
        )


class TestOutOfRangeWireEndpointRaises:
    """graph.py is deliberately permissive about wire port indices; find_matches must not be.

    Fix 2: an out-of-range index in a wire endpoint used to escape as a bare IndexError
    from ``node.legs(direction)[index]``; it must now raise RewriteGrammarError instead,
    for either endpoint, before any dimension work is attempted.
    """

    def test_out_of_range_index_on_the_a_side_raises(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 5), PortRef(b_id, Direction.INPUT, 0))
        with pytest.raises(RewriteGrammarError):
            find_matches(diagram)

    def test_out_of_range_index_on_the_b_side_raises(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 3))
        with pytest.raises(RewriteGrammarError):
            find_matches(diagram)


class TestSharedDimResolvesThroughBinding:
    """Fix 3: shared_dim must be resolved by substituting a unify binding, not taken raw from A.

    Before the fix, ``shared_dim`` was unconditionally ``port_a.dim`` even when ``unify``
    only succeeded by binding a symbol -- e.g. leg dims ``d`` and ``3`` bind ``d := 3``, but
    ``shared_dim`` stayed the still-unbound ``d``.
    """

    def test_shared_dim_is_the_bound_concrete_value_not_the_symbol(self) -> None:
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[three], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].shared_dim == three

    def test_a_legitimate_phase_stated_over_the_bound_dim_now_matches(self) -> None:
        # This is the proof scenario from the Fix 3(a) audit finding: B's phase is
        # legitimately stated over Dim(3), the concrete value d unifies to, not over the
        # still-symbolic leg dim d itself. Before the fix this was wrongly dropped as a
        # non-match because phase_dimension_agreement compared it against raw d.
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[three],
            output_dims=[],
            phase=PhaseVector(three, {1: Phase.turns(1)}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].shared_dim == three

    def test_phases_agreeing_only_after_substitution_but_not_raw_now_matches(self) -> None:
        # Judgement call 2 (Phase 5 post-closing audit), decided: node A's phase is stated
        # over the symbolic leg dim d, node B's over the bound concrete value 3 -- both
        # individually resolve to shared_dim=3 (A's via a binding condition 6 itself now
        # derives, B's outright), but their raw, unsubstituted Dims differ. The pre-fix
        # matcher additionally required two present phases' raw Dims to equal each other,
        # rejecting this candidate on the theory that spider_fusion_builder's
        # PhaseVector.__add__ would otherwise crash on unequal Dims -- but reattach_phase
        # forces *both* operands' container Dim to the identical shared_dim regardless of
        # their raw values (see that function's docstring), so the crash this guarded
        # against was never actually reachable once reattach_phase existed. Removed as
        # unneeded conservatism; the fused diagram is built and oracle-compared for real in
        # TestPhase5Round12AuditDefects-style rules_library tests.
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {1: Phase.turns(1)})
        )
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[three],
            output_dims=[],
            phase=PhaseVector(three, {1: Phase.turns(1)}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].shared_dim == three


class TestPhaseDimensionAgreementSeesSurvivingLegBindings:
    """Defect 2 (Phase 5 round-7 audit): phase_dimension_agreement used to resolve a present
    phase's Dim using only the connecting pair's own unify bindings (``leg_unify.bindings``),
    never a binding a *surviving* leg produced -- even though condition 5's own ``shared_dim``
    is refined by exactly that binding. A phase legally stated over a symbol only a surviving
    leg happens to bind was therefore compared unsubstituted against the refined shared_dim
    and wrongly dropped as a non-match.
    """

    def test_a_phase_over_a_symbol_bound_only_by_a_surviving_leg_now_matches(self) -> None:
        # A's connecting leg (output 0) and B's connecting leg (input 0) are both the
        # symbol d -- unifying them binds nothing. A's *surviving* input leg is a
        # concrete Dim(2), which binds d := 2 once condition 5 unifies it against
        # shared_dim. A's phase, stated over the still-symbolic d, is legal only because
        # it resolves through that surviving-leg binding, not the connecting pair's own.
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[two],
            output_dims=[d],
            phase=PhaseVector(d, {1: Phase.turns(1)}),
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])

        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].shared_dim == two


class TestDeterministicOrder:
    def test_matches_sorted_by_node_ids(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        # Build two independent fusable pairs; node ids interleave with an unrelated node
        # to make accidental insertion-order-only correctness unlikely to pass by luck.
        b1_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        a1_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b2_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        a2_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.add_wire(PortRef(a1_id, Direction.OUTPUT, 0), PortRef(b1_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a2_id, Direction.OUTPUT, 0), PortRef(b2_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 2
        keys = [(int(m.a_id), int(m.b_id)) for m in matches]
        assert keys == sorted(keys)


class TestSurvivingLegDimensionUnification:
    """Defect 1 (Phase 5 audit): surviving-leg dims used to be forced onto ``shared_dim`` by
    the builder with no unification check at all, so a surviving leg that plainly conflicted
    with ``shared_dim`` was silently overwritten (destroying a real dimension, or laundering
    a pre-existing hard error) instead of making the candidate a non-match.
    """

    def test_a_surviving_leg_that_conflicts_with_shared_dim_is_a_non_match(self) -> None:
        # A already carries a hard dimension_policy_violation (outputs 3 and 5 disagree on
        # a Z spider, which is ALL_LEGS_EQUAL). Before the fix, fusing across out0 (dim 3)
        # forced the surviving out1 (dim 5) onto shared_dim=3, silently erasing that
        # pre-existing error with validate() reporting nothing wrong afterward.
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(3), Dim.concrete(5)]
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])
        assert find_matches(diagram) == ()

    def test_a_surviving_concrete_leg_downgrades_shared_dim_and_is_recorded(self) -> None:
        # A's connecting leg is symbolic d, but A's surviving leg is the concrete Dim(2).
        # Before the fix, shared_dim stayed the unbound d and the builder just overwrote
        # the concrete surviving leg with it, discarding the concrete value with nothing
        # recorded. The fix must unify the surviving leg against shared_dim too, refining
        # it to the concrete value and recording the assumed equality.
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, two])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        match = matches[0]
        assert match.shared_dim == two
        assert DimensionConstraint(
            assumed=two,
            equal_to=d,
            source=ConstraintSource.surviving_leg(PortRef(a_id, Direction.OUTPUT, 1)),
            outcome=ConstraintOutcome.BOUND,
        ) in match.dimension_constraints


class TestConsumedPortClaimedElsewhereIsNonMatch:
    """Defect 2 (Phase 5 audit): a consumed port also named by a second wire, or also on a
    boundary list, used to still be returned as a match -- apply() then raised
    RewriteDomainError (unmapped consumed port), breaking the documented "every match this
    function returns can be applied" invariant. find_matches must reject such candidates
    itself, since a port claimed twice like this is not a genuine fusion occurrence.
    """

    def test_consumed_port_also_wired_to_a_third_node_is_a_non_match(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        # A malformed second wire claiming the same consumed port (A.out0) a third time.
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()

    def test_consumed_port_also_on_the_boundary_is_a_non_match(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 0)])
        assert find_matches(diagram) == ()


class TestMalformedWireDetectionIsFullyUnconditional:
    """Defect 3 (Phase 5 audit): the out-of-range-index check used to run only per-candidate,
    after grouping and filtering, so a malformed wire dropped by the parallel-wire-pair
    filter or the self-loop skip escaped detection entirely and returned () instead of
    raising. The check must now run over every wire before any grouping or filtering.
    """

    def test_malformed_wire_inside_a_parallel_pair_still_raises(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        # A second, parallel wire (a pair a fusion candidate now matches across, once per
        # wire -- see TestParallelWiresYieldOneCandidatePerWire) whose B-side index is out
        # of range; malformed-wire detection must still fire unconditionally regardless.
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 7))
        with pytest.raises(RewriteGrammarError):
            find_matches(diagram)

    def test_malformed_self_loop_wire_still_raises(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        # A self-loop wire (skipped by the self-loop filter) whose input index is out of range.
        diagram.add_wire(
            PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.INPUT, 9)
        )
        with pytest.raises(RewriteGrammarError):
            find_matches(diagram)


class TestFusionMatchIsHashable:
    """Task 1 (Phase 5 closing round): ``FusionMatch`` was unhashable because the
    dataclass-generated ``__hash__`` hashed ``bindings`` -- a ``MappingProxyType`` -- verbatim.
    This fired even for the empty default, so no ``FusionMatch`` was ever hashable, and
    ``RewriteStep.__hash__`` (which embeds ``match`` unchanged) inherited the same failure.
    """

    def test_hash_succeeds_with_empty_bindings(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        assert match.bindings == {}
        hash(match)  # must not raise

    def test_hash_succeeds_with_non_empty_bindings(self) -> None:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.symbol("d")])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        match = find_matches(diagram)[0]
        assert dict(match.bindings) == {"d": Dim.concrete(2)}
        hash(match)  # must not raise

    def test_equal_matches_hash_equal_and_are_usable_as_set_members(self) -> None:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.symbol("d")])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        match_one = find_matches(diagram)[0]
        match_two = find_matches(diagram)[0]
        assert match_one == match_two
        assert hash(match_one) == hash(match_two)
        assert {match_one, match_two} == {match_one}
        assert {match_one: "a value"}[match_two] == "a value"


class TestPhaseDimensionAgreementDeferredFidelity:
    """``phase_dimension_agreement`` never reports ``deferred=True`` on a passing outcome
    (Phase 5 post-closing audit, judgement call 2, decided): it now genuinely calls
    :meth:`~qufzx.algebra.dimension.Dim.unify`, but -- unlike condition 5 -- a genuinely
    ``DEFERRED`` per-phase result is rejected outright (see the module docstring, condition
    6, for why a phase cannot tolerate the same unproven assumption a leg can), so a
    *passing* outcome is never itself resting on an undecided unify. A passing outcome can
    still rest on a *binding* -- recorded in ``dimension_constraints`` -- but following
    condition 5's own convention, a binding-only success does not set ``deferred`` either.
    """

    def test_deferred_is_false_when_no_phase_present_on_either_node(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        outcome = next(
            o for o in match.side_condition_outcomes if o.name == "phase_dimension_agreement"
        )
        assert outcome.passed
        assert outcome.deferred is False

    def test_deferred_is_false_even_when_phase_agreement_rests_on_a_binding(self) -> None:
        # A's phase is stated over the still-symbolic d; the connecting leg binds d := 3,
        # so phase agreement holds only because that binding is substituted in first -- an
        # assumption, recorded in dimension_constraints, but not one that sets `deferred`
        # (a binding-only success is not the same as a genuinely undecided DEFERRED result;
        # see the class docstring).
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {1: Phase.turns(1)})
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[three], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        match = find_matches(diagram)[0]
        outcome = next(
            o for o in match.side_condition_outcomes if o.name == "phase_dimension_agreement"
        )
        assert outcome.passed
        assert outcome.deferred is False
        assert DimensionConstraint(
            assumed=d,
            equal_to=three,
            source=ConstraintSource.connecting_pair(),
            outcome=ConstraintOutcome.BOUND,
        ) in match.dimension_constraints

    def test_a_genuinely_deferred_phase_dim_is_a_non_match(self) -> None:
        # Judgement call 2 (Phase 5 post-closing audit): unlike a leg, a phase whose dim
        # only DEFERS against shared_dim (never binds, never a bare identity) cannot be
        # safely reattached -- there is no binding to substitute through its entries, so
        # this must be a non-match, not an accepted-with-assumption pass.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d * e, {1: Phase.turns(1)})
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()

    def test_deferred_is_false_when_phase_agrees_outright_despite_an_unrelated_defer(
        self,
    ) -> None:
        # The connecting pair's own leg dims are both the bare symbol d (syntactic
        # identity, no defer, no binding); A carries an extra surviving leg over d*e,
        # which defers against shared_dim=d (a symbol occurring as a proper subterm of the
        # other side). B's phase is stated directly over d, agreeing with shared_dim
        # outright -- this must not inherit the unrelated surviving leg's defer.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d * e], output_dims=[d])
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[d], output_dims=[], phase=PhaseVector(d, {1: Phase.turns(1)})
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        match = find_matches(diagram)[0]
        dimension_outcome = next(
            o for o in match.side_condition_outcomes if o.name == "dimension_agreement"
        )
        assert dimension_outcome.deferred is True
        phase_outcome = next(
            o for o in match.side_condition_outcomes if o.name == "phase_dimension_agreement"
        )
        assert phase_outcome.passed
        assert phase_outcome.deferred is False


class TestSameGeneratorTypeReportsUnregisteredFusabilityItself:
    """Phase 5 post-closing audit spec/doc defect: a same-typed pair whose shared type is
    not a registered fusable generator (i.e. not Z or X) must fail ``same_generator_type``
    itself, with that as the reported reason -- not pass it (the two types genuinely are
    equal) only to have the real cause surface as an aside buried inside
    ``consumed_wire_direction_permitted_for_color``'s detail, a condition whose own
    declared description is about wire direction, not fusability.
    """

    def test_same_unregistered_type_fails_same_generator_type_with_the_real_reason(self) -> None:
        foreign = GeneratorType(
            name="foreign",
            leg_policy=LegPolicy(),
            phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
            dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
        )
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(foreign, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(foreign, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        same_type = next(o for o in resolution.outcomes if o.name == "same_generator_type")
        assert not same_type.passed
        assert "not a registered fusable generator type" in same_type.detail

        direction = next(
            o
            for o in resolution.outcomes
            if o.name == "consumed_wire_direction_permitted_for_color"
        )
        assert not direction.passed
        assert direction.detail == "not evaluated: same_generator_type failed first"

        assert find_matches(diagram) == ()


class TestResolveFusionMatchIsTheSharedPredicate:
    """A1/A2/A4 (Phase 5 round-12 audit): ``find_matches`` and ``spider_fusion_builder`` must
    call the exact same function object to decide/re-verify a candidate -- not two
    similar-looking computations kept in sync by hand. See ``resolve_fusion_match``'s and the
    module docstring's "One verification predicate" section.
    """

    def test_spider_fusion_builder_calls_the_same_function_object(self) -> None:
        import inspect

        source = inspect.getsource(spider_fusion_builder)
        assert "resolve_fusion_match(" in source

    def test_raises_grammar_error_for_the_same_node_twice(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(a_id, Direction.OUTPUT, 1))
        with pytest.raises(RewriteGrammarError):
            resolve_fusion_match(diagram, a_id, a_id, wire)

    def test_raises_grammar_error_for_an_absent_node_id(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        from qufzx.diagram.graph import NodeId

        with pytest.raises(RewriteGrammarError):
            resolve_fusion_match(diagram, a_id, NodeId(9999), wire)

    def test_raises_grammar_error_when_wire_does_not_join_the_given_pair(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        unrelated_wire = Wire(PortRef(c_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(unrelated_wire.a, unrelated_wire.b)
        with pytest.raises(RewriteGrammarError):
            resolve_fusion_match(diagram, a_id, b_id, unrelated_wire)

    def test_returns_a_failed_resolution_with_six_outcomes_for_a_z_x_pair(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        assert len(resolution.outcomes) == len(FUSION_SIDE_CONDITIONS)
        assert {o.name for o in resolution.outcomes} == {c.name for c in FUSION_SIDE_CONDITIONS}
        same_type = next(o for o in resolution.outcomes if o.name == "same_generator_type")
        assert not same_type.passed
        # Phase 5 post-closing audit: a failed resolution's shared_dim is unrepresentable
        # as a meaningful Dim, not a placeholder concrete value a caller ignoring `passed`
        # could silently read as though it meant something.
        assert resolution.shared_dim is None

    def test_passing_resolution_agrees_with_find_matches_own_fields(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        resolution = resolve_fusion_match(diagram, match.a_id, match.b_id, match.wire)
        assert resolution.passed
        assert resolution.shared_dim == match.shared_dim
        assert dict(resolution.bindings) == dict(match.bindings)
        assert resolution.outcomes == match.side_condition_outcomes

    def test_raises_grammar_error_for_a_wire_incident_on_both_but_not_in_the_diagram(
        self,
    ) -> None:
        """Defect 1 (Phase 5 post-closing audit): a ``Wire`` naming two ports that are each
        genuinely incident on ``a_id``/``b_id`` -- so it passes every earlier structural
        check -- but that was never actually added to the diagram (a caller merely
        constructed it to look like one) must still be refused, not treated as a legal
        fusion occurrence.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])

        # Both endpoints are real, in-range ports on a_id/b_id respectively -- but this
        # particular Wire object was never passed to diagram.add_wire.
        ghost_wire = Wire(PortRef(a_id, Direction.INPUT, 0), PortRef(b_id, Direction.OUTPUT, 0))
        assert ghost_wire not in diagram.wires
        with pytest.raises(RewriteGrammarError):
            resolve_fusion_match(diagram, a_id, b_id, ghost_wire)


class TestFixpointBudgetExhaustion:
    """D4: cap exhaustion is a resolver *iteration budget*, never a dimension disagreement.

    ``_MAX_FIXPOINT_PASSES`` is unreachable under ``Dim.unify``'s current contract for any
    diagram with fewer distinct dimension symbols than the cap (see the module-level
    constant's own docstring), so this test forces the cap artificially low -- to zero, so
    even a diagram that would ordinarily match in a single pass now exhausts the budget
    before ever stabilizing -- and checks the failure is reported honestly: both
    ``dimension_agreement`` and ``phase_dimension_agreement`` fail with a detail naming the
    iteration budget, never a detail claiming a phase or leg dimension disagreement that was
    never actually checked.
    """

    def test_cap_exhaustion_reports_the_budget_not_a_phase_disagreement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import qufzx.rewrite.match as match_module

        monkeypatch.setattr(match_module, "_MAX_FIXPOINT_PASSES", 0)

        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        wire = next(
            w
            for w in diagram.wires
            if {w.a.node_id, w.b.node_id} == {min(a_id, b_id), max(a_id, b_id)}
        )
        resolution = resolve_fusion_match(diagram, min(a_id, b_id), max(a_id, b_id), wire)

        assert not resolution.passed
        by_name = {outcome.name: outcome for outcome in resolution.outcomes}
        assert by_name["dimension_agreement"].passed is False
        assert by_name["phase_dimension_agreement"].passed is False
        for name in ("dimension_agreement", "phase_dimension_agreement"):
            detail = by_name[name].detail.lower()
            assert "budget" in detail or "_max_fixpoint_passes" in detail
            assert "phase dimension" not in detail
            assert "does not unify" not in detail

        # find_matches must not surface this candidate as a match at all -- an exhausted
        # budget is a non-match, not a passing-with-a-caveat one.
        assert find_matches(diagram) == ()
