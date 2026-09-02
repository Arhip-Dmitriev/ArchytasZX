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
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

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
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from qufzx.diagram.validate import IssueKind, validate
from qufzx.rewrite.engine import apply
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
    RewriteDomainError,
    RewriteGrammarError,
    SideConditionOutcome,
)
from qufzx.rewrite.rules_library import SPIDER_FUSION, spider_fusion_builder

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
        """``_unify_surviving_legs`` needs no change for parallel wires -- the
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
        """``(a_id, b_id)`` is no longer a unique key for a candidate, so
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
        # Both legs share the still-symbolic d (shared_dim=d, unbound), A's phase agrees
        # outright (also over d), and B's phase over the concrete 3 binds d := 3 through
        # condition 7's own unify call. Neither phase's entries reference a dimension
        # symbol, so reattach_phase has nothing to substitute.
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
            bound_here=(("d", Dim.concrete(3)),),
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
            bound_here=(("d", Dim.concrete(3)),),
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


class TestPhaseFailureDoesNotMisreportDimensionAgreement:
    """A phase-dim FAILURE inside :func:`resolve_fusion_match`'s fixpoint must not be reported
    as a ``dimension_agreement`` (condition 6) failure -- nothing about a leg's own
    dimension failed here, only a phase's.


    Reproducer straight from the audit brief: a Z-Z pair, every leg ``Dim(2)``, node B's
    phase stated over the concrete ``Dim(3)`` -- plainly non-unifiable with the legs' shared
    ``Dim(2)``. Before the fix, this fell through to ``_verify_fixpoint_closure`` (which
    re-checks phases too, under the identical contract that just failed them) and reported
    BOTH ``dimension_agreement`` and ``phase_dimension_agreement`` as ``False``, both citing
    the closure guard's own "this is unreachable" message -- false on two counts: the legs
    plainly do agree exactly (``Dim(2) == Dim(2)``, no unify even needed), and the guard's
    unreachability claim is not actually true on this path at all.
    """

    def test_phase_failure_reports_leg_accurate_dimension_agreement_and_dedicated_phase_detail(
        self,
    ) -> None:
        two = Dim.concrete(2)
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[two])
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[two], output_dims=[], phase=_phase_at(three, 1)
        )
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed

        by_name = {outcome.name: outcome for outcome in resolution.outcomes}
        dimension_agreement = by_name["dimension_agreement"]
        phase_dimension_agreement = by_name["phase_dimension_agreement"]

        assert dimension_agreement.passed, (
            "every leg here is exactly Dim(2) == Dim(2); condition 6 must pass, not be "
            f"misreported as failed alongside the phase failure: {dimension_agreement!r}"
        )
        assert "2 == 2" in dimension_agreement.detail
        assert not phase_dimension_agreement.passed
        assert "does not unify with the resolved shared leg dimension" in (
            phase_dimension_agreement.detail
        )

        closure_marker = "post-loop closure check failed"
        assert closure_marker not in dimension_agreement.detail
        assert closure_marker not in phase_dimension_agreement.detail

        # find_matches must drop this candidate entirely (all_side_conditions_passed is
        # False), not surface it as a match with a failing outcome.
        assert find_matches(diagram) == ()

    def test_phase_a_binding_before_phase_b_fails_does_not_leak_into_dimension_agreement(
        self,
    ) -> None:
        """The subtlety the fix's docstring calls out: ``_unify_phase_dims`` can bind phase
        A's own symbol before failing on phase B, in the very same call -- advancing
        ``shared_dim``/``bindings`` past what the leg sweep last verified against.
        ``dimension_agreement`` must be reported against the leg-verified snapshot (here:
        the bare identity ``e == e``, nothing bound yet), never against the state
        ``shared_dim`` is advanced to by A's own phase binding partway through the same
        ``_unify_phase_dims`` call that then fails on B's phase.
        """
        e = Dim.symbol("e")
        diagram = Diagram()
        # Both legs are the same bare symbol e: the connecting pair unifies as a bare
        # identity, seeding shared_dim=e with nothing bound and no surviving legs on
        # either side (each node's only leg is the one the wire consumes).
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[e], phase=_phase_at(Dim.concrete(3), 1)
        )
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[e], output_dims=[], phase=_phase_at(Dim.concrete(4), 1)
        )
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        by_name = {outcome.name: outcome for outcome in resolution.outcomes}
        dimension_agreement = by_name["dimension_agreement"]
        # A's phase (3) unifies against the leg-seeded shared_dim=e by binding e := 3,
        # advancing shared_dim to 3 -- all within the same _unify_phase_dims call that then
        # fails B's phase (4) against that advanced 3. dimension_agreement must still be
        # reported True, from the *leg*-verified state (e == e, nothing bound), not from the
        # state a phase's own binding advanced shared_dim to afterwards.
        assert dimension_agreement.passed
        assert "e == e" in dimension_agreement.detail, dimension_agreement.detail
        assert "3" not in dimension_agreement.detail, (
            f"dimension_agreement leaked the phase-advanced shared_dim=3 into its own "
            f"detail, which the leg sweep was never actually checked against: "
            f"{dimension_agreement!r}"
        )
        # phase_dimension_agreement's own detail must name the value B's phase was checked
        # against (3, refined by A's binding within this same _unify_phase_dims call), not
        # the pre-call shared_dim (e).
        phase_dimension_agreement = by_name["phase_dimension_agreement"]
        assert not phase_dimension_agreement.passed
        assert "3" in phase_dimension_agreement.detail, phase_dimension_agreement.detail
        assert "dimension e " not in phase_dimension_agreement.detail, (
            f"phase_dimension_agreement reported a stale pre-call shared_dim: "
            f"{phase_dimension_agreement!r}"
        )


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
                bound_here=(("d", three),),
            ),
        )
        agreement = next(
            o for o in match.side_condition_outcomes if o.name == "dimension_agreement"
        )
        assert not agreement.deferred

    def test_non_concrete_binding_detail_says_something_was_assumed(self) -> None:
        """A connecting pair recorded BOUND via a symbol-to-symbol binding (``d := e``, non-
        concrete -- see the module docstring's "Non-concrete bindings") must not render as a
        bare "d == e" with no indication anything was assumed.


        Root cause: the pre-fix ``_connecting_pair_detail`` derived its "bound to what"
        clause by intersecting the raw legs' free symbols with ``bindings`` -- but
        ``bindings`` (by design) never holds a non-concrete value at all
        (``_merge_bindings`` filters those out before they reach it), so that intersection
        was always empty for exactly this case, even though ``entry.outcome`` -- and
        ``dimension_constraints`` beside it -- correctly recorded BOUND. Fixed by deriving
        the detail from the record entry itself (the single source of truth
        ``dimension_constraints`` also reads from), not from a collection filtered for a
        different purpose.
        """
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[e], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        match = matches[0]

        # The machine-readable record was always right: exactly one BOUND connecting-pair
        # constraint, and match.bindings itself legitimately stays empty (the binding is
        # non-concrete, so _merge_bindings never stores it -- see the module docstring).
        assert match.dimension_constraints == (
            DimensionConstraint(
                assumed=d,
                equal_to=e,
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.BOUND,
                bound_here=(("d", e),),
            ),
        )
        assert dict(match.bindings) == {}

        # The human-readable detail must agree with that record: it must say a binding was
        # assumed, not render as a bare, unqualified equality.
        agreement = next(
            o for o in match.side_condition_outcomes if o.name == "dimension_agreement"
        )
        assert "d == e" in agreement.detail
        assert "bound" in agreement.detail, (
            f"detail gives no indication a non-concrete binding was assumed: {agreement!r}"
        )

    def test_a_leg_already_bound_by_the_connecting_pair_is_not_recorded_again(self) -> None:
        """``dimension_constraints`` records a duplicate assumption exactly once.

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
                bound_here=(("d", two),),
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
                bound_here=(("d", two),),
            ),
        )


class TestOutOfRangeWireEndpointRaises:
    """``graph.py`` is permissive about wire port indices; ``find_matches`` must not be.

    An out-of-range index in either wire endpoint raises ``RewriteGrammarError``, not a
    bare ``IndexError`` from ``node.legs(direction)[index]``, before any dimension work.
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


class TestOutOfRangeBoundaryRefRaises:
    """A boundary entry is held to the identical malformed-reference standard as a wire
    endpoint (see :class:`TestOutOfRangeWireEndpointRaises` above, and the module
    docstring's "Malformed boundary references" section).


    Before this fix, an out-of-range or unknown-node boundary entry was not checked by
    :func:`find_matches` at all; it reached :mod:`qufzx.rewrite.engine`'s ``apply`` step 5
    unexamined, surfacing (if it ever did) as a *different* error class
    (``RewriteDomainError`` from ``_remap_endpoint``, not ``RewriteGrammarError`` from this
    module) and only when the ref happened to sit on a consumed port. Every combination
    below is covered: out-of-range index and unknown node id, on each of
    ``boundary_inputs``/``boundary_outputs``, on both a consumed node (one of the fusing
    pair) and a non-consumed one -- eight cases, since nothing about this check may depend
    on any of those properties.
    """

    @staticmethod
    def _base_diagram() -> tuple[Diagram, PortRef, PortRef]:
        """Two fusable Z spiders (A, B) joined by one wire, plus an unrelated bystander C
        with one free output leg -- the "non-consumed node" boundary target below."""
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs(
            [PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.OUTPUT, 0)]
        )
        return diagram, PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.OUTPUT, 0)

    def test_out_of_range_index_on_boundary_outputs_on_a_consumed_node_raises(self) -> None:
        diagram, consumed_ref, _bystander = self._base_diagram()
        outs = list(diagram.boundary_outputs)
        outs[outs.index(consumed_ref)] = PortRef(consumed_ref.node_id, Direction.OUTPUT, 9)
        diagram.set_boundary_outputs(outs)
        with pytest.raises(RewriteGrammarError, match="out of range"):
            find_matches(diagram)

    def test_out_of_range_index_on_boundary_outputs_on_a_non_consumed_node_raises(self) -> None:
        diagram, _consumed_ref, bystander = self._base_diagram()
        outs = list(diagram.boundary_outputs)
        outs[outs.index(bystander)] = PortRef(bystander.node_id, Direction.OUTPUT, 9)
        diagram.set_boundary_outputs(outs)
        with pytest.raises(RewriteGrammarError, match="out of range"):
            find_matches(diagram)

    def test_unknown_node_id_on_boundary_outputs_on_a_consumed_node_position_raises(
        self,
    ) -> None:
        diagram, consumed_ref, _bystander = self._base_diagram()
        outs = list(diagram.boundary_outputs)
        outs[outs.index(consumed_ref)] = PortRef(NodeId(999999), Direction.OUTPUT, 0)
        diagram.set_boundary_outputs(outs)
        with pytest.raises(RewriteGrammarError, match="absent from the diagram"):
            find_matches(diagram)

    def test_unknown_node_id_on_boundary_outputs_on_a_non_consumed_node_position_raises(
        self,
    ) -> None:
        diagram, _consumed_ref, bystander = self._base_diagram()
        outs = list(diagram.boundary_outputs)
        outs[outs.index(bystander)] = PortRef(NodeId(999999), Direction.OUTPUT, 0)
        diagram.set_boundary_outputs(outs)
        with pytest.raises(RewriteGrammarError, match="absent from the diagram"):
            find_matches(diagram)

    def test_out_of_range_index_on_boundary_inputs_on_a_consumed_node_raises(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 4)])
        with pytest.raises(RewriteGrammarError, match="out of range"):
            find_matches(diagram)

    def test_out_of_range_index_on_boundary_inputs_on_a_non_consumed_node_raises(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(c_id, Direction.INPUT, 4)])
        with pytest.raises(RewriteGrammarError, match="out of range"):
            find_matches(diagram)

    def test_unknown_node_id_on_boundary_inputs_on_a_consumed_node_position_raises(
        self,
    ) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        # Overwrite what would otherwise be a's legitimate boundary input with an unknown
        # node id -- "on a consumed node position" in the sense that the slot this boundary
        # entry occupies is the one a's own surviving input would otherwise need.
        diagram.set_boundary_inputs([PortRef(NodeId(999999), Direction.INPUT, 0)])
        with pytest.raises(RewriteGrammarError, match="absent from the diagram"):
            find_matches(diagram)

    def test_unknown_node_id_on_boundary_inputs_on_a_non_consumed_node_position_raises(
        self,
    ) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(NodeId(999999), Direction.INPUT, 0)])
        with pytest.raises(RewriteGrammarError, match="absent from the diagram"):
            find_matches(diagram)


class TestSharedDimResolvesThroughBinding:
    """Shared_dim must be resolved by substituting a unify binding, not taken raw from A.

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
        # B's phase is stated over Dim(3), the concrete value d unifies to, not over the
        # still-symbolic leg dim d itself.
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
        # Node A's phase is stated over the symbolic leg dim d, node B's over the bound
        # concrete value 3: both resolve to shared_dim=3 (A's via a binding condition 7
        # derives, B's outright), while their raw, unsubstituted Dims differ. Two present
        # phases' raw Dims are not required to be equal -- reattach_phase forces both
        # operands' container Dim to shared_dim before PhaseVector.__add__ sees them.
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
    """``phase_dimension_agreement`` resolves a present phase's Dim through every binding
    accumulated so far, including one a *surviving* leg produced -- not only the connecting
    pair's own. A phase stated over a symbol that only a surviving leg binds still matches.
    """

    def test_a_phase_over_a_symbol_bound_only_by_a_surviving_leg_now_matches(self) -> None:
        # A's connecting leg (output 0) and B's connecting leg (input 0) are both the
        # symbol d -- unifying them binds nothing. A's *surviving* input leg is a
        # concrete Dim(2), which binds d := 2 once condition 6 unifies it against
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
    """A surviving leg that conflicts with ``shared_dim`` makes the candidate a non-match.

    It is never forced onto ``shared_dim`` unchecked, which would overwrite a real
    dimension or launder a pre-existing hard error.
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
            bound_here=(("d", two),),
        ) in match.dimension_constraints


class TestConsumedPortClaimedElsewhereIsNonMatch:
    """A consumed port also named by a second wire, or also on a boundary list, used to still
    be returned as a match -- apply() then raised RewriteDomainError (unmapped consumed
    port), breaking the documented "every match this function returns can be applied"
    invariant. find_matches must reject such candidates itself, since a port claimed twice
    like this is not a genuine fusion occurrence."""

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

    def test_wired_twice_is_reported_as_consumed_ports_singly_claimed_failing(self) -> None:
        """The check is now a real, named side condition decided by ``resolve_fusion_match``
        itself -- not merely a filter inside ``find_matches`` invisible to the certificate.
        Call ``resolve_fusion_match`` directly (bypassing ``find_matches``'s own filter,
        which no longer exists) and check the named outcome appears, failing, with the
        remaining dimension conditions marked "not evaluated"."""
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        by_name = {o.name: o for o in resolution.outcomes}
        assert set(by_name) == {c.name for c in FUSION_SIDE_CONDITIONS}
        claim_outcome = by_name["consumed_ports_singly_claimed"]
        assert claim_outcome.passed is False
        assert "claimed by 1 other wire" in claim_outcome.detail
        assert by_name["dimension_agreement"].passed is False
        assert "not evaluated" in by_name["dimension_agreement"].detail
        assert by_name["phase_dimension_agreement"].passed is False

    def test_on_boundary_is_reported_as_consumed_ports_singly_claimed_failing(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 0)])

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        by_name = {o.name: o for o in resolution.outcomes}
        claim_outcome = by_name["consumed_ports_singly_claimed"]
        assert claim_outcome.passed is False
        assert "listed on a boundary list" in claim_outcome.detail

    def test_a_hand_built_match_over_a_multiply_claimed_port_is_rejected_by_the_builder(
        self,
    ) -> None:
        """The gap Task 2 closes: a hand-built ``FusionMatch`` claiming
        ``consumed_ports_singly_claimed`` passed used to sail through
        ``spider_fusion_builder`` and only fail deep in ``apply`` step 5 (a different error
        class, from a different module). It must now be rejected by the builder's own
        ``resolve_fusion_match`` re-verification, before any graph surgery.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))

        from qufzx.rewrite.match import FusionMatch

        fabricated = FusionMatch(
            a_id=a_id,
            b_id=b_id,
            wire=wire,
            shared_dim=d,
            side_condition_outcomes=tuple(
                SideConditionOutcome(c.name, True, "fabricated") for c in FUSION_SIDE_CONDITIONS
            ),
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, fabricated)


class TestSurvivingLegOverwriteIntroducesDeferral:
    """Forcing a surviving leg onto ``shared_dim`` (module docstring, condition 6) can turn a
    wire to a third node that agreed *exactly* before fusion into one that only defers
    afterward -- a routine outcome (roughly one in ten applications over the random property
    harness, not a rare edge case), and one this pattern's own carve-out
    (:mod:`qufzx.rewrite.engine`'s step-8 relative postcondition) correctly permits.
    Constructed deliberately here, rather than left to be rediscovered as folklore by
    whoever next reads the property harness's floor numbers."""

    def test_exact_third_party_match_becomes_deferred_after_fusion(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        # A's consumed leg and B's consumed leg are both literally `d` -- a bare identity,
        # so the connecting pair unifies trivially and shared_dim stays raw `d`, unchanged.
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d * e])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        # C's port dim is exactly `d * e`, matching A's surviving leg exactly before fusion.
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d * e], output_dims=[])
        consumed = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(consumed.a, consumed.b)
        surviving_wire = Wire(
            PortRef(a_id, Direction.OUTPUT, 1), PortRef(c_id, Direction.INPUT, 0)
        )
        diagram.add_wire(surviving_wire.a, surviving_wire.b)

        # A's own legs (d, d*e) are themselves ALL_LEGS_EQUAL-deferred against each other --
        # an unrelated, pre-existing node-level assumption this test does not care about.
        # What matters is the A-C *wire* specifically: d*e == d*e, an exact match, no issue.
        pre_report = validate(diagram)
        assert not any(
            issue.wire == surviving_wire
            and issue.kind in (IssueKind.DIMENSION_MISMATCH, IssueKind.DIMENSION_DEFERRED)
            for issue in (*pre_report.errors, *pre_report.deferred)
        ), f"the A-C wire must agree exactly before fusion: {pre_report!r}"

        matches = find_matches(diagram)
        match = next(m for m in matches if m.wire == consumed)
        # Surviving leg A.output[1] (d*e) unifies against shared_dim=d as DEFERRED (the same
        # shape as the module docstring's own `d` vs `d*e` example) -- not a FAILURE, so this
        # is a genuine match, not rejected.
        assert match.shared_dim == d

        result = apply(diagram, SPIDER_FUSION, match)

        post_report = validate(result.diagram)
        deferred_kinds = {issue.kind for issue in post_report.deferred}
        assert IssueKind.DIMENSION_DEFERRED in deferred_kinds, (
            f"expected the merged-node-to-C wire to now be DIMENSION_DEFERRED: {post_report!r}"
        )
        assert not post_report.errors, (
            f"this shape must not introduce a hard error -- DEFERRED, not MISMATCH: "
            f"{post_report!r}"
        )


class TestMalformedWireDetectionIsFullyUnconditional:
    """The out-of-range-index check runs over every wire before any grouping or filtering.

    A malformed wire that grouping would drop (a parallel-wire pair, a self-loop) still
    raises rather than being silently skipped.
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
    """``FusionMatch`` was unhashable because the
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
    """``phase_dimension_agreement`` never reports ``deferred=True`` on a passing outcome.

    It calls :meth:`~qufzx.algebra.dimension.Dim.unify`, but -- unlike condition 6 -- a
    ``DEFERRED`` per-phase result is rejected outright (module docstring, condition 7), so
    a passing outcome never rests on an undecided unify. A passing outcome can rest on a
    *binding*, recorded in ``dimension_constraints``; following condition 6's convention,
    a binding-only success does not set ``deferred`` either.
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
            bound_here=(("d", three),),
        ) in match.dimension_constraints

    def test_a_genuinely_deferred_phase_dim_is_a_non_match(self) -> None:
        # Unlike a leg, a phase whose dim only DEFERS against shared_dim (never binds,
        # never a bare identity) is a non-match, not an accepted-with-assumption pass.
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
    """A same-typed pair whose shared type is not a registered fusable generator (not Z or
    X) must fail ``same_generator_type`` itself, with that as the reported reason -- not
    pass it, on the grounds that the two types genuinely are equal, and leave the real cause
    buried inside ``consumed_wire_direction_permitted_for_color``'s detail, a condition
    whose declared description is about wire direction, not fusability.
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
    """``find_matches`` and ``spider_fusion_builder`` call the same function object to
    decide and re-verify a candidate.

    See ``resolve_fusion_match`` and the module docstring's "One verification predicate".
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
        # A failed resolution reports shared_dim as None, never a placeholder Dim.
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
        """A ``Wire`` naming two ports that are each genuinely incident on ``a_id``/``b_id`` --
        so it passes every earlier structural check -- but that was never actually added to
        the diagram (a caller merely constructed it to look like one) must still be refused,
        not treated as a legal fusion occurrence."""
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
    """Cap exhaustion is a resolver *iteration budget*, never a dimension disagreement.

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


class TestD1FixpointTerminationSoundness:
    """The joint condition-5/6 fixpoint used to exit as soon as ``shared_dim`` stopped
    changing, even when ``bindings`` had just grown -- e.g. by binding a symbol that does
    not occur in ``shared_dim`` at all. Legs (and phases) checked earlier in that same pass
    were then never re-checked against the newly accumulated binding, and an unsatisfiable
    constraint set (recorded assumptions that cannot simultaneously hold) was accepted as a
    match. Fixed by terminating only when a full pass leaves both ``shared_dim`` and
    ``bindings`` unchanged, and by rejecting a contradictory rebind outright
    (:func:`~qufzx.rewrite.match._merge_bindings`) rather than silently overwriting."""

    def test_d1_reproduction_e_times_f_e_f_against_2_is_not_a_match(self) -> None:
        """The exact reproduction from the audit brief: A's legs [e*f, e, f] against B's
        single leg 2 record the unsatisfiable set e*f == 2, e == 2, f == 2. Before the fix
        this was silently accepted (the e*f leg overwritten with shared_dim=2, contradicting
        its own recorded DEFERRED assumption)."""
        e = Dim.symbol("e")
        f = Dim.symbol("f")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[e * f, e, f], output_dims=[Dim.concrete(2)])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, i) for i in range(3)])
        # b_id has zero output legs, so there is no boundary_outputs entry to declare for
        # it: naming a nonexistent PortRef(b_id, OUTPUT, 0) here is a malformed boundary
        # reference, which find_matches rejects.
        assert find_matches(diagram) == ()

    def test_legs_2_d_times_e_d_conn_e_is_not_a_match(self) -> None:
        """['2', 'd*e', 'd'] conn e: shared_dim seeds at 'e' (bare identity vs the connecting
        pair). '2' binds e := 2 (refining shared_dim to 2); 'd*e' defers against the
        as-yet-unrefined shared_dim on the same pass, then on the next pass resolves (via
        both e and d now bound) to the concrete 4, contradicting shared_dim=2. The bounded
        'd' leg (d := 2) is independently consistent -- it is 'd*e', not 'd', whose deferral
        must be re-checked once e is bound, not accepted on its first pass's outcome alone."""
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[Dim.concrete(2), d * e, d], output_dims=[e]
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[e], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, i) for i in range(3)])
        assert find_matches(diagram) == ()

    def test_legs_d_d_times_e_e_conn_2_is_not_a_match(self) -> None:
        """['d', 'd*e', 'e'] conn 2: the connecting pair binds nothing new (2 == 2 outright,
        shared_dim=2 from the start). 'd' binds d := 2; 'd*e' defers (its own symbol e is
        still free); 'e' independently binds e := 2. Once both are bound, 'd*e' resolves to
        the concrete 4 on the next pass, contradicting shared_dim=2 -- unsatisfiable once
        every leg is checked consistently against the fully-refined bindings."""
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[d, d * e, e], output_dims=[Dim.concrete(2)]
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, i) for i in range(3)])
        assert find_matches(diagram) == ()

    def test_legs_d_squared_2_d_against_2_is_not_a_match(self) -> None:
        """['d**2', '2', 'd'] conn 2: the 'd**2' leg deferred against shared_dim=2 (a proper
        power, not a bare symbol, so Dim.unify cannot bind it), while the concrete '2' leg
        confirms shared_dim=2 outright, and the final 'd' leg then binds d := 2 -- consistent
        with 'd**2' only at d = sqrt(2), not an integer, so the deferred assumption d**2 == 2
        and the bound assumption d == 2 cannot simultaneously hold. This shape is exactly why
        the post-loop closure check re-verifies every source under the *final* bindings, not
        only the bindings each source happened to see when it was itself last checked."""
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[d**2, Dim.concrete(2), d], output_dims=[Dim.concrete(2)]
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, i) for i in range(3)])
        assert find_matches(diagram) == ()

    def test_leg_order_independence_of_match_and_final_state(self) -> None:
        """Permuting a node's leg order (with its boundary entries permuted correspondingly)
        must not change match-vs-non-match, nor the final shared_dim, nor the final bindings
        set -- constraint order may differ, but the set of what was assumed must not. This is
        the direct regression guard for D1's root cause: leg-order dependence was the
        symptom that exposed it (``[e*f, e, f]`` was accepted, ``[e, f, e*f]`` correctly
        rejected)."""
        e = Dim.symbol("e")
        f = Dim.symbol("f")
        two = Dim.concrete(2)

        def _shape(order: tuple[Dim, ...]) -> tuple[bool, Dim | None, frozenset[tuple[str, Dim]]]:
            diagram = Diagram()
            a_id = diagram.add_node(Z_SPIDER, input_dims=list(order), output_dims=[two])
            b_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[])
            diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
            diagram.set_boundary_inputs(
                [PortRef(a_id, Direction.INPUT, i) for i in range(len(order))]
            )
            matches = find_matches(diagram)
            if not matches:
                return False, None, frozenset()
            m = matches[0]
            return True, m.shared_dim, frozenset(dict(m.bindings).items())

        # A satisfiable permutation family: connecting pair forces shared_dim=2 outright
        # (both sides concrete), and surviving legs {e, f} independently bind e := 2, f := 2
        # regardless of which is checked first.
        satisfiable_orders = [(e, f), (f, e)]
        results = [_shape(order) for order in satisfiable_orders]
        assert all(r == results[0] for r in results)
        assert results[0][0] is True
        assert results[0][1] == two
        assert results[0][2] == frozenset({("e", two), ("f", two)})

        # The D1 reproduction family: [e*f, e, f] is unsatisfiable in every leg order, since
        # the recorded constraint set is the same set regardless of check order.
        unsatisfiable_orders = [
            (e * f, e, f),
            (e, f, e * f),
            (f, e * f, e),
        ]
        for order in unsatisfiable_orders:
            matched, _, _ = _shape(order)
            assert matched is False, f"leg order {order} unexpectedly matched"


class TestMergeBindingsRejectsContradiction:
    """:func:`~qufzx.rewrite.match._merge_bindings` must treat a would-be rebind of an
    already-bound name to a different concrete value as a hard non-match, never a silent
    last-write-wins overwrite.
    """

    def test_rebinding_to_a_different_concrete_value_returns_conflict_and_does_not_mutate(
        self,
    ) -> None:
        from qufzx.rewrite.match import _merge_bindings

        bindings = {"d": Dim.concrete(2)}
        conflict = _merge_bindings(bindings, {"d": Dim.concrete(3)})
        assert conflict == ("d", Dim.concrete(2), Dim.concrete(3))
        assert bindings == {"d": Dim.concrete(2)}

    def test_rebinding_to_the_same_concrete_value_succeeds(self) -> None:
        from qufzx.rewrite.match import _merge_bindings

        bindings = {"d": Dim.concrete(2)}
        conflict = _merge_bindings(bindings, {"d": Dim.concrete(2)})
        assert conflict is None
        assert bindings == {"d": Dim.concrete(2)}

    def test_non_concrete_bindings_are_dropped_not_merged(self) -> None:
        from qufzx.rewrite.match import _merge_bindings

        bindings: dict[str, Dim] = {}
        conflict = _merge_bindings(bindings, {"d": Dim.symbol("e")})
        assert conflict is None
        assert bindings == {}


class TestResolutionFailureReasonDetails:
    """A failure's detail string must be derived from *what actually failed*
    (:class:`~qufzx.rewrite.match._ResolutionFailure`'s ``reason``), not from which call
    site happened to return it. Pins each reason's own wording at the shape that produces
    it."""

    def test_connecting_pair_unify_failure(self) -> None:
        two = Dim.concrete(2)
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[two])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[three], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        by_name = {outcome.name: outcome for outcome in resolution.outcomes}
        detail = by_name["dimension_agreement"].detail
        assert "does not unify" in detail
        assert "2" in detail and "3" in detail
        assert "contradicts an earlier binding" not in detail

    def test_surviving_leg_unify_failure_names_the_side(self) -> None:
        two = Dim.concrete(2)
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[two, three])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        detail = {o.name: o for o in resolution.outcomes}["dimension_agreement"].detail
        assert "A-side" in detail
        assert "does not unify with shared_dim" in detail

    def test_phase_deferred_is_not_a_unify_failure_wording(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d * e], phase=_phase_at(d**2, 1)
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d * e], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        detail = {o.name: o for o in resolution.outcomes}["phase_dimension_agreement"].detail
        assert "DEFERRED" in detail
        assert "not accepted for a phase" in detail
        assert "does not unify" not in detail
        assert "non-concrete Dim" not in detail

    def test_phase_non_concrete_binding_wording(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=_phase_at(e, 1))
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        detail = {o.name: o for o in resolution.outcomes}["phase_dimension_agreement"].detail
        assert "non-concrete Dim" in detail
        assert "DEFERRED" not in detail

    def test_reattach_phase_out_of_range_is_distinct_from_in_loop_wording(self) -> None:
        """The post-loop ``reattach_phase`` failure (an entry falling out of range once
        substituted) must read differently from the in-loop non-unifying-Dim wording, per
        the module docstring's round-23 note -- not the same string with every cause listed.
        """
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[d],
            phase=PhaseVector(d, {5: Phase.turns(sp.Rational(1, 3))}),
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)

        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        assert not resolution.passed
        detail = {o.name: o for o in resolution.outcomes}["phase_dimension_agreement"].detail
        assert "falls out of range" in detail
        assert "DEFERRED" not in detail
        assert "non-concrete Dim" not in detail

    def test_contradictory_rebind_direct_call_pins_each_reachable_helpers_wording(self) -> None:
        """``CONTRADICTORY_REBIND`` is provably unreachable through the normal fixpoint (see
        ``_merge_bindings``' own docstring: every operand is pre-resolved through
        ``_resolve_with_bindings`` first, so an already-bound symbol is never free to be
        rebound) -- reachable only if a caller passes an un-pre-resolved ``shared_dim``, the
        one documented exception. ``_unify_surviving_legs`` and ``_unify_phase_dims`` both
        take ``shared_dim`` as a second, *unresolved* unify operand (only the leg's/phase's
        own dim is pre-resolved), so this shape is directly constructible for them.
        ``_unify_connecting_pair`` has no such raw operand -- both its unify operands are
        pre-resolved through ``bindings`` before ``Dim.unify`` ever sees them -- so
        ``CONTRADICTORY_REBIND`` is not merely unreachable in practice for it but
        unconstructible even by a direct call with contrived state; it is intentionally not
        exercised here for that reason (see the difference in the two functions' own
        docstrings for why one raw operand exists and the other does not).
        """
        from qufzx.rewrite.match import (
            _ConstraintRecord,
            _leg_failure_detail,
            _phase_failure_detail,
            _ResolutionFailure,
            _unify_phase_dims,
            _unify_surviving_legs,
        )

        d = Dim.symbol("d")
        five = Dim.concrete(5)

        # Surviving leg: a raw (un-pre-resolved) shared_dim of bare `d`, with an existing
        # bindings entry d := 3 that a leg's own concrete value (5) would rebind.
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(1), five])
        consumed_ref = PortRef(node_id, Direction.OUTPUT, 0)
        result = _unify_surviving_legs(
            diagram.nodes[node_id], node_id, consumed_ref, d, {"d": Dim.concrete(3)},
            _ConstraintRecord(),
        )
        assert isinstance(result, _ResolutionFailure)
        detail = _leg_failure_detail("A", result)
        assert "contradicts an earlier binding" in detail
        assert "does not unify" not in detail

        # Phase: node's own phase dim is concrete (5), shared_dim is passed raw as `d` --
        # unify(5, d) binds d := 5, contradicting the pre-seeded d := 3.
        phase_a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(1)], phase=_phase_at(five, 1)
        )
        phase_b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(1)], output_dims=[])
        phase_result = _unify_phase_dims(
            diagram.nodes[phase_a_id],
            diagram.nodes[phase_b_id],
            phase_a_id,
            phase_b_id,
            d,
            {"d": Dim.concrete(3)},
            _ConstraintRecord(),
        )
        assert isinstance(phase_result, _ResolutionFailure)
        phase_detail = _phase_failure_detail(phase_result)
        assert "contradicts an earlier binding" in phase_detail


class TestConnectingPairRederivedEachPass:
    """The connecting pair used to be recorded once, before the fixpoint, and never revisited
    -- unlike every ``SURVIVING_LEG`` and ``NODE_PHASE`` source. A phase-driven binding that
    refines the connecting pair's own legs must still show up in the finished
    ``dimension_constraints`` record at its most-resolved form."""

    def test_connecting_pair_constraint_reflects_a_later_phase_driven_binding(self) -> None:
        # Connecting pair: A's output (symbol d) vs B's input (symbol d) -- bare identity,
        # nothing bound yet, shared_dim seeds at d. A's phase, stated over the concrete 3,
        # binds d := 3 through condition 7. The connecting pair's own finished detail must
        # reflect d := 3, not the pre-fixpoint bare-identity state it started in.
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(three, {1: Phase.turns(1)})
        )
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1
        assert matches[0].shared_dim == three
        assert dict(matches[0].bindings) == {"d": three}


class TestStructuralSatisfiabilityOfEveryMatch:
    """N1's required structural-satisfiability arm: for every match ``find_matches`` returns,
    every surviving leg dim, the connecting pair's two dims, and every present phase dim --
    resolved under ``match.bindings`` -- must unify with ``shared_dim`` without ``FAILURE``,
    and the recorded ``dimension_constraints`` must be simultaneously satisfiable.
    """

    @staticmethod
    def _resolved(dim: Dim, bindings: dict[str, Dim]) -> Dim:
        concrete: dict[str | Dim, int | Dim] = {
            name: value for name, value in bindings.items() if value.is_concrete
        }
        return dim.substitute(concrete) if concrete else dim

    def _assert_match_is_structurally_satisfiable(self, diagram: Diagram) -> None:
        for match in find_matches(diagram):
            bindings = dict(match.bindings)
            node_a = diagram.nodes[match.a_id]
            node_b = diagram.nodes[match.b_id]
            ref_a = match.wire.a if match.wire.a.node_id == match.a_id else match.wire.b
            ref_b = match.wire.b if match.wire.a.node_id == match.a_id else match.wire.a

            for node, node_id, consumed_ref in (
                (node_a, match.a_id, ref_a),
                (node_b, match.b_id, ref_b),
            ):
                for direction in (Direction.INPUT, Direction.OUTPUT):
                    for index, port in enumerate(node.legs(direction)):
                        ref = PortRef(node_id, direction, index)
                        if ref == consumed_ref:
                            continue
                        resolved = self._resolved(port.dim, bindings)
                        assert not resolved.unify(match.shared_dim).is_failure

            legs_a = node_a.legs(ref_a.direction)
            legs_b = node_b.legs(ref_b.direction)
            for dim in (legs_a[ref_a.index].dim, legs_b[ref_b.index].dim):
                resolved = self._resolved(dim, bindings)
                assert not resolved.unify(match.shared_dim).is_failure

            for node in (node_a, node_b):
                if node.phase is None:
                    continue
                resolved = self._resolved(node.phase.dim, bindings)
                assert not resolved.unify(match.shared_dim).is_failure

            # The recorded constraint set is simultaneously satisfiable: every entry's own
            # `assumed`, resolved through the final bindings, unifies with its `equal_to`.
            for entry in match.dimension_constraints:
                resolved_assumed = self._resolved(entry.assumed, bindings)
                resolved_equal_to = self._resolved(entry.equal_to, bindings)
                assert not resolved_assumed.unify(resolved_equal_to).is_failure

    def test_ghz_with_copy(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        self._assert_match_is_structurally_satisfiable(diagram)

    def test_a_binding_rich_shape(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(2), d],
            output_dims=[d],
            phase=PhaseVector(d, {1: Phase.turns(1)}),
        )
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[d],
            output_dims=[e],
            phase=PhaseVector(Dim.concrete(2), {1: Phase.turns(1)}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs(
            [PortRef(a_id, Direction.INPUT, 0), PortRef(a_id, Direction.INPUT, 1)]
        )
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
        self._assert_match_is_structurally_satisfiable(diagram)
