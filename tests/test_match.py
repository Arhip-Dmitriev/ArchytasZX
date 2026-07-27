"""Establishes qufzx.rewrite.match: fusion-pattern matching, its side conditions, and ordering."""

from __future__ import annotations

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionPattern, find_matches

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


class TestNoMatchAcrossTwoParallelWires:
    def test_two_wires_between_same_pair_refuses_to_match(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
        assert find_matches(diagram) == ()


class TestNoMatchForSelfLoop:
    def test_self_loop_on_single_spider_is_not_fusion(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()


class TestDimensionMismatchIsNonMatch:
    def test_different_concrete_dims_refuse_to_match(self) -> None:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()


class TestWireDirectionMustBeOutputToInput:
    def test_output_to_output_wire_refuses_to_match(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.OUTPUT, 0))
        assert find_matches(diagram) == ()


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
    def test_both_phases_present_with_mismatched_dims_is_non_match(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d], phase=_phase_at(d, 1)
        )
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[d], output_dims=[], phase=_phase_at(Dim.concrete(3), 1)
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()

    def test_one_sided_phase_differing_from_shared_leg_dim_is_non_match(self) -> None:
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
        assert match.dimension_constraints == ((d, d * e),)
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
        assert match.dimension_constraints == ((d, three),)
        agreement = next(
            o for o in match.side_condition_outcomes if o.name == "dimension_agreement"
        )
        assert not agreement.deferred


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
