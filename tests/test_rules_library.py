"""Establishes qufzx.rewrite.rules_library.spider_fusion_builder's merge and scalar behavior."""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef, Wire
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import RewriteDomainError, RewriteGrammarError, SideConditionOutcome
from qufzx.rewrite.rules_library import RULES, SPIDER_FUSION, lookup_rule, spider_fusion_builder

from .helpers import build_ghz_with_copy


class TestSpiderFusionMergesGhzWithCopy:
    def test_merged_node_has_expected_shape(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.generator_type is Z_SPIDER
        assert merged.num_inputs == 0
        assert merged.num_outputs == 3

    def test_scalar_introduced_is_one(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        assert result.scalar_introduced == Scalar.one()
        assert result.scalar_introduced == SPIDER_FUSION.scalar_introduced

    def test_consumed_node_ids_and_wire(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        assert result.consumed_node_ids == (min(a_id, b_id), max(a_id, b_id))
        assert result.consumed_wires == (match.wire,)

    def test_port_mapping_covers_every_surviving_port(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        # A has one surviving output (index 1); B has two surviving outputs, no
        # surviving inputs -- three surviving ports total, none of them B's consumed input.
        assert len(result.port_mapping) == 3
        assert PortRef(a_id, Direction.OUTPUT, 1) in result.port_mapping
        assert PortRef(b_id, Direction.OUTPUT, 0) in result.port_mapping
        assert PortRef(b_id, Direction.OUTPUT, 1) in result.port_mapping
        assert PortRef(b_id, Direction.INPUT, 0) not in result.port_mapping


class TestDanglingPortSurvives:
    """A leg that is neither wired to anything nor part of any boundary list still survives.

    The builder tracks "surviving" purely by leg identity -- every leg of a consumed node
    except the one the matched wire itself consumes (see :func:`_surviving_legs`) -- never
    by wiring or boundary status, so a genuinely dangling leg needs no special-casing, only
    a test locking the behavior in.
    """

    def test_unwired_non_boundary_leg_on_b_survives_onto_the_merged_node(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        # b_id's output leg is neither wired to a third node nor part of any boundary list.
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.num_inputs == 0
        assert merged.num_outputs == 1
        assert PortRef(b_id, Direction.OUTPUT, 0) in result.port_mapping


class TestLegOrderingConvention:
    def test_merged_inputs_are_a_survivors_then_b_survivors(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        # A: 2 inputs (indices 0, 1), 1 output (the fusion wire).
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[d])
        # B: 1 input (the fusion wire) and 2 inputs that survive (indices 1, 2).
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs(
            [
                PortRef(a_id, Direction.INPUT, 0),
                PortRef(a_id, Direction.INPUT, 1),
                PortRef(b_id, Direction.INPUT, 1),
                PortRef(b_id, Direction.INPUT, 2),
            ]
        )
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.num_inputs == 4
        assert result.port_mapping[PortRef(a_id, Direction.INPUT, 0)] == PortRef(
            result.new_node_ids[0], Direction.INPUT, 0
        )
        assert result.port_mapping[PortRef(a_id, Direction.INPUT, 1)] == PortRef(
            result.new_node_ids[0], Direction.INPUT, 1
        )
        assert result.port_mapping[PortRef(b_id, Direction.INPUT, 1)] == PortRef(
            result.new_node_ids[0], Direction.INPUT, 2
        )
        assert result.port_mapping[PortRef(b_id, Direction.INPUT, 2)] == PortRef(
            result.new_node_ids[0], Direction.INPUT, 3
        )


class TestPhaseAddition:
    def test_both_none_stays_none(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        assert result.diagram.nodes[result.new_node_ids[0]].phase is None

    def test_missing_phase_treated_as_zero(self) -> None:
        d = Dim.concrete(3)
        phase_b = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 3))})
        diagram, _a, _b = build_ghz_with_copy(d, phase_on_b=phase_b)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        merged_phase = result.diagram.nodes[result.new_node_ids[0]].phase
        assert merged_phase is not None
        assert merged_phase.get(1) == phase_b.get(1)

    def test_both_present_are_added(self) -> None:
        d = Dim.concrete(3)
        phase_a = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 3))})
        phase_b = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 5))})
        diagram, _a, _b = build_ghz_with_copy(d, phase_on_a=phase_a, phase_on_b=phase_b)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        merged_phase = result.diagram.nodes[result.new_node_ids[0]].phase
        assert merged_phase is not None
        assert merged_phase.get(1) == phase_a.get(1) + phase_b.get(1)


class TestAllLegsConsumedPhase:
    """The legless-merge corner case: dimension must survive via an explicit zero phase."""

    def test_legless_merge_gets_explicit_zero_phase(self) -> None:
        d = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.num_inputs == 0
        assert merged.num_outputs == 0
        assert merged.phase == PhaseVector(d, {})

    def test_surviving_legs_still_stay_none(self) -> None:
        # Same shape as TestPhaseAddition.test_both_none_stays_none, restated here so a
        # future refactor of _merged_phase's any_legs_survive branch cannot regress this
        # half while fixing the legless case above.
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = spider_fusion_builder(diagram, match)
        assert result.diagram.nodes[result.new_node_ids[0]].phase is None


class TestBuilderTypedErrors:
    def test_rejects_a_non_fusion_match(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        with pytest.raises(RewriteGrammarError):
            spider_fusion_builder(diagram, object())  # type: ignore[arg-type]

    def test_rejects_a_match_with_a_failed_side_condition(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        failing_outcomes = tuple(
            dataclasses.replace(outcome, passed=False) for outcome in match.side_condition_outcomes
        )
        broken_match = dataclasses.replace(match, side_condition_outcomes=failing_outcomes)
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, broken_match)

    def test_rejects_a_hand_built_match_with_disagreeing_phase_dims(self) -> None:
        # find_matches() itself would never produce this (phase_dimension_agreement
        # would reject it); this exercises the builder's own defensive check for a
        # match that reached it some other way, e.g. hand-constructed or from a future
        # pattern that forgot the check.
        d = Dim.symbol("d")
        phase_a = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 3))})
        phase_b = PhaseVector(Dim.concrete(3), {1: Phase.turns(sp.Rational(1, 5))})
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=phase_a)
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[], phase=phase_b)
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        match = FusionMatch(
            a_id=a_id,
            b_id=b_id,
            wire=wire,
            shared_dim=d,
            side_condition_outcomes=(
                SideConditionOutcome("phase_dimension_agreement", True, "hand-built, bypassing find_matches"),
            ),
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, match)


class TestSideConditionCoverageEnforced:
    """Fix 1: an empty (or incomplete) side_condition_outcomes tuple must not be accepted.

    Proof from the audit: a hand-built FusionMatch naming a Z spider wired into an X
    spider, with side_condition_outcomes=(), used to be accepted by spider_fusion_builder
    -- ``all(...)`` over an empty tuple is vacuously True, so the (never actually checked)
    same_generator_type condition never gets a chance to reject it. This builder is
    reachable directly (not only via qufzx.rewrite.engine.apply), so it must enforce
    coverage itself.
    """

    def test_empty_outcomes_on_a_z_into_x_pair_is_rejected(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        match = FusionMatch(
            a_id=a_id,
            b_id=b_id,
            wire=wire,
            shared_dim=d,
            side_condition_outcomes=(),
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, match)

    def test_outcomes_missing_one_declared_condition_is_rejected(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        incomplete_outcomes = match.side_condition_outcomes[:-1]
        broken_match = dataclasses.replace(match, side_condition_outcomes=incomplete_outcomes)
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, broken_match)

    def test_duplicate_outcome_names_are_rejected(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        duplicated_outcomes = match.side_condition_outcomes + (match.side_condition_outcomes[0],)
        broken_match = dataclasses.replace(match, side_condition_outcomes=duplicated_outcomes)
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, broken_match)


class TestRuleRegistry:
    """Fix 5: RewriteStep.rule_name must be resolvable back to a Rule via a registry."""

    def test_rules_contains_spider_fusion_by_name(self) -> None:
        assert RULES["spider_fusion"] is SPIDER_FUSION

    def test_lookup_rule_finds_spider_fusion(self) -> None:
        assert lookup_rule("spider_fusion") is SPIDER_FUSION

    def test_lookup_rule_raises_on_an_unknown_name(self) -> None:
        with pytest.raises(RewriteGrammarError):
            lookup_rule("no_such_rule")
