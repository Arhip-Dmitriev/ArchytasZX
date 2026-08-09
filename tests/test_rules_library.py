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
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import (
    ConstraintOutcome,
    ConstraintSource,
    DimensionConstraint,
    RewriteDomainError,
    RewriteGrammarError,
    SideConditionOutcome,
)
from qufzx.rewrite.rules_library import RULES, SPIDER_FUSION, lookup_rule, spider_fusion_builder
from qufzx.semantics.check import compare

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


class TestPhaseDimensionAgreementJudgementCall2:
    """Judgement call 2 (Phase 5 post-closing audit), decided in
    :mod:`qufzx.rewrite.match`'s ``phase_dimension_agreement``: a fusion whose two phase
    dims unify only by a *concrete* binding is now matched (was previously silently
    refused), and two present phases no longer need to agree raw before any resolution
    (``reattach_phase`` already forces both onto the identical ``shared_dim``). Both are
    exercised here all the way through the builder and the numeric oracle, not merely
    matched.
    """

    def test_a_phase_binding_the_symbolic_shared_leg_dim_builds_and_matches_the_oracle(
        self,
    ) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[d],
            output_dims=[],
            phase=PhaseVector(Dim.concrete(3), {1: Phase.turns(sp.Rational(1, 3))}),
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

        result = apply(diagram, SPIDER_FUSION, matches[0])
        report = compare(diagram, result.diagram, {"d": 3})
        assert report.matched, report.reason

    def test_two_present_phases_agreeing_only_after_substitution_build_and_match_the_oracle(
        self,
    ) -> None:
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[d],
            phase=PhaseVector(d, {1: Phase.turns(sp.Rational(1, 3))}),
        )
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[three],
            output_dims=[],
            phase=PhaseVector(three, {1: Phase.turns(sp.Rational(1, 5))}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        matches = find_matches(diagram)
        assert len(matches) == 1

        result = apply(diagram, SPIDER_FUSION, matches[0])
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.phase is not None
        assert merged.phase.dim == three
        report = compare(diagram, result.diagram, {"d": 3})
        assert report.matched, report.reason


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
                SideConditionOutcome(
                    "phase_dimension_agreement", True, "hand-built, bypassing find_matches"
                ),
            ),
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, match)


def _fabricated_passing_outcomes() -> tuple[SideConditionOutcome, ...]:
    """A complete, all-``passed=True`` outcome tuple naming exactly every declared condition.

    Used to construct a foreign/hand-built ``FusionMatch`` whose ``side_condition_outcomes``
    lies about every condition having actually been checked -- so ``check_side_condition_coverage``
    (which only compares outcome *names* and passedness, never re-evaluates a predicate) lets
    it through, and the builder's own fresh re-verification (:func:`resolve_fusion_match`,
    Phase 5 round-12 audit) is what has to catch the lie.
    """
    return tuple(
        SideConditionOutcome(name, True, "fabricated: claims to pass without being checked")
        for name in (
            "distinct_nodes",
            "same_generator_type",
            "parallel_wires_become_self_loops",
            "consumed_wire_direction_permitted_for_color",
            "dimension_agreement",
            "phase_dimension_agreement",
        )
    )


class TestPhase5Round12AuditDefects:
    """Regression coverage for A1/A2 (Phase 5 round-12 audit): the builder must not trust a
    match's own claimed ``generator_type`` agreement or ``shared_dim``/``bindings`` -- it must
    re-derive them fresh (via :func:`~qufzx.rewrite.match.resolve_fusion_match`) and reject any
    disagreement, since a fabricated-passing ``side_condition_outcomes`` tuple alone (which
    ``check_side_condition_coverage`` cannot catch -- it never re-evaluates a predicate) would
    otherwise let a Z/X pair, or a nonsensical ``shared_dim``, through to graph surgery.
    """

    def test_a1_fabricated_same_generator_type_outcome_on_a_z_x_pair_is_rejected(self) -> None:
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
            side_condition_outcomes=_fabricated_passing_outcomes(),
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, match)

    def test_a2_shared_dim_unrelated_to_the_matched_legs_is_rejected(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        match = FusionMatch(
            a_id=a_id,
            b_id=b_id,
            wire=wire,
            shared_dim=Dim.concrete(7),
            side_condition_outcomes=_fabricated_passing_outcomes(),
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, match)

    def test_a2_bindings_unrelated_to_the_matched_legs_is_rejected(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        wire = Wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(wire.a, wire.b)
        match = FusionMatch(
            a_id=a_id,
            b_id=b_id,
            wire=wire,
            shared_dim=Dim.concrete(2),
            side_condition_outcomes=_fabricated_passing_outcomes(),
            bindings={"d": Dim.concrete(999)},
        )
        with pytest.raises(RewriteDomainError):
            spider_fusion_builder(diagram, match)


class TestGhostWireIsRejected:
    """Defect 1 (Phase 5 post-closing audit): a match naming a wire that is structurally
    incident on both nodes but was never actually added to the diagram used to be accepted
    by ``resolve_fusion_match`` -- so the builder built a merged node consuming a wire the
    diagram never had. Fixed in ``resolve_fusion_match`` itself; this proves the fix closes
    the path all the way through the builder.
    """

    def test_builder_refuses_a_match_over_a_wire_absent_from_the_diagram(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, [d], [d])
        b_id = diagram.add_node(Z_SPIDER, [d], [d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])

        ghost = Wire(PortRef(a_id, Direction.INPUT, 0), PortRef(b_id, Direction.OUTPUT, 0))
        assert ghost not in diagram.wires
        match = FusionMatch(
            a_id=a_id,
            b_id=b_id,
            wire=ghost,
            shared_dim=d,
            side_condition_outcomes=_fabricated_passing_outcomes(),
        )
        with pytest.raises(RewriteGrammarError):
            spider_fusion_builder(diagram.copy(), match)


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


class TestPhaseOnTheUnboundSideOfALegUnifyBinding:
    """Defect 1 reproducer: a phase stated over the still-symbolic side of a bound leg
    must not crash the builder. Before the fix, the absent phase was synthesized at the
    resolved ``shared_dim`` while the present phase kept its own raw, unbound ``Dim``, so
    ``PhaseVector.__add__`` raised ``PhaseDomainError`` -- a phase on one side only, over
    the still-unbound symbol, passed the matcher and raised in the builder.
    """

    def test_phase_on_the_symbolic_side_no_longer_raises(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[d, d],
            phase=PhaseVector(d, {1: Phase.turns(sp.Rational(1, 3))}),
        )
        b_id = diagram.add_node(
            Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[Dim.concrete(3)]
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs(
            [PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.OUTPUT, 0)]
        )
        match = find_matches(diagram)[0]
        three = Dim.concrete(3)
        assert match.shared_dim == three

        result = apply(diagram, SPIDER_FUSION, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert all(port.dim == three for port in (*merged.inputs, *merged.outputs))
        assert merged.phase is not None
        assert merged.phase.dim == three

    def test_the_mirror_orientation_phase_on_the_already_resolved_side_still_works(self) -> None:
        # The pre-fix code already handled this orientation (phase on the side the
        # binding resolves *to*); restated here so a future change to _over_shared_dim
        # cannot regress this half while touching the other.
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(3)],
            output_dims=[Dim.concrete(3)],
            phase=PhaseVector(Dim.concrete(3), {1: Phase.turns(sp.Rational(1, 5))}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs(
            [PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.OUTPUT, 0)]
        )
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        three = Dim.concrete(3)
        assert all(port.dim == three for port in (*merged.inputs, *merged.outputs))
        assert merged.phase is not None
        assert merged.phase.dim == three


class TestSharedDimPropagatesToEverySurvivingPort:
    """Defect 2 reproducer: a resolved ``shared_dim`` used to be computed and then
    discarded -- the merged node's ports kept each leg's own raw ``Dim``, so a fusion
    whose match resolved ``d := 3`` could still leave a surviving leg carrying the
    unbound symbol ``d``, and the same symbol could be bound to contradictory values
    across independent fusion steps with no error anywhere.
    """

    def test_a_surviving_leg_with_a_different_raw_dim_is_forced_to_shared_dim(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
        match = find_matches(diagram)[0]
        assert match.shared_dim == Dim.concrete(3)

        result = apply(diagram, SPIDER_FUSION, match)
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.outputs[0].dim == Dim.concrete(3)

    def test_contradictory_chain_does_not_silently_fuse_twice(self) -> None:
        """A(3) -- B(d) -- C(7): fusing A into B commits ``d := 3`` and propagates that
        onto B's other, surviving leg -- which is wired to C's dim 7, an unresolvable
        concrete mismatch. Before the fix, B's surviving leg kept its own raw, never-
        forced ``d``, so a second, independent fusion could just as freely bind
        ``d := 7``, silently producing one legless node whose ``PhaseVector`` claimed
        dimension 7 with nothing left in the diagram to say it also assumed dimension 3,
        and a validate() report that came back clean. With the dimension propagated,
        apply()'s relative post-condition (see qufzx.rewrite.engine) catches the
        resulting conflict against the still-unfused third node immediately.
        """
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(3)])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(7)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))

        matches = find_matches(diagram)
        assert len(matches) == 2  # each wire looks independently fusable in isolation

        with pytest.raises(RewriteDomainError):
            apply(diagram, SPIDER_FUSION, matches[0])


class TestRuleRegistry:
    """Fix 5: RewriteStep.rule_name must be resolvable back to a Rule via a registry."""

    def test_rules_contains_spider_fusion_by_name(self) -> None:
        assert RULES["spider_fusion"] is SPIDER_FUSION

    def test_lookup_rule_finds_spider_fusion(self) -> None:
        assert lookup_rule("spider_fusion") is SPIDER_FUSION

    def test_lookup_rule_raises_on_an_unknown_name(self) -> None:
        with pytest.raises(RewriteGrammarError):
            lookup_rule("no_such_rule")
