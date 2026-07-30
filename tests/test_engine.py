"""Establishes qufzx.rewrite.engine.apply: non-mutation, remapping, scalar, and provenance."""

from __future__ import annotations

import dataclasses

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import RewriteDomainError, RewriteGrammarError, Rule
from qufzx.rewrite.rules_library import SPIDER_FUSION, spider_fusion_builder

from .helpers import build_ghz_with_copy


class TestApplyNeverMutatesTheInput:
    def test_original_diagram_is_unchanged(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        nodes_before = dict(diagram.nodes)
        wires_before = diagram.wires
        boundary_in_before = diagram.boundary_inputs
        boundary_out_before = diagram.boundary_outputs
        scalar_before = diagram.scalar

        match = find_matches(diagram)[0]
        apply(diagram, SPIDER_FUSION, match)

        assert dict(diagram.nodes) == nodes_before
        assert diagram.wires == wires_before
        assert diagram.boundary_inputs == boundary_in_before
        assert diagram.boundary_outputs == boundary_out_before
        assert diagram.scalar == scalar_before
        assert a_id in diagram.nodes
        assert b_id in diagram.nodes


class TestApplyRemapsBoundaryOrder:
    def test_boundary_output_order_preserved(self) -> None:
        d = Dim.concrete(2)
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        new_id = result.new_node_ids[0]
        assert result.diagram.boundary_outputs == (
            PortRef(new_id, Direction.OUTPUT, 0),
            PortRef(new_id, Direction.OUTPUT, 1),
            PortRef(new_id, Direction.OUTPUT, 2),
        )
        assert result.diagram.boundary_inputs == ()


class TestApplyRemapsThirdNodeWiring:
    def test_wire_to_a_third_node_survives_remapped(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        post = result.diagram
        new_id = result.new_node_ids[0]

        assert c_id in post.nodes
        assert a_id not in post.nodes
        assert b_id not in post.nodes

        b_output_new_ref = result.step.port_mapping[PortRef(b_id, Direction.OUTPUT, 0)]
        assert b_output_new_ref.node_id == new_id
        expected_wire_endpoints = frozenset((b_output_new_ref, PortRef(c_id, Direction.INPUT, 0)))
        actual_endpoints = {wire.endpoints() for wire in post.wires}
        assert expected_wire_endpoints in actual_endpoints


class TestApplyRejectsAnUnmappedSurvivingPort:
    def test_raises_when_builder_omits_a_surviving_port_a_wire_references(self) -> None:
        def _drop_b_output_mapping_builder(working_diagram: Diagram, match_obj: object) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            b_output_ref = PortRef(match_obj.b_id, Direction.OUTPUT, 0)  # type: ignore[attr-defined]
            incomplete_mapping = {
                k: v for k, v in result.port_mapping.items() if k != b_output_ref
            }
            return dataclasses.replace(result, port_mapping=incomplete_mapping)

        incomplete_rule = Rule(
            name="spider_fusion_incomplete",
            pattern=SPIDER_FUSION.pattern,
            builder=_drop_b_output_mapping_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])

        match = find_matches(diagram)[0]
        with pytest.raises(RewriteDomainError):
            apply(diagram, incomplete_rule, match)


class TestApplyRejectsAnUnmappedSurvivingBoundaryPort:
    """Mirrors ``TestApplyRejectsAnUnmappedSurvivingPort`` for the boundary rebuild.

    Before the fix, step 4's boundary rebuild used the silent ``port_mapping.get(ref, ref)``
    fallback unconditionally, so an unmapped ref on a consumed node would survive the
    rebuild unchanged and then be silently deleted by step 5's ``remove_node`` cascade --
    shrinking the returned diagram's boundary arity with no exception. These two tests
    (output and input) are the direct boundary-side mirror of
    ``test_raises_when_builder_omits_a_surviving_port_a_wire_references`` above.
    """

    def test_raises_when_builder_omits_a_surviving_port_a_boundary_output_references(
        self,
    ) -> None:
        def _drop_a_output_1_mapping_builder(
            working_diagram: Diagram, match_obj: object
        ) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            a_output_1_ref = PortRef(match_obj.a_id, Direction.OUTPUT, 1)  # type: ignore[attr-defined]
            incomplete_mapping = {
                k: v for k, v in result.port_mapping.items() if k != a_output_1_ref
            }
            return dataclasses.replace(result, port_mapping=incomplete_mapping)

        incomplete_rule = Rule(
            name="spider_fusion_incomplete_boundary_output",
            pattern=SPIDER_FUSION.pattern,
            builder=_drop_a_output_1_mapping_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.concrete(2)
        diagram, a_id, _b_id = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]

        # A's output 1 is referenced only by the boundary output list, not by any wire --
        # this is the case the pre-fix silent fallback let through unnoticed.
        assert PortRef(a_id, Direction.OUTPUT, 1) in diagram.boundary_outputs

        with pytest.raises(RewriteDomainError):
            apply(diagram, incomplete_rule, match)

    def test_raises_when_builder_omits_a_surviving_port_a_boundary_input_references(
        self,
    ) -> None:
        def _drop_a_input_0_mapping_builder(
            working_diagram: Diagram, match_obj: object
        ) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            a_input_0_ref = PortRef(match_obj.a_id, Direction.INPUT, 0)  # type: ignore[attr-defined]
            incomplete_mapping = {
                k: v for k, v in result.port_mapping.items() if k != a_input_0_ref
            }
            return dataclasses.replace(result, port_mapping=incomplete_mapping)

        incomplete_rule = Rule(
            name="spider_fusion_incomplete_boundary_input",
            pattern=SPIDER_FUSION.pattern,
            builder=_drop_a_input_0_mapping_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])

        match = find_matches(diagram)[0]
        with pytest.raises(RewriteDomainError):
            apply(diagram, incomplete_rule, match)


class TestApplyMultipliesScalar:
    def test_scalar_multiplied_by_the_introduced_factor(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        prior_factor = Scalar.symbol("s")
        diagram.multiply_scalar(prior_factor)

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert result.diagram.scalar == prior_factor * Scalar.one()


class TestRewriteStepProvenance:
    def test_step_records_rule_and_match_location(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        step = result.step

        assert step.rule_name == "spider_fusion"
        assert step.matched_node_ids == (min(a_id, b_id), max(a_id, b_id))
        assert step.consumed_wires == (match.wire,)
        assert step.scalar_introduced == Scalar.one()
        assert step.new_node_ids == result.new_node_ids
        assert step.side_condition_outcomes == match.side_condition_outcomes
        assert step.dimension_constraints == match.dimension_constraints
        assert len(step.port_mapping) == 3


class TestApplyRejectsAFailingMatch:
    def test_raises_before_calling_the_builder(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        failing_outcomes = tuple(
            dataclasses.replace(outcome, passed=False) for outcome in match.side_condition_outcomes
        )
        broken_match = dataclasses.replace(match, side_condition_outcomes=failing_outcomes)
        with pytest.raises(RewriteDomainError):
            apply(diagram, SPIDER_FUSION, broken_match)


class TestApplyRejectsAScalarMismatch:
    def test_raises_when_builder_scalar_disagrees_with_the_rules_declared_scalar(self) -> None:
        def _wrong_scalar_builder(working_diagram: Diagram, match_obj: object) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            return dataclasses.replace(result, scalar_introduced=Scalar.rational(2))

        mismatched_rule = Rule(
            name="spider_fusion_mismatched",
            pattern=SPIDER_FUSION.pattern,
            builder=_wrong_scalar_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=Scalar.one(),
        )
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        with pytest.raises(RewriteDomainError):
            apply(diagram, mismatched_rule, match)


class TestApplyEnforcesSideConditionCoverage:
    """Fix 1's audit proof, exercised through apply(): an empty outcomes tuple must not
    silently bypass the side-condition invariant.

    Before the fix, a hand-built FusionMatch(side_condition_outcomes=()) naming a Z
    spider wired into an X spider was accepted by apply(): it merged the two into one Z
    node, produced a diagram validate() called well-formed, and oracle-compared against
    the input with a nonzero deviation -- while the emitted RewriteStep recorded zero
    side conditions.
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
            apply(diagram, SPIDER_FUSION, match)


class TestApplyRejectsAForeignMatch:
    """Fix 4: apply() must verify the match actually belongs to the diagram it is applied to."""

    def test_raises_when_the_matched_wire_is_absent_from_the_diagram(self) -> None:
        d = Dim.concrete(2)
        diagram, a_id, b_id = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]

        # Same node shapes and ids, but the fusion wire itself was never added -- an
        # already-invalid, dangling-port diagram the match does not actually belong to.
        foreign_diagram = Diagram()
        foreign_diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        foreign_diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
        assert a_id in foreign_diagram.nodes
        assert b_id in foreign_diagram.nodes

        with pytest.raises(RewriteGrammarError):
            apply(foreign_diagram, SPIDER_FUSION, match)

    def test_raises_when_the_build_result_names_a_missing_node_id(self) -> None:
        def _phantom_node_builder(working_diagram: Diagram, match_obj: object) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            return dataclasses.replace(
                result, consumed_node_ids=result.consumed_node_ids + (NodeId(999_999),)
            )

        phantom_rule = Rule(
            name="spider_fusion_phantom",
            pattern=SPIDER_FUSION.pattern,
            builder=_phantom_node_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        with pytest.raises(RewriteGrammarError):
            apply(diagram, phantom_rule, match)


class TestRewriteStepRecordsTheMatch:
    """Fix 6: RewriteStep must carry the located match Phase 6 replays from."""

    def test_step_match_is_the_applied_match(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert result.step.match == match
