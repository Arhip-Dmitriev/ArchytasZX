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

"""Establishes qufzx.rewrite.engine.apply: non-mutation, remapping, scalar, and provenance."""

from __future__ import annotations

import dataclasses

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from qufzx.diagram.validate import IssueKind, ValidationIssue, validate
from qufzx.rewrite import engine as engine_module
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import (
    BuildResult,
    Match,
    Pattern,
    Quantifiers,
    RewriteDomainError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
)
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

    def test_raises_when_new_node_ids_names_a_node_never_added(self) -> None:
        """A builder that reports a ``new_node_ids`` entry it never actually created.

        Neither step 5 (which never reads ``new_node_ids``) nor the pre-fix step 8/9 would
        have caught this: step 9 would have published a phantom id into the ``RewriteStep``
        for Phase 6's certificate to choke on much later, far from the builder bug that
        produced it.
        """

        def _phantom_new_node_builder(working_diagram: Diagram, match_obj: object) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            return dataclasses.replace(
                result, new_node_ids=result.new_node_ids + (NodeId(999_999),)
            )

        phantom_rule = Rule(
            name="spider_fusion_phantom_new_node",
            pattern=SPIDER_FUSION.pattern,
            builder=_phantom_new_node_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        with pytest.raises(RewriteGrammarError, match="new_node_ids"):
            apply(diagram, phantom_rule, match)

    def test_raises_when_port_mapping_names_an_out_of_range_port(self) -> None:
        """A builder that reports a ``port_mapping`` value naming a nonexistent port.

        Undetected, step 5 would have fed this value straight into ``add_wire`` for every
        surviving wire or boundary entry that used to point at the corresponding consumed
        port -- either raising a confusing, unrelated error deep in wire remapping, or (if
        the bogus index happened to be in range for some other leg) silently splicing the
        wire onto the wrong port instead.
        """

        def _phantom_port_mapping_builder(working_diagram: Diagram, match_obj: object) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            bogus_mapping = dict(result.port_mapping)
            new_node_id = result.new_node_ids[0]
            some_old_ref = next(iter(bogus_mapping))
            # An index far out of range for the merged node's actual leg count.
            bogus_mapping[some_old_ref] = PortRef(new_node_id, Direction.OUTPUT, 999)
            return dataclasses.replace(result, port_mapping=bogus_mapping)

        phantom_rule = Rule(
            name="spider_fusion_phantom_port_mapping",
            pattern=SPIDER_FUSION.pattern,
            builder=_phantom_port_mapping_builder,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        with pytest.raises(RewriteGrammarError, match="port_mapping"):
            apply(diagram, phantom_rule, match)


class TestRewriteStepRecordsTheMatch:
    """Fix 6: RewriteStep must carry the located match Phase 6 replays from."""

    def test_step_match_is_the_applied_match(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert result.step.match == match


class TestStep8CatchesAnExtraIssueOfAnAlreadyPresentKind:
    """Defect 5 (Phase 5 audit): step 8's relative post-condition used to compare hard-error
    IssueKinds as a *set*. A set comparison cannot see a second, independent issue of a kind
    the input diagram already carried once (both collapse to the same set element), so a
    builder that left the input's pre-existing violation untouched but introduced a brand
    new, unrelated one of the *same* kind on a fresh node used to slip through undetected.
    The comparison must be a multiset keyed by (kind, offending ref) instead.
    """

    def test_a_second_dimension_policy_violation_on_a_new_node_is_caught(self) -> None:
        def _builder_with_a_second_violation(working_diagram: Diagram, match_obj: object) -> object:
            result = spider_fusion_builder(working_diagram, match_obj)  # type: ignore[arg-type]
            extra_id = working_diagram.add_node(
                Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2), Dim.concrete(3)]
            )
            working_diagram.set_boundary_outputs(
                working_diagram.boundary_outputs
                + (
                    PortRef(extra_id, Direction.OUTPUT, 0),
                    PortRef(extra_id, Direction.OUTPUT, 1),
                )
            )
            return result

        rule_with_extra_violation = Rule(
            name="spider_fusion_extra_violation",
            pattern=SPIDER_FUSION.pattern,
            builder=_builder_with_a_second_violation,  # type: ignore[arg-type]
            side_conditions=SPIDER_FUSION.side_conditions,
            quantifiers=SPIDER_FUSION.quantifiers,
            scalar_introduced=SPIDER_FUSION.scalar_introduced,
        )

        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        # A pre-existing DIMENSION_POLICY_VIOLATION, unrelated to the fused pair, that the
        # rewrite carries over untouched -- a set-of-kinds comparison already tolerates this
        # part correctly on its own.
        c_id = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2), Dim.concrete(3)]
        )
        diagram.set_boundary_outputs(
            [PortRef(c_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.OUTPUT, 1)]
        )

        match = find_matches(diagram)[0]
        with pytest.raises(RewriteDomainError, match="dimension_policy_violation"):
            apply(diagram, rule_with_extra_violation, match)


class TestStep8DoesNotBlockAPreExistingIssueOnAConsumedNode:
    """Defect 1 (Phase 5 round-7 audit): step 8 compared a pre-existing hard-error issue's
    key in its *input*-diagram coordinates against post-rewrite keys, which are always in
    *post*-rewrite coordinates. A consumed node's ports and node id do not survive a
    rewrite -- the merged node gets a fresh id and fresh port indices -- so an issue
    anchored on one of them (e.g. an unwired, non-boundary leg) always looked "introduced"
    even when the rewrite carried it over faithfully, wrongly raising
    ``RewriteDomainError`` for a rewrite that introduced nothing. See the module docstring,
    step 8, and :func:`~qufzx.rewrite.engine._translate_input_issue_key`.
    """

    def test_port_unused_on_a_consumed_nodes_surviving_leg_does_not_block_the_rewrite(
        self,
    ) -> None:
        # The exact reproduction from the audit: A.in0 is left neither wired nor on a
        # boundary (a pre-existing PORT_UNUSED), and is not the leg the fusion consumes
        # (that's A.out0) -- it survives onto the merged node.
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[two])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[two])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
        # A.in0 is deliberately left off every boundary and every wire.

        pre_report = validate(diagram)
        assert IssueKind.PORT_UNUSED in {issue.kind for issue in pre_report.errors}

        matches = find_matches(diagram)
        assert len(matches) == 1

        result = apply(diagram, SPIDER_FUSION, matches[0])

        post_errors = validate(result.diagram).errors
        assert any(issue.kind is IssueKind.PORT_UNUSED for issue in post_errors), (
            "the pre-existing PORT_UNUSED issue should carry over onto the merged node's "
            "corresponding surviving port, not vanish or (as under the defect) block apply()"
        )


class TestTranslateInputIssueKeyMapsConsumedNodeReferences:
    """Direct unit coverage of :func:`~qufzx.rewrite.engine._translate_input_issue_key`.

    A hard-error issue's ``node_id`` or ``port_ref`` anchor cannot, by construction,
    survive a fusion on the very node it names without :meth:`Dim.unify` also failing
    somewhere the matcher checks (see the DIMENSION_POLICY_VIOLATION / PHASE_DIMENSION_
    MISMATCH cases: any real conflict the merge's leg-dim normalization would erase, the
    matcher's own surviving-leg or phase check already refuses to match). Both branches of
    the translation are therefore exercised directly here, independent of what
    ``find_matches`` can organically produce end-to-end -- this is the same class covered,
    end-to-end, by ``TestStep8DoesNotBlockAPreExistingIssueOnAConsumedNode`` above for the
    ``port_ref``-in-``port_mapping`` case.
    """

    def test_node_id_on_a_consumed_node_maps_to_the_sole_new_node_id(self) -> None:
        old_id = NodeId(7)
        new_id = NodeId(99)
        issue = ValidationIssue(
            kind=IssueKind.DIMENSION_POLICY_VIOLATION, message="x", node_id=old_id
        )
        key = engine_module._translate_input_issue_key(
            issue,
            consumed_node_ids=frozenset({old_id, NodeId(8)}),
            port_mapping={},
            new_node_ids=(new_id,),
        )
        assert key == (IssueKind.DIMENSION_POLICY_VIOLATION, new_id)

    def test_node_id_on_a_consumed_node_is_left_unmapped_when_ambiguous(self) -> None:
        # A future rule that consumes N nodes into M != 1 new ones has no principled way
        # to say which new node a consumed node's identity maps to -- see the function's
        # docstring for why this is deliberately fail-closed rather than guessed at.
        old_id = NodeId(7)
        issue = ValidationIssue(
            kind=IssueKind.DIMENSION_POLICY_VIOLATION, message="x", node_id=old_id
        )
        key = engine_module._translate_input_issue_key(
            issue,
            consumed_node_ids=frozenset({old_id}),
            port_mapping={},
            new_node_ids=(),
        )
        assert key == (IssueKind.DIMENSION_POLICY_VIOLATION, old_id)

    def test_port_ref_on_a_consumed_node_maps_through_port_mapping(self) -> None:
        old_ref = PortRef(NodeId(7), Direction.INPUT, 0)
        new_ref = PortRef(NodeId(99), Direction.INPUT, 3)
        issue = ValidationIssue(
            kind=IssueKind.DIMENSION_POLICY_VIOLATION, message="x", port_ref=old_ref
        )
        key = engine_module._translate_input_issue_key(
            issue,
            consumed_node_ids=frozenset({NodeId(7)}),
            port_mapping={old_ref: new_ref},
            new_node_ids=(NodeId(99),),
        )
        assert key == (IssueKind.DIMENSION_POLICY_VIOLATION, new_ref)

    def test_port_ref_on_a_consumed_node_absent_from_port_mapping_is_left_unchanged(
        self,
    ) -> None:
        # The consumed port itself (the matched wire's own endpoint) is never in
        # port_mapping -- a builder only maps surviving ports. Left untranslated, this
        # correctly matches nothing in the post-rewrite diagram's own issue keys, since
        # that port no longer exists at all.
        old_ref = PortRef(NodeId(7), Direction.OUTPUT, 0)
        issue = ValidationIssue(
            kind=IssueKind.DIMENSION_POLICY_VIOLATION, message="x", port_ref=old_ref
        )
        key = engine_module._translate_input_issue_key(
            issue,
            consumed_node_ids=frozenset({NodeId(7)}),
            port_mapping={},
            new_node_ids=(NodeId(99),),
        )
        assert key == (IssueKind.DIMENSION_POLICY_VIOLATION, old_ref)

    def test_reference_on_a_surviving_node_passes_through_unchanged(self) -> None:
        surviving_id = NodeId(3)
        issue = ValidationIssue(
            kind=IssueKind.DIMENSION_POLICY_VIOLATION, message="x", node_id=surviving_id
        )
        key = engine_module._translate_input_issue_key(
            issue,
            consumed_node_ids=frozenset({NodeId(7), NodeId(8)}),
            port_mapping={},
            new_node_ids=(NodeId(99),),
        )
        assert key == (IssueKind.DIMENSION_POLICY_VIOLATION, surviving_id)


class TestRewriteStepIsHashable:
    """Task 1 (Phase 5 closing round): the explicit ``__hash__`` this class already declared
    (to work around ``port_mapping`` being an unhashable ``MappingProxyType``) hashed
    ``self.match`` verbatim -- and every ``FusionMatch`` was itself unhashable for the same
    reason (its own ``bindings`` field), so ``hash(step)`` raised for every step ``apply``
    ever produced, including the empty-``bindings`` default. See ``test_match.py``'s
    ``TestFusionMatchIsHashable`` for the root-cause fix this depends on.
    """

    def test_hash_succeeds_with_empty_bindings(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert match.bindings == {}
        hash(result.step)  # must not raise

    def test_hash_succeeds_with_non_empty_bindings(self) -> None:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.symbol("d")])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert dict(match.bindings) == {"d": Dim.concrete(2)}
        hash(result.step)  # must not raise

    def test_equal_steps_hash_equal_and_are_usable_as_dict_keys(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        step_one = apply(diagram, SPIDER_FUSION, match).step
        step_two = apply(diagram, SPIDER_FUSION, match).step
        assert step_one == step_two
        assert hash(step_one) == hash(step_two)
        assert {step_one, step_two} == {step_one}
        assert {step_one: "a value"}[step_two] == "a value"


class _EmptyPattern(Pattern):
    """A ``Pattern`` that finds nothing; the tests below drive ``apply()`` directly instead
    of going through ``find_matches``."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        return ()


@dataclasses.dataclass(frozen=True, slots=True)
class _ScriptedMatch:
    """A minimal, hand-built ``Match`` for exercising ``apply()`` with an independent builder."""

    side_condition_outcomes: tuple[SideConditionOutcome, ...] = ()
    dimension_constraints: tuple[tuple[Dim, Dim], ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        return True


class TestApplyWithAnIndependentlyScriptedBuilder:
    """Every other test in this file drives ``apply()`` through ``spider_fusion_builder`` with
    one field sabotaged. These exercise the generic splice path with a builder written from
    scratch against the ``BuildResult`` contract alone.
    """

    def test_consumed_wire_between_surviving_nodes_is_dropped(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        s1_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        s2_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[])
        w = Wire(PortRef(s1_id, Direction.OUTPUT, 0), PortRef(s2_id, Direction.INPUT, 0))
        diagram.add_wire(w.a, w.b)
        diagram.set_boundary_outputs([PortRef(s1_id, Direction.OUTPUT, 0)])
        diagram.set_boundary_inputs([PortRef(s2_id, Direction.INPUT, 0)])

        # Both endpoints of w are deliberately also on a boundary.
        pre_errors = validate(diagram).errors
        assert len(pre_errors) == 2
        assert all(issue.kind is IssueKind.PORT_WIRED_AND_BOUNDARY for issue in pre_errors)

        def _builder(working: Diagram, match: Match) -> BuildResult:
            r_id = working.add_node(Z_SPIDER, input_dims=[], output_dims=[])
            return BuildResult(
                diagram=working,
                new_node_ids=(r_id,),
                consumed_node_ids=(c_id,),
                consumed_wires=(w,),
                port_mapping={},
                scalar_introduced=Scalar.one(),
            )

        rule = Rule(
            name="scripted_drop_consumed_wire",
            pattern=_EmptyPattern(),
            builder=_builder,
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )

        result = apply(diagram, rule, _ScriptedMatch())
        post = result.diagram

        assert w not in post.wires
        assert c_id not in post.nodes
        assert result.new_node_ids[0] in post.nodes
        assert post.boundary_outputs == (PortRef(s1_id, Direction.OUTPUT, 0),)
        assert post.boundary_inputs == (PortRef(s2_id, Direction.INPUT, 0),)
        assert validate(post).errors == ()
        assert result.step.consumed_wires == (w,)

    def test_collapsing_port_mapping_raises_rewrite_grammar_error(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        s_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(s_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))

        def _builder(working: Diagram, match: Match) -> BuildResult:
            collapsing_mapping = {
                PortRef(c_id, Direction.INPUT, 0): PortRef(s_id, Direction.OUTPUT, 0),
            }
            return BuildResult(
                diagram=working,
                new_node_ids=(),
                consumed_node_ids=(c_id,),
                consumed_wires=(),
                port_mapping=collapsing_mapping,
                scalar_introduced=Scalar.one(),
            )

        rule = Rule(
            name="scripted_collapsing_port_mapping",
            pattern=_EmptyPattern(),
            builder=_builder,
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )

        with pytest.raises(RewriteGrammarError, match="collapses"):
            apply(diagram, rule, _ScriptedMatch())
