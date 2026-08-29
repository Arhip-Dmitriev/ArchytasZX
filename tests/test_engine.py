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

import ast
import dataclasses
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from qufzx.diagram.validate import IssueKind, ValidationIssue, validate
from qufzx.rewrite import engine as engine_module
from qufzx.rewrite import match as match_module
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionMatch, find_matches
from qufzx.rewrite.rule import (
    BuildResult,
    ConstraintOutcome,
    ConstraintSource,
    DimensionConstraint,
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
        assert step.consumed_node_ids == (min(a_id, b_id), max(a_id, b_id))
        assert step.consumed_wires == (match.wire,)
        assert step.scalar_introduced == Scalar.one()
        assert step.new_node_ids == result.new_node_ids
        assert step.side_condition_outcomes == match.side_condition_outcomes
        assert step.dimension_constraints == match.dimension_constraints
        assert len(step.port_mapping) == 3


class TestCertificateRecordsTheReDerivedFacts:
    """Defect 2 (Phase 5 post-closing audit): the certificate must record what
    ``resolve_fusion_match`` independently re-derives, not a match's own unaudited claim --
    ``spider_fusion_builder`` computes the real ``FusionResolution`` as a side effect of its
    own re-verification (Phase 5 round-12 audit, A1/A2) but, before this fix, had no
    ``BuildResult`` channel to return it, so ``apply`` fell back to recording ``match``'s own
    ``side_condition_outcomes``/``dimension_constraints`` verbatim -- fields a foreign or
    hand-built match can fabricate.
    """

    def test_fabricated_dimension_constraints_and_outcomes_are_rejected_not_silently_recorded(
        self,
    ) -> None:
        d = Dim.symbol("d")
        three = Dim.concrete(3)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[three], output_dims=[three])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])

        match = find_matches(diagram)[0]
        # The real fusion assumes d := 3 exactly once, for the connecting pair. A's
        # surviving input is also stated over d, but ``_unify_surviving_legs`` resolves a
        # leg through the running ``bindings`` accumulator before unifying it, so that leg
        # arrives as concrete 3 against a shared_dim of 3 -- a bare syntactic identity,
        # which records nothing. Re-asserting an identical (d, 3) pair there would be the
        # same fact twice, not a second assumption the certificate must carry.
        assert match.dimension_constraints == (
            DimensionConstraint(
                assumed=d,
                equal_to=three,
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.BOUND,
                bound_here=(("d", three),),
            ),
        )

        fake = dataclasses.replace(
            match,
            dimension_constraints=(),
            side_condition_outcomes=tuple(
                SideConditionOutcome(o.name, True, "fabricated", False)
                for o in match.side_condition_outcomes
            ),
        )
        # Before the fix: apply() recorded fake's claim of "no dimension assumption"
        # verbatim onto the certificate. Now: spider_fusion_builder catches the
        # disagreement against its own fresh resolve_fusion_match derivation and refuses
        # to build at all, rather than let a laundered certificate through.
        with pytest.raises(RewriteDomainError):
            apply(diagram, SPIDER_FUSION, fake)

    def test_every_find_matches_result_has_a_step_matching_resolve_fusion_match_fresh(
        self,
    ) -> None:
        """Positive arm: for every match ``find_matches`` returns, the recorded
        ``step.dimension_constraints``/``side_condition_outcomes`` equal exactly what
        ``resolve_fusion_match`` derives fresh for that wire.
        """
        from qufzx.rewrite.match import resolve_fusion_match

        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        for match in find_matches(diagram):
            resolution = resolve_fusion_match(diagram, match.a_id, match.b_id, match.wire)
            result = apply(diagram, SPIDER_FUSION, match)
            assert result.step.dimension_constraints == resolution.dimension_constraints
            assert result.step.side_condition_outcomes == resolution.outcomes


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


class TestStep8CarriesANodeDimensionUndeterminedIssueOnAThirdUntouchedNode:
    """Round 20, Task 9: NODE_DIMENSION_UNDETERMINED is anchored on ``node_id``, with no
    ``port_ref``/``wire`` -- the same reference shape ``_translate_input_issue_key`` already
    routes through its single-``new_node_ids``-entry fallback for any other node-id-anchored
    kind (see that function's docstring). This pins that routing for the new kind
    specifically: a diagram already carrying this issue on a node the fusion match neither
    consumes nor touches must have step 8 carry it across (translated key equals its own
    untranslated key, since the node id is not in ``consumed_node_ids``), not misreport it as
    "introduced" by the rewrite.
    """

    def test_pre_existing_issue_on_an_untouched_third_node_survives_the_rewrite(self) -> None:
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[two])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[two])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
        # third_id is deliberately untouched by the a_id/b_id fusion match: no legs, no
        # phase, neither consumed nor a new_node_ids entry.
        third_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[])

        pre_report = validate(diagram)
        assert any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED and issue.node_id == third_id
            for issue in pre_report.errors
        )

        matches = find_matches(diagram)
        assert len(matches) == 1
        assert third_id not in (matches[0].a_id, matches[0].b_id)

        result = apply(diagram, SPIDER_FUSION, matches[0])

        assert third_id in result.diagram.nodes
        post_errors = validate(result.diagram).errors
        assert any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED and issue.node_id == third_id
            for issue in post_errors
        ), (
            "the pre-existing NODE_DIMENSION_UNDETERMINED issue on the untouched third node "
            "must carry over unchanged, not be misreported as introduced by the rewrite"
        )


class TestRemovedDeferredIssuesAreRecorded:
    """Judgement call 1 (Phase 5 post-closing audit): fusion is allowed to fire across a
    ``DEFERRED`` dimension pair (see rules_library.py's module docstring, "Phase 5
    judgement call"), but a rewrite that makes the resulting diagram-level
    ``DIMENSION_DEFERRED`` finding disappear entirely must record that fact on the
    certificate, not merely leave it inferable from ``dimension_constraints`` -- step 8's
    own hard-error compare never looks at deferred issues at all, by design, so nothing
    else in ``apply`` would otherwise ever mention this.
    """

    def test_a_consumed_deferred_leg_pair_is_recorded_as_removed(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        # a_id's own two legs (d, d*e) defer against each other under ALL_LEGS_EQUAL --
        # a legal, non-hard-error diagram-level DIMENSION_DEFERRED finding.
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d * e])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])

        pre_report = validate(diagram)
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id == a_id
            for issue in pre_report.deferred
        )

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)

        # The merged node has exactly one surviving leg (forced onto shared_dim=d) -- no
        # second leg left to defer against, so the finding is gone from `working` entirely.
        post_report = validate(result.diagram)
        assert not any(issue.kind is IssueKind.DIMENSION_DEFERRED for issue in post_report.deferred)

        assert len(result.step.removed_deferred_issues) == 1
        removed = result.step.removed_deferred_issues[0]
        assert removed.kind is IssueKind.DIMENSION_DEFERRED
        assert removed.node_id == a_id  # recorded in the input diagram's own coordinates

    def test_a_carried_over_deferred_leg_is_not_recorded_as_removed(self) -> None:
        # Mirror case: the deferred pair survives fusion untouched (on the third,
        # unrelated node c_id) -- removed_deferred_issues must stay empty.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d * e])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs(
            [PortRef(c_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.OUTPUT, 1)]
        )

        pre_report = validate(diagram)
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id == c_id
            for issue in pre_report.deferred
        )

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert result.step.removed_deferred_issues == ()

        post_report = validate(result.diagram)
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id == c_id
            for issue in post_report.deferred
        )


class TestDimensionConstraintsExactContent:
    """Phase 5 post-closing audit, dimension_constraints duplicate-assumption defect.

    Before the fix, ``_unify_surviving_legs`` (match.py) unified each surviving leg's raw,
    unresolved ``Dim`` against ``shared_dim`` -- unlike its sibling ``_unify_phase_dims``,
    which first resolves the checked ``Dim`` through the running ``bindings`` accumulator.
    A leg still mentioning a symbol some earlier leg or phase had already bound concretely
    was therefore re-unified and re-appended to ``dimension_constraints`` as though it were
    a fresh fact, once per such leg and again on every fixpoint pass that left ``shared_dim``
    unchanged. Asserts exact tuple content (not merely length) on ``RewriteStep
    .dimension_constraints`` -- the field Phase 6 will read as the certificate -- for three
    shapes, mirroring the accumulator discipline ``_unify_phase_dims`` already had.
    """

    def test_a_single_bound_leg_contributes_exactly_one_entry(self) -> None:
        # A: output dim d (consumed). B: input dim 2 (consumed), output dim 2 (survives).
        # The connecting pair (d, 2) is a bare syntactic identity only after unify binds
        # d := 2 -- recorded once. B's one surviving leg (2) is then checked against the
        # already-refined shared_dim (2): a bare identity, nothing new to record.
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[two], output_dims=[two])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        expected = (
            DimensionConstraint(
                assumed=d,
                equal_to=two,
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.BOUND,
                bound_here=(("d", two),),
            ),
        )
        assert result.step.dimension_constraints == expected
        assert match.dimension_constraints == expected

    def test_a_surviving_leg_bound_by_an_earlier_leg_contributes_no_duplicate(self) -> None:
        # A: input dims [d, 2], output dim d (consumed). B: input dim d (consumed), output
        # dim d (survives). Connecting pair (d, d) is a bare identity, nothing recorded.
        # A's own surviving input leg 2 unifies against shared_dim d, binding d := 2 --
        # recorded once, as (2, d). Every other surviving leg (A's own d-leg, and B's
        # surviving output d-leg) resolves through that same binding to a bare identity
        # against the now-concrete shared_dim -- including B's leg, checked in the very same
        # fixpoint pass, since bindings is one whole-candidate accumulator both nodes' leg
        # sweeps read from and write into, not one scoped per node or per pass.
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[d, two], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        expected = (
            DimensionConstraint(
                assumed=two,
                equal_to=d,
                source=ConstraintSource.surviving_leg(PortRef(a_id, Direction.INPUT, 1)),
                outcome=ConstraintOutcome.BOUND,
                bound_here=(("d", two),),
            ),
        )
        assert result.step.dimension_constraints == expected
        assert match.dimension_constraints == expected

    def test_two_independent_deferred_legs_are_both_recorded_not_collapsed(self) -> None:
        # A: input dim d*e (survives), output dim d (consumed). B: input dim d (consumed),
        # output dim d*e (survives). shared_dim = d (connecting pair, bare identity). Both
        # surviving legs (A's d*e, B's d*e) unify against d and defer -- neither resolves
        # through bindings (unify(d*e, d) never binds a symbol, since there is no single
        # substitution of d or e alone that equates a product to one of its own factors), so
        # both are genuinely independent, undischarged assumptions and both are recorded --
        # this is not the duplicate-assumption defect, since neither check could have been
        # derived from the other's outcome.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        de = d * e
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[de], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[de])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        expected = (
            DimensionConstraint(
                assumed=de,
                equal_to=d,
                source=ConstraintSource.surviving_leg(PortRef(a_id, Direction.INPUT, 0)),
                outcome=ConstraintOutcome.DEFERRED,
            ),
            DimensionConstraint(
                assumed=de,
                equal_to=d,
                source=ConstraintSource.surviving_leg(PortRef(b_id, Direction.OUTPUT, 0)),
                outcome=ConstraintOutcome.DEFERRED,
            ),
        )
        assert result.step.dimension_constraints == expected
        assert match.dimension_constraints == expected


class TestRemovedDeferredIssuesMultisetCompare:
    """Phase 5 post-closing audit, ``removed_deferred_issues`` key-collision defect.

    Before the fix, the deferred compare in ``apply`` (engine.py) keyed a plain dict
    comprehension on ``_translate_input_issue_key(issue, ...)``. That function maps both
    consumed node ids of a fusion onto the sole surviving ``new_node_ids[0]``, so two
    distinct, node-anchored ``DIMENSION_DEFERRED`` issues -- one on each of the two fused
    spiders -- translate to the *same* key and collapse to one dict entry, silently dropping
    one to last-write-wins. The fix makes this compare multiset-aware (a ``Counter``
    difference), mirroring step 8's own hard-error compare instead of diverging from it.
    """

    def test_two_node_anchored_deferred_issues_are_both_reported(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        de = d * e
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[de], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[de])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])

        pre_report = validate(diagram)
        assert len(pre_report.deferred) == 2
        pre_by_node = {issue.node_id: issue for issue in pre_report.deferred}
        assert set(pre_by_node) == {a_id, b_id}

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)

        post_report = validate(result.diagram)
        assert post_report.deferred == ()

        removed = result.step.removed_deferred_issues
        assert len(removed) == 2
        # Not merely count: both are the input diagram's own pre-rewrite issues, anchored on
        # a_id and b_id respectively (equal to -- not the same object as, since apply() reruns
        # validate(diagram) itself -- pre_report's own issues), never a translated stand-in.
        assert removed[0] == pre_by_node[a_id]
        assert removed[1] == pre_by_node[b_id]
        assert {issue.node_id for issue in removed} == {a_id, b_id}
        for issue in removed:
            assert issue.kind is IssueKind.DIMENSION_DEFERRED


class TestDeferredIssueProvenanceIsSymmetric:
    """D2/D3: a rewrite can introduce a deferred assumption as readily as it removes one, and
    the identity contract on a colliding key is stated and pinned, not left implicit.
    """

    def test_introduced_deferred_issue_is_recorded_in_post_rewrite_coordinates(self) -> None:
        # A: output dims [d, d*e] (d consumed by the fusion, d*e survives on A's own leg,
        # forced onto shared_dim=d by the builder). Before the fusion, the *node* A itself
        # carries a DIMENSION_DEFERRED (its own two legs disagree). After it, A's surviving
        # leg is forced to the merged node's shared_dim=d, but the wire from there to C
        # (whose own leg is still the untouched d*e) now assumes d == d*e at the *wire* --
        # a brand new deferred assumption that did not exist before, on a piece of the
        # diagram (the wire) that did not exist before either.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        de = d * e
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, de])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[de], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(c_id, Direction.INPUT, 0))

        pre_report = validate(diagram)
        assert len(pre_report.deferred) == 1
        assert pre_report.deferred[0].node_id == a_id

        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)

        assert len(result.step.removed_deferred_issues) == 1
        assert result.step.removed_deferred_issues[0] == pre_report.deferred[0]

        post_report = validate(result.diagram)
        assert len(post_report.deferred) == 1
        assert post_report.deferred[0].wire is not None

        assert len(result.step.introduced_deferred_issues) == 1
        assert result.step.introduced_deferred_issues[0] == post_report.deferred[0]
        assert not result.step.deferred_issue_identity_ambiguous

    def test_neither_field_is_populated_when_nothing_changes(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        assert result.step.removed_deferred_issues == ()
        assert result.step.introduced_deferred_issues == ()
        assert not result.step.deferred_issue_identity_ambiguous

    def test_colliding_keys_are_flagged_ambiguous_and_pinned_to_validate_order(self) -> None:
        """Unit-level pin of :func:`~qufzx.rewrite.engine._select_by_key_surplus`'s own
        contract (see :attr:`~qufzx.rewrite.engine.RewriteStep
        .deferred_issue_identity_ambiguous`'s docstring): when several issues collide on one
        translated key and only *some* of them have a surplus, the selection is arbitrary
        but deterministic -- first in the given (``validate``) order -- and the ambiguity
        flag says so. Exercised directly against the selector rather than by engineering a
        diagram-level collision, since the routine is the one, shared implementation both
        ``removed_deferred_issues`` and ``introduced_deferred_issues`` call, and its contract
        is a property of the selection algorithm itself, not of any one diagram shape.
        """
        from collections import Counter

        from qufzx.rewrite.engine import _select_by_key_surplus

        key = (IssueKind.DIMENSION_DEFERRED, None)
        first = ValidationIssue(
            kind=IssueKind.DIMENSION_DEFERRED, message="first", node_id=NodeId(0), deferred=True
        )
        second = ValidationIssue(
            kind=IssueKind.DIMENSION_DEFERRED, message="second", node_id=NodeId(1), deferred=True
        )
        keyed = ((key, first), (key, second))

        # Full surplus (both lack a counterpart): every colliding issue is reported, and the
        # selection -- though every issue happens to be included -- is not ambiguous, since
        # nothing was left out to have been chosen over.
        selected, ambiguous = _select_by_key_surplus(keyed, Counter({key: 2}))
        assert selected == (first, second)
        assert not ambiguous

        # Partial surplus (only one of the two colliding issues has a counterpart): the
        # selection takes the first in the given order and flags the result ambiguous,
        # rather than silently implying `first` was chosen because it is somehow the "real"
        # removed one.
        selected, ambiguous = _select_by_key_surplus(keyed, Counter({key: 1}))
        assert selected == (first,)
        assert ambiguous

        # Zero surplus: nothing selected, and correctly not ambiguous (there is no partial
        # selection to be ambiguous about).
        selected, ambiguous = _select_by_key_surplus(keyed, Counter())
        assert selected == ()
        assert not ambiguous


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
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

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
        # c_id and (below) r_id are deliberately legless: this test is about a consumed
        # wire between two *other* nodes being dropped, not about c_id/r_id's own shape.
        # Each still needs *some* dimension-bearing content (a phase, here, since it has no
        # legs to carry one) so it satisfies validate()'s NODE_DIMENSION_UNDETERMINED check
        # (round 20, Task 9) and does not itself contribute a spurious pre/post error this
        # test is not exercising.
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[], phase=PhaseVector(d, {}))
        w = Wire(PortRef(s1_id, Direction.OUTPUT, 0), PortRef(s2_id, Direction.INPUT, 0))
        diagram.add_wire(w.a, w.b)
        diagram.set_boundary_outputs([PortRef(s1_id, Direction.OUTPUT, 0)])
        diagram.set_boundary_inputs([PortRef(s2_id, Direction.INPUT, 0)])

        # Both endpoints of w are deliberately also on a boundary.
        pre_errors = validate(diagram).errors
        assert len(pre_errors) == 2
        assert all(issue.kind is IssueKind.PORT_WIRED_AND_BOUNDARY for issue in pre_errors)

        def _builder(working: Diagram, match: Match) -> BuildResult:
            r_id = working.add_node(
                Z_SPIDER, input_dims=[], output_dims=[], phase=PhaseVector(d, {})
            )
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

    def test_a3_duplicate_consumed_node_ids_raises_rewrite_grammar_error(self) -> None:
        """A3 (Phase 5 round-12 audit): a repeated entry in ``consumed_node_ids`` passes the
        plain membership check (every entry, including the repeat, names a real node) but
        would otherwise make step 6's removal loop call ``remove_node`` twice on the same,
        by-then-already-removed id, raising ``qufzx.diagram.graph.GraphGrammarError`` -- a
        different module's exception, escaping the ``RewriteError`` hierarchy ``apply``'s own
        docstring promises. It must instead be rejected as a malformed request, before step 6
        is ever reached, with the same ``RewriteGrammarError`` every other malformed
        ``BuildResult`` field raises.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.set_boundary_outputs([PortRef(c_id, Direction.OUTPUT, 0)])

        def _builder(working: Diagram, match: Match) -> BuildResult:
            r_id = working.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
            return BuildResult(
                diagram=working,
                new_node_ids=(r_id,),
                consumed_node_ids=(c_id, c_id),
                consumed_wires=(),
                port_mapping={
                    PortRef(c_id, Direction.OUTPUT, 0): PortRef(r_id, Direction.OUTPUT, 0)
                },
                scalar_introduced=Scalar.one(),
            )

        rule = Rule(
            name="scripted_duplicate_consumed_node_ids",
            pattern=_EmptyPattern(),
            builder=_builder,
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )

        with pytest.raises(RewriteGrammarError, match="more than once"):
            apply(diagram, rule, _ScriptedMatch())

    def test_duplicate_new_node_ids_raises_rewrite_grammar_error(self) -> None:
        """Class 2 sweep (Phase 5 post-closing audit round 19, Task 4): ``new_node_ids`` is a
        structurally identical reference kind to ``consumed_node_ids`` (a tuple of ``NodeId``
        the builder reports about this one rewrite), but only the latter had a duplicate
        check (A3) before this fix. A repeat here drives no imperative loop into a crash --
        unlike A3 -- but it would still misreport, to ``RewriteStep.new_node_ids`` and
        Phase 6's certificate, that a rewrite created two new nodes when it created one.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        diagram.set_boundary_outputs([])

        def _builder(working: Diagram, match: Match) -> BuildResult:
            r_id = working.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
            return BuildResult(
                diagram=working,
                new_node_ids=(r_id, r_id),
                consumed_node_ids=(),
                consumed_wires=(),
                port_mapping={},
                scalar_introduced=Scalar.one(),
            )

        rule = Rule(
            name="scripted_duplicate_new_node_ids",
            pattern=_EmptyPattern(),
            builder=_builder,
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )

        with pytest.raises(RewriteGrammarError, match="more than once"):
            apply(diagram, rule, _ScriptedMatch())

    def test_hardening_5_non_injective_port_mapping_raises_rewrite_grammar_error(self) -> None:
        """Hardening 5 (Phase 5 post-closing audit round 18): port_mapping's injectivity is
        load-bearing (step 5 relies on distinct surviving old ports remapping to distinct new
        ports) but was previously unchecked. Two consumed ports mapped to the *same* new port
        would, once both their wires are remapped, produce two ``Wire`` objects that could be
        identical -- silently collapsing into one entry of ``Diagram``'s set-backed wire
        storage, with no exception anywhere. ``spider_fusion_builder`` is injective by
        construction, so this never fires for Phase 5's one registered rule; a foreign or
        future builder need not be.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        diagram.set_boundary_outputs(
            [PortRef(c_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.OUTPUT, 1)]
        )

        def _builder(working: Diagram, match: Match) -> BuildResult:
            r_id = working.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
            non_injective_mapping = {
                PortRef(c_id, Direction.OUTPUT, 0): PortRef(r_id, Direction.OUTPUT, 0),
                PortRef(c_id, Direction.OUTPUT, 1): PortRef(r_id, Direction.OUTPUT, 0),
            }
            return BuildResult(
                diagram=working,
                new_node_ids=(r_id,),
                consumed_node_ids=(c_id,),
                consumed_wires=(),
                port_mapping=non_injective_mapping,
                scalar_introduced=Scalar.one(),
            )

        rule = Rule(
            name="scripted_non_injective_port_mapping",
            pattern=_EmptyPattern(),
            builder=_builder,
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )

        with pytest.raises(RewriteGrammarError, match="not injective"):
            apply(diagram, rule, _ScriptedMatch())

    def test_hardening_6_wire_count_postcondition_catches_a_silently_lost_wire(self) -> None:
        """Hardening 6 (Phase 5 post-closing audit round 18): a cheap structural postcondition
        on step 5's remapping as a whole -- ``len(working.wires) == len(pre_wires) -
        len(set(consumed_wires))`` -- catching a wire lost by any mechanism, not only the two
        step 5 (single-wire collapse) and hardening 5 (port_mapping injectivity) already
        guard. This reproduction is injective (``port_mapping`` has exactly one entry, so
        injectivity is trivially satisfied) and does not collapse *within* one wire's own two
        endpoints (``new_a != new_b``) -- it aliases a consumed port onto an *already-live*
        port that some other, untouched wire already uses, so the remapped wire silently
        duplicates that pre-existing one once added to ``Diagram``'s set-backed wire storage,
        losing a wire with neither of the other two checks ever firing.
        """
        d = Dim.concrete(2)
        diagram = Diagram()
        c_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        x_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        wire1 = Wire(PortRef(c_id, Direction.OUTPUT, 0), PortRef(x_id, Direction.INPUT, 0))
        # A second, pre-existing wire onto the exact same (already-occupied) x_id port --
        # permissive but weird, mirroring exactly what qufzx.diagram.graph documents as
        # deliberately unchecked at construction time (well-formedness is validate()'s job,
        # never this module's). Both wires exist in the diagram before apply() is ever
        # called; only wire1's own node is reported as consumed.
        wire2 = Wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(x_id, Direction.INPUT, 0))
        diagram.add_wire(wire1.a, wire1.b)
        diagram.add_wire(wire2.a, wire2.b)

        def _builder(working: Diagram, match: Match) -> BuildResult:
            # A builder bug: aliasing c_id's surviving port onto b_id's *existing* port,
            # rather than a freshly created one. This single-entry port_mapping is
            # trivially injective, so hardening 5 does not fire.
            return BuildResult(
                diagram=working,
                new_node_ids=(),
                consumed_node_ids=(c_id,),
                consumed_wires=(),
                port_mapping={
                    PortRef(c_id, Direction.OUTPUT, 0): PortRef(b_id, Direction.OUTPUT, 0),
                },
                scalar_introduced=Scalar.one(),
            )

        rule = Rule(
            name="scripted_wire_count_violation",
            pattern=_EmptyPattern(),
            builder=_builder,
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )

        with pytest.raises(RewriteGrammarError, match="wire-count postcondition"):
            apply(diagram, rule, _ScriptedMatch())


class TestConditionNumberingMatchesDeclaredOrder:
    """Round 24: numbered "condition N" references must agree with FUSION_SIDE_CONDITIONS.

    The seven side conditions are addressed two ways throughout this package -- by *name*
    (``dimension_agreement``) and by *position* ("condition 6"). Only the name is checkable
    by the compiler. Round 23 inserted ``consumed_ports_singly_claimed`` at position 5,
    shifting ``dimension_agreement`` 5 -> 6 and ``phase_dimension_agreement`` 6 -> 7, and
    left **seven** references across three modules stating the old numbers -- including one
    degraded into the meaningless "condition 6/6", and one
    ("condition 5's own convention ... ``leg_deferred``") that no amount of grepping for the
    obvious pattern would have surfaced.

    That is the class this repo's audit history keeps rediscovering: one fact stated in two
    places and kept in sync by hand. Two checks below, deliberately of different strengths:

    * :meth:`test_module_docstring_list_matches_declared_order` is *exact*. The numbered list
      in :mod:`qufzx.rewrite.match`'s own module docstring is the authoritative statement of
      the numbering that every "condition N" elsewhere refers to, and it is machine-readable,
      so it is machine-checked -- no heuristic, no false positives, no escape. A future
      insertion that renumbers the conditions cannot land without this failing.
    * :meth:`test_adjacent_number_and_name_agree` is a *partial net* over cross-references in
      prose, and is documented as partial rather than sold as complete. "Condition 6" and
      ``phase_dimension_agreement`` legitimately appear in one sentence whenever the prose
      contrasts two conditions, so proximity is not reference; the window is calibrated
      (:data:`_WINDOW`) to the tight ``condition N (``name``)`` / ``` ``name`` (condition N)```
      shapes only, and compares the stated number *set* against the named conditions'
      declared position set for equality. It would have caught two of round 24's seven
      findings on its own; the exact check above is what actually guards the renumber,
      and this one adds a second layer over the prose that sits closest to it.
    """

    _WINDOW = 60
    """Characters either side of a "condition N" mention to search for a condition name.

    Calibrated, not guessed: measured over the five modules below, 60 inspects 13 references
    with zero false positives, while 90 pulls in the contrast sentence at ``match.py``'s
    condition-7 body ("unlike condition 6's own ``leg_deferred``, a *passing*
    ``phase_dimension_agreement`` outcome ...") where a number and a *different* condition's
    name sit in one sentence entirely correctly. Widening this without re-checking that
    trade-off will produce false failures, not extra coverage."""

    _MODULES = (
        "qufzx/rewrite/match.py",
        "qufzx/rewrite/rules_library.py",
        "qufzx/rewrite/engine.py",
        "qufzx/rewrite/rule.py",
        "qufzx/diagram/validate.py",
    )

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    """Anchor paths to the repository, never to the process CWD -- pytest may be invoked from
    anywhere, and a silently-missing file would make the sweep below vacuous."""

    _LIST_ITEM_RE = re.compile(r"^(\d+)\. ``([a-z_]+)``", re.MULTILINE)
    _NUMBER_RE = re.compile(r"\bcondition[s]? (\d+)(?: and (\d+))?")
    _HEADING_PREFIX_RE = re.compile(r"^\s*\d+\.\s+$")

    @staticmethod
    def _positions() -> dict[str, int]:
        return {condition.name: i for i, condition in enumerate(FUSION_SIDE_CONDITIONS, 1)}

    def test_module_docstring_list_matches_declared_order(self) -> None:
        """The exact check: the authoritative numbered list *is* FUSION_SIDE_CONDITIONS."""
        docstring = match_module.__doc__
        assert docstring is not None, "qufzx.rewrite.match lost its module docstring"
        listed = [
            (int(number), name) for number, name in self._LIST_ITEM_RE.findall(docstring)
        ]
        expected = [
            (i, condition.name) for i, condition in enumerate(FUSION_SIDE_CONDITIONS, 1)
        ]
        assert listed == expected, (
            "qufzx.rewrite.match's module docstring states the conditions in an order that "
            f"disagrees with FUSION_SIDE_CONDITIONS.\n  docstring: {listed}\n  declared:  "
            f"{expected}"
        )

    @classmethod
    def _is_list_heading(cls, window: str, name_start: int) -> bool:
        """Is the name at ``name_start`` the heading of a numbered list item?

        Inside item 7's *body*, a mention of "condition 6" legitimately refers to a different
        condition while item 7's own heading name is still the nearest one in the window --
        so a heading occurrence is not a referent. Only the name's role is excluded, never
        the surrounding text: a second, non-heading mention of the same name in that body is
        still checked. The heading itself is covered exactly by
        :meth:`test_module_docstring_list_matches_declared_order`.
        """
        line_start = window.rfind("\n", 0, name_start) + 1
        return bool(cls._HEADING_PREFIX_RE.match(window[line_start:name_start]))

    def _cross_references(self) -> list[tuple[str, int, set[int], set[str]]]:
        """Every ``(module, line, stated numbers, adjacent names)`` the net inspects."""
        positions = self._positions()
        name_re = re.compile("``(" + "|".join(map(re.escape, positions)) + ")``")
        found: list[tuple[str, int, set[int], set[str]]] = []
        for relative in self._MODULES:
            path = self._REPO_ROOT / relative
            assert path.is_file(), f"{relative} not found at {path}"
            text = path.read_text(encoding="utf-8")
            for number_match in self._NUMBER_RE.finditer(text):
                numbers = {int(g) for g in number_match.groups() if g}
                start = max(0, number_match.start() - self._WINDOW)
                window = text[start : number_match.end() + self._WINDOW]
                names = {
                    m.group(1)
                    for m in name_re.finditer(window)
                    if not self._is_list_heading(window, m.start())
                }
                if names:
                    line = text.count("\n", 0, number_match.start()) + 1
                    found.append((relative, line, numbers, names))
        return found

    def test_adjacent_number_and_name_agree(self) -> None:
        """The partial net: a number stated right beside a name must be that name's."""
        positions = self._positions()
        # Strict set equality, not mere overlap. Overlap was the first draft's rule and is
        # too lenient for a multi-number mention: the stale round-23 wording
        # "conditions 5 and 6 (``dimension_agreement``, ``phase_dimension_agreement``)"
        # states {5, 6} against true positions {6, 7}, which *do* overlap at 6 -- so an
        # overlap rule would have waved through the exact wording this net exists to catch.
        # Equality is safe here only because _WINDOW is tight enough to exclude contrast
        # sentences (see that constant); measured over the five modules, all 13 inspected
        # references satisfy equality exactly.
        violations = [
            f"{relative}:{line}: 'condition(s) {sorted(numbers)}' sits beside "
            f"{sorted(names)}, whose declared position(s) are "
            f"{sorted(positions[name] for name in names)}"
            for relative, line, numbers, names in self._cross_references()
            if numbers != {positions[name] for name in names}
        ]
        assert not violations, (
            "numbered condition reference(s) disagree with FUSION_SIDE_CONDITIONS' declared "
            "order:\n  " + "\n  ".join(violations)
        )

    def test_the_net_is_not_vacuous(self) -> None:
        """It must actually inspect references, and must reject text it is meant to reject.

        Without this, a regex that silently stopped matching (a changed quoting convention,
        say) would leave the check above passing on zero inspected references forever -- the
        same vacuity failure mode the property harness's own floors exist to prevent.
        """
        inspected = self._cross_references()
        assert len(inspected) >= 8, (
            f"the adjacency net inspected only {len(inspected)} reference(s); it is close to "
            "vacuous -- has the ``name``/'condition N' wording convention changed?"
        )
        positions = self._positions()
        assert positions["dimension_agreement"] == 6
        assert positions["phase_dimension_agreement"] == 7
        # The exact wording round 24 found stale in match.py, before it was fixed.
        stale = "conditions 5 and 6 (``dimension_agreement``, ``phase_dimension_agreement``)"
        numbers = {
            int(g) for m in self._NUMBER_RE.finditer(stale) for g in m.groups() if g
        }
        assert numbers == {5, 6}, numbers
        names = set(re.findall(r"``([a-z_]+)``", stale))
        assert numbers != {positions[name] for name in names}, (
            "the stale round-23 wording must be detectable as a disagreement, or this net "
            "would not have caught it either"
        )
        assert numbers & {positions[name] for name in names}, (
            "and it must be detectable *despite* overlapping, which is exactly why this "
            "check uses set equality rather than intersection"
        )


class TestApplyDocstringMatchesRaiseSites:
    """Round 20, Task 5: ``apply``'s docstring drifted out of date twice (rounds 18-19 each
    added raise sites without updating the prose a caller actually reads), because nothing
    checked the two stayed in sync. This pins the count of raise sites lexically inside
    ``apply``'s own body so a future addition without a matching docstring update fails here,
    naming the offending line, rather than silently drifting a third time.

    Deliberately an *equality* pin on a count, not a semantic check of the docstring's
    content -- the AST can enumerate raise sites cheaply and reliably, but "does the prose
    accurately describe this raise site" is not a question ast.parse can answer. See
    ``apply``'s own docstring for why one of the 11 total documented raise conditions (the
    unmapped-surviving-port raise inside ``_remap_endpoint``, called from step 5) is not
    counted here: it is not a literal ``raise`` statement lexically inside ``apply``'s body,
    so the AST walk below -- by construction -- cannot see it and must not claim to.
    """

    #: Count of ``raise RewriteGrammarError(...)``/``raise RewriteDomainError(...)``
    #: statements lexically inside ``apply``'s own function body, as of round 20. Keep this
    #: adjacent to ``apply``'s docstring (qufzx/rewrite/engine.py) so updating one without the
    #: other is visibly wrong in review.
    _EXPECTED_RAISE_SITE_COUNT = 10

    def test_raise_site_count_matches_the_pinned_constant(self) -> None:
        source = inspect.getsource(engine_module.apply)
        # dedent not needed: inspect.getsource of a module-level function returns
        # unindented source starting at "def apply(...)".
        tree = ast.parse(source)
        (func_node,) = tree.body
        assert isinstance(func_node, ast.FunctionDef)

        raise_sites = []
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Raise):
                continue
            call = node.exc
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in ("RewriteGrammarError", "RewriteDomainError"):
                raise_sites.append(node.lineno)

        assert len(raise_sites) == self._EXPECTED_RAISE_SITE_COUNT, (
            f"apply() now has {len(raise_sites)} RewriteGrammarError/RewriteDomainError "
            f"raise site(s) at (function-relative) line(s) {raise_sites}, but "
            f"_EXPECTED_RAISE_SITE_COUNT is still {self._EXPECTED_RAISE_SITE_COUNT} -- "
            "update apply()'s docstring to document the new/removed raise condition, then "
            "update this constant to match"
        )


class TestCrossProcessDeterminism:
    """Phase 5 post-closing audit round 18, Defect 1: certificate provenance must not vary
    by ``PYTHONHASHSEED``.

    Root cause: :meth:`~qufzx.diagram.validate._check_wire_dimensions` and
    :meth:`~qufzx.diagram.validate._check_port_usage` used to iterate ``diagram.wires``, a
    frozenset. ``PortRef``'s (and so ``Wire``'s) hash folds in ``Direction``, an
    ``enum.Enum`` whose default hash is its member *name*'s string hash -- seed-dependent
    under ``PYTHONHASHSEED`` (Python randomizes string hashing per process by default).
    ``validate()``'s issue order therefore varied per process, and
    :func:`~qufzx.rewrite.engine._select_by_key_surplus` (which walks that order to populate
    :attr:`~qufzx.rewrite.engine.RewriteStep.removed_deferred_issues` /
    ``introduced_deferred_issues`` when a translated key collides across several issues) made
    those fields non-deterministic across processes, contradicting
    :attr:`~qufzx.rewrite.engine.RewriteStep.deferred_issue_identity_ambiguous`'s own
    docstring promise of "first in validate order".

    Fixed by sorting every wire iteration whose order is observable
    (``qufzx.diagram.graph.PortRef.sort_key`` / ``Wire.sort_key``, hash-independent) at every
    site in :mod:`qufzx.diagram.validate`, :mod:`qufzx.rewrite.match`, and
    :mod:`qufzx.rewrite.engine` where the order could reach a returned value, a recorded
    certificate field, or an exception message -- see each site's own comment for why it, in
    particular, needed the sort (or, at two sites, provably did not, and why).

    This test runs the identical rewrite in two child processes with different
    ``PYTHONHASHSEED`` values (rather than merely asserting something in-process, which
    cannot observe seed-dependent hash ordering at all) and compares a full, stable
    serialization of the fields the module docstrings promise are deterministic. It
    deliberately never compares ``hash()`` of anything: ``IssueKind`` and ``Direction`` are
    ``Enum``s hashed by member name, so ``hash(step)`` (or ``hash(issue)``, etc.)
    legitimately -- and permanently -- differs across processes; that is a fact about
    Python's ``Enum``/``str`` hashing this round does not change and could not soundly
    change (``RewriteStep.__hash__``'s own docstring is corrected elsewhere in this round to
    say so explicitly, rather than the pre-round-18 wording that implied more than
    within-process stability).
    """

    SCRIPT = Path(__file__).parent / "_cross_process_determinism_script.py"

    def _run_with_seed(self, seed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, str(self.SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_certificate_and_validation_reports_are_byte_identical_across_hash_seeds(
        self,
    ) -> None:
        # Two seeds, per the acceptance test as specified -- each subprocess pays sympy's
        # import cost, so this stays at two rather than a larger sample to keep the suite's
        # runtime reasonable; two distinct seeds already falsify "no sorting" (verified by
        # hand while developing this fix: reverting the sort in validate.py makes this
        # comparison flake across seeds within a handful of runs).
        first = self._run_with_seed("0")
        second = self._run_with_seed("2147483647")
        assert first, "the driver script printed nothing"
        assert second == first
