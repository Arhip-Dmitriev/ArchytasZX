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

"""Establishes the Rule/BuildResult/SideCondition value objects and their validation."""

from __future__ import annotations

import dataclasses

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionPattern
from qufzx.rewrite.rule import (
    BuildResult,
    ConstraintOutcome,
    ConstraintSource,
    ConstraintSourceKind,
    DimensionConstraint,
    Match,
    Pattern,
    Quantifiers,
    RewriteGrammarError,
    Rule,
    SideCondition,
    SideConditionOutcome,
)
from qufzx.rewrite.rules_library import spider_fusion_builder


def _dummy_builder(diagram: Diagram, match: Match) -> BuildResult:
    """A builder with no ``side_conditions`` attribute, for tests that only need *some*
    callable and must not trip the builder/Rule ``side_conditions`` agreement check that
    ``spider_fusion_builder`` (which does declare one) would otherwise trigger."""
    raise NotImplementedError


class TestSideCondition:
    def test_stores_name_and_description(self) -> None:
        condition = SideCondition("distinct_nodes", "the two matched nodes differ")
        assert condition.name == "distinct_nodes"
        assert condition.description == "the two matched nodes differ"


class TestSideConditionOutcome:
    def test_defaults_not_deferred(self) -> None:
        outcome = SideConditionOutcome("distinct_nodes", True, "1 != 2")
        assert outcome.passed
        assert not outcome.deferred

    def test_deferred_flag_stored(self) -> None:
        outcome = SideConditionOutcome("dimension_agreement", True, "assumed", deferred=True)
        assert outcome.deferred


class TestQuantifiers:
    def test_defaults_are_empty(self) -> None:
        quantifiers = Quantifiers()
        assert quantifiers.leg_counts == ()
        assert quantifiers.dimensions == ()

    def test_stores_declared_names(self) -> None:
        quantifiers = Quantifiers(leg_counts=("m", "n"), dimensions=("d",))
        assert quantifiers.leg_counts == ("m", "n")
        assert quantifiers.dimensions == ("d",)


class TestBuildResult:
    def test_stores_fields(self) -> None:
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        result = BuildResult(
            diagram=diagram,
            new_node_ids=(node_id,),
            consumed_node_ids=(),
            consumed_wires=(),
            port_mapping={},
            scalar_introduced=Scalar.one(),
        )
        assert result.diagram is diagram
        assert result.new_node_ids == (node_id,)
        assert result.consumed_node_ids == ()
        assert result.scalar_introduced == Scalar.one()

    def test_is_frozen(self) -> None:
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        result = BuildResult(
            diagram=diagram,
            new_node_ids=(node_id,),
            consumed_node_ids=(),
            consumed_wires=(),
            port_mapping={},
            scalar_introduced=Scalar.one(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.new_node_ids = (node_id,)  # type: ignore[misc]


class TestRule:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(
                name="",
                pattern=FusionPattern(),
                builder=spider_fusion_builder,
                side_conditions=(),
                quantifiers=Quantifiers(),
                scalar_introduced=Scalar.one(),
            )

    def test_stores_fields(self) -> None:
        rule = Rule(
            name="spider_fusion",
            pattern=FusionPattern(),
            builder=spider_fusion_builder,
            side_conditions=FUSION_SIDE_CONDITIONS,
            quantifiers=Quantifiers(dimensions=("d",)),
            scalar_introduced=Scalar.one(),
        )
        assert rule.name == "spider_fusion"
        assert isinstance(rule.pattern, FusionPattern)
        assert rule.builder is spider_fusion_builder
        assert rule.scalar_introduced == Scalar.one()

    def test_is_frozen(self) -> None:
        rule = Rule(
            name="spider_fusion",
            pattern=FusionPattern(),
            builder=spider_fusion_builder,
            side_conditions=FUSION_SIDE_CONDITIONS,
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.name = "renamed"  # type: ignore[misc]


class TestRuleValidatesEveryField:
    """__post_init__ must validate every field, not only ``name``.

    Checking ``name`` alone lets ``Rule(name="x", pattern="not a pattern", builder="not
    callable", side_conditions=(), quantifiers=Quantifiers(), scalar_introduced=1.0)``
    construct successfully -- including a bare float ``scalar_introduced``, banned
    everywhere else in this codebase by the exact-scalars rule.
    """

    def _kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "name": "spider_fusion",
            "pattern": FusionPattern(),
            "builder": _dummy_builder,
            "side_conditions": (),
            "quantifiers": Quantifiers(),
            "scalar_introduced": Scalar.one(),
        }
        base.update(overrides)
        return base

    def test_rejects_non_pattern_pattern(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(**self._kwargs(pattern="not a pattern"))  # type: ignore[arg-type]

    def test_rejects_non_callable_builder(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(**self._kwargs(builder="not callable"))  # type: ignore[arg-type]

    def test_rejects_side_conditions_not_a_tuple_of_side_condition(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(**self._kwargs(side_conditions=["not", "side", "conditions"]))  # type: ignore[arg-type]

    def test_rejects_side_conditions_tuple_with_wrong_element_type(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(**self._kwargs(side_conditions=("not a SideCondition",)))  # type: ignore[arg-type]

    def test_rejects_non_quantifiers_quantifiers(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(**self._kwargs(quantifiers="not quantifiers"))  # type: ignore[arg-type]

    def test_rejects_float_scalar_introduced(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(**self._kwargs(scalar_introduced=1.0))  # type: ignore[arg-type]

    def test_all_valid_fields_still_construct(self) -> None:
        rule = Rule(**self._kwargs())  # type: ignore[arg-type]
        assert rule.name == "spider_fusion"


class TestRuleRejectsSideConditionsDisagreeingWithItsBuilder:
    """``spider_fusion_builder`` declares its own
    ``side_conditions`` (see its ``.side_conditions`` attribute) and checks a match's
    coverage against exactly that tuple, since it is reachable directly and not only through
    ``apply``. If a ``Rule`` wrapping it declared a *different* ``side_conditions`` tuple,
    a match could pass one check and fail the other -- two contradictory verdicts on the
    same match, with no single source of truth. ``Rule.__post_init__`` must reject that
    combination outright, at construction time, rather than let it compile into an
    inconsistent rule.
    """

    def test_mismatched_side_conditions_raises_at_construction(self) -> None:
        with pytest.raises(RewriteGrammarError):
            Rule(
                name="spider_fusion",
                pattern=FusionPattern(),
                builder=spider_fusion_builder,
                side_conditions=(SideCondition("distinct_nodes", "distinct"),),
                quantifiers=Quantifiers(),
                scalar_introduced=Scalar.one(),
            )

    def test_matching_side_conditions_construct_successfully(self) -> None:
        rule = Rule(
            name="spider_fusion",
            pattern=FusionPattern(),
            builder=spider_fusion_builder,
            side_conditions=FUSION_SIDE_CONDITIONS,
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )
        assert rule.side_conditions == FUSION_SIDE_CONDITIONS

    def test_a_builder_with_no_declared_side_conditions_is_unconstrained(self) -> None:
        rule = Rule(
            name="x",
            pattern=FusionPattern(),
            builder=_dummy_builder,
            side_conditions=(SideCondition("anything", "unrelated to the builder"),),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )
        assert rule.side_conditions == (SideCondition("anything", "unrelated to the builder"),)


class TestDimensionConstraintBoundHereInvariant:
    """DimensionConstraint.__post_init__ enforces structurally what its docstring states --
    BOUND requires a non-empty bound_here, DEFERRED requires an empty one -- the same way
    ConstraintSource.__post_init__ rejects an illegal (kind, reference) combination. Both
    rejections and both legal combinations are pinned here so a future change cannot
    silently relax either half of the invariant.
    """

    def _source(self) -> ConstraintSource:
        return ConstraintSource.node_phase(NodeId(0))

    def test_bound_with_empty_bound_here_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="BOUND"):
            DimensionConstraint(
                assumed=Dim.concrete(2),
                equal_to=Dim.concrete(2),
                source=self._source(),
                outcome=ConstraintOutcome.BOUND,
                bound_here=(),
            )

    def test_deferred_with_non_empty_bound_here_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="DEFERRED"):
            DimensionConstraint(
                assumed=Dim.symbol("d"),
                equal_to=Dim.symbol("e"),
                source=self._source(),
                outcome=ConstraintOutcome.DEFERRED,
                bound_here=(("d", Dim.concrete(2)),),
            )

    def test_bound_with_non_empty_bound_here_constructs(self) -> None:
        constraint = DimensionConstraint(
            assumed=Dim.symbol("d"),
            equal_to=Dim.concrete(2),
            source=self._source(),
            outcome=ConstraintOutcome.BOUND,
            bound_here=(("d", Dim.concrete(2)),),
        )
        assert constraint.bound_here == (("d", Dim.concrete(2)),)

    def test_deferred_with_empty_bound_here_constructs(self) -> None:
        constraint = DimensionConstraint(
            assumed=Dim.symbol("d"),
            equal_to=Dim.symbol("e"),
            source=self._source(),
            outcome=ConstraintOutcome.DEFERRED,
        )
        assert constraint.bound_here == ()

    def test_malformed_bound_here_shape_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError):
            DimensionConstraint(
                assumed=Dim.concrete(2),
                equal_to=Dim.concrete(2),
                source=self._source(),
                outcome=ConstraintOutcome.BOUND,
                bound_here=(("d", "not a Dim"),),  # type: ignore[arg-type]
            )


class TestConstraintSourceRejectsMismatchedReferences:
    """Each ``ConstraintSourceKind`` accepts exactly its own reference field and no other."""

    def test_connecting_pair_with_a_port_ref_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="does not accept"):
            ConstraintSource(
                ConstraintSourceKind.CONNECTING_PAIR,
                port_ref=PortRef(NodeId(0), Direction.OUTPUT, 0),
            )

    def test_surviving_leg_without_a_port_ref_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="does not accept"):
            ConstraintSource(ConstraintSourceKind.SURVIVING_LEG)

    def test_surviving_leg_with_a_node_id_as_well_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="does not accept"):
            ConstraintSource(
                ConstraintSourceKind.SURVIVING_LEG,
                port_ref=PortRef(NodeId(0), Direction.OUTPUT, 0),
                node_id=NodeId(0),
            )

    def test_node_phase_without_a_node_id_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="does not accept"):
            ConstraintSource(ConstraintSourceKind.NODE_PHASE)


class TestConstraintRenderings:
    """``__str__`` on the two record value objects, one case per source kind."""

    def test_connecting_pair(self) -> None:
        assert str(ConstraintSource.connecting_pair()) == "connecting pair"

    def test_surviving_leg(self) -> None:
        source = ConstraintSource.surviving_leg(PortRef(NodeId(7), Direction.OUTPUT, 2))
        assert str(source) == "surviving leg 7.output[2]"

    def test_node_phase(self) -> None:
        assert str(ConstraintSource.node_phase(NodeId(4))) == "phase on node 4"

    def test_dimension_constraint(self) -> None:
        constraint = DimensionConstraint(
            assumed=Dim.symbol("d"),
            equal_to=Dim.concrete(2),
            source=ConstraintSource.node_phase(NodeId(1)),
            outcome=ConstraintOutcome.BOUND,
            bound_here=(("d", Dim.concrete(2)),),
        )
        assert str(constraint) == "d == 2 (bound, phase on node 1)"


class TestPatternIsAbstract:
    """``Pattern.find_matches`` is the seam later phases implement, never a usable default."""

    def test_calling_the_base_implementation_raises(self) -> None:
        class PassThroughPattern(Pattern):
            def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
                # The base body is the seam's contract, not a trivial stub mypy may skip.
                return super().find_matches(diagram)  # type: ignore[safe-super]

        with pytest.raises(NotImplementedError):
            PassThroughPattern().find_matches(Diagram())


class TestBuilderDeclaredSideConditionsMustAgreeWithTheRule:
    """A builder carrying its own ``side_conditions`` attribute pins the ``Rule`` wrapping it,
    so ``check_side_condition_coverage`` cannot be given two different tuples for one match."""

    def test_a_rule_disagreeing_with_its_builder_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="disagrees with its builder"):
            Rule(
                name="spider_fusion",
                pattern=FusionPattern(),
                builder=spider_fusion_builder,
                side_conditions=FUSION_SIDE_CONDITIONS[:-1],
                quantifiers=Quantifiers(),
                scalar_introduced=Scalar.one(),
            )

    def test_a_builder_with_no_declared_side_conditions_is_unconstrained(self) -> None:
        def bare_builder(diagram: Diagram, match: Match) -> BuildResult:
            raise AssertionError("never called")

        rule = Rule(
            name="bare",
            pattern=FusionPattern(),
            builder=bare_builder,
            side_conditions=(SideCondition("only", "the one condition"),),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )
        assert rule.side_conditions == (SideCondition("only", "the one condition"),)


class TestRuleRejectsRepeatedSideConditionNames:
    """A repeated declared name collapses in ``check_side_condition_coverage``'s set compare,
    where one reported outcome would cover both entries."""

    def test_two_conditions_sharing_one_name_are_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="more than once"):
            Rule(
                name="repeats",
                pattern=FusionPattern(),
                builder=_dummy_builder,
                side_conditions=(
                    SideCondition("shared", "first"),
                    SideCondition("shared", "second"),
                ),
                quantifiers=Quantifiers(),
                scalar_introduced=Scalar.one(),
            )

    def test_distinct_names_are_accepted(self) -> None:
        rule = Rule(
            name="distinct",
            pattern=FusionPattern(),
            builder=_dummy_builder,
            side_conditions=(SideCondition("one", "first"), SideCondition("two", "second")),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )
        assert len(rule.side_conditions) == 2
