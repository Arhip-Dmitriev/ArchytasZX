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
from qufzx.diagram.graph import Diagram, NodeId
from qufzx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionPattern
from qufzx.rewrite.rule import (
    BuildResult,
    ConstraintOutcome,
    ConstraintSource,
    DimensionConstraint,
    Match,
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
