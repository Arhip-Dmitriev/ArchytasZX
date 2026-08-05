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
from qufzx.diagram.graph import Diagram
from qufzx.rewrite.match import FusionPattern
from qufzx.rewrite.rule import (
    BuildResult,
    Quantifiers,
    RewriteGrammarError,
    Rule,
    SideCondition,
    SideConditionOutcome,
)
from qufzx.rewrite.rules_library import spider_fusion_builder


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
            side_conditions=(SideCondition("distinct_nodes", "distinct"),),
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
            side_conditions=(),
            quantifiers=Quantifiers(),
            scalar_introduced=Scalar.one(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.name = "renamed"  # type: ignore[misc]


class TestRuleValidatesEveryField:
    """Defect 4 (Phase 5 audit): __post_init__ used to check only ``name``, so
    ``Rule(name="x", pattern="not a pattern", builder="not callable", side_conditions=(),
    quantifiers=Quantifiers(), scalar_introduced=1.0)`` constructed successfully -- including
    a bare float ``scalar_introduced``, banned everywhere else in this codebase by the
    exact-scalars rule. Every field must now be validated the way every other value object
    here validates its own constructor arguments.
    """

    def _kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "name": "spider_fusion",
            "pattern": FusionPattern(),
            "builder": spider_fusion_builder,
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
