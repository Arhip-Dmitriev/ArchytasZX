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
