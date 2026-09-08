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

"""Phase 8 induction module unit suite: error taxonomy, obligation construction, base-value
selection, the discharge ladder, the result shape, and the multiplicity setter.

Covers PHASE8_SPEC.md sections 10, 12, 13 and 14.2.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.diagram.bangbox import (
    BangBox,
    BangBoxDomainError,
    BangBoxGrammarError,
    Mult,
    abstract_port_count,
    abstract_subgraph_count,
    free_mult_symbols,
    instantiate_symbol,
    peel_one,
)
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    GraphGrammarError,
    NodeId,
    PortRef,
)
from qufzx.diagram.validate import IssueKind, validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rule import BuildResult, Match, Rule
from qufzx.rewrite.rules_library import SPIDER_FUSION, spider_fusion_builder
from qufzx.semantics import induction
from qufzx.semantics.certificate import compare_structure
from qufzx.semantics.check import compare, score
from qufzx.semantics.contract_numeric import ContractSizeError
from qufzx.semantics.induction import (
    InductionDomainError,
    InductionError,
    InductionGrammarError,
    InductionResult,
    InductionUnavailableError,
    StepDischarge,
    TierOutcome,
    Verdict,
    build_obligation,
    choose_base_value,
    choose_index,
    rename_multiplicity,
    successor_diagram,
)

from . import test_phase7_oracle as T7
from . import test_phase8_oracle as T8
from .helpers import build_ghz_with_copy


def _build_boxed_fusion_family() -> tuple[Diagram, Diagram]:
    """The boxed A-into-B example before and after one spider fusion, over the symbolic
    dimension ``d`` and the multiplicity symbol ``m``."""
    pre, a_id, b_id = build_ghz_with_copy(Dim("d"))
    pre, _box_id, _m = abstract_subgraph_count(pre, frozenset({a_id, b_id}), 1, stem="m")
    matches = find_matches(pre)
    assert len(matches) == 1, "expected exactly one fusion match on the boxed A-into-B example"
    post = apply(pre, SPIDER_FUSION, matches[0]).diagram
    return pre, post


def _build_plain_two_spider_diagram() -> Diagram:
    """The A-into-B example with no bang box, over the symbolic dimension ``d``."""
    diagram, _a_id, _b_id = build_ghz_with_copy(Dim("d"))
    return diagram


def _build_boxed_ghz_family(dim_stem: str = "d") -> Diagram:
    """One phased Z spider whose single output leg is port-scope boxed under ``n``."""
    d = Dim(dim_stem)
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(node, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    diagram, _box_id, _n = abstract_port_count(diagram, ref, 1, stem="n")
    return diagram


def _build_boxed_product_family(dim_stem: str = "d") -> Diagram:
    """One phased Z spider node-scope boxed under ``n``, giving ``n`` independent single-leg
    spiders."""
    d = Dim(dim_stem)
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, 0)])
    diagram, _box_id, _n = abstract_subgraph_count(diagram, frozenset({node}), 1, stem="n")
    return diagram


def _build_phaseless_boxed_family() -> Diagram:
    """One phaseless Z spider whose single output leg is port-scope boxed under ``n``, so the
    instance at ``n = 0`` is a legless phaseless node."""
    d = Dim("d")
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=None)
    ref = PortRef(node, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    diagram, _box_id, _n = abstract_port_count(diagram, ref, 1, stem="n")
    return diagram


def _build_non_bare_multiplicity_diagram() -> Diagram:
    """One phased Z spider node-scope boxed under the non-bare multiplicity ``m + 1``."""
    d = Dim("d")
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, 0)])
    diagram.add_bang_box(Mult("m") + 1, node_scope=frozenset({node}))
    return diagram


def _build_nested_shared_symbol_family() -> Diagram:
    """One phased Z spider port-scope boxed inside a node-scope box, both boxes carrying the bare
    multiplicity ``m``."""
    d = Dim("d")
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(node, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    diagram, inner_id, _m = abstract_port_count(diagram, ref, 1, stem="m")
    diagram, outer_id, _m2 = abstract_subgraph_count(diagram, frozenset({node}), 1, stem="m2")
    inner = diagram.bang_boxes[inner_id]
    diagram.remove_bang_box(inner_id)
    diagram.add_bang_box(inner.multiplicity, port_scope=inner.port_scope, parent=outer_id)
    diagram.set_bang_box_multiplicity(outer_id, Mult("m"))
    return diagram


class TestErrorTaxonomy:
    """The module's own exceptions, and the foreign ones it must let through unwrapped."""

    def test_the_module_errors_share_one_root_inheriting_exception(self) -> None:
        assert InductionError.__bases__ == (Exception,)
        assert issubclass(InductionDomainError, InductionError)
        assert issubclass(InductionGrammarError, InductionError)
        assert issubclass(InductionUnavailableError, InductionError)

    @pytest.mark.parametrize("base", [2, -1])
    def test_a_base_index_other_than_zero_or_one_is_rejected(self, base: int) -> None:
        pre, post = _build_boxed_fusion_family()
        with pytest.raises(InductionDomainError, match="base index"):
            build_obligation(pre, post, base=base, witness={"d": 2})

    @pytest.mark.parametrize("width", [0, -1])
    def test_a_non_positive_window_width_is_rejected(self, width: int) -> None:
        pre, post = _build_boxed_fusion_family()
        obligation = build_obligation(pre, post, witness={"d": 2})
        with pytest.raises(InductionDomainError, match="window width"):
            induction.discharge_oracle_window(obligation, width=width)

    def test_a_diagram_with_no_bang_box_is_rejected(self) -> None:
        diagram = _build_plain_two_spider_diagram()
        with pytest.raises(InductionGrammarError, match="no bang box"):
            choose_index(diagram, diagram)

    def test_an_unknown_multiplicity_symbol_is_rejected(self) -> None:
        pre, post = _build_boxed_fusion_family()
        with pytest.raises(InductionGrammarError, match="no live bang box"):
            choose_index(pre, post, index="nonexistent")

    def test_a_non_bare_multiplicity_symbol_is_rejected(self) -> None:
        diagram = _build_non_bare_multiplicity_diagram()
        with pytest.raises(InductionDomainError, match="bare multiplicity symbol"):
            rename_multiplicity(diagram, "m", Mult("k"))

    def test_an_unbound_free_symbol_is_rejected_before_the_oracle_sees_it(self) -> None:
        pre, post = _build_boxed_fusion_family()
        with pytest.raises(InductionGrammarError, match="uninstantiated"):
            build_obligation(pre, post)

    def test_symbolic_contraction_falls_through_on_a_variable_rank(self) -> None:
        pre, post = _build_boxed_fusion_family()
        obligation = build_obligation(pre, post, witness={"d": 2})
        outcome = induction.discharge_symbolic_contraction(obligation)
        assert not outcome.settled
        assert "out of scope for Phase 9" in outcome.reason

    def test_symbolic_contraction_is_in_the_default_ladder_before_the_oracle_window(
        self,
    ) -> None:
        ladder = list(induction.DEFAULT_LADDER)
        assert (
            ladder.index(induction.StepDischarge.SYMBOLIC_CONTRACTION)
            == ladder.index(induction.StepDischarge.INDUCTION_REWRITE) + 1
        )
        assert ladder.index(induction.StepDischarge.SYMBOLIC_CONTRACTION) < ladder.index(
            induction.StepDischarge.ORACLE_WINDOW
        )

    def test_an_oversized_instantiation_propagates_the_contract_size_error(self) -> None:
        family = _build_boxed_ghz_family()
        obligation = build_obligation(family, family, witness={"d": 3})
        with pytest.raises(ContractSizeError):
            induction.discharge_oracle_window(obligation, max_elements=2)

    def test_an_unbound_dimension_is_refused_before_any_contraction(self) -> None:
        pre, post = _build_boxed_fusion_family()
        with pytest.raises(induction.InductionGrammarError, match="uninstantiated"):
            induction.prove_by_induction(pre, post)

    def test_a_base_forced_onto_an_unavailable_instance_raises_an_induction_error(
        self,
    ) -> None:
        """A named base with no evaluable instance fails as this module's own error."""
        family = _build_phaseless_boxed_family()
        with pytest.raises(InductionDomainError, match="no evaluable instance"):
            induction.prove_by_induction(family, family, base=0, witness={"d": 2})


class TestObligationConstruction:
    """Renaming a multiplicity index, and the six diagrams an obligation is built from."""

    def test_rename_multiplicity_renames_every_box_carrying_the_index(self) -> None:
        diagram = _build_nested_shared_symbol_family()
        renamed = rename_multiplicity(diagram, "m", Mult("k"))
        carried = [box.multiplicity for box in renamed.bang_boxes.values()]
        assert all(mult.is_bare_symbol for mult in carried)
        assert {mult.bare_symbol_name() for mult in carried} == {"k"}

    def test_rename_multiplicity_drops_the_consumed_parameter_binding(self) -> None:
        diagram = _build_nested_shared_symbol_family()
        renamed = rename_multiplicity(diagram, "m", Mult("k"))
        assert "m" in diagram.parameters
        assert "m" not in renamed.parameters

    def test_rename_multiplicity_leaves_the_input_diagram_untouched(self) -> None:
        pre, _post = _build_boxed_fusion_family()
        before = pre.copy()
        rename_multiplicity(pre, "m", Mult("k"))
        assert compare_structure(pre, before).identical

    def test_the_successor_diagram_carries_a_non_bare_multiplicity(self) -> None:
        pre, _post = _build_boxed_fusion_family()
        successor = successor_diagram(pre, "m")
        for box in successor.bang_boxes.values():
            assert not box.multiplicity.is_bare_symbol

    def test_the_successor_diagram_validates(self) -> None:
        pre, _post = _build_boxed_fusion_family()
        successor = successor_diagram(pre, "m")
        assert validate(successor).errors == ()

    def test_a_step_symbol_colliding_with_a_free_symbol_is_renamed_fresh(self) -> None:
        family = _build_boxed_ghz_family("k")
        obligation = build_obligation(family, family, witness={"k": 2})
        assert obligation.step_symbol != "k"

    def test_the_obligation_records_every_other_symbol_as_held_symbolic(self) -> None:
        pre, post = _build_boxed_fusion_family()
        obligation = build_obligation(pre, post, witness={"d": 2})
        assert "d" in obligation.held_symbolic


class TestBaseCase:
    """Which multiplicity the base is taken at, and what a failure there reports."""

    def test_the_base_is_chosen_at_zero_when_zero_evaluates(self) -> None:
        pre, post = _build_boxed_fusion_family()
        assert choose_base_value(pre, post, "m", {"d": 2}) == 0

    def test_the_base_falls_back_to_one_when_zero_is_structurally_unavailable(self) -> None:
        left = _build_phaseless_boxed_family()
        right = _build_phaseless_boxed_family()
        assert choose_base_value(left, right, "n", {"d": 2}) == 1

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_a_mismatch_at_zero_does_not_advance_the_base_to_one(self, d_value: int) -> None:
        left = _build_boxed_ghz_family()
        right = _build_boxed_product_family()
        at_zero = compare(left, right, {"n": 0, "d": d_value})
        assert not at_zero.matched
        assert at_zero.max_abs_deviation > 0
        at_one = compare(left, right, {"n": 1, "d": d_value})
        assert at_one.matched, at_one.reason
        assert choose_base_value(left, right, "n", {"d": d_value}) == 0
        result = induction.prove_by_induction(left, right, witness={"d": d_value})
        assert result.verdict is Verdict.REFUTED, result.reason

    def test_a_family_false_at_the_base_reports_the_base_as_the_counterexample(self) -> None:
        left = _build_boxed_ghz_family()
        right = _build_boxed_product_family()
        result = induction.prove_by_induction(left, right, witness={"d": 2})
        assert result.counterexample == result.base_value

    def test_a_base_failure_short_circuits_before_any_tier_runs(self) -> None:
        left = _build_boxed_ghz_family()
        right = _build_boxed_product_family()
        result = induction.prove_by_induction(left, right, witness={"d": 2})
        assert result.tiers == ()


def _ladder_build_false_near_identity() -> tuple[Diagram, Diagram]:
    """The bang-boxed GHZ family against the boxed product family, over a symbolic ``d``.

    The two agree at ``n = 1`` and diverge at ``n = 2``.
    """
    d = Dim("d")

    left = Diagram()
    left_spider = left.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    left_ref = PortRef(left_spider, Direction.OUTPUT, 0)
    left.set_boundary_outputs([left_ref])
    left, _left_box, _left_n = abstract_port_count(left, left_ref, 1, stem="n")

    right = Diagram()
    right_spider = right.add_node(
        Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {})
    )
    right.set_boundary_outputs([PortRef(right_spider, Direction.OUTPUT, 0)])
    right, _right_box, _right_n = abstract_subgraph_count(
        right, frozenset({right_spider}), 1, stem="n"
    )
    return left, right


def _ladder_box_killing_builder(diagram: Diagram, match: Match) -> BuildResult:
    """``spider_fusion_builder`` followed by the removal of every live bang box."""
    built = spider_fusion_builder(diagram, match)
    for box_id in sorted(built.diagram.bang_boxes):
        built.diagram.remove_bang_box(box_id)
    return built


def _ladder_symbol_consuming_rule() -> Rule:
    """Spider fusion rebuilt with a builder that drops the multiplicity symbol."""
    return dataclasses.replace(
        SPIDER_FUSION, name="ladder_symbol_consuming_fusion", builder=_ladder_box_killing_builder
    )


def _ladder_uniform_outcome(result: InductionResult) -> TierOutcome:
    """The uniform-rewrite tier's outcome in ``result.tiers``."""
    for outcome in result.tiers:
        if outcome.discharge is StepDischarge.UNIFORM_REWRITE:
            return outcome
    raise AssertionError(f"no uniform-rewrite tier in {[t.discharge for t in result.tiers]!r}")


class TestLadder:
    """Tier order, tier misses, and the verdicts each tier is allowed to produce."""

    def test_the_uniform_tier_is_tried_before_the_induction_tier(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2})
        assert result.verdict is Verdict.PROVED_UNIFORM, result.reason

    def test_a_tier_that_consumes_the_multiplicity_symbol_does_not_settle(self) -> None:
        pre, post = _build_boxed_fusion_family()
        obligation = build_obligation(pre, post, witness={"d": 2})
        outcome = induction.discharge_uniform_rewrite(
            obligation, rules=[_ladder_symbol_consuming_rule()]
        )
        assert outcome.settled is False, outcome.reason
        assert outcome.reason == "a step consumed the multiplicity symbol"

    def test_an_oracle_window_only_verdict_is_not_a_proof(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(
            pre, post, witness={"d": 2}, ladder=(StepDischarge.ORACLE_WINDOW,)
        )
        assert result.verdict is Verdict.SCHEMA_CHECKED, result.reason
        assert result.proved is False

    def test_the_schema_checked_reason_says_it_is_not_a_proof(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(
            pre, post, witness={"d": 2}, ladder=(StepDischarge.ORACLE_WINDOW,)
        )
        assert "not a proof" in result.reason

    def test_a_schema_checked_result_emits_no_certificate(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(
            pre, post, witness={"d": 2}, ladder=(StepDischarge.ORACLE_WINDOW,)
        )
        assert result.verdict is Verdict.SCHEMA_CHECKED, result.reason
        assert result.derivation is None

    def test_an_exhausted_ladder_is_inconclusive_not_refuted(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(
            pre, post, witness={"d": 2}, ladder=(StepDischarge.UNIFORM_REWRITE,), rules=()
        )
        assert result.verdict is Verdict.INCONCLUSIVE, result.reason

    def test_the_step_budget_terminates_a_non_closing_rewrite(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(
            pre, post, witness={"d": 2}, ladder=(StepDischarge.UNIFORM_REWRITE,), max_steps=1
        )
        outcome = _ladder_uniform_outcome(result)
        assert outcome.settled is False, outcome.reason
        assert outcome.reason == "step budget 1 exhausted"


class TestResultShape:
    """The invariants every :class:`~qufzx.semantics.induction.InductionResult` carries."""

    def test_the_result_is_frozen(self) -> None:
        pre, post = _build_boxed_fusion_family()
        obligation = build_obligation(pre, post, witness={"d": 2})
        result = InductionResult(
            verdict=Verdict.PROVED_UNIFORM,
            index=obligation.index,
            base_value=obligation.base_value,
            reason="",
            obligation=obligation,
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.verdict = Verdict.REFUTED  # type: ignore[misc]

    def test_a_refuted_result_always_carries_a_reason_and_a_counterexample(self) -> None:
        left, right = _ladder_build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result.verdict is Verdict.REFUTED, result.reason
        assert result.reason
        assert result.counterexample is not None

    def test_proved_is_true_for_exactly_the_two_proved_verdicts(self) -> None:
        pre, post = _build_boxed_fusion_family()
        obligation = build_obligation(pre, post, witness={"d": 2})
        proved = {Verdict.PROVED_UNIFORM, Verdict.PROVED_INDUCTION}
        for verdict in Verdict:
            result = InductionResult(
                verdict=verdict,
                index=obligation.index,
                base_value=obligation.base_value,
                reason="",
                obligation=obligation,
            )
            assert result.proved is (verdict in proved), verdict

    def test_held_symbolic_is_empty_for_a_window_only_verdict(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(
            pre, post, witness={"d": 2}, ladder=(StepDischarge.ORACLE_WINDOW,)
        )
        assert result.verdict is Verdict.SCHEMA_CHECKED, result.reason
        assert result.held_symbolic == frozenset()


def _mult_node_scope_box() -> BangBox:
    """A node-scope bang box over one node, under the bare symbol ``m``."""
    return BangBox(
        id=BangBoxId(0),
        multiplicity=Mult.symbol("m"),
        node_scope=frozenset({NodeId(0)}),
        parent=None,
    )


def _mult_port_scope_box() -> BangBox:
    """A port-scope bang box over one output port, under the bare symbol ``m``."""
    ref = PortRef(NodeId(0), Direction.OUTPUT, 0)
    return BangBox(
        id=BangBoxId(0),
        multiplicity=Mult.symbol("m"),
        port_scope=frozenset({ref}),
        parent=None,
    )


def _mult_single_spider_diagram() -> tuple[Diagram, NodeId, PortRef]:
    """One Z spider with a single boundary output leg, at the symbolic dimension ``d``."""
    d = Dim("d")
    diagram = Diagram()
    node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(node_id, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    return diagram, node_id, ref


def _mult_port_scope_diagram() -> tuple[Diagram, BangBoxId]:
    """One spider whose single output leg carries a port-scope box. Returns the box id."""
    diagram, _node_id, ref = _mult_single_spider_diagram()
    box_id = diagram.add_bang_box(Mult.symbol("k2"), port_scope=frozenset({ref}), parent=None)
    return diagram, box_id


def _mult_nested_family() -> tuple[Diagram, BangBoxId, BangBoxId]:
    """Two boundary spiders under a node-scope outer box, one of them under a node-scope inner
    box parented to it. Returns the diagram with the outer and inner box ids."""
    d = Dim("d")
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    b_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    diagram.set_boundary_outputs(
        [PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.OUTPUT, 0)]
    )
    outer_id = diagram.add_bang_box(
        Mult.symbol("k1"), node_scope=frozenset({a_id, b_id}), parent=None
    )
    inner_id = diagram.add_bang_box(
        Mult.symbol("k2"), node_scope=frozenset({a_id}), parent=outer_id
    )
    return diagram, outer_id, inner_id


class TestMultiplicitySetter:
    """``BangBox.with_multiplicity`` and ``Diagram.set_bang_box_multiplicity`` replace a
    multiplicity in place of the box, leaving id, scope, parent and children untouched."""

    def test_a_bare_symbol_is_replaced_by_a_concrete_count(self) -> None:
        replaced = _mult_node_scope_box().with_multiplicity(Mult(3))
        assert replaced.multiplicity == Mult(3)

    def test_a_bare_symbol_is_replaced_by_a_compound_multiplicity(self) -> None:
        replaced = _mult_node_scope_box().with_multiplicity(Mult("k") + 1)
        assert replaced.multiplicity == Mult("k") + 1
        assert not replaced.multiplicity.is_bare_symbol

    def test_a_symbol_is_replaced_by_another_symbol(self) -> None:
        replaced = _mult_node_scope_box().with_multiplicity(Mult.symbol("n"))
        assert replaced.multiplicity == Mult.symbol("n")
        assert replaced.multiplicity.is_bare_symbol

    def test_with_multiplicity_returns_a_new_box_and_does_not_mutate_the_receiver(self) -> None:
        box = _mult_node_scope_box()
        replaced = box.with_multiplicity(Mult(3))
        assert replaced is not box
        assert box.multiplicity == Mult.symbol("m")

    def test_with_multiplicity_preserves_every_other_field(self) -> None:
        box = BangBox(
            id=BangBoxId(7),
            multiplicity=Mult.symbol("m"),
            node_scope=frozenset({NodeId(1), NodeId(2)}),
            parent=BangBoxId(4),
        )
        replaced = box.with_multiplicity(Mult(3))
        assert replaced.id == box.id
        assert replaced.node_scope == box.node_scope
        assert replaced.port_scope == box.port_scope
        assert replaced.parent == box.parent

    def test_with_multiplicity_round_trips_a_port_scope_box(self) -> None:
        box = _mult_port_scope_box()
        replaced = box.with_multiplicity(Mult(2))
        assert replaced.port_scope == box.port_scope
        assert replaced.node_scope == frozenset()
        assert replaced.multiplicity == Mult(2)

    @pytest.mark.parametrize("build", [_mult_node_scope_box, _mult_port_scope_box])
    def test_with_multiplicity_preserves_the_scope_invariant(
        self, build: Callable[[], BangBox]
    ) -> None:
        replaced = build().with_multiplicity(Mult(2))
        assert bool(replaced.node_scope) != bool(replaced.port_scope)

    def test_with_multiplicity_cannot_smuggle_a_non_mult_past_post_init(self) -> None:
        box = _mult_node_scope_box()
        with pytest.raises(BangBoxGrammarError):
            box.with_multiplicity("not a mult")  # type: ignore[arg-type]

    def test_the_setter_keeps_the_box_id(self) -> None:
        diagram, outer_id, _inner_id = _mult_nested_family()
        before = set(diagram.bang_boxes)
        diagram.set_bang_box_multiplicity(outer_id, Mult(3))
        assert set(diagram.bang_boxes) == before
        assert diagram.bang_boxes[outer_id].multiplicity == Mult(3)

    def test_the_setter_leaves_every_other_field_unchanged(self) -> None:
        diagram, outer_id, _inner_id = _mult_nested_family()
        before = diagram.bang_boxes[outer_id]
        diagram.set_bang_box_multiplicity(outer_id, Mult(3))
        after = diagram.bang_boxes[outer_id]
        assert after.id == before.id
        assert after.node_scope == before.node_scope
        assert after.port_scope == before.port_scope
        assert after.parent == before.parent

    def test_the_setter_round_trips_a_port_scope_box(self) -> None:
        diagram, box_id = _mult_port_scope_diagram()
        before = diagram.bang_boxes[box_id]
        diagram.set_bang_box_multiplicity(box_id, Mult(2))
        after = diagram.bang_boxes[box_id]
        assert after.multiplicity == Mult(2)
        assert after.port_scope == before.port_scope
        assert after.node_scope == frozenset()
        assert after.parent == before.parent

    def test_resetting_a_parent_multiplicity_does_not_orphan_its_child(self) -> None:
        diagram, outer_id, inner_id = _mult_nested_family()
        diagram.set_bang_box_multiplicity(outer_id, Mult(3))

        child_parent = diagram.bang_boxes[inner_id].parent
        assert child_parent == outer_id
        assert child_parent in diagram.bang_boxes

        report = validate(diagram)
        orphaned = [
            issue for issue in report.issues if issue.kind is IssueKind.BANGBOX_UNKNOWN_PARENT
        ]
        assert orphaned == []
        assert report.errors == ()

    def test_the_setter_rejects_an_unknown_box_id(self) -> None:
        diagram, _outer_id, _inner_id = _mult_nested_family()
        with pytest.raises(GraphGrammarError):
            diagram.set_bang_box_multiplicity(BangBoxId(999), Mult(3))

    def test_the_setter_rejects_a_non_mult_multiplicity(self) -> None:
        diagram, outer_id, _inner_id = _mult_nested_family()
        with pytest.raises(GraphGrammarError):
            diagram.set_bang_box_multiplicity(outer_id, "not a mult")


class TestPeelCommutesWithInstantiate:
    """peel_one is the step case's primitive: it must not change what the family denotes."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize(
        "build,symbol",
        [
            (T7._build_ghz_family, "n"),
            (T7._build_nested_two_index_family, "k1"),
        ],
    )
    def test_peel_then_instantiate_at_k_equals_instantiate_at_k_plus_one(
        self, build: Callable[[int], Diagram], symbol: str, k: int
    ) -> None:
        direct = score(instantiate_symbol(build(2), symbol, k + 1), {}).tensor
        successor = induction.successor_diagram(build(2), symbol)
        step_symbol = min(free_mult_symbols(successor))
        peeled = peel_one(successor, min(successor.bang_boxes))
        assert peeled.separable == bool(peeled.copy_node_ids)
        via_peel = score(instantiate_symbol(peeled.diagram, step_symbol, k), {}).tensor
        assert direct.shape == via_peel.shape
        assert np.allclose(direct, via_peel)

    def test_a_bare_symbol_multiplicity_cannot_be_peeled(self) -> None:
        family = T7._build_ghz_family(2)
        with pytest.raises(BangBoxDomainError):
            peel_one(family, min(family.bang_boxes))


class TestStepCaseIsReachableAndSound:
    """The k to k+1 tier settles a true family and never settles a false one."""

    def test_a_nested_family_is_proved_by_induction(self) -> None:
        left = T7._build_nested_two_index_family(2)
        right = T7._build_nested_two_index_family(2)
        result = induction.prove_by_induction(
            left, right, witness={"d": 2}, ladder=(induction.StepDischarge.INDUCTION_REWRITE,)
        )
        assert result.verdict is induction.Verdict.PROVED_INDUCTION
        assert result.proved

    def test_the_step_tier_refuses_a_false_near_identity(self) -> None:
        left, right = T8._build_false_near_identity()
        result = induction.prove_by_induction(
            left, right, witness={"d": 2}, ladder=(induction.StepDischarge.INDUCTION_REWRITE,)
        )
        assert not result.proved
        assert result.verdict is not induction.Verdict.PROVED_INDUCTION

    def test_the_step_tier_refuses_a_family_off_by_one_leg(self) -> None:
        left = T7._build_nested_two_index_family(2)
        right = T7._build_nested_two_index_family(3)
        result = induction.prove_by_induction(
            left, right, witness={"d": 2}, ladder=(induction.StepDischarge.INDUCTION_REWRITE,)
        )
        assert not result.proved
