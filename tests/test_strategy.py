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

"""Establishes the strategy layer: the termination guard, stop reasons, scalar accumulation,
and ``toward_normal_form``'s rule resolution."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.engine import (
    DEFAULT_GUARD,
    NORMAL_FORM_RULE_NAMES,
    StopReason,
    StrategyOutcome,
    TerminationGuard,
    TerminationGuardTripped,
    apply_until_fixpoint,
    missing_normal_form_rule_names,
    normal_form_rules,
    toward_normal_form,
)
from archytaszx.rewrite.rule import (
    BuildResult,
    DimensionConstraint,
    Match,
    Pattern,
    Quantifiers,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
)
from archytaszx.rewrite.rules_library import SPIDER_FUSION, ZX_CAP

from .helpers import build_ghz_with_copy


@dataclasses.dataclass(frozen=True, slots=True)
class _ScriptedMatch:
    """A minimal, hand-built ``Match`` the looping rules below are driven at."""

    side_condition_outcomes: tuple[SideConditionOutcome, ...] = ()
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        return True


class _AlwaysMatchPattern(Pattern):
    """A ``Pattern`` that reports one match against every diagram, forever."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        return (_ScriptedMatch(),)


def _no_op_builder(working: Diagram, match: Match) -> BuildResult:
    """A builder that consumes nothing, creates nothing and leaves the diagram as it found it."""
    return BuildResult(
        diagram=working,
        new_node_ids=(),
        consumed_node_ids=(),
        consumed_wires=(),
        port_mapping={},
        scalar_introduced=Scalar.one(),
    )


def _scaling_builder(working: Diagram, match: Match) -> BuildResult:
    """A no-op builder that nonetheless reports a factor of two."""
    return BuildResult(
        diagram=working,
        new_node_ids=(),
        consumed_node_ids=(),
        consumed_wires=(),
        port_mapping={},
        scalar_introduced=Scalar.rational(2),
    )


LOOPING_RULE = Rule(
    name="scripted_no_op_loop",
    pattern=_AlwaysMatchPattern(),
    builder=_no_op_builder,
    side_conditions=(),
    quantifiers=Quantifiers(),
    scalar_introduced=Scalar.one(),
)
"""A rule that always matches and always reproduces its input exactly."""

SCALING_LOOPING_RULE = Rule(
    name="scripted_scaling_loop",
    pattern=_AlwaysMatchPattern(),
    builder=_scaling_builder,
    side_conditions=(),
    quantifiers=Quantifiers(),
    scalar_introduced=Scalar.rational(2),
)
"""A rule that always matches and changes nothing but the scalar."""


def _two_spiders() -> Diagram:
    """The GHZ-with-copy worked example at ``d = 2``."""
    diagram, _a_id, _b_id = build_ghz_with_copy(Dim.concrete(2))
    return diagram


def _capped_state(dim: Dim) -> Diagram:
    """A phaseless Z state wired into a phaseless X effect."""
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    return diagram


class TestTerminationGuardShape:
    def test_default_guard_is_a_frozen_termination_guard(self) -> None:
        assert isinstance(DEFAULT_GUARD, TerminationGuard)
        assert DEFAULT_GUARD.max_steps > 0
        assert DEFAULT_GUARD.detect_loops is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_steps": -1},
            {"max_steps": True},
            {"max_steps": 4, "max_nodes": -1},
            {"max_steps": 4, "max_nodes": 1.5},
            {"max_steps": 4, "detect_loops": "yes"},
        ],
    )
    def test_an_ill_shaped_guard_is_a_grammar_error(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(RewriteGrammarError):
            TerminationGuard(**kwargs)  # type: ignore[arg-type]

    def test_a_non_guard_is_a_grammar_error(self) -> None:
        with pytest.raises(RewriteGrammarError, match="must be a TerminationGuard"):
            apply_until_fixpoint(_two_spiders(), [], guard=object())  # type: ignore[arg-type]


class TestFixpoint:
    def test_no_matching_rule_is_an_immediate_fixpoint(self) -> None:
        diagram = _two_spiders()
        outcome = apply_until_fixpoint(diagram, [ZX_CAP])

        assert outcome.stop_reason is StopReason.FIXPOINT
        assert outcome.steps == ()
        assert outcome.steps_attempted == 1
        assert outcome.scalar_accumulated == Scalar.one()
        assert outcome.diagram is diagram

    def test_fusion_runs_to_a_single_spider(self) -> None:
        diagram = _two_spiders()
        outcome = apply_until_fixpoint(diagram, [SPIDER_FUSION])

        assert outcome.stop_reason is StopReason.FIXPOINT
        assert len(outcome.steps) == 1
        assert outcome.steps_attempted == 2
        assert len(outcome.diagram.nodes) == 1
        assert len(diagram.nodes) == 2

    def test_strict_does_not_raise_at_a_fixpoint(self) -> None:
        outcome = apply_until_fixpoint(_two_spiders(), [SPIDER_FUSION], strict=True)
        assert outcome.stop_reason is StopReason.FIXPOINT


class TestLoopDetection:
    """A deliberately looping strategy: a rule that always matches and always reproduces its
    input."""

    def test_a_reproduced_diagram_stops_the_run(self) -> None:
        diagram = _two_spiders()
        guard = TerminationGuard(max_steps=1000, detect_loops=True)
        outcome = apply_until_fixpoint(diagram, [LOOPING_RULE], guard=guard)

        assert outcome.stop_reason is StopReason.LOOP_DETECTED
        assert len(outcome.steps) == 1
        assert outcome.steps_attempted == 1
        assert outcome.steps[0].rule_name == "scripted_no_op_loop"

    def test_strict_raises_carrying_the_same_outcome(self) -> None:
        diagram = _two_spiders()
        guard = TerminationGuard(max_steps=1000, detect_loops=True)

        with pytest.raises(TerminationGuardTripped) as excinfo:
            apply_until_fixpoint(diagram, [LOOPING_RULE], guard=guard, strict=True)

        outcome = excinfo.value.outcome
        assert isinstance(outcome, StrategyOutcome)
        assert outcome.stop_reason is StopReason.LOOP_DETECTED
        assert len(outcome.steps) == 1
        assert "loop_detected" in str(excinfo.value)

    def test_detection_disabled_falls_through_to_the_step_ceiling(self) -> None:
        guard = TerminationGuard(max_steps=3, detect_loops=False)
        outcome = apply_until_fixpoint(_two_spiders(), [LOOPING_RULE], guard=guard)

        assert outcome.stop_reason is StopReason.STEP_LIMIT
        assert len(outcome.steps) == 3
        assert outcome.steps_attempted == 4

    def test_a_cycle_that_moves_the_scalar_hits_the_step_ceiling_instead(self) -> None:
        guard = TerminationGuard(max_steps=3, detect_loops=True)
        outcome = apply_until_fixpoint(_two_spiders(), [SCALING_LOOPING_RULE], guard=guard)

        assert outcome.stop_reason is StopReason.STEP_LIMIT
        assert outcome.scalar_accumulated == Scalar.rational(8)


class TestStepAndNodeCeilings:
    def test_a_zero_step_budget_stops_before_any_rewrite(self) -> None:
        diagram = _two_spiders()
        guard = TerminationGuard(max_steps=0)
        outcome = apply_until_fixpoint(diagram, [SPIDER_FUSION], guard=guard)

        assert outcome.stop_reason is StopReason.STEP_LIMIT
        assert outcome.steps == ()
        assert outcome.diagram is diagram

    def test_strict_raises_on_the_step_ceiling(self) -> None:
        guard = TerminationGuard(max_steps=0)
        with pytest.raises(TerminationGuardTripped, match="step_limit"):
            apply_until_fixpoint(_two_spiders(), [SPIDER_FUSION], guard=guard, strict=True)

    def test_the_node_ceiling_stops_the_run(self) -> None:
        diagram = _two_spiders()
        guard = TerminationGuard(max_steps=10, max_nodes=1)
        outcome = apply_until_fixpoint(diagram, [SPIDER_FUSION], guard=guard)

        assert outcome.stop_reason is StopReason.NODE_LIMIT
        assert outcome.steps == ()

    def test_the_node_ceiling_admits_a_diagram_within_it(self) -> None:
        guard = TerminationGuard(max_steps=10, max_nodes=2)
        outcome = apply_until_fixpoint(_two_spiders(), [SPIDER_FUSION], guard=guard)

        assert outcome.stop_reason is StopReason.FIXPOINT
        assert len(outcome.steps) == 1

    def test_strict_raises_on_the_node_ceiling(self) -> None:
        guard = TerminationGuard(max_steps=10, max_nodes=1)
        with pytest.raises(TerminationGuardTripped, match="node_limit"):
            apply_until_fixpoint(_two_spiders(), [SPIDER_FUSION], guard=guard, strict=True)


class TestScalarAccumulation:
    """``scalar_accumulated`` is the exact product of the steps' factors, and that product is
    the delta between the input and output diagram scalars."""

    @pytest.mark.parametrize("value", [2, 3, 5])
    def test_the_cap_factor_is_the_scalar_delta(self, value: int) -> None:
        dim = Dim.concrete(value)
        diagram = _capped_state(dim)
        outcome = apply_until_fixpoint(diagram, [ZX_CAP])

        assert outcome.stop_reason is StopReason.FIXPOINT
        assert len(outcome.steps) == 1
        assert outcome.scalar_accumulated == Scalar.dim_power(dim, 1, 2)
        assert outcome.diagram.scalar == diagram.scalar * outcome.scalar_accumulated

    def test_the_accumulated_scalar_is_the_product_of_every_step(self) -> None:
        guard = TerminationGuard(max_steps=4, detect_loops=False)
        diagram = _two_spiders()
        outcome = apply_until_fixpoint(diagram, [SCALING_LOOPING_RULE], guard=guard)

        product = Scalar.one()
        for step in outcome.steps:
            product = product * step.scalar_introduced

        assert outcome.scalar_accumulated == product
        assert outcome.diagram.scalar == diagram.scalar * outcome.scalar_accumulated

    def test_an_empty_run_accumulates_one(self) -> None:
        diagram = _two_spiders()
        outcome = apply_until_fixpoint(diagram, [ZX_CAP])

        assert outcome.scalar_accumulated == Scalar.one()
        assert outcome.diagram.scalar == diagram.scalar


class TestNormalFormRuleResolution:
    """``toward_normal_form``'s rule set is resolved at call time, and a name not yet
    registered is skipped rather than raising."""

    def test_bialgebra_is_not_in_the_set(self) -> None:
        assert "bialgebra" not in NORMAL_FORM_RULE_NAMES

    def test_the_declared_order_is_the_spec_order(self) -> None:
        assert NORMAL_FORM_RULE_NAMES == (
            "identity_removal",
            "triangle_inverse_cancellation",
            "fourier_cancellation",
            "fourier_state_color_change",
            "spider_fusion",
            "hopf",
            "state_copy",
            "zx_cap",
        )

    def test_resolved_and_skipped_names_partition_the_declared_tuple(self) -> None:
        resolved = tuple(rule.name for rule in normal_form_rules())
        missing = missing_normal_form_rule_names()

        assert set(resolved).isdisjoint(missing)
        assert set(resolved) | set(missing) == set(NORMAL_FORM_RULE_NAMES)
        assert len(resolved) + len(missing) == len(NORMAL_FORM_RULE_NAMES)

    def test_resolved_rules_keep_the_declared_relative_order(self) -> None:
        resolved = [rule.name for rule in normal_form_rules()]
        declared = [name for name in NORMAL_FORM_RULE_NAMES if name in resolved]

        assert resolved == declared

    def test_every_resolved_name_is_a_registered_rule(self) -> None:
        from archytaszx.rewrite.rules_library import RULES

        for rule in normal_form_rules():
            assert RULES[rule.name] is rule

    def test_the_set_now_resolves_complete(self) -> None:
        assert missing_normal_form_rule_names() == ()
        assert len(normal_form_rules()) == len(NORMAL_FORM_RULE_NAMES)

    def test_every_rule_in_the_set_is_non_growing_at_a_match(self) -> None:
        from archytaszx.rewrite.rules_library import RULES

        assert RULES["bialgebra"] not in normal_form_rules()

    def test_a_non_unit_rule_scalar_reaches_the_accumulator(self) -> None:
        from archytaszx.rewrite.rules_library import HOPF

        assert not HOPF.scalar_introduced.is_one

    def test_no_missing_name_is_registered(self) -> None:
        from archytaszx.rewrite.rules_library import RULES

        for name in missing_normal_form_rule_names():
            assert name not in RULES


class TestTowardNormalForm:
    def test_a_spider_chain_collapses_to_one_node(self) -> None:
        dim = Dim.concrete(2)
        diagram = Diagram()
        head = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        middle = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        tail = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(PortRef(head, Direction.OUTPUT, 0), PortRef(middle, Direction.INPUT, 0))
        diagram.add_wire(PortRef(middle, Direction.OUTPUT, 0), PortRef(tail, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(tail, Direction.OUTPUT, 0)])

        outcome = toward_normal_form(diagram)

        assert outcome.stop_reason is StopReason.FIXPOINT
        assert outcome.steps
        assert len(outcome.diagram.nodes) == 1
        (survivor,) = outcome.diagram.nodes.values()
        assert survivor.num_inputs == 0
        assert survivor.num_outputs == 1
        assert outcome.diagram.boundary_outputs == (
            PortRef(next(iter(outcome.diagram.nodes)), Direction.OUTPUT, 0),
        )

    def test_a_capped_state_is_driven_to_the_empty_diagram(self) -> None:
        dim = Dim.concrete(3)
        diagram = _capped_state(dim)

        outcome = toward_normal_form(diagram)

        assert outcome.stop_reason is StopReason.FIXPOINT
        assert outcome.diagram.nodes == {}
        assert outcome.diagram.wires == frozenset()
        assert outcome.diagram.scalar == Scalar.dim_power(dim, 1, 2)

    def test_every_step_names_a_rule_from_the_resolved_set(self) -> None:
        diagram = _capped_state(Dim.concrete(2))
        resolved = {rule.name for rule in normal_form_rules()}

        outcome = toward_normal_form(diagram)

        assert {step.rule_name for step in outcome.steps} <= resolved


class TestStrategyCrossProcessDeterminism:
    """A whole ``toward_normal_form`` run must not vary by ``PYTHONHASHSEED``.

    The same shape as ``test_engine.py::TestCrossProcessDeterminism``, one level up: the
    strategy layer chooses which rule and which match to fire at every iteration, and that
    choice must come out of the matchers' own deterministic orders rather than any set or
    dict iteration order.
    """

    SCRIPT = Path(__file__).parent / "_strategy_determinism_script.py"

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

    def test_the_whole_run_is_byte_identical_across_hash_seeds(self) -> None:
        first = self._run_with_seed("0")
        second = self._run_with_seed("2147483647")
        assert first, "the driver script printed nothing"
        assert second == first


def _renumbering_builder(working: Diagram, match: Match) -> BuildResult:
    """A builder that replaces the diagram's single Z spider with a structurally identical
    fresh one."""
    (old_id,) = tuple(working.nodes)
    old = working.nodes[old_id]
    new_id = working.add_node(
        Z_SPIDER,
        input_dims=[port.dim for port in old.inputs],
        output_dims=[port.dim for port in old.outputs],
    )
    port_mapping = {
        PortRef(old_id, Direction.INPUT, index): PortRef(new_id, Direction.INPUT, index)
        for index in range(old.num_inputs)
    }
    port_mapping.update(
        {
            PortRef(old_id, Direction.OUTPUT, index): PortRef(new_id, Direction.OUTPUT, index)
            for index in range(old.num_outputs)
        }
    )
    return BuildResult(
        diagram=working,
        new_node_ids=(new_id,),
        consumed_node_ids=(old_id,),
        consumed_wires=(),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


RENUMBERING_RULE = Rule(
    name="scripted_renumbering_loop",
    pattern=_AlwaysMatchPattern(),
    builder=_renumbering_builder,
    side_conditions=(),
    quantifiers=Quantifiers(),
    scalar_introduced=Scalar.one(),
)
"""A rule that always matches and rebuilds its input under fresh node ids."""


class TestRenumberingLoopDetection:
    def test_a_cycle_under_fresh_node_ids_stops_the_run(self) -> None:
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, 0)])
        guard = TerminationGuard(max_steps=25, detect_loops=True)

        outcome = apply_until_fixpoint(diagram, [RENUMBERING_RULE], guard=guard)

        assert outcome.stop_reason is StopReason.LOOP_DETECTED
        assert len(outcome.steps) == 1
