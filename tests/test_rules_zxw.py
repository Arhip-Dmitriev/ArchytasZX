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

"""The W, triangle and dimension-connective rules, each checked against the numeric oracle."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.normal_form import normal_form
from archytaszx.rewrite.rule import Rule
from archytaszx.rewrite.rules_library import (
    CONNECTIVE_INVERSE,
    CONNECTIVE_INVERSE_BIND_FIRST,
    CONNECTIVE_STATES,
    CONNECTIVE_STATES_SWAPPED,
    RULES,
    TRIANGLE_ZERO_EFFECT,
    TRIANGLE_ZERO_STATE,
    W_FUSION,
    W_IDENTITY,
    W_ZERO_COPY,
)
from archytaszx.semantics.certificate import certify, replay
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import DecisionMethod, EqualityVerdict, decide_equal

D = Dim("d")
S = Dim("s")
T = Dim("t")
SINGLE = ({"d": 2}, {"d": 3}, {"d": 4})
MIXED = ({"s": 2, "t": 3}, {"s": 3, "t": 2}, {"s": 2, "t": 2})


def out(node: int, index: int) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def inp(node: int, index: int) -> PortRef:
    return PortRef(node, Direction.INPUT, index)


def w_into_w(first: int = 3, position: int = 1, second: int = 2) -> Diagram:
    """A ``first``-output W whose output ``position`` feeds a ``second``-output W."""
    diagram = Diagram()
    a = diagram.add_node(W_NODE, [D], [D] * first)
    b = diagram.add_node(W_NODE, [D], [D] * second)
    diagram.add_wire(out(a, position), inp(b, 0))
    diagram.set_boundary_inputs([inp(a, 0)])
    rest = [out(a, k) for k in range(first) if k != position]
    diagram.set_boundary_outputs(rest[::-1] + [out(b, k) for k in range(second)])
    return diagram


def single_w(outputs: int) -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(W_NODE, [D], [D] * outputs)
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, k) for k in range(outputs)])
    return diagram


def w_after_z() -> Diagram:
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, [D], [D])
    w = diagram.add_node(W_NODE, [D], [D])
    diagram.add_wire(out(z, 0), inp(w, 0))
    diagram.set_boundary_inputs([inp(z, 0)])
    diagram.set_boundary_outputs([out(w, 0)])
    return diagram


def state_into(target: GeneratorType, outputs: int, phase: PhaseVector | None = None) -> Diagram:
    diagram = Diagram()
    x = diagram.add_node(X_SPIDER, [], [D], phase=phase)
    node = diagram.add_node(target, [D], [D] * outputs)
    diagram.add_wire(out(x, 0), inp(node, 0))
    diagram.set_boundary_outputs([out(node, k) for k in range(outputs)])
    return diagram


def triangle_into_effect(triangle: GeneratorType = TRIANGLE) -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(triangle, [D], [D])
    x = diagram.add_node(X_SPIDER, [D], [])
    diagram.add_wire(out(node, 0), inp(x, 0))
    diagram.set_boundary_inputs([inp(node, 0)])
    return diagram


def split_then_bind() -> Diagram:
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, [], [S * T, S * T])
    split = diagram.add_node(DIM_SPLITTER, [S * T], [S, T])
    bind = diagram.add_node(DIM_BINDER, [S, T], [S * T])
    diagram.add_wire(out(z, 0), inp(split, 0))
    diagram.add_wire(out(split, 0), inp(bind, 0))
    diagram.add_wire(out(split, 1), inp(bind, 1))
    diagram.set_boundary_outputs([out(bind, 0), out(z, 1)])
    return diagram


def bind_then_split() -> Diagram:
    diagram = Diagram()
    zs = diagram.add_node(Z_SPIDER, [], [S, S])
    xt = diagram.add_node(X_SPIDER, [T], [T])
    bind = diagram.add_node(DIM_BINDER, [S, T], [S * T])
    split = diagram.add_node(DIM_SPLITTER, [S * T], [S, T])
    diagram.add_wire(out(zs, 0), inp(bind, 0))
    diagram.add_wire(out(xt, 0), inp(bind, 1))
    diagram.add_wire(out(bind, 0), inp(split, 0))
    diagram.set_boundary_inputs([inp(xt, 0)])
    diagram.set_boundary_outputs([out(split, 1), out(zs, 1), out(split, 0)])
    return diagram


def connective_ends(colour: GeneratorType, effects: bool) -> Diagram:
    diagram = Diagram()
    if effects:
        split = diagram.add_node(DIM_SPLITTER, [S * T], [S, T])
        a = diagram.add_node(colour, [S], [])
        b = diagram.add_node(colour, [T], [])
        diagram.add_wire(out(split, 0), inp(a, 0))
        diagram.add_wire(out(split, 1), inp(b, 0))
        diagram.set_boundary_inputs([inp(split, 0)])
    else:
        a = diagram.add_node(colour, [], [S])
        b = diagram.add_node(colour, [], [T])
        bind = diagram.add_node(DIM_BINDER, [S, T], [S * T])
        diagram.add_wire(out(a, 0), inp(bind, 0))
        diagram.add_wire(out(b, 0), inp(bind, 1))
        diagram.set_boundary_outputs([out(bind, 0)])
    return diagram


CASES: dict[str, tuple[Rule, Callable[[], Diagram], int, tuple[dict[str, int], ...]]] = {
    "w_fusion": (W_FUSION, w_into_w, 1, SINGLE),
    "w_fusion_counit": (W_FUSION, lambda: w_into_w(3, 0, 0), 1, SINGLE),
    "w_fusion_binary": (W_FUSION, lambda: w_into_w(2, 1, 2), 1, SINGLE),
    "w_identity": (W_IDENTITY, w_after_z, 1, SINGLE),
    "w_zero_copy": (W_ZERO_COPY, lambda: state_into(W_NODE, 3), 3, SINGLE),
    "w_zero_copy_counit": (W_ZERO_COPY, lambda: state_into(W_NODE, 0), 0, SINGLE),
    "triangle_zero_state": (TRIANGLE_ZERO_STATE, lambda: state_into(TRIANGLE, 1), 1, SINGLE),
    "triangle_inverse_zero_state": (
        TRIANGLE_ZERO_STATE,
        lambda: state_into(TRIANGLE_INVERSE, 1),
        1,
        SINGLE,
    ),
    "triangle_zero_effect": (TRIANGLE_ZERO_EFFECT, triangle_into_effect, 1, SINGLE),
    "connective_inverse": (CONNECTIVE_INVERSE, split_then_bind, 1, MIXED),
    "connective_inverse_bind_first": (CONNECTIVE_INVERSE_BIND_FIRST, bind_then_split, 2, MIXED),
    "connective_states": (CONNECTIVE_STATES, lambda: connective_ends(Z_SPIDER, False), 1, MIXED),
    "connective_effects": (CONNECTIVE_STATES, lambda: connective_ends(Z_SPIDER, True), 1, MIXED),
    "connective_states_swapped": (
        CONNECTIVE_STATES_SWAPPED,
        lambda: connective_ends(X_SPIDER, False),
        1,
        MIXED,
    ),
    "connective_effects_swapped": (
        CONNECTIVE_STATES_SWAPPED,
        lambda: connective_ends(X_SPIDER, True),
        1,
        MIXED,
    ),
}


class TestEachRuleAgainstTheOracle:
    @pytest.mark.parametrize("name", sorted(CASES))
    def test_the_rewrite_preserves_the_map(self, name: str) -> None:
        rule, build, nodes_after, assignments = CASES[name]
        diagram = build()
        assert not validate(diagram).errors
        matches = rule.pattern.find_matches(diagram)
        assert matches, name
        result = apply(diagram, rule, matches[0])
        assert len(result.diagram.nodes) == nodes_after
        assert not validate(result.diagram).errors
        for assignment in assignments:
            assert compare(diagram, result.diagram, assignment).matched, assignment

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_the_step_certifies_and_replays(self, name: str) -> None:
        rule, build, _, _ = CASES[name]
        diagram = build()
        result = apply(diagram, rule, rule.pattern.find_matches(diagram)[0])
        assert replay(certify(diagram, [result], label=name)).reproduced

    def test_every_new_rule_is_registered(self) -> None:
        for rule, *_ in CASES.values():
            assert RULES[rule.name] is rule


class TestNonMatches:
    def test_a_phased_state_does_not_copy_through_w(self) -> None:
        phase = PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))})
        assert not W_ZERO_COPY.pattern.find_matches(state_into(W_NODE, 2, phase))

    def test_a_z_state_does_not_copy_through_w(self) -> None:
        diagram = Diagram()
        z = diagram.add_node(Z_SPIDER, [], [D])
        w = diagram.add_node(W_NODE, [D], [D, D])
        diagram.add_wire(out(z, 0), inp(w, 0))
        diagram.set_boundary_outputs([out(w, 0), out(w, 1)])
        assert not W_ZERO_COPY.pattern.find_matches(diagram)

    def test_the_inverse_triangle_has_no_zero_effect_rule(self) -> None:
        assert not TRIANGLE_ZERO_EFFECT.pattern.find_matches(triangle_into_effect(TRIANGLE_INVERSE))

    def test_crossed_connective_wires_are_not_an_inverse(self) -> None:
        diagram = Diagram()
        z = diagram.add_node(Z_SPIDER, [], [S * S, S * S])
        split = diagram.add_node(DIM_SPLITTER, [S * S], [S, S])
        bind = diagram.add_node(DIM_BINDER, [S, S], [S * S])
        diagram.add_wire(out(z, 0), inp(split, 0))
        diagram.add_wire(out(split, 0), inp(bind, 1))
        diagram.add_wire(out(split, 1), inp(bind, 0))
        diagram.set_boundary_outputs([out(bind, 0), out(z, 1)])
        assert not CONNECTIVE_INVERSE.pattern.find_matches(diagram)
        assert not compare(diagram, split_then_bind_over(S), {"s": 2}).matched

    def test_a_lone_w_is_not_fused(self) -> None:
        assert not W_FUSION.pattern.find_matches(single_w(3))


def split_then_bind_over(dim: Dim) -> Diagram:
    """:func:`split_then_bind` with both split dimensions ``dim``."""
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, [], [dim * dim, dim * dim])
    split = diagram.add_node(DIM_SPLITTER, [dim * dim], [dim, dim])
    bind = diagram.add_node(DIM_BINDER, [dim, dim], [dim * dim])
    diagram.add_wire(out(z, 0), inp(split, 0))
    diagram.add_wire(out(split, 0), inp(bind, 0))
    diagram.add_wire(out(split, 1), inp(bind, 1))
    diagram.set_boundary_outputs([out(bind, 0), out(z, 1)])
    return diagram


class TestNormalForms:
    def test_a_w_tree_reduces_to_one_w(self) -> None:
        nf = normal_form(w_into_w(2, 1, 2))
        assert [node.generator_type.name for node in nf.diagram.nodes.values()] == ["W"]
        assert nf.diagram.nodes[next(iter(nf.diagram.nodes))].num_outputs == 3

    def test_a_w_tree_decides_equal_to_the_wide_w(self) -> None:
        left, right = w_into_w(2, 1, 2), single_w(3)
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.NORMAL_FORM

    def test_the_state_through_a_triangle_decides_equal_to_the_state(self) -> None:
        state = Diagram()
        x = state.add_node(X_SPIDER, [], [D])
        state.set_boundary_outputs([out(x, 0)])
        decision = decide_equal(state_into(TRIANGLE, 1), state)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason

    def test_the_triangle_effect_carries_its_scalar(self) -> None:
        nf = normal_form(triangle_into_effect())
        assert nf.diagram.scalar == Scalar.dim_power(D, 1, 2)
        assert [node.generator_type.name for node in nf.diagram.nodes.values()] == ["Z"]
