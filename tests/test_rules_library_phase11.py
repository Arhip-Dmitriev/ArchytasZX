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

"""Establishes the six Phase 11 rules: their matchers, their surgery, and their exact scalars.

Every scalar is checked against :func:`archytaszx.semantics.check.compare` at several
dimensions and leg counts, never against the algebra it was derived from.
"""

from __future__ import annotations

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, abstract_subgraph_count, instantiate_symbol
from archytaszx.diagram.generators import (
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import (
    find_bialgebra_matches,
    find_fourier_state_matches,
    find_hopf_matches,
    find_identity_matches,
    find_state_copy_matches,
    find_triangle_matches,
)
from archytaszx.rewrite.rule import RewriteDomainError, RewriteGrammarError
from archytaszx.rewrite.rules_library import (
    BIALGEBRA,
    FOURIER_STATE_COLOR_CHANGE,
    HOPF,
    IDENTITY_REMOVAL,
    RULES,
    STATE_COPY,
    TRIANGLE_INVERSE_CANCELLATION,
    bialgebra_builder,
    bialgebra_scalar,
    hopf_builder,
    hopf_scalar,
    identity_removal_builder,
    state_copy_builder,
    state_copy_scalar,
)
from archytaszx.semantics.check import compare

DIMENSIONS = (2, 3, 4, 5)
"""The concrete dimensions every oracle test below sweeps."""


def out(node_id: NodeId, index: int = 0) -> PortRef:
    """The node's output port at ``index``."""
    return PortRef(node_id, Direction.OUTPUT, index)


def inp(node_id: NodeId, index: int = 0) -> PortRef:
    """The node's input port at ``index``."""
    return PortRef(node_id, Direction.INPUT, index)


def phase_vector(d: int) -> PhaseVector:
    """A non-zero phase vector over ``Dim.concrete(d)``, entry ``k`` at ``k / d`` turns."""
    return PhaseVector(Dim.concrete(d), {k: Phase.turns(sp.Rational(k, d)) for k in range(1, d)})


def identity_chain(d: int, generator: GeneratorType = Z_SPIDER) -> Diagram:
    """A Z state, a one-in-one-out spider, and an X effect in series."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    middle = diagram.add_node(generator, input_dims=[dim], output_dims=[dim])
    effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(out(state), inp(middle))
    diagram.add_wire(out(middle), inp(effect))
    return diagram


def triangle_chain(d: int, *, inverse_first: bool = False) -> Diagram:
    """A Z state into a T/Ti pair in series, the pair's far leg on the output boundary."""
    dim = Dim.concrete(d)
    first, second = (TRIANGLE_INVERSE, TRIANGLE) if inverse_first else (TRIANGLE, TRIANGLE_INVERSE)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    a = diagram.add_node(first, input_dims=[dim], output_dims=[dim])
    b = diagram.add_node(second, input_dims=[dim], output_dims=[dim])
    diagram.add_wire(out(state), inp(a))
    diagram.add_wire(out(a), inp(b))
    diagram.set_boundary_outputs([out(b)])
    return diagram


def state_copy_diagram(d: int, n: int) -> Diagram:
    """A phaseless X state into a phaseless Z spider with ``n`` outputs, all on the boundary."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    state = diagram.add_node(X_SPIDER, input_dims=[], output_dims=[dim])
    spider = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim] * n)
    diagram.add_wire(out(state), inp(spider))
    diagram.set_boundary_outputs([out(spider, i) for i in range(n)])
    return diagram


def state_copy_right_hand_side(d: int, n: int) -> Diagram:
    """``n`` separate phaseless X states, each output on the boundary."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    states = [diagram.add_node(X_SPIDER, input_dims=[], output_dims=[dim]) for _ in range(n)]
    diagram.set_boundary_outputs([out(node_id) for node_id in states])
    return diagram


def hopf_diagram(d: int, m: int, n: int, p: int, q: int, *, phases: bool = False) -> Diagram:
    """A Z_{m->n+2} and an X_{p+2->q} joined by a plain wire and by a wire through two F boxes."""
    dim = Dim.concrete(d)
    vector = phase_vector(d) if phases else None
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, [dim] * m, [dim] * (n + 2), phase=vector)
    x = diagram.add_node(X_SPIDER, [dim] * (p + 2), [dim] * q, phase=vector)
    f1 = diagram.add_node(FOURIER_BOX, [dim], [dim])
    f2 = diagram.add_node(FOURIER_BOX, [dim], [dim])
    diagram.add_wire(out(z, n), inp(x, p))
    diagram.add_wire(out(z, n + 1), inp(f1))
    diagram.add_wire(out(f1), inp(f2))
    diagram.add_wire(out(f2), inp(x, p + 1))
    diagram.set_boundary_inputs([inp(z, i) for i in range(m)] + [inp(x, i) for i in range(p)])
    diagram.set_boundary_outputs([out(z, i) for i in range(n)] + [out(x, i) for i in range(q)])
    return diagram


def bialgebra_diagram(d: int) -> Diagram:
    """A phaseless X_{2->1} feeding a phaseless Z_{1->2}, all four free legs on the boundary."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    x = diagram.add_node(X_SPIDER, [dim, dim], [dim])
    z = diagram.add_node(Z_SPIDER, [dim], [dim, dim])
    diagram.add_wire(out(x), inp(z))
    diagram.set_boundary_inputs([inp(x, 0), inp(x, 1)])
    diagram.set_boundary_outputs([out(z, 0), out(z, 1)])
    return diagram


def bialgebra_right_hand_side(d: int) -> Diagram:
    """Two Z_{1->2} and two X_{2->1}, each Z feeding both Xs."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    z_ids = [diagram.add_node(Z_SPIDER, [dim], [dim, dim]) for _ in range(2)]
    x_ids = [diagram.add_node(X_SPIDER, [dim, dim], [dim]) for _ in range(2)]
    for z_index in range(2):
        for x_index in range(2):
            diagram.add_wire(out(z_ids[z_index], x_index), inp(x_ids[x_index], z_index))
    diagram.set_boundary_inputs([inp(node_id) for node_id in z_ids])
    diagram.set_boundary_outputs([out(node_id) for node_id in x_ids])
    return diagram


def fourier_state_diagram(d: int, *, is_state: bool) -> Diagram:
    """An F box in series with a phaseless Z state, or with a phaseless Z effect."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    fourier = diagram.add_node(FOURIER_BOX, [dim], [dim])
    if is_state:
        spider = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        diagram.add_wire(out(spider), inp(fourier))
        diagram.set_boundary_outputs([out(fourier)])
    else:
        spider = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[])
        diagram.add_wire(out(fourier), inp(spider))
        diagram.set_boundary_inputs([inp(fourier)])
    return diagram


class TestEveryRuleIsRegistered:
    def test_the_six_new_rules_resolve_by_name(self) -> None:
        for rule in (
            IDENTITY_REMOVAL,
            TRIANGLE_INVERSE_CANCELLATION,
            STATE_COPY,
            HOPF,
            BIALGEBRA,
            FOURIER_STATE_COLOR_CHANGE,
        ):
            assert RULES[rule.name] is rule


class TestIdentityRemoval:
    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("generator", [Z_SPIDER, X_SPIDER])
    def test_splicing_out_preserves_the_denotation(self, d: int, generator: GeneratorType) -> None:
        diagram = identity_chain(d, generator)
        match = find_identity_matches(diagram)[0]
        result = apply(diagram, IDENTITY_REMOVAL, match)
        assert result.step.scalar_introduced == Scalar.one()
        assert len(result.diagram.nodes) == 2
        assert len(result.diagram.wires) == 1
        assert compare(diagram, result.diagram, {}).matched

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_a_boundary_leg_follows_the_splice(self, d: int) -> None:
        dim = Dim.concrete(d)
        diagram = Diagram()
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        middle = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(out(state), inp(middle))
        diagram.set_boundary_outputs([out(middle)])
        match = find_identity_matches(diagram)[0]
        result = apply(diagram, IDENTITY_REMOVAL, match)
        assert result.diagram.boundary_outputs == (out(state),)
        assert compare(diagram, result.diagram, {}).matched

    def test_a_phased_spider_is_not_a_match(self) -> None:
        diagram = identity_chain(3)
        middle = sorted(diagram.nodes)[1]
        diagram.set_phase(middle, phase_vector(3))
        assert find_identity_matches(diagram) == ()

    def test_a_zero_phase_vector_is_still_a_match(self) -> None:
        diagram = identity_chain(3)
        middle = sorted(diagram.nodes)[1]
        diagram.set_phase(middle, PhaseVector(Dim.concrete(3), {}))
        assert len(find_identity_matches(diagram)) == 1

    def test_other_leg_counts_are_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
        diagram.set_boundary_inputs([inp(node)])
        diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
        assert find_identity_matches(diagram) == ()

    def test_a_self_loop_on_the_node_is_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(out(node), inp(node))
        assert find_identity_matches(diagram) == ()

    def test_both_legs_on_the_boundary_is_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        diagram.set_boundary_inputs([inp(node)])
        diagram.set_boundary_outputs([out(node)])
        assert find_identity_matches(diagram) == ()

    def test_legs_of_different_dimensions_are_not_a_match(self) -> None:
        diagram = Diagram()
        node = diagram.add_node(
            Z_SPIDER, input_dims=[Dim.symbol("d")], output_dims=[Dim.symbol("e")]
        )
        other = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.symbol("e")])
        diagram.add_wire(out(other), inp(node))
        assert find_identity_matches(diagram) == ()

    def test_a_foreign_match_type_is_a_grammar_error(self) -> None:
        diagram = bialgebra_diagram(3)
        match = find_bialgebra_matches(diagram)[0]
        with pytest.raises(RewriteGrammarError):
            identity_removal_builder(diagram, match)

    def test_a_match_from_another_diagram_is_a_domain_error(self) -> None:
        match = find_identity_matches(identity_chain(3))[0]
        with pytest.raises(RewriteDomainError):
            identity_removal_builder(identity_chain(2), match)


class TestTriangleInverseCancellation:
    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("inverse_first", [False, True])
    def test_the_pair_cancels_at_every_dimension(self, d: int, inverse_first: bool) -> None:
        diagram = triangle_chain(d, inverse_first=inverse_first)
        match = find_triangle_matches(diagram)[0]
        result = apply(diagram, TRIANGLE_INVERSE_CANCELLATION, match)
        assert result.step.scalar_introduced == Scalar.one()
        assert len(result.diagram.nodes) == 1
        assert compare(diagram, result.diagram, {}).matched

    def test_two_triangles_of_the_same_kind_are_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        a = diagram.add_node(TRIANGLE, input_dims=[dim], output_dims=[dim])
        b = diagram.add_node(TRIANGLE, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(out(state), inp(a))
        diagram.add_wire(out(a), inp(b))
        diagram.set_boundary_outputs([out(b)])
        assert find_triangle_matches(diagram) == ()

    def test_both_outer_legs_on_the_boundary_is_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        a = diagram.add_node(TRIANGLE, input_dims=[dim], output_dims=[dim])
        b = diagram.add_node(TRIANGLE_INVERSE, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(out(a), inp(b))
        diagram.set_boundary_inputs([inp(a)])
        diagram.set_boundary_outputs([out(b)])
        assert find_triangle_matches(diagram) == ()

    def test_nodes_in_different_bang_boxes_are_not_a_match(self) -> None:
        diagram = triangle_chain(3)
        a = sorted(diagram.nodes)[1]
        diagram.add_bang_box(Mult.symbol("k"), node_scope=frozenset({a}))
        assert find_triangle_matches(diagram) == ()


class TestStateCopy:
    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_the_scalar_is_the_oracle_s(self, d: int, n: int) -> None:
        left = state_copy_diagram(d, n)
        right = state_copy_right_hand_side(d, n)
        right.multiply_scalar(state_copy_scalar(Dim.concrete(d), n))
        assert compare(left, right, {}).matched

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_the_builder_reports_that_scalar(self, n: int) -> None:
        diagram = state_copy_diagram(3, n)
        match = find_state_copy_matches(diagram)[0]
        assert match.output_count == n
        result = state_copy_builder(diagram.copy(), match)
        assert result.scalar_introduced == state_copy_scalar(Dim.concrete(3), n)
        assert len(result.new_node_ids) == n

    def test_the_rule_evaluates_its_scalar_from_the_match(self) -> None:
        for n in (0, 1, 2, 3):
            match = find_state_copy_matches(state_copy_diagram(3, n))[0]
            assert STATE_COPY.scalar_for_match(match) == state_copy_scalar(Dim.concrete(3), n)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_applying_preserves_the_denotation(self, d: int, n: int) -> None:
        diagram = state_copy_diagram(d, n)
        match = find_state_copy_matches(diagram)[0]
        result = apply(diagram, STATE_COPY, match)
        assert compare(diagram, result.diagram, {}).matched

    def test_a_z_state_into_a_z_spider_is_not_a_match(self) -> None:
        diagram = state_copy_diagram(3, 2)
        state = min(diagram.nodes)
        assert diagram.nodes[state].generator_type.name == X_SPIDER.name
        other = Diagram()
        dim = Dim.concrete(3)
        z_state = other.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        spider = other.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
        other.add_wire(out(z_state), inp(spider))
        other.set_boundary_outputs([out(spider, 0), out(spider, 1)])
        assert find_state_copy_matches(other) == ()

    def test_a_phased_spider_is_not_a_match(self) -> None:
        diagram = state_copy_diagram(3, 2)
        spider = sorted(diagram.nodes)[1]
        diagram.set_phase(spider, phase_vector(3))
        assert find_state_copy_matches(diagram) == ()


class TestHopf:
    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("legs", [(1, 0, 0, 1), (1, 1, 1, 1), (2, 1, 0, 2), (0, 1, 1, 0)])
    def test_the_spiders_fall_apart_times_one_over_d(
        self, d: int, legs: tuple[int, int, int, int]
    ) -> None:
        diagram = hopf_diagram(d, *legs)
        match = find_hopf_matches(diagram)[0]
        result = apply(diagram, HOPF, match)
        assert result.step.scalar_introduced == hopf_scalar(Dim.concrete(d))
        assert len(result.diagram.nodes) == 2
        assert result.diagram.wires == frozenset()
        assert compare(diagram, result.diagram, {}).matched

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_phases_ride_through_unchanged(self, d: int) -> None:
        diagram = hopf_diagram(d, 1, 1, 1, 1, phases=True)
        match = find_hopf_matches(diagram)[0]
        result = apply(diagram, HOPF, match)
        assert compare(diagram, result.diagram, {}).matched

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_a_plain_double_edge_is_not_a_match(self, d: int) -> None:
        dim = Dim.concrete(d)
        diagram = Diagram()
        z = diagram.add_node(Z_SPIDER, [dim], [dim, dim])
        x = diagram.add_node(X_SPIDER, [dim, dim], [dim])
        diagram.add_wire(out(z, 0), inp(x, 0))
        diagram.add_wire(out(z, 1), inp(x, 1))
        diagram.set_boundary_inputs([inp(z)])
        diagram.set_boundary_outputs([out(x)])
        assert find_hopf_matches(diagram) == ()

    def test_a_single_fourier_box_on_the_second_path_is_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        z = diagram.add_node(Z_SPIDER, [dim], [dim, dim])
        x = diagram.add_node(X_SPIDER, [dim, dim], [dim])
        f = diagram.add_node(FOURIER_BOX, [dim], [dim])
        diagram.add_wire(out(z, 0), inp(x, 0))
        diagram.add_wire(out(z, 1), inp(f))
        diagram.add_wire(out(f), inp(x, 1))
        diagram.set_boundary_inputs([inp(z)])
        diagram.set_boundary_outputs([out(x)])
        assert find_hopf_matches(diagram) == ()

    def test_a_foreign_match_type_is_a_grammar_error(self) -> None:
        diagram = hopf_diagram(3, 1, 0, 0, 1)
        with pytest.raises(RewriteGrammarError):
            hopf_builder(diagram, find_identity_matches(identity_chain(3))[0])


class TestBialgebra:
    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_the_scalar_is_the_oracle_s(self, d: int) -> None:
        left = bialgebra_diagram(d)
        right = bialgebra_right_hand_side(d)
        right.multiply_scalar(bialgebra_scalar(Dim.concrete(d)))
        assert compare(left, right, {}).matched

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_the_builder_reports_four_new_nodes_and_four_new_wires(self, d: int) -> None:
        diagram = bialgebra_diagram(d)
        match = find_bialgebra_matches(diagram)[0]
        result = bialgebra_builder(diagram.copy(), match)
        assert len(result.new_node_ids) == 4
        assert len(result.new_wires) == 4
        assert result.scalar_introduced == bialgebra_scalar(Dim.concrete(d))

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_applying_preserves_the_denotation(self, d: int) -> None:
        diagram = bialgebra_diagram(d)
        match = find_bialgebra_matches(diagram)[0]
        result = apply(diagram, BIALGEBRA, match)
        assert len(result.diagram.nodes) == 4
        assert compare(diagram, result.diagram, {}).matched

    def test_the_reversed_colors_are_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        z = diagram.add_node(Z_SPIDER, [dim, dim], [dim])
        x = diagram.add_node(X_SPIDER, [dim], [dim, dim])
        diagram.add_wire(out(z), inp(x))
        diagram.set_boundary_inputs([inp(z, 0), inp(z, 1)])
        diagram.set_boundary_outputs([out(x, 0), out(x, 1)])
        assert find_bialgebra_matches(diagram) == ()


class TestFourierStateColorChange:
    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("is_state", [True, False])
    def test_the_fourier_box_recolors_the_spider(self, d: int, is_state: bool) -> None:
        diagram = fourier_state_diagram(d, is_state=is_state)
        match = find_fourier_state_matches(diagram)[0]
        assert match.is_state is is_state
        result = apply(diagram, FOURIER_STATE_COLOR_CHANGE, match)
        assert result.step.scalar_introduced == Scalar.one()
        assert len(result.diagram.nodes) == 1
        new_node = result.diagram.nodes[result.new_node_ids[0]]
        assert new_node.generator_type.name == X_SPIDER.name
        assert compare(diagram, result.diagram, {}).matched

    def test_a_phased_spider_is_not_a_match(self) -> None:
        diagram = fourier_state_diagram(3, is_state=True)
        spider = next(
            node_id
            for node_id in diagram.nodes
            if diagram.nodes[node_id].generator_type.name == Z_SPIDER.name
        )
        diagram.set_phase(spider, phase_vector(3))
        assert find_fourier_state_matches(diagram) == ()

    def test_a_multi_leg_spider_is_not_a_match(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        spider = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
        fourier = diagram.add_node(FOURIER_BOX, [dim], [dim])
        diagram.add_wire(out(spider, 0), inp(fourier))
        diagram.set_boundary_outputs([out(fourier), out(spider, 1)])
        assert find_fourier_state_matches(diagram) == ()


class TestBangBoxes:
    """A node-scope bang box replicates its contents, so a non-unit scalar must not fire inside."""

    @pytest.mark.parametrize(
        "build,finder",
        [
            (lambda d: state_copy_diagram(d, 2), find_state_copy_matches),
            (bialgebra_diagram, find_bialgebra_matches),
            (lambda d: hopf_diagram(d, 1, 0, 0, 1), find_hopf_matches),
        ],
    )
    def test_a_non_unit_scalar_rule_does_not_match_inside_a_node_scope_box(
        self, build: object, finder: object
    ) -> None:
        diagram = build(3)  # type: ignore[operator]
        diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset(diagram.nodes))
        assert finder(diagram) == ()  # type: ignore[operator]

    @pytest.mark.parametrize("m_value", [0, 1, 2, 3])
    @pytest.mark.parametrize("d", [2, 3])
    def test_identity_removal_inside_a_box_holds_at_every_multiplicity(
        self, d: int, m_value: int
    ) -> None:
        dim = Dim.concrete(d)
        diagram = Diagram()
        middle = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        other = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[dim, dim])
        diagram.add_wire(out(middle), inp(other))
        diagram.set_boundary_inputs([inp(middle)])
        diagram.set_boundary_outputs([out(other, 0), out(other, 1)])
        pre, _box_id, _m = abstract_subgraph_count(diagram, frozenset({middle, other}), 1, stem="m")
        post = apply(pre, IDENTITY_REMOVAL, find_identity_matches(pre)[0]).diagram
        result = compare(
            instantiate_symbol(pre, "m", m_value),
            instantiate_symbol(post, "m", m_value),
            {},
        )
        assert result.matched, result.reason

    def test_a_box_holding_only_the_matched_node_is_not_a_match(self) -> None:
        diagram = identity_chain(3)
        middle = sorted(diagram.nodes)[1]
        diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({middle}))
        assert find_identity_matches(diagram) == ()

    def test_a_box_holding_only_the_matched_pair_is_not_a_match(self) -> None:
        diagram = triangle_chain(3)
        a, b = sorted(diagram.nodes)[1:3]
        diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({a, b}))
        assert find_triangle_matches(diagram) == ()
