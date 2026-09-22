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

"""Tests for the Phase 10 denotations: T, Ti, W, B, S, and the per-leg dimension readers."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, Direction, Node, PortRef
from archytaszx.semantics.contract_symbolic import _Indices, _node_entry
from archytaszx.semantics.denote import (
    DenoteDomainError,
    DenoteGrammarError,
    denote,
    leg_dimensions,
    leg_dims,
)

_DIMS = (2, 3, 4, 5)
_PAIRS = [(s, t) for s in (2, 3, 4, 5) for t in (2, 3, 4, 5)]


def _node(
    generator: object, input_dims: list[Dim], output_dims: list[Dim], phase: object = None
) -> Node:
    """Build a one-node diagram and hand back the node."""
    diagram = Diagram()
    node_id = diagram.add_node(generator, input_dims, output_dims, phase=phase)  # type: ignore[arg-type]
    return diagram.nodes[node_id]


def _unary(generator: object, d: int) -> Node:
    dim = Dim.concrete(d)
    return _node(generator, [dim], [dim])


def _w(d: int) -> Node:
    dim = Dim.concrete(d)
    return _node(W_NODE, [dim], [dim, dim])


def _binder(s: int, t: int) -> Node:
    return _node(DIM_BINDER, [Dim.concrete(s), Dim.concrete(t)], [Dim.concrete(s * t)])


def _splitter(s: int, t: int) -> Node:
    return _node(DIM_SPLITTER, [Dim.concrete(s * t)], [Dim.concrete(s), Dim.concrete(t)])


class TestTriangle:
    @pytest.mark.parametrize("d", _DIMS)
    def test_entrywise(self, d: int) -> None:
        tensor = denote(_unary(TRIANGLE, d))
        assert tensor.shape == (d, d)
        for out in range(d):
            for inp in range(d):
                expected = 1 if (out == 0 or out == inp) else 0
                assert tensor[out][inp] == expected

    def test_d2_is_the_known_matrix(self) -> None:
        assert np.array_equal(denote(_unary(TRIANGLE, 2)), np.array([[1, 1], [0, 1]]))

    @pytest.mark.parametrize("d", _DIMS)
    def test_first_row_and_diagonal_are_one(self, d: int) -> None:
        tensor = denote(_unary(TRIANGLE, d))
        assert np.array_equal(tensor[0], np.ones(d))
        assert np.array_equal(np.diagonal(tensor), np.ones(d))


class TestTriangleInverse:
    @pytest.mark.parametrize("d", _DIMS)
    def test_entrywise(self, d: int) -> None:
        tensor = denote(_unary(TRIANGLE_INVERSE, d))
        assert tensor.shape == (d, d)
        for out in range(d):
            for inp in range(d):
                if out == inp:
                    expected = 1
                elif out == 0 and inp >= 1:
                    expected = -1
                else:
                    expected = 0
                assert tensor[out][inp] == expected

    def test_d2_is_the_known_matrix(self) -> None:
        assert np.array_equal(denote(_unary(TRIANGLE_INVERSE, 2)), np.array([[1, -1], [0, 1]]))

    @pytest.mark.parametrize("d", _DIMS)
    def test_is_the_two_sided_inverse_of_the_triangle(self, d: int) -> None:
        triangle = denote(_unary(TRIANGLE, d))
        inverse = denote(_unary(TRIANGLE_INVERSE, d))
        identity = np.eye(d, dtype=np.complex128)
        assert np.allclose(triangle @ inverse, identity)
        assert np.allclose(inverse @ triangle, identity)

    @pytest.mark.parametrize("d", _DIMS)
    def test_entries_are_exact_integers(self, d: int) -> None:
        tensor = denote(_unary(TRIANGLE_INVERSE, d))
        assert np.array_equal(tensor, np.round(tensor.real).astype(np.complex128))


class TestWNode:
    @pytest.mark.parametrize("d", _DIMS)
    def test_entrywise(self, d: int) -> None:
        tensor = denote(_w(d))
        assert tensor.shape == (d, d, d)
        expected = np.zeros((d, d, d), dtype=np.complex128)
        expected[0][0][0] = 1
        for i in range(1, d):
            expected[0][i][i] = 1
            expected[i][0][i] = 1
        assert np.array_equal(tensor, expected)

    @pytest.mark.parametrize("d", _DIMS)
    def test_the_all_zero_entry_is_assigned_not_accumulated(self, d: int) -> None:
        assert denote(_w(d))[0][0][0] == 1

    def test_d2_is_the_qubit_w_node(self) -> None:
        expected = np.zeros((2, 2, 2), dtype=np.complex128)
        expected[0][0][0] = 1
        expected[0][1][1] = 1
        expected[1][0][1] = 1
        assert np.array_equal(denote(_w(2)), expected)

    @pytest.mark.parametrize("d", _DIMS)
    def test_symmetric_under_swapping_the_output_axes(self, d: int) -> None:
        tensor = denote(_w(d))
        assert np.array_equal(tensor, np.swapaxes(tensor, 0, 1))

    @pytest.mark.parametrize("d", _DIMS)
    def test_coassociative(self, d: int) -> None:
        tensor = denote(_w(d))
        left = np.einsum("abm,mci->abci", tensor, tensor)
        right = np.einsum("ami,bcm->abci", tensor, tensor)
        assert np.allclose(left, right)


class TestConnectives:
    @pytest.mark.parametrize(("s", "t"), _PAIRS)
    def test_binder_entrywise(self, s: int, t: int) -> None:
        tensor = denote(_binder(s, t))
        assert tensor.shape == (s * t, s, t)
        for out in range(s * t):
            for a in range(s):
                for b in range(t):
                    assert tensor[out][a][b] == (1 if out == a * t + b else 0)

    @pytest.mark.parametrize(("s", "t"), _PAIRS)
    def test_splitter_entrywise(self, s: int, t: int) -> None:
        tensor = denote(_splitter(s, t))
        assert tensor.shape == (s, t, s * t)
        for a in range(s):
            for b in range(t):
                for inp in range(s * t):
                    assert tensor[a][b][inp] == (1 if inp == a * t + b else 0)

    @pytest.mark.parametrize(("s", "t"), _PAIRS)
    def test_binder_after_splitter_is_the_joint_identity(self, s: int, t: int) -> None:
        binder = denote(_binder(s, t))
        splitter = denote(_splitter(s, t))
        composed = np.einsum("oab,abi->oi", binder, splitter)
        assert np.allclose(composed, np.eye(s * t, dtype=np.complex128))

    @pytest.mark.parametrize(("s", "t"), _PAIRS)
    def test_splitter_after_binder_is_the_factor_identity(self, s: int, t: int) -> None:
        binder = denote(_binder(s, t))
        splitter = denote(_splitter(s, t))
        composed = np.einsum("abk,kcd->abcd", splitter, binder)
        expected = np.einsum("ac,bd->abcd", np.eye(s), np.eye(t))
        assert np.allclose(composed, expected)

    def test_binder_rejects_a_non_product_output_dim(self) -> None:
        node = _node(DIM_BINDER, [Dim.concrete(2), Dim.concrete(3)], [Dim.concrete(5)])
        with pytest.raises(DenoteGrammarError):
            denote(node)

    def test_splitter_rejects_a_non_product_input_dim(self) -> None:
        node = _node(DIM_SPLITTER, [Dim.concrete(5)], [Dim.concrete(2), Dim.concrete(3)])
        with pytest.raises(DenoteGrammarError):
            denote(node)

    def test_binder_rejects_a_symbolic_port_dim(self) -> None:
        d1 = Dim.symbol("d1")
        node = _node(DIM_BINDER, [d1, Dim.concrete(3)], [d1 * 3])
        with pytest.raises(DenoteDomainError):
            denote(node)


class TestLegDims:
    def test_leg_dims_are_outputs_then_inputs(self) -> None:
        node = _binder(2, 3)
        assert leg_dims(node) == (Dim.concrete(6), Dim.concrete(2), Dim.concrete(3))

    def test_leg_dimensions_are_outputs_then_inputs(self) -> None:
        assert leg_dimensions(_binder(2, 3)) == (6, 2, 3)
        assert leg_dimensions(_splitter(2, 3)) == (2, 3, 6)

    def test_leg_dimensions_on_a_mixed_all_legs_equal_node(self) -> None:
        assert leg_dimensions(_w(4)) == (4, 4, 4)

    def test_leg_dimensions_raises_on_a_symbolic_port_dim(self) -> None:
        d1 = Dim.symbol("d1")
        node = _node(DIM_BINDER, [d1, Dim.concrete(3)], [d1 * 3])
        with pytest.raises(DenoteDomainError):
            leg_dimensions(node)

    def test_leg_dims_never_gates_on_concreteness(self) -> None:
        d1 = Dim.symbol("d1")
        node = _node(DIM_BINDER, [d1, Dim.concrete(3)], [d1 * 3])
        assert leg_dims(node) == (d1 * 3, d1, Dim.concrete(3))

    def test_zero_leg_node_with_a_phase_yields_an_empty_tuple(self) -> None:
        phase = PhaseVector(Dim.concrete(3), {1: Phase.turns(sp.Rational(1, 3))})
        assert leg_dimensions(_node(Z_SPIDER, [], [], phase=phase)) == ()

    def test_zero_leg_node_without_a_phase_is_rejected(self) -> None:
        with pytest.raises(DenoteGrammarError):
            leg_dimensions(_node(Z_SPIDER, [], []))


class TestArityGuards:
    def test_triangle_rejects_two_inputs(self) -> None:
        dim = Dim.concrete(3)
        node = _node(TRIANGLE, [dim, dim], [dim])
        with pytest.raises(DenoteGrammarError):
            denote(node)

    def test_w_rejects_one_output(self) -> None:
        dim = Dim.concrete(3)
        node = _node(W_NODE, [dim], [dim])
        with pytest.raises(DenoteGrammarError):
            denote(node)

    def test_binder_rejects_one_input(self) -> None:
        node = _node(DIM_BINDER, [Dim.concrete(6)], [Dim.concrete(6)])
        with pytest.raises(DenoteGrammarError):
            denote(node)


class TestSymbolicEntriesAgreeWithDenote:
    """Each new symbolic node entry, enumerated densely, equals the numeric denotation."""

    @staticmethod
    def _dense(generator: object, input_dims: list[int], output_dims: list[int]) -> np.ndarray:
        diagram = Diagram()
        node_id = diagram.add_node(
            generator,  # type: ignore[arg-type]
            [Dim.concrete(value) for value in input_dims],
            [Dim.concrete(value) for value in output_dims],
        )
        port_index = {
            PortRef(node_id, Direction.OUTPUT, i): sp.Symbol(
                f"_o{i}", integer=True, nonnegative=True
            )
            for i in range(len(output_dims))
        }
        port_index.update(
            {
                PortRef(node_id, Direction.INPUT, i): sp.Symbol(
                    f"_n{i}", integer=True, nonnegative=True
                )
                for i in range(len(input_dims))
            }
        )
        entry = _node_entry(diagram, node_id, port_index, _Indices())
        order = [PortRef(node_id, Direction.OUTPUT, i) for i in range(len(output_dims))] + [
            PortRef(node_id, Direction.INPUT, i) for i in range(len(input_dims))
        ]
        symbols = [port_index[ref] for ref in order]
        shape = (*output_dims, *input_dims)
        out = np.zeros(shape, dtype=np.complex128)
        for index in np.ndindex(*shape):
            bound = dict(zip(symbols, (sp.Integer(v) for v in index), strict=True))
            out[index] = Scalar(entry.xreplace(bound)).simplify().to_complex()
        return out

    @pytest.mark.parametrize("d", (2, 3, 4))
    @pytest.mark.parametrize("generator", (TRIANGLE, TRIANGLE_INVERSE), ids=lambda g: g.name)
    def test_triangles(self, generator: object, d: int) -> None:
        assert np.allclose(self._dense(generator, [d], [d]), denote(_unary(generator, d)))

    @pytest.mark.parametrize("d", (2, 3, 4))
    def test_w_node(self, d: int) -> None:
        assert np.allclose(self._dense(W_NODE, [d], [d, d]), denote(_w(d)))

    @pytest.mark.parametrize(("s", "t"), ((2, 2), (2, 3), (3, 2)))
    def test_binder(self, s: int, t: int) -> None:
        assert np.allclose(self._dense(DIM_BINDER, [s, t], [s * t]), denote(_binder(s, t)))

    @pytest.mark.parametrize(("s", "t"), ((2, 2), (2, 3), (3, 2)))
    def test_splitter(self, s: int, t: int) -> None:
        assert np.allclose(self._dense(DIM_SPLITTER, [s * t], [s, t]), denote(_splitter(s, t)))
