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


"""Phase 10 done-when: every new generator and a mixed-dimension diagram checked two ways."""

from __future__ import annotations

import numpy as np
import pytest

from archytaszx.algebra.dimension import Dim
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
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.semantics.contract_numeric import contract
from archytaszx.semantics.contract_symbolic import contract_symbolic

ATOL = 1e-10
"""Every tensor in this suite has entries in {-1, 0, 1}, so the tolerance is absolute."""

D = Dim.symbol("d")
S = Dim.symbol("s")
T = Dim.symbol("t")


def _expose(diagram: Diagram, node_id: NodeId) -> None:
    """Put every leg of one node onto the matching boundary list, in port order."""
    node = diagram.nodes[node_id]
    diagram.set_boundary_outputs(
        [PortRef(node_id, Direction.OUTPUT, index) for index in range(len(node.outputs))]
    )
    diagram.set_boundary_inputs(
        [PortRef(node_id, Direction.INPUT, index) for index in range(len(node.inputs))]
    )


def _single(generator: GeneratorType, input_dims: list[Dim], output_dims: list[Dim]) -> Diagram:
    """A one-node diagram with all of that node's legs on the boundary."""
    diagram = Diagram()
    node_id = diagram.add_node(generator, input_dims=input_dims, output_dims=output_dims)
    _expose(diagram, node_id)
    return diagram


def _numeric(diagram: Diagram) -> np.ndarray:
    """The numeric oracle's tensor for a fully concrete diagram."""
    report = validate(diagram)
    assert report.is_valid, [issue.kind.value for issue in report.errors]
    return contract(diagram).tensor


def _symbolic(diagram: Diagram, environment: dict[str, int]) -> np.ndarray:
    """Contract symbolically, substitute the concrete dims, then enumerate entries."""
    report = validate(diagram)
    assert not report.errors, [issue.kind.value for issue in report.errors]
    dense = contract_symbolic(diagram).substitute(environment).to_dense()
    return np.asarray(dense, dtype=np.complex128)


def _agree(symbolic_diagram: Diagram, environment: dict[str, int], concrete: Diagram) -> None:
    """Assert the symbolic contraction and the numeric oracle agree entrywise."""
    left = _symbolic(symbolic_diagram, environment)
    right = _numeric(concrete)
    assert left.shape == right.shape
    assert np.allclose(left, right, rtol=0.0, atol=ATOL), (left, right)


class TestTheUnaryGeneratorsAgreeWithTheOracle:
    """T, Ti and W: symbolic in d, substituted, against the dense oracle."""

    @pytest.mark.parametrize("generator", [TRIANGLE, TRIANGLE_INVERSE])
    @pytest.mark.parametrize("value", [2, 3, 4, 5])
    def test_a_triangle_agrees(self, generator: GeneratorType, value: int) -> None:
        dim = Dim.concrete(value)
        _agree(_single(generator, [D], [D]), {"d": value}, _single(generator, [dim], [dim]))

    @pytest.mark.parametrize("value", [2, 3, 4, 5])
    def test_the_w_node_agrees(self, value: int) -> None:
        dim = Dim.concrete(value)
        _agree(
            _single(W_NODE, [D], [D, D]),
            {"d": value},
            _single(W_NODE, [dim], [dim, dim]),
        )


class TestTheDimensionConnectivesAgreeWithTheOracle:
    """B and S at every (s, t) over 2..4, with the two port dims kept formal."""

    @pytest.mark.parametrize("s_value", [2, 3, 4])
    @pytest.mark.parametrize("t_value", [2, 3, 4])
    def test_the_binder_agrees(self, s_value: int, t_value: int) -> None:
        s_dim, t_dim = Dim.concrete(s_value), Dim.concrete(t_value)
        _agree(
            _single(DIM_BINDER, [S, T], [S * T]),
            {"s": s_value, "t": t_value},
            _single(DIM_BINDER, [s_dim, t_dim], [s_dim * t_dim]),
        )

    @pytest.mark.parametrize("s_value", [2, 3, 4])
    @pytest.mark.parametrize("t_value", [2, 3, 4])
    def test_the_splitter_agrees(self, s_value: int, t_value: int) -> None:
        s_dim, t_dim = Dim.concrete(s_value), Dim.concrete(t_value)
        _agree(
            _single(DIM_SPLITTER, [S * T], [S, T]),
            {"s": s_value, "t": t_value},
            _single(DIM_SPLITTER, [s_dim * t_dim], [s_dim, t_dim]),
        )


def _binder_then_splitter(s_dim: Dim, t_dim: Dim) -> Diagram:
    """B taking (s, t) to s*t, then S taking it back: boundary is (s, t) in and out."""
    diagram = Diagram()
    binder = diagram.add_node(DIM_BINDER, input_dims=[s_dim, t_dim], output_dims=[s_dim * t_dim])
    splitter = diagram.add_node(
        DIM_SPLITTER, input_dims=[s_dim * t_dim], output_dims=[s_dim, t_dim]
    )
    diagram.add_wire(PortRef(binder, Direction.OUTPUT, 0), PortRef(splitter, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [PortRef(splitter, Direction.OUTPUT, 0), PortRef(splitter, Direction.OUTPUT, 1)]
    )
    diagram.set_boundary_inputs(
        [PortRef(binder, Direction.INPUT, 0), PortRef(binder, Direction.INPUT, 1)]
    )
    return diagram


def _splitter_then_binder(s_dim: Dim, t_dim: Dim) -> Diagram:
    """S taking s*t to (s, t), then B taking it back: boundary is one s*t leg each way."""
    diagram = Diagram()
    splitter = diagram.add_node(
        DIM_SPLITTER, input_dims=[s_dim * t_dim], output_dims=[s_dim, t_dim]
    )
    binder = diagram.add_node(DIM_BINDER, input_dims=[s_dim, t_dim], output_dims=[s_dim * t_dim])
    diagram.add_wire(PortRef(splitter, Direction.OUTPUT, 0), PortRef(binder, Direction.INPUT, 0))
    diagram.add_wire(PortRef(splitter, Direction.OUTPUT, 1), PortRef(binder, Direction.INPUT, 1))
    diagram.set_boundary_outputs([PortRef(binder, Direction.OUTPUT, 0)])
    diagram.set_boundary_inputs([PortRef(splitter, Direction.INPUT, 0)])
    return diagram


class TestTheConnectiveIdentityOnTheOracle:
    """The pairing convention is an isomorphism, both composites, at every (d1, d2) in 2..4."""

    @pytest.mark.parametrize("d1", [2, 3, 4])
    @pytest.mark.parametrize("d2", [2, 3, 4])
    def test_binder_then_splitter_is_the_identity_on_the_tensor_product(
        self, d1: int, d2: int
    ) -> None:
        tensor = _numeric(_binder_then_splitter(Dim.concrete(d1), Dim.concrete(d2)))
        expected = np.einsum("ac,bd->abcd", np.eye(d1), np.eye(d2))
        assert tensor.shape == (d1, d2, d1, d2)
        assert np.allclose(tensor, expected, rtol=0.0, atol=ATOL)

    @pytest.mark.parametrize("d1", [2, 3, 4])
    @pytest.mark.parametrize("d2", [2, 3, 4])
    def test_splitter_then_binder_is_the_identity_on_the_product_dimension(
        self, d1: int, d2: int
    ) -> None:
        tensor = _numeric(_splitter_then_binder(Dim.concrete(d1), Dim.concrete(d2)))
        assert tensor.shape == (d1 * d2, d1 * d2)
        assert np.allclose(tensor, np.eye(d1 * d2), rtol=0.0, atol=ATOL)

    @pytest.mark.slow
    @pytest.mark.parametrize("d1,d2", [(2, 3), (3, 2), (2, 2)])
    def test_both_composites_agree_symbolically_too(self, d1: int, d2: int) -> None:
        environment = {"s": d1, "t": d2}
        _agree(
            _binder_then_splitter(S, T),
            environment,
            _binder_then_splitter(Dim.concrete(d1), Dim.concrete(d2)),
        )
        _agree(
            _splitter_then_binder(S, T),
            environment,
            _splitter_then_binder(Dim.concrete(d1), Dim.concrete(d2)),
        )


def _mixed_spiders(s_dim: Dim, t_dim: Dim) -> Diagram:
    """Z spiders at s and at t joined through a B into a Z spider at s*t."""
    diagram = Diagram()
    low = diagram.add_node(Z_SPIDER, input_dims=[s_dim], output_dims=[s_dim])
    high = diagram.add_node(Z_SPIDER, input_dims=[t_dim], output_dims=[t_dim])
    binder = diagram.add_node(DIM_BINDER, input_dims=[s_dim, t_dim], output_dims=[s_dim * t_dim])
    joined = diagram.add_node(Z_SPIDER, input_dims=[s_dim * t_dim], output_dims=[s_dim * t_dim])
    diagram.add_wire(PortRef(low, Direction.OUTPUT, 0), PortRef(binder, Direction.INPUT, 0))
    diagram.add_wire(PortRef(high, Direction.OUTPUT, 0), PortRef(binder, Direction.INPUT, 1))
    diagram.add_wire(PortRef(binder, Direction.OUTPUT, 0), PortRef(joined, Direction.INPUT, 0))
    diagram.set_boundary_outputs([PortRef(joined, Direction.OUTPUT, 0)])
    diagram.set_boundary_inputs(
        [PortRef(low, Direction.INPUT, 0), PortRef(high, Direction.INPUT, 0)]
    )
    return diagram


class TestAMixedDimensionDiagram:
    """Dim 2 and dim 3 legs meeting a dim 6 spider, both contraction paths."""

    def test_the_mixed_diagram_validates_clean(self) -> None:
        report = validate(_mixed_spiders(Dim.concrete(2), Dim.concrete(3)))
        assert report.is_valid, [issue.kind.value for issue in report.errors]

    def test_the_numeric_oracle_gives_the_paired_copy_tensor(self) -> None:
        tensor = _numeric(_mixed_spiders(Dim.concrete(2), Dim.concrete(3)))
        assert tensor.shape == (6, 2, 3)
        expected = np.zeros((6, 2, 3), dtype=np.complex128)
        for a in range(2):
            for b in range(3):
                expected[a * 3 + b][a][b] = 1.0
        assert np.allclose(tensor, expected, rtol=0.0, atol=ATOL)

    @pytest.mark.slow
    def test_it_contracts_identically_both_ways(self) -> None:
        _agree(
            _mixed_spiders(S, T),
            {"s": 2, "t": 3},
            _mixed_spiders(Dim.concrete(2), Dim.concrete(3)),
        )


def _triangle_and_z(dim: Dim) -> Diagram:
    """T into a Z spider, one of whose outputs passes through Ti."""
    diagram = Diagram()
    triangle = diagram.add_node(TRIANGLE, input_dims=[dim], output_dims=[dim])
    spider = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    inverse = diagram.add_node(TRIANGLE_INVERSE, input_dims=[dim], output_dims=[dim])
    diagram.add_wire(PortRef(triangle, Direction.OUTPUT, 0), PortRef(spider, Direction.INPUT, 0))
    diagram.add_wire(PortRef(spider, Direction.OUTPUT, 1), PortRef(inverse, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [PortRef(spider, Direction.OUTPUT, 0), PortRef(inverse, Direction.OUTPUT, 0)]
    )
    diagram.set_boundary_inputs([PortRef(triangle, Direction.INPUT, 0)])
    return diagram


def _w_and_x(dim: Dim) -> Diagram:
    """A W node whose second output feeds an X spider."""
    diagram = Diagram()
    w_node = diagram.add_node(W_NODE, input_dims=[dim], output_dims=[dim, dim])
    spider = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[dim])
    diagram.add_wire(PortRef(w_node, Direction.OUTPUT, 1), PortRef(spider, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [PortRef(w_node, Direction.OUTPUT, 0), PortRef(spider, Direction.OUTPUT, 0)]
    )
    diagram.set_boundary_inputs([PortRef(w_node, Direction.INPUT, 0)])
    return diagram


class TestCompositeDiagramsWithTheNewGenerators:
    """A triangle-and-Z diagram and a W-and-X diagram, each checked both ways."""

    def test_the_w_and_x_diagram_agrees(self) -> None:
        _agree(_w_and_x(D), {"d": 2}, _w_and_x(Dim.concrete(2)))

    @pytest.mark.slow
    @pytest.mark.parametrize("value", [2, 3])
    def test_the_triangle_and_z_diagram_agrees(self, value: int) -> None:
        _agree(_triangle_and_z(D), {"d": value}, _triangle_and_z(Dim.concrete(value)))

    @pytest.mark.slow
    @pytest.mark.parametrize("value", [3, 4])
    def test_the_w_and_x_diagram_agrees_at_larger_dimensions(self, value: int) -> None:
        _agree(_w_and_x(D), {"d": value}, _w_and_x(Dim.concrete(value)))
