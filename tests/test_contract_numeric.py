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

"""Tests for qufzx.semantics.contract_numeric: contracting a fully concrete diagram."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.semantics.contract_numeric import (
    ContractDomainError,
    ContractSizeError,
    ContractValidationError,
    _assign_labels,
    contract,
)

from .helpers import build_ghz_with_copy


class TestEmptyDiagram:
    def test_empty_diagram_is_the_scalar(self) -> None:
        diagram = Diagram()
        diagram.multiply_scalar(Scalar.rational(3))
        result = contract(diagram)
        assert result.tensor.shape == ()
        assert complex(result.tensor) == complex(3, 0)
        assert result.axis_refs == ()


class TestGhzWithCopy:
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_contracts_to_ghz_vector(self, d_value: int) -> None:
        d = Dim.concrete(d_value)
        diagram, _a, _b = build_ghz_with_copy(d)
        result = contract(diagram)
        assert result.tensor.shape == (d_value, d_value, d_value)
        expected = np.zeros((d_value,) * 3, dtype=complex)
        for k in range(d_value):
            expected[k, k, k] = 1
        np.testing.assert_allclose(result.tensor, expected)

    def test_boundary_order_determines_axis_order(self) -> None:
        d = Dim.concrete(2)
        diagram, a_id, b_id = build_ghz_with_copy(d)
        result_default = contract(diagram)

        diagram.set_boundary_outputs(
            [
                PortRef(b_id, Direction.OUTPUT, 1),
                PortRef(a_id, Direction.OUTPUT, 1),
                PortRef(b_id, Direction.OUTPUT, 0),
            ]
        )
        result_reordered = contract(diagram)
        assert result_reordered.tensor.shape == (2, 2, 2)
        # transpose reordered back to the default order: (2,1,0) axis mapping since
        # reordered = [B1, A1, B0] and default = [A1, B0, B1]
        transposed_back = np.transpose(result_reordered.tensor, (1, 2, 0))
        np.testing.assert_allclose(transposed_back, result_default.tensor)


class TestDisconnectedComponents:
    def test_tensor_product_of_two_components(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        phase_b = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 2))})
        diagram.set_phase(b, phase_b)
        diagram.set_boundary_outputs(
            [PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.OUTPUT, 0)]
        )
        result = contract(diagram)
        expected = np.outer([1, 1], [1, -1])
        np.testing.assert_allclose(result.tensor, expected, atol=1e-10)


class TestSelfLoop:
    def test_partial_trace_of_two_leg_spider(self) -> None:
        d = Dim.concrete(3)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(node, Direction.OUTPUT, 0), PortRef(node, Direction.INPUT, 0))
        result = contract(diagram)
        assert result.tensor.shape == ()
        # Trace of the identity-diagonal Z spider (no phase) at d=3 is 3.
        np.testing.assert_allclose(complex(result.tensor), complex(3, 0))


class TestWireBetweenTwoOutputs:
    def test_wire_joining_two_output_ports(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 1), PortRef(b, Direction.OUTPUT, 0))
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 0)])
        result = contract(diagram)
        # A denotes |00>+|11> on (out0,out1); B denotes |0>+|1> on out0. Wiring A.out1 to
        # B.out0 contracts those two axes: sum_k (A[k,k'] terms) -> sum_j A[i,j]*B[j].
        assert result.tensor.shape == (2,)
        np.testing.assert_allclose(result.tensor, [1, 1])


class TestScalarAccumulator:
    def test_multiply_scalar_scales_result_exactly(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        diagram.set_boundary_outputs(
            [PortRef(node, Direction.OUTPUT, 0), PortRef(node, Direction.OUTPUT, 1)]
        )
        base = contract(diagram).tensor
        diagram.multiply_scalar(Scalar.rational(5))
        scaled = contract(diagram).tensor
        np.testing.assert_allclose(scaled, base * 5)


class TestConcretenessGuards:
    def test_symbolic_dim_refused(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, 0)])
        diagram.set_boundary_inputs([PortRef(node, Direction.INPUT, 0)])
        with pytest.raises(ContractDomainError):
            contract(diagram)

    def test_symbolic_phase_refused(self) -> None:
        d = Dim.concrete(3)
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
        diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, 0)])
        diagram.set_boundary_inputs([PortRef(node, Direction.INPUT, 0)])
        with pytest.raises(ContractDomainError):
            contract(diagram)

    def test_symbolic_scalar_refused(self) -> None:
        diagram = Diagram()
        diagram.multiply_scalar(Scalar.symbol("s"))
        with pytest.raises(ContractDomainError):
            contract(diagram)


class TestValidationRefusal:
    def test_invalid_diagram_is_refused_with_report_attached(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        # The one port is neither wired nor on the boundary -> PORT_UNUSED, a hard error.
        with pytest.raises(ContractValidationError) as excinfo:
            contract(diagram)
        assert not excinfo.value.report.is_valid


class TestSizeGuard:
    def test_oversized_node_tensor_is_refused(self) -> None:
        d = Dim.concrete(17)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d] * 10, output_dims=[d] * 10)
        diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, i) for i in range(10)])
        diagram.set_boundary_inputs([PortRef(node, Direction.INPUT, i) for i in range(10)])
        with pytest.raises(ContractSizeError):
            contract(diagram, max_elements=1_000_000)


class TestOrderIndependence:
    def test_node_insertion_order_does_not_affect_result(self) -> None:
        d = Dim.concrete(2)

        diagram1 = Diagram()
        a1 = diagram1.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b1 = diagram1.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
        diagram1.add_wire(PortRef(a1, Direction.OUTPUT, 0), PortRef(b1, Direction.INPUT, 0))
        diagram1.set_boundary_outputs(
            [
                PortRef(a1, Direction.OUTPUT, 1),
                PortRef(b1, Direction.OUTPUT, 0),
                PortRef(b1, Direction.OUTPUT, 1),
            ]
        )

        diagram2 = Diagram()
        b2 = diagram2.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
        a2 = diagram2.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        diagram2.add_wire(PortRef(a2, Direction.OUTPUT, 0), PortRef(b2, Direction.INPUT, 0))
        diagram2.set_boundary_outputs(
            [
                PortRef(a2, Direction.OUTPUT, 1),
                PortRef(b2, Direction.OUTPUT, 0),
                PortRef(b2, Direction.OUTPUT, 1),
            ]
        )

        result1 = contract(diagram1)
        result2 = contract(diagram2)
        np.testing.assert_allclose(result1.tensor, result2.tensor)


class TestAssignLabelsUnionFind:
    """Direct unit tests of ``_assign_labels``, targeting the helper rather than ``contract``.

    The three-wire chain below (p1-p2, p3-p4, then p2-p3) wires p2 and p3 twice each --
    a shape ``validate`` rejects as ``PORT_WIRED_TWICE``, so it is not ``contract()``-legal
    and this bug is unreachable through ``contract`` itself. It is reachable directly
    through ``_assign_labels``, which Phase 7's bang-box instantiation is expected to call
    on wire sets it does not build (and therefore does not get to pre-validate) itself --
    see that function's own docstring.
    """

    def test_chained_merge_unifies_all_four_ports(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        n1 = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        n2 = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        n3 = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        n4 = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        p1 = PortRef(n1, Direction.OUTPUT, 0)
        p2 = PortRef(n2, Direction.OUTPUT, 0)
        p3 = PortRef(n3, Direction.OUTPUT, 0)
        p4 = PortRef(n4, Direction.OUTPUT, 0)
        diagram.add_wire(p1, p2)
        diagram.add_wire(p3, p4)
        diagram.add_wire(p2, p3)

        labels = _assign_labels(diagram)

        assert labels[p1] == labels[p2] == labels[p3] == labels[p4]
