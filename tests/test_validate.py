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

"""Tests for qufzx.diagram.validate: the Phase 3 well-formedness checker."""

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import (
    IssueKind,
    ValidationFailedError,
    validate,
    validate_or_raise,
)

from .helpers import build_ghz_with_copy


class TestGhzWithCopyValidates:
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_concrete_dims_pass(self, d_value: int) -> None:
        diagram, _a, _b = build_ghz_with_copy(Dim.concrete(d_value))
        report = validate(diagram)
        assert report.is_valid, report.errors

    def test_symbolic_dim_with_symbolic_phase_passes(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, _a, _b = build_ghz_with_copy(d, phase_on_a=phase)
        report = validate(diagram)
        assert report.is_valid, report.errors

    def test_validate_or_raise_does_not_raise_on_valid_diagram(self) -> None:
        diagram, _a, _b = build_ghz_with_copy(Dim.concrete(2))
        validate_or_raise(diagram)


class TestDimensionMismatch:
    def test_mismatched_concrete_dims_fail_with_specific_kind(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_MISMATCH for issue in report.errors)

    def test_validate_or_raise_raises_on_mismatch(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        with pytest.raises(ValidationFailedError):
            validate_or_raise(diagram)


class TestDeferredDimensionConstraint:
    def test_symbol_against_a_product_containing_it_is_deferred_not_error(self) -> None:
        # Dim.unify's occurs check defers exactly this shape (see its docstring): "d"
        # occurs as a proper subterm of "d * e", so it is neither bound (as bare
        # symbol-vs-symbol would be) nor rejected -- it is a residual constraint.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b = diagram.add_node(Z_SPIDER, input_dims=[d * e], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        report = validate(diagram)
        assert report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_DEFERRED for issue in report.deferred)
        assert not any(issue.kind is IssueKind.DIMENSION_MISMATCH for issue in report.errors)


class TestBoundaryViolations:
    def test_duplicate_boundary_entry(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.set_boundary_outputs([ref, ref])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DUPLICATE_BOUNDARY_ENTRY for issue in report.errors)

    def test_boundary_port_also_wired(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.add_wire(ref, PortRef(b, Direction.INPUT, 0))
        diagram.set_boundary_outputs([ref])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_WIRED_AND_BOUNDARY for issue in report.errors)

    def test_out_of_range_index(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 5)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_INDEX_OUT_OF_RANGE for issue in report.errors)

    def test_unknown_node_id(self) -> None:
        diagram = Diagram()
        diagram.set_boundary_outputs([PortRef(NodeId(999), Direction.OUTPUT, 0)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.UNKNOWN_NODE for issue in report.errors)

    def test_wrong_direction_in_boundary_inputs(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_inputs([PortRef(a, Direction.OUTPUT, 0)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.BOUNDARY_DIRECTION_MISMATCH for issue in report.errors)

    def test_port_wired_twice(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        c = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.add_wire(ref, PortRef(b, Direction.INPUT, 0))
        diagram.add_wire(ref, PortRef(c, Direction.INPUT, 0))
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_WIRED_TWICE for issue in report.errors)


class TestGeneratorPolicyConformance:
    def test_all_legs_equal_violation(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[Dim.concrete(3)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors)

    def test_phase_dimension_mismatch(self) -> None:
        diagram = Diagram()
        mismatched_phase = PhaseVector(Dim.concrete(3), {1: Phase.turns(1)})
        diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(2)],
            output_dims=[Dim.concrete(2)],
            phase=mismatched_phase,
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors)

    def test_symbolic_leg_dims_deferred_not_hard_error(self) -> None:
        # d occurs as a proper subterm of d * e, so Dim.unify defers this pair rather
        # than binding or failing it (see its docstring) -- this must land as
        # DIMENSION_DEFERRED, not DIMENSION_POLICY_VIOLATION, mirroring the wire-level
        # case in TestDeferredDimensionConstraint.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d * e])
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors
        )
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id is not None
            for issue in report.deferred
        )

    def test_concrete_leg_dims_still_hard_error(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[Dim.concrete(3)])
        report = validate(diagram)
        assert any(
            issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION and not issue.deferred
            for issue in report.errors
        )

    def test_symbolic_phase_dim_deferred_not_hard_error(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        phase = PhaseVector(d * e, {1: Phase.symbol("alpha")})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
        report = validate(diagram)
        assert not any(issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors)
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id is not None
            for issue in report.deferred
        )

    def test_concrete_phase_dim_still_hard_error(self) -> None:
        diagram = Diagram()
        mismatched_phase = PhaseVector(Dim.concrete(3), {1: Phase.turns(1)})
        diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(2)],
            output_dims=[Dim.concrete(2)],
            phase=mismatched_phase,
        )
        report = validate(diagram)
        assert any(
            issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH and not issue.deferred
            for issue in report.errors
        )


class TestDanglingPorts:
    def test_dangling_output_port(self) -> None:
        diagram, a_id, _b = build_ghz_with_copy(Dim.concrete(2))
        # A's output 1 is on the boundary by default; strip it so it dangles.
        diagram.set_boundary_outputs(
            [ref for ref in diagram.boundary_outputs if ref != PortRef(a_id, Direction.OUTPUT, 1)]
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.PORT_UNUSED
            and issue.port_ref == PortRef(a_id, Direction.OUTPUT, 1)
            for issue in report.errors
        )

    def test_dangling_input_port(self) -> None:
        diagram = Diagram()
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.PORT_UNUSED and issue.port_ref == PortRef(b, Direction.INPUT, 0)
            for issue in report.errors
        )

    def test_dangling_port_suppressed_when_node_reference_already_broken(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 5)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_INDEX_OUT_OF_RANGE for issue in report.errors)
        assert not any(issue.kind is IssueKind.PORT_UNUSED for issue in report.errors)


class TestValidationIsPure:
    def test_validate_does_not_mutate_diagram(self) -> None:
        diagram, _a, _b = build_ghz_with_copy(Dim.concrete(2))
        nodes_before = dict(diagram.nodes)
        wires_before = diagram.wires
        boundary_before = diagram.boundary_outputs
        scalar_before = diagram.scalar
        validate(diagram)
        assert dict(diagram.nodes) == nodes_before
        assert diagram.wires == wires_before
        assert diagram.boundary_outputs == boundary_before
        assert diagram.scalar == scalar_before
