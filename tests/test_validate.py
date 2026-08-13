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


class TestAllLegsEqualJointSatisfiability:
    """F1, Phase 5 post-closing audit round 21: leg dims must be *jointly* unifiable, not
    merely pairwise-unifiable against the first leg -- see qufzx.algebra.dimension.unify_all.
    """

    def test_jointly_unsatisfiable_bindings_are_rejected(self) -> None:
        # d unifies with 2 (binds d := 2) and with 3 (binds d := 3) independently, but the
        # two bindings contradict -- validate() must reject this, not silently discard both.
        d = Dim.symbol("d")
        diagram = Diagram()
        diagram.add_node(
            Z_SPIDER, input_dims=[d, Dim.concrete(2), Dim.concrete(3)], output_dims=[]
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors)

    def test_verdict_is_order_independent(self) -> None:
        d = Dim.symbol("d")
        for dims in ([d, Dim.concrete(2), Dim.concrete(3)], [Dim.concrete(2), d, Dim.concrete(3)]):
            diagram = Diagram()
            diagram.add_node(Z_SPIDER, input_dims=list(dims), output_dims=[])
            report = validate(diagram)
            assert not report.is_valid
            assert any(
                issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors
            )

    def test_multiple_residual_deferred_pairs_are_all_reported(self) -> None:
        # R3: three legs, each pair deferred against the others -- one DIMENSION_DEFERRED
        # issue per residual pair, not one collapsed "strongest" issue for the whole node.
        d, e, f = Dim.symbol("d"), Dim.symbol("e"), Dim.symbol("f")
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d * e, d * f, e * f], output_dims=[])
        diagram.set_boundary_inputs(
            [PortRef(node, Direction.INPUT, i) for i in range(3)]
        )
        report = validate(diagram)
        assert report.is_valid
        deferred_on_node = [
            issue for issue in report.deferred if issue.kind is IssueKind.DIMENSION_DEFERRED
        ]
        assert len(deferred_on_node) >= 2


class TestSymbolRoleCollision:
    """F2, Phase 5 post-closing audit round 21: a name cannot legally serve as both a
    dimension symbol and a phase parameter within one diagram (see
    qufzx.diagram.validate._classify_symbol_role)."""

    def test_same_name_dimension_and_phase_symbol_is_rejected(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("d")})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[], phase=phase)
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.SYMBOL_ROLE_COLLISION for issue in report.errors)

    def test_dimension_symbol_embedded_in_root_of_unity_phase_is_not_a_collision(self) -> None:
        # A root-of-unity phase entry over the node's own symbolic dim legitimately embeds
        # that same dimension symbol (not a distinct phase-role symbol of the same name) --
        # this must not be flagged.
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.root_of_unity(1, d)})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[], phase=phase)
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.SYMBOL_ROLE_COLLISION for issue in report.errors
        )

    def test_distinct_symbols_of_the_same_name_are_never_equal(self) -> None:
        # The discriminator itself: Dim.symbol/Phase.symbol/Scalar.symbol build distinct
        # sympy Symbol objects for the same name string, via different assumptions.
        from qufzx.algebra.scalar import Scalar

        dim_symbol = Dim.symbol("d").to_sympy()
        phase_symbol = Phase.symbol("d").to_sympy_turns()
        scalar_symbol = Scalar.symbol("d").to_sympy()
        assert dim_symbol != phase_symbol
        assert dim_symbol != scalar_symbol
        assert phase_symbol != scalar_symbol


class TestNodeDimensionUndetermined:
    """Round 20, Task 9: a node with zero legs and no phase carries its dimension nowhere at
    all (per CLAUDE.md, "dimension is stored per port, not as one global parameter"), so it
    is not well-formed -- yet this module used to accept it as valid, while
    :mod:`qufzx.semantics.denote` correctly refused it. The invariant this closes:
    ``validate(d).is_valid`` implies every node in ``d`` is denotable (see
    ``tests/test_phase5_exhaustive_oracle.py``'s exhaustive sweep, which now checks this
    over its whole space, and ``denote.py``'s own "has no legs and no phase vector" message,
    which this issue's message deliberately echoes).
    """

    def test_legless_phaseless_node_is_a_hard_error(self) -> None:
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[])
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED
            and issue.node_id == node_id
            and not issue.deferred
            for issue in report.errors
        )

    def test_legless_node_with_a_phase_is_not_flagged(self) -> None:
        diagram = Diagram()
        diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[], phase=PhaseVector(Dim.concrete(2), {})
        )
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED for issue in report.issues
        )

    def test_node_with_a_leg_and_no_phase_is_not_flagged(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED for issue in report.issues
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
