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

"""Tests for archytaszx.diagram.validate's Phase 10 additions: the PRODUCT_OF_LEGS_EQUAL
per-node branch and the diagram-global dimension-consistency pass."""

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import DIM_BINDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.validate import IssueKind, validate


def _binder_into_spider(in_a: Dim, in_b: Dim, out: Dim, spider_dim: Dim) -> Diagram:
    """A B node joining two input dims into ``out``, wired into a one-leg-in Z spider."""
    diagram = Diagram()
    binder = diagram.add_node(DIM_BINDER, input_dims=[in_a, in_b], output_dims=[out])
    spider = diagram.add_node(Z_SPIDER, input_dims=[spider_dim], output_dims=[spider_dim])
    diagram.add_wire(PortRef(binder, Direction.OUTPUT, 0), PortRef(spider, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(binder, Direction.INPUT, i) for i in range(2)])
    diagram.set_boundary_outputs([PortRef(spider, Direction.OUTPUT, 0)])
    return diagram


class TestProductOfLegsEqual:
    def test_mixed_concrete_dims_validate_clean(self) -> None:
        diagram = _binder_into_spider(
            Dim.concrete(2), Dim.concrete(3), Dim.concrete(6), Dim.concrete(6)
        )
        report = validate(diagram)
        assert report.is_valid, report.errors
        assert report.issues == ()

    def test_output_that_is_not_the_product_is_a_policy_violation(self) -> None:
        diagram = _binder_into_spider(
            Dim.concrete(2), Dim.concrete(3), Dim.concrete(5), Dim.concrete(5)
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION and not issue.deferred
            for issue in report.errors
        )

    def test_symbolic_product_validates_clean_with_no_bindings(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        diagram = _binder_into_spider(d1, d2, d1 * d2, d1 * d2)
        report = validate(diagram)
        assert report.is_valid, report.errors
        assert report.issues == ()

    def test_legs_are_not_required_to_agree_with_each_other(self) -> None:
        # A B node's 2 and 3 legs disagree; only their product is checked, so no
        # ALL_LEGS_EQUAL finding may appear.
        diagram = _binder_into_spider(
            Dim.concrete(2), Dim.concrete(3), Dim.concrete(6), Dim.concrete(6)
        )
        assert all(
            issue.kind is not IssueKind.DIMENSION_POLICY_VIOLATION
            for issue in validate(diagram).issues
        )


class TestPerPortEnforcementSurvivesMixing:
    def test_a_wire_joining_dim_2_to_dim_3_is_still_a_mismatch(self) -> None:
        diagram = _binder_into_spider(
            Dim.concrete(2), Dim.concrete(3), Dim.concrete(6), Dim.concrete(7)
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_MISMATCH for issue in report.errors)


class TestGlobalConsistencyPass:
    def test_a_cross_node_contradiction_is_reported(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        node_x = diagram.add_node(Z_SPIDER, input_dims=[d, Dim.concrete(2)], output_dims=[])
        node_y = diagram.add_node(Z_SPIDER, input_dims=[d, Dim.concrete(3)], output_dims=[])
        diagram.set_boundary_inputs(
            [PortRef(node_x, Direction.INPUT, i) for i in range(2)]
            + [PortRef(node_y, Direction.INPUT, i) for i in range(2)]
        )
        report = validate(diagram)
        assert not report.is_valid
        globals_ = [
            issue
            for issue in report.issues
            if issue.kind is IssueKind.DIMENSION_GLOBALLY_INCONSISTENT
        ]
        assert len(globals_) == 1
        assert globals_[0].deferred is False

    def test_global_issues_append_after_every_per_node_issue(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        node_x = diagram.add_node(Z_SPIDER, input_dims=[d, Dim.concrete(2)], output_dims=[])
        node_y = diagram.add_node(Z_SPIDER, input_dims=[d, Dim.concrete(3)], output_dims=[])
        diagram.set_boundary_inputs(
            [PortRef(node_x, Direction.INPUT, i) for i in range(2)]
            + [PortRef(node_y, Direction.INPUT, i) for i in range(2)]
        )
        kinds = [issue.kind for issue in validate(diagram).issues]
        assert kinds[-1] is IssueKind.DIMENSION_GLOBALLY_INCONSISTENT
        assert IssueKind.DIMENSION_BOUND in kinds[:-1]

    def test_a_converged_deferred_global_solve_emits_nothing(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d1 * d2, Dim.concrete(6)], output_dims=[])
        diagram.set_boundary_inputs([PortRef(node, Direction.INPUT, i) for i in range(2)])
        report = validate(diagram)
        assert report.is_valid
        assert all(
            issue.kind
            not in (
                IssueKind.DIMENSION_GLOBALLY_INCONSISTENT,
                IssueKind.DIMENSION_GLOBAL_RESOLUTION_EXHAUSTED,
            )
            for issue in report.issues
        )

    def test_a_malformed_port_reference_skips_the_pass(self) -> None:
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.set_boundary_inputs([PortRef(node, Direction.INPUT, 7)])
        report = validate(diagram)
        assert any(issue.kind is IssueKind.PORT_INDEX_OUT_OF_RANGE for issue in report.errors)
        assert all(
            issue.kind
            not in (
                IssueKind.DIMENSION_GLOBALLY_INCONSISTENT,
                IssueKind.DIMENSION_GLOBAL_RESOLUTION_EXHAUSTED,
            )
            for issue in report.issues
        )

    def _staged_system(self) -> Diagram:
        """Three nodes asserting a*b == a, a == d1*d2, d1*d2 == 6: a system whose fixpoint
        needs more than one solve pass and still converges to DEFERRED."""
        a, b = Dim.symbol("a"), Dim.symbol("b")
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        diagram = Diagram()
        legs = ([a * b, a], [a, d1 * d2], [d1 * d2, Dim.concrete(6)])
        for dims in legs:
            node = diagram.add_node(Z_SPIDER, input_dims=list(dims), output_dims=[])
            diagram.set_boundary_inputs(
                (
                    *diagram.boundary_inputs,
                    *(PortRef(node, Direction.INPUT, i) for i in range(2)),
                )
            )
        return diagram

    def test_exhaustion_is_a_hard_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import archytaszx.algebra.dimension as dimension_module

        monkeypatch.setattr(dimension_module, "_MAX_SOLVE_PASSES", 1)

        report = validate(self._staged_system())

        assert not report.is_valid
        exhausted = [
            issue
            for issue in report.issues
            if issue.kind is IssueKind.DIMENSION_GLOBAL_RESOLUTION_EXHAUSTED
        ]
        assert len(exhausted) == 1
        assert exhausted[0].deferred is False

    def test_the_same_system_is_clean_at_the_default_budget(self) -> None:
        report = validate(self._staged_system())
        assert report.is_valid, report.errors
