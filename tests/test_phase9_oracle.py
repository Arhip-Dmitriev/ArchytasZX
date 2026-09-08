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


"""Phase 9 done-when: the character sum, and one root-of-unity rule verified two ways."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar, ScalarBudgetError, ScalarGrammarError, ScalarSumError
from qufzx.diagram.bangbox import Mult
from qufzx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import (
    CAP_SIDE_CONDITIONS,
    FOURIER_SIDE_CONDITIONS,
    CapMatch,
    FourierMatch,
    find_cap_matches,
    find_fourier_matches,
)
from qufzx.rewrite.rule import RewriteDomainError, SideConditionOutcome
from qufzx.rewrite.rules_library import FOURIER_CANCELLATION, RULES, ZX_CAP
from qufzx.semantics.check import compare
from qufzx.semantics.contract_symbolic import contract_symbolic
from qufzx.semantics.denote import DenoteGrammarError, denote

D = Dim.symbol("d")


def _integer(name: str) -> sp.Symbol:
    return sp.Symbol(name, integer=True, nonnegative=True)


def _fourier_chain(dim: Dim, count: int) -> Diagram:
    diagram = Diagram()
    ids = [diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim]) for _ in range(count)]
    for left, right in itertools.pairwise(ids):
        diagram.add_wire(PortRef(left, Direction.OUTPUT, 0), PortRef(right, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(ids[0], Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(ids[-1], Direction.OUTPUT, 0)])
    return diagram


def _identity_wire(dim: Dim) -> Diagram:
    diagram = Diagram()
    node_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    diagram.set_boundary_inputs([PortRef(node_id, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(node_id, Direction.OUTPUT, 0)])
    return diagram


class TestTheCharacterSum:
    """FULL_PLAN.md Phase 9, test clause one."""

    def test_a_constant_summand_gives_the_dimension(self) -> None:
        assert Scalar.index_sum(D, lambda _k: Scalar.one()).simplify() == Scalar.from_dim(D)

    def test_an_index_that_is_a_multiple_of_d_closes_to_d(self) -> None:
        m = _integer("m")
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, m * D.to_sympy() * k.to_sympy()))
        assert summed.simplify() == Scalar.from_dim(D)

    def test_a_zero_index_closes_to_d_with_d_symbolic(self) -> None:
        assert Scalar.index_sum(D, lambda k: Scalar.omega(D, 0 * k.to_sympy())).simplify() == (
            Scalar.from_dim(D)
        )

    @pytest.mark.parametrize("value,index", [(4, 1), (4, 2), (4, 3), (5, 1), (6, 4)])
    def test_a_concrete_nonzero_index_vanishes(self, value: int, index: int) -> None:
        dim = Dim.concrete(value)
        summed = Scalar.index_sum(dim, lambda k: Scalar.omega(dim, index * k.to_sympy()))
        assert summed.simplify() == Scalar.zero()

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5, 6])
    def test_it_agrees_with_direct_summation_at_concrete_d(self, value: int) -> None:
        dim = Dim.concrete(value)
        for index in range(value + 2):

            def summand(k: Scalar, j: int = index) -> Scalar:
                return Scalar.omega(dim, j * k.to_sympy())

            closed = Scalar.index_sum(dim, summand).simplify()
            direct = sum(np.exp(2j * np.pi * index * k / value) for k in range(value))
            assert np.isclose(closed.to_complex(), direct)

    def test_a_concrete_index_at_symbolic_d_stays_unevaluated(self) -> None:
        """The lemma: d = 1 is always live, so no universal negative is provable."""
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, 3 * k.to_sympy())).simplify()
        assert summed != Scalar.zero()
        assert summed.to_sympy().atoms(sp.Sum)

    def test_the_degenerate_dimension_one_gives_one(self) -> None:
        one = Dim.concrete(1)
        assert Scalar.index_sum(one, lambda k: Scalar.omega(one, k.to_sympy())).simplify() == (
            Scalar.one()
        )

    def test_orthogonality_closes_on_the_index_difference(self) -> None:
        m = _integer("m")
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, (m * D.to_sympy()) * k.to_sympy()))
        assert summed.simplify() == Scalar.from_dim(D)

    def test_a_shift_factor_is_never_dropped(self) -> None:
        offset = _integer("c")
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, offset + 0 * k.to_sympy()))
        result = summed.simplify()
        assert result == Scalar.omega(D, offset) * Scalar.from_dim(D)
        assert result != Scalar.from_dim(D)

    def test_a_free_scalar_factor_is_pulled_out_and_kept(self) -> None:
        s = Scalar.symbol("s")
        summed = Scalar.index_sum(D, lambda _k: s)
        assert summed.simplify() == s * Scalar.from_dim(D)

    def test_simplify_is_idempotent(self) -> None:
        a, b = _integer("a"), _integer("b")
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, k.to_sympy() * (a - b)))
        once = summed.simplify()
        assert once.simplify() == once

    def test_the_budget_is_enforced_and_nothing_partial_escapes(self) -> None:
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, k.to_sympy()))
        before = summed.to_sympy()
        with pytest.raises(ScalarBudgetError, match="step budget"):
            summed.simplify(max_steps=0)
        assert summed.to_sympy() == before


class TestTheScalarGrammar:
    def test_the_engine_index_namespace_is_reserved(self) -> None:
        for name in ("_k0", "_i0", "_k12"):
            with pytest.raises(ScalarGrammarError, match="reserved"):
                Scalar.symbol(name)

    def test_a_bound_index_cannot_be_substituted(self) -> None:
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, 3 * k.to_sympy())).simplify()
        with pytest.raises(ScalarSumError, match="bound summation index"):
            summed.substitute({"_k0": 1})

    def test_substituting_d_rewrites_the_limit_without_expanding(self) -> None:
        summed = Scalar.index_sum(D, lambda k: Scalar.omega(D, 3 * k.to_sympy())).simplify()
        answered = summed.substitute({"d": 4096})
        sums = answered.to_sympy().atoms(sp.Sum)
        assert len(sums) == 1
        assert next(iter(sums)).limits[0][2] == sp.Integer(4095)

    def test_a_rational_dimension_power_is_exact(self) -> None:
        assert Scalar.dim_power(D, -1, 2) ** 2 * Scalar.from_dim(D) == Scalar.one()


class TestFourierCancellationVerifiedTwoWays:
    """FULL_PLAN.md Phase 9, test clause two: symbolically in d, and on the numeric oracle."""

    def test_verified_by_symbolic_contraction_with_d_formal(self) -> None:
        assert (
            contract_symbolic(_fourier_chain(D, 4)).entry
            == contract_symbolic(_identity_wire(D)).entry
        )

    @pytest.mark.parametrize("value", [2, 3, 4, 5, 7])
    def test_verified_by_the_numeric_oracle(self, value: int) -> None:
        dim = Dim.concrete(value)
        before = _fourier_chain(dim, 4)
        match = find_fourier_matches(before)[0]
        after = apply(before, FOURIER_CANCELLATION, match).diagram
        assert not validate(after).errors
        assert compare(before, after, {}).matched

    def test_the_introduced_scalar_is_exactly_one(self) -> None:
        assert FOURIER_CANCELLATION.scalar_introduced == Scalar.one()

    @pytest.mark.parametrize("power", [1, -1, -2, 2])
    def test_a_wrong_global_factor_is_caught(self, power: int) -> None:
        """The d from the character sum must cancel the four d^(-1/2) factors exactly."""
        dim = Dim.concrete(3)
        before = _fourier_chain(dim, 4)
        shifted = _identity_wire(dim)
        shifted.multiply_scalar(Scalar.from_dim(dim) ** power)
        assert not compare(before, shifted, {}).matched

    def test_the_rule_is_registered(self) -> None:
        assert RULES["fourier_cancellation"] is FOURIER_CANCELLATION


class TestTheFourierMatcher:
    @pytest.mark.parametrize("count,expected", [(1, 0), (2, 0), (3, 0), (4, 1), (5, 2), (6, 3)])
    def test_only_chains_of_four_match(self, count: int, expected: int) -> None:
        assert len(find_fourier_matches(_fourier_chain(Dim.concrete(2), count))) == expected

    def test_matches_are_ordered_by_the_chains_first_node(self) -> None:
        matches = find_fourier_matches(_fourier_chain(Dim.concrete(2), 6))
        starts = [match.node_ids[0] for match in matches]
        assert starts == sorted(starts)

    def test_a_chain_broken_by_a_spider_does_not_match(self) -> None:
        dim = Dim.concrete(2)
        diagram = Diagram()
        boxes = [
            diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim]) for _ in range(3)
        ]
        spider = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        order = [boxes[0], boxes[1], spider, boxes[2]]
        for left, right in itertools.pairwise(order):
            diagram.add_wire(PortRef(left, Direction.OUTPUT, 0), PortRef(right, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(order[0], Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(order[-1], Direction.OUTPUT, 0)])
        assert find_fourier_matches(diagram) == ()


class TestFourierBuilderRederivesItsMatch:
    """A fabricated FourierMatch never reaches graph surgery."""

    @staticmethod
    def _chain(length: int, d: int = 3) -> tuple[Diagram, list[int]]:
        dim = Dim.concrete(d)
        diagram = Diagram()
        ids = [
            diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim])
            for _ in range(length)
        ]
        for first, second in itertools.pairwise(ids):
            diagram.add_wire(
                PortRef(first, Direction.OUTPUT, 0), PortRef(second, Direction.INPUT, 0)
            )
        diagram.set_boundary_inputs([PortRef(ids[0], Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(ids[-1], Direction.OUTPUT, 0)])
        return diagram, ids

    @pytest.mark.parametrize("length", [2, 3, 5])
    def test_a_chain_that_is_not_four_boxes_is_refused(self, length: int) -> None:
        diagram, ids = self._chain(length)
        fabricated = FourierMatch(
            node_ids=tuple(ids),
            wires=tuple(diagram.wires),
            shared_dim=Dim.concrete(3),
            side_condition_outcomes=tuple(
                SideConditionOutcome(condition.name, True, "fabricated")
                for condition in FOURIER_SIDE_CONDITIONS
            ),
        )
        with pytest.raises(RewriteDomainError):
            apply(diagram, FOURIER_CANCELLATION, fabricated)

    def test_the_genuine_four_box_chain_still_fires(self) -> None:
        diagram, _ids = self._chain(4)
        matches = find_fourier_matches(diagram)
        assert len(matches) == 1
        assert apply(diagram, FOURIER_CANCELLATION, matches[0]).diagram is not None


class TestFourierBoxArity:
    """An F box denotes a d-by-d matrix and takes exactly one leg of each direction."""

    @pytest.mark.parametrize("inputs,outputs", [(2, 1), (1, 2), (3, 3), (0, 1)])
    def test_a_wrong_arity_f_box_is_refused(self, inputs: int, outputs: int) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        node_id = diagram.add_node(
            FOURIER_BOX, input_dims=[dim] * inputs, output_dims=[dim] * outputs
        )
        with pytest.raises(DenoteGrammarError):
            denote(diagram.nodes[node_id])

    def test_the_one_in_one_out_f_box_still_denotes(self) -> None:
        dim = Dim.concrete(3)
        diagram = Diagram()
        node_id = diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim])
        assert denote(diagram.nodes[node_id]).shape == (3, 3)


class TestFourierCancellationUnderABangBox:
    """The rule fires inside a node-scope box and leaves the box intact (Phase 7)."""

    @staticmethod
    def _boxed_chain(d_value: int) -> tuple[Diagram, list[int]]:
        dim = Dim.concrete(d_value)
        diagram = Diagram()
        ids = [diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim]) for _ in range(4)]
        for first, second in itertools.pairwise(ids):
            diagram.add_wire(
                PortRef(first, Direction.OUTPUT, 0), PortRef(second, Direction.INPUT, 0)
            )
        diagram.set_boundary_inputs([PortRef(ids[0], Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(ids[-1], Direction.OUTPUT, 0)])
        diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset(ids))
        return diagram, ids

    def test_the_chain_collapses_and_the_box_gains_the_identity_spider(self) -> None:
        diagram, ids = self._boxed_chain(3)
        assert validate(diagram).is_valid
        matches = find_fourier_matches(diagram)
        assert len(matches) == 1
        fused = apply(diagram, FOURIER_CANCELLATION, matches[0]).diagram
        report = validate(fused)
        assert report.is_valid, [issue.kind.value for issue in report.errors]
        (box,) = fused.bang_boxes.values()
        assert str(box.multiplicity) == "m"
        assert box.node_scope == frozenset(fused.nodes)
        assert not (box.node_scope & set(ids))


class TestZXCapRootOfUnityScalar:
    """Phase 9 iii's rule that needs a root-of-unity scalar, with its exact scalar output."""

    @staticmethod
    def _cap(dim: Dim) -> Diagram:
        diagram = Diagram()
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
        diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
        return diagram

    def test_the_scalar_it_introduces_is_not_one(self) -> None:
        assert not ZX_CAP.scalar_introduced.is_one
        assert ZX_CAP.scalar_introduced == Scalar.dim_power(D, 1, 2)
        assert ZX_CAP.scalar_for(Dim.concrete(9)) == Scalar.dim_power(Dim.concrete(9), 1, 2)

    @pytest.mark.parametrize("d_value", [2, 3, 4, 5, 6, 7])
    def test_the_numeric_oracle_confirms_the_rewrite(self, d_value: int) -> None:
        """First of the two independent checks Phase 9 asks for: the oracle at concrete d."""
        before = self._cap(Dim.concrete(d_value))
        matches = find_cap_matches(before)
        assert len(matches) == 1
        result = apply(before, ZX_CAP, matches[0])
        assert result.step.scalar_introduced == Scalar.dim_power(Dim.concrete(d_value), 1, 2)
        assert not result.diagram.nodes
        assert validate(result.diagram).is_valid
        comparison = compare(before, result.diagram, {})
        assert comparison.matched, comparison.reason

    def test_full_symbolic_contraction_in_d_confirms_the_same_scalar(self) -> None:
        """Second, independent of the oracle: contract with d formal and close the sum."""
        before = self._cap(D)
        result = apply(before, ZX_CAP, find_cap_matches(before)[0])
        contracted_before = contract_symbolic(before).entry.simplify()
        contracted_after = contract_symbolic(result.diagram).entry.simplify()
        assert contracted_before == Scalar.dim_power(D, 1, 2)
        assert contracted_before == contracted_after

    def test_a_dropped_factor_would_be_caught(self) -> None:
        """The scalar is load-bearing: without it the two sides disagree."""
        before = self._cap(Dim.concrete(3))
        result = apply(before, ZX_CAP, find_cap_matches(before)[0])
        unscaled = result.diagram.copy()
        unscaled.multiply_scalar(Scalar.dim_power(Dim.concrete(3), -1, 2))
        assert not compare(before, unscaled, {}).matched

    def test_the_builder_refuses_a_fabricated_match(self) -> None:
        phased = self._cap(Dim.concrete(3))
        fabricated = CapMatch(
            state_id=0,
            effect_id=1,
            wire=next(iter(phased.wires)),
            shared_dim=Dim.concrete(5),
            side_condition_outcomes=tuple(
                SideConditionOutcome(condition.name, True, "fabricated")
                for condition in CAP_SIDE_CONDITIONS
            ),
        )
        with pytest.raises(RewriteDomainError):
            apply(phased, ZX_CAP, fabricated)

    def test_a_phased_or_boundary_cap_is_not_matched(self) -> None:
        dim = Dim.concrete(3)
        boundary = self._cap(dim)
        boundary.set_boundary_outputs([PortRef(0, Direction.OUTPUT, 0)])
        assert find_cap_matches(boundary) == ()

        phased = Diagram()
        state = phased.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[dim],
            phase=PhaseVector(dim, {1: Phase.symbol("a")}),
        )
        effect = phased.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
        phased.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
        assert find_cap_matches(phased) == ()
