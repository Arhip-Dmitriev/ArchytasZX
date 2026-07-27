"""Tests for qufzx.semantics.check: instantiation and oracle comparison."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.semantics.check import (
    CheckGrammarError,
    ComparisonResult,
    EqualityMode,
    compare,
    compare_tensors,
    instantiate,
    score,
)

from .helpers import build_ghz_with_copy


class TestInstantiate:
    def test_substitutes_dim_and_phase(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, _a, b_id = build_ghz_with_copy(d, phase_on_b=phase)
        instantiated = instantiate(diagram, {"d": 3, "alpha": sp.Rational(1, 6)})
        node = instantiated.nodes[b_id]
        assert node.inputs[0].dim == Dim.concrete(3)
        assert node.phase is not None
        assert node.phase.get(1) == Phase.turns(sp.Rational(1, 6))

    def test_missing_symbol_raises(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, _a, _b = build_ghz_with_copy(d, phase_on_b=phase)
        with pytest.raises(CheckGrammarError):
            instantiate(diagram, {"d": 3})

    def test_fully_concrete_diagram_needs_no_assignment(self) -> None:
        d = Dim.concrete(3)
        diagram, _a, _b = build_ghz_with_copy(d)
        instantiated = instantiate(diagram, {})
        assert instantiated.nodes.keys() == diagram.nodes.keys()


class TestScore:
    def test_scores_symbolic_ghz_at_several_d(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        for d_value in (2, 3, 5):
            result = score(diagram, {"d": d_value})
            assert result.tensor.shape == (d_value,) * 3
            expected = np.zeros((d_value,) * 3, dtype=complex)
            for k in range(d_value):
                expected[k, k, k] = 1
            np.testing.assert_allclose(result.tensor, expected)

    def test_missing_symbol_refused(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        with pytest.raises(CheckGrammarError):
            score(diagram, {})


class TestCompareTensorsExact:
    def test_identical_tensors_match(self) -> None:
        a = np.array([1.0 + 0j, 0j])
        result = compare_tensors(a, a.copy())
        assert result.matched
        assert result.mode is EqualityMode.EXACT

    def test_different_tensors_do_not_match(self) -> None:
        a = np.array([1.0 + 0j])
        b = np.array([2.0 + 0j])
        result = compare_tensors(a, b)
        assert not result.matched

    def test_shape_mismatch_does_not_match(self) -> None:
        a = np.zeros((2, 2))
        b = np.zeros((2, 2, 2))
        result = compare_tensors(a, b)
        assert not result.matched
        assert result.max_abs_deviation == float("inf")


class TestCompareTensorsUpToGlobalPhase:
    def test_unit_modulus_shift_matches_only_in_global_phase_mode(self) -> None:
        a = np.array([1.0 + 0j, 1j])
        lam = Scalar.omega(Dim.concrete(4), 1).to_complex()
        b = a * lam

        exact = compare_tensors(a, b, mode=EqualityMode.EXACT)
        assert not exact.matched

        phase_mode = compare_tensors(a, b, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert phase_mode.matched
        assert phase_mode.recovered_lambda is not None
        assert abs(abs(phase_mode.recovered_lambda) - 1.0) < 1e-9

    def test_shift_by_two_fails_in_both_modes(self) -> None:
        a = np.array([1.0 + 0j, 1j])
        b = a * 2

        exact = compare_tensors(a, b, mode=EqualityMode.EXACT)
        assert not exact.matched

        phase_mode = compare_tensors(a, b, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert not phase_mode.matched

    def test_both_zero_matches(self) -> None:
        a = np.zeros(3, dtype=complex)
        b = np.zeros(3, dtype=complex)
        result = compare_tensors(a, b, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert result.matched

    def test_one_zero_one_nonzero_does_not_match(self) -> None:
        a = np.zeros(3, dtype=complex)
        b = np.array([1.0, 0.0, 0.0], dtype=complex)
        result = compare_tensors(a, b, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert not result.matched


class TestCompareDiagrams:
    def test_scalar_shifted_copy_of_ghz_fails_exact_passes_global_phase(self) -> None:
        d = Dim.concrete(2)
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)
        diagram_b.multiply_scalar(Scalar.omega(Dim.concrete(4), 1))

        exact = compare(diagram_a, diagram_b, {})
        assert not exact.matched

        phase_mode = compare(diagram_a, diagram_b, {}, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert phase_mode.matched

    def test_scalar_shift_by_two_fails_both_modes(self) -> None:
        d = Dim.concrete(2)
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)
        diagram_b.multiply_scalar(Scalar.rational(2))

        exact = compare(diagram_a, diagram_b, {})
        assert not exact.matched
        phase_mode = compare(diagram_a, diagram_b, {}, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert not phase_mode.matched

    def test_identical_symbolic_diagrams_match_at_several_d(self) -> None:
        d = Dim.symbol("d")
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)
        for d_value in (2, 3, 5):
            result = compare(diagram_a, diagram_b, {"d": d_value})
            assert result.matched

    def test_default_mode_is_exact(self) -> None:
        d = Dim.concrete(2)
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)
        diagram_b.multiply_scalar(Scalar.omega(Dim.concrete(4), 1))
        result = compare(diagram_a, diagram_b, {})
        assert result.mode is EqualityMode.EXACT
        assert not result.matched


class TestComparisonResultStructure:
    def test_reason_and_deviation_are_populated(self) -> None:
        a = np.array([1.0 + 0j])
        b = np.array([2.0 + 0j])
        result = compare_tensors(a, b)
        assert isinstance(result, ComparisonResult)
        assert result.reason
        assert result.max_abs_deviation > 0


class TestUnknownMode:
    def test_unknown_mode_raises_grammar_error(self) -> None:
        a = np.array([1.0])
        b = np.array([1.0])
        with pytest.raises(CheckGrammarError):
            compare_tensors(a, b, mode="bogus")  # type: ignore[arg-type]


class TestCompareInterfaceMismatch:
    def test_different_output_input_split_does_not_match(self) -> None:
        d = Dim.concrete(2)
        two_out_one_in = Diagram()
        node_id = two_out_one_in.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
        two_out_one_in.set_boundary_outputs(
            [PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.OUTPUT, 1)]
        )
        two_out_one_in.set_boundary_inputs([PortRef(node_id, Direction.INPUT, 0)])

        one_out_two_in = Diagram()
        node_id2 = one_out_two_in.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[d])
        one_out_two_in.set_boundary_outputs([PortRef(node_id2, Direction.OUTPUT, 0)])
        one_out_two_in.set_boundary_inputs(
            [PortRef(node_id2, Direction.INPUT, 0), PortRef(node_id2, Direction.INPUT, 1)]
        )

        result = compare(two_out_one_in, one_out_two_in, {})
        assert not result.matched
        assert "arity" in result.reason
        assert "deviation" not in result.reason

    def test_same_arity_different_dimensions_does_not_match(self) -> None:
        diagram_a = Diagram()
        node_a1 = diagram_a.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        node_a2 = diagram_a.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram_a.set_boundary_outputs(
            [PortRef(node_a1, Direction.OUTPUT, 0), PortRef(node_a2, Direction.OUTPUT, 0)]
        )

        diagram_b = Diagram()
        node_b1 = diagram_b.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        node_b2 = diagram_b.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(3)])
        diagram_b.set_boundary_outputs(
            [PortRef(node_b1, Direction.OUTPUT, 0), PortRef(node_b2, Direction.OUTPUT, 0)]
        )

        result = compare(diagram_a, diagram_b, {})
        assert not result.matched
        assert "dimension" in result.reason

    def test_reordered_symmetric_boundary_does_not_spuriously_match(self) -> None:
        d = Dim.concrete(2)

        forward = Diagram()
        node_id = forward.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        forward.set_boundary_outputs(
            [PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.OUTPUT, 1)]
        )

        reversed_ = Diagram()
        node_id2 = reversed_.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        reversed_.set_boundary_outputs(
            [PortRef(node_id2, Direction.OUTPUT, 1), PortRef(node_id2, Direction.OUTPUT, 0)]
        )

        result = compare(forward, reversed_, {})
        assert not result.matched
        assert "order" in result.reason

    def test_independently_built_ghz_copies_still_match(self) -> None:
        d = Dim.concrete(3)
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)

        result = compare(diagram_a, diagram_b, {})
        assert result.matched
        assert result.reason != ""
