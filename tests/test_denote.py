"""Tests for qufzx.semantics.denote: the Z and X spider tensor formulas."""

from __future__ import annotations

import cmath

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram
from qufzx.semantics.denote import (
    DenoteDomainError,
    DenoteGrammarError,
    _fourier_matrix,
    denote,
    resolve_dimension,
)


class TestZSpider:
    def test_zero_to_two_at_d2_gives_ghz_pair_vector(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        tensor = denote(diagram.nodes[node])
        assert tensor.shape == (2, 2)
        np.testing.assert_allclose(tensor.flatten(), [1, 0, 0, 1])

    def test_no_phase_is_all_ones_diagonal(self) -> None:
        for d_value in (2, 3, 5):
            d = Dim.concrete(d_value)
            diagram = Diagram()
            node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
            tensor = denote(diagram.nodes[node])
            np.testing.assert_allclose(tensor, np.eye(d_value, dtype=complex))

    def test_nonzero_phase_matches_direct_exponential(self) -> None:
        for d_value in (2, 3, 5, 7):
            d = Dim.concrete(d_value)
            phase = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 3))})
            diagram = Diagram()
            node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
            tensor = denote(diagram.nodes[node])
            expected = np.zeros((d_value, d_value), dtype=complex)
            for k in range(d_value):
                angle = 2 * cmath.pi * (1 / 3) if k == 1 else 0.0
                expected[k, k] = cmath.exp(1j * angle)
            np.testing.assert_allclose(tensor, expected, atol=1e-10)

    def test_zero_in_zero_out_is_scalar_sum(self) -> None:
        d = Dim.concrete(3)
        phase = PhaseVector(
            d, {1: Phase.turns(sp.Rational(1, 3)), 2: Phase.turns(sp.Rational(1, 6))}
        )
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[], phase=phase)
        tensor = denote(diagram.nodes[node])
        assert tensor.shape == ()
        expected = sum(phase.get(k).to_complex() for k in range(3))
        np.testing.assert_allclose(complex(tensor), expected, atol=1e-10)

    def test_zero_leg_no_phase_raises_grammar_error(self) -> None:
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[])
        with pytest.raises(DenoteGrammarError):
            denote(diagram.nodes[node])

    def test_symbolic_dim_raises_domain_error(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        with pytest.raises(DenoteDomainError):
            denote(diagram.nodes[node])

    def test_symbolic_phase_raises_domain_error(self) -> None:
        d = Dim.concrete(3)
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
        with pytest.raises(DenoteDomainError):
            denote(diagram.nodes[node])

    def test_phase_dim_mismatch_raises_domain_error(self) -> None:
        d2 = Dim.concrete(2)
        d3 = Dim.concrete(3)
        phase = PhaseVector(d3, {1: Phase.turns(sp.Rational(1, 3))})
        # Force construction of a Node with mismatched leg dim and phase dim, bypassing
        # validate() -- denote() must catch this itself.
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d2], output_dims=[d2])
        diagram.set_phase(node_id, phase)
        with pytest.raises(DenoteDomainError):
            denote(diagram.nodes[node_id])

    def test_unequal_legs_raise_domain_error(self) -> None:
        d2 = Dim.concrete(2)
        d3 = Dim.concrete(3)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d2], output_dims=[d3])
        with pytest.raises(DenoteDomainError):
            denote(diagram.nodes[node_id])


class TestUnknownGenerator:
    def test_unregistered_generator_name_raises_grammar_error(self) -> None:
        from qufzx.diagram.generators import DimensionPolicy, GeneratorType, LegPolicy, PhaseSchema

        bogus = GeneratorType(
            name="W",
            leg_policy=LegPolicy(),
            phase_schema=PhaseSchema.NONE,
            dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
        )
        d = Dim.concrete(2)
        diagram = Diagram()
        node_id = diagram.add_node(bogus, input_dims=[d], output_dims=[d])
        with pytest.raises(DenoteGrammarError):
            denote(diagram.nodes[node_id])


class TestFourierMatrix:
    @pytest.mark.parametrize("d", [2, 3, 4, 5, 7])
    def test_unitary(self, d: int) -> None:
        f = _fourier_matrix(d)
        np.testing.assert_allclose(f @ f.conj().T, np.eye(d), atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3, 4, 5, 7])
    def test_symmetric(self, d: int) -> None:
        f = _fourier_matrix(d)
        np.testing.assert_allclose(f, f.T, atol=1e-12)

    def test_d2_is_hadamard(self) -> None:
        f = _fourier_matrix(2)
        np.testing.assert_allclose(f, np.array([[1, 1], [1, -1]]) / np.sqrt(2), atol=1e-12)


class TestXSpider:
    def test_matches_f_z_fdagger_independently_constructed(self) -> None:
        for d_value in (2, 3, 5):
            d = Dim.concrete(d_value)
            phase = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 4))})
            diagram_x = Diagram()
            node_x = diagram_x.add_node(X_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
            diagram_z = Diagram()
            node_z = diagram_z.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase)

            got = denote(diagram_x.nodes[node_x])
            z = denote(diagram_z.nodes[node_z])
            f = _fourier_matrix(d_value)
            expected = f @ z @ f.conj().T
            np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_d2_one_to_one_no_phase_is_identity(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[d])
        tensor = denote(diagram.nodes[node])
        np.testing.assert_allclose(tensor, np.eye(2, dtype=complex), atol=1e-10)

    def test_d2_one_to_one_pi_phase_is_bit_flip(self) -> None:
        # X_{1,1}(pi) = |+><+| - |-><-| = [[0, 1], [1, 0]], the Pauli-X-like bit flip.
        # (X_{1,1}(0) is the identity, per the general spider-fusion identity law that
        # holds for any color at phase 0 -- see the note in the module docstring.)
        d = Dim.concrete(2)
        phase = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 2))})
        diagram = Diagram()
        node = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
        tensor = denote(diagram.nodes[node])
        np.testing.assert_allclose(tensor, np.array([[0, 1], [1, 0]]), atol=1e-10)

    def test_d2_zero_to_one_no_phase_matches_plus_minus_bra_ket_formula(self) -> None:
        # X_{0,1}(0) = |+> + |-> = sqrt(2)|0>, by the standard bra-ket formula
        # X_{m,n}(alpha) = |+>^n <+|^m + e^{i alpha} |->^n <-|^m at m=0, n=1, alpha=0.
        d = Dim.concrete(2)
        diagram = Diagram()
        node = diagram.add_node(X_SPIDER, input_dims=[], output_dims=[d])
        tensor = denote(diagram.nodes[node])
        plus = np.array([1, 1]) / np.sqrt(2)
        minus = np.array([1, -1]) / np.sqrt(2)
        np.testing.assert_allclose(tensor, plus + minus, atol=1e-10)


class TestResolveDimension:
    def test_derives_from_phase_when_no_legs(self) -> None:
        d = Dim.concrete(4)
        phase = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 4))})
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[], phase=phase)
        assert resolve_dimension(diagram.nodes[node_id]) == 4

    def test_derives_from_legs_when_present(self) -> None:
        d = Dim.concrete(5)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        assert resolve_dimension(diagram.nodes[node_id]) == 5
