"""Tests for qufzx.algebra.scalar: the Phase 2 exact scalar algebra."""

import cmath

import pytest
import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase
from qufzx.algebra.scalar import Scalar, ScalarDomainError, ScalarGrammarError


class TestConstruction:
    def test_zero(self) -> None:
        assert Scalar.zero().is_zero

    def test_one(self) -> None:
        assert Scalar.one().is_one

    def test_rational(self) -> None:
        s = Scalar.rational(1, 2)
        assert s.is_concrete
        assert s.to_complex() == 0.5

    def test_rational_zero_denominator_rejected(self) -> None:
        with pytest.raises(ScalarDomainError):
            Scalar.rational(1, 0)

    def test_gaussian_rational(self) -> None:
        s = Scalar.gaussian_rational(1, 1)
        assert s.to_complex() == complex(1, 1)

    def test_omega_concrete(self) -> None:
        s = Scalar.omega(Dim.concrete(4), 1)
        got = s.to_complex()
        expected = cmath.exp(2j * cmath.pi / 4)
        assert abs(got - expected) < 1e-12

    def test_omega_symbolic_dim(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.omega(d, 1)
        assert not s.is_concrete
        assert s.free_symbols == frozenset({"d"})

    def test_from_dim(self) -> None:
        s = Scalar.from_dim(Dim.concrete(5))
        assert s.to_complex() == complex(5, 0)

    def test_from_dim_symbolic(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.from_dim(d)
        assert not s.is_concrete
        assert s.free_symbols == frozenset({"d"})

    def test_symbol(self) -> None:
        s = Scalar.symbol("s")
        assert not s.is_concrete
        assert s.free_symbols == frozenset({"s"})

    def test_from_phase(self) -> None:
        p = Phase.turns(sp.Rational(1, 4))
        s = Scalar.from_phase(p)
        got = s.to_complex()
        expected = cmath.exp(1j * cmath.pi / 2)
        assert abs(got - expected) < 1e-12

    def test_omega_index_rejects_bool(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar.omega(Dim.concrete(4), True)


class TestOmegaModDReduction:
    def test_omega_d_to_d_is_one_when_concrete(self) -> None:
        d = Dim.concrete(4)
        s = Scalar.omega(d, 4)
        assert s == Scalar.one()

    def test_omega_d_to_j_not_claimed_symbolic(self) -> None:
        d = Dim.symbol("d")
        j = sp.Symbol("j", integer=True)
        s = Scalar.omega(d, j)
        assert s != Scalar.one()


class TestArithmeticAndEquality:
    def test_product_of_roots_of_unity_adds_indices(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.omega(d, 1) * Scalar.omega(d, 2)
        assert s == Scalar.omega(d, 3)

    def test_no_global_factor_quotient_scale_by_two(self) -> None:
        s = Scalar.symbol("s")
        assert s != s + s  # i.e. s != 2*s
        assert s != Scalar.rational(2) * s

    def test_no_global_factor_quotient_omega(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.symbol("s")
        assert s != Scalar.omega(d, 1) * s

    def test_sum(self) -> None:
        a, b = Scalar.rational(1, 2), Scalar.rational(1, 3)
        assert (a + b) == Scalar.rational(5, 6)

    def test_difference(self) -> None:
        a, b = Scalar.rational(1, 2), Scalar.rational(1, 4)
        assert (a - b) == Scalar.rational(1, 4)

    def test_negation(self) -> None:
        a = Scalar.rational(1, 2)
        assert -a == Scalar.rational(-1, 2)

    def test_power(self) -> None:
        d = Dim.concrete(4)
        s = Scalar.omega(d, 1)
        assert s**4 == Scalar.one()

    def test_conjugate_of_root_of_unity(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.omega(d, 1)
        assert s.conjugate() == Scalar.omega(d, -1)

    def test_conjugate_gaussian(self) -> None:
        s = Scalar.gaussian_rational(1, 1)
        assert s.conjugate() == Scalar.gaussian_rational(1, -1)

    def test_eq_against_non_scalar_is_false(self) -> None:
        assert (Scalar.one() == 1) is False

    def test_eq_hash_agree(self) -> None:
        a, b = Scalar.rational(1, 2), Scalar.rational(2, 4)
        assert a == b
        assert hash(a) == hash(b)

    def test_dedup_in_set(self) -> None:
        s = {Scalar.rational(1, 2), Scalar.rational(2, 4)}
        assert len(s) == 1


class TestSubstitution:
    def test_partial_substitution(self) -> None:
        s = Scalar.symbol("a") * Scalar.symbol("b")
        result = s.substitute({"a": 2})
        assert result.free_symbols == frozenset({"b"})

    def test_never_mutates(self) -> None:
        s = Scalar.symbol("a")
        s.substitute({"a": 2})
        assert not s.is_concrete

    def test_dimension_symbol_substitution(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.omega(d, 1)
        result = s.substitute({"d": 4})
        assert result.is_concrete
        expected = cmath.exp(2j * cmath.pi / 4)
        assert abs(result.to_complex() - expected) < 1e-12


class TestNumericGate:
    def test_to_complex_on_symbolic_raises(self) -> None:
        with pytest.raises(ScalarDomainError):
            Scalar.symbol("s").to_complex()

    def test_product_matches_complex_arithmetic(self) -> None:
        a = Scalar.rational(1, 2)
        b = Scalar.gaussian_rational(1, 1)
        lhs = (a * b).to_complex()
        rhs = a.to_complex() * b.to_complex()
        assert abs(lhs - rhs) < 1e-12

    def test_sum_matches_complex_arithmetic(self) -> None:
        a = Scalar.rational(1, 2)
        b = Scalar.gaussian_rational(1, 1)
        lhs = (a + b).to_complex()
        rhs = a.to_complex() + b.to_complex()
        assert abs(lhs - rhs) < 1e-12


class TestFloatRejection:
    def test_rational_rejects_float(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar.rational(1.5, 1)  # type: ignore[arg-type]

    def test_gaussian_rational_rejects_float_real(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar.gaussian_rational(0.5, 1)

    def test_gaussian_rational_rejects_float_imag(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar.gaussian_rational(1, 0.25)

    def test_gaussian_rational_rejects_bool(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar.gaussian_rational(True, 1)

    def test_bare_constructor_rejects_float(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar(1.5)

    def test_bare_constructor_rejects_float_buried_in_expression(self) -> None:
        with pytest.raises(ScalarGrammarError):
            Scalar(sp.Symbol("a") + sp.Float(0.5))

    def test_bare_constructor_accepts_exact_rational(self) -> None:
        s = Scalar(sp.Rational(1, 3))
        assert s.is_concrete

    def test_bare_constructor_accepts_integer(self) -> None:
        s = Scalar(sp.Integer(2))
        assert s.is_concrete

    def test_bare_constructor_accepts_bare_symbol(self) -> None:
        s = Scalar(sp.Symbol("a", complex=True))
        assert not s.is_concrete

    def test_bare_constructor_accepts_symbolic_omega(self) -> None:
        d = Dim.symbol("d")
        s = Scalar.omega(d, sp.Symbol("j"))
        assert not s.is_concrete


class TestNoGlobalFactorQuotientAPI:
    def test_no_method_named_normalize_or_quotient(self) -> None:
        forbidden = {"quotient_global_phase", "normalize_global_phase", "up_to_phase"}
        assert not (forbidden & set(dir(Scalar)))
