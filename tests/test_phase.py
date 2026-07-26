"""Tests for qufzx.algebra.phase: the Phase 2 phase algebra."""

import cmath

import pytest
import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import (
    Phase,
    PhaseDomainError,
    PhaseGrammarError,
    PhaseVector,
)


class TestPhaseConstruction:
    def test_zero(self) -> None:
        assert Phase.zero().is_zero

    def test_turns_int(self) -> None:
        p = Phase.turns(1)
        assert p.is_zero

    def test_turns_rational(self) -> None:
        p = Phase.turns(sp.Rational(1, 2))
        assert p.is_concrete
        assert not p.is_zero

    def test_turns_rejects_float(self) -> None:
        with pytest.raises(PhaseGrammarError):
            Phase.turns(0.5)

    def test_root_of_unity_concrete(self) -> None:
        p = Phase.root_of_unity(1, Dim.concrete(4))
        assert p.is_concrete
        assert p == Phase.turns(sp.Rational(1, 4))

    def test_root_of_unity_symbolic_dim(self) -> None:
        d = Dim.symbol("d")
        p = Phase.root_of_unity(1, d)
        assert not p.is_concrete
        assert p.free_symbols == frozenset({"d"})

    def test_root_of_unity_symbolic_index(self) -> None:
        d = Dim.concrete(4)
        j = sp.Symbol("j", integer=True)
        p = Phase.root_of_unity(j, d)
        assert p.free_symbols == frozenset({"j"})

    def test_symbol(self) -> None:
        p = Phase.symbol("theta")
        assert not p.is_concrete
        assert p.free_symbols == frozenset({"theta"})


class TestPhaseAddition:
    def test_symbol_plus_symbol(self) -> None:
        a, b = Phase.symbol("alpha"), Phase.symbol("beta")
        s = a + b
        assert s.free_symbols == frozenset({"alpha", "beta"})
        assert not s.is_concrete

    def test_symbol_plus_concrete(self) -> None:
        a = Phase.symbol("alpha")
        s = a + Phase.turns(sp.Rational(1, 4))
        assert s.free_symbols == frozenset({"alpha"})
        assert not s.is_concrete

    def test_root_of_unity_plus_root_of_unity_same_symbolic_dim(self) -> None:
        d = Dim.symbol("d")
        p1 = Phase.root_of_unity(1, d)
        p2 = Phase.root_of_unity(2, d)
        s = p1 + p2
        assert s == Phase.root_of_unity(3, d)

    def test_concrete_plus_concrete_reduces_mod_one_turn(self) -> None:
        a = Phase.turns(sp.Rational(3, 4))
        b = Phase.turns(sp.Rational(3, 4))
        s = a + b
        assert s == Phase.turns(sp.Rational(1, 2))

    def test_subtraction(self) -> None:
        a = Phase.turns(sp.Rational(1, 2))
        b = Phase.turns(sp.Rational(1, 4))
        assert (a - b) == Phase.turns(sp.Rational(1, 4))

    def test_negation(self) -> None:
        a = Phase.turns(sp.Rational(1, 4))
        assert (-a) == Phase.turns(sp.Rational(3, 4))

    def test_scale(self) -> None:
        a = Phase.turns(sp.Rational(1, 4))
        assert a.scale(2) == Phase.turns(sp.Rational(1, 2))


class TestModOneTurnNormalization:
    def test_concrete_rational_reduced(self) -> None:
        p = Phase.turns(sp.Rational(5, 4))
        assert p == Phase.turns(sp.Rational(1, 4))

    def test_symbolic_expression_not_reduced(self) -> None:
        d = Dim.symbol("d")
        j = sp.Symbol("j", integer=True)
        p = Phase.root_of_unity(j, d)
        # j/d with symbolic d cannot be reduced mod 1; the raw expression survives.
        assert p.to_sympy_turns() == j / d.to_sympy()

    def test_bare_symbol_not_reduced(self) -> None:
        p = Phase.symbol("theta")
        assert p.to_sympy_turns() == sp.Symbol("theta", real=True)


class TestFloatRejection:
    def test_bare_constructor_rejects_float(self) -> None:
        with pytest.raises(PhaseGrammarError):
            Phase(0.5)

    def test_bare_constructor_rejects_float_buried_in_expression(self) -> None:
        with pytest.raises(PhaseGrammarError):
            Phase(sp.Symbol("a") + sp.Float(0.5))

    def test_bare_constructor_accepts_exact_rational(self) -> None:
        p = Phase(sp.Rational(1, 3))
        assert p.is_concrete

    def test_bare_constructor_accepts_integer(self) -> None:
        p = Phase(sp.Integer(2))
        assert p.is_concrete

    def test_bare_constructor_accepts_bare_symbol(self) -> None:
        p = Phase(sp.Symbol("theta", real=True))
        assert not p.is_concrete

    def test_bare_constructor_accepts_symbolic_root_of_unity(self) -> None:
        d = Dim.symbol("d")
        j = sp.Symbol("j", integer=True)
        p = Phase.root_of_unity(j, d)
        assert not p.is_concrete


class TestSubstitution:
    def test_partial_substitution(self) -> None:
        a, b = Phase.symbol("alpha"), Phase.symbol("beta")
        s = (a + b).substitute({"alpha": sp.Rational(1, 4)})
        assert s.free_symbols == frozenset({"beta"})

    def test_full_substitution_yields_concrete(self) -> None:
        a = Phase.symbol("alpha")
        s = a.substitute({"alpha": sp.Rational(1, 2)})
        assert s.is_concrete
        assert s == Phase.turns(sp.Rational(1, 2))

    def test_substitute_does_not_mutate(self) -> None:
        a = Phase.symbol("alpha")
        a.substitute({"alpha": 1})
        assert not a.is_concrete

    def test_root_of_unity_dimension_symbol_substitution(self) -> None:
        d = Dim.symbol("d")
        p = Phase.root_of_unity(1, d)
        s = p.substitute({"d": 4})
        assert s == Phase.turns(sp.Rational(1, 4))


class TestNumericGate:
    def test_to_complex_on_symbolic_raises(self) -> None:
        with pytest.raises(PhaseDomainError):
            Phase.symbol("theta").to_complex()

    def test_to_complex_matches_cmath(self) -> None:
        p = Phase.turns(sp.Rational(1, 4))
        got = p.to_complex()
        expected = cmath.exp(1j * cmath.pi / 2)
        assert abs(got - expected) < 1e-12

    def test_addition_homomorphism(self) -> None:
        a = Phase.turns(sp.Rational(1, 3))
        b = Phase.turns(sp.Rational(1, 5))
        lhs = (a + b).to_complex()
        rhs = a.to_complex() * b.to_complex()
        assert abs(lhs - rhs) < 1e-12


class TestEqualityAndHashing:
    def test_eq_hash_agree(self) -> None:
        a = Phase.turns(sp.Rational(1, 2))
        b = Phase.turns(sp.Rational(3, 6))
        assert a == b
        assert hash(a) == hash(b)

    def test_eq_against_non_phase_is_false(self) -> None:
        assert (Phase.zero() == 0) is False

    def test_dedup_in_set(self) -> None:
        s = {Phase.turns(sp.Rational(1, 2)), Phase.turns(sp.Rational(3, 6))}
        assert len(s) == 1


class TestPhaseVectorConstruction:
    def test_symbolic_dim_constructible(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        assert pv.dim == d

    def test_symbolic_length_is_none(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        assert pv.length is None

    def test_concrete_length(self) -> None:
        d = Dim.concrete(3)
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        assert pv.length == 2

    def test_index_zero_zero_phase_allowed(self) -> None:
        d = Dim.concrete(3)
        pv = PhaseVector(d, {0: Phase.zero(), 1: Phase.symbol("a")})
        assert pv.get(0).is_zero

    def test_index_zero_nonzero_phase_rejected(self) -> None:
        d = Dim.concrete(3)
        with pytest.raises(PhaseDomainError):
            PhaseVector(d, {0: Phase.turns(sp.Rational(1, 4))})

    def test_out_of_range_index_rejected_at_concrete_dim(self) -> None:
        d = Dim.concrete(3)
        with pytest.raises(PhaseDomainError):
            PhaseVector(d, {5: Phase.symbol("a")})

    def test_same_index_accepted_at_symbolic_dim(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {5: Phase.symbol("a")})
        assert pv.get(5).free_symbols == frozenset({"a"})

    def test_index_rejected_after_dim_substituted(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {5: Phase.symbol("a")})
        with pytest.raises(PhaseDomainError):
            pv.substitute({"d": 3})

    def test_negative_index_rejected(self) -> None:
        d = Dim.symbol("d")
        with pytest.raises(PhaseDomainError):
            PhaseVector(d, {-1: Phase.symbol("a")})

    def test_explicit_zero_entry_dropped(self) -> None:
        d = Dim.concrete(3)
        pv = PhaseVector(d, {1: Phase.zero()})
        assert pv.is_zero
        assert pv.entries() == {}


class TestPhaseVectorAddition:
    def test_componentwise_addition(self) -> None:
        d = Dim.concrete(3)
        pv1 = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 4))})
        pv2 = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 4)), 2: Phase.turns(sp.Rational(1, 2))})
        result = pv1 + pv2
        assert result.get(1) == Phase.turns(sp.Rational(1, 2))
        assert result.get(2) == Phase.turns(sp.Rational(1, 2))

    def test_mismatched_dim_raises(self) -> None:
        pv1 = PhaseVector(Dim.concrete(3), {1: Phase.zero()})
        pv2 = PhaseVector(Dim.concrete(4), {1: Phase.zero()})
        with pytest.raises(PhaseDomainError):
            pv1 + pv2

    def test_addition_over_symbolic_dim(self) -> None:
        d = Dim.symbol("d")
        pv1 = PhaseVector(d, {1: Phase.symbol("a")})
        pv2 = PhaseVector(d, {1: Phase.symbol("b")})
        result = pv1 + pv2
        assert result.get(1).free_symbols == frozenset({"a", "b"})


class TestPhaseVectorSubstitution:
    def test_partial_substitution_never_mutates(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        pv.substitute({"a": sp.Rational(1, 4)})
        assert not pv.is_concrete

    def test_full_substitution_yields_concrete_vector(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        result = pv.substitute({"d": 3, "a": sp.Rational(1, 4)})
        assert result.is_concrete
        assert result.length == 2

    def test_sp_integer_dim_substitution_concretizes_dim(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        result = pv.substitute({"d": sp.Integer(3)})
        assert result.dim == Dim.concrete(3)
        assert result.length == 2

    def test_sp_integer_dim_substitution_raises_on_out_of_range_index(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {5: Phase.turns(sp.Rational(1, 3))})
        # Matches the int case: 5 is out of range for a concretized dim of 3.
        with pytest.raises(PhaseDomainError):
            pv.substitute({"d": 3})
        with pytest.raises(PhaseDomainError):
            pv.substitute({"d": sp.Integer(3)})

    def test_integral_rational_dim_substitution_concretizes_dim(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        result = pv.substitute({"d": sp.Rational(3, 1)})
        assert result.dim == Dim.concrete(3)

    def test_non_integral_dim_substitution_rejected(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        with pytest.raises(PhaseDomainError):
            pv.substitute({"d": sp.Rational(3, 2)})

    def test_phase_value_for_dim_symbol_rejected(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a")})
        with pytest.raises(PhaseDomainError):
            pv.substitute({"d": Phase.turns(sp.Rational(1, 2))})

    def test_non_integral_dim_substitution_does_not_desync_or_mutate(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {5: Phase.turns(sp.Rational(1, 3))})
        with pytest.raises(PhaseDomainError):
            pv.substitute({"d": sp.Rational(3, 2)})
        # Original vector is untouched: dim still symbolic, entry still present.
        assert not pv.dim.is_concrete
        assert pv.get(5) == Phase.turns(sp.Rational(1, 3))

    def test_partial_substitution_leaves_unmentioned_phase_symbols_symbolic(self) -> None:
        d = Dim.symbol("d")
        pv = PhaseVector(d, {1: Phase.symbol("a"), 2: Phase.symbol("b")})
        result = pv.substitute({"d": 4, "a": sp.Rational(1, 4)})
        assert result.get(1) == Phase.turns(sp.Rational(1, 4))
        assert result.get(2).free_symbols == frozenset({"b"})


class TestPhaseVectorEqualityAndHashing:
    def test_explicit_zero_equals_absent(self) -> None:
        d = Dim.concrete(3)
        pv1 = PhaseVector(d, {1: Phase.zero()})
        pv2 = PhaseVector(d, {})
        assert pv1 == pv2
        assert hash(pv1) == hash(pv2)

    def test_different_entries_unequal(self) -> None:
        d = Dim.concrete(3)
        pv1 = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 4))})
        pv2 = PhaseVector(d, {1: Phase.turns(sp.Rational(1, 2))})
        assert pv1 != pv2

    def test_different_dims_unequal(self) -> None:
        pv1 = PhaseVector(Dim.concrete(3), {})
        pv2 = PhaseVector(Dim.concrete(4), {})
        assert pv1 != pv2
