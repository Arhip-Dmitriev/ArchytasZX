"""Tests for qufzx.algebra.dimension: the Phase 1 dimension algebra."""

import pytest

from qufzx.algebra.dimension import (
    Dim,
    DimensionDomainError,
    DimensionGrammarError,
    UnifyStatus,
)


class TestConstruction:
    def test_concrete_builds(self) -> None:
        assert Dim.concrete(2).to_int() == 2
        assert Dim(5).to_int() == 5

    def test_symbol_builds(self) -> None:
        d = Dim.symbol("d")
        assert not d.is_concrete
        assert d.free_symbols == frozenset({"d"})

    def test_product_builds(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        assert (d1 * d2).free_symbols == frozenset({"d1", "d2"})

    def test_power_builds(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        p = d**n
        assert p.free_symbols == frozenset({"d", "n"})

    def test_zero_rejected(self) -> None:
        with pytest.raises(DimensionDomainError):
            Dim(0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(DimensionDomainError):
            Dim(-1)

    def test_non_integer_rejected(self) -> None:
        with pytest.raises(TypeError):
            Dim(2.5)  # type: ignore[arg-type]

    def test_non_integer_exponent_rejected(self) -> None:
        with pytest.raises(TypeError):
            Dim.symbol("d") ** 2.5  # type: ignore[arg-type]

    def test_negative_exponent_rejected(self) -> None:
        with pytest.raises(DimensionDomainError):
            Dim.symbol("d") ** (-1)

    def test_bool_rejected(self) -> None:
        with pytest.raises(DimensionGrammarError):
            Dim(True)


class TestEqualityAndNormalization:
    def test_commutative_product(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        assert d1 * d2 == d2 * d1

    def test_product_equals_power(self) -> None:
        d = Dim.symbol("d")
        assert d * d == d**2

    def test_power_one_is_identity(self) -> None:
        d = Dim.symbol("d")
        assert d**1 == d

    def test_power_zero_is_one(self) -> None:
        d = Dim.symbol("d")
        assert d**0 == Dim.concrete(1)

    def test_concrete_folding(self) -> None:
        assert Dim.concrete(2) * Dim.concrete(3) == Dim.concrete(6)

    def test_concrete_symbol_product_commutes(self) -> None:
        d = Dim.symbol("d")
        assert 2 * d == d * 2

    def test_different_symbols_unequal(self) -> None:
        assert Dim.symbol("d") != Dim.symbol("d1")

    def test_different_concretes_unequal(self) -> None:
        assert Dim.concrete(2) != Dim.concrete(3)

    def test_different_exponent_symbols_unequal(self) -> None:
        d = Dim.symbol("d")
        n, m = Dim.symbol("n"), Dim.symbol("m")
        assert d**n != d**m

    def test_eq_against_non_dim_is_false(self) -> None:
        assert (Dim.concrete(2) == 2) is False

    def test_ne_against_non_dim_is_true(self) -> None:
        assert (Dim.concrete(2) != 2) is True


class TestHashing:
    def test_equal_dims_hash_equal(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        assert hash(d1 * d2) == hash(d2 * d1)

    def test_dedup_in_set(self) -> None:
        d = Dim.symbol("d")
        s = {d * d, d**2, Dim.symbol("d") ** 2}
        assert len(s) == 1

    def test_dedup_in_dict_keys(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        m = {d1 * d2: "a", d2 * d1: "b"}
        assert len(m) == 1
        assert m[d1 * d2] == "b"


class TestIntrospection:
    def test_is_concrete_true_for_int(self) -> None:
        assert Dim.concrete(4).is_concrete

    def test_is_concrete_false_for_symbolic(self) -> None:
        assert not Dim.symbol("d").is_concrete

    def test_to_int_on_symbolic_raises(self) -> None:
        with pytest.raises(DimensionDomainError):
            Dim.symbol("d").to_int()

    def test_free_symbols_nested(self) -> None:
        d1, d2, n = Dim.symbol("d1"), Dim.symbol("d2"), Dim.symbol("n")
        expr = d1 * d2**n
        assert expr.free_symbols == frozenset({"d1", "d2", "n"})


class TestSubstitution:
    def test_total_substitution(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        expr = d**n
        result = expr.substitute({"d": 3, "n": 4})
        assert result.to_int() == 3**4 == 81

    def test_partial_substitution_stays_symbolic(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        result = (d**n).substitute({"d": 2})
        assert not result.is_concrete
        assert result.free_symbols == frozenset({"n"})
        assert result.substitute({"n": 3}) == Dim.concrete(8)

    def test_substitution_reaches_exponents(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        result = (d**n).substitute({d: 2, n: 3})
        assert result == Dim.concrete(8)

    def test_dim_key_accepted(self) -> None:
        d = Dim.symbol("d")
        assert d.substitute({d: 5}) == Dim.concrete(5)

    def test_dim_value_accepted(self) -> None:
        d = Dim.symbol("d")
        assert d.substitute({"d": Dim.concrete(5)}) == Dim.concrete(5)

    def test_invalid_dimension_value_rejected(self) -> None:
        d = Dim.symbol("d")
        with pytest.raises(DimensionDomainError):
            d.substitute({"d": 0})

    def test_invalid_exponent_value_rejected(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        with pytest.raises(DimensionDomainError):
            (d**n).substitute({"d": 2, "n": -1})

    def test_original_unchanged(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        expr = d**n
        expr.substitute({"d": 2, "n": 3})
        assert expr.free_symbols == frozenset({"d", "n"})
        assert not expr.is_concrete


class TestUnifyPlaceholder:
    def test_concrete_equal_succeeds(self) -> None:
        result = Dim.concrete(2).unify(Dim.concrete(2))
        assert result.is_success

    def test_concrete_unequal_fails_hard(self) -> None:
        result = Dim.concrete(2).unify(Dim.concrete(3))
        assert result.is_failure
        assert not result.is_deferred

    def test_syntactically_equal_succeeds_no_constraints(self) -> None:
        d1, d2 = Dim.symbol("d1"), Dim.symbol("d2")
        result = (d1 * d2).unify(d1 * d2)
        assert result.is_success
        assert result.constraints == ()

    def test_bare_symbol_binds(self) -> None:
        d = Dim.symbol("d")
        result = d.unify(Dim.concrete(5))
        assert result.is_success
        assert result.bindings == {"d": Dim.concrete(5)}

    def test_bare_symbol_binds_symmetrically(self) -> None:
        d = Dim.symbol("d")
        result = Dim.concrete(5).unify(d)
        assert result.is_success
        assert result.bindings == {"d": Dim.concrete(5)}

    def test_product_vs_symbol_deferred(self) -> None:
        d, d1, d2 = Dim.symbol("d"), Dim.symbol("d1"), Dim.symbol("d2")
        result = d.unify(d1 * d2)
        # d is a bare symbol, so this binds rather than defers.
        assert result.is_success

    def test_two_compound_expressions_deferred(self) -> None:
        d1, d2, n, m = (
            Dim.symbol("d1"),
            Dim.symbol("d2"),
            Dim.symbol("n"),
            Dim.symbol("m"),
        )
        lhs, rhs = d1**n, d2**m
        result = lhs.unify(rhs)
        assert result.status is UnifyStatus.DEFERRED
        assert result.constraints == ((lhs, rhs),)

    def test_deferred_case_recorded_not_guessed(self) -> None:
        d, n, m = Dim.symbol("d"), Dim.symbol("n"), Dim.symbol("m")
        result = (d**n).unify(d**m)
        assert result.is_deferred
        assert not result.is_success
        assert not result.is_failure

    def test_occurs_check_defers_symbol_in_product(self) -> None:
        d, e = Dim.symbol("d"), Dim.symbol("e")
        result = d.unify(d * e)
        assert result.is_deferred
        assert result.bindings == {}
        assert result.constraints == ((d, d * e),)

    def test_occurs_check_defers_symmetrically(self) -> None:
        d, e = Dim.symbol("d"), Dim.symbol("e")
        result = (d * e).unify(d)
        assert result.is_deferred
        assert result.bindings == {}
        assert result.constraints == ((d * e, d),)

    def test_occurs_check_defers_symbol_in_power(self) -> None:
        d, n = Dim.symbol("d"), Dim.symbol("n")
        result = d.unify(d**n)
        assert result.is_deferred

    def test_occurs_check_does_not_regress_plain_self_unify(self) -> None:
        d = Dim.symbol("d")
        result = d.unify(d)
        assert result.is_success
        assert result.bindings == {}
        assert result.constraints == ()

    def test_occurs_check_not_over_broad(self) -> None:
        d, e, f = Dim.symbol("d"), Dim.symbol("e"), Dim.symbol("f")
        result = d.unify(e * f)
        assert result.is_success
        assert result.bindings == {"d": e * f}


class TestCompletionCondition:
    """The Phase 1 stand-in for the numeric oracle: fully substituted Dims
    evaluate to an independently computed Python int."""

    def test_represent_and_compare(self) -> None:
        d = Dim.symbol("d")
        two = Dim.concrete(2)
        n = Dim.symbol("n")
        power = d**n

        assert d != two
        assert d != power
        assert two != power

    def test_repr_round_trips(self) -> None:
        d = Dim.symbol("d")
        n = Dim.symbol("n")
        assert repr(Dim.concrete(2)) == "Dim(2)"
        assert repr(d) == "Dim(d)"
        assert repr(d**n) == "Dim(d**n)"
        assert str(d**n) == "d**n"

    def test_each_substitutes_to_expected_int(self) -> None:
        d = Dim.symbol("d")
        n = Dim.symbol("n")
        power = d**n

        assert Dim.concrete(2).to_int() == 2
        assert d.substitute({"d": 7}).to_int() == 7
        assert power.substitute({"d": 3, "n": 5}).to_int() == 3**5 == 243
