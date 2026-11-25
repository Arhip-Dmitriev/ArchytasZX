"""Regression tests for the F61 engine items."""

from __future__ import annotations

from fractions import Fraction

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.pathsum import SQRT2, Cyclo
from archytaszx.algebra.phase import INDEX
from archytaszx.algebra.scalar import Scalar
from archytaszx.rewrite import rules_axioms as R


def test_index_sum_nested_keeps_both_indices() -> None:
    d = Dim.concrete(3)
    w = Scalar.omega(d)
    nested = Scalar.index_sum(
        d,
        lambda k: Scalar.index_sum(
            d, lambda m: Scalar(w.to_sympy() ** (k.to_sympy() * m.to_sympy()))
        ),
    )
    expr = nested.to_sympy()
    assert len(expr.function.free_symbols) == 2
    assert sp.simplify(nested.simplify().to_sympy() - 3) == 0


def test_cyclo_equal_values_hash_equal() -> None:
    pairs = [
        (Cyclo(1, {0: Fraction(1)}), Cyclo(4, {0: Fraction(1)})),
        (Cyclo(3, {1: Fraction(1), 2: Fraction(1)}), Cyclo(1, {0: Fraction(-1)})),
        (SQRT2, SQRT2.lift(24)),
        (Cyclo(6, {1: Fraction(1)}), Cyclo(12, {2: Fraction(1)})),
    ]
    for a, b in pairs:
        assert a == b
        assert len({a, b}) == 1


def test_unit_index_compares_bound_value() -> None:
    three = Dim.concrete(3)
    unit = R.unit_vector(1, three)
    assert R.UnitIndex("j").match(unit, three, {"j": sp.Integer(2)}) is None
    assert R.UnitIndex("j").match(unit, three, {"j": sp.Integer(1)}) == {"j": 1}
    assert R.UnitIndex("j", -1).match(unit, three, {}) == {"j": 2}


def test_mixed_var_and_unit_index_equality() -> None:
    assert R.UnitIndex("j") == R.UnitIndex("j")
    assert R.UnitIndex("j") != R.UnitIndex("j", -1)
    assert len({R.MixedVar("a", 2), R.MixedVar("a", 2)}) == 1
    assert R.MixedVar("a", 2) != R.MixedVar("a", 3)


def test_symbolic_unit_vector_reduces_index_mod_d() -> None:
    d = Dim.symbol("d")
    top = d.to_sympy() - 1
    for index in (sp.Integer(-1), top):
        rule = R._rule_coefficient(R.unit_vector(index, d))
        assert rule.xreplace({INDEX: top}) == 1
    j = sp.Symbol("j", integer=True)
    assert R._rule_coefficient(R.unit_vector(-j, d)).xreplace({INDEX: d.to_sympy() - j}) == 1


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
