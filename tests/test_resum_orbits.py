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

"""Character-orbit resummation in the symbolic comparison (check._resummed)."""

from __future__ import annotations

import itertools

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import ModDelta
from archytaszx.semantics import check

X = [sp.Symbol(f"_i{k}", integer=True, nonnegative=True) for k in range(4)]


def _e(turns: sp.Expr) -> sp.Expr:
    return sp.exp(2 * sp.pi * sp.I * turns)


def _ranges(d: int) -> dict[str, Dim]:
    return {str(x): Dim(d) for x in X}


def _zero_everywhere(expr: sp.Expr, d: int) -> bool:
    symbols = sorted(expr.free_symbols, key=str)
    for values in itertools.product(range(d), repeat=len(symbols)):
        v = expr.xreplace(dict(zip(symbols, map(sp.Integer, values), strict=True)))
        if abs(complex(sp.N(v, 30))) > 1e-12:
            return False
    return True


def _draper_left(d: int) -> sp.Expr:
    """``sum_c [i0 == c] [i0 == i2] (1/d) sum_k e(k (i0 - i1 + i3) / d)`` with the case
    deltas mod d^2, as the draper_adder n = 1 contraction gives it."""
    i0, i1, i2, i3 = X
    return sp.Add(
        *(
            ModDelta(d * i0 - d * c, d * d) * ModDelta(i0 - i2, d) * _e(k * (i0 - i1 + i3) / d) / d
            for c in range(d)
            for k in range(d)
        )
    )


def test_case_split_character_sum_resums_to_its_delta() -> None:
    i0, i1, i2, i3 = X
    d = 6
    diff = sp.expand(_draper_left(d) - ModDelta(i0 - i2, d) * ModDelta(i0 - i1 + i3, d))
    assert check._vanishes(diff, _ranges(d), resum=True)


def test_orbit_closed_only_under_a_delta_resums() -> None:
    i0, i1, i2, i3 = X
    a = i1 - i3 + 1
    t1 = (3 * i0 + 7 * i1 + 6 * i2 + 2 * i3 + 4) / sp.Integer(9)
    t2 = (6 * i0 + 8 * i1 + 3 * i2 + i3 + 2) / sp.Integer(9)
    left = ModDelta(a, 3) * (1 + _e(t1) + _e(t2)) / 3
    right = ModDelta(-3 * i0 - i1 + 3 * i2 + i3 + 2, 9)
    diff = sp.expand(left - right)
    assert _zero_everywhere(diff, 3)
    assert not check._vanishes(diff, _ranges(3))
    assert check._vanishes(diff, _ranges(3), resum=True)


def test_a_wrong_delta_is_not_resummed_away() -> None:
    i0, i1, i2, i3 = X
    d = 6
    for wrong in (i0 - i1 - i3, i0 - i1 + i3 + 1, 2 * i0 - i1 + i3):
        diff = sp.expand(_draper_left(d) - ModDelta(i0 - i2, d) * ModDelta(wrong, d))
        assert not _zero_everywhere(diff, d)
        assert not check._vanishes(diff, _ranges(d), resum=True)


def test_a_partial_orbit_is_kept() -> None:
    i1, i3 = X[1], X[3]
    d = 4
    partial = sp.Add(*(_e(k * (i1 + i3) / d) for k in range(d - 1))) / d
    diff = sp.expand(partial - ModDelta(i1 + i3, d))
    assert not _zero_everywhere(diff, d)
    assert not check._vanishes(diff, _ranges(d), resum=True)


def test_normal_delta_divides_the_common_factor_and_a_unit() -> None:
    i0, i1 = X[0], X[1]
    assert check._normal_delta(10 * i0 - 40, 100) == ModDelta(i0 - 4, 10)
    assert check._normal_delta(2 * i0 + 1, 4) == 0
    assert check._normal_delta(3 * i0 + 6 * i1, 9) == check._normal_delta(i0 + 2 * i1, 3)
    assert check._normal_delta(5 * i0 + 5 * i1, 7) == check._normal_delta(i0 + i1, 7)


def test_polynomials_over_a_thousand_indices_need_no_deep_recursion() -> None:
    from archytaszx.algebra import scalar

    many = [sp.Symbol(f"_i{k}", integer=True, nonnegative=True) for k in range(1100)]
    turns = sp.Rational(9, 8) * sp.Add(*many) + sp.Rational(5, 2) * many[0] * many[1]
    reduced = sp.Rational(1, 8) * sp.Add(*many) + many[0] * many[1] / 2
    assert check._reduce_exponents(_e(turns)) == _e(reduced)
    found = check._turns(turns, tuple(many))
    assert found is not None and found[(0,) * 1099 + (1,)] == sp.Rational(1, 8)
    assert scalar._reduce_integer_phase(2 * sp.pi * sp.I * turns) == 2 * sp.pi * sp.I * reduced
