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

"""Checks the integer-polynomial paths of ModDelta, _reduce_mod, _linear_form and _integer_slope
against sympy expansion."""

from __future__ import annotations

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import (
    ModDelta,
    _integer_slope,
    _integer_terms,
    _linear_form,
    _reduce_mod,
)

A, B, C, E = (sp.Symbol(f"_i{k}", integer=True, nonnegative=True) for k in range(4))
X = sp.Symbol("x")


def reference_reduce(expr: sp.Expr, modulus: int) -> sp.Expr:
    """Each coefficient of the expanded ``expr`` taken into ``(-modulus/2, modulus/2]``."""
    symbols = sorted(expr.free_symbols, key=lambda s: str(s.name))
    out = []
    for monomial, c in sp.Poly(expr, *symbols).terms():
        r = int(c) % modulus
        r = r - modulus if 2 * r > modulus else r
        if r:
            out.append(r * sp.Mul(*(s**e for s, e in zip(symbols, monomial))))
    return sp.expand(sp.Add(*out))


CARRY = A * B + A * C + B * C + A * B * C * E


@pytest.mark.parametrize(
    "expr,modulus",
    [
        (CARRY.xreplace({B: A + C + 1}), 2),
        (3 * A * B - 5 * C + 7, 4),
        ((A + B) ** 3 - A * (B + 2) ** 2, 5),
        (A**2 * B - A * A * B + C, 3),
    ],
)
def test_reduce_mod_matches_expansion(expr: sp.Expr, modulus: int) -> None:
    assert _reduce_mod(expr, modulus) == reference_reduce(expr, modulus)


def test_moddelta_takes_reduced_argument() -> None:
    expr = CARRY.xreplace({B: A + C + 1})
    assert ModDelta(expr, 2).args[0] == reference_reduce(expr, 2)
    assert ModDelta(2 * A - 2 * A + 3, 2) == 0
    assert ModDelta(A - A, 2) == 1


def test_integer_terms_refuses_rationals_and_functions() -> None:
    assert _integer_terms(A / 2 + B) is None
    assert _integer_terms(sp.exp(A) + B) is None
    assert _integer_terms(A**3 * B) == {frozenset({(A, 3), (B, 1)}): 1}


def test_linear_form_and_slope() -> None:
    assert _linear_form(2 * A - B + 3) == {A: 2, B: -1}
    assert _linear_form(A * B + C) is None
    assert _linear_form((A + B) ** 2 - A**2 - B**2 - 2 * A * B + A - C) == {A: 1, B: 0, C: -1}
    assert _integer_slope(CARRY, A) is True
    assert _integer_slope(A * X + B, A) is False
    assert _integer_slope(A / 2 + B, A) is False
