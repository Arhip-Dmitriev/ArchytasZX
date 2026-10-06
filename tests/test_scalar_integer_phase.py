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

"""Checks :meth:`Scalar.simplify`'s reduction of exponentials over integer-assumed symbols."""

from __future__ import annotations

import itertools

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import Scalar

I0 = sp.Symbol("_i0", integer=True, nonnegative=True)
I1 = sp.Symbol("_i1", integer=True, nonnegative=True)
D = sp.Symbol("d", integer=True, positive=True)
S = sp.Symbol("s", complex=True)


def twopi_i(q: sp.Expr) -> sp.Expr:
    """``exp(2*pi*i*q)``."""
    return sp.exp(2 * sp.pi * sp.I * q)


@pytest.mark.parametrize(
    "left,right",
    [
        (sp.exp(sp.I * sp.pi * (I0 - I1)), sp.exp(sp.I * sp.pi * (I0 + I1))),
        (twopi_i((I0 - I1) / 3), twopi_i((I0 + 2 * I1) / 3)),
        (twopi_i(sp.Rational(3, 2) * I0 * I1 + sp.Rational(5, 4)), sp.I * twopi_i(I0 * I1 / 2)),
        (twopi_i(I0 + I1), sp.Integer(1)),
    ],
)
def test_equal_phases_over_integers_simplify_to_zero(left: sp.Expr, right: sp.Expr) -> None:
    assert Scalar(left - right).simplify().is_zero


@pytest.mark.parametrize(
    "expr",
    [
        twopi_i(I0 / D),
        twopi_i(S * I0 / 2),
        twopi_i(I0 / 2) - twopi_i(I1 / 2),
        sp.exp(sp.I * I0),
    ],
)
def test_reduction_is_sound_at_integer_points(expr: sp.Expr) -> None:
    simplified = Scalar(expr).simplify().to_sympy()
    for i0, i1, d, s in itertools.product(range(4), range(4), (2, 3, 5), (sp.Rational(1, 3), 2)):
        point = {I0: i0, I1: i1, D: d, S: s}
        before = complex(sp.N(expr.xreplace(point)))
        after = complex(sp.N(simplified.xreplace(point)))
        assert abs(before - after) < 1e-9, point


def test_symbolic_dimension_and_non_integer_symbols_are_left_alone() -> None:
    assert Scalar(twopi_i(I0 / D)).simplify() == Scalar(twopi_i(I0 / D))
    assert Scalar(twopi_i(S * I0 / 2)).simplify() == Scalar(twopi_i(S * I0 / 2))
