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

import random

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import (
    ModDelta,
    _expand,
    _FreeSymbols,
    _integer_slope,
    _integer_terms,
    _linear_form,
    _reduce_mod,
    _terms_expr,
    _xreplace_symbol,
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


def reference_canonical(expr: sp.Expr, modulus: int) -> sp.Expr:
    """The residue of ``+-expr`` ModDelta keeps: both built, a sign-free one first, then the
    least by sort key."""
    candidates = [reference_reduce(expr, modulus), reference_reduce(-expr, modulus)]
    preferred = [c for c in candidates if not c.could_extract_minus_sign()] or candidates
    return min(preferred, key=sp.default_sort_key)


def test_moddelta_sign_choice_is_the_built_one() -> None:
    rng = random.Random(31)
    symbols = [A, B, C, E]
    for _ in range(400):
        expr = sp.Integer(rng.randint(-5, 5))
        for _t in range(rng.randint(1, 6)):
            mono = sp.Mul(
                *(s ** rng.choice((1, 1, 2)) for s in rng.sample(symbols, rng.randint(1, 3)))
            )
            expr += rng.randint(-9, 9) * mono
        if not expr.free_symbols:
            continue
        modulus = rng.choice((2, 3, 4, 5, 6, 8, 12))
        want = reference_canonical(expr, modulus)
        got = ModDelta(expr, modulus)
        if want.free_symbols:
            assert same_tree(got, ModDelta(want, modulus, evaluate=False))
        else:
            assert got == (1 if int(want) % modulus == 0 else 0)


def same_tree(a: sp.Expr, b: sp.Expr) -> bool:
    """Same class, value, hash and arguments in the same order, recursively."""
    if type(a) is not type(b) or a != b or hash(a) != hash(b) or str(a) != str(b):
        return False
    return len(a.args) == len(b.args) and all(same_tree(x, y) for x, y in zip(a.args, b.args))


def test_terms_expr_is_the_flattened_sum() -> None:
    rng = random.Random(41)
    symbols = [A, B, C, E, X, sp.Symbol("d", integer=True, positive=True)]
    for _ in range(400):
        terms = {}
        for _t in range(rng.randint(0, 10)):
            mono = frozenset(
                (s, rng.choice((1, 1, 2, 3))) for s in rng.sample(symbols, rng.randint(0, 4))
            )
            terms[mono] = rng.choice((1, -1, 2, -2, 3, -7))
        want = sp.Add(
            *(
                c * sp.Mul(*(s**e for s, e in sorted(mono, key=lambda item: str(item[0].name))))
                for mono, c in terms.items()
            )
        )
        assert same_tree(_terms_expr(terms), want)


def random_polynomial(rng: random.Random, symbols: list[sp.Symbol]) -> sp.Expr:
    """An expanded integer polynomial of up to six terms in ``symbols``."""
    expr = sp.Integer(rng.randint(-3, 3))
    for _t in range(rng.randint(1, 6)):
        mono = sp.Mul(*(s ** rng.choice((1, 1, 2)) for s in rng.sample(symbols, rng.randint(1, 3))))
        expr += rng.randint(-5, 5) * mono
    return sp.expand(expr)


def test_delta_substitution_is_xreplace() -> None:
    rng = random.Random(53)
    symbols = [A, B, C, E]
    frees = _FreeSymbols()
    for _ in range(300):
        modulus = rng.choice((2, 3, 4, 6))
        factors = [ModDelta(random_polynomial(rng, symbols), modulus) for _f in range(3)]
        product = sp.Mul(*factors, sp.exp(sp.I * sp.pi * A / 3))
        var = rng.choice(symbols)
        value = random_polynomial(rng, [s for s in symbols if s != var])
        got = _xreplace_symbol(product, var, value, frees)
        assert same_tree(got, product.xreplace({var: value}))


def test_fixed_expand_is_sympy_expand() -> None:
    rng = random.Random(59)
    symbols = [A, B, C, E]
    for _ in range(200):
        deltas = [ModDelta(random_polynomial(rng, symbols), 2) for _f in range(3)]
        for expr in (
            sp.Mul(*deltas),
            deltas[0] ** 2 * deltas[1] - 3 * deltas[2] * A,
            deltas[0] * (deltas[1] + 1),
            sp.exp(sp.I * sp.pi * (A + B) / 2) * deltas[0],
        ):
            assert same_tree(_expand(expr), sp.expand(expr))
