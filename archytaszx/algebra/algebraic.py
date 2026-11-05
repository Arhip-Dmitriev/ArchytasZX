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

"""Algebraic numbers in canonical form, and an enumeration of all of them.

The canonical form of an algebraic number is a rational, or ``CRootOf(p, i)`` with ``p`` its
minimal polynomial (primitive, integer coefficients, positive leading coefficient) and ``i``
the index of the root in sympy's fixed root order. Two algebraic numbers are equal exactly when
their canonical forms are identical.

:func:`level` gives the finitely many numbers of each level; every algebraic number has a level
(a rational its height, any other number the degree plus the largest coefficient of its minimal
polynomial), so :func:`algebraic_numbers` lists every algebraic number exactly once.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterator

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.algebra.scalar import Scalar

_X = sp.Symbol("x")


class NotAlgebraicError(ValueError):
    """A value with free symbols or no algebraic minimal polynomial."""


def _primitive(poly: sp.Poly) -> sp.Poly:
    _, prim = poly.primitive()
    return prim if prim.LC() > 0 else -prim


@functools.lru_cache(maxsize=4096)
def _approx_roots(poly: sp.Poly, digits: int) -> tuple[sp.Expr, ...]:
    return tuple(sp.CRootOf(poly, i).evalf(digits) for i in range(poly.degree()))


def _root_index(poly: sp.Poly, value: sp.Expr) -> int:
    """The index of ``value`` among the roots of the irreducible ``poly``: the unique root
    within a third of the least root separation, at rising precision."""
    for digits in (15, 30, 60, 120, 240, 480):
        v = sp.N(value, digits)
        approx = _approx_roots(poly, digits)
        sep = min(abs(a - b) for a, b in itertools.combinations(approx, 2))
        near = [i for i, a in enumerate(approx) if abs(a - v) < sep / 3]
        if len(near) == 1:
            return near[0]
    raise NotAlgebraicError(f"no root of {poly.as_expr()} isolates {value}")


def _gaussian(a: sp.Rational, b: sp.Rational) -> sp.Expr:
    """The canonical form of ``a + b i`` for rationals ``a`` and ``b != 0``: the root of its
    minimal polynomial with the sign of ``b`` (negative imaginary part first)."""
    poly = _primitive(sp.Poly(_X**2 - 2 * a * _X + a**2 + b**2, _X))
    return sp.CRootOf(poly, 0 if b < 0 else 1)


def canonical(expr: sp.Expr) -> sp.Expr:
    """The canonical form of the algebraic number ``expr``."""
    expr = sp.sympify(expr)
    if expr.free_symbols:
        raise NotAlgebraicError(f"{expr} has free symbols")
    if expr.is_Rational:
        return expr
    re, im = expr.as_real_imag()
    if re.is_Rational and im.is_Rational:
        return _gaussian(re, im)
    if isinstance(expr, sp.CRootOf):
        poly = _primitive(sp.Poly(expr.poly.as_expr(), _X))
        if poly.degree() > 1 and poly.is_irreducible:
            return sp.CRootOf(poly, _root_index(poly, expr))
    try:
        mp = sp.minimal_polynomial(expr, _X, polys=True)
    except (NotImplementedError, ValueError, sp.polys.polyerrors.PolynomialError) as exc:
        raise NotAlgebraicError(f"{expr} has no minimal polynomial here: {exc}") from exc
    poly = _primitive(sp.Poly(mp.as_expr(), _X))
    if poly.degree() == 1:
        a, b = poly.all_coeffs()
        return sp.Rational(-b, a)
    return sp.CRootOf(poly, _root_index(poly, expr))


def canonical_scalar(value: Scalar) -> Scalar:
    """``value`` with its expression in canonical form."""
    return Scalar(canonical(value.to_sympy()))


def canonical_vector(pv: PhaseVector | None, dim: Dim) -> PhaseVector | None:
    """``pv`` as a weight vector whose every coefficient is in canonical form; ``None`` stays
    ``None``. Needs a concrete dimension."""
    if pv is None:
        return None
    if not dim.is_concrete:
        raise NotAlgebraicError("a symbolic dimension")
    weights = {k: canonical_scalar(pv.coefficient(k)) for k in range(1, dim.to_int())}
    return PhaseVector.from_weights(dim, weights)


def _irreducible(coeffs: tuple[int, ...]) -> sp.Poly | None:
    if coeffs[0] <= 0:
        return None
    poly = sp.Poly(list(coeffs), _X)
    if poly.degree() < 2 or sp.gcd_list(list(coeffs)) != 1:
        return None
    return poly if poly.is_irreducible else None


def _height(q: sp.Rational) -> int:
    return max(abs(int(q.p)), int(q.q))


def _rationals(n: int) -> list[sp.Expr]:
    return [
        sp.Rational(p, q)
        for q in range(1, n + 1)
        for p in range(-n, n + 1)
        if sp.igcd(p, q) == 1 and max(abs(p), q) == n
    ]


def level(n: int) -> list[sp.Expr]:
    """The algebraic numbers of level ``n``, canonical, in a fixed order: rationals of height
    ``n``; primitive roots of unity of order ``n``; ``a + b i`` (``b != 0``) with ``n - 1`` the
    larger height of ``a`` and ``b``; the roots of irreducible integer polynomials of degree
    ``D >= 2`` and largest coefficient ``H`` with ``D + H = n``. A number may occur at several
    levels."""
    out: list[sp.Expr] = list(_rationals(n))
    if n >= 3:
        out += [
            canonical(sp.exp(2 * sp.pi * sp.I * sp.Rational(k, n)))
            for k in range(1, n)
            if sp.igcd(k, n) == 1
        ]
    if n >= 2:
        small = [q for m in range(1, n) for q in _rationals(m)]
        for a in small:
            for b in small:
                if b != 0 and max(_height(a), _height(b)) == n - 1:
                    out.append(_gaussian(a, b))
    for degree in range(2, n):
        height = n - degree
        for coeffs in itertools.product(range(-height, height + 1), repeat=degree + 1):
            if max(map(abs, coeffs)) != height:
                continue
            poly = _irreducible(coeffs)
            if poly is not None:
                out += [sp.CRootOf(poly, i) for i in range(degree)]
    return out


def algebraic_numbers() -> Iterator[sp.Expr]:
    """Every algebraic number exactly once, in canonical form: the levels 1, 2, ... in order,
    each number at its first level."""
    seen: set[sp.Expr] = set()
    for n in itertools.count(1):
        for value in level(n):
            if value not in seen:
                seen.add(value)
                yield value


def up_to_level(bound: int) -> tuple[Scalar, ...]:
    """Every algebraic number of level at most ``bound``, each once, as scalars."""
    seen: list[sp.Expr] = []
    for n in range(1, bound + 1):
        for value in level(n):
            if value not in seen:
                seen.append(value)
    return tuple(Scalar(v) for v in seen)


__all__ = [
    "NotAlgebraicError",
    "algebraic_numbers",
    "canonical",
    "canonical_scalar",
    "canonical_vector",
    "level",
    "up_to_level",
]
