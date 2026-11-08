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

"""Number-field certificates: finitely many algebraic numbers written over one real root.

A :class:`FieldCertificate` holds an irreducible integer polynomial ``p``, rationals
``lo < hi`` between which ``p`` has exactly one root ``θ`` (``p(lo) < 0 < p(hi)`` and ``p'``
positive there, its Bernstein coefficients on ``[lo, hi]`` all positive), and for each atom
``a`` (a CRootOf, a radical, a root of unity, any closed algebraic subterm) rational
polynomials ``qa``, ``qb`` of degree below ``deg p`` with ``a = qa(θ) + i qb(θ)``.

:func:`certificate` finds one by resultants, factoring and polynomial gcds over ``Q(θ)``.
:func:`check` verifies one from its data alone: ``p`` irreducible, the interval isolating
one root, and every atom's ``qa(θ) + i qb(θ)`` a root of the atom's minimal polynomial lying
in the atom's isolating rectangle, by exact polynomial arithmetic modulo ``p`` and rational
interval arithmetic. :func:`element` writes any expression over the atoms as a reduced pair.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.algebraic import NotAlgebraicError

T = sp.Symbol("theta_nf")
_Y = sp.Symbol("y_nf")
_Z = sp.Symbol("z_nf")
_DIGITS = 60


class NumberFieldError(ValueError):
    """A certificate that fails its check, or values with no certificate."""


Coeffs = tuple[sp.Rational, ...]
"""Rational coefficients, constant term first."""

Rect = tuple[sp.Rational, sp.Rational, sp.Rational, sp.Rational]
"""``(ax, ay, bx, by)``: the closed rectangle ``[ax, bx] × [ay, by]``."""


@dataclass(frozen=True)
class FieldCertificate:
    """``p`` (integer coefficients, constant term first), the isolating interval, and each
    atom with the coefficients of its real and imaginary parts as polynomials in ``θ`` and a
    rectangle ``(ax, ay, bx, by)`` holding the atom and one root of its minimal polynomial."""

    poly: tuple[int, ...]
    lo: sp.Rational
    hi: sp.Rational
    atoms: tuple[tuple[sp.Expr, Coeffs, Coeffs, Rect], ...]

    @property
    def degree(self) -> int:
        """The degree of ``p``."""
        return len(self.poly) - 1

    def p(self) -> sp.Poly:
        """``p`` as a sympy polynomial in :data:`T`."""
        return sp.Poly(list(reversed(self.poly)), T, domain="QQ")

    def table(self) -> dict[sp.Expr, tuple[sp.Poly, sp.Poly]]:
        """Each atom's real and imaginary parts as polynomials in :data:`T`."""
        return {a: (_poly(re), _poly(im)) for a, re, im, _ in self.atoms}


def _poly(cs: Coeffs) -> sp.Poly:
    return sp.Poly(list(reversed(cs)) or [0], T, domain="QQ")


def _coeffs(p: sp.Poly) -> Coeffs:
    return tuple(sp.Rational(c) for c in reversed(p.all_coeffs())) if not p.is_zero else ()


# atoms


def atoms(expr: sp.Expr) -> list[sp.Expr]:
    """The algebraic atoms of the closed expression ``expr``: every maximal subterm that is not
    a rational, the imaginary unit, a sum, a product or an integer power."""
    expr = sp.sympify(expr)
    if expr.free_symbols:
        raise NotAlgebraicError(f"{expr} has free symbols")
    if expr.is_Rational or expr is sp.I:
        return []
    if expr.is_Add or expr.is_Mul:
        out: list[sp.Expr] = []
        for a in expr.args:
            out += [x for x in atoms(a) if x not in out]
        return out
    if expr.is_Pow and expr.args[1].is_Integer:
        return atoms(expr.args[0])
    if expr.is_Pow and expr.args[0] is sp.I:
        return atoms(sp.exp(sp.I * sp.pi * expr.args[1] / 2))
    return [expr]


@dataclass(frozen=True)
class _Info:
    """An irrational atom's minimal polynomial in :data:`_Y`, its value to :data:`_DIGITS`
    digits, an isolating rectangle and whether it is real."""

    m: sp.Poly
    value: sp.Expr
    own: Rect
    real: bool


def _mp(z: object) -> sp.Expr:
    import mpmath  # type: ignore[import-untyped]  # mpmath ships no py.typed marker

    c = mpmath.mpc(z)
    return sp.Float(str(c.real), _DIGITS) + sp.I * sp.Float(str(c.imag), _DIGITS)


def _roots(m: sp.Poly) -> list[sp.Expr]:
    import mpmath

    with mpmath.workdps(_DIGITS + 20):
        rs = mpmath.polyroots(
            [int(c) for c in m.clear_denoms()[1].all_coeffs()], maxsteps=400, extraprec=400
        )
        return [_mp(r) for r in rs]


@functools.lru_cache(maxsize=1024)
def _info(atom: sp.Expr) -> _Info | sp.Rational:
    """The atom as a rational, or its :class:`_Info`."""
    if atom.is_Rational:
        return sp.Rational(atom)
    if isinstance(atom, sp.CRootOf):
        m = sp.Poly(atom.poly.as_expr().subs(atom.poly.gens[0], _Y), _Y, domain="QQ")
        if m.degree() > 1 and m.is_irreducible:
            iv = atom._get_interval()
            if atom.is_real:
                own: Rect = (sp.Rational(iv.a), sp.Integer(0), sp.Rational(iv.b), sp.Integer(0))
            else:
                own = (
                    sp.Rational(iv.ax),
                    sp.Rational(iv.ay),
                    sp.Rational(iv.bx),
                    sp.Rational(iv.by),
                )
            rs = _roots(m)
            mid = (own[0] + own[2]) / 2 + sp.I * (own[1] + own[3]) / 2
            dist = sorted((sp.N(abs(r - mid), 30), i) for i, r in enumerate(rs))
            reach = sp.N(own[2] - own[0] + own[3] - own[1], 30)
            if dist[0][0] > reach + 1e-40 or dist[1][0] <= reach + 1e-40:
                raise NumberFieldError(f"no numeric root of {m.as_expr()} picked by {atom}")
            near = rs[dist[0][1]]
            return _Info(m, sp.re(near) if atom.is_real else near, own, bool(atom.is_real))
    try:
        mp = sp.minimal_polynomial(atom, _Y, polys=True)
    except (NotImplementedError, ValueError, sp.polys.polyerrors.BasePolynomialError) as exc:
        raise NotAlgebraicError(f"{atom} has no minimal polynomial here: {exc}") from exc
    m = sp.Poly(mp.as_expr(), _Y, domain="QQ")
    if m.degree() == 1:
        a, b = m.all_coeffs()
        return sp.Rational(-b, a)
    value = sp.N(atom, _DIGITS)
    rs = _roots(m)
    near = min(rs, key=lambda r: abs(complex(r) - complex(value)))
    sep = min(abs(complex(a) - complex(b)) for a, b in itertools.combinations(rs, 2))
    if abs(complex(near) - complex(value)) > sep / 100:
        raise NumberFieldError(f"no numeric root of {m.as_expr()} near {atom}")
    half = sp.Rational(1, 2 ** max(1, int(-math.log2(sep / 4)) + 1))
    cx = sp.Rational(sp.floor(sp.re(near) / half)) * half
    real = abs(complex(near).imag) < 1e-40
    if real:
        own = (cx - half, sp.Integer(0), cx + 2 * half, sp.Integer(0))
        return _Info(m, sp.re(near), own, True)
    cy = sp.Rational(sp.floor(sp.im(near) / half)) * half
    own = (cx - half, cy - half, cx + 2 * half, cy + 2 * half)
    return _Info(m, near, own, False)


# arithmetic in Q(θ)(i)


class _Field:
    """Exact arithmetic on pairs ``(re, im)`` of polynomials in :data:`T` modulo ``p``."""

    def __init__(self, p: sp.Poly) -> None:
        self.p = p

    def red(self, a: sp.Poly) -> sp.Poly:
        return a.rem(self.p)

    def const(self, c: sp.Expr) -> tuple[sp.Poly, sp.Poly]:
        re, im = sp.sympify(c).as_real_imag()
        return sp.Poly(re, T, domain="QQ"), sp.Poly(im, T, domain="QQ")

    def add(
        self, x: tuple[sp.Poly, sp.Poly], y: tuple[sp.Poly, sp.Poly]
    ) -> tuple[sp.Poly, sp.Poly]:
        return x[0] + y[0], x[1] + y[1]

    def mul(
        self, x: tuple[sp.Poly, sp.Poly], y: tuple[sp.Poly, sp.Poly]
    ) -> tuple[sp.Poly, sp.Poly]:
        a, b = x
        c, e = y
        return self.red(a * c - b * e), self.red(a * e + b * c)

    def inv(self, x: tuple[sp.Poly, sp.Poly]) -> tuple[sp.Poly, sp.Poly]:
        a, b = x
        norm = self.red(a * a + b * b)
        if norm.is_zero:
            raise NumberFieldError("division by zero in the number field")
        n_inv = sp.Poly(sp.invert(norm.as_expr(), self.p.as_expr(), T), T, domain="QQ")
        return self.red(a * n_inv), self.red(-b * n_inv)

    def pow(self, x: tuple[sp.Poly, sp.Poly], n: int) -> tuple[sp.Poly, sp.Poly]:
        if n < 0:
            return self.pow(self.inv(x), -n)
        out = self.const(1)
        base = x
        while n:
            if n & 1:
                out = self.mul(out, base)
            base = self.mul(base, base)
            n >>= 1
        return out


def element(cert: FieldCertificate, expr: sp.Expr) -> tuple[sp.Poly, sp.Poly]:
    """``expr`` over the certificate's atoms as reduced ``(re, im)`` polynomials in :data:`T`."""
    field = _Field(cert.p())
    table = cert.table()

    def go(e: sp.Expr) -> tuple[sp.Poly, sp.Poly]:
        if e.is_Rational:
            return field.const(e)
        if e is sp.I:
            return field.const(sp.I)
        if e.is_Add:
            out = field.const(0)
            for a in e.args:
                out = field.add(out, go(a))
            return out
        if e.is_Mul:
            out = field.const(1)
            for a in e.args:
                out = field.mul(out, go(a))
            return out
        if e.is_Pow and e.args[1].is_Integer:
            return field.pow(go(e.args[0]), int(e.args[1]))
        if e.is_Pow and e.args[0] is sp.I:
            return go(sp.exp(sp.I * sp.pi * e.args[1] / 2))
        if e not in table:
            raise NumberFieldError(f"{e} is not an atom of the certificate")
        re, im = table[e]
        return field.red(re), field.red(im)

    return go(sp.sympify(expr))


def key(cert: FieldCertificate, expr: sp.Expr) -> sp.Expr:
    """``re(θ) + i im(θ)`` for ``expr``: equal values have equal keys."""
    re, im = element(cert, expr)
    return sp.expand(re.as_expr() + sp.I * im.as_expr())


def _split(expr: sp.Expr) -> tuple[sp.Poly, sp.Poly]:
    """A polynomial in :data:`T` and ``i`` as its real and imaginary parts."""
    j = sp.Dummy("j")
    poly = sp.Poly(sp.expand(sp.sympify(expr).xreplace({sp.I: j})), T, j, domain="QQ")
    re = sp.Integer(0)
    im = sp.Integer(0)
    for (a, b), c in poly.terms():
        sign = -1 if b % 4 in (2, 3) else 1
        if b % 2 == 0:
            re += sign * c * T**a
        else:
            im += sign * c * T**a
    return sp.Poly(re, T, domain="QQ"), sp.Poly(im, T, domain="QQ")


def inverse(cert: FieldCertificate, expr: sp.Expr) -> sp.Expr:
    """The inverse of ``expr``, a polynomial in :data:`T` and ``i``, as ``re + i im``."""
    re, im = _Field(cert.p()).inv(_split(expr))
    return sp.expand(re.as_expr() + sp.I * im.as_expr())


# intervals


def _ival(cs: Coeffs, lo: Fraction, hi: Fraction) -> tuple[Fraction, Fraction]:
    """A rational interval containing ``Σ cs_k x^k`` for every ``x`` in ``[lo, hi]``."""
    a = b = Fraction(0)
    for c in reversed(cs):
        prods = (a * lo, a * hi, b * lo, b * hi)
        a, b = min(prods), max(prods)
        f = _frac(c)
        a, b = a + f, b + f
    return a, b


def _frac(r: sp.Rational) -> Fraction:
    r = sp.Rational(r)
    return Fraction(int(r.p), int(r.q))


def _bernstein(cs: Coeffs, lo: sp.Rational, hi: sp.Rational) -> list[sp.Rational]:
    """The Bernstein coefficients of ``Σ cs_k x^k`` on ``[lo, hi]``."""
    n = max(len(cs) - 1, 0)
    t = sp.Symbol("t_nf")
    shifted = sp.Poly(sum(c * (lo + (hi - lo) * t) ** k for k, c in enumerate(cs)), t)
    a = [shifted.coeff_monomial(t**k) for k in range(n + 1)]
    return [
        sum(sp.binomial(j, k) / sp.binomial(n, k) * a[k] for k in range(j + 1))
        for j in range(n + 1)
    ]


def derivative(poly: tuple[int, ...]) -> tuple[int, ...]:
    """The coefficients of ``p'``, constant term first."""
    return tuple(k * c for k, c in enumerate(poly))[1:]


def bernstein_derivative(cert: FieldCertificate) -> list[sp.Rational]:
    """The Bernstein coefficients of ``p'`` on ``[lo, hi]``."""
    return _bernstein(tuple(sp.Integer(c) for c in derivative(cert.poly)), cert.lo, cert.hi)


def algebraic(atom: sp.Expr) -> bool:
    """Whether the atom has a minimal polynomial and an isolated numeric value."""
    try:
        _info(atom)
    except (NotAlgebraicError, NumberFieldError):
        return False
    return True


def _rect(atom: sp.Expr) -> tuple[sp.Poly, Rect | None]:
    """The atom's minimal polynomial in :data:`_Y` and an isolating rectangle of it, ``None``
    for a rational."""
    info = _info(atom)
    if not isinstance(info, _Info):
        return sp.Poly(_Y - info, _Y, domain="QQ"), None
    return info.m, info.own


def _within(inner: Rect, outer: Rect) -> bool:
    return bool(
        outer[0] <= inner[0]
        and inner[2] <= outer[2]
        and outer[1] <= inner[1]
        and inner[3] <= outer[3]
    )


def _count(m: sp.Poly, rect: Rect) -> int:
    ax, ay, bx, by = rect
    if ay == by == 0:
        return int(m.count_roots(ax, bx))
    return int(m.count_roots(ax + sp.I * ay, bx + sp.I * by))


def _box(re_cs: Coeffs, im_cs: Coeffs, lo: sp.Rational, hi: sp.Rational) -> Rect:
    """A rectangle containing ``re(x) + i im(x)`` for every ``x`` in ``[lo, hi]``."""
    a, b = _ival(re_cs, _frac(lo), _frac(hi))
    c, e = _ival(im_cs, _frac(lo), _frac(hi))
    return (
        sp.Rational(a.numerator, a.denominator),
        sp.Rational(c.numerator, c.denominator),
        sp.Rational(b.numerator, b.denominator),
        sp.Rational(e.numerator, e.denominator),
    )


# checking


def check(cert: FieldCertificate) -> None:
    """Raise NumberFieldError unless every claim of ``cert`` holds exactly."""
    p = cert.p()
    if cert.degree < 1 or not p.is_irreducible:
        raise NumberFieldError("p is not an irreducible polynomial of positive degree")
    if not cert.lo < cert.hi:
        raise NumberFieldError("the interval is empty")
    if not (p.eval(cert.lo) < 0 < p.eval(cert.hi)):
        raise NumberFieldError("p does not change sign from negative to positive on the interval")
    if p.count_roots(cert.lo, cert.hi) != 1:
        raise NumberFieldError("the interval does not isolate one root of p")
    if not all(b > 0 for b in bernstein_derivative(cert)):
        raise NumberFieldError("p' has a non-positive Bernstein coefficient on the interval")
    field = _Field(p)
    for atom, re_cs, im_cs, rect in cert.atoms:
        if len(re_cs) > cert.degree or len(im_cs) > cert.degree:
            raise NumberFieldError(f"the coefficients of {atom} are not reduced")
        m, own = _rect(atom)
        w = (field.red(_poly(re_cs)), field.red(_poly(im_cs)))
        value = field.const(0)
        for c in m.all_coeffs():
            value = field.add(field.mul(value, w), field.const(c))
        if not (value[0].is_zero and value[1].is_zero):
            raise NumberFieldError(f"the value written for {atom} is not a root of {m.as_expr()}")
        if own is None:
            if w[0].degree() > 0 or not w[1].is_zero:
                raise NumberFieldError(f"the rational {atom} is written as an irrational")
            continue
        if own[1] == own[3] == 0 and not w[1].is_zero:
            raise NumberFieldError(f"the real {atom} is written with an imaginary part")
        if not _within(own, rect):
            raise NumberFieldError(f"the rectangle of {atom} does not hold the atom")
        if not _within(_box(re_cs, im_cs, cert.lo, cert.hi), rect):
            raise NumberFieldError(f"the value written for {atom} is not in its rectangle")
        if _count(m, rect) != 1:
            raise NumberFieldError(f"the rectangle of {atom} does not isolate one root")


# finding


def _num(e: sp.Expr) -> sp.Expr:
    return sp.N(e, _DIGITS)


def _choose(poly: sp.Poly, value: sp.Expr) -> sp.Poly:
    """The irreducible factor of ``poly`` with ``value`` as a root."""
    best: tuple[float, sp.Poly] | None = None
    for f, _ in poly.factor_list()[1]:
        if f.degree() < 1:
            continue
        roots = f.nroots(n=_DIGITS, maxsteps=500)
        dist = min(abs(complex(r) - complex(value)) for r in roots)
        if best is None or dist < best[0]:
            best = (dist, f)
    if best is None or best[0] > 1e-20:
        raise NumberFieldError(f"no factor has the root {value}")
    return best[1]


Pair = tuple[sp.Poly, sp.Poly]
YPoly = list[Pair]
"""A polynomial in ``y`` over ``Q(θ)(i)``, highest coefficient first."""


def _yzero(c: Pair) -> bool:
    return bool(c[0].is_zero and c[1].is_zero)


def _ynorm(field: _Field, f: YPoly) -> YPoly:
    f = [(field.red(a), field.red(b)) for a, b in f]
    while f and _yzero(f[0]):
        f = f[1:]
    return f


def _yadd(field: _Field, f: YPoly, g: YPoly) -> YPoly:
    n = max(len(f), len(g))
    f = [field.const(0)] * (n - len(f)) + f
    g = [field.const(0)] * (n - len(g)) + g
    return _ynorm(field, [field.add(x, y) for x, y in zip(f, g, strict=True)])


def _ymul(field: _Field, f: YPoly, g: YPoly) -> YPoly:
    if not f or not g:
        return []
    out = [field.const(0)] * (len(f) + len(g) - 1)
    for i, x in enumerate(f):
        for j, y in enumerate(g):
            out[i + j] = field.add(out[i + j], field.mul(x, y))
    return _ynorm(field, out)


def _ycompose(field: _Field, m: sp.Poly, lin: YPoly) -> YPoly:
    """``m(lin(y))`` for a rational ``m``."""
    out: YPoly = []
    for c in m.all_coeffs():
        out = _yadd(field, _ymul(field, out, lin), [field.const(c)])
    return out


def _ygcd(field: _Field, f: YPoly, g: YPoly) -> YPoly:
    """The monic gcd of ``f`` and ``g``."""

    def rem(f: YPoly, g: YPoly) -> YPoly:
        f = _ynorm(field, f)
        inv = field.inv(g[0])
        while len(f) >= len(g):
            q = field.mul(f[0], inv)
            f = _ynorm(
                field,
                [
                    field.add(fc, field.mul((-q[0], -q[1]), g[i])) if i < len(g) else fc
                    for i, fc in enumerate(f)
                ],
            )
        return f

    f, g = _ynorm(field, f), _ynorm(field, g)
    while g:
        f, g = g, rem(f, g)
    inv = field.inv(f[0])
    return [field.mul(c, inv) for c in f]


def _near(pair: Pair, theta: sp.Expr, value: sp.Expr) -> bool:
    v = _num(pair[0].as_expr().subs(T, theta) + sp.I * pair[1].as_expr().subs(T, theta))
    return abs(complex(v) - complex(value)) < 1e-20


def _linear_root(field: _Field, g: YPoly, theta: sp.Expr, value: sp.Expr) -> Pair | None:
    if len(g) != 2:
        return None
    root = (field.red(-g[1][0]), field.red(-g[1][1]))
    return root if _near(root, theta, value) else None


def _common_root(
    p: sp.Poly, theta: sp.Expr, m: sp.Poly, old: sp.Poly, c: int, value: sp.Expr
) -> sp.Poly | None:
    """``value`` as a polynomial in ``θ`` when ``θ = o + c·value``, ``o`` a root of ``old`` and
    ``value`` a root of ``m``: the common root of ``m(y)`` and ``old(θ - c y)``, from their gcd
    over ``Q(θ)``; None when that gcd is not linear."""
    field = _Field(p)
    lin = [field.const(-c), (sp.Poly(T, T, domain="QQ"), sp.Poly(0, T, domain="QQ"))]
    g = _ygcd(
        field, _ycompose(field, m, [field.const(1), field.const(0)]), _ycompose(field, old, lin)
    )
    root = _linear_root(field, g, theta, value)
    return None if root is None else root[0]


def _complex_root(
    p: sp.Poly, theta: sp.Expr, gen: sp.Poly, c: int, m: sp.Poly, value: sp.Expr
) -> Pair | None:
    """The root ``α`` of ``m`` with ``re α + c im α`` the element ``gen`` of ``Q(θ)``: the common
    root of ``m(y)`` and ``m((2 gen - (1 - ci) y) / (1 + ci))`` over ``Q(θ)(i)``."""
    field = _Field(p)
    u = field.inv(field.const(1 + c * sp.I))
    a = field.mul((field.red(2 * gen), sp.Poly(0, T, domain="QQ")), u)
    b = field.mul(field.const(-(1 - c * sp.I)), u)
    g = _ygcd(
        field, _ycompose(field, m, [field.const(1), field.const(0)]), _ycompose(field, m, [b, a])
    )
    return _linear_root(field, g, theta, value)


@dataclass
class _Gen:
    """A real generator: its minimal polynomial in :data:`_Z` and its value."""

    poly: sp.Poly
    value: sp.Expr


def _combo_poly(m: sp.Poly, c: int) -> sp.Poly:
    """A rational polynomial in :data:`_Z` whose roots include ``re α + c im α`` for every root
    ``α`` of ``m`` (in :data:`_Y`): the resultant over pairs of roots ``(α, ᾱ)``."""
    n = m.degree()
    w = 1 + c * sp.I
    other = sp.expand(
        sum(
            co * (2 * _Z - (1 - c * sp.I) * _Y) ** k * w ** (n - k)
            for k, co in enumerate(reversed(m.all_coeffs()))
        )
    )
    res = sp.Poly(sp.expand(sp.resultant(m.as_expr(), other, _Y)), _Z)
    coeffs = [sp.expand(co) for co in res.all_coeffs()]
    lead = next(co for co in coeffs if co != 0)
    coeffs = [sp.expand(co / lead) for co in coeffs]
    if any(sp.im(co) != 0 for co in coeffs):
        raise NumberFieldError("no rational polynomial for a real combination")
    return sp.Poly([sp.re(co) for co in coeffs], _Z, domain="QQ")


Part = tuple[str, object]
"""How an atom is written: ``("rational", r)``, ``("real", gen)`` or ``("complex", (gen, c))``."""


def _parts(atom: sp.Expr, gens: list[_Gen]) -> Part:
    """The atom's form, its generator appended to ``gens``."""
    info = _info(atom)
    if not isinstance(info, _Info):
        return ("rational", info)
    if info.real:
        gens.append(_Gen(sp.Poly(info.m.as_expr().subs(_Y, _Z), _Z, domain="QQ"), info.value))
        return ("real", len(gens) - 1)
    for c in _SHIFTS:
        gv = _num(sp.re(info.value) + c * sp.im(info.value))
        f = _choose(_combo_poly(info.m, c), gv)
        pt = sp.Poly(f.as_expr().subs(_Z, T), T, domain="QQ")
        if _complex_root(pt, gv, sp.Poly(T, T, domain="QQ"), c, info.m, info.value) is None:
            continue
        gens.append(_Gen(f, gv))
        return ("complex", (len(gens) - 1, c))
    raise NumberFieldError(f"no real generator for {atom}")


def _isolate(p: sp.Poly, value: sp.Expr) -> tuple[sp.Poly, sp.Rational, sp.Rational]:
    """``p`` signed so that ``p'`` is positive near ``value``, with a dyadic interval around
    ``value`` on which ``p'`` has positive Bernstein coefficients and ``p`` changes sign."""
    v = sp.Rational(sp.Float(sp.re(value), _DIGITS))
    if p.diff().eval(v) < 0:
        p = -p
    cs = tuple(sp.Integer(c) for c in reversed(p.all_coeffs()))
    der = tuple(sp.Integer(k * c) for k, c in enumerate(cs))[1:]
    for k in range(2, 180):
        w = sp.Rational(1, 2**k)
        lo = (sp.floor(v / w) - 1) * w
        hi = lo + 3 * w
        if (
            p.eval(lo) < 0 < p.eval(hi)
            and all(b > 0 for b in _bernstein(der, lo, hi))
            and p.count_roots(lo, hi) == 1
        ):
            return p, lo, hi
    raise NumberFieldError("no isolating interval found")


_SHIFTS = (1, 2, -1, 3, -2, 5, 7, -3, 11, 13)


def certificate(values: list[sp.Expr]) -> FieldCertificate:
    """A checked certificate for every atom of ``values``; NotAlgebraicError when one is not
    algebraic, NumberFieldError when none is found."""
    found: list[sp.Expr] = []
    for v in values:
        found += [a for a in atoms(v) if a not in found]
    gens: list[_Gen] = []
    parts = {a: _parts(a, gens) for a in found}
    if gens:
        p = sp.Poly(gens[0].poly.as_expr().subs(_Z, T), T, domain="QQ")
        theta = gens[0].value
    else:
        p = sp.Poly(T**2 - 2, T, domain="QQ")
        theta = _num(sp.sqrt(2))
    exprs: list[sp.Poly] = [sp.Poly(T, T, domain="QQ")] if gens else []
    for g in gens[1:]:
        for c in _SHIFTS:
            cand = _num(theta + c * g.value)
            res = sp.resultant(
                sp.expand(p.as_expr().subs(T, _Z - c * _Y)), g.poly.as_expr().subs(_Z, _Y), _Y
            )
            new_p = _choose(sp.Poly(res, _Z, domain="QQ"), cand)
            new_p = sp.Poly(new_p.as_expr().subs(_Z, T), T, domain="QQ")
            root = _common_root(new_p, cand, g.poly, p, c, g.value)
            if root is None:
                continue
            old = (sp.Poly(T, T, domain="QQ") - c * root).rem(new_p)
            exprs = [
                sp.Poly(sp.expand(e.as_expr().subs(T, old.as_expr())), T, domain="QQ").rem(new_p)
                for e in exprs
            ]
            exprs.append(root)
            p, theta = new_p, cand
            break
        else:
            raise NumberFieldError("no primitive element found")
    _, p_int = p.clear_denoms()
    p_int = sp.Poly(p_int.as_expr(), T, domain="ZZ").primitive()[1]
    p_int, lo, hi = _isolate(p_int, theta)

    p_q = sp.Poly(p_int.as_expr(), T, domain="QQ")

    def coeffs(atom: sp.Expr, part: Part) -> tuple[Coeffs, Coeffs]:
        kind, data = part
        if kind == "rational":
            return _coeffs(sp.Poly(data, T, domain="QQ")), ()
        if kind == "real":
            return _coeffs(exprs[cast(int, data)]), ()
        gen, c = cast(tuple[int, int], data)
        info = _info(atom)
        assert isinstance(info, _Info)
        root = _complex_root(p_q, theta, exprs[gen], c, info.m, info.value)
        if root is None:
            raise NumberFieldError(f"{atom} is not in the field")
        return _coeffs(root[0]), _coeffs(root[1])

    poly = tuple(int(c) for c in reversed(p_int.all_coeffs()))
    coeff = {a: coeffs(a, part) for a, part in parts.items()}
    for _ in range(60):
        rows = []
        for a, (re_cs, im_cs) in coeff.items():
            _, own = _rect(a)
            if own is None:
                rows.append((a, re_cs, im_cs, (sp.Integer(0),) * 4))
                continue
            box = _box(re_cs, im_cs, lo, hi)
            rows.append((a, re_cs, im_cs, _hull(own, box)))
        cert = FieldCertificate(poly, lo, hi, tuple(rows))
        try:
            check(cert)
            return cert
        except NumberFieldError as exc:
            if "isolate one root" not in str(exc):
                raise
        narrowed = cert
        for _ in range(6):
            narrowed = _narrow(narrowed)
        lo, hi = narrowed.lo, narrowed.hi
    check(cert)
    return cert


def _hull(a: Rect, b: Rect) -> Rect:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _narrow(cert: FieldCertificate) -> FieldCertificate:
    """``cert`` with its interval halved around the root."""
    p = cert.p()
    mid = (cert.lo + cert.hi) / 2
    if p.eval(mid) < 0:
        return FieldCertificate(cert.poly, mid, cert.hi, cert.atoms)
    return FieldCertificate(cert.poly, cert.lo, mid, cert.atoms)


__all__ = [
    "FieldCertificate",
    "NumberFieldError",
    "T",
    "algebraic",
    "atoms",
    "bernstein_derivative",
    "certificate",
    "check",
    "derivative",
    "element",
    "inverse",
    "key",
]
