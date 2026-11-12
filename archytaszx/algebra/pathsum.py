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

"""Path sums over Boolean variables: cyclotomic constants, multilinear phase polynomials, the
reduction rules [Elim], [HH] and [omega], and the canonical form of a reduced sum.

A :class:`PathSum` denotes ``E(i) = c * sqrt(2)^k * [constraints(i) = 0] * Sum_{y in Z_2^m}
exp(2 pi i P(i, y))`` over its boundary variables ``i`` and path variables ``y``. ``P`` is a
:class:`PhasePoly`, a multilinear polynomial with coefficients in Q/Z (turns); every constraint
is a polynomial over F_2, a set of monomials read as their XOR. Pure Python integers and
Fractions throughout; no sum is ever enumerated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import Any, cast

Mono = frozenset[int]
F2Poly = frozenset[Mono]
EMPTY: Mono = frozenset()
QMono = tuple[tuple[int, int], ...]
"""A monomial over Z_p: sorted (variable, exponent) pairs, every exponent in 1 .. p-1."""


class PathSumError(Exception):
    """Base class for every error raised by the path-sum modules."""


class PathSumUnsupported(PathSumError):
    """A diagram or a sum outside the class the path-sum calculus covers."""


# -- cyclotomic constants -----------------------------------------------------------------------


def _poly_divmod(num: list[int], den: list[int]) -> tuple[list[int], list[int]]:
    """Integer polynomial division by a monic ``den``, coefficient lists lowest degree first."""
    num = list(num)
    q = [0] * max(1, len(num) - len(den) + 1)
    for i in range(len(num) - len(den), -1, -1):
        c = num[i + len(den) - 1]
        q[i] = c
        if c:
            for j, dc in enumerate(den):
                num[i + j] -= c * dc
    return q, num[: len(den) - 1]


@lru_cache(maxsize=256)
def cyclotomic(n: int) -> tuple[int, ...]:
    """The coefficients of the n-th cyclotomic polynomial, lowest degree first."""
    poly = [-1] + [0] * (n - 1) + [1]
    for d in range(1, n):
        if n % d == 0:
            poly, rest = _poly_divmod(poly, list(cyclotomic(d)))
            if any(rest):
                raise PathSumError(f"cyclotomic({n}): inexact division by Phi_{d}")
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return tuple(poly)


class Cyclo:
    """An element of Q(zeta_n) as rational coefficients over zeta_n^0 .. zeta_n^(n-1)."""

    __slots__ = ("c", "n")

    def __init__(self, n: int, c: Mapping[int, Fraction] | None = None) -> None:
        """Build from exponent -> coefficient, exponents taken mod ``n``."""
        self.n = n
        self.c: dict[int, Fraction] = {}
        for e, v in (c or {}).items():
            k = e % n
            s = self.c.get(k, Fraction(0)) + v
            if s:
                self.c[k] = s
            else:
                self.c.pop(k, None)

    @staticmethod
    def rational(v: Fraction | int) -> Cyclo:
        """The rational ``v``."""
        return Cyclo(1, {0: Fraction(v)})

    @staticmethod
    def root(turns: Fraction) -> Cyclo:
        """``exp(2 pi i turns)``."""
        t = Fraction(turns) % 1
        return Cyclo(t.denominator, {t.numerator: Fraction(1)})

    def lift(self, m: int) -> Cyclo:
        """The same number over zeta_m, ``m`` a multiple of ``n``."""
        f = m // self.n
        return Cyclo(m, {e * f: v for e, v in self.c.items()})

    def _common(self, other: Cyclo) -> tuple[Cyclo, Cyclo]:
        m = self.n * other.n // math.gcd(self.n, other.n)
        return self.lift(m), other.lift(m)

    def __add__(self, other: Cyclo) -> Cyclo:
        a, b = self._common(other)
        out = dict(a.c)
        for e, v in b.c.items():
            out[e] = out.get(e, Fraction(0)) + v
        return Cyclo(a.n, out)

    def __neg__(self) -> Cyclo:
        return Cyclo(self.n, {e: -v for e, v in self.c.items()})

    def __sub__(self, other: Cyclo) -> Cyclo:
        return self + (-other)

    def __mul__(self, other: Cyclo) -> Cyclo:
        a, b = self._common(other)
        out: dict[int, Fraction] = {}
        for e1, v1 in a.c.items():
            for e2, v2 in b.c.items():
                k = (e1 + e2) % a.n
                out[k] = out.get(k, Fraction(0)) + v1 * v2
        return Cyclo(a.n, out)

    def scale(self, v: Fraction | int) -> Cyclo:
        """This number times the rational ``v``."""
        return Cyclo(self.n, {e: x * v for e, x in self.c.items()})

    def conj(self) -> Cyclo:
        """The complex conjugate."""
        return Cyclo(self.n, {-e: v for e, v in self.c.items()})

    def reduced(self, m: int | None = None) -> tuple[Fraction, ...]:
        """The coefficients over zeta_m^0 .. zeta_m^(phi(m)-1) modulo Phi_m."""
        x = self if m is None else self.lift(m)
        phi = cyclotomic(x.n)
        deg = len(phi) - 1
        poly = [Fraction(0)] * max(x.n, deg)
        for e, v in x.c.items():
            poly[e] += v
        for i in range(len(poly) - 1, deg - 1, -1):
            c = poly[i]
            if c:
                for j, p in enumerate(phi):
                    poly[i - deg + j] -= c * p
        return tuple(poly[:deg])

    @property
    def is_zero(self) -> bool:
        """Whether this number is zero."""
        return not any(self.reduced())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cyclo):
            return NotImplemented
        a, b = self._common(other)
        return a.reduced() == b.reduced()

    def __hash__(self) -> int:
        return hash(self.n)

    def to_complex(self) -> complex:
        """The value as a complex float."""
        import cmath

        return sum(
            (float(v) * cmath.exp(2j * math.pi * e / self.n) for e, v in self.c.items()),
            complex(0),
        )

    def __repr__(self) -> str:
        return f"Cyclo({self.n}, {dict(sorted(self.c.items()))})"


SQRT2 = Cyclo(8, {1: Fraction(1), 7: Fraction(1)})
"""sqrt(2) = zeta_8 + zeta_8^-1."""


def with_sqrt2(c: Cyclo, k: int) -> Cyclo:
    """``c * sqrt(2)^k`` as one cyclotomic number."""
    out = c
    if k % 2:
        out = out * SQRT2
        k -= 1
    half = k // 2
    return out.scale(Fraction(2) ** half)


def _sqrt_integer(n: int) -> tuple[Cyclo, int]:
    """sqrt(n) as (cyclotomic number, power of sqrt 2): odd primes by their quadratic Gauss
    sums."""
    out, k = Cyclo.rational(1), 0
    m, p = n, 2
    while m > 1:
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        if e:
            out = out.scale(Fraction(p) ** (e // 2))
            if e % 2:
                if p == 2:
                    k += 1
                else:
                    g = Cyclo(
                        p,
                        {
                            a: Fraction(1 if pow(a, (p - 1) // 2, p) == 1 else -1)
                            for a in range(1, p)
                        },
                    )
                    if p % 4 == 3:
                        g = g * Cyclo.root(Fraction(-1, 4))
                    out = out * g
        p += 1
    return out, k


def _cyclo_sqrt(square: Cyclo, approx: complex) -> Cyclo | None:
    """A cyclotomic r with r * r == square exactly and r near ``approx``, searched among
    a z^i + b z^j over z = zeta_M, M a multiple of square's order up to 8 times it, a and b
    rationals of denominator at most 64 fitted to ``approx``."""
    import cmath

    for factor in (1, 2, 4, 8):
        m = math.lcm(square.n, 8) * factor
        roots = [cmath.exp(2j * math.pi * e / m) for e in range(m)]
        for i in range(m):
            for j in range(i, m):
                u, v = roots[i], roots[j]
                det = (u.conjugate() * v).imag
                if abs(det) < 1e-12:
                    if i != j:
                        continue
                    a = (approx / u).real
                    cand = Cyclo(m, {i: Fraction(a).limit_denominator(64)})
                else:
                    a = (approx.conjugate() * v).imag / det
                    b = (u.conjugate() * approx).imag / det
                    cand = Cyclo(
                        m,
                        {
                            i: Fraction(a).limit_denominator(64),
                            j: Fraction(b).limit_denominator(64),
                        },
                    )
                if abs(cand.to_complex() - approx) < 1e-9 and cand * cand == square:
                    return cand
    return None


def _cyclo_inverse(c: Cyclo) -> Cyclo | None:
    """1 / c in Q(zeta_n): the solution of c x = 1 over the basis zeta_n^0 .. zeta_n^(phi-1)
    by Gauss-Jordan elimination over the rationals; None when c is zero."""
    n = c.n
    cols = [(c * Cyclo(n, {j: Fraction(1)})).reduced(n) for j in range(len(c.reduced(n)))]
    size = len(cols)
    rows = [[cols[j][i] for j in range(size)] + [Fraction(1 if i == 0 else 0)] for i in range(size)]
    for col in range(size):
        piv = next((r for r in range(col, size) if rows[r][col]), None)
        if piv is None:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        lead = rows[col][col]
        rows[col] = [v / lead for v in rows[col]]
        for r in range(size):
            if r != col and rows[r][col]:
                f = rows[r][col]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[col], strict=True)]
    return Cyclo(n, {j: rows[j][size] for j in range(size) if rows[j][size]})


def constant(expr: Any) -> tuple[Cyclo, int]:
    """A diagram scalar as (cyclotomic number, power of sqrt 2); raises when it is not one."""
    import sympy as sp  # type: ignore[import-untyped]

    expr = cast(Any, sp.sympify(expr))
    if expr.is_Rational:
        return Cyclo.rational(Fraction(int(expr.p), int(expr.q))), 0
    if expr == sp.I:
        return Cyclo.root(Fraction(1, 4)), 0
    if isinstance(expr, sp.exp):
        t = sp.nsimplify(expr.args[0] / (2 * sp.pi * sp.I)) if expr.args[0].is_number else None
        if t is not None and t.is_Rational:
            return Cyclo.root(Fraction(int(t.p), int(t.q))), 0
    if expr.is_Mul:
        c, k = Cyclo.rational(1), 0
        for arg in expr.args:
            ca, ka = constant(arg)
            c, k = c * ca, k + ka
        return c, k
    if expr.is_Add:
        total = Cyclo.rational(0)
        for arg in expr.args:
            ca, ka = constant(arg)
            total = total + with_sqrt2(ca, ka)
        return total, 0
    if expr.is_Pow:
        base, exp = expr.args
        if (
            base.is_Integer
            and int(base) > 2
            and exp.is_Rational
            and (2 * exp).is_Integer
            and not exp.is_Integer
        ):
            root, k = _sqrt_integer(int(base))
            c, kk = Cyclo.rational(Fraction(int(base)) ** (int(2 * exp) // 2)), 0
            if int(2 * exp) < 0:
                c = Cyclo.rational(Fraction(1, int(base)) ** ((-int(2 * exp) + 1) // 2))
                return c * root, k
            return c * root, k + kk
        if base == 2 and exp.is_Rational and (2 * exp).is_Integer:
            return Cyclo.rational(1), int(2 * exp)
        if base == -1 and exp.is_Rational:
            return Cyclo.root(Fraction(int(exp.p), 2 * int(exp.q))), 0
        if base == sp.I and exp.is_Rational:
            return Cyclo.root(Fraction(int(exp.p), 4 * int(exp.q))), 0
        if base.is_Rational and exp.is_Integer:
            return Cyclo.rational(Fraction(int(base.p), int(base.q)) ** int(exp)), 0
        if exp.is_Rational and abs(exp) == sp.Rational(1, 2) and not base.is_Integer:
            cb, kb = constant(base)
            square = with_sqrt2(cb, kb)
            half_power = _cyclo_sqrt(square, complex(sp.N(sp.sqrt(base), 30)))
            if half_power is not None:
                if exp < 0:
                    inv = _cyclo_inverse(half_power)
                    if inv is None:
                        raise PathSumUnsupported(f"the scalar {expr} has no inverse found")
                    half_power = inv
                return half_power, 0
        if exp.is_Integer and int(exp) >= 0:
            cb, kb = constant(base)
            c, k = Cyclo.rational(1), 0
            for _ in range(int(exp)):
                c, k = c * cb, k + kb
            return c, k
    raise PathSumUnsupported(
        f"the scalar {expr} is not a cyclotomic number times a power of sqrt 2"
    )


# -- phase polynomials ----------------------------------------------------------------------------


def _mod1(v: Fraction) -> Fraction:
    return v % 1


@dataclass
class PhasePoly:
    """A multilinear polynomial ``Sum_S c_S prod_{v in S} v`` with every ``c_S`` in [0, 1)."""

    terms: dict[Mono, Fraction] = field(default_factory=dict)

    def copy(self) -> PhasePoly:
        """An independent copy."""
        return PhasePoly(dict(self.terms))

    def add(self, mono: Iterable[int], coeff: Fraction) -> None:
        """Add ``coeff * prod(mono)``."""
        key = frozenset(mono)
        v = _mod1(self.terms.get(key, Fraction(0)) + coeff)
        if v:
            self.terms[key] = v
        else:
            self.terms.pop(key, None)

    def variables(self) -> set[int]:
        """Every variable some term carries."""
        out: set[int] = set()
        for m in self.terms:
            out |= m
        return out

    def substitute(self, var: int, value: F2Poly) -> None:
        """Replace ``var`` by the Boolean lift of the F_2 polynomial ``value``."""
        hit = [(m, c) for m, c in self.terms.items() if var in m]
        for m, _c in hit:
            del self.terms[m]
        items = sorted(value, key=_mono_key)
        for m, c in hit:
            rest = m - {var}
            for mono, coeff in _lift_terms(items, c):
                self.add(rest | mono, coeff)

    def assign(self, values: Mapping[int, int]) -> None:
        """Set each variable of ``values`` to its 0 or 1."""
        old = self.terms
        self.terms = {}
        for m, c in old.items():
            if any(values.get(v, 1) == 0 for v in m if v in values):
                continue
            self.add(m - set(values), c)


def _mono_key(m: Mono) -> tuple[int, tuple[int, ...]]:
    return (len(m), tuple(sorted(m)))


def _lift_terms(items: Sequence[Mono], coeff: Fraction) -> list[tuple[Mono, Fraction]]:
    """``coeff`` times the Boolean lift of the XOR of ``items``, terms whose coefficient is an
    integer dropped: ``Sum_{S nonempty} (-2)^(|S|-1) prod S``."""
    out: list[tuple[Mono, Fraction]] = []
    den = coeff.denominator
    depth = den.bit_length() if den & (den - 1) == 0 else len(items)
    for size in range(1, min(len(items), depth) + 1):
        factor = coeff * (-2) ** (size - 1)
        if factor.denominator == 1:
            break
        for group in combinations(items, size):
            mono: frozenset[int] = frozenset().union(*group)
            out.append((mono, factor))
    return out


def f2_add(a: F2Poly, b: F2Poly) -> F2Poly:
    """The XOR of two F_2 polynomials."""
    return frozenset(a ^ b)


def f2_substitute(p: F2Poly, var: int, value: F2Poly) -> F2Poly:
    """``p`` with ``var`` replaced by ``value``, over F_2."""
    out: set[Mono] = set()
    for m in p:
        if var not in m:
            out ^= {m}
            continue
        rest = m - {var}
        for v in value:
            out ^= {rest | v}
    return frozenset(out)


def f2_mul(a: F2Poly, b: F2Poly) -> F2Poly:
    """The product of two F_2 polynomials, multilinear."""
    out: set[Mono] = set()
    for m1 in a:
        for m2 in b:
            out ^= {m1 | m2}
    return frozenset(out)


ONE: F2Poly = frozenset({EMPTY})


def int_lift(p: F2Poly, modulus: int) -> dict[Mono, int]:
    """The integer multilinear polynomial equal to the F_2 polynomial ``p`` on Booleans,
    coefficients mod ``modulus``: ``a xor b = a + b - 2ab`` folded over the monomials."""
    out: dict[Mono, int] = {}
    for m in sorted(p, key=_mono_key):
        nxt = dict(out)
        nxt[m] = nxt.get(m, 0) + 1
        for m2, c in out.items():
            key = m | m2
            nxt[key] = nxt.get(key, 0) - 2 * c
        out = {k: v % modulus for k, v in nxt.items() if v % modulus}
    return out


def f2_assign(p: F2Poly, values: Mapping[int, int]) -> F2Poly:
    """``p`` with each variable of ``values`` set."""
    out: set[Mono] = set()
    for m in p:
        if any(values.get(v, 1) == 0 for v in m if v in values):
            continue
        out ^= {m - set(values)}
    return frozenset(out)


# -- path sums -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One rule application of a reduction: its rule, variable, pivot and substitution; a
    dyadic step also lists its digit group, fresh variables and digit polynomials, a qudit
    step its F_p polynomial."""

    rule: str
    var: int
    pivot: int | None = None
    value: F2Poly | None = None
    group: tuple[int, ...] = ()
    fresh: tuple[int, ...] = ()
    values: tuple[F2Poly, ...] = ()
    poly: tuple[tuple[QMono, int], ...] = ()


@dataclass
class PathSum:
    """``const * sqrt(2)^sqrt2 * [constraints = 0] * Sum_path exp(2 pi i phase)``."""

    boundary: tuple[int, ...]
    path: set[int]
    phase: PhasePoly
    const: Cyclo
    sqrt2: int = 0
    constraints: list[F2Poly] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)

    def copy(self) -> PathSum:
        """An independent copy."""
        return PathSum(
            self.boundary,
            set(self.path),
            self.phase.copy(),
            self.const,
            self.sqrt2,
            list(self.constraints),
            list(self.trace),
        )

    @property
    def is_zero(self) -> bool:
        """Whether the constant is zero or a constraint is the constant one."""
        return self.const.is_zero or any(c == frozenset({EMPTY}) for c in self.constraints)


def _occurrences(phase: PhasePoly) -> dict[int, list[tuple[Mono, Fraction]]]:
    out: dict[int, list[tuple[Mono, Fraction]]] = {}
    for m, c in phase.terms.items():
        for v in m:
            out.setdefault(v, []).append((m, c))
    return out


def _try_elim(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    for y in sorted(ps.path):
        if y not in occ:
            ps.path.discard(y)
            ps.sqrt2 += 2
            ps.trace.append(Step("elim", y))
            return True
    return False


def _hh_quotient(y: int, terms: Sequence[tuple[Mono, Fraction]]) -> F2Poly | None:
    """The F_2 polynomial Q with the y-terms equal to y Q / 2, or None."""
    half = Fraction(1, 2)
    if any(c != half for _m, c in terms):
        return None
    return frozenset(m - {y} for m, _c in terms)


def _pivot(q: F2Poly, path: set[int], y: int) -> int | None:
    count: dict[int, int] = {}
    for m in q:
        for v in m:
            count[v] = count.get(v, 0) + 1
    for z in sorted(path):
        if z != y and frozenset({z}) in q and count.get(z) == 1:
            return z
    return None


def _drop_terms(ps: PathSum, terms: Sequence[tuple[Mono, Fraction]]) -> None:
    for m, _c in terms:
        ps.phase.terms.pop(m, None)


def _try_hh_pivot(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    for y in sorted(ps.path):
        terms = occ.get(y, [])
        q = _hh_quotient(y, terms)
        if q is None:
            continue
        z = _pivot(q, ps.path, y)
        if z is None:
            continue
        value = frozenset(q - {frozenset({z})})
        _drop_terms(ps, terms)
        ps.phase.substitute(z, value)
        ps.constraints = [f2_substitute(c, z, value) for c in ps.constraints]
        ps.path.discard(y)
        ps.path.discard(z)
        ps.sqrt2 += 2
        ps.trace.append(Step("hh_pivot", y, z, value))
        return True
    return False


def _try_omega(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    quarter, three = Fraction(1, 4), Fraction(3, 4)
    for y in sorted(ps.path):
        terms = occ.get(y, [])
        single = frozenset({y})
        lin = [c for m, c in terms if m == single]
        if not lin or lin[0] not in (quarter, three):
            continue
        rest = [(m, c) for m, c in terms if m != single]
        q = _hh_quotient(y, rest) if rest else frozenset()
        if q is None:
            continue
        _drop_terms(ps, terms)
        sign = -1 if lin[0] == quarter else 1
        for mono, coeff in _lift_terms(sorted(q, key=_mono_key), Fraction(sign, 4)):
            ps.phase.add(mono, coeff)
        ps.const = ps.const * Cyclo.root(Fraction(1 if sign < 0 else -1, 8))
        ps.sqrt2 += 1
        ps.path.discard(y)
        ps.trace.append(Step("omega" if sign < 0 else "omega_conj", y, None, q))
        return True
    return False


def _try_hh_gen(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    """[HHgen]: y Q / 2 with Q = z A + B, z a path variable, (1 + A)(1 + B) = 0: z := B."""
    for y in sorted(ps.path):
        q = _hh_quotient(y, occ.get(y, []))
        if not q:
            continue
        for z in sorted({v for m in q for v in m} & ps.path):
            a = frozenset(m - {z} for m in q if z in m)
            if a == ONE:
                continue
            b = frozenset(m for m in q if z not in m)
            if f2_mul(f2_add(ONE, a), f2_add(ONE, b)):
                continue
            ps.phase.substitute(z, b)
            ps.constraints = [f2_substitute(c, z, b) for c in ps.constraints]
            ps.path.discard(z)
            ps.trace.append(Step("hh_gen", y, z, b))
            return True
    return False


def _dyadic_group(
    y: int, ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]
) -> tuple[list[int], dict[Mono, int], int] | None:
    """The digit group ``Y_0 = y, Y_1, ..`` with phase ``Y S / 2^k`` and S's integer
    coefficients mod 2^k, or None."""
    terms = occ.get(y, [])
    if not terms:
        return None
    den = max(c.denominator for _m, c in terms)
    if den < 4 or den & (den - 1):
        return None
    k = den.bit_length() - 1
    s0 = {m - {y}: int(c * den) % den for m, c in terms}
    inside = {v for m in s0 for v in m}
    digits = [y]
    for a in range(1, k):
        expected = {m: Fraction((s << a) % den, den) for m, s in s0.items() if (s << a) % den}
        hit = None
        for z in sorted(ps.path - set(digits) - inside):
            zt = occ.get(z, [])
            if len(zt) == len(expected) and all(expected.get(m - {z}) == c for m, c in zt):
                hit = z
                break
        if hit is None:
            return None
        digits.append(hit)
    return digits, s0, k


def _try_dyadic(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    """[Dyadic]: Sum_{Y in Z_2^k} exp(2 pi i Y S / 2^k) = Sum_{y' in Z_2^k} (-1)^(Sum_j y'_j
    b_j), b_j the j-th binary digit of S as an F_2 polynomial."""
    for y in sorted(ps.path):
        found = _dyadic_group(y, ps, occ)
        if found is None:
            continue
        digits, s, k = found
        bits: list[F2Poly] = []
        for j in range(k):
            mod = 1 << (k - j)
            b = frozenset(m for m, c in s.items() if c % 2)
            bits.append(b)
            lift = int_lift(b, mod)
            rest: dict[Mono, int] = {}
            for m in set(s) | set(lift):
                v = (s.get(m, 0) - lift.get(m, 0)) % mod
                if v % 2:
                    raise PathSumError("dyadic digit: an odd remainder")
                if v // 2:
                    rest[m] = v // 2
            s = rest
        for z in digits:
            _drop_terms(ps, occ.get(z, []))
            ps.path.discard(z)
        top = max([*ps.boundary, *ps.path, *digits, *ps.phase.variables()]) + 1
        fresh = tuple(range(top, top + k))
        half = Fraction(1, 2)
        for f, b in zip(fresh, bits, strict=True):
            ps.path.add(f)
            for m in b:
                ps.phase.add(m | {f}, half)
        ps.trace.append(Step("dyadic", y, None, None, tuple(digits), fresh, tuple(bits)))
        return True
    return False


def _try_rem(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    """[Rem] for a monomial quotient: y m / 2 forces m = 0, so terms divisible by m drop."""
    for y in sorted(ps.path):
        q = _hh_quotient(y, occ.get(y, []))
        if q is None or len(q) != 1:
            continue
        (m,) = q
        if not m:
            continue
        hit = [t for t in ps.phase.terms if m <= t and y not in t]
        if not hit:
            continue
        for t in hit:
            del ps.phase.terms[t]
        ps.trace.append(Step("rem", y, None, q))
        return True
    return False


def _try_hh_constraint(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    for y in sorted(ps.path):
        terms = occ.get(y, [])
        q = _hh_quotient(y, terms)
        if q is None or not q:
            continue
        if any(v in ps.path for m in q for v in m):
            continue
        _drop_terms(ps, terms)
        mono = next((m for m in q if m), EMPTY)
        if len(q) == 2 and EMPTY in q and len(mono) > 1:
            ps.constraints.extend(frozenset({EMPTY, frozenset({v})}) for v in sorted(mono))
        else:
            ps.constraints.append(q)
        ps.path.discard(y)
        ps.sqrt2 += 2
        ps.trace.append(Step("hh_constraint", y, None, q))
        return True
    return False


def _try_hh_nl(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    """[HHnl]: y Q / 2 + y' Q' / 2, y and y' in neither quotient: y' := y + y Q, factor 2."""
    forms = {}
    for y in sorted(ps.path):
        q = _hh_quotient(y, occ.get(y, []))
        if q:
            forms[y] = q
    for y, q in forms.items():
        qv = {v for m in q for v in m}
        for y2, q2 in forms.items():
            if y2 <= y or y2 in qv or y in {v for m in q2 for v in m}:
                continue
            value = frozenset({frozenset({y})} ^ {m | {y} for m in q})
            ps.phase.substitute(y2, value)
            ps.path.discard(y2)
            ps.sqrt2 += 2
            ps.trace.append(Step("hh_nl", y, y2, value))
            return True
    return False


def reduce(ps: PathSum) -> PathSum:
    """``ps`` with [Elim], [HH] on a path pivot, [HHgen], [omega], [Dyadic], [Rem], [HH] on a
    boundary constraint and [HHnl] applied to a fixpoint, in that order, the lowest variable
    first."""
    out = ps.copy()
    while out.path and not out.is_zero:
        occ = _occurrences(out.phase)
        if not (
            _try_elim(out, occ)
            or _try_hh_pivot(out, occ)
            or _try_hh_gen(out, occ)
            or _try_omega(out, occ)
            or _try_dyadic(out, occ)
            or _try_rem(out, occ)
            or _try_hh_constraint(out, occ)
            or _try_hh_nl(out, occ)
        ):
            break
    return out


# -- canonical form ----------------------------------------------------------------------


@dataclass(frozen=True)
class Canon:
    """A reduced path sum with no path variable: ``const * exp(2 pi i phase) * [rows]``. Each
    row sets a pivot boundary variable to an F_2 polynomial (monomials, XOR) of the free
    variables; the phase is over the free variables."""

    zero: bool
    rows: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]
    phase: tuple[tuple[tuple[int, ...], Fraction], ...]
    const: Cyclo
    boundary: tuple[int, ...]

    def same(self, other: Canon) -> bool:
        """Whether both forms carry the same data, so denote one function."""
        if self.zero or other.zero:
            return self.zero and other.zero
        return (
            self.boundary == other.boundary
            and self.rows == other.rows
            and self.phase == other.phase
            and self.const == other.const
        )

    def value(self, values: Mapping[int, int]) -> Cyclo:
        """The entry at a full boundary assignment."""
        if self.zero:
            return Cyclo.rational(0)
        for pivot, rest in self.rows:
            if values[pivot] != _f2_eval(rest, values):
                return Cyclo.rational(0)
        turns = sum(
            (c for m, c in self.phase if all(values[v] for v in m)),
            Fraction(0),
        )
        return self.const * Cyclo.root(turns)

    def support_point(self) -> dict[int, int]:
        """The support point with every free variable 0."""
        return self.complete({})

    def complete(self, free: Mapping[int, int]) -> dict[int, int]:
        """The support point with the given free values, the rest of the free variables 0."""
        pivots = {p for p, _r in self.rows}
        values = {v: int(free.get(v, 0)) for v in self.boundary if v not in pivots}
        for pivot, rest in self.rows:
            values[pivot] = _f2_eval(rest, values)
        return values

    @property
    def affine(self) -> bool:
        """Whether every row is affine."""
        return all(len(m) <= 1 for _p, rest in self.rows for m in rest)


def _f2_eval(rest: Iterable[Iterable[int]], values: Mapping[int, int]) -> int:
    return sum(1 for m in rest if all(values[v] for v in m)) % 2


def eliminate(
    constraints: Iterable[F2Poly], boundary: Sequence[int]
) -> list[tuple[int, F2Poly]] | list[None] | None:
    """Gauss-Jordan elimination over F_2 polynomials, in passes over the variables ascending
    until no pivot is found: each pivot is a variable occurring in a constraint only as a
    degree-one monomial, substituted out of every other constraint and row. Returns (pivot,
    polynomial) rows, ``[None]`` when a constraint is the constant one, and None when a
    constraint is left without a pivot."""
    work = sorted({c for c in constraints if c}, key=lambda c: sorted(_mono_key(m) for m in c))
    rows: list[tuple[int, F2Poly]] = []
    if any(c == frozenset({EMPTY}) for c in work):
        return [None]
    progress = True
    while work and progress:
        progress = False
        for v in sorted(set(boundary) - {q for q, _r in rows}):
            single = frozenset({v})
            hit = next(
                (c for c in work if single in c and not any(v in m for m in c if m != single)),
                None,
            )
            if hit is None:
                continue
            work.remove(hit)
            rest = frozenset(hit - {single})
            work = [f2_substitute(c, v, rest) for c in work]
            rows = [(q, f2_substitute(r, v, rest)) for q, r in rows]
            rows.append((v, rest))
            work = [c for c in work if c]
            progress = True
            if any(c == frozenset({EMPTY}) for c in work):
                return [None]
    if work:
        return None
    return sorted(rows, key=lambda r: r[0])


def _zero_canon(boundary: tuple[int, ...]) -> Canon:
    return Canon(True, (), (), Cyclo.rational(0), boundary)


def canonical(ps: PathSum) -> Canon | None:
    """The canonical form of a reduced path sum; None when a path variable remains or a
    constraint has no pivot."""
    if ps.is_zero:
        return _zero_canon(ps.boundary)
    if ps.path:
        return None
    rows = eliminate(ps.constraints, ps.boundary)
    if rows is None:
        return None
    if rows and rows[0] is None:
        return _zero_canon(ps.boundary)
    pivots = cast(list[tuple[int, F2Poly]], rows)

    phase = ps.phase.copy()
    for pivot, rest in pivots:
        phase.substitute(pivot, rest)
    const = with_sqrt2(ps.const, ps.sqrt2)
    c0 = phase.terms.pop(EMPTY, Fraction(0))
    if c0:
        const = const * Cyclo.root(c0)
    if const.is_zero:
        return _zero_canon(ps.boundary)
    return Canon(
        False,
        tuple((p, tuple(sorted(tuple(sorted(m)) for m in rest))) for p, rest in pivots),
        tuple(sorted(((tuple(sorted(m)), c) for m, c in phase.terms.items()))),
        const,
        ps.boundary,
    )


@dataclass(frozen=True)
class Residual:
    """A reduced path sum with its path variables renamed canonically: constant, support
    rows (or the raw constraints when they have no pivots), phase over boundary and renamed
    path variables ``len(boundary) ..``."""

    rows: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]
    raw: tuple[tuple[tuple[int, ...], ...], ...]
    phase: tuple[tuple[tuple[int, ...], Fraction], ...]
    const: Cyclo
    boundary: tuple[int, ...]
    paths: int

    def same(self, other: Residual) -> bool:
        """Whether both forms carry the same data, so denote one function."""
        return (
            self.boundary == other.boundary
            and self.paths == other.paths
            and self.rows == other.rows
            and self.raw == other.raw
            and self.phase == other.phase
            and self.const == other.const
        )


RESIDUAL_ORDERINGS = 5040
"""The most orderings of colour-tied path variables residual_form compares."""


def _refine(phase: PhasePoly, path: Sequence[int]) -> dict[int, int]:
    """Colour refinement of path variables by the terms they carry, colours as ranks."""
    colour = {v: 0 for v in path}
    occ = _occurrences(phase)
    for _ in range(len(path) + 1):
        sig = {
            v: (
                colour[v],
                tuple(
                    sorted(
                        (
                            c,
                            tuple(sorted(u for u in m if u not in colour)),
                            tuple(sorted(colour[u] for u in m if u in colour and u != v)),
                        )
                        for m, c in occ.get(v, [])
                    )
                ),
            )
            for v in path
        }
        ranks = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        new = {v: ranks[sig[v]] for v in path}
        if len(set(new.values())) == len(set(colour.values())):
            return new
        colour = new
    return colour


def residual_form(ps: PathSum) -> Residual | None:
    """The residual form of a reduced, nonzero path sum: support rows substituted, path
    variables renamed by colour refinement and the least encoding over tied orderings; None
    when the tied orderings exceed RESIDUAL_ORDERINGS."""
    rows = eliminate(ps.constraints, ps.boundary)
    phase = ps.phase.copy()
    raw: tuple[tuple[tuple[int, ...], ...], ...] = ()
    pivots: list[tuple[int, F2Poly]] = []
    if rows is None:
        raw = tuple(
            sorted({tuple(sorted(tuple(sorted(m)) for m in c)) for c in ps.constraints if c})
        )
    elif rows and rows[0] is None:
        return None
    else:
        pivots = cast(list[tuple[int, F2Poly]], rows)
    for pivot, rest in pivots:
        phase.substitute(pivot, rest)
    const = with_sqrt2(ps.const, ps.sqrt2)
    c0 = phase.terms.pop(EMPTY, Fraction(0))
    if c0:
        const = const * Cyclo.root(c0)
    path = sorted(ps.path)
    colour = _refine(phase, path)
    classes: dict[int, list[int]] = {}
    for v in path:
        classes.setdefault(colour[v], []).append(v)
    count = 1
    for members in classes.values():
        count *= math.factorial(len(members))
        if count > RESIDUAL_ORDERINGS:
            return None
    base = max(ps.boundary, default=-1) + 1
    ordered = [classes[c] for c in sorted(classes)]
    best: tuple[tuple[tuple[int, ...], Fraction], ...] | None = None
    for choice in product(*(permutations(m) for m in ordered)):
        name: dict[int, int] = {}
        for group in choice:
            for v in group:
                name[v] = base + len(name)
        enc = tuple(
            sorted((tuple(sorted(name.get(u, u) for u in m)), c) for m, c in phase.terms.items())
        )
        if best is None or enc < best:
            best = enc
    assert best is not None
    return Residual(
        tuple((p, tuple(sorted(tuple(sorted(m)) for m in rest))) for p, rest in pivots),
        raw,
        best,
        const,
        ps.boundary,
        len(path),
    )


def entry_at(ps: PathSum, values: Mapping[int, int]) -> Cyclo:
    """The entry at one boundary assignment: the boundary substituted, then reduced; raises
    :class:`PathSumUnsupported` when a path variable remains."""
    at = ps.copy()
    at.phase.assign(values)
    at.constraints = [f2_assign(c, values) for c in at.constraints]
    if any(c == frozenset({EMPTY}) for c in at.constraints):
        return Cyclo.rational(0)
    at = reduce(at)
    if at.is_zero:
        return Cyclo.rational(0)
    if at.path:
        raise PathSumUnsupported(f"{len(at.path)} path variable(s) left at this entry")
    if any(c for c in at.constraints):
        return Cyclo.rational(0)
    turns = at.phase.terms.get(EMPTY, Fraction(0))
    return with_sqrt2(at.const, at.sqrt2) * Cyclo.root(turns)


def rref(rows: Iterable[tuple[set[int], int]]) -> list[tuple[int, set[int], int]] | None:
    """Reduced row echelon form over F_2 with the variables in ascending order, as (pivot,
    other variables, constant); None when the system has no solution."""
    work = [(set(v), b) for v, b in rows]
    columns = sorted(set().union(*(v for v, _b in work))) if work else []
    done: list[tuple[set[int], int]] = []
    for col in columns:
        hit = next((i for i, (v, _b) in enumerate(work) if col in v), None)
        if hit is None:
            continue
        pv, pb = work.pop(hit)
        work = [(v ^ pv, b ^ pb) if col in v else (v, b) for v, b in work]
        done = [(v ^ pv, b ^ pb) if col in v else (v, b) for v, b in done]
        done.append((pv, pb))
    if any(b for v, b in work if not v):
        return None
    out = [(min(v), v - {min(v)}, b) for v, b in done]
    return sorted(out, key=lambda r: r[0])


def _solve(
    rows: Sequence[tuple[int, tuple[tuple[int, ...], ...]]],
    boundary: Sequence[int],
    extra: tuple[set[int], int],
) -> dict[int, int] | None:
    """A boundary point meeting affine ``rows`` and the extra affine row, free variables 0."""
    system = rref(
        [
            *(
                ({p} ^ {m[0] for m in rest if m}, sum(1 for m in rest if not m) % 2)
                for p, rest in rows
            ),
            extra,
        ]
    )
    if system is None:
        return None
    pivots = {p for p, _o, _b in system}
    values = {v: 0 for v in boundary if v not in pivots}
    for p, others, bit in system:
        values[p] = (bit + sum(values.get(v, 0) for v in others)) % 2
    return values


def witness_points(a: Canon, b: Canon) -> list[dict[int, int]]:
    """Boundary points to evaluate when two canonical forms differ: support points of each, a
    point of each affine support outside the other, and the indicator of each free monomial."""
    points: list[dict[int, int]] = []
    for x, y in ((a, b), (b, a)):
        if x.zero:
            continue
        points.append(x.support_point())
        if x.affine and not y.zero and y.affine:
            for p, rest in y.rows:
                row = ({p} ^ {m[0] for m in rest if m}, (sum(1 for m in rest if not m) + 1) % 2)
                got = _solve(x.rows, x.boundary, row)
                if got is not None:
                    points.append(got)
        monos = {m for m, _c in x.phase} | {m for _p, rest in x.rows for m in rest}
        for mono in sorted(monos, key=lambda t: (len(t), t)):
            points.append(x.complete({v: 1 for v in mono}))
    return points


# -- odd prime d ------------------------------------------------------------------------------

ONE_Q: QMono = ()


def _qmono_mul(a: QMono, b: QMono, p: int) -> QMono:
    """The product of two monomials with x^p = x."""
    exps = dict(a)
    for v, e in b:
        exps[v] = exps.get(v, 0) + e
    return tuple(sorted((v, (e - 1) % (p - 1) + 1) for v, e in exps.items()))


@dataclass
class ModPoly:
    """A polynomial over F_p in reduced form, each exponent below p, coefficients mod p."""

    p: int
    terms: dict[QMono, int] = field(default_factory=dict)

    def copy(self) -> ModPoly:
        """An independent copy."""
        return ModPoly(self.p, dict(self.terms))

    @staticmethod
    def const(p: int, c: int) -> ModPoly:
        """The constant ``c``."""
        return ModPoly(p, {ONE_Q: c % p} if c % p else {})

    def add(self, mono: QMono, coeff: int) -> None:
        """Add ``coeff * mono``."""
        v = (self.terms.get(mono, 0) + coeff) % self.p
        if v:
            self.terms[mono] = v
        else:
            self.terms.pop(mono, None)

    def __add__(self, other: ModPoly) -> ModPoly:
        out = self.copy()
        for m, c in other.terms.items():
            out.add(m, c)
        return out

    def scale(self, c: int) -> ModPoly:
        """This polynomial times the constant ``c``."""
        return ModPoly(self.p, {m: v * c % self.p for m, v in self.terms.items() if v * c % self.p})

    def __mul__(self, other: ModPoly) -> ModPoly:
        out = ModPoly(self.p)
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                out.add(_qmono_mul(m1, m2, self.p), c1 * c2)
        return out

    def power(self, e: int) -> ModPoly:
        """This polynomial to the power ``e``."""
        out = ModPoly.const(self.p, 1)
        for _ in range(e):
            out = out * self
        return out

    def variables(self) -> set[int]:
        """Every variable some term carries."""
        return {v for m in self.terms for v, _e in m}

    def degree_in(self, var: int) -> int:
        """The largest exponent of ``var``."""
        return max((e for m in self.terms for v, e in m if v == var), default=0)

    def coefficient(self, var: int, e: int) -> ModPoly:
        """The polynomial multiplying ``var^e``."""
        out = ModPoly(self.p)
        for m, c in self.terms.items():
            if (var, e) in m:
                out.add(tuple(t for t in m if t[0] != var), c)
        return out

    def without(self, var: int) -> ModPoly:
        """The terms free of ``var``."""
        return ModPoly(
            self.p, {m: c for m, c in self.terms.items() if all(v != var for v, _e in m)}
        )

    def substitute(self, var: int, value: ModPoly) -> ModPoly:
        """``var`` replaced by ``value``."""
        out = self.without(var)
        powers: dict[int, ModPoly] = {}
        for m, c in self.terms.items():
            e = next((e for v, e in m if v == var), 0)
            if not e:
                continue
            if e not in powers:
                powers[e] = value.power(e)
            rest = ModPoly(self.p, {tuple(t for t in m if t[0] != var): c})
            out = out + rest * powers[e]
        return out

    def assign(self, values: Mapping[int, int]) -> ModPoly:
        """Each variable of ``values`` set to its value."""
        out = ModPoly(self.p)
        for m, c in self.terms.items():
            coeff = c
            rest = []
            for v, e in m:
                if v in values:
                    coeff = coeff * pow(values[v], e, self.p) % self.p
                else:
                    rest.append((v, e))
            if coeff:
                out.add(tuple(rest), coeff)
        return out

    def value(self, values: Mapping[int, int]) -> int:
        """The value at a full assignment, mod p."""
        return self.assign(values).terms.get(ONE_Q, 0)

    def key(self) -> tuple[tuple[QMono, int], ...]:
        """The sorted terms."""
        return tuple(sorted(self.terms.items()))


def legendre(a: int, p: int) -> int:
    """The Legendre symbol (a / p), p an odd prime."""
    r = pow(a % p, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def gauss_sum(a: int, p: int) -> Cyclo:
    """``Sum_{y in Z_p} omega^(a y^2)`` = (a/p) epsilon_p sqrt(p), as the Legendre-weighted sum
    of the p-th roots."""
    g = Cyclo(p, {b: Fraction(legendre(b, p)) for b in range(1, p)})
    return g.scale(legendre(a, p))


def sqrt_prime(p: int) -> Cyclo:
    """sqrt(p) for an odd prime p."""
    return _sqrt_integer(p)[0]


@dataclass
class QuditPathSum:
    """``const * [constraints = 0] * Sum_{path in Z_p^m} omega_p^phase``, the constraints and
    the phase polynomials over F_p."""

    p: int
    boundary: tuple[int, ...]
    path: set[int]
    phase: ModPoly
    const: Cyclo
    constraints: list[ModPoly] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)

    def copy(self) -> QuditPathSum:
        """An independent copy."""
        return QuditPathSum(
            self.p,
            self.boundary,
            set(self.path),
            self.phase.copy(),
            self.const,
            [c.copy() for c in self.constraints],
            list(self.trace),
        )

    @property
    def is_zero(self) -> bool:
        """Whether the constant is zero or a constraint is a nonzero constant."""
        return self.const.is_zero or any(
            bool(c.terms) and set(c.terms) == {ONE_Q} for c in self.constraints
        )


def _q_linear_pivot(q: ModPoly, candidates: set[int], y: int) -> tuple[int, int] | None:
    """A candidate variable z in Q only as ``a z``, a a constant, with its a; lowest first."""
    for z in sorted(q.variables() & candidates):
        if z == y:
            continue
        hits = [(m, c) for m, c in q.terms.items() if any(v == z for v, _e in m)]
        if len(hits) == 1 and hits[0][0] == ((z, 1),):
            return z, hits[0][1]
    return None


def _q_step(rule: str, y: int, pivot: int | None, poly: ModPoly) -> Step:
    """A qudit trace step with its F_p polynomial."""
    return Step(rule, y, pivot, poly=poly.key())


def _q_try_elim(ps: QuditPathSum) -> bool:
    used = ps.phase.variables()
    for y in sorted(ps.path):
        if y not in used:
            ps.path.discard(y)
            ps.const = ps.const.scale(ps.p)
            ps.trace.append(_q_step("q_elim", y, None, ModPoly(ps.p)))
            return True
    return False


def _q_try_hh_pivot(ps: QuditPathSum) -> bool:
    for y in sorted(ps.path):
        if ps.phase.degree_in(y) != 1:
            continue
        q = ps.phase.coefficient(y, 1)
        found = _q_linear_pivot(q, ps.path, y)
        if found is None:
            continue
        z, a = found
        rest = q + ModPoly(ps.p, {((z, 1),): -a % ps.p})
        value = rest.scale(-pow(a, -1, ps.p))
        ps.phase = ps.phase.without(y).substitute(z, value)
        ps.constraints = [c.substitute(z, value) for c in ps.constraints]
        ps.path.discard(y)
        ps.path.discard(z)
        ps.const = ps.const.scale(ps.p)
        ps.trace.append(_q_step("q_hh_pivot", y, z, value))
        return True
    return False


def _q_try_gauss(ps: QuditPathSum) -> bool:
    for y in sorted(ps.path):
        if ps.phase.degree_in(y) != 2:
            continue
        sq = ps.phase.coefficient(y, 2)
        if set(sq.terms) != {ONE_Q}:
            continue
        a = sq.terms[ONE_Q]
        q = ps.phase.coefficient(y, 1)
        ps.phase = ps.phase.without(y) + (q * q).scale(-pow(4 * a, -1, ps.p))
        ps.const = ps.const * gauss_sum(a, ps.p)
        ps.path.discard(y)
        ps.trace.append(_q_step("q_gauss", y, a, q))
        return True
    return False


def _q_try_hh_constraint(ps: QuditPathSum) -> bool:
    for y in sorted(ps.path):
        if ps.phase.degree_in(y) != 1:
            continue
        q = ps.phase.coefficient(y, 1)
        if q.variables() & ps.path:
            continue
        ps.phase = ps.phase.without(y)
        ps.constraints.append(q)
        ps.path.discard(y)
        ps.const = ps.const.scale(ps.p)
        ps.trace.append(_q_step("q_hh_constraint", y, None, q))
        return True
    return False


def reduce_qudit(ps: QuditPathSum) -> QuditPathSum:
    """``ps`` with [Elim], [HH] on a path pivot (z := -Q'/a), [Gauss] and [HH] on a boundary
    constraint applied to a fixpoint over Z_p, in that order, the lowest variable first."""
    out = ps.copy()
    while out.path and not out.is_zero:
        if not (
            _q_try_elim(out)
            or _q_try_hh_pivot(out)
            or _q_try_gauss(out)
            or _q_try_hh_constraint(out)
        ):
            break
    return out


QRow = tuple[int, tuple[tuple[QMono, int], ...]]


@dataclass(frozen=True)
class QuditCanon:
    """A reduced qudit path sum with no path variable: ``const * omega_p^phase * [rows]``, each
    row setting a pivot boundary variable to a polynomial of the free variables."""

    p: int
    zero: bool
    rows: tuple[QRow, ...]
    phase: tuple[tuple[QMono, int], ...]
    const: Cyclo
    boundary: tuple[int, ...]

    def same(self, other: QuditCanon) -> bool:
        """Whether both forms carry the same data, so denote one function."""
        if self.zero or other.zero:
            return self.zero and other.zero
        return (
            self.p == other.p
            and self.boundary == other.boundary
            and self.rows == other.rows
            and self.phase == other.phase
            and self.const == other.const
        )

    def value(self, values: Mapping[int, int]) -> Cyclo:
        """The entry at a full boundary assignment."""
        if self.zero:
            return Cyclo.rational(0)
        for pivot, rest in self.rows:
            if values[pivot] % self.p != ModPoly(self.p, dict(rest)).value(values):
                return Cyclo.rational(0)
        e = ModPoly(self.p, dict(self.phase)).value(values)
        return self.const * Cyclo.root(Fraction(e, self.p))

    def complete(self, free: Mapping[int, int]) -> dict[int, int]:
        """The support point with the given free values, the rest of the free variables 0."""
        pivots = {q for q, _r in self.rows}
        values = {v: int(free.get(v, 0)) % self.p for v in self.boundary if v not in pivots}
        for pivot, rest in self.rows:
            values[pivot] = ModPoly(self.p, dict(rest)).value(values)
        return values

    @property
    def affine(self) -> bool:
        """Whether every row is affine."""
        return all(sum(e for _v, e in m) <= 1 for _q, rest in self.rows for m, _c in rest)


def eliminate_qudit(
    constraints: Iterable[ModPoly], boundary: Sequence[int]
) -> list[tuple[int, ModPoly]] | list[None] | None:
    """Gauss-Jordan elimination over F_p polynomials in passes over the variables ascending,
    each pivot a variable occurring in a constraint only as ``a v`` with a constant: rows,
    ``[None]`` when a constraint is a nonzero constant, None when one has no pivot."""
    work = sorted((c for c in constraints if c.terms), key=lambda c: c.key())
    rows: list[tuple[int, ModPoly]] = []
    progress = True
    while work and progress:
        if any(set(c.terms) == {ONE_Q} for c in work):
            return [None]
        progress = False
        for v in sorted(set(boundary) - {q for q, _r in rows}):
            hit = None
            for c in work:
                found = _q_linear_pivot(c, {v}, -1)
                if found is not None:
                    hit = (c, found[1])
                    break
            if hit is None:
                continue
            c, a = hit
            work.remove(c)
            value = (c + ModPoly(c.p, {((v, 1),): -a % c.p})).scale(-pow(a, -1, c.p))
            work = [w.substitute(v, value) for w in work]
            work = [w for w in work if w.terms]
            rows = [(q, r.substitute(v, value)) for q, r in rows]
            rows.append((v, value))
            progress = True
            if any(set(w.terms) == {ONE_Q} for w in work):
                return [None]
    if work:
        return None
    return sorted(rows, key=lambda r: r[0])


def canonical_qudit(ps: QuditPathSum) -> QuditCanon | None:
    """The canonical form of a reduced qudit path sum; None when a path variable remains or a
    constraint has no pivot."""
    zero = QuditCanon(ps.p, True, (), (), Cyclo.rational(0), ps.boundary)
    if ps.is_zero:
        return zero
    if ps.path:
        return None
    rows = eliminate_qudit(ps.constraints, ps.boundary)
    if rows is None:
        return None
    if rows and rows[0] is None:
        return zero
    pivots = cast(list[tuple[int, ModPoly]], rows)
    phase = ps.phase.copy()
    for pivot, rest in pivots:
        phase = phase.substitute(pivot, rest)
    const = ps.const
    c0 = phase.terms.pop(ONE_Q, 0)
    if c0:
        const = const * Cyclo.root(Fraction(c0, ps.p))
    if const.is_zero:
        return zero
    return QuditCanon(
        ps.p, False, tuple((q, r.key()) for q, r in pivots), phase.key(), const, ps.boundary
    )


def _solve_mod(
    eqs: Sequence[tuple[dict[int, int], int]], boundary: Sequence[int], p: int
) -> dict[int, int] | None:
    """A solution of affine equations ``Sum a_v v = b`` over F_p, free variables 0."""
    work = [({v: c % p for v, c in a.items() if c % p}, b % p) for a, b in eqs]
    done: list[tuple[int, dict[int, int], int]] = []
    for col in sorted(boundary):
        hit = next((i for i, (a, _b) in enumerate(work) if col in a), None)
        if hit is None:
            continue
        a, b = work.pop(hit)
        inv = pow(a[col], -1, p)
        a = {v: c * inv % p for v, c in a.items()}
        b = b * inv % p
        work = [_clear(r, col, a, b, p) for r in work]
        done = [(q, *_clear((ra, rb), col, a, b, p)) for q, ra, rb in done]
        done.append((col, a, b))
    if any(b for a, b in work if not a):
        return None
    pivots = {q for q, _a, _b in done}
    values = {v: 0 for v in boundary if v not in pivots}
    for q, a, b in done:
        values[q] = (b - sum(c * values.get(v, 0) for v, c in a.items() if v != q)) % p
    return values


def _clear(
    row: tuple[dict[int, int], int], col: int, a: Mapping[int, int], b: int, p: int
) -> tuple[dict[int, int], int]:
    """``row`` minus the multiple of the pivot row ``(a, b)`` that clears ``col``."""
    ra, rb = row
    f = ra.get(col, 0)
    if not f:
        return row
    out = dict(ra)
    for v, c in a.items():
        out[v] = (out.get(v, 0) - f * c) % p
    return {v: c for v, c in out.items() if c}, (rb - f * b) % p


def _affine_row(
    pivot: int, rest: Sequence[tuple[QMono, int]], p: int, rhs: int
) -> tuple[dict[int, int], int]:
    """The row ``pivot = rest`` as ``pivot - linear(rest) = const(rest) + rhs``."""
    a = {pivot: 1}
    b = rhs
    for m, c in rest:
        if not m:
            b += c
        else:
            v = m[0][0]
            a[v] = (a.get(v, 0) - c) % p
    return a, b % p


def witness_points_qudit(a: QuditCanon, b: QuditCanon) -> list[dict[int, int]]:
    """Boundary points to evaluate when two qudit canonical forms differ: support points, a
    point of each affine support off the other's rows, and for each phase or row monomial the
    grid of its variables' values up to their exponents."""
    points: list[dict[int, int]] = []
    for x, y in ((a, b), (b, a)):
        if x.zero:
            continue
        points.append(x.complete({}))
        if x.affine and not y.zero and y.affine:
            base = [_affine_row(q, r, x.p, 0) for q, r in x.rows]
            for q, r in y.rows:
                got = _solve_mod([*base, _affine_row(q, r, x.p, 1)], x.boundary, x.p)
                if got is not None:
                    points.append(got)
        monos = {m for m, _c in (*x.phase, *y.phase)} | {m for _q, rest in x.rows for m, _c in rest}
        for mono in sorted(monos):
            for vals in product(*(range(e + 1) for _v, e in mono)):
                points.append(x.complete({v: c for (v, _e), c in zip(mono, vals, strict=True)}))
    return points


def entry_at_qudit(ps: QuditPathSum, values: Mapping[int, int]) -> Cyclo:
    """The entry at one boundary assignment: the boundary substituted, then reduced; raises
    :class:`PathSumUnsupported` when a path variable remains."""
    at = ps.copy()
    at.phase = at.phase.assign(values)
    at.constraints = [c.assign(values) for c in at.constraints]
    if any(c.terms for c in at.constraints):
        return Cyclo.rational(0)
    at = reduce_qudit(at)
    if at.is_zero:
        return Cyclo.rational(0)
    if at.path:
        raise PathSumUnsupported(f"{len(at.path)} path variable(s) left at this entry")
    if any(c.terms for c in at.constraints):
        return Cyclo.rational(0)
    return at.const * Cyclo.root(Fraction(at.phase.terms.get(ONE_Q, 0), at.p))


__all__ = [
    "SQRT2",
    "Canon",
    "Cyclo",
    "ModPoly",
    "PathSum",
    "PathSumError",
    "PathSumUnsupported",
    "PhasePoly",
    "QuditCanon",
    "QuditPathSum",
    "Residual",
    "Step",
    "canonical",
    "canonical_qudit",
    "constant",
    "cyclotomic",
    "eliminate",
    "eliminate_qudit",
    "entry_at",
    "entry_at_qudit",
    "gauss_sum",
    "legendre",
    "reduce",
    "reduce_qudit",
    "residual_form",
    "sqrt_prime",
    "with_sqrt2",
    "witness_points",
    "witness_points_qudit",
]
