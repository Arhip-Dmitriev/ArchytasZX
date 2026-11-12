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
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    """``const * sqrt(2)^sqrt2 * [constraints = 0] * Sum_path exp(2 pi i phase) * Prod a^F``,
    each amplitude atom ``a^F`` a cyclotomic base to the power of an F_2 polynomial."""

    boundary: tuple[int, ...]
    path: set[int]
    phase: PhasePoly
    const: Cyclo
    sqrt2: int = 0
    constraints: list[F2Poly] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)
    amps: list[tuple[Cyclo, F2Poly]] = field(default_factory=list)

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
            list(self.amps),
        )

    def amp_vars(self) -> set[int]:
        """Every variable an amplitude exponent carries."""
        return {v for _a, f in self.amps for m in f for v in m}

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


def _fold_amps(ps: PathSum) -> None:
    """Amplitude atoms with a constant exponent into the constant; a zero base with an exponent
    free of path variables into the constraint ``F = 0``."""
    keep = []
    for a, f in ps.amps:
        if not f:
            continue
        if f == ONE:
            ps.const = ps.const * a
        elif a.is_zero and not any(v in ps.path for m in f for v in m):
            ps.constraints.append(f)
        else:
            keep.append((a, f))
    ps.amps = keep


def _try_average(ps: PathSum) -> bool:
    """[Average]: y in no phase term or constraint, its atoms alpha^(f (1 + y)) and
    beta^(f y), f free of y: Sum_y = 2 ((alpha + beta) / 2)^f."""
    phase_vars = ps.phase.variables()
    cons_vars = {v for c in ps.constraints for m in c for v in m}
    for y in sorted(ps.path & ps.amp_vars() - phase_vars - cons_vars):
        alpha, beta = Cyclo.rational(1), Cyclo.rational(1)
        guard: F2Poly | None = None
        hit = []
        ok = True
        for i, (a, f) in enumerate(ps.amps):
            if not any(y in m for m in f):
                continue
            quotient = frozenset(m - {y} for m in f if y in m)
            rest = frozenset(m for m in f if y not in m)
            if guard is None:
                guard = quotient
            if quotient != guard or rest not in (frozenset(), guard):
                ok = False
                break
            if rest:
                alpha = alpha * a
            else:
                beta = beta * a
            hit.append(i)
        if not ok or guard is None:
            continue
        ps.amps = [t for i, t in enumerate(ps.amps) if i not in hit]
        ps.amps.append(((alpha + beta).scale(Fraction(1, 2)), guard))
        ps.path.discard(y)
        ps.sqrt2 += 2
        ps.trace.append(Step("average", y, None, guard))
        _fold_amps(ps)
        return True
    return False


def reduce(ps: PathSum) -> PathSum:
    """``ps`` with [Elim], [HH] on a path pivot, [HHgen], [omega], [Dyadic], [Rem], [HH] on a
    boundary constraint, [HHnl] and [Average] applied to a fixpoint, in that order, the lowest
    variable first; variables in an amplitude exponent take only [Average], and substitutions
    reach the exponents."""
    out = ps.copy()
    _fold_amps(out)
    while out.path and not out.is_zero:
        if out.amps:
            blocked = out.path & out.amp_vars()
            out.path -= blocked
            before = (frozenset(out.path), len(out.trace))
            _reduce_round(out)
            stepped = len(out.trace) != before[1]
            out.path |= blocked
            if stepped:
                last = out.trace[-1]
                for sub in ("hh_pivot", "hh_gen"):
                    if last.rule == sub and last.pivot is not None and last.value is not None:
                        out.amps = [
                            (a, f2_substitute(f, last.pivot, last.value)) for a, f in out.amps
                        ]
                _fold_amps(out)
                continue
            if _try_average(out):
                continue
            break
        if not _reduce_round(out):
            break
    return out


def _reduce_round(out: PathSum) -> bool:
    """One rule application of reduce's balanced rules, or False."""
    if not out.path:
        return False
    occ = _occurrences(out.phase)
    return (
        _try_elim(out, occ)
        or _try_hh_pivot(out, occ)
        or _try_hh_gen(out, occ)
        or _try_omega(out, occ)
        or _try_dyadic(out, occ)
        or _try_rem(out, occ)
        or _try_hh_constraint(out, occ)
        or _try_hh_nl(out, occ)
    )


# -- canonical form ----------------------------------------------------------------------


@dataclass(frozen=True)
class Canon:
    """A reduced path sum with no path variable: ``const * exp(2 pi i phase) * [rows]``. Each
    row sets a pivot boundary variable to an F_2 polynomial (monomials, XOR) of the free
    variables; the phase is over the free variables; ``support``, when not empty, is the
    indicator (an F_2 polynomial) the free variables must satisfy."""

    zero: bool
    rows: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]
    phase: tuple[tuple[tuple[int, ...], Fraction], ...]
    const: Cyclo
    boundary: tuple[int, ...]
    support: tuple[tuple[int, ...], ...] = ()
    amps: tuple[tuple[tuple[tuple[int, ...], ...], Cyclo], ...] = ()

    def same(self, other: Canon) -> bool:
        """Whether both forms carry the same data, so denote one function."""
        if self.zero or other.zero:
            return self.zero and other.zero
        return (
            self.boundary == other.boundary
            and self.rows == other.rows
            and self.support == other.support
            and len(self.amps) == len(other.amps)
            and all(
                a[0] == b[0] and a[1] == b[1] for a, b in zip(self.amps, other.amps, strict=True)
            )
            and self.phase == other.phase
            and self.const == other.const
        )

    def value(self, values: Mapping[int, int]) -> Cyclo:
        """The entry at a full boundary assignment."""
        if self.zero:
            return Cyclo.rational(0)
        if self.support and _f2_eval(self.support, values) != 1:
            return Cyclo.rational(0)
        scale = Cyclo.rational(1)
        for expo, base in self.amps:
            if _f2_eval(expo, values):
                scale = scale * base
        for pivot, rest in self.rows:
            if values[pivot] != _f2_eval(rest, values):
                return Cyclo.rational(0)
        turns = sum(
            (c for m, c in self.phase if all(values[v] for v in m)),
            Fraction(0),
        )
        return scale * self.const * Cyclo.root(turns)

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


def _bmul(a: set[int], b: set[int]) -> set[int]:
    """The product of two F_2 polynomials over bitmask monomials."""
    out: set[int] = set()
    for x in a:
        for y in b:
            k = x | y
            if k in out:
                out.remove(k)
            else:
                out.add(k)
    return out


def _bsub(p: set[int], bit: int, value: set[int]) -> set[int]:
    """``p`` with the variable at ``bit`` replaced by ``value``, over bitmask monomials."""
    out: set[int] = set()
    for x in p:
        for k in ((x & ~bit) | y for y in value) if x & bit else (x,):
            if k in out:
                out.remove(k)
            else:
                out.add(k)
    return out


def boolean_eliminate(
    constraints: Iterable[F2Poly], boundary: Sequence[int]
) -> tuple[list[tuple[int, F2Poly]], F2Poly]:
    """Rows and residual indicator of nonlinear F_2 constraints: the support indicator g =
    Prod (1 + c) (the Boolean ideal of the constraints is the principal ideal of 1 + g); each
    variable v ascending with g = v A + B and B (A + B) = 0 is determined, v = A + B on the
    support, and is substituted out of g and the rows. An empty indicator is the empty
    support. Monomials are bitmasks internally."""
    cons = [c for c in constraints if c]
    names = sorted({v for c in cons for m in c for v in m} | set(boundary))
    bit = {v: 1 << i for i, v in enumerate(names)}

    def enc(p: F2Poly) -> set[int]:
        out: set[int] = set()
        for m in p:
            k = sum(bit[v] for v in m)
            if k in out:
                out.remove(k)
            else:
                out.add(k)
        return out

    def dec(p: set[int]) -> F2Poly:
        return frozenset(frozenset(v for v in names if k & bit[v]) for k in p)

    g: set[int] = {0}
    for c in cons:
        g = _bmul(g, {0} ^ enc(c))
        if not g:
            return [], frozenset()
    rows: list[tuple[int, set[int]]] = []
    for v in sorted(boundary):
        b_v = bit[v]
        a = {k & ~b_v for k in g if k & b_v}
        if not a:
            continue
        b = {k for k in g if not k & b_v}
        value = a ^ b
        if _bmul(b, value):
            continue
        g = _bsub(g, b_v, value)
        rows = [(q, _bsub(r, b_v, value)) for q, r in rows]
        rows.append((v, value))
    return sorted(((q, dec(r)) for q, r in rows), key=lambda r: r[0]), dec(g)


def _zero_canon(boundary: tuple[int, ...]) -> Canon:
    return Canon(True, (), (), Cyclo.rational(0), boundary)


def canonical(ps: PathSum) -> Canon | None:
    """The canonical form of a reduced path sum; None when a path variable remains. Constraints
    without a degree-one pivot go through boolean_eliminate."""
    if ps.is_zero:
        return _zero_canon(ps.boundary)
    if ps.path:
        return None
    if ps.amps:
        ps = ps.copy()
        _fold_amps(ps)
        if ps.is_zero:
            return _zero_canon(ps.boundary)
    rows = eliminate(ps.constraints, ps.boundary)
    support: F2Poly = ONE
    if rows is None:
        rows, support = boolean_eliminate(ps.constraints, ps.boundary)
        if not support:
            return _zero_canon(ps.boundary)
    if rows and rows[0] is None:
        return _zero_canon(ps.boundary)
    pivots = cast(list[tuple[int, F2Poly]], rows)

    phase = ps.phase.copy()
    amps = list(ps.amps)
    for pivot, rest in pivots:
        phase.substitute(pivot, rest)
        amps = [(a, f2_substitute(f, pivot, rest)) for a, f in amps]
    const = with_sqrt2(ps.const, ps.sqrt2)
    merged: dict[F2Poly, Cyclo] = {}
    for a, f in amps:
        if not f:
            continue
        if f == ONE:
            const = const * a
            continue
        terms = [(e, q) for e, q in a.c.items() if q]
        if len(terms) == 1 and abs(terms[0][1]) == 1:
            ((e, q),) = terms
            turns = (Fraction(e, a.n) + (Fraction(1, 2) if q < 0 else 0)) % 1
            for mono, coeff in _lift_terms(sorted(f, key=_mono_key), turns):
                phase.add(mono, coeff)
            continue
        merged[f] = merged[f] * a if f in merged else a
    c0 = phase.terms.pop(EMPTY, Fraction(0))
    if c0:
        const = const * Cyclo.root(c0)
    if const.is_zero:
        return _zero_canon(ps.boundary)
    amp_key = tuple(
        sorted(
            ((tuple(sorted(tuple(sorted(m)) for m in f)), a) for f, a in merged.items()),
            key=lambda t: t[0],
        )
    )
    return Canon(
        False,
        tuple((p, tuple(sorted(tuple(sorted(m)) for m in rest))) for p, rest in pivots),
        tuple(sorted(((tuple(sorted(m)), c) for m, c in phase.terms.items()))),
        const,
        ps.boundary,
        () if support == ONE else tuple(sorted(tuple(sorted(m)) for m in support)),
        amp_key,
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
    if ps.amps:
        return None
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
    at.amps = [(a, f2_assign(f, values)) for a, f in at.amps]
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
    """``const * [constraints = 0] * Sum_{path in Z_p^m} omega_p^phase * Prod f(arg)``, the
    constraints, the phase and each atom's argument polynomials over F_p, each atom table ``f``
    its p values in Q(zeta)."""

    p: int
    boundary: tuple[int, ...]
    path: set[int]
    phase: ModPoly
    const: Cyclo
    constraints: list[ModPoly] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)
    atoms: list[tuple[ModPoly, tuple[Cyclo, ...]]] = field(default_factory=list)
    katoms: list[tuple[tuple[ModPoly, ...], tuple[Cyclo, ...]]] = field(default_factory=list)

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
            [(a.copy(), t) for a, t in self.atoms],
            [(tuple(c.copy() for c in a), t) for a, t in self.katoms],
        )

    def atom_vars(self) -> set[int]:
        """Every variable some atom argument carries."""
        return {v for a, _t in self.atoms for v in a.variables()} | {
            v for args, _t in self.katoms for a in args for v in a.variables()
        }

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


def _q_fold_atoms(ps: QuditPathSum) -> None:
    """Atoms whose argument is a constant multiplied into the constant."""
    keep = []
    for arg, table in ps.atoms:
        if arg.variables():
            keep.append((arg, table))
        else:
            ps.const = ps.const * table[arg.terms.get(ONE_Q, 0) % ps.p]
    ps.atoms = keep
    p = ps.p
    kkeep = []
    for args, table in ps.katoms:
        live = [j for j, a in enumerate(args) if a.variables()]
        fixed = {j: a.terms.get(ONE_Q, 0) % p for j, a in enumerate(args) if j not in live}
        if len(live) >= 2:
            kkeep.append((args, table))
            continue
        if not live:
            ps.const = ps.const * table[sum(fixed[j] * p**j for j in fixed)]
            continue
        (j0,) = live
        base = sum(fixed[j] * p**j for j in fixed)
        ps.atoms.append((args[j0], tuple(table[base + t * p**j0] for t in range(p))))
    ps.katoms = kkeep


def _q_try_elim(ps: QuditPathSum) -> bool:
    used = ps.phase.variables() | ps.atom_vars()
    for y in sorted(ps.path):
        if y not in used:
            ps.path.discard(y)
            ps.const = ps.const.scale(ps.p)
            ps.trace.append(_q_step("q_elim", y, None, ModPoly(ps.p)))
            return True
    return False


def _q_try_hh_pivot(ps: QuditPathSum) -> bool:
    in_atoms = ps.atom_vars()
    for y in sorted(ps.path - in_atoms):
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
        ps.atoms = [(a.substitute(z, value), t) for a, t in ps.atoms]
        ps.katoms = [(tuple(c.substitute(z, value) for c in a), t) for a, t in ps.katoms]
        _q_fold_atoms(ps)
        ps.path.discard(y)
        ps.path.discard(z)
        ps.const = ps.const.scale(ps.p)
        ps.trace.append(_q_step("q_hh_pivot", y, z, value))
        return True
    return False


def _q_try_gauss(ps: QuditPathSum) -> bool:
    for y in sorted(ps.path - ps.atom_vars()):
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
    for y in sorted(ps.path - ps.atom_vars()):
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


def dft_table(table: Sequence[Cyclo], p: int) -> tuple[Cyclo, ...]:
    """``t -> Sum_u zeta_p^(u t) f(u)``."""
    return tuple(
        sum((Cyclo.root(Fraction(u * t, p)) * table[u] for u in range(p)), Cyclo.rational(0))
        for t in range(p)
    )


def _q_try_fourier(ps: QuditPathSum) -> bool:
    """[Fourier]: y of degree at most one in the phase, y Q, and in one atom f(a y + c), a a
    unit constant: Sum_y omega^(y Q) f(a y + c) = omega^(-Q c / a) f^(Q / a), f^ the DFT of f."""
    cons_vars = {v for c in ps.constraints for v in c.variables()}
    kvars = {v for args, _t in ps.katoms for a in args for v in a.variables()}
    for y in sorted(ps.path - cons_vars - kvars):
        if ps.phase.degree_in(y) > 1:
            continue
        hits = [i for i, (a, _t) in enumerate(ps.atoms) if y in a.variables()]
        if len(hits) != 1:
            continue
        arg, table = ps.atoms[hits[0]]
        if arg.degree_in(y) != 1:
            continue
        lin = arg.coefficient(y, 1)
        if set(lin.terms) != {ONE_Q}:
            continue
        inv = pow(lin.terms[ONE_Q], -1, ps.p)
        q = ps.phase.coefficient(y, 1)
        c = arg.without(y)
        ps.phase = ps.phase.without(y) + (q * c).scale(-inv)
        ps.atoms = [t for i, t in enumerate(ps.atoms) if i != hits[0]]
        ps.atoms.append((q.scale(inv), dft_table(table, ps.p)))
        ps.path.discard(y)
        _q_fold_atoms(ps)
        ps.trace.append(_q_step("q_fourier", y, lin.terms[ONE_Q], arg))
        return True
    return False


def _q_try_hh_atom(ps: QuditPathSum) -> bool:
    """[HH] into an atom: y of degree one, y Q, Q with a path variable and no pivot, y in no
    atom: Sum_y omega^(y Q) = p delta(Q), the atom (Q, delta)."""
    p = ps.p
    delta = tuple(Cyclo.rational(1 if t == 0 else 0) for t in range(p))
    for y in sorted(ps.path - ps.atom_vars()):
        if ps.phase.degree_in(y) != 1:
            continue
        q = ps.phase.coefficient(y, 1)
        if not q.variables() & ps.path:
            continue
        ps.phase = ps.phase.without(y)
        ps.atoms.append((q, delta))
        ps.path.discard(y)
        ps.const = ps.const.scale(p)
        ps.trace.append(_q_step("q_hh_atom", y, None, q))
        return True
    return False


def _q_try_convolve(ps: QuditPathSum) -> bool:
    """[Convolve]: x in no constraint, in the phase only through terms in x alone, in atoms on
    x alone (product A, the phase terms absorbed) and at most one atom g(a x + M), a a unit
    constant: Sum_x A(x) g(a x + M) = H(M), H(m) = Sum_v A(v) g(a v + m)."""
    p = ps.p
    cons_vars = {v for c in ps.constraints for v in c.variables()}
    kvars = {v for args, _t in ps.katoms for a in args for v in a.variables()}
    for x in sorted(ps.path - cons_vars - kvars):
        mixed = [m for m in ps.phase.terms if any(v == x for v, _e in m) and len(m) > 1]
        if mixed:
            continue
        alone = [i for i, (a, _t) in enumerate(ps.atoms) if a.variables() == {x}]
        other = [i for i, (a, _t) in enumerate(ps.atoms) if x in a.variables() and i not in alone]
        if not alone and not other:
            continue
        if len(other) > 1:
            continue
        table = [Cyclo.rational(1)] * p
        for i in alone:
            arg, t = ps.atoms[i]
            table = [table[v] * t[arg.value({x: v})] for v in range(p)]
        own = ModPoly(p, {m: c for m, c in ps.phase.terms.items() if any(u == x for u, _e in m)})
        table = [table[v] * Cyclo.root(Fraction(own.value({x: v}), p)) for v in range(p)]
        if other:
            arg, g = ps.atoms[other[0]]
            if arg.degree_in(x) != 1:
                continue
            lin = arg.coefficient(x, 1)
            if set(lin.terms) != {ONE_Q}:
                continue
            a = lin.terms[ONE_Q]
            rest = arg.without(x)
            h = tuple(
                sum((table[v] * g[(a * v + m) % p] for v in range(p)), Cyclo.rational(0))
                for m in range(p)
            )
            new_atoms = [(rest, h)]
        else:
            ps.const = ps.const * sum(table, Cyclo.rational(0))
            new_atoms = []
        ps.phase = ps.phase.without(x)
        drop = set(alone) | set(other)
        ps.atoms = [t for i, t in enumerate(ps.atoms) if i not in drop] + new_atoms
        ps.path.discard(x)
        _q_fold_atoms(ps)
        ps.trace.append(_q_step("q_convolve", x, None, ModPoly(p)))
        return True
    return False


def _kslot(ps: QuditPathSum, y: int) -> tuple[int, int, int] | None:
    """The one k-ary atom and component carrying ``y``, linearly with a unit constant
    coefficient, with that coefficient; None otherwise."""
    found = [
        (i, j)
        for i, (args, _t) in enumerate(ps.katoms)
        for j, a in enumerate(args)
        if y in a.variables()
    ]
    if len(found) != 1:
        return None
    i, j = found[0]
    arg = ps.katoms[i][0][j]
    if arg.degree_in(y) != 1:
        return None
    lin = arg.coefficient(y, 1)
    if set(lin.terms) != {ONE_Q}:
        return None
    return i, j, lin.terms[ONE_Q]


def _axis_map(
    table: Sequence[Cyclo],
    k: int,
    j: int,
    p: int,
    fn: Callable[[Callable[[int], Cyclo], int], Cyclo],
) -> tuple[Cyclo, ...]:
    """A new k-ary table whose entry at t is ``fn(g, t_j)``, g reading the old table along
    axis j with the other coordinates of t."""
    out = []
    for idx in range(p**k):
        tj = idx // p**j % p
        base = idx - tj * p**j

        def read(u: int, base: int = base) -> Cyclo:
            return cast(Cyclo, table[base + (u % p) * p**j])

        out.append(fn(read, tj))
    return tuple(out)


def _q_try_kfourier(ps: QuditPathSum) -> bool:
    """[Fourier] along one component of a k-ary atom: y of degree at most one in the phase,
    in no constraint or unary atom, in one component a y + c."""
    p = ps.p
    cons_vars = {v for c in ps.constraints for v in c.variables()}
    unary = {v for a, _t in ps.atoms for v in a.variables()}
    for y in sorted(ps.path - cons_vars - unary):
        if ps.phase.degree_in(y) > 1:
            continue
        slot = _kslot(ps, y)
        if slot is None:
            continue
        i, j, a = slot
        args, table = ps.katoms[i]
        if any(y in b.variables() for jj, b in enumerate(args) if jj != j):
            continue
        inv = pow(a, -1, p)
        q = ps.phase.coefficient(y, 1)
        c = args[j].without(y)
        ps.phase = ps.phase.without(y) + (q * c).scale(-inv)
        new_args = tuple(q.scale(inv) if jj == j else b for jj, b in enumerate(args))

        def dft(g: Callable[[int], Cyclo], t: int) -> Cyclo:
            return sum((Cyclo.root(Fraction(u * t, p)) * g(u) for u in range(p)), Cyclo.rational(0))

        new_table = _axis_map(table, len(args), j, p, dft)
        ps.katoms[i] = (new_args, new_table)
        ps.path.discard(y)
        _q_fold_atoms(ps)
        ps.trace.append(_q_step("q_kfourier", y, a, args[j]))
        return True
    return False


def _q_try_kconvolve(ps: QuditPathSum) -> bool:
    """[Convolve] along one component of a k-ary atom: x in no constraint, in the phase only
    through terms in x alone, in unary atoms on x alone and in one component a x + M."""
    p = ps.p
    cons_vars = {v for c in ps.constraints for v in c.variables()}
    for x in sorted(ps.path - cons_vars):
        if any(any(v == x for v, _e in m) and len(m) > 1 for m in ps.phase.terms):
            continue
        if any(x in a.variables() and a.variables() != {x} for a, _t in ps.atoms):
            continue
        slot = _kslot(ps, x)
        if slot is None:
            continue
        i, j, a = slot
        args, g = ps.katoms[i]
        if any(x in b.variables() for jj, b in enumerate(args) if jj != j):
            continue
        alone = [n for n, (b, _t) in enumerate(ps.atoms) if b.variables() == {x}]
        table = [Cyclo.rational(1)] * p
        for n in alone:
            b, t = ps.atoms[n]
            table = [table[v] * t[b.value({x: v})] for v in range(p)]
        own = ModPoly(p, {m: c for m, c in ps.phase.terms.items() if any(u == x for u, _e in m)})
        table = [table[v] * Cyclo.root(Fraction(own.value({x: v}), p)) for v in range(p)]
        rest = args[j].without(x)
        weights = list(table)

        def conv(
            read: Callable[[int], Cyclo], m: int, weights: list[Cyclo] = weights, a: int = a
        ) -> Cyclo:
            return sum((weights[v] * read(a * v + m) for v in range(p)), Cyclo.rational(0))

        new_table = _axis_map(g, len(args), j, p, conv)
        ps.phase = ps.phase.without(x)
        ps.atoms = [t for n, t in enumerate(ps.atoms) if n not in alone]
        ps.katoms[i] = (tuple(rest if jj == j else b for jj, b in enumerate(args)), new_table)
        ps.path.discard(x)
        _q_fold_atoms(ps)
        ps.trace.append(_q_step("q_kconvolve", x, a, ModPoly(p)))
        return True
    return False


def reduce_qudit(ps: QuditPathSum) -> QuditPathSum:
    """``ps`` with [Elim], [HH] on a path pivot (z := -Q'/a), [Gauss], [HH] on a boundary
    constraint, [Fourier], [HH] into an atom and [Convolve] applied to a fixpoint over Z_p, in
    that order, the lowest variable first; variables in an atom take only the last three."""
    out = ps.copy()
    _q_fold_atoms(out)
    while out.path and not out.is_zero:
        if not (
            _q_try_elim(out)
            or _q_try_hh_pivot(out)
            or _q_try_gauss(out)
            or _q_try_hh_constraint(out)
            or _q_try_fourier(out)
            or _q_try_hh_atom(out)
            or _q_try_convolve(out)
            or _q_try_kfourier(out)
            or _q_try_kconvolve(out)
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
    atoms: tuple[tuple[tuple[tuple[QMono, int], ...], tuple[Cyclo, ...]], ...] = ()
    katoms: tuple[tuple[tuple[tuple[tuple[QMono, int], ...], ...], tuple[Cyclo, ...]], ...] = ()

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
            and len(self.atoms) == len(other.atoms)
            and all(
                a[0] == b[0] and all(x == y for x, y in zip(a[1], b[1], strict=True))
                for a, b in zip(self.atoms, other.atoms, strict=True)
            )
            and len(self.katoms) == len(other.katoms)
            and all(
                a[0] == b[0] and all(x == y for x, y in zip(a[1], b[1], strict=True))
                for a, b in zip(self.katoms, other.katoms, strict=True)
            )
        )

    def value(self, values: Mapping[int, int]) -> Cyclo:
        """The entry at a full boundary assignment."""
        if self.zero:
            return Cyclo.rational(0)
        for pivot, rest in self.rows:
            if values[pivot] % self.p != ModPoly(self.p, dict(rest)).value(values):
                return Cyclo.rational(0)
        e = ModPoly(self.p, dict(self.phase)).value(values)
        out = self.const * Cyclo.root(Fraction(e, self.p))
        for arg, table in self.atoms:
            out = out * table[ModPoly(self.p, dict(arg)).value(values)]
        for args, table in self.katoms:
            idx = sum(
                ModPoly(self.p, dict(a)).value(values) * self.p**j for j, a in enumerate(args)
            )
            out = out * table[idx]
        return out

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
    hold = ps.copy()
    for pivot, rest in pivots:
        phase = phase.substitute(pivot, rest)
        hold.atoms = [(a.substitute(pivot, rest), t) for a, t in hold.atoms]
        hold.katoms = [(tuple(c.substitute(pivot, rest) for c in a), t) for a, t in hold.katoms]
    hold.const = ps.const
    _q_fold_atoms(hold)
    phase, const, canon_atoms = _canon_atoms(hold.atoms, phase, hold.const, ps.p)
    phase, const, canon_katoms = _canon_katoms(hold.katoms, phase, const, ps.p)
    c0 = phase.terms.pop(ONE_Q, 0)
    if c0:
        const = const * Cyclo.root(Fraction(c0, ps.p))
    if const.is_zero or any(all(v.is_zero for v in t) for _a, t in (*canon_atoms, *canon_katoms)):
        return zero
    return QuditCanon(
        ps.p,
        False,
        tuple((q, r.key()) for q, r in pivots),
        phase.key(),
        const,
        ps.boundary,
        canon_atoms,
        canon_katoms,
    )


def _canon_katoms(
    katoms: Sequence[tuple[tuple[ModPoly, ...], tuple[Cyclo, ...]]],
    phase: ModPoly,
    const: Cyclo,
    p: int,
) -> tuple[
    ModPoly,
    Cyclo,
    tuple[tuple[tuple[tuple[tuple[QMono, int], ...], ...], tuple[Cyclo, ...]], ...],
]:
    """k-ary atoms with each component monic and constant-free (axes permuted), components
    sorted, equal argument tuples merged, a root-of-unity table written back as phase and
    constant, and every other table scaled to 1 at its first nonzero entry."""
    merged: dict[tuple[tuple[tuple[QMono, int], ...], ...], list[Cyclo]] = {}
    for args, table0 in katoms:
        k = len(args)
        table = list(table0)
        keys = []
        for j, arg in enumerate(args):
            lead = min(m for m in arg.terms if m)
            a = arg.terms[lead]
            c0 = arg.terms.get(ONE_Q, 0)
            monic = (arg + ModPoly.const(p, -c0)).scale(pow(a, -1, p))

            def shift(read: Callable[[int], Cyclo], t: int, a: int = a, c0: int = c0) -> Cyclo:
                return read(a * t + c0)

            table = list(_axis_map(table, k, j, p, shift))
            keys.append(monic.key())
        order = sorted(range(k), key=lambda j: keys[j])
        new = []
        for idx in range(p**k):
            coords = [idx // p**n % p for n in range(k)]
            old = sum(coords[n] * p ** order[n] for n in range(k))
            new.append(table[old])
        key = tuple(keys[j] for j in order)
        if key in merged:
            merged[key] = [x * y for x, y in zip(merged[key], new, strict=True)]
        else:
            merged[key] = new
    out = []
    for key in sorted(merged):
        tab = merged[key]
        k = len(key)
        if all(not v.is_zero for v in tab):
            inv0 = _inverse_or_none(tab[0])
            exps = [_root_exponent(v * inv0, p) for v in tab]
            if all(e is not None for e in exps):
                const = const * tab[0]
                e_poly = ModPoly(p)
                for idx, e in enumerate(exps):
                    if not e:
                        continue
                    term = ModPoly.const(p, int(e))
                    for n in range(k):
                        t = idx // p**n % p
                        holder = -1 - n
                        shifted = ModPoly(
                            p, {((holder, 1),): 1, (): -t % p} if t else {((holder, 1),): 1}
                        )
                        term = term * (ModPoly.const(p, 1) + shifted.power(p - 1).scale(-1))
                    e_poly = e_poly + term
                for n in range(k):
                    e_poly = e_poly.substitute(-1 - n, ModPoly(p, dict(key[n])))
                phase = phase + e_poly
                continue
        if all(v.is_zero for v in tab):
            const = Cyclo.rational(0)
            continue
        lead_value = next(v for v in tab if not v.is_zero)
        inv = _inverse_or_none(lead_value)
        const = const * lead_value
        out.append((key, tuple(v * inv for v in tab)))
    return phase, const, tuple(out)


def _interp_mod(values: Sequence[int], var: int, p: int) -> ModPoly:
    """The F_p polynomial in ``var`` taking ``values[c]`` at c (Lagrange indicators)."""
    out = ModPoly(p)
    for c, e in enumerate(values):
        if e % p:
            shifted = ModPoly(p, {((var, 1),): 1, (): -c % p} if c else {((var, 1),): 1})
            out = out + (ModPoly.const(p, 1) + shifted.power(p - 1).scale(-1)).scale(e)
    return out


def _root_exponent(ratio: Cyclo, p: int) -> int | None:
    """The e with ratio = zeta_p^e, or None."""
    return next((e for e in range(p) if ratio == Cyclo.root(Fraction(e, p))), None)


def _canon_atoms(
    atoms: Sequence[tuple[ModPoly, tuple[Cyclo, ...]]], phase: ModPoly, const: Cyclo, p: int
) -> tuple[ModPoly, Cyclo, tuple[tuple[tuple[tuple[QMono, int], ...], tuple[Cyclo, ...]], ...]]:
    """Atoms with arguments made monic and constant-free (tables permuted), equal arguments
    merged, a single-variable atom absorbing the phase terms in that variable alone, a table
    that is a constant times p-th roots of unity written back as phase and constant, and every
    other table scaled to 1 at its first nonzero entry."""
    merged: dict[tuple[tuple[QMono, int], ...], list[Cyclo]] = {}
    for arg, table in atoms:
        if not arg.variables():
            const = const * table[arg.terms.get(ONE_Q, 0) % p]
            continue
        lead = min(m for m in arg.terms if m)
        a = arg.terms[lead]
        c0 = arg.terms.get(ONE_Q, 0)
        monic = (arg + ModPoly.const(p, -c0)).scale(pow(a, -1, p))
        permuted = [table[(a * t + c0) % p] for t in range(p)]
        key = monic.key()
        if key in merged:
            merged[key] = [x * y for x, y in zip(merged[key], permuted, strict=True)]
        else:
            merged[key] = permuted
    phase = phase.copy()
    out: list[tuple[tuple[tuple[QMono, int], ...], tuple[Cyclo, ...]]] = []
    for key in sorted(merged):
        tab = merged[key]
        if len(key) == 1 and key[0][1] == 1 and len(key[0][0]) == 1 and key[0][0][0][1] == 1:
            x = key[0][0][0][0]
            alone = {m: c for m, c in phase.terms.items() if m and all(v == x for v, _e in m)}
            for m in alone:
                del phase.terms[m]
            poly = ModPoly(p, alone)
            tab = [tab[t] * Cyclo.root(Fraction(poly.value({x: t}), p)) for t in range(p)]
        if all(not v.is_zero for v in tab):
            exps = [_root_exponent(v * _inverse_or_none(tab[0]), p) for v in tab]
            if all(e is not None for e in exps):
                const = const * tab[0]
                arg = ModPoly(p, dict(key))
                placeholder = -1
                e_poly = _interp_mod([int(e) for e in exps if e is not None], placeholder, p)
                phase = phase + e_poly.substitute(placeholder, arg)
                continue
        if all(v.is_zero for v in tab):
            const = Cyclo.rational(0)
            continue
        lead_value = next(v for v in tab if not v.is_zero)
        inv = _inverse_or_none(lead_value)
        const = const * lead_value
        out.append((key, tuple(v * inv for v in tab)))
    return phase, const, tuple(out)


def _inverse_or_none(c: Cyclo) -> Cyclo:
    """1 / c for a nonzero cyclotomic c."""
    inv = _cyclo_inverse(c)
    if inv is None:
        raise PathSumError(f"{c} has no inverse")
    return inv


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
        atom_vars = sorted(
            {v for arg, _t in (*x.atoms, *y.atoms) for m, _c in arg for v, _e in m}
            | {
                v
                for args, _t in (*x.katoms, *y.katoms)
                for arg in args
                for m, _c in arg
                for v, _e in m
            }
        )
        for v in atom_vars:
            points.extend(x.complete({v: t}) for t in range(x.p))
        for args, _t in (*x.katoms, *y.katoms):
            kv = sorted({v for arg in args for m, _c in arg for v, _e in m})
            if x.p ** len(kv) <= 625:
                for vals in product(range(x.p), repeat=len(kv)):
                    points.append(x.complete(dict(zip(kv, vals, strict=True))))
    return points


def entry_at_qudit(ps: QuditPathSum, values: Mapping[int, int]) -> Cyclo:
    """The entry at one boundary assignment: the boundary substituted, then reduced; raises
    :class:`PathSumUnsupported` when a path variable remains."""
    at = ps.copy()
    at.phase = at.phase.assign(values)
    at.constraints = [c.assign(values) for c in at.constraints]
    at.atoms = [(a.assign(values), t) for a, t in at.atoms]
    at.katoms = [(tuple(c.assign(values) for c in a), t) for a, t in at.katoms]
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
    "boolean_eliminate",
    "canonical",
    "canonical_qudit",
    "constant",
    "cyclotomic",
    "dft_table",
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
