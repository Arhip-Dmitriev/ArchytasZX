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
from itertools import combinations
from typing import Any, cast

Mono = frozenset[int]
F2Poly = frozenset[Mono]
EMPTY: Mono = frozenset()


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
    """One rule application of a reduction: its rule, variable, pivot and substitution."""

    rule: str
    var: int
    pivot: int | None = None
    value: F2Poly | None = None


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


def _try_hh_constraint(ps: PathSum, occ: Mapping[int, list[tuple[Mono, Fraction]]]) -> bool:
    for y in sorted(ps.path):
        terms = occ.get(y, [])
        q = _hh_quotient(y, terms)
        if q is None or not q:
            continue
        if any(v in ps.path for m in q for v in m):
            continue
        _drop_terms(ps, terms)
        ps.constraints.append(q)
        ps.path.discard(y)
        ps.sqrt2 += 2
        ps.trace.append(Step("hh_constraint", y, None, q))
        return True
    return False


def reduce(ps: PathSum) -> PathSum:
    """``ps`` with [Elim], [HH] on a path pivot, [omega] and [HH] on a boundary constraint
    applied to a fixpoint, in that order, the lowest variable first."""
    out = ps.copy()
    while out.path and not out.is_zero:
        occ = _occurrences(out.phase)
        if not (
            _try_elim(out, occ)
            or _try_hh_pivot(out, occ)
            or _try_omega(out, occ)
            or _try_hh_constraint(out, occ)
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
    """Gauss-Jordan elimination over F_2 polynomials, the variables ascending: each pivot is a
    variable occurring in a constraint only as a degree-one monomial, substituted out of every
    other constraint and row. Returns (pivot, polynomial) rows, ``[None]`` when a constraint
    is the constant one, and None when a constraint is left without a pivot."""
    work = sorted({c for c in constraints if c}, key=lambda c: sorted(_mono_key(m) for m in c))
    rows: list[tuple[int, F2Poly]] = []
    if any(c == frozenset({EMPTY}) for c in work):
        return [None]
    for v in sorted(boundary):
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


__all__ = [
    "SQRT2",
    "Canon",
    "Cyclo",
    "PathSum",
    "PathSumError",
    "PathSumUnsupported",
    "PhasePoly",
    "Step",
    "canonical",
    "constant",
    "cyclotomic",
    "eliminate",
    "entry_at",
    "reduce",
    "with_sqrt2",
    "witness_points",
]
