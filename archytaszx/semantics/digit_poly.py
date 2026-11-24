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

"""Phase tables as polynomials in the mixed-radix digits of their index.

A table of numerators ``N(k)`` over ``den`` is recognised when ``N(k) = F(x_1, ..., x_m) mod den``
at every index ``k < dim``, with ``x_i`` the digits of ``k`` in bases ``b_1, ..., b_m`` (most
significant first) and ``F`` a polynomial with integer coefficients over ``den2``. The fit takes
Newton coefficients in the binomial basis from a grid of degree at most `MAX_DEGREE` per digit; the
check compares every stored entry exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

MAX_DEGREE = 3
"""Highest degree per digit tried."""
MAX_DIGITS = 4
"""Most digits in a split."""
MAX_POLY_TERMS = 64
"""Most monomials in an accepted formula."""


@dataclass(frozen=True)
class DigitPoly:
    """``F(k) = sum c * prod x_i^e_i`` over the digits ``x_i`` of ``k`` in ``bases``; the phase at
    ``k`` is ``(F(k) mod den) / den`` turns."""

    bases: tuple[int, ...]
    terms: tuple[tuple[int, tuple[int, ...]], ...]
    den: int

    def digits(self, k: int) -> list[int]:
        out = []
        for b in reversed(self.bases):
            k, r = divmod(k, b)
            out.append(r)
        return out[::-1]

    def value(self, k: int) -> int:
        """``F(k) mod den``."""
        xs = self.digits(k)
        total = 0
        for c, es in self.terms:
            t = c
            for x, e in zip(xs, es, strict=True):
                t *= x**e
            total += t
        return total % self.den

    def digit_text(self, i: int, var: str = "k") -> str:
        """Digit ``i`` of ``var`` as a Lean integer."""
        below = math.prod(self.bases[i + 1 :])
        if len(self.bases) == 1:
            body = var
        elif i == 0:
            body = f"{var} / {below}"
        elif i == len(self.bases) - 1:
            body = f"{var} % {self.bases[i]}"
        else:
            body = f"{var} / {below} % {self.bases[i]}"
        return f"(({body} : ℕ) : ℤ)"

    def lean(self, var: str = "k") -> str:
        """``F(var) % den`` as a Lean integer expression."""
        parts = []
        for c, es in self.terms:
            fs = [f"({c} : ℤ)" if c < 0 else str(c)]
            for i, e in enumerate(es):
                if e:
                    d = self.digit_text(i, var)
                    fs.append(d if e == 1 else f"{d} ^ {e}")
            parts.append(" * ".join(fs))
        body = " + ".join(parts) if parts else "0"
        return f"(({body}) % {self.den})"


def _splits(n: int, m: int) -> Iterator[tuple[int, ...]]:
    """Ordered factorizations of ``n`` into ``m`` factors above 1."""
    if m == 1:
        if n > 1:
            yield (n,)
        return
    for f in sp.divisors(n)[1:-1]:
        for rest in _splits(n // f, m - 1):
            yield (int(f), *rest)


def _index(bases: Sequence[int], xs: Sequence[int]) -> int:
    k = 0
    for b, x in zip(bases, xs, strict=True):
        k = k * b + x
    return k


def _fit(
    vals: Sequence[int], den: int, bases: tuple[int, ...], deg: int
) -> dict[tuple[int, ...], int] | None:
    """Binomial-basis coefficients mod ``den`` from the grid of degree ``deg`` per digit, or None
    when the grid is the whole table."""
    ds = [min(deg, b - 1) for b in bases]
    if math.prod(d + 1 for d in ds) >= len(vals):
        return None
    shape = tuple(d + 1 for d in ds)
    g = np.zeros(shape, dtype=object)
    for xs in product(*(range(s) for s in shape)):
        g[xs] = vals[_index(bases, xs)]
    for ax, d in enumerate(ds):
        for j in range(1, d + 1):
            for i in range(d, j - 1, -1):
                hi: list[int | slice] = [slice(None)] * len(ds)
                lo: list[int | slice] = [slice(None)] * len(ds)
                hi[ax], lo[ax] = i, i - 1
                g[tuple(hi)] = g[tuple(hi)] - g[tuple(lo)]
    out = {}
    for xs in product(*(range(s) for s in shape)):
        c = int(g[xs]) % den
        if c:
            out[xs] = c if 2 * c <= den else c - den
    return out


PROBES = 64
"""Entries compared before the full check."""


def _probe(
    vals: Sequence[int], den: int, bases: tuple[int, ...], coefs: dict[tuple[int, ...], int]
) -> bool:
    """The binomial-basis polynomial mod ``den`` matches ``vals`` at `PROBES` spread indices."""
    n = len(vals)
    for j in range(PROBES):
        k = (j * (n - 1)) // (PROBES - 1)
        xs, r = [], k
        for b in reversed(bases):
            r, x = divmod(r, b)
            xs.append(x)
        xs.reverse()
        total = 0
        for alpha, c in coefs.items():
            t = c
            for x, a in zip(xs, alpha, strict=True):
                if a:
                    t *= math.comb(x, a)
            total += t
        if total % den != vals[k]:
            return False
    return True


def _binom_col(x: np.ndarray, a: int) -> np.ndarray:
    out = np.ones_like(x)
    for i in range(a):
        out = out * (x - i)
    return out // math.factorial(a)


def _check(
    vals: np.ndarray, den: int, bases: tuple[int, ...], coefs: dict[tuple[int, ...], int]
) -> bool:
    """Every entry of ``vals`` equals the binomial-basis polynomial mod ``den``."""
    n = len(vals)
    big = den > 1 << 30 or max(bases) ** MAX_DEGREE > 1 << 62
    k = np.arange(n, dtype=object if big or vals.dtype == object else np.int64)
    xs, rest = [], k
    for b in reversed(bases):
        xs.append(rest % b)
        rest = rest // b
    xs = xs[::-1]
    total = np.zeros(n, dtype=k.dtype)
    cols: dict[tuple[int, int], np.ndarray] = {}
    for alpha, c in coefs.items():
        t = np.full(n, c % den, dtype=k.dtype)
        for i, a in enumerate(alpha):
            if a:
                if (i, a) not in cols:
                    cols[(i, a)] = _binom_col(xs[i], a) % den
                t = (t * cols[(i, a)]) % den
        total = (total + t) % den
    return bool(np.array_equal(total, vals))


def _monomials(
    bases: tuple[int, ...], coefs: dict[tuple[int, ...], int], den: int
) -> DigitPoly | None:
    xs = sp.symbols(f"x0:{len(bases)}")
    expr = sp.Integer(0)
    for alpha, c in coefs.items():
        t = sp.Integer(c)
        for x, a in zip(xs, alpha, strict=True):
            t *= sp.ff(x, a) / sp.factorial(a)
        expr += t
    poly = sp.Poly(sp.expand(expr), *xs)
    rat = [(Fraction(int(sp.Rational(c).p), int(sp.Rational(c).q)), m) for m, c in poly.terms()]
    lcd = math.lcm(1, *(f.denominator for f, _ in rat))
    terms = tuple(
        (int(f * lcd), tuple(int(e) for e in m)) for f, m in sorted(rat, key=lambda t: t[1])
    )
    if len(terms) > MAX_POLY_TERMS:
        return None
    return DigitPoly(tuple(bases), terms, den * lcd)


def digit_poly(nums: Sequence[int], den: int, dim: int) -> DigitPoly | None:
    """The digit polynomial of the table ``nums`` over ``den`` of dimension ``dim``, fewest digits
    and lowest degree first; None when no split up to `MAX_DIGITS` digits fits or an entry lies
    outside ``[0, den)``."""
    if dim < 2 or len(nums) > dim or any(not 0 <= v < den for v in nums):
        return None
    vals = list(nums) + [0] * (dim - len(nums))
    arr: np.ndarray | None = None
    for m in range(1, MAX_DIGITS + 1):
        for deg in range(1, MAX_DEGREE + 1):
            for bases in _splits(dim, m):
                coefs = _fit(vals, den, bases, deg)
                if coefs is None:
                    continue
                if not _probe(vals, den, bases, coefs):
                    continue
                if arr is None:
                    arr = np.array(vals, dtype=object if den > 1 << 30 else np.int64)
                if _check(arr, den, bases, coefs):
                    return _monomials(bases, coefs, den)
    return None
