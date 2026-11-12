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

"""Whether an F pair differs at a point (BENCHMARKS.txt, sections 4 and 5).

Decided from the F record's kind and edit parameters by exact integer arithmetic:

  beyond_oracle, prime_only, coincident_dims  equal iff the phase rule is an integer at every
                                              k < d
  scalar_small_one, box_boundary              equal iff the scalar factor is 1
  beyond_N                                    unrolled: equal iff n <= N; boxed: equal unless
                                              n = N + 1 mod N + 2
  qubit_only                                  equal iff d <= 2

An F record's ``differs_at`` table (breaks.differs_table: [n, d, differs] by a dense check at
small points) decides its own points first. Off the table, a kind without a rule above is
equal where the table shows it equal along the whole line: at d = 1 when every tabled d = 1
point is equal, at n = 0 when every tabled n = 0 point is, at every point when every tabled
point is (a no-op edit), and beyond the largest tabled n (d) when the last TAIL tabled values
of n (d) are equal at every tabled d (n), and at an n beyond the table where the tabled n line
of its d repeats with a period p (at least PERIODS checks of v(m) = v(m + p)) and the tabled m
= n mod p is equal. Otherwise it differs at every point of its domain,
unless its witness is an interface mismatch between a pure side and a doubled side (one side's
boundary is the other's taken twice): such a witness reads no entry, and the pair counts as
not differing.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

_K, _N, _D = sp.symbols("k n d", integer=True)
TAIL = 3
PERIODS = 3


@functools.cache
def _expr(text: str) -> Any:
    return sp.sympify(text.replace("^", "**"), locals={"k": _K, "n": _N, "d": _D})


@functools.cache
def _ratio(text: str) -> tuple[Any, Any] | None:
    """Integer functions of (n, d) for the numerator and denominator of ``text`` divided by k,
    or None when ``text`` is not k times an expression in n and d."""
    expr = sp.expand(_expr(text))
    coefficient, rest = expr.as_independent(_K)
    body = sp.simplify(rest / _K)
    if coefficient != 0 or _K in body.free_symbols or body.free_symbols - {_N, _D}:
        return None
    num, den = sp.fraction(sp.together(body))
    return sp.lambdify((_N, _D), num, "math"), sp.lambdify((_N, _D), den, "math")


@functools.cache
def _difference(text: str) -> Any:
    """An integer function of (n, d) for ``text`` minus 1, or None when ``text`` has a symbol
    other than n and d."""
    expr = _expr(text) - 1
    if expr.free_symbols - {_N, _D}:
        return None
    return sp.lambdify((_N, _D), expr, "math")


def _phase_trivial(rule: str, n: int | None, d: int) -> bool:
    """Whether the phase rule is an integer at every k < d."""
    if rule.replace(" ", "") == "(k^d-k)/d":
        return _fermat(d)
    ratio = _ratio(rule)
    if ratio is None:
        return False
    num, den = ratio[0](n or 0, d), ratio[1](n or 0, d)
    return bool(den != 0 and num % den == 0)


@functools.cache
def _fermat(d: int) -> bool:
    """Whether k^d = k mod d for every k < d."""
    return all(pow(k, d, d) == k % d for k in range(d))


def _factor_one(factor: str, n: int | None, d: int) -> bool:
    """Whether the scalar factor is 1."""
    difference = _difference(factor)
    return difference is not None and difference(n or 0, d) == 0


_INTERFACE = re.compile(r"left \[(.*?)\]->\[(.*?)\], right \[(.*?)\]->\[(.*?)\]")


def _dims(text: str) -> Counter[int]:
    return Counter(int(v) for v in text.split(",") if v.strip())


def doubling_mismatch(interface: str | None) -> bool:
    """Whether ``interface`` pairs a boundary with the same boundary taken twice."""
    m = _INTERFACE.fullmatch(interface or "")
    if m is None:
        return False
    lo, li, ro, ri = (_dims(g) for g in m.groups())

    def twice(c: Counter[int]) -> Counter[int]:
        return Counter({k: 2 * v for k, v in c.items()})

    return (lo, li) == (twice(ro), twice(ri)) or (ro, ri) == (twice(lo), twice(li))


@functools.lru_cache(maxsize=4096)
def _table(rows: tuple[tuple[Any, int, bool], ...]) -> dict[tuple[Any, int], bool]:
    return {(n, d): v for n, d, v in rows}


def _table_of(false: dict[str, Any]) -> dict[tuple[Any, int], bool] | None:
    rows = false.get("differs_at")
    if not rows:
        return None
    return _table(tuple((r[0], int(r[1]), bool(r[2])) for r in rows))


def _tail_equal(table: dict[tuple[Any, int], bool], axis: int) -> bool:
    """Whether the last TAIL tabled values of ``axis`` (0: n, 1: d) are equal at every tabled
    value of the other axis."""
    others = {k[1 - axis] for k in table}
    for o in others:
        line = sorted(
            (k[axis], v) for k, v in table.items() if k[1 - axis] == o and k[axis] is not None
        )
        if len(line) < TAIL or any(v for _, v in line[-TAIL:]):
            return False
    return bool(others)


def _periodic_equal(table: dict[tuple[Any, int], bool], n: int, d: int) -> bool:
    """Whether d's tabled n line has a least period p with the tabled m = n mod p equal."""
    line = {k[0]: v for k, v in table.items() if k[1] == d and k[0] is not None}
    if len(line) < 2 or all(line.values()):
        return False
    lo, hi = min(line), max(line)
    for p in range(1, (hi - lo) // 2 + 1):
        pairs = [(m, m + p) for m in line if m + p in line]
        if len(pairs) < PERIODS or any(line[a] != line[b] for a, b in pairs):
            continue
        reps = [m for m in line if (m - n) % p == 0]
        return bool(reps) and not line[reps[0]]
    return False


def _table_equal(table: dict[tuple[Any, int], bool], n: int | None, d: int) -> bool:
    """Whether the table's lines show the pair equal at an untabled ``(n, d)``."""
    if not any(table.values()):
        return True
    for axis, at in ((1, 1), (0, 0)):
        value = d if axis == 1 else n
        line = [v for k, v in table.items() if k[axis] == at]
        if value == at and line and not any(line):
            return True
    ns = [k[0] for k in table if k[0] is not None]
    if (
        n is not None
        and ns
        and n > max(ns)
        and (_tail_equal(table, 0) or _periodic_equal(table, n, d))
    ):
        return True
    return d > max(k[1] for k in table) and _tail_equal(table, 1)


def equal_at(false: dict[str, Any] | None, n: int | None, d: int | None) -> bool | None:
    """True when the F pair of ``false`` is equal at ``(n, d)`` by its differs_at table, its
    kind's closed-form rule or the table's lines, False when it differs there, None when only
    the F record's witness speaks (a witness that pairs a pure boundary with the same boundary
    doubled); False for an S equation and a symbolic point (``d`` None)."""
    if false is None or d is None:
        return False
    table = _table_of(false)
    if table is not None and (n, d) in table:
        return not table[(n, d)]
    kind = false.get("kind")
    params = false.get("params") or {}
    if kind in ("beyond_oracle", "prime_only", "coincident_dims") and "rule" in params:
        return _phase_trivial(params["rule"], n, d)
    if kind in ("scalar_small_one", "box_boundary") and "factor" in params:
        return _factor_one(params["factor"], n, d)
    if kind == "beyond_N" and n is not None:
        big_n = int(params["N"])
        if "index" in params:
            return n <= big_n
        return n % (big_n + 2) != big_n + 1
    if kind == "qubit_only":
        return d <= 2
    if table is not None and _table_equal(table, n, d):
        return True
    if kind == "off_by_one_n":
        return False
    witness = false.get("witness") or {}
    if witness.get("method", "entry") == "entry" and doubling_mismatch(witness.get("interface")):
        return None
    return False


def differs(false: dict[str, Any] | None, n: int | None, d: int | None) -> bool:
    """Whether the F pair of ``false`` differs at ``(n, d)`` (:func:`equal_at` False); True for
    an S equation, a symbolic point (``d`` None) or a kind with no equal region."""
    return equal_at(false, n, d) is False


__all__ = ["differs", "doubling_mismatch", "equal_at"]
