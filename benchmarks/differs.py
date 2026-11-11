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

Every other kind differs at every point of its domain, unless its witness is an interface
mismatch between a pure side and a doubled side (one side's boundary is the other's taken
twice): such a witness reads no entry, and the pair counts as not differing.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

_K, _N, _D = sp.symbols("k n d", integer=True)


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


def differs(false: dict[str, Any] | None, n: int | None, d: int | None) -> bool:
    """Whether the F pair of ``false`` differs at ``(n, d)``; True for an S equation, a
    symbolic point (``d`` None) or a kind with no equal region."""
    if false is None or d is None:
        return True
    kind = false.get("kind")
    params = false.get("params") or {}
    if kind in ("beyond_oracle", "prime_only", "coincident_dims") and "rule" in params:
        return not _phase_trivial(params["rule"], n, d)
    if kind in ("scalar_small_one", "box_boundary") and "factor" in params:
        return not _factor_one(params["factor"], n, d)
    if kind == "beyond_N" and n is not None:
        big_n = int(params["N"])
        if "index" in params:
            return n > big_n
        return n % (big_n + 2) == big_n + 1
    if kind == "qubit_only":
        return d > 2
    witness = false.get("witness") or {}
    return not (
        witness.get("method", "entry") == "entry" and doubling_mismatch(witness.get("interface"))
    )


__all__ = ["differs", "doubling_mismatch"]
