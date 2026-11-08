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

"""The point schedule of one (equation, tool) (BENCHMARKS.txt, section 5).

A point is ``(n, d)``, n None for a subject without a size parameter. Phase 1 is a fixed
growing ladder, ended by its first failure; phase 2 draws log-uniform points from the region not
dominated by a failed point until K successes or a cap on failures.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

Point = tuple[int | None, int]
MAX_DRAWS = 200_000


@dataclass(frozen=True)
class Axes:
    """The ranges a schedule draws from: n in [n_lo, n_hi] (None: no n axis), d in [d_lo,
    d_hi] (equal: a fixed d), with the domain's membership test."""

    n_lo: int | None
    n_hi: int | None
    d_lo: int
    d_hi: int
    has_n: Any
    has_d: Any

    def contains(self, p: Point) -> bool:
        """Whether ``p`` lies in the domain and the ranges."""
        n, d = p
        if n is not None and (self.n_lo is None or not (self.n_lo <= n <= (self.n_hi or n))):
            return False
        if not (self.d_lo <= d <= self.d_hi):
            return False
        return (n is None or self.has_n(n)) and self.has_d(d)


def axes(inst: Any, cfg: dict[str, Any]) -> Axes:
    """The schedule ranges of ``inst`` under the ``points`` config."""
    pc = cfg["points"]
    dom = inst.domain
    if inst.subject.size_parameter is None:
        n_lo = n_hi = None
    else:
        n_lo = max(pc["n_min"], dom.n_min)
        n_hi = pc["n_max"] if dom.n_max is None else min(pc["n_max"], dom.n_max)
    if inst.subject.general_d:
        d_lo = max(pc["d_min"], dom.d_min)
        d_hi = pc["d_max"] if dom.d_max is None else min(pc["d_max"], dom.d_max)
    else:
        d_lo = d_hi = int(inst.subject.dimension)
    return Axes(n_lo, n_hi, d_lo, d_hi, dom.has_n, dom.has_d)


def seed(base: int, equation: str, tool: str) -> int:
    """The schedule seed derived from (base, equation, tool)."""
    h = hashlib.sha256(f"{base}|{equation}|{tool}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _n_of(u: float, n_max: int) -> int:
    return math.floor((n_max + 1) ** u) - 1


def _d_of(u: float, d_max: int) -> int:
    return math.floor((d_max + 1) ** u)


def _next_n(ax: Axes, n: int) -> int | None:
    """The least domain n >= ``n`` within the range, or None."""
    hi = ax.n_hi if ax.n_hi is not None else n
    v = max(n, ax.n_lo or 0)
    while v <= hi:
        if ax.has_n(v):
            return v
        v += 1
    return None


def _next_d(ax: Axes, d: int) -> int | None:
    """The least domain d >= ``d`` within the range, or None."""
    v = max(d, ax.d_lo)
    while v <= ax.d_hi:
        if ax.has_d(v):
            return v
        v += 1
    return None


def ladder(ax: Axes, cfg: dict[str, Any]) -> list[Point]:
    """The phase 1 steps: k = 1..K at ((n_max+1)^(k/K) - 1, (d_max+1)^(k/K)), clamped,
    moved up into the domain and onto points not used before."""
    pc = cfg["points"]
    k_total = pc["K"]
    out: list[Point] = []
    used: set[Point] = set()
    for k in range(1, k_total + 1):
        u = k / k_total
        n: int | None = None
        if ax.n_lo is not None:
            n = _next_n(ax, min(_n_of(u, pc["n_max"]), ax.n_hi if ax.n_hi is not None else 10**18))
            if n is None:
                n = _next_n(ax, ax.n_lo)
        d = _next_d(ax, min(_d_of(u, pc["d_max"]), ax.d_hi))
        if d is None:
            d = _next_d(ax, ax.d_lo)
        if d is None or (ax.n_lo is not None and n is None):
            continue
        p = _distinct(ax, (n, d), used)
        if p is None:
            continue
        used.add(p)
        out.append(p)
    return out


def _distinct(ax: Axes, p: Point, used: set[Point]) -> Point | None:
    """``p``, or the next domain point not in ``used``: n raised first, then d."""
    n, d = p
    if p not in used:
        return p
    if n is not None:
        v = _next_n(ax, n + 1)
        while v is not None:
            if (v, d) not in used:
                return (v, d)
            v = _next_n(ax, v + 1)
    w = _next_d(ax, d + 1)
    while w is not None:
        q = (n, w)
        if q not in used:
            return q
        w = _next_d(ax, w + 1)
    for q in enumerate_domain(ax, limit=10_000):
        if q not in used:
            return q
    return None


def enumerate_domain(ax: Axes, limit: int) -> Iterator[Point]:
    """Domain points in (n, d) order, at most ``limit``."""
    ns: Any = [None] if ax.n_lo is None else range(ax.n_lo, (ax.n_hi or ax.n_lo) + 1)
    count = 0
    for n in ns:
        if n is not None and not ax.has_n(n):
            continue
        for d in range(ax.d_lo, ax.d_hi + 1):
            if ax.has_d(d):
                yield (n, d)
                count += 1
                if count >= limit:
                    return


def domain_size(ax: Axes, cap: int) -> int:
    """The number of domain points, counted up to ``cap``."""
    return sum(1 for _ in enumerate_domain(ax, cap))


def dominated(p: Point, failed: list[Point]) -> bool:
    """Whether ``p`` is at least as large in n and in d as some failed point."""
    n, d = p
    for fn, fd in failed:
        if (n is None or fn is None or n >= fn) and d >= fd:
            return True
    return False


@dataclass
class Phase2:
    """The seeded phase 2 draws of one (equation, tool)."""

    ax: Axes
    n_max: int
    d_max: int
    rng: random.Random
    draws: int = field(default=0)

    @classmethod
    def of(cls, ax: Axes, cfg: dict[str, Any], equation: str, tool: str) -> Phase2:
        """The sampler seeded from (points.seed, equation, tool)."""
        pc = cfg["points"]
        return cls(ax, pc["n_max"], pc["d_max"], random.Random(seed(pc["seed"], equation, tool)))

    def draw(self) -> Point:
        """One log-uniform domain point (out-of-domain values redrawn)."""
        while True:
            self.draws += 1
            n: int | None = None
            if self.ax.n_lo is not None:
                n = _n_of(self.rng.random(), self.n_max)
            d = (
                self.ax.d_lo
                if self.ax.d_lo == self.ax.d_hi
                else _d_of(self.rng.random(), self.d_max)
            )
            if self.ax.contains((n, d)):
                return (n, d)
            if self.draws > MAX_DRAWS:
                raise StopIteration

    def next(self, failed: list[Point], used: set[Point]) -> Point | None:
        """The next drawn point in the valid region and not used, or None when the draws run
        out."""
        try:
            while self.draws <= MAX_DRAWS:
                p = self.draw()
                if p not in used and not dominated(p, failed):
                    return p
        except StopIteration:
            return None
        return None
