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

"""Python rechecks of the evidence behind a refutation, at its one point.

:func:`recheck_residue` re-evaluates a residue refutation's boundary assignment ``σ`` mod
``p`` on a fresh :class:`~archytaszx.semantics.lean_refute._Refuter`, its leaf tables built
anew; :func:`recheck_numeric` recomputes a ball refutation's entries at ``σ`` through the
pinned contraction and checks the ball holds their difference; :func:`recheck_interface`
compares the two boundaries' leg dimensions at a sample point. Each returns None when the
evidence holds, else what failed; :func:`checked_residue` and :func:`checked_numeric` raise
:class:`RecheckFailed` instead.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_net import LNet, NetUnsupported

NUMERIC_SLACK = Fraction(1, 10**30)
NUMERIC_FLOOR = 10**-25


class RecheckFailed(NetUnsupported):
    """A refutation's evidence did not hold when rechecked."""


def _order_ok(p: int, w: int, m: int) -> bool:
    """``p`` is prime and ``w`` has order exactly ``m`` mod ``p``."""
    if not sp.isprime(p) or pow(w, m, p) != 1:
        return False
    return all(pow(w, m // q, p) != 1 for q in sp.primefactors(m))


def _net_residue(r: Any, net: LNet, sigma: Mapping[int, int]) -> int:
    """The residue of ``net``'s entry at ``sigma`` on ``r``: leaf tables tabulated without a
    cache and eliminated, else the substitution contraction."""
    from archytaszx.semantics.lean_refute import SYM_FIRST
    from archytaszx.semantics.lean_subst import sym_plan

    widest = max(
        (
            math.prod(r.labdim(x) for x in dict.fromkeys((*n.outs, *n.ins)) if x not in sigma)
            for n in net.nodes
        ),
        default=1,
    )

    def by_tables() -> int:
        r.__dict__["_res_entries"] = 0
        leaves = []
        for n in net.nodes:
            scope = tuple(x for x in dict.fromkeys((*n.outs, *n.ins)) if x not in sigma)

            def leaf(tau: dict[int, int], n: Any = n) -> int:
                return int(r.res(r.node(n, {**tau, **sigma})))

            leaves.append((scope, r.res_table(scope, leaf)))
        _, _, final = r.res_eliminate(leaves, list(dict.fromkeys(net.summed)))
        return int(r.res_final(final, sigma))

    def by_substitution() -> int:
        return int(sym_plan(r, net, sigma).value)

    routes = (by_substitution, by_tables) if widest > SYM_FIRST else (by_tables, by_substitution)
    try:
        return routes[0]()
    except NetUnsupported:
        return routes[1]()


def recheck_residue(
    left: Diagram,
    right: Diagram,
    point: Mapping[str, object],
    describe: Callable[[Diagram], str] | None,
    found: Any,
) -> str | None:
    """None when a fresh refuter at ``point`` gives the residues ``found.residues`` and
    ``f₀(w) r₀ ≠ f₁(w) r₁`` mod ``p`` at the assignment ``found.sigma``, with ``found``'s
    order, prime and root; else what failed."""
    from archytaszx.semantics.lean_refute import _coeff_list, _kw, _Refuter, _root_factor

    if found.arity:
        sizes = [len(d.boundary_outputs) + len(d.boundary_inputs) for d in (left, right)]
        return None if sizes[0] != sizes[1] else f"both boundaries have {sizes[0]} slots"
    sigma = getattr(found, "sigma", None)
    if sigma is None:
        return "the search recorded no boundary assignment"
    p, w, m = found.p, found.w, found.M
    if not _order_ok(p, w, m):
        return f"{w} does not have order {m} mod the prime {p}"
    fresh = _Refuter(left, right, point, describe)
    fresh.sparse_all = found.sparse_all
    try:
        fresh.setup_env()
        fresh.choose_m()
        if fresh.M != m:
            return f"the order of ζ is {fresh.M}, not {m}"
        fresh.p, fresh.w = p, w
        a, b = fresh.a.net, fresh.b.net
        k0, k1 = _kw(fresh.sqw_list(a)), _kw(fresh.sqw_list(b))
        s0, s1 = fresh.at_point(fresh.s0), fresh.at_point(fresh.s1)
        if fresh.sparse_all:
            fresh.lmul = _root_factor([s0 * k0, s1 * k1])
        fs = [fresh.poly(fresh.lmul * s0 * k0), fresh.poly(fresh.lmul * s1 * k1)]
        lcd = math.lcm(*(int(sp.Rational(c).q) for f in fs for c in f.coeffs()))
        fr = [sum(c * pow(w, i, p) for i, c in enumerate(_coeff_list(f, lcd))) % p for f in fs]
        res = [_net_residue(fresh, net, sigma) for net in (a, b)]
    except NetUnsupported as exc:
        return f"the fresh evaluation failed ({exc})"
    searched = getattr(found, "residues", None)
    if searched is not None and [int(x) % p for x in searched] != res:
        return f"the residues {res} mod {p} at σ differ from the search's {list(searched)}"
    if (fr[0] * res[0] - fr[1] * res[1]) % p == 0:
        return f"the scaled entries agree mod {p} at σ = {dict(sorted(sigma.items()))}"
    return None


def checked_residue(
    source: str,
    left: Diagram,
    right: Diagram,
    point: Mapping[str, object],
    describe: Callable[[Diagram], str] | None,
    found: Any,
) -> str:
    """``source`` when :func:`recheck_residue` passes; raises :class:`RecheckFailed`
    otherwise."""
    why = recheck_residue(left, right, point, describe, found)
    if why is not None:
        raise RecheckFailed(f"residue recheck failed: {why}")
    return source


def recheck_numeric(
    left: Diagram,
    right: Diagram,
    point: Mapping[str, object],
    describe: Callable[[Diagram], str] | None,
    found: Any,
) -> str | None:
    """None when a fresh ball refuter's pinned contractions at ``found.evidence``'s assignment
    give its groups, and the ball ``(x, y, r)`` holds their difference and excludes 0; else
    what failed."""
    import mpmath  # type: ignore[import-untyped]

    from archytaszx.semantics.lean_numeric import _NumRefuter
    from archytaszx.semantics.lean_scalar import Inv, Rt

    sigma, g0, g1, x, y, rad = found.evidence
    if max(abs(x), abs(y)) <= rad:
        return "the ball contains 0"
    fresh = _NumRefuter(left, right, point, describe)
    try:
        fresh.setup_env()
        fresh.choose_m()
        if fresh.M != found.M or fresh.bases != found.bases:
            return "the order of ζ or the atoms differ"
        gs = []
        for net in (fresh.a.net, fresh.b.net):
            if fresh.use_plan(net):
                gs.append(fresh.final_groups(fresh.plan(net, sigma)[2], sigma))
            else:
                gs.append(fresh.groups_brute(net, sigma))
        if gs != [g0, g1]:
            return f"the entries at σ = {dict(sorted(sigma.items()))} differ from the search's"
        sq0, sq1 = fresh.sq_list(fresh.a.net), fresh.sq_list(fresh.b.net)
        k0 = sp.Mul(*(Inv(Rt(d)) ** c for d, c in sq0))
        k1 = sp.Mul(*(Inv(Rt(d)) ** c for d, c in sq1))
        v = fresh.num(fresh.difference(k0, k1, gs[0], gs[1]))
    except NetUnsupported as exc:
        return f"the fresh evaluation failed ({exc})"
    if abs(v) < NUMERIC_FLOOR:
        return "the difference is 0 numerically"
    dist = abs(mpmath.re(v) - mpmath.mpf(x.numerator) / x.denominator) + abs(
        mpmath.im(v) - mpmath.mpf(y.numerator) / y.denominator
    )
    bound = rad + NUMERIC_SLACK
    if dist > mpmath.mpf(bound.numerator) / bound.denominator:
        return "the ball does not hold the difference"
    return None


def checked_numeric(
    source: str,
    left: Diagram,
    right: Diagram,
    point: Mapping[str, object],
    describe: Callable[[Diagram], str] | None,
    found: Any,
) -> str:
    """``source`` when :func:`recheck_numeric` passes; raises :class:`RecheckFailed`
    otherwise."""
    why = recheck_numeric(left, right, point, describe, found)
    if why is not None:
        raise RecheckFailed(f"numeric recheck failed: {why}")
    return source


def _side(d: Diagram, refs: Any) -> list[tuple[sp.Expr, sp.Expr]]:
    """Each slot of ``refs``: its dimension and its number of legs."""
    from archytaszx.semantics.lean_net import _port_dim

    out = []
    for ref in refs:
        count = sp.Integer(1)
        for box in d.bang_boxes.values():
            if ref.node_id in box.node_scope or ref in box.port_scope:
                count *= box.multiplicity.to_sympy()
        out.append((_port_dim(d, ref).to_sympy(), count))
    return out


def _flat(slots: list[tuple[sp.Expr, sp.Expr]], vals: Mapping[sp.Symbol, Any]) -> list[int] | None:
    """The leg dimensions of ``slots`` at ``vals``, one per leg; None when a dimension is not a
    positive integer or a count not a natural number."""
    out: list[int] = []
    for dim, n in slots:
        dv, nv = dim.xreplace(vals), n.xreplace(vals)
        if not (dv.is_Integer and nv.is_Integer) or int(dv) <= 0 or int(nv) < 0:
            return None
        out += [int(dv)] * int(nv)
    return out


def recheck_interface(left: Diagram, right: Diagram) -> str | None:
    """None when, at a sample point, the input or the output leg dimensions of the two
    diagrams differ; else what failed."""
    from archytaszx.semantics.decide import sample_grid
    from archytaszx.semantics.lean_refute import MAX_POINTS

    sides = [
        (_side(left, left.boundary_inputs), _side(right, right.boundary_inputs)),
        (_side(left, left.boundary_outputs), _side(right, right.boundary_outputs)),
    ]
    syms = sorted(
        {s for pair in sides for slots in pair for e in slots for x in e for s in x.free_symbols},
        key=str,
    )
    points: list[Mapping[str, Any]] = [{}]
    if syms:
        points = []
        with contextlib.suppress(Exception):
            points = list(sample_grid(left, right))[:MAX_POINTS]
    for cand in points:
        if any(str(s) not in cand for s in syms):
            continue
        vals = {s: sp.nsimplify(cand[str(s)]) for s in syms}
        for ls, rs in sides:
            fl, fr = _flat(ls, vals), _flat(rs, vals)
            if fl is not None and fr is not None and fl != fr:
                return None
    return "interface recheck failed: the leg dimensions agree at every sample point"


__all__ = [
    "RecheckFailed",
    "checked_numeric",
    "checked_residue",
    "recheck_interface",
    "recheck_numeric",
    "recheck_residue",
]
