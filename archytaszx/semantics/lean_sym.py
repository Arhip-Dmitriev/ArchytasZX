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

"""Lean 4 proofs of equalities over symbolic dimensions by the verified normaliser of
``ArchytasZX.SymNet``.

:func:`export_symbolic` writes both diagrams as networks, their dimensions as monomials in
the dimension symbols (``DT``), their phases as roots of unity times atoms ``ex g`` (``PT``),
their scalars as constant expressions (``S0``, ``S1``) and proves ``Equal`` by ``equal_sym``:
the kernel normalises the scaled difference of the two networks' expressions to nothing
(``decide +kernel``), for every value of the dimension symbols and phase parameters.

The functions below the exporter mirror their Lean namesakes on tuples, so that
:func:`sym_check` predicts ``symCheck``: ``DMon`` is ``(c, e)``, ``IPoly`` a tuple of
``(monomial, coefficient)``, ``CMon`` ``(q, r, sq, rc, am, gc)``, a factor ``(DMon, IPoly)``
and a term ``(CMon, sums, dels, phs)``.
"""

from __future__ import annotations

import functools
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_net import NetUnsupported, _Export, _join, _paren
from archytaszx.semantics.lean_scalar import II, NC, Ex, Inv, Om, Rt, ScalarUnsupported

DMon = tuple[int, tuple[int, ...]]
IPoly = tuple[tuple[tuple[int, ...], int], ...]
Fac = tuple[DMon, IPoly]
CMon = tuple[
    Fraction,
    Fraction,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, DMon], ...],
]
STm = tuple[CMon, tuple[tuple[int, DMon], ...], tuple[Fac, ...], tuple[Fac, ...]]

OFF = 1000000
ROUNDS = 3
FULL_TERMS = 4096
BS_NODES = False
"""Whether the translation writes binder and splitter nodes (``bsTerm``)."""
BS_SMALL = 64
"""The most nodes of both networks for which :func:`export_symbolic` writes binder and
splitter nodes with BS_NODES off."""
_BS_ON = False
"""Whether the current export writes binder and splitter nodes."""
_FOLD = False
"""Whether the compact check folds binder, compact spider and splitter blocks (``trNetK``)."""
BLOCK_MAX = 8
"""The most nodes of one ``trBlockM`` theorem."""
BLOCK_TERMS = 24
"""The most nodes times terms of the largest stated expression in one ``trBlockM`` theorem."""
NORM_TERMS = 1 << 14
"""The most terms :func:`sym_check` multiplies out in the unnormalised translation."""


def _emod(a: int, b: int) -> int:
    """Lean's ``Int`` remainder: nonnegative, ``a`` itself for ``b = 0``."""
    return a if b == 0 else a % abs(b)


def _ediv(a: int, b: int) -> int:
    """Lean's ``Int`` division, ``a = b q + r`` with ``0 ≤ r < |b|``."""
    return 0 if b == 0 else (a - _emod(a, b)) // b


# dense vectors


def nadd(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    n = max(len(a), len(b))
    return tuple((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n))


def nsub(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    return tuple(max(0, a[i] - (b[i] if i < len(b) else 0)) for i in range(len(a)))


def nle(a: Sequence[int], b: Sequence[int]) -> bool:
    return all(x <= (b[i] if i < len(b) else 0) for i, x in enumerate(a))


def trim(a: Sequence[int]) -> tuple[int, ...]:
    out = list(a)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


# dimension monomials


def dm_mul(a: DMon, b: DMon) -> DMon:
    return (a[0] * b[0], trim(nadd(a[1], b[1])))


def dm_dvd(a: DMon, b: DMon) -> bool:
    return a[0] != 0 and b[0] % a[0] == 0 and nle(a[1], b[1])


def dm_div(b: DMon, a: DMon) -> DMon:
    return (b[0] // a[0] if a[0] else 0, trim(nsub(b[1], a[1])))


def dm_const(m: DMon) -> bool:
    return all(x == 0 for x in m[1])


def dm_den(m: DMon, dv: Callable[[int], int]) -> int:
    out = m[0]
    for i, e in enumerate(m[1]):
        out *= dv(i) ** e
    return out


# polynomials


def im_ins(x: int, m: Sequence[int]) -> tuple[int, ...]:
    out = list(m)
    for i, y in enumerate(out):
        if x <= y:
            return tuple(out[:i] + [x] + out[i:])
    return tuple(out + [x])


def im_mul(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    out = tuple(a)
    for x in reversed(b):
        out = im_ins(x, out)
    return out


def ip_ins(m: tuple[int, ...], c: int, p: IPoly) -> IPoly:
    out = list(p)
    for i, (m2, c2) in enumerate(out):
        if m == m2:
            if c + c2 == 0:
                return tuple(out[:i] + out[i + 1 :])
            return tuple(out[:i] + [(m, c + c2)] + out[i + 1 :])
        if m < m2:
            if c == 0:
                return tuple(out)
            return tuple(out[:i] + [(m, c)] + out[i:])
    return tuple(out + ([(m, c)] if c != 0 else []))


def ip_add(p: IPoly, q: IPoly) -> IPoly:
    out = p
    for m, c in reversed(q):
        out = ip_ins(m, c, out)
    return out


def ip_smul(k: int, p: IPoly) -> IPoly:
    return () if k == 0 else tuple((m, k * c) for m, c in p)


def ip_mul(p: IPoly, q: IPoly) -> IPoly:
    acc: IPoly = ()
    for m, c in reversed(p):
        for m2, c2 in reversed(q):
            acc = ip_ins(im_mul(m, m2), c * c2, acc)
    return acc


def ip_c(c: int) -> IPoly:
    return ip_ins((), c, ())


def ip_v(x: int) -> IPoly:
    return (((x,), 1),)


def ip_sub(p: IPoly, q: IPoly) -> IPoly:
    return ip_add(p, ip_smul(-1, q))


def ip_pow(r: IPoly, k: int) -> IPoly:
    out = ip_c(1)
    for _ in range(k):
        out = ip_mul(r, out)
    return out


@functools.lru_cache(maxsize=1 << 18)
def ip_varset(p: IPoly) -> frozenset[int]:
    """The variables of ``p``."""
    return frozenset(x for m, _ in p for x in m)


def ip_subst(x: int, r: IPoly, p: IPoly) -> IPoly:
    if x not in ip_varset(p):
        return p
    acc: IPoly = ()
    for m, c in reversed(p):
        rest = tuple(y for y in m if y != x)
        acc = ip_add(acc, ip_mul(ip_smul(c, ((rest, 1),)), ip_pow(r, m.count(x))))
    return acc


def ip_has(x: int, p: IPoly) -> bool:
    return x in ip_varset(p)


@functools.lru_cache(maxsize=1 << 18)
def ip_lin(x: int, p: IPoly) -> tuple[IPoly, IPoly] | None:
    if not all(m.count(x) <= 1 for m, _ in p):
        return None
    a: IPoly = ()
    b: IPoly = ()
    for m, c in reversed(p):
        if x in m:
            mm = list(m)
            mm.remove(x)
            a = ip_ins(tuple(mm), c, a)
        else:
            b = ip_ins(m, c, b)
    return a, b


def ip_const(p: IPoly) -> int | None:
    if not p:
        return 0
    if len(p) == 1 and p[0][0] == ():
        return p[0][1]
    return None


def ip_modc(k: int, p: IPoly) -> IPoly:
    acc: IPoly = ()
    for m, c in reversed(p):
        acc = ip_ins(m, _emod(c, k), acc)
    return acc


def ip_divby(k: int, p: IPoly) -> bool:
    return all(_emod(c, k) == 0 for _, c in p)


def ip_divc(k: int, p: IPoly) -> IPoly:
    acc: IPoly = ()
    for m, c in reversed(p):
        acc = ip_ins(m, _ediv(c, k), acc)
    return acc


def ip_sign(p: IPoly) -> IPoly:
    if p and p[0][1] < 0:
        return ip_smul(-1, p)
    return p


def ip_vars(p: IPoly) -> list[int]:
    return [x for m, _ in p for x in m]


# coefficient monomials


def rnorm(q: Fraction, r: Fraction) -> tuple[Fraction, Fraction]:
    r2 = r - math.floor(r)
    if Fraction(1, 2) <= r2:
        return -q, r2 - Fraction(1, 2)
    return q, r2


def rc_ins(c: int, lst: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    out = list(lst)
    for i, x in enumerate(out):
        if c == x:
            return c, tuple(out[:i] + out[i + 1 :])
        if c < x:
            return 1, tuple(out[:i] + [c] + out[i:])
    return 1, tuple(out + [c])


def rc_mul(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    f, lst = 1, a
    for c in reversed(b):
        g, lst = rc_ins(c, lst)
        f = g * f
    return f, lst


def gc_key(g: tuple[int, DMon]) -> tuple[int, int, tuple[int, ...]]:
    return (g[0], g[1][0], g[1][1])


def gc_ins(g: tuple[int, DMon], lst: tuple[tuple[int, DMon], ...]) -> tuple[tuple[int, DMon], ...]:
    out = list(lst)
    for i, h in enumerate(out):
        if not gc_key(h) < gc_key(g):
            return tuple(out[:i] + [g] + out[i:])
    return tuple(out + [g])


def cm_mul(a: CMon, b: CMon) -> CMon:
    q, r = rnorm(a[0] * b[0], a[1] + b[1])
    f, rc = rc_mul(a[3], b[3])
    gc = a[5]
    for g in reversed(b[5]):
        gc = gc_ins(g, gc)
    return (q * f, r, trim(nadd(a[2], b[2])), rc, trim(nadd(a[4], b[4])), gc)


def of_q(q: Fraction | int) -> CMon:
    return (Fraction(q), Fraction(0), (), (), (), ())


def _isqrt_exact(c: int) -> bool:
    s = math.isqrt(c)
    return s * s == c


def rt_c(c: int, k: int) -> CMon:
    half = _ediv(k, 2)
    if _emod(k, 2) == 0:
        return of_q(Fraction(c) ** half)
    if _isqrt_exact(c):
        return of_q(Fraction(c) ** half * math.isqrt(c))
    return (Fraction(c) ** half, Fraction(0), (), (c,), (), ())


def rt_d(d: DMon, k: int) -> CMon:
    m = rt_c(d[0], k)
    return (m[0], m[1], trim(tuple(k * x for x in d[1])), m[3], m[4], m[5])


def gcd_a(a: int, d: DMon) -> CMon:
    if dm_const(d) or abs(a) == 1:
        return of_q(math.gcd(abs(a), d[0]))
    return (Fraction(1), Fraction(0), (), (), (), ((abs(a), d),))


def ex_q(r: Fraction) -> CMon:
    q, r2 = rnorm(Fraction(1), r)
    return (q, r2, (), (), (), ())


def atom_p(j: int, k: int) -> CMon:
    return (Fraction(1), Fraction(0), (), (), trim((0,) * j + (k,)), ())


# terms


def fac_vars(f: Fac) -> list[int]:
    return ip_vars(f[1])


def tm_vars(t: STm) -> list[int]:
    return [x for f in (*t[2], *t[3]) for x in fac_vars(f)]


def bvs(t: STm) -> list[int]:
    return [x for x, _ in t[1]]


def tm_mul(s: STm, t: STm) -> STm:
    return (cm_mul(s[0], t[0]), s[1] + t[1], s[2] + t[2], s[3] + t[3])


def mul_ok(s: STm, t: STm) -> bool:
    vt, vs = set(tm_vars(t)), set(tm_vars(s))
    return all(x not in vt for x in bvs(s)) and all(x not in vs for x in bvs(t))


def fac_has(x: int, f: Fac) -> bool:
    return x in ip_varset(f[1])


def fac_sub(x: int, v: IPoly, f: Fac) -> Fac:
    return (f[0], ip_subst(x, v, f[1]))


def pin_val(s: int, b: IPoly) -> IPoly:
    return ip_smul(-s, b)


def _splits(lst: Sequence[Any]) -> list[tuple[tuple[Any, ...], Any, tuple[Any, ...]]]:
    return [(tuple(lst[:i]), lst[i], tuple(lst[i + 1 :])) for i in range(len(lst))]


# eliminations


def elim_unused(
    x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    if d[0] != 0 and all(not fac_has(x, f) for f in (*ds, *ps)):
        return rt_d(d, 2), ds, ps
    return None


def elim_delta1(
    x: int, d: DMon, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    lin = ip_lin(x, f[1])
    if lin is None:
        return None
    a, b = lin
    s = ip_const(a)
    if s is None:
        return None
    if (
        (s in (1, -1))
        and d[0] != 0
        and dm_dvd(f[0], d)
        and all(not fac_has(x, g) or dm_dvd(g[0], f[0]) for g in (*pre, *post, *ps))
    ):
        v = pin_val(s, b)
        return (
            rt_d(dm_div(d, f[0]), 2),
            tuple(fac_sub(x, v, g) for g in (*pre, *post)),
            tuple(fac_sub(x, v, g) for g in ps),
        )
    return None


def elim_delta(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    for i, f in enumerate(ds):
        if x not in ip_varset(f[1]):
            continue
        e = elim_delta1(x, d, ds[:i], f, ds[i + 1 :], ps)
        if e is not None:
            return e
    return None


def char_data(x: int, d: DMon, lst: tuple[Fac, ...]) -> tuple[IPoly, IPoly] | None:
    if not lst:
        return (), ()
    rest = char_data(x, d, lst[1:])
    if rest is None:
        return None
    big_a, big_b = rest
    f = lst[0]
    if fac_has(x, f):
        if f[0] == d:
            lin = ip_lin(x, f[1])
            if lin is None:
                return None
            return ip_add(big_a, lin[0]), ip_add(big_b, lin[1])
        return None
    return big_a, big_b


def elim_char(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    if d[0] != 0 and all(not fac_has(x, f) for f in ds):
        cd = char_data(x, d, ps)
        if cd is not None:
            big_a, big_b = cd
            return (
                rt_d(d, 2),
                ds + ((d, big_a),),
                tuple(f for f in ps if not fac_has(x, f)) + ((d, big_b),),
            )
    return None


def elim_gcd1(
    x: int, d: DMon, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> Any:
    if f[0] == d and d[0] != 0 and all(not fac_has(x, g) for g in (*pre, *post, *ps)):
        lin = ip_lin(x, f[1])
        if lin is None:
            return None
        a, b = lin
        alpha = ip_const(a)
        if alpha is None:
            return None
        if ip_divby(alpha, b):
            return gcd_a(alpha, d), pre + post, ps
    return None


def elim_gcd(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    for pre, f, post in _splits(ds):
        e = elim_gcd1(x, d, pre, f, post, ps)
        if e is not None:
            return e
    return None


@functools.lru_cache(maxsize=1 << 18)
def elim_var(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    for fn in (elim_unused, elim_delta, elim_char, elim_gcd):
        e = fn(x, d, ds, ps)
        if e is not None:
            return e
    return None


def ip_quo(k: int, p: IPoly) -> IPoly:
    return tuple((m, c // k) for m, c in p)


def split_del(r: Sequence[tuple[int, DMon]], y: int, d: DMon, f: Fac) -> tuple[Fac, Fac] | None:
    """``splitDel``: the deltas ``[D | L]`` and ``[M / D | U + m]`` of ``f``."""
    rv = [x for x, _ in r]
    if not (dm_const(d) and d[0] >= 2 and dm_dvd(d, f[0]) and rv.count(y) == 1 and (y, d) in r):
        return None
    lin = ip_lin(y, f[1])
    if lin is None:
        return None
    s = ip_const(lin[0])
    if s not in (1, -1):
        return None
    b = lin[1]
    k = d[0]
    c = sum(t[1] for t in b if not t[0])
    rest = tuple(t for t in b if t[0])
    zs = [t[0][0] for t in rest if t[1] == -s and len(t[0]) == 1]
    for z in zs:
        if (
            z != y
            and rv.count(z) == 1
            and (z, d) in r
            and [t for t in rest if t[0] == (z,)] == [((z,), -s)]
        ):
            rem = tuple(t for t in rest if t[0] != (z,))
            if ip_divby(k, rem) and c % k == 0:
                return (
                    (d, ip_add(ip_smul(s, ip_v(y)), ip_smul(-s, ip_v(z)))),
                    (dm_div(f[0], d), ip_add(ip_quo(k, rem), ip_c(c // k))),
                )
            return None
    if not ip_divby(k, rest):
        return None
    cp = c % k
    m = 1 if (s == 1 and cp != 0) else 0
    return (
        (d, ip_add(ip_smul(s, ip_v(y)), ip_c(cp))),
        (dm_div(f[0], d), ip_add(ip_quo(k, rest), ip_c((c - cp) // k + m))),
    )


def split_step(t: STm) -> STm | None:
    """``splitStep``: the first delta split by a bound variable, in place."""
    for i, f in enumerate(t[2]):
        vs = ip_varset(f[1])
        for y, d in t[1]:
            if y not in vs:
                continue
            got = split_del(t[1], y, d, f)
            if got is not None:
                return (t[0], t[1], (*t[2][:i], got[0], got[1], *t[2][i + 1 :]), t[3])
    return None


def elim_step(t: STm) -> STm | None:
    """``elimStep``, through :func:`elim_choice`."""
    got = elim_choice(t)
    return None if got is None else got[0]


def elim_all(n: int, t: STm) -> STm:
    for _ in range(n):
        t2 = elim_step(t)
        if t2 is None:
            return t
        t = t2
    return t


# canonical forms


def _pick(p: IPoly, ok: Callable[[int], bool]) -> tuple[int, int, IPoly] | None:
    acc: tuple[int, int, IPoly] | None = None
    for x in ip_vars(p):
        lin = ip_lin(x, p)
        if lin is None:
            continue
        c = ip_const(lin[0])
        if c is None or not ok(c):
            continue
        if acc is None or acc[0] < x:
            acc = (x, c, lin[1])
    return acc


@functools.lru_cache(maxsize=1 << 16)
def pivot_of(p: IPoly) -> tuple[int, int, IPoly] | None:
    return _pick(p, lambda c: c in (1, -1))


@functools.lru_cache(maxsize=1 << 16)
def div_of(p: IPoly) -> tuple[int, int, IPoly] | None:
    return _pick(p, lambda c: c != 0)


def piv_sub(m: DMon, x: int, v: IPoly, g: Fac) -> Fac:
    return fac_sub(x, v, g) if dm_dvd(g[0], m) else g


def div_sub(m: DMon, x: int, c: int, b: IPoly, g: Fac) -> Fac:
    if not dm_dvd(g[0], m):
        return g
    lin = ip_lin(x, g[1])
    if lin is None:
        return g
    ag, bg = lin
    if ip_divby(c, ag):
        return (g[0], ip_sub(bg, ip_mul(ip_divc(c, ag), b)))
    return g


@functools.lru_cache(maxsize=1 << 16)
def _step(f: Fac) -> tuple[Any, ...] | None:
    """``stepOf f`` as data: ``(True, m, x, v)`` for a pivot, ``(False, m, x, c, b)`` for a
    division."""
    q = pivot_of(f[1])
    if q is not None:
        return (True, f[0], q[0], pin_val(q[1], q[2]))
    q2 = div_of(f[1])
    if q2 is not None:
        return (False, f[0], q2[0], q2[1], q2[2])
    return None


def _apply(st: tuple[Any, ...], g: Fac) -> Fac:
    """:func:`piv_sub` or :func:`div_sub` of a :func:`_step`; a pivot leaves a factor without
    its variable as it is."""
    if st[0]:
        if st[2] not in ip_varset(g[1]):
            return g
        return piv_sub(st[1], st[2], st[3], g)
    return div_sub(st[1], st[2], st[3], st[4], g)


def step_of(f: Fac) -> Callable[[Fac], Fac] | None:
    st = _step(f)
    return None if st is None else functools.partial(_apply, st)


def piv_pass(
    n: int, a: tuple[Fac, ...], b: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> tuple[tuple[Fac, ...], tuple[Fac, ...]]:
    """``pivPass``, on lists with each factor's variable set alongside; a pivot rewrites only
    the factors holding its variable."""
    la, lb, lp = list(a), list(b), list(ps)
    va = [ip_varset(g[1]) for g in la]
    vb = [ip_varset(g[1]) for g in lb]
    vp = [ip_varset(g[1]) for g in lp]

    def run(st: tuple[Any, ...], fs: list[Fac], vs: list[frozenset[int]], start: int) -> None:
        for j in range(start, len(fs)):
            if st[0] and st[2] not in vs[j]:
                continue
            g = _apply(st, fs[j])
            if g is not fs[j]:
                fs[j] = g
                vs[j] = ip_varset(g[1])

    i = 0
    while True:
        if n == 0:
            return tuple(la) + tuple(lb[i:]), tuple(lp)
        if i == len(lb):
            return tuple(la), tuple(lp)
        n -= 1
        f, vf = lb[i], vb[i]
        i += 1
        st = _step(f)
        if st is not None:
            run(st, la, va, 0)
            run(st, lb, vb, i)
            run(st, lp, vp, 0)
        la.append(f)
        va.append(vf)


def flv(fl: Sequence[int], i: int) -> int:
    return fl[i] if i < len(fl) else 0


def sym_mod(c: int, k: int) -> int:
    r = _emod(c, k)
    return r - k if 2 * r > k else r


def ip_mods(k: int, p: IPoly) -> IPoly:
    acc: IPoly = ()
    for m, c in reversed(p):
        acc = ip_ins(m, sym_mod(c, k), acc)
    return acc


def del_poly(f: Fac) -> IPoly:
    return ip_sign(ip_mods(f[0][0], f[1])) if dm_const(f[0]) else ip_sign(f[1])


_ZERO = object()


def norm_del(fl: Sequence[int], f: Fac) -> Any:
    """``None`` when the delta vanishes, ``()`` when it is one, else the factor."""
    p = del_poly(f)
    c = ip_const(p)
    if c is not None:
        if c == 0:
            return ()
        if dm_const(f[0]):
            return () if _emod(c, f[0][0]) == 0 else None
        if abs(c) < dm_den(f[0], lambda i: flv(fl, i)):
            return None
        return (f[0], p)
    return (f[0], p)


def norm_dels(fl: Sequence[int], lst: tuple[Fac, ...]) -> tuple[Fac, ...] | None:
    out = []
    for f in lst:
        r = norm_del(fl, f)
        if r is None:
            return None
        if r != ():
            out.append(r)
    return tuple(out)


def norm_ph(f: Fac) -> tuple[CMon, Fac | None]:
    if dm_const(f[0]) and f[0][0] != 0:
        p = ip_modc(f[0][0], f[1])
        c = ip_const(p)
        if c is not None:
            return ex_q(Fraction(c, f[0][0])), None
        return of_q(1), (f[0], p)
    if not f[1]:
        return of_q(1), None
    return of_q(1), f


def norm_phs(lst: tuple[Fac, ...]) -> tuple[CMon, tuple[Fac, ...]]:
    if not lst:
        return of_q(1), ()
    k, rest = norm_phs(lst[1:])
    k1, g = norm_ph(lst[0])
    return cm_mul(k1, k), (rest if g is None else (g,) + rest)


def ph_ins(f: Fac, lst: tuple[Fac, ...]) -> tuple[Fac, ...]:
    out = list(lst)
    for i, g in enumerate(out):
        if g[0] == f[0]:
            out[i] = (g[0], ip_add(g[1], f[1]))
            return tuple(out)
    return tuple(out + [f])


def ph_merge(ps: tuple[Fac, ...]) -> tuple[Fac, ...]:
    acc: tuple[Fac, ...] = ()
    for f in ps:
        acc = ph_ins(f, acc)
    return acc


def _fkey(f: Fac) -> tuple[Any, ...]:
    return (f[0][0], f[0][1], f[1])


def fac_ins(f: Fac, lst: tuple[Fac, ...]) -> tuple[Fac, ...]:
    out = list(lst)
    for i, g in enumerate(out):
        if not _fkey(g) < _fkey(f):
            return tuple(out[:i] + [f] + out[i:])
    return tuple(out + [f])


def del_ins(f: Fac, lst: tuple[Fac, ...]) -> tuple[Fac, ...]:
    out = list(lst)
    for i, g in enumerate(out):
        if g == f:
            return tuple(out)
        if not _fkey(g) < _fkey(f):
            return tuple(out[:i] + [f] + out[i:])
    return tuple(out + [f])


RBASE = 2000000


def ren_one(t: STm, x: int, y: int) -> STm | None:
    b = bvs(t)
    if b.count(x) == 1 and y not in b and y not in tm_vars(t):
        v = ip_v(y)
        return (
            t[0],
            tuple((y, p[1]) if p[0] == x else p for p in t[1]),
            tuple(fac_sub(x, v, f) for f in t[2]),
            tuple(fac_sub(x, v, f) for f in t[3]),
        )
    return None


def ren_list(t: STm, k: int, xs: Sequence[int]) -> STm:
    for x in xs:
        r = ren_one(t, x, RBASE + k)
        t = t if r is None else r
        k += 1
    return t


def canon_tm(fl: Sequence[int], t: STm) -> STm | None:
    t = ren_list(t, 0, bvs(t))
    ds0, ps0 = piv_pass(len(t[2]), (), t[2], t[3])
    ds = norm_dels(fl, ds0)
    if ds is None:
        return None
    k, ps = norm_phs(ph_merge(ps0))
    dsort: tuple[Fac, ...] = ()
    for f in reversed(ds):
        dsort = del_ins(f, dsort)
    psort: tuple[Fac, ...] = ()
    for f in reversed(ps):
        psort = fac_ins(f, psort)
    return (cm_mul(t[0], k), t[1], dsort, psort)


def norm_tm(fl: Sequence[int], r: int, t: STm) -> STm | None:
    for _ in range(r):
        t2 = canon_tm(fl, elim_all(len(t[1]) + 1, t))
        if t2 is None:
            return None
        t = t2
    return t


def shape(t: STm) -> STm:
    c = t[0]
    return ((Fraction(0), c[1], c[2], c[3], c[4], c[5]), t[1], t[2], t[3])


def with_q(t: STm, q: Fraction) -> STm:
    c = t[0]
    return ((q, c[1], c[2], c[3], c[4], c[5]), t[1], t[2], t[3])


def tm_add(t: STm, lst: tuple[STm, ...]) -> tuple[STm, ...]:
    out = list(lst)
    st = shape(t)
    for i, u in enumerate(out):
        if shape(u) == st:
            q = u[0][0] + t[0][0]
            if q == 0:
                return tuple(out[:i] + out[i + 1 :])
            out[i] = with_q(u, q)
            return tuple(out)
    return tuple(out + ([t] if t[0][0] != 0 else []))


@functools.lru_cache(maxsize=1 << 16)
def _norm_cached(fl: tuple[int, ...], t: STm) -> STm | None:
    return norm_tm(fl, ROUNDS, t)


def nf_ex(fl: Sequence[int], e: Sequence[STm]) -> tuple[STm, ...]:
    """``tm_add`` folded over the normalised terms, by a shape-keyed dict in insertion order."""
    key = tuple(fl)
    acc: dict[STm, STm] = {}
    for t in e:
        n = _norm_cached(key, t)
        if n is None:
            continue
        st = shape(n)
        u = acc.get(st)
        if u is None:
            if n[0][0] != 0:
                acc[st] = n
            continue
        q = u[0][0] + n[0][0]
        if q == 0:
            del acc[st]
        else:
            acc[st] = with_q(u, q)
    return tuple(acc.values())


# characters of sum-free residuals (SymFourier)

FZCert = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[tuple[int, ...], IPoly, int, int, Fraction], ...],
]


def _fz_mod(e: Sequence[STm]) -> int | None:
    """The modulus N of :func:`fz_cert`: the lcm of the constant moduli and the denominators of
    the constant phases; None when a term has a sum, a symbolic modulus or another constant."""
    n = 1
    for t in e:
        c = t[0]
        if t[1] or c[2] or c[3] or c[4] or c[5]:
            return None
        for f in (*t[2], *t[3]):
            if not dm_const(f[0]) or f[0][0] <= 0:
                return None
            n = math.lcm(n, f[0][0])
        n = math.lcm(n, Fraction(c[1]).denominator)
    return n


def exp_term(n: int, t: STm) -> list[tuple[Fraction, IPoly]] | None:
    """``expTerm``: a sum-free term as characters ``w omega_N^P``."""
    c = t[0]
    if t[1] or c[2] or c[3] or c[4] or c[5]:
        return None
    if not all(dm_const(f[0]) and f[0][0] > 0 and n % f[0][0] == 0 for f in (*t[2], *t[3])):
        return None
    rn = Fraction(c[1]) * n
    if rn.denominator != 1:
        return None
    out = [(Fraction(c[0]), ip_c(rn.numerator))]
    for f in reversed(t[2]):
        m = f[0][0]
        out = [(w / m, ip_add(p, ip_smul(k * (n // m), f[1]))) for w, p in out for k in range(m)]
    for f in reversed(t[3]):
        m = f[0][0]
        out = [(w, ip_add(p, ip_smul(n // m, f[1]))) for w, p in out]
    return out


def _cycle_cert(n: int, v: dict[int, Fraction]) -> tuple[tuple[int, int, Fraction], ...] | None:
    """Coefficients of the cycles ``Sum_{i<p} omega_N^(a + i N/p)`` (primes ``p | N``, ``a <
    N/p``) summing to the vector ``v``; None when there are none."""
    rels = [(p, a) for p in sp.primefactors(n) for a in range(n // p)]
    if not rels:
        return () if not any(v.values()) else None
    mat = sp.zeros(n, len(rels))
    for col, (p, a) in enumerate(rels):
        for i in range(p):
            mat[a + i * (n // p), col] = 1
    rhs = sp.Matrix([sp.Rational(v.get(j, Fraction(0))) for j in range(n)])
    try:
        sol, params = mat.gauss_jordan_solve(rhs)
    except ValueError:
        return None
    sol = sol.subs({s: 0 for s in params})
    out = []
    for (p, a), x in zip(rels, sol, strict=True):
        q = Fraction(int(sp.fraction(x)[0]), int(sp.fraction(x)[1]))
        if q:
            out.append((p, a, q))
    return tuple(out)


def _cdel(f: Fac) -> bool | None:
    """``cdel``: whether a delta with a constant argument over a positive constant modulus
    holds; None for any other delta."""
    c = ip_const(f[1])
    if c is None or not (dm_const(f[0]) and f[0][0] > 0):
        return None
    return c % f[0][0] == 0


def _prune(t: STm) -> STm | None:
    """``pruneTm``."""
    if any(_cdel(f) is False for f in t[2]):
        return None
    return (t[0], t[1], tuple(f for f in t[2] if _cdel(f) is not True), t[3])


def _explode(
    n: int, pv: Sequence[int], vv: tuple[int, ...], e: Sequence[STm]
) -> list[tuple[tuple[int, ...], STm]]:
    """``explode``: every pinned label split into its residues mod ``n``."""
    if not pv:
        return [(vv, t) for t in e]
    x, rest = pv[0], pv[1:]
    out: list[tuple[tuple[int, ...], STm]] = []
    for v in range(n):
        sub = []
        for t in e:
            s = (
                t[0],
                t[1],
                tuple(fac_sub(x, ip_c(v), f) for f in t[2]),
                tuple(fac_sub(x, ip_c(v), f) for f in t[3]),
            )
            p = _prune(s)
            if p is not None:
                sub.append(p)
        out += _explode(n, rest, (*vv, v), sub)
    return out


def _pinned(e: Sequence[STm]) -> list[int]:
    """The labels some delta pins alone: ``[s x + c]`` with ``s = +-1``."""
    out = set()
    for t in e:
        for _m, p in t[2]:
            xs = {m[0] for m, _c in p if m}
            if len(xs) == 1 and all(len(m) <= 1 and (not m or abs(c) == 1) for m, c in p):
                out |= xs
    return sorted(out)


def fz_cert(e: Sequence[STm]) -> tuple[int, FZCert] | None:
    """``fzOK``'s modulus and certificate for a sum-free expression: the labels pinned alone
    split into residues, the rest as characters grouped by residues and non-constant exponent
    mod N, each group vanishing by cycles; None otherwise."""
    if not e:
        return None
    n = _fz_mod(e)
    if n is None or n < 2:
        return None
    pv = _pinned(e)
    groups: dict[tuple[tuple[int, ...], IPoly], dict[int, Fraction]] = {}
    for vv, t in _explode(n, pv, (), e):
        chars = exp_term(n, t)
        if chars is None:
            return None
        for w, p in chars:
            if any(len(m) > 1 for m, _c in p):
                return None
            key = ip_modc(n, tuple(s for s in p if s[0]))
            idx = _emod(sum(c for m, c in p if not m), n)
            g = groups.setdefault((vv, key), {})
            g[idx] = g.get(idx, Fraction(0)) + w
    rows = []
    for (vv, key), v in groups.items():
        cyc = _cycle_cert(n, v)
        if cyc is None:
            return None
        rows += [(vv, key, p, a, q) for p, a, q in cyc]
    vs = tuple(sorted({m[0] for _vv, key in groups for m, _c in key}))
    return n, (tuple(pv), vs, tuple(rows))


def _lean_cert(cert: FZCert) -> str:
    """A certificate literal."""
    rows = ", ".join(
        f"({_lean_ints(list(vv))}, {_lean_ip(k)}, {_rn(p)}, {_rn(a)}, {_rq(q)})"
        for vv, k, p, a, q in cert[2]
    )
    return f"(({_lean_ints(list(cert[0]))}, {_lean_ints(list(cert[1]))}, [{rows}]) : FzCert)"


# residuals vanishing on the labels' ranges (SymRange)

RgCert = tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, FZCert], ...]]
RG_CASES = 729
"""The most cases :func:`rg_cert` splits the labels into."""
RG_WORK = 1 << 16
"""The most cases times residual terms :func:`rg_cert` normalises."""
RG_FUEL = 256
"""The most pins and eliminations :func:`rg_cert` runs on one term."""
_NO_CERT: FZCert = ((), (), ())


def elim_pin1(
    x: int, d: DMon, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    """``elimPin1``: ``x`` of constant range ``D`` held by ``[D | s x + c]`` at ``(-s c) mod D``."""
    if not (f[0] == d and dm_const(d) and d[0] > 0):
        return None
    lin = ip_lin(x, f[1])
    if lin is None:
        return None
    s, c = ip_const(lin[0]), ip_const(lin[1])
    if s not in (1, -1) or c is None:
        return None
    v = ip_c((-s * c) % d[0])
    return (
        of_q(1),
        tuple(fac_sub(x, v, g) for g in (*pre, *post)),
        tuple(fac_sub(x, v, g) for g in ps),
    )


def elim_var_r(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    """``elimVarR``: a pin by range first, else :func:`elim_var`."""
    for pre, f, post in _splits(ds):
        e = elim_pin1(x, d, pre, f, post, ps)
        if e is not None:
            return e
    return elim_var(x, d, ds, ps)


def split_step_r(t: STm) -> STm | None:
    """``splitStepR``: :func:`split_step` without splits of a delta whose modulus is the
    variable's range."""
    for pre, f, post in _splits(t[2]):
        for y, d in t[1]:
            got = None if f[0] == d else split_del(t[1], y, d, f)
            if got is not None:
                return (t[0], t[1], (*pre, got[0], got[1], *post), t[3])
    return None


def elim_step_r(t: STm) -> STm | None:
    """``elimStepR``: the first elimination by :func:`elim_var_r`, else
    :func:`split_step_r`."""
    for pre, (x, d), post in _splits(t[1]):
        if x in [y for y, _ in (*pre, *post)]:
            continue
        e = elim_var_r(x, d, t[2], t[3])
        if e is not None:
            return (cm_mul(t[0], e[0]), pre + post, e[1], e[2])
    return split_step_r(t)


def _pin_count(t: STm) -> tuple[STm, int]:
    """The term after :func:`elim_step_r` to a fixed point within RG_FUEL, and the steps."""
    for k in range(RG_FUEL):
        t2 = elim_step_r(t)
        if t2 is None:
            return t, k
        t = t2
    return t, RG_FUEL


def sub_tm(x: int, v: int, t: STm) -> STm:
    """``subTm``."""
    w = ip_c(v)
    return (
        t[0],
        t[1],
        tuple(fac_sub(x, w, f) for f in t[2]),
        tuple(fac_sub(x, w, f) for f in t[3]),
    )


def rsplit(r: Sequence[tuple[int, int]], e: Sequence[STm]) -> list[list[STm]]:
    """``rsplit``: each label of ``r`` substituted by each value of its range."""
    if not r:
        return [list(e)]
    (x, d), rest = r[0], r[1:]
    return [c for v in range(d) for c in rsplit(rest, [sub_tm(x, v, t) for t in e])]


def lab_ranges(
    labs: Mapping[int, int], dt: Callable[[int], DMon], ix: Sequence[int], bnd: Sequence[int]
) -> dict[int, int]:
    """The boundary labels ``labOK`` accepts, each with its constant dimension."""
    out = {}
    for lab in bnd:
        i = labs.get(lab, 0)
        m = dt(i)
        if i in ix and dm_const(m):
            out[lab] = m[0]
    return out


def rg_cert(e: Sequence[STm], ranges: Mapping[int, int]) -> RgCert | None:
    """``rgOK``'s certificate for a residual: pins to sum-free terms, the labels some character
    reads with a period beyond their range split over ``ranges``, each case's normal form empty
    or certified by :func:`fz_cert` after :func:`_prune`; None otherwise."""
    if not e:
        return None
    pinned, fuel = [], 1
    for t in e:
        u, k = _pin_count(t)
        if u[1] or k >= RG_FUEL:
            return None
        pinned.append(u)
        fuel = max(fuel, k)
    split: set[int] = set()
    for t in pinned:
        for m, p in (*t[2], *t[3]):
            if not (dm_const(m) and m[0] > 0):
                return None
            for mon, c in p:
                if len(mon) == 1 and mon[0] in ranges and (c * ranges[mon[0]]) % m[0]:
                    split.add(mon[0])
    r = tuple((x, ranges[x]) for x in sorted(split))
    cases = math.prod(d for _x, d in r)
    if not r or cases > RG_CASES or cases * len(pinned) > RG_WORK:
        return None
    certs = []
    for case in rsplit(r, pinned):
        rest = [p for p in (_prune(t) for t in case) if p is not None]
        if not rest:
            certs.append((0, _NO_CERT))
            continue
        got = fz_cert(rest)
        if got is None:
            return None
        certs.append(got)
    return fuel, r, tuple(certs)


def _lean_rg(rc: RgCert) -> str:
    """A range certificate literal."""
    labs = ", ".join(f"({x}, {d})" for x, d in rc[1])
    cases = ", ".join(f"({n}, {_lean_cert(c)})" for n, c in rc[2])
    return f"(({rc[0]}, [{labs}], [{cases}]) : RgCert)"


# geometric sums and pins by the labels' ranges (SymRange)

RX_ROUNDS = 256
"""The most rounds :func:`rx_hints` runs on one residual term."""
RX_STEPS = 32
"""The most steps of one round of :func:`rx_hints` (one ``decide`` each in the export)."""


def geo_data(x: int, big_l: int, lst: Sequence[Fac]) -> tuple[IPoly, IPoly] | None:
    """``geoData``."""
    big_a: IPoly = ()
    big_b: IPoly = ()
    for f in reversed(lst):
        if not fac_has(x, f):
            continue
        if not (dm_const(f[0]) and f[0][0] > 0 and big_l % f[0][0] == 0):
            return None
        lin = ip_lin(x, f[1])
        if lin is None:
            return None
        k = big_l // f[0][0]
        big_a, big_b = ip_add(big_a, ip_smul(k, lin[0])), ip_add(big_b, ip_smul(k, lin[1]))
    return big_a, big_b


def elim_geo1(
    x: int, d: DMon, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    """``elimGeo1``."""
    if not (dm_const(d) and d[0] > 0 and dm_const(f[0]) and f[0][0] > 0):
        return None
    if any(fac_has(x, g) for g in (*pre, f, *post)):
        return None
    k = f[0][0]
    big_l = k * d[0]
    got = geo_data(x, big_l, ps)
    if got is None:
        return None
    big_a, big_b = got
    am = ip_modc(k, big_a)
    if ip_modc(k, f[1]) != am and ip_modc(k, ip_smul(-1, f[1])) != am:
        return None
    lm = (big_l, ())
    return (
        rt_d(d, 2),
        (*pre, (lm, big_a), *post),
        tuple(g for g in ps if not fac_has(x, g)) + ((lm, big_b),),
    )


def _lookup(rg: Sequence[tuple[int, DMon]], u: int) -> DMon | None:
    for v, m in rg:
        if v == u:
            return m
    return None


def ivl(rg: Sequence[tuple[int, DMon]], p: IPoly) -> tuple[int, int] | None:
    """``ivl``."""
    lo = hi = 0
    for m, c in p:
        if not m:
            lo, hi = lo + c, hi + c
        elif len(m) == 1:
            dm = _lookup(rg, m[0])
            if dm is None or not (dm_const(dm) and dm[0] > 0):
                return None
            w = c * (dm[0] - 1)
            lo, hi = lo + min(0, w), hi + max(0, w)
        else:
            return None
    return lo, hi


def elim_xpin1(
    rg: Sequence[tuple[int, DMon]],
    x: int,
    d: DMon,
    pre: tuple[Fac, ...],
    f: Fac,
    post: tuple[Fac, ...],
    ps: tuple[Fac, ...],
) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    """``elimXpin1``."""
    if not (f[0] == d and dm_const(d) and d[0] > 0):
        return None
    lin = ip_lin(x, f[1])
    if lin is None:
        return None
    s = ip_const(lin[0])
    if s not in (1, -1):
        return None
    v = pin_val(s, lin[1])
    iv = ivl(rg, v)
    if iv is None:
        return None
    lo, hi = iv
    k = _ediv(lo, d[0]) * d[0]
    if not hi < k + d[0]:
        return None
    val = ip_add(v, ip_c(-k))
    return (
        of_q(1),
        tuple(fac_sub(x, val, g) for g in (*pre, *post)),
        tuple(fac_sub(x, val, g) for g in ps),
    )


def rg_of(
    r: Sequence[tuple[int, int]], sums: Sequence[tuple[int, DMon]]
) -> tuple[tuple[int, DMon], ...]:
    """``rgOf``."""
    names = {v for v, _ in sums}
    return tuple(sums) + tuple((lab, (d, ())) for lab, d in r if lab not in names)


def x_choice(r: Sequence[tuple[int, int]], t: STm) -> tuple[STm, tuple[int, int, int]] | None:
    """A step of ``elimStepX`` and its hint: a geometric sum, else a pin by ranges, else
    :func:`elim_choice`. A geometric sum (rule 5) is tried only for variables no delta reads; a
    pin (rule 20, the ranges ``rgLoc``) only by deltas reading the variable."""
    dd, pp = _occ(t)
    names = [v for v, _ in t[1]]
    if len(set(names)) == len(names):
        ds, ps = t[2], t[3]
        for i, (x, d) in enumerate(t[1]):
            if x not in pp or x in dd or not (dm_const(d) and d[0] > 0):
                continue
            cache: dict[int, tuple[IPoly, IPoly] | None] = {}
            for j, f in enumerate(ds):
                if not (dm_const(f[0]) and f[0][0] > 0):
                    continue
                k = f[0][0]
                if k not in cache:
                    cache[k] = geo_data(x, k * d[0], ps)
                got = cache[k]
                if got is None:
                    continue
                am = ip_modc(k, got[0])
                if ip_modc(k, f[1]) != am and ip_modc(k, ip_smul(-1, f[1])) != am:
                    continue
                e = elim_geo1(x, d, ds[:j], f, ds[j + 1 :], ps)
                if e is not None:
                    rest = t[1][:i] + t[1][i + 1 :]
                    return (cm_mul(t[0], e[0]), rest, e[1], e[2]), (i, 5, j)
    ds, ps = t[2], t[3]
    counts = Counter(names)
    first: dict[int, DMon] = {}
    for v, m in t[1]:
        first.setdefault(v, m)
    labs: dict[int, DMon] = {}
    for lab, dv in r:
        if lab not in first:
            labs.setdefault(lab, (dv, ()))
    rng = {v: m for v, m in first.items() if counts[v] == 1}
    for j, f in enumerate(ds):
        if not (dm_const(f[0]) and f[0][0] > 0):
            continue
        for i, (x, d) in enumerate(t[1]):
            if d != f[0] or counts[x] != 1 or j not in dd.get(x, ()):
                continue
            e = _xpin_fast(rng, labs, x, d, ds[:j], f, ds[j + 1 :], ps)
            if e is not None:
                rest = t[1][:i] + t[1][i + 1 :]
                return (cm_mul(t[0], e[0]), rest, e[1], e[2]), (i, 20, j)
    return elim_choice(t)


def _xpin_fast(
    rng: Mapping[int, DMon],
    labs: Mapping[int, DMon],
    x: int,
    d: DMon,
    pre: tuple[Fac, ...],
    f: Fac,
    post: tuple[Fac, ...],
    ps: tuple[Fac, ...],
) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    """:func:`elim_xpin1` with ``rg_of`` read through dictionaries (the sums without
    duplicates, ``x`` absent from the pin's value)."""
    lin = ip_lin(x, f[1])
    if lin is None:
        return None
    s = ip_const(lin[0])
    if s not in (1, -1):
        return None
    v = pin_val(s, lin[1])
    lo = hi = 0
    for m, c in v:
        if not m:
            lo, hi = lo + c, hi + c
            continue
        if len(m) != 1 or m[0] == x:
            return None
        dm = rng.get(m[0], labs.get(m[0]))
        if dm is None or not (dm_const(dm) and dm[0] > 0):
            return None
        w = c * (dm[0] - 1)
        lo, hi = lo + min(0, w), hi + max(0, w)
    k = _ediv(lo, d[0]) * d[0]
    if not hi < k + d[0]:
        return None
    val = ip_add(v, ip_c(-k))
    return (
        of_q(1),
        tuple(fac_sub(x, val, g) for g in (*pre, *post)),
        tuple(fac_sub(x, val, g) for g in ps),
    )


def rx_hints(
    fl: Sequence[int],
    r: Sequence[tuple[int, int]],
    t: STm,
    choice: Callable[[Sequence[tuple[int, int]], STm], Any] | None = None,
    steps: int = 0,
    mids: list[list[STm]] | None = None,
) -> tuple[STm | None, list[tuple[int, list[tuple[int, int, int]]]], list[STm]]:
    """``normTmX``'s rounds for ``t`` (``normTmC``'s with ``choice`` :func:`c_choice`): each
    the pivot hint and at most ``steps`` (RX_STEPS when 0) steps; the result (None when it
    vanishes) and the term before each round. With ``mids``, each round's terms after every
    RX_STEPS steps appended to it."""
    choice = choice or x_choice
    steps = steps or RX_STEPS
    hints: list[tuple[int, list[tuple[int, int, int]]]] = []
    terms = [t]
    for _ in range(RX_ROUNDS):
        t1, es = t, []
        cut: list[STm] = []
        for _ in range(steps):
            got = choice(r, t1)
            if got is None:
                break
            t1, h = got
            es.append(h)
            if len(es) % RX_STEPS == 0:
                cut.append(t1)
        t2 = canon_tm(fl, t1)
        if not cut or cut[-1] is not t1:
            cut.append(t1)
        if mids is not None and t2 != t:
            mids.append(cut)
        if t2 == t:
            break
        hints.append((canon_hint(t1), es))
        if t2 is None:
            return None, hints, terms
        t = t2
        terms.append(t)
    return t, hints, terms


RxCert = tuple[
    tuple[tuple[int, int], ...],
    list[list[tuple[int, list[tuple[int, int, int]]]]],
    list[tuple[list[STm], STm | None]],
]


def rx_cert(fl: Sequence[int], e: Sequence[STm], ranges: Mapping[int, int]) -> RxCert | None:
    """``rxNf``'s labels and rounds for a residual whose terms, normalised with the labels'
    ranges, cancel, with each term's path (the term before each round, the result); None
    otherwise."""
    if not e or not ranges:
        return None
    r = tuple(sorted(ranges.items()))
    acc: tuple[STm, ...] = ()
    hs = []
    paths: list[tuple[list[STm], STm | None]] = []
    for t in e:
        got, h, terms = rx_hints(fl, r, t)
        hs.append(h)
        paths.append((terms, got))
        if got is not None:
            acc = tm_add(got, acc)
    if not acc:
        return (r, hs, paths)
    return rx_cert_c(fl, e, r)


def rx_cert_c(fl: Sequence[int], e: Sequence[STm], r: tuple[tuple[int, int], ...]) -> RxCert | None:
    """``rxNfC``'s labels, rounds and paths as :func:`rx_cert`'s, and as a fourth entry each
    round's terms after every RX_STEPS steps (rounds of at most C_STEPS steps), for a residual
    whose terms, normalised with the carry steps of :func:`c_choice`, cancel; None otherwise."""
    acc: tuple[STm, ...] = ()
    hs = []
    paths: list[tuple[list[STm], STm | None]] = []
    cuts: list[list[list[STm]]] = []
    for t in e:
        mids: list[list[STm]] = []
        got, h, terms = rx_hints(fl, r, t, c_choice, C_STEPS, mids)
        hs.append(h)
        paths.append((terms, got))
        cuts.append(mids)
        if got is not None:
            acc = tm_add(got, acc)
    return (r, hs, paths, cuts) if not acc else None  # type: ignore[return-value]


C_STEPS = 4096
"""The most steps of one round of :func:`rx_cert_c` (RX_STEPS steps per ``decide`` in the
export)."""


def carry17(i: int, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...]) -> Fac | None:
    """``carry17``: ``f = [M ∣ p]`` as ``[M ∣ (p - c s q) mod M]`` beside the delta
    ``[m ∣ q]`` at ``i`` of ``pre + post``, ``q = s x + b`` at its pivot ``x``, ``c`` the
    coefficient of ``x`` in ``p``, ``M ∣ c s m``."""
    rest = pre + post
    if i >= len(rest):
        return None
    g = rest[i]
    if not (dm_const(g[0]) and dm_const(f[0]) and f[0][0] > 0):
        return None
    pv = pivot_of(g[1])
    if pv is None:
        return None
    lin = ip_lin(pv[0], f[1])
    if lin is None:
        return None
    c = ip_const(lin[0])
    if c is None or (c * pv[1] * g[0][0]) % f[0][0]:
        return None
    return (f[0], ip_modc(f[0][0], ip_add(f[1], ip_smul(-(c * pv[1]), g[1]))))


def _c17(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 17 at the first delta whose pivot is a sum read by another delta with a coefficient
    that the moduli allow to clear."""
    ds = t[2]
    names = {v for v, _ in t[1]}
    for j, f in enumerate(ds):
        if not (dm_const(f[0]) and f[0][0] > 1):
            continue
        rest = _without(ds, j)
        for i, g in enumerate(rest):
            pv = pivot_of(g[1])
            if pv is None or pv[0] not in names or not fac_has(pv[0], f):
                continue
            got = carry17(i, ds[:j], f, ds[j + 1 :])
            if got is not None and not fac_has(pv[0], got):
                return (t[0], t[1], (*ds[:j], got, *ds[j + 1 :]), t[3]), (i, 17, j)
    return None


def ph_split(t: STm, i: int, j: int) -> STm | None:
    """``phSplit``: the root of unity ``[M ∣ p]`` at ``i`` as ``[M ∣ p₀]`` in place and
    ``[d ∣ p₁ / k]`` appended, ``x`` the sum at ``j`` of constant range ``d``, ``M = k d``, ``p₁``
    the monomials reading ``x``."""
    if i >= len(t[3]) or j >= len(t[1]):
        return None
    g, (x, d) = t[3][i], t[1][j]
    if not (dm_const(d) and d[0] > 0 and dm_const(g[0]) and g[0][0] > 0 and g[0][0] % d[0] == 0):
        return None
    k = g[0][0] // d[0]
    with_x = tuple(m for m in g[1] if x in m[0])
    if any(c % k for _, c in with_x):
        return None
    without = tuple(m for m in g[1] if x not in m[0])
    phs = (*t[3][:i], (g[0], without), *t[3][i + 1 :], (d, tuple((m, c // k) for m, c in with_x)))
    return (t[0], t[1], t[2], phs)


def _c18(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 18 at the first root of unity of a modulus above ``d`` reading a sum ``x`` of range
    ``d`` that no delta reads and that every root reading it allows to split."""
    names = [v for v, _ in t[1]]
    if len(set(names)) != len(names):
        return None
    dd, pp = _occ(t)
    for j, (x, d) in enumerate(t[1]):
        if x in dd or x not in pp or not (dm_const(d) and d[0] > 1):
            continue
        cand = None
        for i, g in enumerate(t[3]):
            if not fac_has(x, g) or g[0] == d:
                continue
            if ph_split(t, i, j) is None:
                cand = None
                break
            if cand is None:
                cand = i
        if cand is not None:
            u = ph_split(t, cand, j)
            if u is not None:
                return u, (cand, 18, j)
    return None


def f_choice(r: Sequence[tuple[int, int]], t: STm) -> tuple[STm, tuple[int, int, int]] | None:
    """A step of ``elimStepF`` and its hint: :func:`x_choice`, else rule 17, else rule 18."""
    for fn in (x_choice, _c17, _c18):
        got: tuple[STm, tuple[int, int, int]] | None = fn(r, t)
        if got is not None:
            return got
    return None


def rx_cert_f(fl: Sequence[int], e: Sequence[STm], ranges: Mapping[int, int]) -> RxCert | None:
    """``rxNfF``'s labels, rounds and paths as :func:`rx_cert`'s, marked by a fourth entry
    ``"F"``, for a residual whose terms, normalised with :func:`f_choice`, cancel; None
    otherwise."""
    if not e or not ranges:
        return None
    r = tuple(sorted(ranges.items()))
    acc: tuple[STm, ...] = ()
    hs = []
    paths: list[tuple[list[STm], STm | None]] = []
    for t in e:
        got, h, terms = rx_hints(fl, r, t, f_choice)
        hs.append(h)
        paths.append((terms, got))
        if got is not None:
            acc = tm_add(got, acc)
    return (r, hs, paths, "F") if not acc else None  # type: ignore[return-value]


# carry steps over the labels' ranges (SymCarry)


def _nvars(p: IPoly) -> int:
    return len(ip_varset(p))


def _without(lst: Sequence[Any], j: int) -> tuple[Any, ...]:
    return tuple(lst[:j]) + tuple(lst[j + 1 :])


def c_fac(
    r: Sequence[tuple[int, int]], t: STm, i: int, rule: int, j: int
) -> tuple[tuple[Fac, ...], tuple[Fac, ...]] | None:
    """``carryFac``: the deltas and roots of unity after step ``(i, rule, j)``, rules 7 to 14,
    at the delta ``j``."""
    ds, ps = t[2], t[3]
    if j >= len(ds):
        return None
    pre, f, post = ds[:j], ds[j], ds[j + 1 :]
    rest = pre + post
    fc = dm_const(f[0]) and f[0][0] > 0
    if rule == 7:
        if i >= len(t[1]) or not fc:
            return None
        x = t[1][i][0]
        lin = ip_lin(x, f[1])
        if lin is None:
            return None
        s = ip_const(lin[0])
        if s not in (1, -1):
            return None
        g = (f[0], ip_sub(ip_v(x), ip_modc(f[0][0], pin_val(s, lin[1]))))
        return (*pre, g, *post), ps
    if rule in (8, 9):
        if i >= len(rest) or rest[i][0] != f[0] or not fc:
            return None
        o = rest[i][1]
        q = ip_add(f[1], ip_smul(1 if rule == 8 else -1, o))
        return (*pre, (f[0], ip_modc(f[0][0], q)), *post), ps
    if rule == 10:
        if i >= len(t[1]):
            return None
        d = t[1][i][1]
        if not (dm_const(d) and d[0] > 0 and dm_const(f[0]) and f[0][0] % d[0] == 0):
            return None
        return (*pre, f, (d, ip_modc(d[0], f[1])), *post), ps
    if rule == 11:
        if i >= len(rest):
            return None
        g = rest[i]
        if not (dm_const(g[0]) and g[0][0] > 0 and fc and g[0][0] % f[0][0] == 0):
            return None
        k = f[0][0]
        fm = ip_modc(k, f[1])
        if ip_modc(k, g[1]) != fm and ip_modc(k, ip_smul(-1, g[1])) != fm:
            return None
        return rest, ps
    if rule == 12:
        rg = rg_u(r, t[1], i, f)
        dy = _lookup(rg, i)
        if dy is None:
            return None
        got = None if f[0] == dy else split_del(rg, i, dy, f)
        if got is None:
            return None
        return (*pre, got[0], got[1], *post), ps
    if rule == 13:
        pv = pivot_of(f[1])
        if i >= len(ps) or pv is None or not fc:
            return None
        g = ps[i]
        if not (dm_const(g[0]) and g[0][0] > 0):
            return None
        x, m, big = pv[0], f[0][0], g[0][0]
        if not all(x not in mon or (c * m) % big == 0 for mon, c in g[1]):
            return None
        return ds, (*ps[:i], fac_sub(x, pin_val(pv[1], pv[2]), g), *ps[i + 1 :])
    if rule == 14:
        return (*pre, (f[0], ip_smul(-1, f[1])), *post), ps
    if rule == 16:
        rg = rg_u(r, t[1], i, f)
        x, dx = i, _lookup(rg, i)
        if dx is None or not (f[0] == dx and dm_const(dx) and dx[0] > 0):
            return None
        lin = ip_lin(x, f[1])
        if lin is None:
            return None
        s = ip_const(lin[0])
        if s not in (1, -1):
            return None
        v = pin_val(s, lin[1])
        iv = ivl(rg, v)
        if iv is None:
            return None
        k = _ediv(iv[0], dx[0]) * dx[0]
        if not iv[1] < k + dx[0]:
            return None
        val = ip_add(v, ip_c(-k))
        return (
            (*(fac_sub(x, val, g) for g in pre), f, *(fac_sub(x, val, g) for g in post)),
            tuple(fac_sub(x, val, g) for g in ps),
        )
    return None


def c_rc(t: STm, i: int, j: int) -> STm | None:
    """``rcStep``: the square roots ``i`` and ``j`` of the coefficient merged when their
    product is a square."""
    c = t[0]
    rc = list(c[3])
    if i >= len(rc) or j >= len(rc):
        return None
    a, b = rc[i], rc[j]
    rc.remove(a)
    if b not in rc or not _isqrt_exact(a * b):
        return None
    rc.remove(b)
    return ((c[0] * math.isqrt(a * b), c[1], c[2], tuple(rc), c[4], c[5]), t[1], t[2], t[3])


def rg_loc(
    r: Sequence[tuple[int, int]], sums: Sequence[tuple[int, DMon]], x: int, f: Fac
) -> list[tuple[int, DMon]]:
    """``rgLoc``: for each variable of ``f`` other than ``x``, in order, its sum when ``sums``
    binds it once, else its label range in ``r``."""
    first: dict[int, DMon] = {}
    counts: dict[int, int] = {}
    for v, m in sums:
        first.setdefault(v, m)
        counts[v] = counts.get(v, 0) + 1
    labs: dict[int, int] = {}
    for lab, dv in r:
        labs.setdefault(lab, dv)
    out: list[tuple[int, DMon]] = []
    for mon, _c in f[1]:
        for u in mon:
            if u == x:
                continue
            if u in first:
                if counts[u] == 1:
                    out.append((u, first[u]))
            elif u in labs:
                out.append((u, (labs[u], ())))
    return out


def rg_u(
    r: Sequence[tuple[int, int]], sums: Sequence[tuple[int, DMon]], y: int, f: Fac
) -> list[tuple[int, DMon]]:
    """``rgU``: the range of ``y``, then :func:`rg_loc` of ``f`` without ``y``."""
    return rg_loc(r, sums, y + 1, ((0, ()), (((y,), 1),))) + rg_loc(r, sums, y, f)


def c_step(r: Sequence[tuple[int, int]], t: STm, i: int, rule: int, j: int) -> STm | None:
    """``elimStepC`` for the rules 7 to 16."""
    if rule == 15:
        return c_rc(t, i, j)
    got = c_fac(r, t, i, rule, j)
    return None if got is None else (t[0], t[1], got[0], got[1])


def _c_try(
    r: Sequence[tuple[int, int]], t: STm, i: int, rule: int, j: int
) -> tuple[STm, tuple[int, int, int]] | None:
    u = c_step(r, t, i, rule, j)
    return None if u is None else (u, (i, rule, j))


def _c_pin(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 7 at a delta of a sum's range that rule 20 then pins."""
    names = [v for v, _ in t[1]]
    for i, (x, d) in enumerate(t[1]):
        if names.count(x) != 1:
            continue
        rest = t[1][:i] + t[1][i + 1 :]
        for j, f in enumerate(t[2]):
            if f[0] != d or not fac_has(x, f):
                continue
            got = _c_try(r, t, i, 7, j)
            if got is None or got[0][2][j] == f:
                continue
            g = got[0][2][j]
            rg = rg_loc(r, rest, x, g)
            if elim_xpin1(rg, x, d, t[2][:j], g, t[2][j + 1 :], t[3]) is not None:
                return got
    return None


def _c_rc(r: Sequence[tuple[int, int]], t: STm) -> Any:
    n = len(t[0][3])
    for i in range(n):
        for j in range(i + 1, n):
            got = _c_try(r, t, i, 15, j)
            if got is not None:
                return got
    return None


def _c_phase(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 13 at a root of unity reading a delta's pivot, the root's modulus not dividing the
    delta's."""
    for j, f in enumerate(t[2]):
        if not (dm_const(f[0]) and f[0][0] > 1):
            continue
        pv = pivot_of(f[1])
        if pv is None:
            continue
        for i, g in enumerate(t[3]):
            if fac_has(pv[0], g) and not dm_dvd(g[0], f[0]):
                got = _c_try(r, t, i, 13, j)
                if got is not None:
                    return got
    return None


def _c_tie(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 14 at a delta whose leading coefficient is half its modulus and whose next one is
    negative."""
    for j, f in enumerate(t[2]):
        if not (dm_const(f[0]) and f[0][0] > 2 and f[0][0] % 2 == 0):
            continue
        k = f[0][0]
        cs = [c for _, c in ip_mods(k, f[1]) if c % k]
        if cs and cs[0] == k // 2:
            nxt = [c for c in cs[1:] if 2 * c not in (k, -k)]
            if nxt and nxt[0] < 0:
                return _c_try(r, t, 0, 14, j)
    return None


def _c_row(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 8 or 9 at the first delta a delta of its modulus thins."""
    for j, f in enumerate(t[2]):
        if not (dm_const(f[0]) and f[0][0] > 1):
            continue
        for i, g in enumerate(_without(t[2], j)):
            if g[0] != f[0]:
                continue
            for rule in (8, 9):
                got = _c_try(r, t, i, rule, j)
                if got is not None and _nvars(got[0][2][j][1]) < _nvars(f[1]):
                    return got
    return None


def _c_absorb(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 11 at a delta that a delta of a larger modulus implies."""
    for j, f in enumerate(t[2]):
        if not (dm_const(f[0]) and f[0][0] > 1):
            continue
        for i, g in enumerate(_without(t[2], j)):
            if dm_const(g[0]) and g[0][0] > f[0][0]:
                got = _c_try(r, t, i, 11, j)
                if got is not None:
                    return got
    return None


def _c_split(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 12 by a label's range."""
    names = {v for v, _ in t[1]}
    labs = [lab for lab, _d in r if lab not in names]
    for j in range(len(t[2])):
        for y in labs:
            got = _c_try(r, t, y, 12, j)
            if got is not None:
                return got
    return None


def _c_lpin(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 16 at a delta's greatest variable of the delta's range read by another factor."""
    for j, f in enumerate(t[2]):
        if not (dm_const(f[0]) and f[0][0] > 0) or not f[1]:
            continue
        x = max(ip_varset(f[1]))
        if not any(fac_has(x, g) for k, g in enumerate((*t[2], *t[3])) if k != j):
            continue
        if _lookup(rg_u(r, t[1], x, f), x) == f[0]:
            got = _c_try(r, t, x, 16, j)
            if got is not None:
                return got
    return None


def _c_weaken(r: Sequence[tuple[int, int]], t: STm) -> Any:
    """Rule 10 at a delta reading a sum of a smaller range with coefficient ±1, when a delta of
    that range then thins the new one."""
    ds = t[2]
    for j, f in enumerate(ds):
        if not (dm_const(f[0]) and f[0][0] > 1):
            continue
        for i, (x, d) in enumerate(t[1]):
            if not (dm_const(d) and 1 < d[0] < f[0][0] and f[0][0] % d[0] == 0):
                continue
            lin = ip_lin(x, f[1])
            if lin is None or ip_const(lin[0]) not in (1, -1):
                continue
            k = d[0]
            q = ip_modc(k, f[1])
            if any(
                g[0] == d and q in (ip_modc(k, g[1]), ip_modc(k, ip_smul(-1, g[1]))) for g in ds
            ):
                continue
            if not any(
                g[0] == d
                and min(_nvars(ip_modc(k, ip_add(q, g[1]))), _nvars(ip_modc(k, ip_sub(q, g[1]))))
                < _nvars(q)
                for g in ds
            ):
                continue
            got = _c_try(r, t, i, 10, j)
            if got is not None:
                return got
    return None


def c_choice(r: Sequence[tuple[int, int]], t: STm) -> tuple[STm, tuple[int, int, int]] | None:
    """A step of ``elimStepC`` and its hint: a geometric sum or a pin by ranges, else rule 7
    before such a pin, else :func:`elim_choice`, else a carry step (rules 15, 13, 14, 8 or 9,
    11, 12, 16, 10)."""
    got = x_choice(r, t)
    if got is not None and got[1][1] in (5, 6, 20):
        return got
    pin: tuple[STm, tuple[int, int, int]] | None = _c_pin(r, t)
    if pin is not None:
        return pin
    if got is not None:
        return got
    for fn in (_c_rc, _c_phase, _c_tie, _c_row, _c_absorb, _c_split, _c_lpin, _c_weaken):
        step: tuple[STm, tuple[int, int, int]] | None = fn(r, t)
        if step is not None:
            return step
    return None


def _lean_rx(rc: RxCert) -> tuple[str, str]:
    """The labels' ranges and the rounds as literals."""
    labs = "[" + ", ".join(f"({x}, {d})" for x, d in rc[0]) + "]"
    terms = ", ".join(
        "["
        + ", ".join(
            f"({_rn(c)}, [" + ", ".join(f"({_rn(i)}, {_rn(k)}, {_rn(j)})" for i, k, j in es) + "])"
            for c, es in rounds
        )
        + "]"
        for rounds in rc[1]
    )
    return labs, f"([{terms}] : List (List RHint))"


# networks


def sum_l(ls: Sequence[int]) -> IPoly:
    acc: IPoly = ()
    for x in reversed(ls):
        acc = ip_add(acc, ip_v(x))
    return acc


def z_legs(d: DMon, s: int, ls: Sequence[int]) -> tuple[Fac, ...]:
    return tuple((d, ip_sub(ip_v(x), ip_v(s))) for x in ls)


def spider(
    k: CMon,
    d: DMon,
    s: int,
    pt: Sequence[tuple[int, CMon]],
    ds: tuple[Fac, ...],
    ps: tuple[Fac, ...],
) -> list[STm]:
    out: list[STm] = [(k, ((s, d),), ds, ps)]
    for j, a in pt:
        delta = (d, ip_sub(ip_v(s), ip_c(j)))
        out.append((cm_mul(k, a), ((s, d),), (delta,) + ds, ps))
        out.append((cm_mul(k, of_q(-1)), ((s, d),), (delta,) + ds, ps))
    return out


def w_zeros(d: DMon, os: Sequence[int]) -> tuple[Fac, ...]:
    return tuple((d, ip_v(o)) for o in os)


def w_s(d: DMon, i: int, acc: tuple[Fac, ...], os: Sequence[int]) -> list[STm]:
    out: list[STm] = []
    for p, o in enumerate(os):
        out.append(
            (of_q(1), (), acc + ((d, ip_sub(ip_v(o), ip_v(i))),) + w_zeros(d, os[p + 1 :]), ())
        )
        acc = acc + ((d, ip_v(o)),)
    return out


class PhRow(list):  # type: ignore[type-arg]
    """A phase vector's stored entries, with its rule ``(DMon, coefficients)`` or None."""

    rule: tuple[DMon, tuple[int, ...]] | None = None


def r_poly(s: int, cs: Sequence[int]) -> IPoly:
    """``rPoly``."""
    if not cs:
        return ()
    return ip_add(ip_c(cs[0]), ip_mul(ip_v(s), r_poly(s, cs[1:])))


def r_f(s: int, rule: tuple[DMon, tuple[int, ...]] | None) -> tuple[Fac, ...]:
    """``rF``."""
    if rule is None or not rule[1]:
        return ()
    return ((rule[0], r_poly(s, rule[1])),)


CP_MARK: DMon = (0, ())
"""The modulus of a rule marking a compact table (``cpData``)."""
CP_BASE = 3000000
"""``cpX s = CP_BASE + 2 s``."""

CpData = tuple[int, int, int, int]
CP_ROW_MIN = 256
"""The fewest entries of a compact table stored as ``cpRow`` instead of entry by entry."""


def cp_data(r: tuple[DMon, tuple[int, ...]] | None) -> CpData | None:
    """``cpData``: kind, modulus, coefficient and digit range of a marking rule."""
    if r is None or r[0] != CP_MARK or len(r[1]) != 4:
        return None
    k, m, c, d = r[1]
    return max(k, 0), max(m, 0), c, max(d, 0)


def cp_row(big_d: int, q: CpData) -> list[tuple[int, CMon, sp.Expr]]:
    """``cpRow``: the entries off multiples of the modulus, each phase with its value."""
    out = []
    for j in range(big_d):
        n = cp_num(q, j)
        if _emod(n, q[1]):
            out.append((j, ph_c(Fraction(n, q[1]), ()), sp.Rational(n, q[1])))
    return out


def cp_num(q: CpData, j: int) -> int:
    """``cpNum``."""
    if q[0] == 1:
        return q[2] * j
    d = q[3]
    return q[2] * (j // d if d else 0) * (j % d if d else j)


def pt_ok(lst: Sequence[int], c: int, pt: Sequence[tuple[int, CMon]]) -> bool:
    """``ptOK``."""
    keys = [j for j, _ in pt]
    if c <= 0 or len(set(keys)) != len(keys):
        return False
    if not all(m == ph_c(Fraction(lst[j] if j < len(lst) else 0, c), ()) for j, m in pt):
        return False
    return all(_emod(v, c) == 0 or i in keys for i, v in enumerate(lst))


def cp_terms(
    k: CMon, d: DMon, s: int, q: CpData, ds: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> list[STm]:
    """``cpTerms``."""
    if q[0] == 1:
        return [(k, ((s, d),), ds, ps + (((q[1], ()), ip_smul(q[2], ip_v(s))),))]
    x, y = CP_BASE + 2 * s, CP_BASE + 2 * s + 1
    delta = (d, ip_sub(ip_sub(ip_v(s), ip_smul(q[3], ip_v(x))), ip_v(y)))
    phase = ((q[1], ()), ip_smul(q[2], ip_mul(ip_v(x), ip_v(y))))
    return [(k, ((s, d), (x, (q[3], ())), (y, (q[3], ()))), ds + (delta,), ps + (phase,))]


def cp_chk(
    d: DMon,
    pt: Sequence[tuple[int, CMon]],
    q: CpData,
    s: int,
    ds: tuple[Fac, ...],
    ps: tuple[Fac, ...],
) -> bool:
    """``cpChk``."""
    if not (dm_const(d) and d[0] > 0 and pt_ok([cp_num(q, j) for j in range(d[0])], q[1], pt)):
        return False
    if q[0] == 1:
        return True
    x = CP_BASE + 2 * s
    return q[3] * q[3] == d[0] and not any(fac_has(x, f) or fac_has(x + 1, f) for f in (*ds, *ps))


def spider_c(
    k: CMon,
    d: DMon,
    s: int,
    pt: Sequence[tuple[int, CMon]],
    r: tuple[DMon, tuple[int, ...]] | None,
    ds: tuple[Fac, ...],
    ps: tuple[Fac, ...],
) -> list[STm]:
    """``spiderC``."""
    q = cp_data(r)
    if q is not None and cp_chk(d, pt, q, s, ds, ps):
        return cp_terms(k, d, s, q, ds, ps)
    return spider(k, d, s, pt, ds, ps + r_f(s, r))


def tr_node(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    s: int,
    kind: str,
    os: Sequence[int],
    is_: Sequence[int],
    dim: int,
    ph: int,
) -> list[STm] | None:
    d = dt(dim)
    rule = getattr(pt(ph), "rule", None)
    if kind == "Z":
        return spider_c(of_q(1), d, s, pt(ph), rule, z_legs(d, s, (*os, *is_)), ())
    if kind == "X":
        poly = ip_mul(ip_v(s), ip_sub(sum_l(os), sum_l(is_)))
        return spider_c(rt_d(d, -(len(os) + len(is_))), d, s, pt(ph), rule, (), ((d, poly),))
    if kind == "F" and len(os) == 1 and len(is_) == 1:
        return [(rt_d(d, -1), (), (), ((d, ip_mul(ip_v(os[0]), ip_v(is_[0]))),))]
    if kind == "T" and len(os) == 1 and len(is_) == 1:
        o, i = os[0], is_[0]
        return [
            (of_q(1), (), ((d, ip_v(o)),), ()),
            (of_q(1), (), ((d, ip_sub(ip_v(o), ip_v(i))),), ()),
            (of_q(-1), (), ((d, ip_v(o)), (d, ip_v(i))), ()),
        ]
    if kind == "Ti" and len(os) == 1 and len(is_) == 1:
        o, i = os[0], is_[0]
        return [
            (of_q(1), (), ((d, ip_sub(ip_v(o), ip_v(i))),), ()),
            (of_q(-1), (), ((d, ip_v(o)),), ()),
            (of_q(1), (), ((d, ip_v(o)), (d, ip_v(i))), ()),
        ]
    if kind == "W" and len(is_) == 1:
        i = is_[0]
        return w_s(d, i, (), os) + [(of_q(1 - len(os)), (), ((d, ip_v(i)),) + w_zeros(d, os), ())]
    if kind == "I" and len(os) == 1 and len(is_) == 1:
        return [(of_q(1), (), ((d, ip_sub(ip_v(os[0]), ip_v(is_[0]))),), ())]
    if (BS_NODES or _BS_ON) and kind == "B" and len(os) == 1 and len(is_) == 2:
        return bs_term(d, dt(ph), os[0], is_[0], is_[1])
    if (BS_NODES or _BS_ON) and kind == "S" and len(os) == 2 and len(is_) == 1:
        return bs_term(d, dt(ph), is_[0], os[0], os[1])
    return None


def bs_term(ds: DMon, dt_: DMon, k: int, a: int, b: int) -> list[STm] | None:
    """``bsTerm``."""
    if dt_[1]:
        return None
    poly = ip_sub(ip_sub(ip_v(k), ip_mul(ip_c(dt_[0]), ip_v(a))), ip_v(b))
    return [(of_q(1), (), ((dm_mul(ds, dt_), poly),), ())]


def leg_dms(
    dt: Callable[[int], DMon], kind: str, os: Sequence[int], is_: Sequence[int], dim: int, ph: int
) -> list[DMon]:
    """``legDMs``."""
    if kind == "B" and len(os) == 1 and len(is_) == 2:
        return [dm_mul(dt(dim), dt(ph)), dt(dim), dt(ph)]
    if kind == "S" and len(os) == 2 and len(is_) == 1:
        return [dt(dim), dt(ph), dm_mul(dt(dim), dt(ph))]
    return [dt(dim)] * (len(os) + len(is_))


def ex_mul(a: Sequence[STm], b: Sequence[STm]) -> list[STm]:
    return [tm_mul(s, t) for s in a for t in b]


def ex_mul_ok(a: Sequence[STm], b: Sequence[STm]) -> bool:
    return all(mul_ok(s, t) for s in a for t in b)


def tr_nodes(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    internal: Sequence[int],
    pre: tuple[int, ...],
    i: int,
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    budget: list[int] | None = None,
) -> list[STm] | None:
    """``trNodesB``, iterated from the last node; :class:`NetUnsupported` once the terms
    normalised exceed ``budget[0]``."""
    steps: list[tuple[list[STm] | None, tuple[int, ...], tuple[int, ...]]] = []
    for k, node in enumerate(nodes):
        _kind, os, is_, _dim, _ph = node
        legs = (*os, *is_)
        steps.append((tr_node(dt, pt, OFF + i + k, *node), legs, pre))
        pre = pre + tuple(legs)
    b: list[STm] | None = [(of_q(1), (), (), ())]
    for a, legs, before in reversed(steps):
        if a is None or b is None or not ex_mul_ok(a, b):
            return None
        if budget is not None:
            budget[0] -= len(a) * len(b)
            if budget[0] < 0:
                raise NetUnsupported(f"the normaliser's expansion exceeds {NORM_TERMS} terms")
        first = tuple(
            (lab, lab_dm(dt, labs, lab)) for lab in internal if lab in legs and lab not in before
        )
        b = list(nf_ex(fl, [(t[0], first + t[1], t[2], t[3]) for t in ex_mul(a, b)]))
    return b


def tr_nodes_full(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    i: int,
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
) -> list[STm] | None:
    """``trNodesF``, iterated from the last node."""
    parts = [tr_node(dt, pt, OFF + i + k, *node) for k, node in enumerate(nodes)]
    b: list[STm] | None = [(of_q(1), (), (), ())]
    for a in reversed(parts):
        if a is None or b is None or not ex_mul_ok(a, b):
            return None
        b = ex_mul(a, b)
    return b


def tr_net_full(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    internal: Sequence[int],
) -> list[STm] | None:
    """``trNetF``."""
    e = tr_nodes_full(dt, pt, 0, nodes)
    if e is None:
        return None
    sums = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal)
    return [(t[0], sums + t[1], t[2], t[3]) for t in e]


def tr_net_id(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    internal: Sequence[int],
) -> list[STm] | None:
    """``trNetG`` with the identity normaliser: each node's terms times the tail's, the labels
    first met at the node summed there."""
    pres: list[tuple[int, ...]] = []
    pre: tuple[int, ...] = ()
    for node in nodes:
        pres.append(pre)
        pre = pre + (*node[1], *node[2])
    b: list[STm] = [(of_q(1), (), (), ())]
    for k in range(len(nodes) - 1, -1, -1):
        node = nodes[k]
        a = tr_node(dt, pt, OFF + k, *node)
        if a is None or not ex_mul_ok(a, b):
            return None
        legs = (*node[1], *node[2])
        first = tuple(
            (lab, lab_dm(dt, labs, lab)) for lab in internal if lab in legs and lab not in pres[k]
        )
        b = [(t[0], first + t[1], t[2], t[3]) for t in ex_mul(a, b)]
    used = {lab for n in nodes for lab in (*n[1], *n[2])}
    sums = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal if lab not in used)
    return [(t[0], sums + t[1], t[2], t[3]) for t in b]


Node = tuple[str, Sequence[int], Sequence[int], int, int]


def fold_bzs(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    n1: Node,
    n2: Node,
    n3: Node,
) -> tuple[STm, int, int] | None:
    """``foldBZS``: the term ``[d ∣ a' - a] [d ∣ b' - b] ω_M^(c a b)`` of ``B (a, b) → k``,
    ``Z k → k'`` with a compact table of kind 2 on ``d²`` and ``S k' → (a', b')``, with ``k``,
    ``k'``."""
    if not (len(n1[1]) == 1 and len(n1[2]) == 2 and len(n2[1]) == 1 and len(n2[2]) == 1):
        return None
    if not (len(n3[1]) == 2 and len(n3[2]) == 1):
        return None
    q = cp_data(getattr(pt(n2[4]), "rule", None))
    if q is None:
        return None
    (k,), (a, b), (k1,), (k2,), (a1, b1), (k3,) = n1[1], n1[2], n2[1], n2[2], n3[1], n3[2]
    d, big = dt(n1[3]), dt(n2[3])
    labels = [k, k1, a, b, a1, b1]
    if not (
        (n1[0], n2[0], n3[0]) == ("B", "Z", "S")
        and k2 == k
        and k3 == k1
        and len(set(labels)) == len(labels)
        and dt(n1[4]) == d
        and dt(n3[3]) == d
        and dt(n3[4]) == d
        and dm_const(d)
        and dm_const(big)
        and big[0] == d[0] * d[0]
        and q[0] == 2
        and q[3] == d[0]
        and pt_ok([cp_num(q, j) for j in range(big[0])], q[1], pt(n2[4]))
    ):
        return None
    dels = ((d, ip_sub(ip_v(a1), ip_v(a))), (d, ip_sub(ip_v(b1), ip_v(b))))
    phs = (((q[1], ()), ip_smul(q[2], ip_mul(ip_v(a), ip_v(b)))),)
    return (of_q(1), (), dels, phs), k, k1


def fold_ok(
    internal: Sequence[int], pre: Sequence[int], k: int, k1: int, rest: Sequence[Node]
) -> bool:
    """``foldOK``."""
    later = {lab for n in rest for lab in (*n[1], *n[2])}
    inner = [lab for lab in internal if lab in (k, k1)]
    return (
        k not in pre
        and k1 not in pre
        and k not in later
        and k1 not in later
        and inner in ([k, k1], [k1, k])
    )


def fold_steps(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    nodes: Sequence[Node],
    internal: Sequence[int],
) -> list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[STm, int, int] | None]]:
    """``trNodesK``'s steps from the first node: start, labels before, labels read, the folded
    block or None for one node."""
    out: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[STm, int, int] | None]] = []
    pre: tuple[int, ...] = ()
    i = 0
    while i < len(nodes):
        got = None
        if i + 2 < len(nodes):
            got = fold_bzs(dt, pt, nodes[i], nodes[i + 1], nodes[i + 2])
            if got is not None and not fold_ok(internal, pre, got[1], got[2], nodes[i + 3 :]):
                got = None
        span = 3 if got is not None else 1
        legs = tuple(lab for n in nodes[i : i + span] for lab in (*n[1], *n[2]))
        out.append((i, pre, legs, got))
        pre = pre + legs
        i += span
    return out


def tr_net_k(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    nodes: Sequence[Node],
    internal: Sequence[int],
) -> list[STm] | None:
    """``trNetK``: :func:`tr_net_id` with each block of :func:`fold_steps` one term."""
    b: list[STm] = [(of_q(1), (), (), ())]
    for i, pre, legs, got in reversed(fold_steps(dt, pt, nodes, internal)):
        if got is None:
            a = tr_node(dt, pt, OFF + i, *nodes[i])
            skip: tuple[int, ...] = ()
        else:
            a, skip = [got[0]], (got[1], got[2])
        if a is None or not ex_mul_ok(a, b):
            return None
        first = tuple(
            (lab, lab_dm(dt, labs, lab))
            for lab in internal
            if lab in legs and lab not in pre and lab not in skip
        )
        b = [(t[0], first + t[1], t[2], t[3]) for t in ex_mul(a, b)]
    used = {lab for n in nodes for lab in (*n[1], *n[2])}
    sums = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal if lab not in used)
    return [(t[0], sums + t[1], t[2], t[3]) for t in b]


def lab_dm(dt: Callable[[int], DMon], labs: dict[int, int], lab: int) -> DMon:
    return dt(labs.get(lab, 0))


def tr_net(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    internal: Sequence[int],
) -> list[STm] | None:
    e = tr_nodes(fl, dt, pt, labs, internal, (), 0, nodes, [NORM_TERMS])
    if e is None:
        return None
    used = {lab for n in nodes for lab in (*n[1], *n[2])}
    sums = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal if lab not in used)
    return [(t[0], sums + t[1], t[2], t[3]) for t in e]


def node_chk(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    dim: int,
    ph: int,
) -> bool:
    d = dt(dim)
    return d[0] != 0 and all(j < dm_den(d, lambda i: flv(fl, i)) for j, _ in pt(ph))


def net_chk(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    ix: Sequence[int],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    internal: Sequence[int],
    bnd: Sequence[int],
) -> bool:
    for kind, os, is_, dim, ph in nodes:
        if not node_chk(fl, dt, pt, dim, ph) or dim not in ix:
            return False
        if kind in ("B", "S") and ph not in ix:
            return False
        for lab in (*os, *is_):
            if not (lab < OFF and (lab in bnd or lab in internal) and labs.get(lab, 0) in ix):
                return False
        for lab, dm in zip((*os, *is_), leg_dms(dt, kind, os, is_, dim, ph), strict=False):
            if lab_dm(dt, labs, lab) != dm:
                return False
    return all(labs.get(lab, 0) in ix for lab in internal)


def ex_neg(e: Sequence[STm]) -> list[STm]:
    return [with_q(t, -t[0][0]) for t in e]


def ex_const(e: Sequence[STm]) -> bool:
    return all(not t[1] and not t[2] and not tm_vars(t) for t in e)


def _full_terms(
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
) -> int:
    """The term count of ``tr_nodes_full`` on ``nodes``: the product of each node's count."""
    total = 1
    for i, node in enumerate(nodes):
        terms = tr_node(dt, pt, OFF + i, *node)
        total *= 0 if terms is None else len(terms)
    return total


def sym_check(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    ix: Sequence[int],
    n0: Any,
    n1: Any,
    s0: Sequence[STm],
    s1: Sequence[STm],
    fz: list[tuple[int, FZCert]] | None = None,
    rg: list[RgCert] | None = None,
    rx: list[RxCert] | None = None,
) -> bool:
    """``symCheck``; a network is ``(nodes, internal, boundary)``. With ``fz``, a sum-free
    residual whose characters vanish (:func:`fz_cert`) also passes (``symCheckF``), its
    modulus and certificate appended to ``fz``; with ``rg``, failing that, a residual
    vanishing on the labels' ranges (:func:`rg_cert`, ``symCheckR``), its certificate
    appended to ``rg``; with ``rx``, failing that, a residual cancelling under the labels'
    ranges (:func:`rx_cert`), its labels and rounds appended to ``rx``."""
    if sorted(n0[2]) != sorted(n1[2]):
        return False
    if not (BS_NODES or _BS_ON) and any(n[0] in ("B", "S") for n in (*n0[0], *n1[0])):
        return False
    if not (net_chk(fl, dt, pt, labs, ix, *n0) and net_chk(fl, dt, pt, labs, ix, *n1)):
        return False
    if not (ex_const(s0) and ex_const(s1)):
        return False
    if rx is not None:
        trx = tr_net_k if _FOLD else tr_net_id
        a0 = trx(dt, pt, labs, n0[0], n0[1])
        a1 = trx(dt, pt, labs, n1[0], n1[1])
        neg = ex_neg(s1)
        if a0 is None or a1 is None or not (ex_mul_ok(s0, a0) and ex_mul_ok(neg, a1)):
            return False
        cert = rx_cert_f if _FOLD else rx_cert
        got_x = cert(fl, ex_mul(s0, a0) + ex_mul(neg, a1), lab_ranges(labs, dt, ix, n0[2]))
        if got_x is None:
            return False
        rx.append(got_x)
        return True
    for tr in (tr_net, tr_net_full):
        if tr is tr_net_full and max(_full_terms(dt, pt, n[0]) for n in (n0, n1)) > FULL_TERMS:
            continue
        e0 = tr(fl, dt, pt, labs, n0[0], n0[1])
        e1 = tr(fl, dt, pt, labs, n1[0], n1[1])
        if e0 is None or e1 is None:
            continue
        neg = ex_neg(s1)
        if not (ex_mul_ok(s0, e0) and ex_mul_ok(neg, e1)):
            continue
        rest = nf_ex(fl, ex_mul(s0, e0) + ex_mul(neg, e1))
        if not rest:
            return True
        if fz is not None:
            got = fz_cert(rest)
            if got is not None:
                fz.append(got)
                return True
        if rg is not None:
            got_r = rg_cert(rest, lab_ranges(labs, dt, ix, n0[2]))
            if got_r is not None:
                rg.append(got_r)
                return True
    return False


def tr_steps(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    internal: Sequence[int],
) -> list[list[STm]] | None:
    """``trNodesB`` on every suffix of ``nodes``: entry ``k`` the expression of ``nodes[k:]``
    after the labels of ``nodes[:k]``."""
    pres: list[tuple[int, ...]] = []
    pre: tuple[int, ...] = ()
    for node in nodes:
        pres.append(pre)
        pre = pre + (*node[1], *node[2])
    b: list[STm] = [(of_q(1), (), (), ())]
    out = [b]
    for k in range(len(nodes) - 1, -1, -1):
        node = nodes[k]
        a = tr_node(dt, pt, OFF + k, *node)
        if a is None or not ex_mul_ok(a, b):
            return None
        legs = (*node[1], *node[2])
        first = tuple(
            (lab, lab_dm(dt, labs, lab)) for lab in internal if lab in legs and lab not in pres[k]
        )
        b = list(nf_ex(fl, [(t[0], first + t[1], t[2], t[3]) for t in ex_mul(a, b)]))
        out.append(b)
    out.reverse()
    return out


def piv_pass_b(
    m: int | None, ds: tuple[Fac, ...], ps: tuple[Fac, ...]
) -> tuple[tuple[Fac, ...], tuple[Fac, ...], int]:
    """``pivPassM m ds ps`` (every step when ``m`` is ``None``), with the mask of the steps that
    changed a factor."""
    la: list[Fac] = []
    va: list[frozenset[int]] = []
    lb, lp = list(ds), list(ps)
    vb = [ip_varset(g[1]) for g in lb]
    vp = [ip_varset(g[1]) for g in lp]

    def run(st: tuple[Any, ...], fs: list[Fac], vs: list[frozenset[int]], start: int) -> bool:
        hit = False
        for j in range(start, len(fs)):
            if st[0] and st[2] not in vs[j]:
                continue
            g = _apply(st, fs[j])
            if g != fs[j]:
                fs[j] = g
                vs[j] = ip_varset(g[1])
                hit = True
        return hit

    used = 0
    for i in range(len(lb)):
        f, vf = lb[i], vb[i]
        st = _step(f) if m is None or (m >> i) & 1 else None
        if st is not None:
            hit = run(st, la, va, 0)
            hit = run(st, lb, vb, i + 1) or hit
            hit = run(st, lp, vp, 0) or hit
            if hit:
                used |= 1 << i
        la.append(f)
        va.append(vf)
    return tuple(la), tuple(lp), used


def canon_hint(t: STm) -> int:
    """The hint ``canonTmH`` takes for ``t``: the mask of the steps of the full pass that change
    a factor, or every step when that mask does not reproduce it."""
    r = ren_list(t, 0, bvs(t))
    full = piv_pass(len(r[2]), (), r[2], r[3])
    *_, used = piv_pass_b(None, r[2], r[3])
    if piv_pass_b(used, r[2], r[3])[:2] == full:
        return used
    return (1 << len(r[2])) - 1


def elim_choice(t: STm) -> tuple[STm, tuple[int, int, int]] | None:
    """``elim_step`` with its choice as ``elimStepH`` reads it: sum index, rule (0 unused,
    1 delta, 2 characters, 3 gcd, 4 split) and delta index. Only the deltas reading a variable
    are tried for it (a delta without it pins, divides and splits nothing)."""
    dd, pp = _occ(t)
    names = [y for y, _ in t[1]]
    ds, ps = t[2], t[3]
    for i, (x, d) in enumerate(t[1]):
        if names.count(x) > 1:
            continue
        js = dd.get(x, [])
        e, r, j = None, 0, 0
        if not js and x not in pp:
            e = elim_unused(x, d, ds, ps)
        if e is None:
            mods = [ds[k][0] for k in js]
            if x in pp:
                mods += [g[0] for g in ps if fac_has(x, g)]
            for j in js:
                fm = ds[j][0]
                if not all(dm_dvd(m, fm) for m in mods):
                    continue
                e = elim_delta1(x, d, ds[:j], ds[j], ds[j + 1 :], ps)
                if e is not None:
                    r = 1
                    break
        if e is None and not js:
            e, r, j = elim_char(x, d, ds, ps), 2, 0
        if e is None and len(js) == 1 and x not in pp:
            j = js[0]
            e = elim_gcd1(x, d, ds[:j], ds[j], ds[j + 1 :], ps)
            r = 3
        if e is not None:
            return (cm_mul(t[0], e[0]), t[1][:i] + t[1][i + 1 :], e[1], e[2]), (i, r, j)
    for j, f in enumerate(ds):
        vs = ip_varset(f[1])
        for i, (y, d) in enumerate(t[1]):
            if y not in vs:
                continue
            got = split_del(t[1], y, d, f)
            if got is not None:
                return (t[0], t[1], (*ds[:j], got[0], got[1], *ds[j + 1 :]), t[3]), (i, 4, j)
    return None


def _occ(t: STm) -> tuple[dict[int, list[int]], set[int]]:
    """The deltas reading each variable, in order, and the variables the roots of unity read."""
    dd: dict[int, list[int]] = {}
    for j, f in enumerate(t[2]):
        for x in ip_varset(f[1]):
            dd.setdefault(x, []).append(j)
    pp: set[int] = set()
    for f in t[3]:
        pp |= ip_varset(f[1])
    return dd, pp


def elim_hints(n: int, t: STm) -> tuple[STm, list[tuple[int, int, int]]]:
    """``elim_all`` and the choice of each elimination."""
    hints = []
    for _ in range(n):
        found = elim_choice(t)
        if found is None:
            break
        t, h = found
        hints.append(h)
    return t, hints


def norm_hints(fl: Sequence[int], t: STm) -> list[tuple[int, list[tuple[int, int, int]]]]:
    """``norm_tm`` at ``ROUNDS`` with the hints of each round: the pivot hint and the
    eliminations; the rounds from the first one that keeps the term left out."""
    hints = []
    for _ in range(ROUNDS):
        t1, es = elim_hints(len(t[1]) + 1, t)
        assert t1 == elim_all(len(t[1]) + 1, t)
        t2 = canon_tm(fl, t1)
        if t2 == t:
            break
        hints.append((canon_hint(t1), es))
        if t2 is None:
            break
        t = t2
    return hints


def _mask(labels: Iterable[int]) -> int:
    """The bit mask with bit ``l`` set for each label ``l`` (``maskOf``)."""
    m = 0
    for lab in labels:
        m |= 1 << lab
    return m


THint = tuple[int, int, list[tuple[int, list[tuple[int, int, int]]]]]


def ilv_mask(a: Sequence[Fac], b: Sequence[Fac], out: Sequence[Fac]) -> int | None:
    """The mask with which ``ilv`` interleaves ``a`` and ``b`` into ``out``, if one does."""
    i = j = 0
    m = 0
    for pos, g in enumerate(out):
        if i < len(a) and a[i] == g:
            m |= 1 << pos
            i += 1
        elif j < len(b) and b[j] == g:
            j += 1
        else:
            return None
    return m if i == len(a) and j == len(b) else None


def term_hint(fl: Sequence[int], t: STm) -> THint:
    """``normTmK``'s hint for ``t``: the mask of the deltas set aside, those without a summed
    variable that the full normalisation keeps, the mask interleaving them back, and the
    rounds of the rest; empty masks when that does not give the full result."""
    key = tuple(fl)
    full = _norm_cached(key, t)
    if full is not None:
        sv = {x for x, _ in t[1]}
        kept = set(full[2])
        idx = [j for j, g in enumerate(t[2]) if g in kept and not (ip_varset(g[1]) & sv)]
        if idx:
            iset = set(idx)
            keep = tuple(t[2][j] for j in idx)
            tw = (t[0], t[1], tuple(g for j, g in enumerate(t[2]) if j not in iset), t[3])
            r = _norm_cached(key, tw)
            if r is not None and (r[0], r[1], r[3]) == (full[0], full[1], full[3]):
                rv = {x for x, _ in r[1]}
                m = ilv_mask(keep, r[2], full[2])
                if m is not None and not any(ip_varset(g[1]) & rv for g in keep):
                    return _mask(idx), m, norm_hints(fl, tw)
    return 0, 0, norm_hints(fl, t)


def tr_hints(
    fl: Sequence[int],
    dt: Callable[[int], DMon],
    pt: Callable[[int], Sequence[tuple[int, CMon]]],
    labs: dict[int, int],
    nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
    internal: Sequence[int],
    steps: Sequence[Sequence[STm]],
) -> list[list[THint]]:
    """For node ``k``, the hints of each term of its product with ``steps[k + 1]``."""
    pres: list[tuple[int, ...]] = []
    pre: tuple[int, ...] = ()
    for node in nodes:
        pres.append(pre)
        pre = pre + (*node[1], *node[2])
    out = []
    for k, node in enumerate(nodes):
        a = tr_node(dt, pt, OFF + k, *node)
        assert a is not None
        legs = (*node[1], *node[2])
        first = tuple(
            (lab, lab_dm(dt, labs, lab)) for lab in internal if lab in legs and lab not in pres[k]
        )
        prods = [(t[0], first + t[1], t[2], t[3]) for t in ex_mul(a, steps[k + 1])]
        out.append([term_hint(fl, t) for t in prods])
    return out


def _rn(x: int) -> str:
    """A natural number as a raw literal."""
    return f"nat_lit {x}"


def _lean_th(h: THint) -> str:
    """A term hint on raw literals."""
    rounds = ", ".join(
        f"({_rn(c)}, [" + ", ".join(f"({_rn(i)}, {_rn(r)}, {_rn(j)})" for i, r, j in es) + "])"
        for c, es in h[2]
    )
    return f"({_rn(h[0])}, {_rn(h[1])}, [{rounds}])"


def _rz(x: int) -> str:
    """An integer through its constructors on raw literals."""
    return f"Int.ofNat ({_rn(x)})" if x >= 0 else f"Int.negSucc ({_rn(-x - 1)})"


def _rq(q: Fraction) -> str:
    """A rational as ``mkRat`` of raw literals."""
    return f"mkRat ({_rz(q.numerator)}) ({_rn(q.denominator)})"


def _rdm(m: DMon) -> str:
    return f"⟨{_rn(m[0])}, [{', '.join(_rn(e) for e in m[1])}]⟩"


def _lean_cm(c: CMon) -> str:
    gc = ", ".join(f"({_rz(g)}, {_rdm(d)})" for g, d in c[5])
    return (
        f"⟨{_rq(c[0])}, {_rq(c[1])}, [{', '.join(_rz(x) for x in c[2])}], "
        f"[{', '.join(_rn(x) for x in c[3])}], [{', '.join(_rz(x) for x in c[4])}], [{gc}]⟩"
    )


def _lean_ip(p: IPoly) -> str:
    return "[" + ", ".join(f"([{', '.join(_rn(x) for x in m)}], {_rz(c)})" for m, c in p) + "]"


def _lean_fac(f: Fac) -> str:
    return f"({_rdm(f[0])}, {_lean_ip(f[1])})"


def _lean_facs(fs: Sequence[Fac], name: Callable[[Fac], str] | None = None) -> str:
    return "[" + ", ".join((name or _lean_fac)(f) for f in fs) + "]"


def _lean_tm(t: STm, name: Callable[[Fac], str] | None = None) -> str:
    sums = ", ".join(f"({_rn(x)}, {_rdm(d)})" for x, d in t[1])
    return f"⟨{_lean_cm(t[0])}, [{sums}], {_lean_facs(t[2], name)}, {_lean_facs(t[3], name)}⟩"


def _lean_ex(e: Sequence[STm], name: Callable[[Fac], str] | None = None) -> str:
    """An expression literal; with ``name``, each factor written as ``name`` gives it."""
    return "([" + ", ".join(_lean_tm(t, name) for t in e) + "] : List STm)"


# the export


def _dmon(e: sp.Expr, syms: Sequence[sp.Symbol]) -> DMon:
    """A dimension expression as ``c Π d_i^e_i``."""
    e = sp.expand(sp.sympify(e))
    if e.has(sp.pi):
        raise NetUnsupported("a phase that is not a rational number of turns (a real angle)")
    c, rest = e.as_coeff_Mul()
    if not c.is_Integer or int(c) <= 0:
        raise NetUnsupported(f"the dimension {e} is not a monomial")
    exps = [0] * len(syms)
    for f in sp.Mul.make_args(rest):
        if f == 1:
            continue
        if f in syms:
            exps[syms.index(f)] += 1
            continue
        base, k = f.as_base_exp()
        if base not in syms or not k.is_Integer or int(k) < 0:
            raise NetUnsupported(f"the dimension {e} is not a monomial")
        exps[syms.index(base)] += int(k)
    return (int(c), trim(exps))


def _lean_dm(m: DMon) -> str:
    return f"⟨{m[0]}, [{', '.join(map(str, m[1]))}]⟩"


def _lean_q(q: Fraction) -> str:
    if q.denominator == 1:
        return f"({q.numerator} : ℚ)"
    return f"(({q.numerator} : ℚ) / {q.denominator})"


def _lean_ints(xs: Sequence[int]) -> str:
    return "[" + ", ".join(f"({x} : ℤ)" if x < 0 else str(x) for x in xs) + "]"


class _Atoms:
    """Phase values split as a rational plus integer multiples of atoms ``b / L``; in a phase
    entry, integer multiples of integer symbols drop out."""

    def __init__(self, values: Sequence[sp.Expr], phases: Sequence[sp.Expr] = ()) -> None:
        dens: dict[sp.Expr, int] = {}
        for v, drop in [(v, False) for v in values] + [(v, True) for v in phases]:
            for b, c in self.split(v, drop)[1].items():
                dens[b] = math.lcm(dens.get(b, 1), int(c.q))
        self.bases = sorted(dens, key=str)
        self.scale = [dens[b] for b in self.bases]

    @staticmethod
    def split(
        v: sp.Expr, drop: bool = False
    ) -> tuple[sp.Rational, dict[sp.Expr, sp.Rational], sp.Expr]:
        q = sp.Integer(0)
        parts: dict[sp.Expr, sp.Rational] = {}
        for term in sp.Add.make_args(sp.expand(v)):
            if term.is_Rational:
                q += term
                continue
            c, b = term.as_coeff_Mul()
            if not c.is_Rational or not b.free_symbols:
                raise NetUnsupported(f"the phase term {term}")
            parts[b] = parts.get(b, sp.Integer(0)) + sp.Rational(c)
        z = sp.Integer(0)
        if drop:
            for b in list(parts):
                if b.is_Symbol and b.is_integer:
                    n = sp.floor(parts[b])
                    z += n * b
                    parts[b] -= n
                    if parts[b] == 0:
                        del parts[b]
        return sp.Rational(q), parts, z

    def ks(self, v: sp.Expr, drop: bool = False) -> tuple[Fraction, list[int], sp.Expr]:
        q, parts, z = self.split(v, drop)
        ks = [int(parts.get(b, 0) * L) for b, L in zip(self.bases, self.scale, strict=True)]
        return Fraction(int(q.p), int(q.q)), ks, z


def ph_c(q: Fraction, ks: Sequence[int]) -> CMon:
    """``phC``."""
    a, r = rnorm(Fraction(1), q)
    return (a, r, (), (), trim(ks), ())


def _unnest(d: Diagram) -> tuple[Diagram, dict[str, str]]:
    """``d`` with each box nested in boxes of its own multiplicity symbol given a fresh symbol
    per depth, and each fresh symbol's original."""
    from archytaszx.diagram.bangbox import Mult

    out = d.copy()
    alias: dict[str, str] = {}
    for bid, box in sorted(d.bang_boxes.items()):
        m = box.multiplicity.to_sympy()
        if m.is_number:
            continue
        depth, b = 0, box.parent
        while b is not None:
            if d.bang_boxes[b].multiplicity.to_sympy() == m:
                depth += 1
            b = d.bang_boxes[b].parent
        if depth:
            fresh = "_".join(sorted(map(str, m.free_symbols))) + f"_nest{depth}"
            while fresh in alias and alias[fresh] != str(m):
                fresh += "_"
            out.set_bang_box_multiplicity(bid, Mult(fresh))
            alias[fresh] = str(m)
    return out, alias


def _family_mults(left: Diagram, right: Diagram) -> list[sp.Expr]:
    """The multiplicities of the port boxes and of the node boxes, each node box meeting the
    rest only at the boundary or at wires into Z or X spiders, bare symbols first;
    :class:`NetUnsupported` for any other box."""
    from archytaszx.diagram.bangbox import crossing_wires

    names = set()
    for d in (left, right):
        for box in d.bang_boxes.values():
            if box.is_node_scope and not all(
                d.nodes[(w.b if w.a.node_id in box.node_scope else w.a).node_id].generator_type.name
                in ("Z", "X")
                for w in crossing_wires(d, box.node_scope)
            ):
                raise NetUnsupported("a box meeting the rest at a wire")
            names.add(box.multiplicity.to_sympy())
    return sorted(names, key=lambda m: (not m.is_Symbol, str(m)))


def _no_negative_powers(model: sp.Expr) -> sp.Expr:
    """``model`` with every ``a^(-k)`` written as ``Inv(a)^k`` and ``Inv(a)^(-k)`` as ``a^k``."""

    def flip(e: sp.Expr) -> sp.Expr:
        base, k = e.args[0], -int(e.args[1])
        return base.args[0] ** k if isinstance(base, Inv) else Inv(base) ** k

    return sp.sympify(model).replace(
        lambda e: e.is_Pow and e.args[1].is_Integer and e.args[1] < 0 and not e.args[0].is_Number,
        flip,
    )


def _peeled(split: Any, net: Any) -> Any:
    """The nodes, summed and boundary labels of ``split`` that ``net`` lacks."""
    from archytaszx.semantics.lean_net import LNet

    rest = list(split.nodes)
    for n in net.nodes:
        if n not in rest:
            raise NetUnsupported("the split network does not contain the network")
        rest.remove(n)
    return LNet(
        tuple(rest),
        tuple(x for x in split.summed if x not in net.summed),
        tuple(x for x in split.bnd if x not in net.bnd),
    )


def _split_ok(m: int, h: int, t: str) -> str:
    """A proof of ``Spec.splitOK`` at the family from the kernel's check at ``SpC m``."""
    return (
        f"(Spec.splitOK_struct _ {m} _ _ {h} {t} "
        f"(by decide +kernel : Spec.splitOK (SpC {m}) {h} {t} = true))"
    )


def _port_legs(net: Any, m: int, isb: Callable[[int], bool]) -> list[tuple[int, int]] | None:
    """Python mirror of ``Net.ports`` when ``Spec.portNet`` holds, else None."""
    from archytaszx.semantics.lean_net import _tail

    def legs(c: int, tl: Sequence[int], cs: Sequence[Any]) -> list[tuple[int, int]]:
        return [(x, c) for x, ch in zip(tl, cs, strict=False) if tuple(ch[:1]) == (m,)]

    def nested(cs: Sequence[Any]) -> bool:
        return any(tuple(c[:1]) == (m,) and tuple(c) != (m,) for c in cs)

    ports: list[tuple[int, int]] = []
    for n in net.nodes:
        h = n.kind if isinstance(n.kind, str) else n.kind[0]
        plain = all(not isb(lab) for lab in n.labels())
        if h == "ZB":
            cs = n.kind[1]
            if nested(cs):
                return None
            ports += legs(0, _tail(n.ins, cs), cs)
        elif h == "XB":
            co, ci = n.kind[1], n.kind[2]
            if nested((*co, *ci)):
                return None
            ports += legs(1, _tail(n.outs, co), co) + legs(1, _tail(n.ins, ci), ci)
        elif h == "WB":
            cs = n.kind[1]
            if nested(cs):
                return None
            ports += legs(2, _tail(n.outs, cs), cs)
        elif h == "Bun":
            if n.kind[1] == m or not plain:
                return None
        elif not plain:
            return None
    labels = [lab for lab, _ in ports]
    if len(set(labels)) != len(labels) or any(isb(lab) for lab in net.summed):
        return None
    return ports


class _SymExport:
    """The tables, networks and proof of one symbolic export."""

    cp_rows: dict[int, tuple[int, CpData]]
    rx: list[RxCert]
    fold: bool = False

    def __init__(
        self,
        left: Diagram,
        right: Diagram,
        describe: Callable[[Diagram], str] | None,
        floors: Mapping[str, int],
        roots: bool = False,
    ) -> None:
        out = _Export(describe or (lambda d: f"a diagram of {len(d.nodes)} nodes"))
        out.mixed_ok, out.mixed = True, set()
        self.out = out
        left, alias_l = _unnest(left)
        right, alias_r = _unnest(right)
        self.alias = {**alias_l, **alias_r}
        dim_syms = {
            x
            for d in (left, right)
            for n in d.nodes.values()
            for p in (*n.inputs, *n.outputs)
            for x in p.dim.to_sympy().free_symbols
        }
        for d in (left, right):
            for n in d.nodes.values():
                if n.phase is None:
                    continue
                for ph in n.phase.entries().values():
                    for x in sorted(ph.to_sympy_turns().free_symbols, key=str):
                        if x.is_integer and x not in dim_syms:
                            out.sym.name(x, "ℤ")
        for m in _family_mults(left, right):
            out.mult_of(m)
        self.mults = [n for n, bare in zip(out.mult_ix, out.mult_bare, strict=True) if bare]
        self.a = out.labelled(left)
        self.b = out.labelled(right)
        out.register(left)
        out.register(right)
        m0, m1 = out.scalar(left.scalar), out.scalar(right.scalar)
        if roots:
            m0, m1 = (self.with_roots(m, dim_syms) for m in (m0, m1))
        self.tables(m0, m1, floors)

    def phase_value(self, v: sp.Expr) -> sp.Expr:
        """A phase entry as the atoms read it."""
        return v

    def tables(self, m0: Any, m1: Any, floors: Mapping[str, int]) -> None:
        """The scalars ``m0``, ``m1`` and the dimension, phase and atom tables of ``self.out``."""
        out = self.out
        self.m0, self.m1 = m0, m1
        self.common: Any = None
        opaque = {v[0] for v in out.sc.opaques.values()}
        varying = set(out.mult_exprs)
        m0, m1 = sp.sympify(self.m0), sp.sympify(self.m1)
        if (m0.free_symbols | m1.free_symbols) & (opaque | varying) and m1 != 0:
            ratio = _no_negative_powers(sp.simplify(sp.powsimp(m0 / m1, force=True)))
            if not ratio.free_symbols & (opaque | varying):
                self.common = m1
                self.stated = (_paren(out.text(m0)), _paren(out.text(m1)))
                self.stated_models = (m0, m1)
                self.m0, self.m1 = ratio, sp.Integer(1)
        if out.sc.opaques and self.common is None:
            raise NetUnsupported("the diagrams' scalars differ by a factor outside the exact model")
        if out.family and not self.mults:
            raise NetUnsupported("bang boxes")
        out.set_params()
        params = [
            (raw, n, ty)
            for raw, (n, ty) in out.sym.params.items()
            if raw != "k_lean_index" and raw not in out.mults
        ]
        self.params = params

        def atom(e: sp.Expr) -> bool:
            e = sp.sympify(e)
            return bool(e.is_Pow and e.args[0].is_Symbol and not e.args[1].is_number)

        dsyms = sorted(
            {
                x
                for e in out.dim_exprs
                for x in ({sp.sympify(e).args[0]} if atom(e) else sp.sympify(e).free_symbols)
            },
            key=str,
        )
        self.nsym = len(dsyms)
        dsyms += [sp.sympify(e) for e in out.dim_exprs if atom(e)]
        self.dsyms = dsyms
        self.dnames = [
            out.sym.name(x, "ℕ") if x.is_Symbol else _paren(out.sym.render(x, "ℕ", "ℕ"))
            for x in dsyms
        ]
        self.dt = [_dmon(e, dsyms) for e in out.dim_exprs]
        self.constraints: list[str] = []
        for iq, ip in sorted(out.mixed):
            self.dt[iq] = self.dt[ip]
            self.constraints.append(f"{_paren(out.dims[ip])} = {_paren(out.dims[iq])}")
        if self.constraints and out.family:
            raise NetUnsupported("a family whose nodes' legs differ in dimension")
        values: list[sp.Expr] = []
        self.rules: list[tuple[DMon, tuple[int, ...]] | None] = []
        for entries, rule in out.phase_specs:
            self.rules.append(None if rule is None else self.rule_of(rule))
            values += [self.phase_value(v) for v in entries.values()]
        self.exs: list[sp.Expr] = []
        for model in (self.m0, self.m1):
            for x in sp.preorder_traversal(sp.sympify(model)):
                if isinstance(x, Ex):
                    self.exs.append(sp.expand(x.args[0] / (2 * sp.pi * sp.I)))
        self.atoms = _Atoms(self.exs, values)
        self.pt: list[list[tuple[int, CMon, sp.Expr]]] = []
        for entries, _ in out.phase_specs:
            row = []
            for j, v0 in sorted(entries.items()):
                v = self.phase_value(v0)
                q, ks, _ = self.atoms.ks(v, True)
                row.append((int(j), ph_c(q, ks), v))
            self.pt.append(row)
        fl = {str(k): int(v) for k, v in floors.items()}
        for nd in (*self.a.net.nodes, *self.b.net.nodes):
            if nd.kind in ("Z", "X") and nd.ph < len(self.pt) and self.pt[nd.ph]:
                need = max(j for j, _, _ in self.pt[nd.ph]) + 1
                c, e = self.dt[nd.dim]
                if sum(e) == 1 and c == 1:
                    name = str(dsyms[e.index(1)])
                    fl[name] = max(fl.get(name, 1), need)
        self.fl = trim([fl.get(str(x), 0) if fl.get(str(x), 0) > 1 else 0 for x in dsyms])
        msyms = set(out.mult_exprs)
        mult_varies = any(sp.sympify(m).free_symbols & msyms for m in (self.m0, self.m1))
        self.s0 = None if mult_varies else self.scalar_terms(self.m0)
        self.s1 = None if mult_varies else self.scalar_terms(self.m1)
        self.stabs: dict[str, tuple[str, str, list[STm]]] = {}
        self.stab_decls: list[str] = []
        self.ix = [i for i in range(len(self.dt)) if i not in out.bund_of]
        self.fam_decls: list[str] = []
        self.g_args: dict[str, str] = {}
        self.net_decls: list[str] = []
        self.planned: bool | None = None
        self.shift: tuple[str, int] | None = None
        self.shifted: Any = None
        self.fixed: Sequence[tuple[str, int] | tuple[str, int, bool]] | None = None
        self.fixed_steps: Any = None
        self.q: Any = None
        self.fz: list[tuple[int, FZCert]] = []
        self.rg: list[RgCert] = []
        self.g_lemma: dict[str, str] = {}

    # tables as Python

    def dtf(self, i: int) -> DMon:
        return self.dt[i] if i < len(self.dt) else (1, ())

    def ptf(self, i: int) -> list[tuple[int, CMon]]:
        row = PhRow((j, c) for j, c, _ in self.pt[i]) if i < len(self.pt) else PhRow()
        row.rule = self.rules[i] if i < len(self.rules) else None
        return row

    def rule_of(self, turns: sp.Expr) -> tuple[DMon, tuple[int, ...]]:
        """A phase rule ``p(k) / D`` as ``D`` and the integer coefficients of ``p``."""
        from archytaszx.algebra.phase import INDEX

        num, den = sp.fraction(sp.cancel(sp.together(sp.sympify(turns))))
        if num.free_symbols - {INDEX}:
            raise NetUnsupported(f"the phase rule {turns} is not a polynomial over a dimension")
        poly = sp.Poly(num, INDEX)
        cs = [poly.coeff_monomial(INDEX**i) for i in range(poly.degree() + 1)]
        if not all(c.is_Integer for c in cs) or cs[0] != 0:
            raise NetUnsupported(f"the phase rule {turns} has no integer form")
        return _dmon(den, self.dsyms), tuple(int(c) for c in cs)

    @staticmethod
    def net(
        lnet: Any,
    ) -> tuple[
        list[tuple[str, tuple[int, ...], tuple[int, ...], int, int]],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        nodes = []
        for n in lnet.nodes:
            if not isinstance(n.kind, str):
                raise NetUnsupported(f"the kind {n.kind}")
            nodes.append((n.kind, n.outs, n.ins, n.dim, n.ph))
        return nodes, lnet.summed, lnet.bnd

    def q_plan(self) -> bool:
        """Whether the F_2 checker of ``ArchytasZX.SymF2`` proves the pair: every dimension 2,
        no phase, Z, X and F nodes only, scalars ``a / b √2^K``, equal final terms; each side's
        order, hints and blocks, and the scalars, in ``self.q``."""
        from archytaszx.semantics import lean_f2 as Q

        out = self.out
        if self.mults or self.constraints or self.params or self.common is not None:
            return False
        if out.sc.opaques:
            return False
        if self.dt != [(2, ())] or any(self.pt) or any(r is not None for r in self.rules):
            return False
        if any(v != 0 for v in out.labs.values()):
            return False
        try:
            nets = [self.net(self.a.net), self.net(self.b.net)]
        except NetUnsupported:
            return False
        if sorted(nets[0][2]) != sorted(nets[1][2]):
            return False
        for nodes, _, _ in nets:
            if any(nd[3] != 0 or Q.tr_q(nd[0], nd[1], nd[2]) is None for nd in nodes):
                return False
        s0, s1 = Q.scalar_of(self.m0), Q.scalar_of(self.m1)
        if s0 is None or s1 is None:
            return False
        plans = [Q.plan(n[0], n[1]) for n in nets]
        if plans[0] is None or plans[1] is None:
            return False
        if not Q.cmp_q(s0, s1, plans[0][2][0], plans[1][2][0]):
            return False
        self.q = ([(p[0], p[1], p[2][1]) for p in plans], (s0, s1))
        return True

    def compact_rules(self) -> bool:
        """Marks with a ``cpData`` rule each phase vector without a rule whose stored phases
        are a compact table at every Z or X node reading it (kind 1 ``c j / M``, else for Z
        nodes on ``d²`` kind 2 ``c (j / d) (j % d) / M``); whether one was marked."""
        nets = [self.net(self.a.net), self.net(self.b.net)]
        uses: dict[int, list[tuple[str, DMon, tuple[int, ...]]]] = {}
        for nodes, _, _ in nets:
            for k, (kind, os, is_, dim, ph) in enumerate(nodes):
                if kind in ("Z", "X"):
                    uses.setdefault(ph, []).append((kind, self.dtf(dim), (*os, *is_)))
                elif ph < len(self.pt) and self.pt[ph] and kind not in ("B", "S"):
                    uses.setdefault(ph, []).append(("", (0, ()), ()))
        marked = False
        for p, row in enumerate(self.pt):
            if p >= len(self.rules) or self.rules[p] is not None or not row or p not in uses:
                continue
            pairs = [(j, c) for j, c, _ in row]
            turns = {}
            for j, c in pairs:
                if c[0] not in (1, -1) or c[2] or c[3] or c[4] or c[5]:
                    break
                turns[j] = (Fraction(c[1]) + (Fraction(1, 2) if c[0] == -1 else 0)) % 1
            else:
                dims = {dm for _, dm, _ in uses[p]}
                if len(dims) != 1 or any(kind == "" for kind, _, _ in uses[p]):
                    continue
                (dm,) = dims
                if not dm_const(dm) or dm[0] < 2:
                    continue
                big_d = dm[0]
                m = 1
                for v in turns.values():
                    m = math.lcm(m, v.denominator)
                vals = [turns.get(j, Fraction(0)) for j in range(big_d)]
                d = math.isqrt(big_d)
                cands = [(1, m, int(vals[1] * m) % m, 0)]
                if d * d == big_d and d > 1 and all(kind == "Z" for kind, _, _ in uses[p]):
                    cands.append((2, m, int(vals[d + 1] * m) % m, d))
                for q in cands:
                    if all(cp_chk(dm, pairs, q, OFF, (), ()) for _ in uses[p]):
                        self.rules[p] = (CP_MARK, q)
                        if big_d > CP_ROW_MIN:
                            self.cp_rows[p] = (big_d, q)
                            self.pt[p] = cp_row(big_d, q)
                        marked = True
                        break
        return marked

    def chained_gadgets(self) -> bool:
        """Whether a splitter output of either network is a binder input."""
        for lnet in (self.a.net, self.b.net):
            nodes = self.net(lnet)[0]
            outs = {lab for kind, os, _, _, _ in nodes if kind == "S" for lab in os}
            if any(lab in outs for kind, _, is_, _, _ in nodes if kind == "B" for lab in is_):
                return True
        return False

    def compact_check(self, args: tuple[Any, ...]) -> bool:
        """The compact check: the tables marked by :meth:`compact_rules`, binder and splitter
        nodes on, the difference cancelled by ``rx_cert``; the rules, tables and node setting
        restored when it fails."""
        saved = list(self.rules)
        saved_pt = list(self.pt)
        self.cp_rows = {}
        if not self.compact_rules():
            return False
        self.fz, self.rg = [], []
        global _BS_ON
        bs = _BS_ON
        _BS_ON = True
        fl, dtf, ptf, labs, ix, n0, n1, s0, s1 = args
        global _FOLD
        if any(
            got is not None
            for nodes, internal, _ in (n0, n1)
            for _, _, _, got in fold_steps(dtf, ptf, nodes, internal)
        ):
            _FOLD = True
            try:
                if sym_check(fl, dtf, ptf, labs, ix, n0, n1, s0, s1, None, None, self.rx):
                    self.fold = True
                    return True
            finally:
                _FOLD = False
            self.rx = []
        if sym_check(fl, dtf, ptf, labs, ix, n0, n1, s0, s1, None, None, self.rx):
            return True
        _BS_ON = bs
        self.rules = saved
        self.pt = saved_pt
        self.cp_rows = {}
        self.rx = []
        return False

    def check(self) -> bool:
        if not self.mults:
            assert self.s0 is not None and self.s1 is not None
            if self.q_plan():
                return True
            self.fz = []
            self.rg = []
            self.rx = []
            args = (
                self.fl,
                self.dtf,
                self.ptf,
                dict(self.out.labs),
                self.ix,
                self.net(self.a.net),
                self.net(self.b.net),
                self.s0,
                self.s1,
            )
            self.cp_rows = {}
            first = self.chained_gadgets()
            if first and self.compact_check(args):
                return True
            self.fz, self.rg = [], []
            try:
                if sym_check(*args, self.fz, self.rg):
                    return True
                failed: NetUnsupported | None = None
            except NetUnsupported as exc:
                failed = exc
            if not first and self.compact_check(args):
                return True
            if failed is not None:
                raise failed
            return False
        if self.planned is None:
            try:
                self.texts()
                if self.fixed:
                    self.plan_fixed()
                elif self.shift is not None:
                    self.plan_shift()
                else:
                    self.fam(
                        self.a.net,
                        self.a.name,
                        self.b.net,
                        self.b.name,
                        (sp.sympify(self.m0), sp.sympify(self.m1), self.s0t, self.s1t),
                    )
                self.planned = True
            except NetUnsupported as exc:
                self.planned, self.why = False, str(exc)
        return self.planned

    def plan_shift(self) -> None:
        """The family lemma for both networks split ``c`` times along the shifted index, its
        scalars at ``n + c``; the splits' texts and offsets in ``shifted``."""
        from archytaszx.semantics.lean_net import _split_net

        assert self.shift is not None
        out = self.out
        sym_name, c = self.shift
        if self.common is not None:
            raise NetUnsupported("a shifted family with a common scalar")
        if sym_name not in out.mult_ix:
            raise NetUnsupported("the shifted symbol carries no box")
        j = out.mult_ix[sym_name]
        isb = out.isb(j)
        H = self.offset_base()
        offs = [H << t for t in range(c)]
        nets = [(self.a.net, self.a.name, self.b.net, self.b.name)]
        for off in offs:
            x, xt, y, yt = nets[-1]
            self.peel_regs((x, y), j, off)
            spc = f"(SpC {j})"
            nets.append(
                (
                    _split_net(x, j, isb, off),
                    f"(Net.split {spc} {off} {xt})",
                    _split_net(y, j, isb, off),
                    f"(Net.split {spc} {off} {yt})",
                )
            )
        sym = self.mult_sym(j)
        name = self.var_names()[j]
        if sym is None:
            raise NetUnsupported("the shifted index is not a bare multiplicity")
        m0 = sp.sympify(self.m0).subs(sym, sym + c)
        m1 = sp.sympify(self.m1).subs(sym, sym + c)
        t0 = self.retext(self.s0t, name, f"({name} + {c})")
        t1 = self.retext(self.s1t, name, f"({name} + {c})")
        x, xt, y, yt = nets[-1]
        self.fam(x, xt, y, yt, (m0, m1, t0, t1))
        self.shifted = (j, c, offs, nets)

    def offset_base(self) -> int:
        """The first split offset: a power of two above every label of both networks."""
        labels = [
            lb
            for net in (self.a.net, self.b.net)
            for lb in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels()))
        ]
        return 1 << max(16, max(labels, default=0).bit_length())

    def peel_regs(self, nets: Sequence[Any], j: int, off: int) -> None:
        """Declare the copies at ``off`` of the labels of ``nets`` bundled along ``j``."""
        for net in nets:
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                if self.out.label_chain(lab)[:1] == (j,):
                    self.out.peel_reg(lab, off)

    def plan_fixed(self) -> None:
        """The family lemma for both networks, along each clause of ``fixed`` in turn split
        ``c`` times, and zeroed at ``n = c``, its scalars at the clauses' values (``n + c`` at
        ``n >= c``, ``n`` standing for the step); the steps in ``fixed_steps``."""
        from archytaszx.semantics.lean_net import _split_net, _zero_net

        assert self.fixed is not None
        out = self.out
        if self.common is not None:
            raise NetUnsupported("a fixed family with a common scalar")
        js = []
        for sym_name, *_ in self.fixed:
            if sym_name not in out.mult_ix:
                raise NetUnsupported("a fixed symbol carries no box")
            js.append(out.mult_ix[sym_name])
        fixed = {j: bool(f[2]) if len(f) > 2 else True for j, f in zip(js, self.fixed)}
        depth = dict.fromkeys(js, 0)
        for net in (self.a.net, self.b.net):
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                inner = [j for j in out.label_chain(lab) if j in depth]
                for pos, j in enumerate(inner):
                    depth[j] = max(depth[j], pos)
                if any(not fixed[j] for j in inner[:-1]):
                    raise NetUnsupported("a lifted box inside a box at n >= c")
        order = sorted(range(len(js)), key=lambda q: depth[js[q]])
        H = self.offset_base()
        t_off = 0
        x, xt, y, yt = self.a.net, self.a.name, self.b.net, self.b.name
        cur = {j: c for j, (_, c, *_) in zip(js, self.fixed, strict=True)}
        steps: list[tuple[str, int, int, dict[int, int], str, str]] = []
        m0, m1, t0, t1 = sp.sympify(self.m0), sp.sympify(self.m1), self.s0t, self.s1t
        names = self.var_names()
        for j, (_, c, *_) in ((js[q], self.fixed[q]) for q in order):
            isb = out.isb(j)
            spc = f"(SpC {j})"
            for _ in range(c):
                off = H << t_off
                t_off += 1
                self.peel_regs((x, y), j, off)
                cur[j] -= 1
                steps.append(("split", j, off, dict(cur), xt, yt))
                x, y = _split_net(x, j, isb, off), _split_net(y, j, isb, off)
                xt, yt = f"(Net.split {spc} {off} {xt})", f"(Net.split {spc} {off} {yt})"
            sym = self.mult_sym(j)
            if sym is None:
                raise NetUnsupported("a fixed index is not a bare multiplicity")
            if not fixed[j]:
                if c:
                    m0, m1 = m0.subs(sym, sym + c), m1.subs(sym, sym + c)
                    t0 = self.retext(t0, names[j], f"({names[j]} + {c})")
                    t1 = self.retext(t1, names[j], f"({names[j]} + {c})")
                continue
            off = H << t_off
            t_off += 1
            self.peel_regs((x, y), j, off)
            steps.append(("zero", j, off, dict(cur), xt, yt))
            x, xt = _zero_net(x, j, isb), f"(Net.zero {spc} {xt})"
            y, yt = _zero_net(y, j, isb), f"(Net.zero {spc} {yt})"
            m0, m1 = m0.subs(sym, c), m1.subs(sym, c)
            t0, t1 = self.retext(t0, names[j], str(c)), self.retext(t1, names[j], str(c))
        self.fam(x, xt, y, yt, (m0, m1, t0, t1))
        self.fixed_steps = (js, steps, cur, fixed)

    def fixed_proof(self, vals: Sequence[str]) -> tuple[list[str], str]:
        """The values at the clauses and the proof lifting the split and zeroed nets' lemma
        there by ``Equal.of_zero_at`` and ``Equal.of_split_at``, one per step."""
        js, steps, zero, fixed = self.fixed_steps
        args = self.args
        fam = f"(F {args})" if args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(args, hyp0)} _)" if (args or hyp0) else "(Fpos _)"

        def one(q: int, t: int) -> str:
            if fixed[q]:
                return str(t)
            return vals[q] if t == 0 else f"({vals[q].strip('()')} + {t})"

        def at(cur: dict[int, int]) -> list[str]:
            return [one(q, cur[q]) if q in cur else v for q, v in enumerate(vals)]

        proof = self.call(self.top, *at(zero))
        for kind, j, off, cur, xt, yt in reversed(steps):
            lst = ", ".join(at(cur))
            oks = f"{_split_ok(j, off, xt)} {_split_ok(j, off, yt)}"
            if kind == "zero":
                proof = (
                    f"(Equal.of_zero_at {fam} {j} {off} [{lst}] (by simp) {xt} {yt} {oks} "
                    f"(by decide) {proof})"
                )
            else:
                proof = (
                    f"(Equal.of_split_at {fam} {j} {off} {one(j, cur[j])} [{lst}] (by simp) "
                    f"{xt} {yt} {oks} (fun _ _ _ => Fam.dimv_pos _ _ _ {fpos}) {proof})"
                )
        final = {j: c for j, (_, c, *_) in zip(js, self.fixed or (), strict=True)}
        return at(final), proof

    def shift_proof(self, vals: Sequence[str]) -> tuple[list[str], str]:
        """The values at ``n + c`` and the proof lifting the split family lemma there by
        ``Equal.of_split_at`` once per split."""
        j, c, offs, nets = self.shifted
        name = vals[j]
        args = self.args
        fam = f"(F {args})" if args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(args, hyp0)} _)" if (args or hyp0) else "(Fpos _)"

        def plus(t: int) -> str:
            return name if t == 0 else f"({name} + {t})"

        proof = self.call(self.top, *vals)
        for level in range(c - 1, -1, -1):
            t = c - 1 - level
            lst = [plus(t) if q == j else v for q, v in enumerate(vals)]
            xt, yt = nets[level][1], nets[level][3]
            proof = (
                f"(Equal.of_split_at {fam} {j} {offs[level]} {plus(t)} [{', '.join(lst)}] "
                f"(by simp) {xt} {yt} {_split_ok(j, offs[level], xt)} "
                f"{_split_ok(j, offs[level], yt)} (fun _ _ _ => Fam.dimv_pos _ _ _ {fpos}) "
                f"{proof})"
            )
        return [plus(c) if q == j else v for q, v in enumerate(vals)], proof

    # scalars

    def scalar_terms(self, model: sp.Expr) -> list[STm]:
        """The scalar as constant terms, each with its Lean constructor in ``self.texts_of``."""
        terms: list[STm] = []
        self.texts_of: dict[STm, str] = getattr(self, "texts_of", {})
        for term in sp.Add.make_args(sp.expand(sp.sympify(model))):
            rest, phs = self.root_facs(term)
            cm, text = self.monomial(rest)
            t: STm = (cm, (), (), phs)
            terms.append(t)
            self.texts_of[t] = text
        return terms

    def root_facs(self, term: sp.Expr) -> tuple[sp.Expr, tuple[Fac, ...]]:
        """``term`` without its roots of unity of symbolic order (``Om(M)^c``, ``M`` a
        monomial in the dimension symbols), and those roots as constant phase factors
        ``(M, c)``."""
        rest: list[sp.Expr] = []
        phs: tuple[Fac, ...] = ()
        for f in sp.Mul.make_args(term):
            base, k = f.as_base_exp()
            sign = -1 if isinstance(base, Inv) else 1
            inner = base.args[0] if sign == -1 else base
            if isinstance(inner, Om) and not inner.args[0].is_number and k.is_Integer:
                phs += ((_dmon(inner.args[0], self.dsyms), ip_c(sign * int(k))),)
                continue
            rest.append(f)
        return sp.Mul(*rest), phs

    @staticmethod
    def with_roots(model: sp.Expr, dims: Iterable[sp.Symbol]) -> sp.Expr:
        """``model`` with each ``Ex(2 pi i c / M)`` (``c`` an integer, ``M`` a monomial in the
        dimension symbols) written ``Om(M)^c``."""

        def root(x: sp.Expr) -> sp.Expr:
            turns = x.args[0].replace(lambda e: isinstance(e, Inv), lambda e: 1 / e.args[0])
            c, mm = sp.fraction(sp.factor(sp.expand(turns / (2 * sp.pi * sp.I))))
            syms = sorted(mm.free_symbols, key=str)
            if c.is_Integer and syms and set(syms) <= set(dims):
                try:
                    _dmon(mm, syms)
                except NetUnsupported:
                    return x
                return Om(mm) ** int(c)
            return x

        return _no_negative_powers(sp.sympify(model).replace(lambda e: isinstance(e, Ex), root))

    def scalar_text(self, t: STm) -> str:
        """A constant scalar term as a Lean ``STm``."""
        return f"⟨{self.texts_of[t]}, [], [], {_lean_facs(t[3])}⟩"

    def monomial(self, term: sp.Expr) -> tuple[CMon, str]:
        cm, text = of_q(1), "CMon.ofQ 1"

        def mul(c2: CMon, t2: str) -> None:
            nonlocal cm, text
            cm, text = cm_mul(cm, c2), f"CMon.mul ({text}) ({t2})"

        for f in sp.Mul.make_args(term):
            base, k = f.as_base_exp()
            if not k.is_Integer:
                raise NetUnsupported(f"the scalar factor {f}")
            k = int(k)
            if base.is_Rational:
                q = Fraction(int(base.p), int(base.q)) ** k
                mul(of_q(q), f"CMon.ofQ {_lean_q(q)}")
            elif base == II:
                r = Fraction(k, 4)
                mul(ex_q(r), f"CMon.exQ {_lean_q(r)}")
            elif isinstance(base, Om) and base.args[0].is_Integer:
                r = Fraction(k, int(base.args[0]))
                mul(ex_q(r), f"CMon.exQ {_lean_q(r)}")
            elif isinstance(base, Rt) or (isinstance(base, Inv) and isinstance(base.args[0], Rt)):
                sign = 1 if isinstance(base, Rt) else -1
                inner = base.args[0] if sign == 1 else base.args[0].args[0]
                dm = _dmon(inner, self.dsyms)
                mul(rt_d(dm, sign * k), f"CMon.rtD {_lean_dm(dm)} ({sign * k})")
            elif isinstance(base, Inv) and (
                base.args[0] in self.dsyms or isinstance(base.args[0], NC)
            ):
                inner = base.args[0] if base.args[0] in self.dsyms else base.args[0].args[0]
                dm = _dmon(inner, self.dsyms)
                mul(rt_d(dm, -2 * k), f"dP {_lean_dm(dm)} ({-k})")
            elif base in self.dsyms or isinstance(base, NC):
                dm = _dmon(base if base in self.dsyms else base.args[0], self.dsyms)
                mul(rt_d(dm, 2 * k), f"dP {_lean_dm(dm)} ({k})")
            elif isinstance(base, Ex):
                q, ks, _ = self.atoms.ks(sp.expand(base.args[0] / (2 * sp.pi * sp.I)))
                q, ks = q * k, [k * x for x in ks]
                mul(ph_c(q, ks), f"phC {_lean_q(q)} {_lean_ints(ks)}")
            else:
                raise NetUnsupported(f"the scalar factor {f}")
        return cm, text

    # the source

    def env_at(self, vs: Sequence[str] = ()) -> str:
        """The environment text, at the multiplicities ``vs`` for a family."""
        args = " ".join(n for _, n, _ in self.params)
        if not self.mults:
            return f"(E {args})" if args else "E"
        fam = f"(F {args})" if args else "F"
        return f"({fam}.env (ksOf [{', '.join(vs)}]))"

    def texts(self) -> None:
        """The binder, argument and scalar texts the declarations share."""
        out = self.out
        binders = " ".join(f"({n} : {ty})" for _, n, ty in self.params)
        args = " ".join(n for _, n, _ in self.params)
        hyps = []
        for i, nm in enumerate(self.dnames[: self.nsym]):
            low = self.fl[i] if i < len(self.fl) else 0
            hyps.append(f"(h_{nm} : {max(low, 1)} ≤ {nm})")
        hyps += [f"(hc{j} : {c})" for j, c in enumerate(self.constraints)]
        hyp_args = " ".join(
            [
                *(f"h_{nm}" for nm in self.dnames[: self.nsym]),
                *(f"hc{j}" for j in range(len(self.constraints))),
            ]
        )
        self.binders, self.args = binders, args
        self.bh, self.ah = _join(binders, *hyps), _join(args, hyp_args)
        self.rho = f"(ρ {args})" if args else "ρ"
        self.hr = self.call("rhoOK")
        self.s0t, self.s1t = _paren(out.text(self.m0)), _paren(out.text(self.m1))

    def common_wrap(self, proof: str) -> tuple[str | None, str]:
        """A scalar both sides share outside the exact model, multiplied onto ``proof``."""
        if self.common is None:
            return None, proof
        ct = _paren(self.out.text(self.common))
        tactics = []
        for model, stated in zip((self.m0, self.m1), self.stated_models, strict=True):
            try:
                tactics.append(self.out.sc.prove(self.common * model, stated))
            except ScalarUnsupported:
                tactics.append("by ring")
        return ct, f"Equal.congr_s (Equal.smul {ct} ({proof})) ({tactics[0]}) ({tactics[1]})"

    def stated_s(self, side: int) -> str:
        """The scalar the theorem states for ``side``."""
        if self.common is None:
            return self.s0t if side == 0 else self.s1t
        return self.stated[side]

    def call(self, lemma: str, *more: str) -> str:
        text = _join(lemma, self.ah, *more)
        return f"({text})" if " " in text else text

    def sym(
        self,
        envt: str,
        ks: str,
        n0: str,
        n1: str,
        a0: str,
        a1: str,
        chk: str,
        l0: str = "scal0",
        l1: str = "scal1",
    ) -> str:
        """``equal_sym`` at the environment ``envt``, multiplicities ``ks``; ``equal_symS`` on the
        parts of ``chk`` when :meth:`check_decls` wrote it node by node."""
        g = self.g_args.get(chk)
        if g:
            lemma = self.g_lemma.get(chk, "equal_symS")
        else:
            lemma = "equal_symF" if self.fz else "equal_symR" if self.rg else "equal_sym"
        fzargs = " FZN FZC" if self.fz and not g else " RGC" if self.rg and not g else ""
        return (
            f"{lemma} {envt} {self.rho} {self.hr} FL {self.call('flOK')}{fzargs} DT PT RL labs IX "
            f"{self.call('envD', ks)} {self.call('envP', ks)} rfl {n0} {n1} {a0} {a1} _ _ "
            f"{self.call(l0)} {self.call(l1)} {g or chk}"
        )

    def q_source(self, name: str) -> str:
        """The proof by the F_2 checker: the runs through ``qchain`` checkpoints, ``qCheck`` on
        the final states, and the diagrams bridged by ``bridge_ok``."""
        from archytaszx.diagram.graph import Direction
        from archytaszx.semantics import lean_f2 as Q

        out = self.out
        sides, scal = self.q
        nets = (self.a, self.b)
        labels = [lab for lb in nets for lab in (*lb.net.summed, *lb.net.bnd)]
        labels += [lab for lb in nets for n in lb.net.nodes for lab in (*n.outs, *n.ins)]
        top = max([*labels, *out.labs, 0]) + 1
        decls = ["open ArchytasZX.SymF2"]
        revs = []
        for tag, lb, (rev, h, sizes) in zip("01", nets, sides, strict=True):
            n = lb.name
            revs.append("true" if rev else "false")
            decls += [
                (
                    f"/-- Each node's labels summed after it. -/\ndef H{tag} : List (List ℕ) := "
                    f"{Q.lean_hints(h)}"
                ),
                (
                    f"/-- The steps. -/\ndef R{tag} : List (Node × List ℕ) := "
                    f"(ordQ {revs[-1]} {n}).zip H{tag}"
                ),
                (
                    f"/-- The summed labels as a mask. -/\ndef IM{tag} : ℕ := "
                    f"0x{Q.mask_of(lb.net.summed):x}"
                ),
                f"theorem IM{tag}_eq : maskL {n}.int = IM{tag} := by decide +kernel",
                f"/-- The blocks' sizes. -/\ndef KS{tag} : List ℕ := {_lean_ints(sizes)}",
                f"set_option maxHeartbeats 0 in\nqchain q{tag} IM{tag} R{tag} KS{tag}",
            ]
        nr = len(out.phases)
        cases = " | ".join(["_"] * (nr + 1))
        (a0, b0, k0), (a1, b1, k1) = scal
        sc = f"({a0} : ℤ) {b0} ({k0} : ℤ) ({a1} : ℤ) {b1} ({k1} : ℤ)"
        decls += [
            (
                "/-- Every phase is zero. -/\ntheorem hph : ∀ ph j, E.phase ph j = 0 := by\n"
                f"  intro ph j\n  rcases ph with {cases} <;> first | rfl | simp [E, Φ]"
            ),
            (
                "/-- Every label has dimension index 0. -/\n"
                "theorem hL : E.labs.all (fun p => p.2 == 0) = true := by decide +kernel"
            ),
            (
                "set_option maxHeartbeats 0 in\n/-- The final terms agree. -/\n"
                f"theorem qc : qCheck N0 N1 H0 H1 q0_fin q1_fin {sc} = true := by decide +kernel"
            ),
        ]
        A, B = self.a.name, self.b.name
        stated = [self.stated_s(0), self.stated_s(1)]
        form = [f"((({a} : ℤ) : ℂ) / (({b} : ℕ) : ℂ) * rt 2 ^ ({k} : ℤ))" for a, b, k in scal]
        theorem = (
            "/-- The diagrams are equal. -/\n"
            f"theorem {name} :\n    Equal E {stated[0]} {A} {stated[1]} {B} :=\n"
            f"  Equal.congr_s (equal_q E rfl hph hL {A} {B} {revs[0]} {revs[1]} H0 H1\n"
            f"    q0_fin q1_fin {sc} (by rw [IM0_eq]; exact q0) (by rw [IM1_eq]; exact q1) qc)\n"
            f"    (show {form[0]} = {stated[0]} by qscal) (show {form[1]} = {stated[1]} by qscal)"
        )
        out.decls += [*decls, theorem]
        plain = out.diagram_literal

        def literal(d: Diagram, net: Any, nm: str, own: bool = False) -> tuple[str, int] | None:
            start = len(out.decls)
            got = plain(d, net, nm, own)
            dname = f"D{nm[1:]}"
            if got is None or own or got[0] != dname:
                return got
            off = got[1]
            ports = [
                (int(r.node_id), r.direction is Direction.OUTPUT, r.index)
                for r in [*d.boundary_outputs, *d.boundary_inputs]
            ]
            wires = [
                (
                    (int(w.a.node_id), w.a.direction is Direction.OUTPUT, w.a.index),
                    (int(w.b.node_id), w.b.direction is Direction.OUTPUT, w.b.index),
                )
                for w in sorted(d.wires, key=lambda w: w.sort_key())
            ]
            width = max([1, *(max(len(n.outputs), len(n.inputs)) for n in d.nodes.values())])
            ends = [*ports, *(p for w in wires for p in w)]
            bits = max([1, *(Q.pkey(width, *p).bit_length() for p in ends)])
            wb = Q.pack_w(width, bits, wires)
            args = f"{width} {bits} {off} {top} WB{nm[1:]} {dname} {nm}"
            del out.decls[start + 1 :]
            out.checked.add(dname)
            out.decls += [
                f"/-- {dname}'s wires, packed. -/\ndef WB{nm[1:]} : ℕ := 0x{wb:x}",
                f"theorem {dname}_bq : bridgeQ {args} = true := by decide +kernel",
                (
                    f"theorem {dname}_net : {dname}.toNet {off} = {nm} := "
                    f"(bridge_ok {args} {dname}_bq).2"
                ),
                (
                    f"theorem {dname}_chk : {dname}.check labs {off} = true := "
                    f"(bridge_ok {args} {dname}_bq).1"
                ),
            ]
            return got

        out.diagram_literal = literal  # type: ignore[method-assign,assignment]
        try:
            out.diagram_theorem(name, "", "", stated[0], self.a, stated[1], self.b, env="E")
        finally:
            del out.diagram_literal
        lead = "  equality of qubit networks by the verified normaliser over F_2 (ArchytasZX.SymF2)"
        src = out.source(lead=lead)
        return re.sub(
            r"def labs : List \(ℕ × ℕ\) := \[[^\n]*\]",
            f"def labs : List (ℕ × ℕ) := (List.range {top}).map fun l => (l, 0)",
            src,
            count=1,
        )

    def prelude(self) -> list[str]:
        """The tables DT, PT, RL, IX, FL, GS and ρ with rhoOK, flOK, envD and envP (and the
        scalars S0, S1 of a pair without boxes), after :meth:`texts`."""
        out = self.out
        bh, args, rho = self.bh, self.args, self.rho
        dims = "[" + ", ".join(self.dnames) + "]"
        gs = [
            out.sym.render(b / L, "ℝ", "ℝ")
            for b, L in zip(self.atoms.bases, self.atoms.scale, strict=True)
        ]
        gtext = "[" + ", ".join(gs) + "]"
        gsa = f"(GS {args})" if args else "GS"
        family = bool(self.mults)
        kb = "(ks : ℕ → ℕ)" if family else ""
        fam = f"(F {args})" if args else "F"
        env = f"({fam}.env ks)" if family else self.env_at()
        unfold = "Fam.env, F, bundL, Fam.dimv, List.lookup" if family else "E"
        decls = [
            "open ArchytasZX.Sym\n\nset_option linter.unusedSimpArgs false",
            "set_option Elab.async false\nset_option linter.all false",
        ]
        decls.append(
            "/-- Each dimension index as a monomial in the dimension symbols. -/\n"
            "def DT (i : ℕ) : DMon := (["
            + ", ".join(_lean_dm(m) for m in self.dt)
            + "] : List DMon).getD i ⟨1, []⟩"
        )
        rows = []
        for p, row in enumerate(self.pt):
            if p in getattr(self, "cp_rows", {}):
                dd, (k, m, c, d) = self.cp_rows[p]
                rows.append(f"cpRow {dd} (({k}, {m}, {c}, {d}) : ℕ × ℕ × ℤ × ℕ)")
                continue
            items = []
            for j, _, v in row:
                q, ks, _ = self.atoms.ks(v, True)
                items.append(f"({j}, phC {_lean_q(q)} {_lean_ints(ks)})")
            rows.append("[" + ", ".join(items) + "]")
        decls.append(
            "/-- Each phase vector's stored entries. -/\n"
            "def PT (i : ℕ) : List (ℕ × CMon) := (["
            + ", ".join(rows)
            + "] : List (List (ℕ × CMon))).getD i []"
        )
        rl = ", ".join(
            f"({_lean_dm(r[0])}, {_lean_ints(list(r[1]))})" if r is not None else "(⟨1, []⟩, [])"
            for r in self.rules
        )
        decls.append(
            "/-- Each phase vector's rule as a dimension monomial and the coefficients of "
            "its numerator. -/\ndef RL (i : ℕ) : DMon × List ℤ := (["
            + rl
            + "] : List (DMon × List ℤ)).getD i (⟨1, []⟩, [])"
        )
        decls.append(
            "/-- The dimension indices the networks use. -/\ndef IX : List ℕ := "
            + _lean_ints(self.ix)
        )
        decls.append(
            f"/-- Floors of the dimension symbols. -/\ndef FL : List ℕ := "
            f"{_lean_ints(list(self.fl))}"
        )
        decls.append(
            f"/-- The atoms' arguments. -/\nnoncomputable def GS {self.binders} : "
            "List ℝ := "
            f"{gtext}".replace("GS  :", "GS :")
        )
        decls.append(
            "/-- Dimension symbol values and atoms. -/\nnoncomputable def ρ "
            f"{self.binders} : "
            f"Val := ⟨fun i => ({dims} : List ℕ).getD i 1, fun j => ex ({gsa}.getD j 0)⟩".replace(
                "ρ  :", "ρ :"
            )
        )

        def stext(terms: list[STm]) -> str:
            return "[" + ", ".join(self.scalar_text(t) for t in terms) + "]"

        if not family:
            assert self.s0 is not None and self.s1 is not None
            decls.append(f"/-- The left scalar. -/\ndef S0 : List STm := {stext(self.s0)}")
            decls.append(f"/-- The right scalar. -/\ndef S1 : List STm := {stext(self.s1)}")
        nds = len(self.dnames)
        cases = " | ".join(["_"] * (nds + 1))
        decls.append(
            f"theorem rhoOK {bh} : {rho}.OK :=\n  ⟨fun i => by rcases i with {cases} <;> "
            "simp [ρ] <;> first | omega | exact Nat.pos_of_ne_zero (pow_ne_zero _ "
            "(by omega)), fun j => Complex.exp_ne_zero _⟩"
        )
        decls.append(
            f"theorem flOK {bh} : FloorOK FL {rho} := fun i => by\n  rcases i with "
            f"{cases} <;> simp [flv, FL, ρ] <;> omega"
        )
        nix = len(self.ix)
        cases_d = (
            f"\n  obtain {' | '.join(['rfl'] * nix)} := hi <;> "
            f"simp [{unfold}, Δ, DT, ρ, DMon.den, npow"
            + "".join(f", hc{j}" for j in range(len(self.constraints)))
            + "]"
            if nix
            else ""
        )
        decls.append(
            f"theorem envD {_join(bh, kb)} : ∀ i ∈ IX, {env}.dimv i = (DT i).den {rho}.dv := by\n"
            "  intro i hi\n  simp only [IX, List.mem_cons, List.not_mem_nil, or_false] at hi"
            + cases_d
        )
        nr = len(self.pt)
        ph_lines = [
            f"· {self.rule_pre(p, unfold, rho)}{self.ph_proof(p, row, unfold, rho, gsa)}"
            for p, row in enumerate(self.pt)
        ]
        ph_lines.append(f"· simp [{unfold}, Φ, PT, phF, ex_zero, rV, rEv, RL]")
        decls.append(
            f"theorem envP {_join(bh, kb)} : ∀ ph j, ex ({env}.phase ph j) = phF {rho} (PT ph) j "
            f"* rV {rho} (RL ph) j "
            f":= by\n  intro ph j\n  rcases ph with {' | '.join(['_'] * (nr + 1))}\n"
            + "\n".join("  " + line for line in ph_lines)
        )
        hr = self.hr
        for side, sm, st in (("0", self.m0, self.s0t), ("1", self.m1, self.s1t)):
            if family:
                break
            decls.append(
                f"theorem scal{side} {bh} : {st} = exDen {rho} (fun _ => 0) "
                f"S{side} := by\n  have hρ := {hr}\n  have hg : ∀ j, {rho}.atom j = ex "
                f"({gsa}.getD j 0) := fun j => rfl\n  {self.scalar_proof(sm, rho)}"
            )
        return decls

    def source(self, name: str) -> str:
        out = self.out
        self.texts()
        if self.q is not None:
            return self.q_source(name)
        decls = self.prelude()
        bh = self.bh
        family = bool(self.mults)
        env = "" if family else self.env_at()
        A, B = self.a.name, self.b.name
        lead = "  equality over symbolic dimensions by the verified normaliser of ArchytasZX.SymNet"
        if not family:
            decls += self.chk_decls(A, B)
            _, proof = self.common_wrap(self.sym(env, "", A, B, "S0", "S1", "chk"))
            theorem = (
                "/-- The diagrams are equal for every value of their dimensions and "
                f"phases. -/\ntheorem {name} {bh} :\n"
                f"    Equal {env} {self.stated_s(0)} {A} {self.stated_s(1)} {B} :=\n"
                f"  {proof}"
            )
            out.decls += [*decls, theorem]
            out.diagram_theorem(
                name, bh, self.ah, self.stated_s(0), self.a, self.stated_s(1), self.b, env=env
            )
        else:
            if not self.check():
                raise NetUnsupported(self.why)
            keys = list(out.mult_ix)
            names = [
                n
                for k, n, bare in zip(keys, out.mult_names, out.mult_bare, strict=True)
                if bare and k not in self.alias
            ]

            def val(k: str) -> str:
                j = out.mult_ix[self.alias.get(k, k)]
                return _paren(out.mult_names[j])

            vals = [val(k) for k in keys]
            s0, s1 = self.stated_s(0), self.stated_s(1)
            if self.fixed_steps is not None:
                js, fixed = self.fixed_steps[0], self.fixed_steps[3]
                raws = [vals[j].strip("()") for j in js]
                names = [n for n in names if n not in {out.mult_names[j] for j in js if fixed[j]}]
                vals, proof = self.fixed_proof(vals)
                for j, raw in zip(js, raws, strict=True):
                    s0 = self.retext(s0, raw, vals[j])
                    s1 = self.retext(s1, raw, vals[j])
            elif self.shifted is not None:
                j = self.shifted[0]
                raw = vals[j].strip("()")
                vals, proof = self.shift_proof(vals)
                s0 = self.retext(s0, raw, vals[j])
                s1 = self.retext(s1, raw, vals[j])
            else:
                _, proof = self.common_wrap(self.call(self.top, *vals))
            quant = f"∀ {' '.join(names)} : ℕ, " if names else ""
            lam = f"fun {' '.join(names)} => " if names else ""
            theorem = (
                "/-- The families are equal for every multiplicity and every value of "
                f"their dimensions and phases. -/\ntheorem {name} {bh} :\n"
                f"    {quant}Equal {self.env_at(vals)} {s0} "
                f"{A} {s1} {B} :=\n  {lam}{proof}"
            )
            out.decls += [
                *decls,
                *self.stab_decls,
                *self.net_decls,
                *reversed(self.fam_decls),
                theorem,
            ]
            fargs = " ".join(n for _, n, _ in self.params)
            hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
            out.family_diagrams(
                name,
                _join(bh, *(f"({n} : ℕ)" for n in names)),
                _join(self.ah, *names),
                f"(ksOf [{', '.join(vals)}])",
                self.env_at(vals),
                f"(F {fargs})" if fargs else "F",
                _join("Fpos", self.args, hyp0),
                s0,
                self.a,
                s1,
                self.b,
            )
            mm = ", ".join(names)
            if names:
                lead += (
                    f"\n  by induction on the multiplicit{'ies' if len(names) > 1 else 'y'} {mm}, "
                    "one copy peeled per step"
                )
            else:
                lead += "\n  at fixed multiplicities, every copy peeled"
        floors = [
            f"{n} ≥ {self.fl[i]}"
            for i, n in enumerate(self.dnames)
            if i < len(self.fl) and self.fl[i] > 1
        ]
        if floors:
            lead += "\n  assumes the dimension floors " + ", ".join(floors)
        if self.constraints:
            lead += (
                "\n  assumes the dimension constraints "
                + ", ".join(self.constraints)
                + " (hypotheses of the theorem)"
            )
        return out.source(lead=lead)

    # families

    def var_names(self) -> list[str]:
        """Each multiplicity index's variable in the family lemmas."""
        out = self.out
        keys = list(out.mult_ix)
        return [
            out.mult_names[j] if out.mult_bare[j] and keys[j] not in self.alias else f"v{j}"
            for j in range(len(out.mult_names))
        ]

    def mult_sym(self, j: int) -> sp.Symbol | None:
        """The multiplicity symbol a scalar may read for index ``j``, or None."""
        out = self.out
        keys = list(out.mult_ix)
        if out.mult_bare[j] and keys[j] not in self.alias:
            return out.mult_exprs[j]
        return None

    @staticmethod
    def retext(text: str, name: str, value: str) -> str:
        return re.sub(rf"(?<![\w'.]){re.escape(name)}(?![\w'])", value, text)

    def scal_at(self, sc: Any, j: int, text: str, value: Any) -> Any:
        """The scalars ``sc`` with index ``j`` set to ``value`` (Lean ``text``)."""
        mx, my, tx, ty = sc
        sym = self.mult_sym(j)
        if sym is None:
            return sc
        name = self.var_names()[j]
        return (
            mx.subs(sym, value),
            my.subs(sym, value),
            self.retext(tx, name, text),
            self.retext(ty, name, text),
        )

    def ratio(self, model: Any, j: int) -> Any:
        """The factor ``model`` gains when index ``j`` grows by one."""
        sym = self.mult_sym(j)
        if sym is None or sym not in model.free_symbols:
            return sp.Integer(1)
        if model == 0:
            raise NetUnsupported("a zero scalar depending on a multiplicity")
        r = sp.simplify(
            sp.powsimp(sp.expand_power_exp(model.subs(sym, sym + 1) / model), force=True)
        )
        if r.free_symbols & set(self.out.mult_exprs):
            raise NetUnsupported("a scalar not geometric in a multiplicity")
        return r

    def stab(self, model: Any, text: str) -> tuple[str, str, list[STm]]:
        """A constant scalar's term table, its proof lemma and terms."""
        key = text
        if key in self.stabs:
            return self.stabs[key]
        if model.free_symbols & set(self.out.mult_exprs):
            raise NetUnsupported("a scalar depending on a multiplicity outside its boxes")
        terms = self.scalar_terms(model)
        n = len(self.stabs)
        name, lem = f"SS{n}", f"scalS{n}"
        body = "[" + ", ".join(self.scalar_text(t) for t in terms) + "]"
        rho, gsa = self.rho, (f"(GS {self.args})" if self.args else "GS")
        self.stab_decls += [
            f"/-- A scalar's terms. -/\ndef {name} : List STm := {body}",
            (
                f"theorem {lem} {self.bh} : {_paren(text)} = exDen {rho} "
                f"(fun _ => 0) {name} := by\n"
                f"  have hρ := {self.hr}\n  have hg : ∀ j, {rho}.atom j = ex ({gsa}.getD j 0) := "
                f"fun j => rfl\n  {self.scalar_proof(model, rho, name)}"
            ),
        ]
        self.stabs[key] = (name, lem, terms)
        return self.stabs[key]

    def fam(self, x: Any, xt: str, y: Any, yt: str, sc: Any) -> str:
        """The name of a lemma ``∀ ks, Equal (F.env (ksOf ks)) sx X sy Y``, ``sc`` the scalars
        as models and Lean texts over the lemma's variables; its declarations in
        ``fam_decls``."""
        out = self.out
        if not hasattr(self, "bh"):
            self.texts()
        i = len(self.fam_decls)
        self.fam_decls.append("")
        mx, my, sxt, syt = sc
        vs = self.var_names()
        vb = " ".join(vs)
        from archytaszx.semantics.lean_net import LNet, _norm

        nx = LNet(tuple(_norm(n) for n in x.nodes), x.summed, x.bnd)
        ny = LNet(tuple(_norm(n) for n in y.nodes), y.summed, y.bnd)
        if nx != x or ny != y:
            na, nb = f"T{i}n", f"T{i}m"
            note = "/-- A network with its unbundled tails plain. -/\ndef"
            self.net_decls += [
                f"{note} {na} : Net := {nx.lean()}",
                f"{note} {nb} : Net := {ny.lean()}",
            ]
            lj = self.fam(nx, na, ny, nb, sc)
            env = self.env_at(vs)

            def iso(a_text: str, a: Any, b_text: str, b: Any) -> str:
                return out.iso_b(env, a_text, a, b_text, b)

            self.fam_decls[i] = (
                f"theorem P{i} {self.bh} : ∀ {vb} : ℕ, Equal {env} {sxt} {xt} {syt} {yt} :=\n"
                f"  fun {vb} => Equal.congr_s (Equal.mul_trans {iso(xt, x, na, nx)}\n"
                f"    (Equal.mul_trans {self.call(lj, *vs)} {iso(nb, ny, yt, y)})) (by ring) "
                "(by ring)"
            )
            if i == 0:
                self.top = f"P{i}"
            return f"P{i}"
        heads: set[int] = set()
        inner: set[int] = set()
        ported: set[int] = set()
        for net in (x, y):
            for lab in (*net.summed, *net.bnd, *(lb for n in net.nodes for lb in n.labels())):
                c = out.label_chain(lab)
                heads |= set(c[:1])
                inner |= set(c[1:])
            for n in net.nodes:
                if not isinstance(n.kind, str) and n.kind[0] in ("ZB", "XB", "WB"):
                    ported |= {c[0] for cs in n.kind[1:] for c in cs if c}
        if not heads:
            sx, lx, tx = self.stab(mx, sxt)
            sy, ly, ty = self.stab(my, syt)
            nodes0, nodes1 = self.net(x), self.net(y)
            labs = dict(out.labs)
            if not sym_check(self.fl, self.dtf, self.ptf, labs, self.ix, nodes0, nodes1, tx, ty):
                raise NetUnsupported("the normal forms of the two networks differ")
            env = self.env_at(vs)
            checks = self.check_decls(
                f"C{i}",
                "/-- The kernel's normalisation. -/\n",
                [(f"C{i}x", xt, x), (f"C{i}y", yt, y)],
                tx,
                ty,
                sx,
                sy,
            )
            self.fam_decls[i] = (
                "\n\n".join(checks)
                + f"\n\ntheorem P{i} {self.bh} : ∀ {vb} : ℕ, Equal {env} {sxt} {xt} "
                f"{syt} {yt} :=\n  fun {vb} => "
                + self.sym(env, f"(ksOf [{', '.join(vs)}])", xt, yt, sx, sy, f"C{i}", lx, ly)
            )
            if i == 0:
                self.top = f"P{i}"
            return f"P{i}"
        cands = sorted(heads - inner - ported) or sorted(heads - inner)
        if not cands:
            raise NetUnsupported("boxes that do not nest")
        m = cands[0]
        top = max(
            (
                lb
                for net in (x, y)
                for lb in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels()))
            ),
            default=0,
        )
        H = 1 << max(16, top.bit_length())
        for net in (x, y):
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                if out.label_chain(lab)[:1] == (m,):
                    out.peel_reg(lab, H)
        from archytaszx.semantics.lean_net import _split_net, _zero_net

        isb = out.isb(m)
        spc = f"(SpC {m})"
        if m in ported:
            try:
                return self.port(i, x, xt, y, yt, sc, m, H, vs)
            except NetUnsupported:
                if not any(isb(lab) for lab in x.summed):
                    raise
        al, be = self.ratio(mx, m), self.ratio(my, m)
        zx, zy = _zero_net(x, m, isb), _zero_net(y, m, isb)
        lz = self.fam(
            zx, f"(Net.zero {spc} {xt})", zy, f"(Net.zero {spc} {yt})", self.scal_at(sc, m, "0", 0)
        )
        sp0, sp1 = _split_net(x, m, isb, H), _split_net(y, m, isb, H)
        try:
            t0, t1 = _peeled(sp0, x), _peeled(sp1, y)
        except NetUnsupported:
            return self.tower(i, x, xt, y, yt, sc, m, H, vs, lz, (al, be))
        if sorted(t0.bnd) != sorted(t1.bnd):
            raise NetUnsupported("the peeled copies have different boundaries")
        ta, tb = f"T{i}a", f"T{i}b"
        alt, bet = _paren(out.text(al)), _paren(out.text(be))
        lt = self.fam(t0, ta, t1, tb, (al, be, alt, bet))
        name = vs[m]
        kvs = [("k_" if j == m else v) for j, v in enumerate(vs)]
        envk = self.env_at(kvs)
        bnd = sp0.bnd
        sk = self.scal_at(sc, m, "k_", sp.Symbol("k_"))
        sk1 = self.scal_at(sc, m, "(k_ + 1)", sp.Symbol("k_") + 1)

        def iso_s(a_text: str, a: Any, b_text: str, b: Any) -> str:
            return out.iso_b(envk, a_text, a, b_text, b)

        p0 = LNet(t0.nodes + x.nodes, t0.summed + x.summed, bnd)
        q0 = LNet(t0.nodes + y.nodes, t0.summed + y.summed, bnd)
        q0r = LNet(y.nodes + t0.nodes, y.summed + t0.summed, bnd)
        q1 = LNet(y.nodes + t1.nodes, y.summed + t1.summed, bnd)
        bl = "[" + ", ".join(map(str, bnd)) + "]"
        tp0 = f"⟨{ta}.nodes ++ {xt}.nodes, {ta}.int ++ {xt}.int, {bl}⟩"
        tq0 = f"⟨{ta}.nodes ++ {yt}.nodes, {ta}.int ++ {yt}.int, {bl}⟩"
        tq0r = f"⟨{yt}.nodes ++ {ta}.nodes, {yt}.int ++ {ta}.int, {bl}⟩"
        tq1 = f"⟨{yt}.nodes ++ {tb}.nodes, {yt}.int ++ {tb}.int, {bl}⟩"
        plug0 = (
            f"(plug {envk} {ta}.nodes {xt}.nodes {yt}.nodes {ta}.int {xt}.int {yt}.int {bl} "
            f"{xt}.bnd {sk[2]} {sk[3]} (Equal.toLocal _ _ _ ih) (by chk) (by chk) (by chk))"
        )
        plug1 = (
            f"(plug {envk} {yt}.nodes {ta}.nodes {tb}.nodes {yt}.int {ta}.int {tb}.int {bl} "
            f"{ta}.bnd {alt} {bet} (Equal.toLocal _ _ _ {self.call(lt, *kvs)}) (by chk) "
            "(by chk) (by chk))"
        )
        chain = (
            f"Equal.congr_s (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans "
            f"(Equal.mul_trans {iso_s(f'(Net.split {spc} {H} {xt})', sp0, tp0, p0)}\n"
            f"    {plug0})\n    {iso_s(tq0, q0, tq0r, q0r)})\n    {plug1})\n"
            f"    {iso_s(tq1, q1, f'(Net.split {spc} {H} {yt})', sp1)}) (by ring) (by ring)"
        )
        ok0, ok1 = _split_ok(m, H, xt), _split_ok(m, H, yt)
        args = self.args
        fam = f"(F {args})" if args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(args, hyp0)} _)" if (args or hyp0) else "(Fpos _)"
        zvs = [("0" if j == m else v) for j, v in enumerate(vs)]
        kb = " ".join(f"({v} : ℕ)" for v in kvs)
        self.net_decls += [
            f"/-- A peeled copy. -/\ndef {ta} : Net := {t0.lean()}",
            f"/-- A peeled copy. -/\ndef {tb} : Net := {t1.lean()}",
        ]
        self.fam_decls[i] = "\n\n".join(
            [
                (
                    f"theorem Q{i} {self.bh} {kb} (ih : Equal {envk} {sk[2]} {xt} {sk[3]} {yt}) :\n"
                    f"    Equal {envk} {sk1[2]} (Net.split {spc} {H} {xt}) {sk1[3]} "
                    f"(Net.split {spc} {H} {yt}) :=\n  {chain}"
                ),
                (
                    f"theorem P{i} {self.bh} : ∀ {vb} : ℕ, "
                    f"Equal {self.env_at(vs)} {sxt} {xt} {syt} {yt} "
                    f":= by\n  intro {vb}\n  induction {name} with\n"
                    f"  | zero => exact Equal.of_zero_at {fam} {m} {H} [{', '.join(zvs)}] "
                    f"(by simp) {xt} "
                    f"{yt} {ok0} {ok1} (by decide +kernel) (Equal.congr_s {self.call(lz, *zvs)} "
                    "(by ring) (by ring))\n"
                    f"  | succ k_ ih => exact Equal.of_split_at {fam} {m} {H} k_ "
                    f"[{', '.join(kvs)}] "
                    "(by simp) "
                    f"{xt} {yt} {ok0} {ok1} (fun _ _ _ => Fam.dimv_pos _ _ _ {fpos}) "
                    f"{self.call(f'Q{i}', *kvs, 'ih')}"
                ),
            ]
        )
        if i == 0:
            self.top = f"P{i}"
        return f"P{i}"

    def tower_cut(
        self, x: Any, zn: Any, rest: Sequence[Any]
    ) -> tuple[Any, int, tuple[int, ...], str]:
        """The split spider of ``zn`` in ``rest``, the boundary leg of ``zn`` its unfused part
        takes, the legs the split adds and the unfusion's form: ``zout`` (a Z spider's boundary
        output), ``zin`` (a Z spider's plain boundary input) or ``xout`` (an X spider's boundary
        output)."""
        from archytaszx.semantics.lean_net import _head, _tail

        xs = _head(zn.kind) == "XB"
        if xs and zn.kind[1]:
            raise NetUnsupported("an X spider meeting the node box with bundled outputs")
        cs1 = tuple(zn.kind[-1])
        i1 = tuple(zn.ins[: len(zn.ins) - len(cs1)])
        o = next((lab for lab in zn.outs if lab in x.bnd), None)
        mode = "xout" if xs else "zout"
        if o is None and not xs:
            o = next((lab for lab in i1 if lab in x.bnd), None)
            mode = "zin"
        if o is None:
            raise NetUnsupported("the spider meeting the node box has no boundary leg")
        found = [n for n in rest if _head(n.kind) == _head(zn.kind) and n.outs == zn.outs]
        if len(found) != 1:
            raise NetUnsupported("the split spider is not found")
        zs = found[0]
        es = tuple(lab for lab in zs.ins if lab not in zn.ins)
        if not es:
            raise NetUnsupported("the split spider gains no leg")
        t1 = _tail(zn.ins, cs1)
        if tuple(zs.ins) != i1 + tuple(t1) + es or tuple(zs.kind[-1]) != cs1 + ((),) * len(es):
            raise NetUnsupported("the split spider's legs are not the unfused form")
        if xs and zs.kind[1]:
            raise NetUnsupported("the split X spider has bundled outputs")
        return zs, o, es, mode

    @staticmethod
    def tower_leg(x: Any, zn: Any) -> int | None:
        """A boundary output of ``zn``, or for a Z spider a plain boundary input; None if
        neither."""
        from archytaszx.semantics.lean_net import _head

        o = next((lab for lab in zn.outs if lab in x.bnd), None)
        if o is None and _head(zn.kind) == "ZB":
            plain = zn.ins[: len(zn.ins) - len(zn.kind[-1])]
            o = next((lab for lab in plain if lab in x.bnd), None)
        return o

    def tower_target(self, x: Any, zn: Any, spiders: Sequence[Any], held: Sequence[Any]) -> Any:
        """The phaseless spider of ``spiders`` with a boundary leg and ``zn``'s kind that
        phaseless ``zn`` meets at one summed plain leg (:meth:`tower_shared`)."""
        from archytaszx.semantics.lean_net import _head

        for zm in spiders:
            if any(zm is h for h in held) or _head(zm.kind) != _head(zn.kind):
                continue
            if zn.ph or zm.ph or self.tower_shared(zn, zm, x) is None:
                continue
            return zm
        raise NetUnsupported(
            "the spider meeting the node box has no boundary leg and no neighbour of its colour "
            "with one"
        )

    @staticmethod
    def tower_shared(a: Any, b: Any, x: Any) -> int | None:
        """The one summed label joining a plain leg of ``a`` and one of ``b`` (``a``'s output
        and ``b``'s input for X spiders without bundled outputs); None if there is not one."""
        from archytaszx.semantics.lean_net import _head

        if _head(a.kind) == "XB":
            if a.kind[1] or b.kind[1]:
                return None
            ao = tuple(a.outs)
            bi = tuple(b.ins[: len(b.ins) - len(b.kind[2])])
            both = [lab for lab in ao if lab in bi]
        else:
            al = (*a.outs, *a.ins[: len(a.ins) - len(a.kind[1])])
            bl = (*b.outs, *b.ins[: len(b.ins) - len(b.kind[1])])
            both = [lab for lab in al if lab in bl]
        both = [lab for lab in both if lab in x.summed]
        return both[0] if len(both) == 1 else None

    def tower_fuse(self, envk: str, a: Any, b: Any, w: int) -> tuple[Any, str]:
        """The spider fusing phaseless ``a`` and ``b`` along ``w`` and the local equation
        (``fusionZG`` or ``fusionXG``) from the two to it."""
        from archytaszx.semantics.lean_net import LNode, _head, _list, _xk, _zk

        chk = "(by chk)"
        if _head(a.kind) == "ZB":
            c1, c2 = tuple(a.kind[1]), tuple(b.kind[1])
            i1, t1 = tuple(a.ins[: len(a.ins) - len(c1)]), tuple(a.ins[len(a.ins) - len(c1) :])
            i2, t2 = tuple(b.ins[: len(b.ins) - len(c2)]), tuple(b.ins[len(b.ins) - len(c2) :])
            o3 = tuple(lab for lab in (*a.outs, *b.outs) if lab != w)
            i3 = tuple(lab for lab in (*i1, *i2) if lab != w)
            f = LNode(_zk(c1 + c2), o3, i3 + t1 + t2, a.dim, 0)
            iface = tuple(dict.fromkeys(f.labels()))
            text = (
                f"fusionZG {envk} {self.fmt_cs(c1)} {self.fmt_cs(c2)} 0 0 0 {a.dim} {w} "
                f"{_list(a.outs)} {_list(i1)} {_list(t1)} {_list(b.outs)} {_list(i2)} "
                f"{_list(t2)} {_list(o3)} {_list(i3)} {_list(iface)} "
                + " ".join([chk] * 9)
                + " (by phs)"
            )
            return f, text
        ci1, ci2 = tuple(a.kind[2]), tuple(b.kind[2])
        o1, o2 = tuple(a.outs), tuple(b.outs)
        i1, t1 = tuple(a.ins[: len(a.ins) - len(ci1)]), tuple(a.ins[len(a.ins) - len(ci1) :])
        i2, t2 = tuple(b.ins[: len(b.ins) - len(ci2)]), tuple(b.ins[len(b.ins) - len(ci2) :])
        o3 = tuple(lab for lab in o1 if lab != w) + o2
        i3 = i1 + tuple(lab for lab in i2 if lab != w)
        f = LNode(_xk((), ci1 + ci2), o3, i3 + t1 + t2, a.dim, 0)
        iface = tuple(dict.fromkeys(f.labels()))
        hyp_d = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        pos = f"(Fpos {_join(self.args, hyp_d)} _)" if (self.args or hyp_d) else "(Fpos _)"
        text = (
            f"fusionXG {envk} [] {self.fmt_cs(ci1)} [] {self.fmt_cs(ci2)} 0 0 0 {a.dim} {w} "
            f"{_list(o1)} [] {_list(i1)} {_list(t1)} {_list(o2)} [] {_list(i2)} {_list(t2)} "
            f"{_list(o3)} {_list(i3)} {_list(iface)} "
            + " ".join([chk] * 4)
            + f" (Fam.env_dimv_pos _ _ _ {pos}) "
            + " ".join([chk] * 10)
            + " (by phs)"
        )
        return f, text

    def tower_move(
        self,
        envk: str,
        x: Any,
        zn: Any,
        zm: Any,
        rest: Sequence[Any],
        ints: tuple[int, ...],
        others: Sequence[Any],
        nb: tuple[int, ...],
    ) -> tuple[Any, ...]:
        """The legs the split adds to ``zn`` moved onto ``zm`` through the spider fusing the
        two (fused, reordered, unfused): the network before (joining label last), the joining
        label, the fusion step, the fused network, the reordered network, the unfusion step,
        the network after, the two split spiders and ``zm``'s split spider with both spiders'
        added legs."""
        from archytaszx.semantics.lean_net import LNet, LNode, _head, _list

        def split_of(z: Any) -> Any:
            found = [n for n in rest if _head(n.kind) == _head(z.kind) and n.outs == z.outs]
            if len(found) != 1:
                raise NetUnsupported("the split spider is not found")
            return found[0]

        def lit(nodes: Sequence[Any]) -> str:
            return "[" + ", ".join(n.lean() for n in nodes) + "]"

        sn, sm = split_of(zn), split_of(zm)
        w = self.tower_shared(zn, zm, x)
        assert w is not None
        en = tuple(lab for lab in sn.ins if lab not in zn.ins)
        tail = tuple(zn.kind[-1]) + ((),) * len(en)
        if not en or tuple(sn.ins) != tuple(zn.ins) + en or tuple(sn.kind[-1]) != tail:
            raise NetUnsupported("the split spider's legs are not the unfused form")
        kind = list(sm.kind)
        kind[-1] = tuple(sm.kind[-1]) + ((),) * len(en)
        sm2 = LNode(tuple(kind), sm.outs, tuple(sm.ins) + en, sm.dim, sm.ph)
        ctx = tuple(n for n in rest if n is not sn and n is not sm) + tuple(others)
        ic = tuple(lab for lab in ints if lab != w)
        f1, fuse1 = self.tower_fuse(envk, sn, sm, w)
        f2, fuse2 = self.tower_fuse(envk, zn, sm2, w)
        before = LNet(ctx + (sn, sm), ic + (w,), nb)
        fused = LNet(ctx + (f1,), ic, nb)
        refused = LNet(ctx + (f2,), ic, nb)
        after = LNet(ctx + (zn, sm2), ic + (w,), nb)
        bl = _list(nb)
        iface1 = tuple(dict.fromkeys(f1.labels()))
        iface2 = tuple(dict.fromkeys(f2.labels()))
        step1 = (
            f"(plug {envk} {lit(ctx)} {lit([sn, sm])} {lit([f1])} {_list(ic)} [{w}] [] {bl} "
            f"{_list(iface1)} 1 1 (fun τ hτ => ({fuse1}) τ hτ) (by chk) (by chk) (by chk))"
        )
        step2 = (
            f"(plug {envk} {lit(ctx)} {lit([f2])} {lit([zn, sm2])} {_list(ic)} [] [{w}] {bl} "
            f"{_list(iface2)} 1 1 (fun τ hτ => (({fuse2}) τ hτ).symm) (by chk) (by chk) "
            "(by chk))"
        )
        return before, w, step1, fused, refused, step2, after, sn, sm, sm2

    def tower_fusion(
        self, envk: str, zn: Any, zs: Any, o: int, w: int, es: tuple[int, ...], mode: str
    ) -> tuple[Any, Any, str]:
        """``zn`` with its leg ``o`` renamed ``w``, the spider on ``o``, ``w`` and ``es``, and
        the local equation fusing the two into ``zs`` (``fusionZG`` or ``fusionXG``)."""
        from archytaszx.semantics.lean_net import LNode, _list, _tail, _xk, _zk

        cs1 = tuple(zn.kind[-1])
        t1 = _tail(zn.ins, cs1)
        i1 = tuple(zn.ins[: len(zn.ins) - len(cs1)])
        cs2 = self.fmt_cs(((),) * len(es))
        iface = tuple(dict.fromkeys(zs.labels()))
        chk = " ".join(["(by chk)"] * 9)
        if mode == "zin":
            ia = tuple(w if lab == o else lab for lab in i1)
            za = LNode(zn.kind, zn.outs, ia + tuple(t1), zn.dim, zn.ph)
            zb = LNode(_zk(((),) * len(es)), (), (o, w, *es), zs.dim, 0)
            fusion = (
                f"fusionZG {envk} {self.fmt_cs(cs1)} {cs2} {za.ph} {zb.ph} "
                f"{zs.ph} {zs.dim} {w} {_list(zn.outs)} {_list(ia)} {_list(t1)} [] "
                f"{_list((o, w))} {_list(es)} {_list(zs.outs)} {_list(i1)} {_list(iface)} "
                "(by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) "
                "(by chk) (by phs)"
            )
            return za, zb, fusion
        o1 = tuple(w if lab == o else lab for lab in zn.outs)
        za = LNode(zn.kind, o1, zn.ins, zn.dim, zn.ph)
        if mode == "zout":
            zb = LNode(_zk(((),) * len(es)), (o,), (w, *es), zs.dim, 0)
            fusion = (
                f"fusionZG {envk} {self.fmt_cs(cs1)} {cs2} {za.ph} {zb.ph} {zs.ph} "
                f"{zs.dim} {w} {_list(o1)} {_list(i1)} {_list(t1)} {_list(zb.outs)} {_list((w,))} "
                f"{_list(es)} {_list(zs.outs)} {_list(i1)} {_list(iface)} {chk} (by phs)"
            )
            return za, zb, fusion
        zb = LNode(_xk((), ((),) * len(es)), (o,), (w, *es), zs.dim, 0)
        hyp_d = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        pos = f"(Fpos {_join(self.args, hyp_d)} _)" if (self.args or hyp_d) else "(Fpos _)"
        fusion = (
            f"fusionXG {envk} [] {self.fmt_cs(cs1)} [] {cs2} {za.ph} {zb.ph} "
            f"{zs.ph} {zs.dim} {w} {_list(o1)} [] {_list(i1)} {_list(t1)} {_list(zb.outs)} [] "
            f"{_list((w,))} {_list(es)} {_list(zs.outs)} {_list(i1)} {_list(iface)} "
            "(by chk) (by chk) (by chk) (by chk) "
            f"(Fam.env_dimv_pos _ _ _ {pos}) (by chk) (by chk) (by chk) (by chk) (by chk) "
            "(by chk) (by chk) (by chk) (by chk) (by chk) (by phs)"
        )
        return za, zb, fusion

    @staticmethod
    def tower_open_leg(x: Any, zn: Any) -> tuple[int, Any, int] | None:
        """For a Z spider ``zn`` without a boundary leg: a summed plain leg ``u`` of it that is
        the input of an F node of ``x`` whose output is a boundary label, as ``(u, F node,
        that label)``; None if there is none."""
        from archytaszx.semantics.lean_net import _head

        if _head(zn.kind) != "ZB":
            return None
        plain = (*zn.outs, *zn.ins[: len(zn.ins) - len(zn.kind[-1])])
        for u in plain:
            if u not in x.summed:
                continue
            for f in x.nodes:
                if f.kind == "F" and tuple(f.ins) == (u,) and f.outs[0] in x.bnd:
                    return u, f, f.outs[0]
        return None

    def tower_cut_open(
        self, x: Any, zn: Any, u: int, rest: Sequence[Any]
    ) -> tuple[Any, int, tuple[int, ...], str]:
        """:meth:`tower_cut` at the summed leg ``u`` of ``zn`` (``zout`` when ``u`` is an
        output, else ``zin``)."""
        from archytaszx.semantics.lean_net import _head, _tail

        cs1 = tuple(zn.kind[-1])
        i1 = tuple(zn.ins[: len(zn.ins) - len(cs1)])
        mode = "zout" if u in zn.outs else "zin"
        found = [n for n in rest if _head(n.kind) == _head(zn.kind) and n.outs == zn.outs]
        if len(found) != 1:
            raise NetUnsupported("the split spider is not found")
        zs = found[0]
        es = tuple(lab for lab in zs.ins if lab not in zn.ins)
        t1 = _tail(zn.ins, cs1)
        if not es or tuple(zs.ins) != i1 + tuple(t1) + es:
            raise NetUnsupported("the split spider's legs are not the unfused form")
        return zs, u, es, mode

    def tower_opening(
        self, envk: str, kvs: Sequence[str], zb: Any, w: int, f: Any
    ) -> tuple[list[Any], Any, str, Any, Any]:
        """The F chain ``F (w -> w1), F (w1 -> w2), F (w2 -> w3), F (w3 -> w4)`` put before
        ``zb`` on its leg ``w`` (four F boxes are the identity): the first F box, the other
        three, ``zb`` on ``w4``, the local equation from ``zb`` to them (``Equal.toLocal`` of
        the normaliser's proof on the two small networks) and the two networks."""
        from archytaszx.semantics.lean_net import LNet, LNode

        out = self.out
        ws = [out.fresh_like(w) for _ in range(4)]
        chain = [w, *ws]
        fs = [LNode(f.kind, (chain[j + 1],), (chain[j],), f.dim, f.ph) for j in range(4)]
        zb2 = LNode(
            zb.kind,
            tuple(ws[3] if lab == w else lab for lab in zb.outs),
            tuple(ws[3] if lab == w else lab for lab in zb.ins),
            zb.dim,
            zb.ph,
        )
        iface = tuple(dict.fromkeys(zb.labels()))
        a = LNet((zb,), (), iface)
        b = LNet((*fs, zb2), tuple(ws), iface)
        i = len(self.fam_decls)
        at, bt = f"T{i}oa", f"T{i}ob"
        self.net_decls += [
            f"/-- A spider before an opening. -/\ndef {at} : Net := {a.lean()}",
            f"/-- The spider after four F boxes on one leg. -/\ndef {bt} : Net := {b.lean()}",
        ]
        one = sp.Integer(1)
        lj = self.fam(a, at, b, bt, (one, one, "1", "1"))
        eq = f"(Equal.toLocal _ _ _ {self.call(lj, *kvs)})"
        return [fs[0]], (fs[1], fs[2], fs[3], zb2), eq, a, b

    def tower(
        self,
        i: int,
        x: Any,
        xt: str,
        y: Any,
        yt: str,
        sc: Any,
        m: int,
        H: int,
        vs: list[str],
        lz: str,
        ratios: tuple[Any, Any],
    ) -> str:
        """The lemma ``P{i}`` for a node box along ``m`` whose copies meet Z or X spiders of
        ``x`` at their bundled inputs: zero by the zero networks; ``k + 1`` by splitting a copy
        off, unfusing each split spider at a boundary leg (:meth:`tower_fusion`) so that ``x``
        at ``k``, those legs renamed to the new wires, is a region, the induction hypothesis
        plugged there, and the rest normalised against ``y``'s split."""
        from archytaszx.semantics.lean_net import (
            LNet,
            _head,
            _list,
            _pairs,
            _perm_to_bijection,
            _rename_net,
            _split_net,
        )

        out = self.out
        isb = out.isb(m)
        spiders = [
            n
            for n in x.nodes
            if _head(n.kind) in ("ZB", "XB") and any(isb(lab) and lab in x.summed for lab in n.ins)
        ]
        if not spiders:
            raise NetUnsupported("a node box meeting no Z or X spider")
        if any(_head(n.kind) == "Bun" and n.kind[1] != m for n in x.nodes):
            raise NetUnsupported("towers along two multiplicities")
        sp0, sp1 = _split_net(x, m, isb, H), _split_net(y, m, isb, H)
        others = [n for n in x.nodes if all(n is not z for z in spiders)]
        rest = list(sp0.nodes)
        for n in others:
            if n not in rest:
                raise NetUnsupported("the split network does not contain the network")
            rest.remove(n)
        spc = f"(SpC {m})"
        kvs = [("k_" if j == m else v) for j, v in enumerate(vs)]
        envk = self.env_at(kvs)
        sk = self.scal_at(sc, m, "k_", sp.Symbol("k_"))
        sk1 = self.scal_at(sc, m, "(k_ + 1)", sp.Symbol("k_") + 1)
        al, be = ratios
        alt, bet = _paren(out.text(al)), _paren(out.text(be))
        bnd = sp0.bnd
        bl = _list(bnd)

        def iso_s(a_text: str, a: Any, b_text: str, b: Any) -> str:
            return out.iso_b(envk, a_text, a, b_text, b)

        def lit(nodes: Sequence[Any]) -> str:
            return "[" + ", ".join(n.lean() for n in nodes) + "]"

        steps: list[str] = []
        ints = tuple(sp0.summed)
        prev_t, prev = f"(Net.split {spc} {H} {xt})", sp0
        held = [z for z in spiders if self.tower_leg(x, z) is None]
        opened: dict[int, tuple[int, Any, int]] = {}
        for zn in held:
            try:
                self.tower_target(x, zn, spiders, held)
            except NetUnsupported:
                got = self.tower_open_leg(x, zn)
                if got is None:
                    raise
                opened[id(zn)] = got
        held = [z for z in held if id(z) not in opened]
        for j, zn in enumerate(held):
            zm = self.tower_target(x, zn, spiders, held)
            mv = self.tower_move(envk, x, zn, zm, rest, ints, others, bnd)
            names = [f"T{i}m{j}{c}" for c in "abcd"]
            notes = [
                "A split network before a leg move.",
                "A split network with two spiders fused.",
                "The fused spider with its legs reordered.",
                "A split network after a leg move.",
            ]
            for nm, note, net in zip(names, notes, (mv[0], mv[3], mv[4], mv[6]), strict=True):
                self.net_decls.append(f"/-- {note} -/\ndef {nm} : Net := {net.lean()}")
            steps += [
                iso_s(prev_t, prev, names[0], mv[0]),
                mv[2],
                iso_s(names[1], mv[3], names[2], mv[4]),
                mv[5],
            ]
            prev_t, prev = names[3], mv[6]
            ints = tuple(mv[6].summed)
            rest[:] = [n for n in rest if n is not mv[7] and n is not mv[8]] + [mv[9]]
            others.append(zn)
        cuts = []
        for zn in spiders:
            if any(zn is h for h in held):
                continue
            if id(zn) in opened:
                cut = self.tower_cut_open(x, zn, opened[id(zn)][0], rest)
            else:
                cut = self.tower_cut(x, zn, rest)
            rest.remove(cut[0])
            cuts.append((zn, *cut))
        copy = list(rest)
        one = len(cuts) == 1
        done: tuple[Any, ...] = ()
        mapping: dict[int, int] = {}
        zbs: list[Any] = []
        for j, (zn, zs, o, es, mode) in enumerate(cuts):
            w = out.fresh_like(o)
            mapping[o] = w
            za, zb, fusion = self.tower_fusion(envk, zn, zs, o, w, es, mode)
            zbs.append(zb)
            ctx = tuple(copy) + tuple(others) + done + tuple(c[1] for c in cuts[j + 1 :])
            n1 = LNet(ctx + (zs,), ints, bnd)
            n2 = LNet(ctx + (za, zb), ints + (w,), bnd)
            n1t, n2t = (f"T{i}s", f"T{i}u") if one else (f"T{i}s{j}", f"T{i}u{j}")
            self.net_decls += [
                f"/-- The split network, its spider last. -/\ndef {n1t} : Net := {n1.lean()}",
                (
                    f"/-- The split network with its spider unfused. -/\ndef {n2t} : Net := "
                    f"{n2.lean()}"
                ),
            ]
            iface = tuple(dict.fromkeys(zs.labels()))
            steps.append(iso_s(prev_t, prev, n1t, n1))
            steps.append(
                f"(plug {envk} {lit(ctx)} {lit([zs])} {lit([za, zb])} {_list(ints)} [] "
                f"[{w}] {bl} {_list(iface)} 1 1 (fun τ hτ => (({fusion}) τ hτ).symm) (by chk) "
                "(by chk) (by chk))"
            )
            done += (za, zb)
            ints += (w,)
            prev_t, prev = n2t, n2
            if id(zn) in opened:
                _u, fnode, o_out = opened[id(zn)]
                head, tail, eq, na, nb = self.tower_opening(envk, kvs, zb, w, fnode)
                mapping[o_out] = head[0].outs[0]
                done = done[:-1]
                ctx = tuple(copy) + tuple(others) + done + tuple(c[1] for c in cuts[j + 1 :])
                n5 = LNet(ctx + (zb,), ints, bnd)
                n6 = LNet(ctx + tuple(head) + tail, ints + na.summed + nb.summed, bnd)
                n5t, n6t = f"T{i}o{j}", f"T{i}p{j}"
                self.net_decls += [
                    f"/-- The network before its opening. -/\ndef {n5t} : Net := {n5.lean()}",
                    f"/-- The network after its opening. -/\ndef {n6t} : Net := {n6.lean()}",
                ]
                steps.append(iso_s(prev_t, prev, n5t, n5))
                steps.append(
                    f"(plug {envk} {lit(ctx)} {lit([zb])} {lit([*head, *tail])} {_list(ints)} [] "
                    f"{_list(nb.summed)} {bl} {_list(nb.bnd)} 1 1 {eq} (by chk) (by chk) "
                    "(by chk))"
                )
                done += tuple(head) + tail
                ints += tuple(nb.summed)
                prev_t, prev = n6t, n6
                zbs.pop()
                zbs.extend((*tail, fnode))
        for lab in y.summed:
            mapping.setdefault(lab, out.fresh_like(lab))
        pairs = _perm_to_bijection(mapping)
        pt = _pairs(pairs)
        rx, ry = _rename_net(x, pairs), _rename_net(y, pairs)
        ctx3 = tuple(copy) + tuple(zbs)
        ic3 = tuple(lab for lab in prev.summed if lab not in set(rx.summed))
        n3 = LNet(ctx3 + rx.nodes, ic3 + rx.summed, bnd)
        n4 = LNet(ctx3 + ry.nodes, ic3 + ry.summed, bnd)

        def plug_text(t: str) -> str:
            return (
                f"⟨{lit(ctx3)} ++ (Net.rename {pt} {t}).nodes, {_list(ic3)} ++ "
                f"(Net.rename {pt} {t}).int, {bl}⟩"
            )

        hr = f"(Equal.rename {envk} {pt} {xt} {yt} (by chk) (by chk) ih)"
        plug = (
            f"(plug {envk} {lit(ctx3)} (Net.rename {pt} {xt}).nodes (Net.rename {pt} {yt}).nodes "
            f"{_list(ic3)} (Net.rename {pt} {xt}).int (Net.rename {pt} {yt}).int {bl} "
            f"(Net.rename {pt} {xt}).bnd {sk[2]} {sk[3]} (Equal.toLocal _ _ _ {hr}) (by chk) "
            "(by chk) (by chk))"
        )
        rt = f"T{i}r"
        self.net_decls.append(
            f"/-- The rest after the hypothesis. -/\ndef {rt} : Net := {n4.lean()}"
        )
        lt = self.fam(n4, rt, sp1, f"(Net.split {spc} {H} {yt})", (al, be, alt, bet))
        steps += [
            iso_s(prev_t, prev, plug_text(xt), n3),
            plug,
            iso_s(plug_text(yt), n4, rt, n4),
            self.call(lt, *kvs),
        ]
        chain = steps[0]
        for t in steps[1:]:
            chain = f"(Equal.mul_trans {chain}\n    {t})"
        chain = f"Equal.congr_s {chain} (by ring) (by ring)"
        ok0, ok1 = _split_ok(m, H, xt), _split_ok(m, H, yt)
        args = self.args
        fam = f"(F {args})" if args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(args, hyp0)} _)" if (args or hyp0) else "(Fpos _)"
        zvs = [("0" if j == m else v) for j, v in enumerate(vs)]
        kb = " ".join(f"({v} : ℕ)" for v in kvs)
        vb = " ".join(vs)
        name = vs[m]
        _mx, _my, sxt, syt = sc
        self.fam_decls[i] = "\n\n".join(
            [
                (
                    f"theorem Q{i} {self.bh} {kb} (ih : Equal {envk} {sk[2]} {xt} {sk[3]} {yt}) :\n"
                    f"    Equal {envk} {sk1[2]} (Net.split {spc} {H} {xt}) {sk1[3]} "
                    f"(Net.split {spc} {H} {yt}) :=\n  {chain}"
                ),
                (
                    f"theorem P{i} {self.bh} : ∀ {vb} : ℕ, "
                    f"Equal {self.env_at(vs)} {sxt} {xt} {syt} {yt} "
                    f":= by\n  intro {vb}\n  induction {name} with\n"
                    f"  | zero => exact Equal.of_zero_at {fam} {m} {H} [{', '.join(zvs)}] "
                    f"(by simp) {xt} "
                    f"{yt} {ok0} {ok1} (by decide +kernel) (Equal.congr_s {self.call(lz, *zvs)} "
                    "(by ring) (by ring))\n"
                    f"  | succ k_ ih => exact Equal.of_split_at {fam} {m} {H} k_ "
                    f"[{', '.join(kvs)}] "
                    "(by simp) "
                    f"{xt} {yt} {ok0} {ok1} (fun _ _ _ => Fam.dimv_pos _ _ _ {fpos}) "
                    f"{self.call(f'Q{i}', *kvs, 'ih')}"
                ),
            ]
        )
        if i == 0:
            self.top = f"P{i}"
        return f"P{i}"

    @staticmethod
    def fmt_cs(cs: Sequence[Any]) -> str:
        """Chains as a Lean list of lists."""
        return "[" + ", ".join("[" + ", ".join(map(str, c)) + "]" for c in cs) + "]"

    def port(
        self, i: int, x: Any, xt: str, y: Any, yt: str, sc: Any, m: int, H: int, vs: list[str]
    ) -> str:
        """The lemma ``P{i}`` for a port box along ``m``: zero, one copy by the split's zero,
        more copies by ``Equal.of_port_at`` from the induction hypothesis."""
        from archytaszx.semantics.lean_net import _split_net, _zero_net

        out = self.out
        isb = out.isb(m)
        px, py = _port_legs(x, m, isb), _port_legs(y, m, isb)
        if px is None or py is None or sorted(px) != sorted(py):
            raise NetUnsupported("a port box the port step does not reach")
        mx, my, sxt, syt = sc
        al, be = self.ratio(mx, m), self.ratio(my, m)
        if sp.simplify(al - be) != 0:
            raise NetUnsupported("a port box whose scalars grow unequally")
        alt = _paren(out.text(al))
        spc = f"(SpC {m})"
        zx, zy = _zero_net(x, m, isb), _zero_net(y, m, isb)
        lz = self.fam(
            zx, f"(Net.zero {spc} {xt})", zy, f"(Net.zero {spc} {yt})", self.scal_at(sc, m, "0", 0)
        )
        sx, sy = _split_net(x, m, isb, H), _split_net(y, m, isb, H)
        H2 = 2 * H
        for net in (sx, sy):
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                if out.label_chain(lab)[:1] == (m,):
                    out.peel_reg(lab, H2)
        sxt_, syt_ = f"(Net.split {spc} {H} {xt})", f"(Net.split {spc} {H} {yt})"
        l1 = self.fam(
            _zero_net(sx, m, isb),
            f"(Net.zero {spc} {sxt_})",
            _zero_net(sy, m, isb),
            f"(Net.zero {spc} {syt_})",
            self.scal_at(sc, m, "1", 1),
        )
        args = self.args
        fam = f"(F {args})" if args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(args, hyp0)})" if (args or hyp0) else "Fpos"
        name = vs[m]
        zvs = [("0" if j == m else v) for j, v in enumerate(vs)]
        kvs = [("k_" if j == m else v) for j, v in enumerate(vs)]
        jvs = [("(j_ + 1)" if j == m else v) for j, v in enumerate(vs)]
        vb = " ".join(vs)

        def ok(h: int, t: str) -> str:
            return _split_ok(m, h, t)

        self.fam_decls[i] = (
            f"theorem P{i} {self.bh} : ∀ {vb} : ℕ, Equal {self.env_at(vs)} {sxt} {xt} {syt} {yt} "
            f":= by\n  intro {vb}\n  induction {name} with\n"
            f"  | zero => exact Equal.of_zero_at {fam} {m} {H} [{', '.join(zvs)}] (by simp) {xt} "
            f"{yt} {ok(H, xt)} {ok(H, yt)} (by decide +kernel) (Equal.congr_s "
            f"{self.call(lz, *zvs)} (by ring) (by ring))\n"
            f"  | succ k_ ih =>\n    refine Equal.of_split_at {fam} {m} {H} k_ [{', '.join(kvs)}] "
            f"(by simp) {xt} {yt} {ok(H, xt)} {ok(H, yt)} (fun _ _ _ => Fam.dimv_pos _ _ _ "
            f"({fpos} _)) ?_\n    cases k_ with\n"
            f"    | zero => exact Equal.of_zero_at {fam} {m} {H2} [{', '.join(zvs)}] (by simp) "
            f"{sxt_} {syt_} {ok(H2, sxt_)} {ok(H2, syt_)} (by decide +kernel) "
            f"(Equal.congr_s {self.call(l1, *zvs)} (by ring) (by ring))\n"
            f"    | succ j_ => exact Equal.congr_s (Equal.smul {alt} (Equal.of_port_at {fam} {m} "
            f"{H} (j_ + 1) [{', '.join(jvs)}] (by simp) (by omega) {fpos} {xt} {yt} {ok(H, xt)} "
            f"{ok(H, yt)} (Spec.portOK_struct _ {m} _ _ {xt} {yt} "
            f"(by decide +kernel : Spec.portOK {spc} {xt} {yt} = true)) ih)) (by ring) (by ring)"
        )
        if i == 0:
            self.top = f"P{i}"
        return f"P{i}"

    def rule_pre(self, p: int, unfold: str, rho: str) -> str:
        """The tactic lines splitting phase vector ``p``'s rule off its stored entries."""
        rule = self.rules[p] if p < len(self.rules) else None
        if rule is None:
            return f"rw [show rV {rho} (RL {p}) j = 1 by simp [rV, rEv, RL], mul_one]\n    "
        if cp_data(rule) is not None:
            return (
                f"rw [show RL {p} = ({_lean_dm(rule[0])}, {_lean_ints(list(rule[1]))}) from rfl, "
                f"rV_mark, mul_one]\n    "
            )
        from archytaszx.algebra.phase import INDEX

        k = sp.Symbol("k_lean_index", integer=True, nonnegative=True)
        spec = self.out.phase_specs[p][1]
        assert spec is not None
        text = self.out.sym.render(spec.xreplace({INDEX: k}), "ℝ", "ℝ")
        dm, cs = _lean_dm(rule[0]), _lean_ints(list(rule[1]))
        return (
            f"simp only [{unfold}, Φ]\n    have hr := rule_ex {rho} {dm} {cs} (fun k : ℕ => "
            f"{text}) (fun k => by simp [DMon.den, npow, ρ, rEv]; push_cast; ring) "
            f"(by decide) j\n    try simp only at hr\n    rw [ex_add, hr, show RL {p} = "
            f"({dm}, {cs}) from rfl]\n    congr 1\n    "
        )

    def tab_row(self, p: int, row: list[tuple[int, CMon, sp.Expr]], chunked: bool = False) -> bool:
        """Whether ``ptOK`` holds for phase vector ``p``'s flat table (with ``chunked``, one
        read in chunks) and stored ``row``."""
        from archytaszx.semantics.lean_net import TABLE_CHUNK

        tab = getattr(self.out, "phase_tables", {}).get(p)
        if tab is None or (len(tab[0]) > TABLE_CHUNK) != chunked:
            return False
        nums, den = tab
        keys = [j for j, _, _ in row]
        if len(set(keys)) != len(keys):
            return False
        for j, m, _ in row:
            n = nums[j] if j < len(nums) else 0
            if m != ph_c(Fraction(n, den), []):
                return False
        return all(n % den == 0 or i in keys for i, n in enumerate(nums))

    def ph_proof(
        self, p: int, row: list[tuple[int, CMon, sp.Expr]], unfold: str, rho: str, gsa: str
    ) -> str:
        if not row:
            return f"simp [{unfold}, Φ, PT, phF, ex_zero]"
        if self.tab_row(p, row):
            den = self.out.phase_tables[p][1]
            return (
                f"simp only [{unfold}, Φ]\n    exact tab_ph {rho} _ {den} _ (by norm_num) _ "
                "(by decide +kernel) j"
            )
        rule = self.rules[p] if p < len(self.rules) else None
        if cp_data(rule) is not None and self.tab_row(p, row, chunked=True):
            from archytaszx.semantics.lean_net import TABLE_CHUNK

            nums, den = self.out.phase_tables[p]
            name = self.out.table_names[tuple(nums)]
            ok = "(by decide +kernel)"
            if p in self.cp_rows and list(nums) == [
                cp_num(self.cp_rows[p][1], j) % self.cp_rows[p][1][1]
                for j in range(self.cp_rows[p][0])
            ]:
                big_d, q = self.cp_rows[p]
                qt = f"(({q[0]}, {q[1]}, {q[2]}, {q[3]}) : ℕ × ℕ × ℤ × ℕ)"
                ok = (
                    f"(ptOK_cpRow_of _ {big_d} {qt} _ (by decide) (by decide +kernel) "
                    "(by decide +kernel))"
                )
            return (
                f"simp only [{unfold}, Φ]\n    exact tab_phC {rho} {name} {TABLE_CHUNK} "
                f"(by norm_num) {den} _ (by norm_num) _ (by decide +kernel) {ok} j"
            )
        lines = [
            (
                f"simp only [{unfold}, Φ, PT, phF, List.getD_cons_succ, List.getD_cons_zero, "
                "List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]"
            )
        ]
        indent = ""
        for idx, (j, _, v) in enumerate(row):
            q, ks, z = self.atoms.ks(v, True)
            theta = self.out.sym.render(v, "ℝ", "ℝ")
            lines.append(f"{indent}by_cases h{idx} : j = {j}")
            others = "".join(
                f"show ((({j} : ℕ) = {k}) ↔ False) from by decide, " for k, _, _ in row if k != j
            )
            lines.append(
                f"{indent}· subst h{idx}; simp only [{others}eq_self_iff_true, if_true, "
                "if_false, add_zero, List.getD_cons_succ, List.getD_cons_zero]"
            )
            if z == 0:
                lines.append(
                    f"{indent}  rw [phC_den {rho} {gsa} (fun _ => rfl) {_lean_q(q)} "
                    f"{_lean_ints(ks)} {_paren(theta)} (by first | (simp [lin, GS]; done) "
                    "| (simp [lin, GS]; push_cast; ring))]; ring"
                )
            else:
                zt = self.out.sym.render(z, "ℤ", "ℤ")
                lines.append(
                    f"{indent}  rw [phC_den_int {rho} {gsa} (fun _ => rfl) {_lean_q(q)} "
                    f"{_lean_ints(ks)} {_paren(theta)} ⟨{zt}, by first | "
                    "(simp [lin, GS]; done) | (simp [lin, GS]; push_cast; ring)⟩]; ring"
                )
            lines.append(f"{indent}· ")
            indent += "  "
        hs = ", ".join(f"h{i}" for i in range(len(row)))
        idxs = "[" + ", ".join(str(j) for j, _, _ in row) + "]"
        lines[-1] += (
            f"first | (simp [{hs}, ex_zero]; done) | (simp [{hs}, ex_zero]; first "
            f"| refine tab_zeroZ _ {idxs} _ j ?_ ?_ | refine tab_zero _ {idxs} _ j ?_ ?_) "
            f"<;> first | decide | simp [{hs}]"
        )
        return "\n    ".join(lines)

    def chk_decls(self, a: str, b: str) -> list[str]:
        """``chk : symCheck .. a b S0 S1 = true`` (see :meth:`check_decls`)."""
        assert self.s0 is not None and self.s1 is not None
        return self.check_decls(
            "chk",
            "/-- The kernel's normalisation of the scaled difference. -/\n",
            [(a, a, self.a.net), (b, b, self.b.net)],
            self.s0,
            self.s1,
            "S0",
            "S1",
        )

    def rx_decls(
        self,
        name: str,
        doc: str,
        own: Mapping[str, tuple[str, Any]],
        na_: tuple[str, str, Any],
        nb_: tuple[str, str, Any],
        s0: Sequence[STm],
        s1: Sequence[STm],
        s0t: str,
        s1t: str,
    ) -> list[str]:
        """The check of an export with compact tables: each network's product unnormalised
        (``trNetG`` with the identity), the scaled difference normalised by ``rxNf`` with the
        labels' ranges."""
        (ta, a, _), (tb, b, nb) = na_, nb_
        labs = dict(self.out.labs)
        out: list[str] = []
        e: dict[str, list[STm]] = {}
        nets_ok: dict[str, str] = {}
        for tag, (text, lnet) in own.items():
            nt = _paren(text)
            nodes, internal, bnd = self.net(lnet)
            fold = self.fold
            got = (tr_net_k if fold else tr_net_id)(self.dtf, self.ptf, labs, nodes, internal)
            if got is None:
                raise NetUnsupported("a compact network whose products capture variables")
            e[tag] = got
            out.append(f"def {tag}X : List STm := {_lean_ex(got)}")
            trn = "trNetK" if fold else "trNetG (fun _ e => e)"
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {tag}R : {trn} DT PT RL "
                f"labs {nt} = some {tag}X := by decide +kernel"
            )
            if all(v == 0 for v in labs.values()) and 0 in self.ix:
                out.append(f"def {tag}MK : ℕ := {hex(_mask((*bnd, *internal)))}")
                out.append(
                    f"set_option maxHeartbeats 0 in\ntheorem {name}c{tag} : netChkF FL DT PT RL "
                    f"labs IX {tag}MK {nt} = true := by decide +kernel"
                )
                nets_ok[tag] = f"(netChkF_ok FL DT PT RL labs IX {tag}MK {nt} {name}c{tag})"
            else:
                out.append(
                    f"set_option maxHeartbeats 0 in\ntheorem {name}c{tag} : netChk FL DT PT RL "
                    f"labs IX {nt} = true := by decide +kernel"
                )
                nets_ok[tag] = f"{name}c{tag}"
        tb2 = tb if tb in own else ta
        neg = ex_neg(s1)
        if not (ex_mul_ok(s0, e[ta]) and ex_mul_ok(neg, e[tb2])):
            raise NetUnsupported("a compact export whose products capture variables")
        diff = ex_mul(s0, e[ta]) + ex_mul(neg, e[tb2])
        fold = self.fold
        rxs = (rx_cert_f if fold else rx_cert)(
            self.fl, diff, lab_ranges(labs, self.dtf, self.ix, self.net(nb)[2])
        )
        if rxs is None:
            raise NetUnsupported("the compact residual does not cancel")
        rx_labs, _ = _lean_rx(rxs)
        cx = "X" if len(rxs) <= 3 else ("F" if rxs[3] == "F" else "C")
        cuts: list[list[list[STm]]] = list(rxs[3:4][0]) if cx == "C" else []  # type: ignore[misc]
        rxn = "rxNf" if cx == "X" else f"rxNf{cx}"
        out.append(f"def {name}RR : List (ℕ × ℕ) := {rx_labs}")
        rr = f"{name}RR"
        term_rows, steps_by = [], []
        for i, (rounds, (terms, last)) in enumerate(zip(rxs[1], rxs[2], strict=True)):
            names = []
            tdefs = [f"def {name}T{i}_{k} : STm := {_lean_tm(t)}" for k, t in enumerate(terms)]
            if cx == "C":
                out += tdefs
            for k, (c, es) in enumerate(rounds):
                hn = f"{name}H{i}_{k}"
                names.append(hn)
                if cx == "C":
                    out += self.chunk_decls(name, i, k, c, es, cuts[i][k])
                    continue
                hint = ", ".join(f"({_rn(x)}, {_rn(y)}, {_rn(z)})" for x, y, z in es)
                out.append(f"def {hn} : RHint := ({_rn(c)}, [{hint}])")
            if cx != "C":
                out += tdefs
            m = len(rounds)
            for k in range(m):
                res = "none" if last is None and k == m - 1 else f"some {name}T{i}_{k + 1}"
                proof = "by decide +kernel"
                if cx == "C":
                    proof = (
                        f"by\n  rw [{name}H{i}_{k}, normTmC_one _ _ _ _ _ _ {name}E{i}_{k}]\n"
                        "  decide +kernel"
                    )
                out.append(
                    f"set_option maxHeartbeats 0 in\ntheorem {name}S{i}_{k} : normTm{cx} {rr} FL "
                    f"[{name}H{i}_{k}] {name}T{i}_{k} = {res} := {proof}"
                )
            row = "[" + ", ".join(names) + "]"
            term_rows.append(row)
            lemma = [
                f"normTm{cx}_{'stop' if last is None and k == m - 1 else 'step'} _ _ _ _ _ "
                + ("" if last is None and k == m - 1 else "_ ")
                + f"{name}S{i}_{k}"
                for k in range(m)
            ]
            res = "none" if last is None else f"some {name}T{i}_{m}"
            proof = f"by rw [{', '.join(lemma)}]; try rfl" if m else "rfl"
            out.append(
                f"theorem {name}N{i} : normTm{cx} {rr} FL {row} {name}T{i}_0 = {res} := {proof}"
            )
            steps_by.append(
                f"nfGo{cx}_{'none' if last is None else 'some'} _ _ _ _ _ _ "
                + ("" if last is None else "_ ")
                + f"_ {name}N{i}"
            )
        out.append(f"def {name}XH : List (List RHint) := [{', '.join(term_rows)}]")
        e0, e1 = f"{ta}X", f"{tb2}X"
        dtext = f"(exMul {s0t} {e0} ++ exMul (exNeg {s1t}) {e1})"
        terms0 = "[" + ", ".join(f"{name}T{i}_0" for i in range(len(rxs[2]))) + "]"
        parts = [
            ("p", f"{a}.bnd.isPerm {b}.bnd"),
            ("k", f"(exConst {s0t} && exConst {s1t})"),
            ("m", f"(exMulOK {s0t} {e0} && exMulOK (exNeg {s1t}) {e1})"),
            ("r", f"labOK labs IX DT {a}.bnd {name}RR"),
        ]
        for sfx, stmt in parts:
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {name}{sfx} : {stmt} = true := "
                "by decide +kernel"
            )
        out.append(
            f"set_option maxHeartbeats 0 in\ntheorem {name}d : {dtext} = {terms0} := "
            "by decide +kernel"
        )
        out.append(
            f"set_option maxHeartbeats 0 in\n{doc}theorem {name}z : ({rxn} {rr} FL {name}XH "
            f"{dtext}).isEmpty = true := by\n  rw [{name}d]\n  unfold {rxn} {name}XH\n"
            f"  rw [{', '.join(steps_by)}]\n  decide +kernel"
        )
        fl_ok = self.call("flOK")
        self.g_lemma[name] = "equal_symSK" if fold else "equal_symSR"
        nfs = "" if fold else "(fun _ e => e) (fun _ e => e) (fun _ _ _ => rfl) (fun _ _ _ => rfl) "
        self.g_args[name] = (
            f"{nfs}_ _ {ta}R "
            f"{tb2}R {name}p {nets_ok[ta]} {nets_ok[tb2]} {name}k {name}m {name}RR "
            f"({rxn} {name}RR FL {name}XH) (fun e ν hν => {rxn}_den {self.rho} {self.hr} {name}RR "
            f"FL {fl_ok} {name}XH (fun e => e) (fun _ _ => rfl) e ν hν) {name}z {name}r"
        )
        return ["noncomputable section", *out, "end"]

    @staticmethod
    def chunk_decls(
        name: str, i: int, k: int, c: int, es: Sequence[tuple[int, int, int]], cut: Sequence[STm]
    ) -> list[str]:
        """Round ``k`` of term ``i`` of the carry route: the hint ``{name}H{i}_{k}`` with its
        steps as chunks ``{name}C{i}_{k}_{c}`` of RX_STEPS steps, each run by ``elimAllCS`` from
        the term before it to the next (``{name}K{i}_{k}_{c}``, one ``decide`` each, the terms
        ``{name}U{i}_{k}_{c}`` from ``cut``), and ``{name}E{i}_{k}``: ``elimAllC`` of the round
        from ``{name}T{i}_{k}`` to its last term."""
        rr = f"{name}RR"
        chunks = [list(es[a : a + RX_STEPS]) for a in range(0, len(es), RX_STEPS)] or [[]]
        assert len(cut) == len(chunks)
        tag = f"{i}_{k}"
        terms = [f"{name}T{tag}"] + [f"{name}U{tag}_{c2 + 1}" for c2 in range(len(chunks))]
        out = [f"def {terms[c2 + 1]} : STm := {_lean_tm(t)}" for c2, t in enumerate(cut)]
        for c2, ch in enumerate(chunks):
            hint = ", ".join(f"({_rn(x)}, {_rn(y)}, {_rn(z)})" for x, y, z in ch)
            out.append(f"def {name}C{tag}_{c2} : List (ℕ × ℕ × ℕ) := [{hint}]")
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {name}K{tag}_{c2} : elimAllCS {rr} "
                f"{name}C{tag}_{c2} {terms[c2]} = some {terms[c2 + 1]} := by decide +kernel"
            )
        cat = f"{name}C{tag}_{len(chunks) - 1}"
        for c2 in reversed(range(len(chunks) - 1)):
            cat = f"{name}C{tag}_{c2} ++ ({cat})"
        out.append(f"def {name}H{tag} : RHint := ({_rn(c)}, {cat})")
        rws = [f"elimAllC_append _ _ _ _ _ {name}K{tag}_{c2}" for c2 in range(len(chunks) - 1)]
        rws.append(f"elimAllC_of_S _ _ _ _ {name}K{tag}_{len(chunks) - 1}")
        out.append(
            f"theorem {name}E{tag} : elimAllC {rr} ({cat}) {name}T{tag} = {terms[-1]} := by\n"
            f"  rw [{', '.join(rws)}]"
        )
        return out

    def check_decls(
        self,
        name: str,
        doc: str,
        nets: Sequence[tuple[str, str, Any]],
        s0: Sequence[STm],
        s1: Sequence[STm],
        s0t: str,
        s1t: str,
    ) -> list[str]:
        """``name : symCheck .. A B s0t s1t = true`` for ``nets`` = ``[(tag, text, net)]`` of
        ``A`` and ``B``: each network's ``trNodesB`` one node per ``decide`` of ``trStepS``,
        through the stated intermediate expressions, when ``tr_net`` decides the pair; else
        one ``decide`` of the whole check."""
        (ta, a, na), (tb, b, nb) = nets
        if self.fz and name == "chk":
            fzn, fzc = self.fz[0]
            whole = [
                f"/-- The characters' modulus. -/\ndef FZN : ℕ := {fzn}",
                (
                    f"/-- The cycles each character group vanishes by. -/\ndef FZC : "
                    f"FzCert := {_lean_cert(fzc)}"
                ),
                "set_option maxHeartbeats 0 in\n"
                + doc
                + f"theorem {name} : symCheckF FZN FZC FL DT PT RL labs IX {a} {b} {s0t} {s1t} = "
                "true := by decide +kernel",
            ]
        elif self.rg and name == "chk":
            whole = [
                (
                    "/-- The pins, the labels split over their ranges and each case's "
                    f"certificate. -/\ndef RGC : RgCert := {_lean_rg(self.rg[0])}"
                ),
                "set_option maxHeartbeats 0 in\n"
                + doc
                + f"theorem {name} : symCheckR RGC FL DT PT RL labs IX {a} {b} {s0t} {s1t} = "
                "true := by decide +kernel",
            ]
        else:
            whole = [
                "set_option maxHeartbeats 0 in\n"
                + doc
                + f"theorem {name} : symCheck FL DT PT RL labs IX {a} {b} {s0t} {s1t} = true := "
                "by decide +kernel"
            ]
        labs = dict(self.out.labs)
        rx_on = bool(getattr(self, "rx", None)) and name == "chk"
        own = {ta: (a, na)} if a == b else {ta: (a, na), tb: (b, nb)}
        if rx_on:
            return self.rx_decls(name, doc, own, (ta, a, na), (tb, b, nb), s0, s1, s0t, s1t)
        steps: dict[str, list[list[STm]]] = {}
        e: dict[str, list[STm]] = {}
        for tag, (_text, lnet) in own.items():
            nodes, internal, _bnd = self.net(lnet)
            st = tr_steps(self.fl, self.dtf, self.ptf, labs, nodes, internal)
            if st is None:
                return whole
            steps[tag] = st
            used = {lab for n in nodes for lab in (*n[1], *n[2])}
            sums = tuple((lab, lab_dm(self.dtf, labs, lab)) for lab in internal if lab not in used)
            e[tag] = [(t[0], sums + t[1], t[2], t[3]) for t in st[0]]
        eb = e[tb] if tb in e else e[ta]
        neg = ex_neg(s1)
        if not (ex_mul_ok(s0, e[ta]) and ex_mul_ok(neg, eb)):
            return whole
        rest = nf_ex(self.fl, ex_mul(s0, e[ta]) + ex_mul(neg, eb))
        fzs = fz_cert(rest) if rest and self.fz and name == "chk" else None
        rgs = None
        if rest and fzs is None and self.rg and name == "chk":
            rgs = rg_cert(rest, lab_ranges(labs, self.dtf, self.ix, self.net(na)[2]))
        if rest and fzs is None and rgs is None:
            return whole
        out: list[str] = []
        nfn = "nfHP" if rgs is not None else "nfH"
        if fzs is not None:
            out += [
                f"def {name}FN : ℕ := {fzs[0]}",
                f"def {name}FC : FzCert := {_lean_cert(fzs[1])}",
            ]
        if rgs is not None:
            out.append(f"def {name}RC : RgCert := {_lean_rg(rgs)}")
        texts_e: dict[str, str] = {}
        nets_ok: dict[str, str] = {}
        for tag, (text, lnet) in own.items():
            st = steps[tag]
            nt = _paren(text)
            nodes, internal, bnd = self.net(lnet)
            hints = tr_hints(self.fl, self.dtf, self.ptf, labs, nodes, internal, st)
            table = "[" + ", ".join("[" + ", ".join(map(_lean_th, h)) + "]" for h in hints) + "]"
            out.append(
                f"def {tag}H (i : ℕ) : List THint := (({table} : List (List THint))).getD i []"
            )
            args = f"({nfn} FL {tag}H) DT PT RL labs"
            m = len(lnet.nodes)
            for k, n in enumerate(lnet.nodes):
                out.append(f"def {tag}n{k} : Node := {n.lean()}")
            out.append(f"def {tag}D{m} : List Node := []")
            for k in range(m - 1, -1, -1):
                out.append(f"def {tag}D{k} : List Node := {tag}n{k} :: {tag}D{k + 1}")
            per = min(BLOCK_MAX, max(1, BLOCK_TERMS // max(1, *map(len, st))))
            marks = sorted({*range(0, m, per), m})
            pm = 0
            for k in range(m + 1):
                if k in marks:
                    out.append(f"def {tag}M{k} : ℕ := {hex(pm)}")
                if k < m:
                    pm |= _mask((*nodes[k][1], *nodes[k][2]))
            fac_names: dict[Fac, str] = {}

            def fac_name(f: Fac, tag: str = tag, names: dict[Fac, str] = fac_names) -> str:
                if f not in names:
                    names[f] = f"{tag}F{len(names)}"
                return names[f]

            stated = [_lean_ex(st[k], fac_name) for k in marks]
            out += [f"def {n} : ArchytasZX.Sym.Fac := {_lean_fac(f)}" for f, n in fac_names.items()]
            for k, text in zip(marks, stated, strict=True):
                out.append(f"def {tag}E{k} : List STm := {text}")
            out.append(
                f"theorem {tag}B{m} : trNodesM {args} {nt}.int {tag}M{m} {m} {tag}D{m} = "
                f"some {tag}E{m} := by decide +kernel"
            )
            for k, k2 in reversed(list(itertools.pairwise(marks))):
                xs = "[" + ", ".join(f"{tag}n{j}" for j in range(k, k2)) + "]"
                out.append(
                    f"set_option maxHeartbeats 0 in\ntheorem {tag}T{k} : trBlockM {args} {nt}.int "
                    f"{tag}E{k2} {tag}M{k} {k} {xs} = some {tag}E{k} := by decide +kernel"
                )
                out.append(
                    f"theorem {tag}B{k} : trNodesM {args} {nt}.int {tag}M{k} {k} {tag}D{k} = "
                    f"some {tag}E{k} :=\n  (trNodesM_block {args} {nt}.int {tag}E{k2} {xs} "
                    f"{tag}D{k2} {tag}M{k} {tag}M{k2} {k} (by decide +kernel) {tag}B{k2}).trans "
                    f"{tag}T{k}"
                )
            e_text = (
                f"(exSums (labSums DT labs ({nt}.int.filter fun l => "
                f"!(labelsOf {nt}.nodes).contains l)) {tag}E0)"
            )
            out.append(
                f"theorem {tag}R : trNetG {args} {nt} = some {e_text} :=\n"
                f"  trNetG_of {args} {nt} {tag}D0 {tag}E0 rfl\n"
                f"    ((trNodesM_eq {args} {nt}.int {tag}D0 [] 0).symm.trans {tag}B0)"
            )
            used = {lab for n in nodes for lab in (*n[1], *n[2])}
            free = [lab for lab in internal if lab not in used]
            out.append(f"def {tag}ML : ℕ := {hex(_mask(used))}")
            out.append(f"def {tag}X : List STm := exSums (labSums DT labs {free}) {tag}E0")
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {tag}XE : {e_text} = {tag}X := by\n"
                f"  rw [filter_free {nt}.int (labelsOf {nt}.nodes) {tag}ML (by decide +kernel)]\n"
                "  decide +kernel"
            )
            texts_e[tag] = f"{tag}X"
            if all(v == 0 for v in labs.values()) and 0 in self.ix:
                out.append(f"def {tag}MK : ℕ := {hex(_mask((*bnd, *internal)))}")
                out.append(
                    f"set_option maxHeartbeats 0 in\ntheorem {name}c{tag} : netChkF FL DT PT RL "
                    f"labs IX {tag}MK {nt} = true := by decide +kernel"
                )
                nets_ok[tag] = f"(netChkF_ok FL DT PT RL labs IX {tag}MK {nt} {name}c{tag})"
            else:
                out.append(
                    f"set_option maxHeartbeats 0 in\ntheorem {name}c{tag} : netChk FL DT PT RL "
                    f"labs IX {nt} = true := by decide +kernel"
                )
                nets_ok[tag] = f"{name}c{tag}"
        tb2 = tb if tb in own else ta
        e0, e1 = texts_e[ta], texts_e[tb2]
        zh = [term_hint(self.fl, t) for t in ex_mul(s0, e[ta]) + ex_mul(neg, eb)]
        out.append(f"def {name}ZH (_ : ℕ) : List THint := [" + ", ".join(map(_lean_th, zh)) + "]")
        fl_ok = self.call("flOK")
        nfz = f"({nfn} FL {name}ZH 0)"
        hnfz = f"({nfn}_den {self.rho} {self.hr} FL {fl_ok} {name}ZH 0)"
        if fzs is not None:
            nfz = f"(fzNf {name}FN {name}FC (nfH FL {name}ZH 0))"
            hnfz = f"(fzNf_den {self.rho} {self.hr} {name}FN {name}FC _ {hnfz})"
        rgx = ""
        if rgs is not None:
            nfz = f"(rgNf {name}RC ({nfn} FL {name}ZH 0))"
            hnfz = f"(rgNf_den {self.rho} {self.hr} {name}RC _ {hnfz})"
        parts = [
            ("p", f"{a}.bnd.isPerm {b}.bnd"),
            ("k", f"(exConst {s0t} && exConst {s1t})"),
            ("m", f"(exMulOK {s0t} {e0} && exMulOK (exNeg {s1t}) {e1})"),
            ("z", f"({nfz[1:-1]} (exMul {s0t} {e0} ++ exMul (exNeg {s1t}) {e1})).isEmpty"),
        ]
        if rgs is not None:
            parts.insert(3, ("r", f"labOK labs IX DT {a}.bnd {name}RC.2.1"))
            rgx = f" {name}RC.2.1"
            self.g_lemma[name] = "equal_symSR"
        for k, (sfx, stmt) in enumerate(parts):
            out.append(
                "set_option maxHeartbeats 0 in\n"
                + (doc if k == len(parts) - 1 else "")
                + f"theorem {name}{sfx} : {stmt} = true := by decide +kernel"
            )
        self.g_args[name] = (
            f"({nfn} FL {ta}H) ({nfn} FL {tb2}H) ({nfn}_den {self.rho} {self.hr} FL {fl_ok} {ta}H) "
            f"({nfn}_den {self.rho} {self.hr} FL {fl_ok} {tb2}H) _ _ "
            f"({ta}R.trans (congrArg some {ta}XE)) ({tb2}R.trans (congrArg some {tb2}XE)) "
            f"{name}p {nets_ok[ta]} {nets_ok[tb2]} {name}k {name}m{rgx} {nfz} {hnfz} {name}z"
            + (f" {name}r" if rgs is not None else "")
        )
        return ["noncomputable section", *out, "end"]

    def scalar_proof(self, model: sp.Expr, rho: str, tab: str = "S0, S1") -> str:
        """The tactic proving a scalar equal to its constant terms; roots of unity of
        symbolic order (:meth:`root_facs`) read through ``phV``."""
        roots = any(
            isinstance(x, Om) and not x.args[0].is_number for x in sp.preorder_traversal(model)
        )
        more = "phV, ipDen, imDen, List.prod_cons, " if roots else ""
        return (
            f"simp (disch := decide) only [{tab}, exDen, List.map_cons, List.map_nil, "
            f"List.sum_cons, List.sum_nil, add_zero, den_zero, bodyOf, List.prod_nil, "
            f"{more}mul_one, CMon.den_mul {rho} hρ, ofQ_den, exQ_den, rtD_den, dP_den, "
            f"phC_den {rho} _ hg] <;> "
            f"simp [DMon.den, npow, ρ, zpow_neg, inv_pow] <;> push_cast <;> ring_nf <;> "
            "(try simp) <;> (try (simp only [om_eq_ex]; norm_num))"
        )


def export_symbolic(
    left: Diagram,
    right: Diagram,
    name: str = "main",
    describe: Callable[[Diagram], str] | None = None,
    floors: Mapping[str, int] | None = None,
    shift: tuple[str, int] | None = None,
    original: tuple[dict[str, int], list[tuple[Diagram, Diagram]]] | None = None,
    fixed: Sequence[tuple[str, int] | tuple[str, int, bool]] | None = None,
) -> str:
    """Lean 4 source proving ``left = right`` for every value of the dimensions and phases by
    the verified normaliser, at ``n + c`` for every ``n`` when ``shift`` is ``(n, c)``, at
    each clause ``(n, c, at)`` of ``fixed`` (``n = c`` when ``at``, else ``n + c``); with
    ``original``, restated over the original diagrams as in
    :func:`~archytaszx.semantics.lean_net.export`;
    :class:`NetUnsupported` when the normaliser does not reach it."""
    from archytaszx.semantics.decide import dimension_floors

    floors = floors if floors is not None else dimension_floors(left, right)
    try:
        return _export_symbolic(left, right, name, describe, floors, shift, original, fixed)
    except NetUnsupported:
        if not any(isinstance(x, sp.exp) for d in (left, right) for x in _scalar_parts(d)):
            raise
    return _export_symbolic(left, right, name, describe, floors, shift, original, fixed, True)


def _scalar_parts(d: Diagram) -> Iterable[sp.Expr]:
    """The subexpressions of ``d``'s scalar."""
    return list(sp.preorder_traversal(sp.sympify(d.scalar.to_sympy())))


def _export_symbolic(
    left: Diagram,
    right: Diagram,
    name: str,
    describe: Callable[[Diagram], str] | None,
    floors: Mapping[str, int],
    shift: tuple[str, int] | None,
    original: tuple[dict[str, int], list[tuple[Diagram, Diagram]]] | None,
    fixed: Sequence[tuple[str, int] | tuple[str, int, bool]] | None,
    roots: bool = False,
) -> str:
    """:func:`export_symbolic` with ``floors`` given; with ``roots``, scalar roots of unity of
    symbolic order as constant phase factors (:meth:`_SymExport.with_roots`)."""
    ex = _SymExport(left, right, describe, floors, roots)
    ex.shift = shift
    ex.fixed = fixed
    ex.out.original = original
    global _BS_ON
    saved = _BS_ON
    _BS_ON = _bs_small(ex)
    try:
        if not ex.check():
            raise NetUnsupported(getattr(ex, "why", "the normal forms of the two networks differ"))
        return ex.source(name)
    except ScalarUnsupported as exc:
        raise NetUnsupported(str(exc)) from exc
    finally:
        _BS_ON = saved


def _bs_small(ex: _SymExport) -> bool:
    """Whether a pair without multiplicities has binder or splitter nodes and at most BS_SMALL
    nodes in all."""
    if ex.mults:
        return False
    nodes = [n for lnet in (ex.a.net, ex.b.net) for n in ex.net(lnet)[0]]
    return len(nodes) <= BS_SMALL and any(n[0] in ("B", "S") for n in nodes)
