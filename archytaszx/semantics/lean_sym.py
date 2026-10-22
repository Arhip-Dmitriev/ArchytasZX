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

import math
import re
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_net import NetUnsupported, _Export, _join, _paren
from archytaszx.semantics.lean_scalar import II, NC, Ex, Inv, Om, Rt

DMon = tuple[int, tuple[int, ...]]
IPoly = tuple[tuple[tuple[int, ...], int], ...]
Fac = tuple[DMon, IPoly]
CMon = tuple[Fraction, Fraction, tuple[int, ...], tuple[int, ...], tuple[int, ...],
             tuple[tuple[int, DMon], ...]]
STm = tuple[CMon, tuple[tuple[int, DMon], ...], tuple[Fac, ...], tuple[Fac, ...]]

OFF = 1000000
ROUNDS = 3


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
                return tuple(out[:i] + out[i + 1:])
            return tuple(out[:i] + [(m, c + c2)] + out[i + 1:])
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


def ip_subst(x: int, r: IPoly, p: IPoly) -> IPoly:
    if not any(x in m for m, _ in p):
        return p
    acc: IPoly = ()
    for m, c in reversed(p):
        rest = tuple(y for y in m if y != x)
        acc = ip_add(acc, ip_mul(ip_smul(c, ((rest, 1),)), ip_pow(r, m.count(x))))
    return acc


def ip_has(x: int, p: IPoly) -> bool:
    return any(x in m for m, _ in p)


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
            return c, tuple(out[:i] + out[i + 1:])
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
    return ip_has(x, f[1])


def fac_sub(x: int, v: IPoly, f: Fac) -> Fac:
    return (f[0], ip_subst(x, v, f[1]))


def pin_val(s: int, b: IPoly) -> IPoly:
    return ip_smul(-s, b)


def _splits(lst: Sequence[Any]) -> list[tuple[tuple[Any, ...], Any, tuple[Any, ...]]]:
    return [(tuple(lst[:i]), lst[i], tuple(lst[i + 1:])) for i in range(len(lst))]


# eliminations

def elim_unused(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]
                ) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    if d[0] != 0 and all(not fac_has(x, f) for f in (*ds, *ps)):
        return rt_d(d, 2), ds, ps
    return None


def elim_delta1(x: int, d: DMon, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...],
                ps: tuple[Fac, ...]) -> tuple[CMon, tuple[Fac, ...], tuple[Fac, ...]] | None:
    lin = ip_lin(x, f[1])
    if lin is None:
        return None
    a, b = lin
    s = ip_const(a)
    if s is None:
        return None
    if (s in (1, -1)) and d[0] != 0 and dm_dvd(f[0], d) and all(
            not fac_has(x, g) or dm_dvd(g[0], f[0]) for g in (*pre, *post, *ps)):
        v = pin_val(s, b)
        return (rt_d(dm_div(d, f[0]), 2), tuple(fac_sub(x, v, g) for g in (*pre, *post)),
                tuple(fac_sub(x, v, g) for g in ps))
    return None


def elim_delta(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    for pre, f, post in _splits(ds):
        e = elim_delta1(x, d, pre, f, post, ps)
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
            return (rt_d(d, 2), ds + ((d, big_a),),
                    tuple(f for f in ps if not fac_has(x, f)) + ((d, big_b),))
    return None


def elim_gcd1(x: int, d: DMon, pre: tuple[Fac, ...], f: Fac, post: tuple[Fac, ...],
              ps: tuple[Fac, ...]) -> Any:
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


def elim_var(x: int, d: DMon, ds: tuple[Fac, ...], ps: tuple[Fac, ...]) -> Any:
    for fn in (elim_unused, elim_delta, elim_char, elim_gcd):
        e = fn(x, d, ds, ps)
        if e is not None:
            return e
    return None


def elim_step(t: STm) -> STm | None:
    for pre, (x, d), post in _splits(t[1]):
        if x in [y for y, _ in (*pre, *post)]:
            continue
        e = elim_var(x, d, t[2], t[3])
        if e is not None:
            return (cm_mul(t[0], e[0]), pre + post, e[1], e[2])
    return None


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


def pivot_of(p: IPoly) -> tuple[int, int, IPoly] | None:
    return _pick(p, lambda c: c in (1, -1))


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


def step_of(f: Fac) -> Callable[[Fac], Fac] | None:
    q = pivot_of(f[1])
    if q is not None:
        v = pin_val(q[1], q[2])
        return lambda g: piv_sub(f[0], q[0], v, g)
    q2 = div_of(f[1])
    if q2 is not None:
        return lambda g: div_sub(f[0], q2[0], q2[1], q2[2], g)
    return None


def piv_pass(n: int, a: tuple[Fac, ...], b: tuple[Fac, ...], ps: tuple[Fac, ...]
             ) -> tuple[tuple[Fac, ...], tuple[Fac, ...]]:
    while True:
        if n == 0:
            return a + b, ps
        if not b:
            return a, ps
        n -= 1
        f, b = b[0], b[1:]
        st = step_of(f)
        if st is None:
            a = a + (f,)
        else:
            a = tuple(st(g) for g in a) + (f,)
            b = tuple(st(g) for g in b)
            ps = tuple(st(g) for g in ps)


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
        return (t[0], tuple((y, p[1]) if p[0] == x else p for p in t[1]),
                tuple(fac_sub(x, v, f) for f in t[2]), tuple(fac_sub(x, v, f) for f in t[3]))
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
                return tuple(out[:i] + out[i + 1:])
            out[i] = with_q(u, q)
            return tuple(out)
    return tuple(out + ([t] if t[0][0] != 0 else []))


def nf_ex(fl: Sequence[int], e: Sequence[STm]) -> tuple[STm, ...]:
    acc: tuple[STm, ...] = ()
    for t in e:
        n = norm_tm(fl, ROUNDS, t)
        if n is not None:
            acc = tm_add(n, acc)
    return acc


# networks

def sum_l(ls: Sequence[int]) -> IPoly:
    acc: IPoly = ()
    for x in reversed(ls):
        acc = ip_add(acc, ip_v(x))
    return acc


def z_legs(d: DMon, s: int, ls: Sequence[int]) -> tuple[Fac, ...]:
    return tuple((d, ip_sub(ip_v(x), ip_v(s))) for x in ls)


def spider(k: CMon, d: DMon, s: int, pt: Sequence[tuple[int, CMon]], ds: tuple[Fac, ...],
           ps: tuple[Fac, ...]) -> list[STm]:
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
        out.append((of_q(1), (), acc + ((d, ip_sub(ip_v(o), ip_v(i))),) + w_zeros(d, os[p + 1:]),
                    ()))
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


def tr_node(dt: Callable[[int], DMon], pt: Callable[[int], Sequence[tuple[int, CMon]]], s: int,
            kind: str, os: Sequence[int], is_: Sequence[int], dim: int, ph: int
            ) -> list[STm] | None:
    d = dt(dim)
    rf = r_f(s, getattr(pt(ph), "rule", None))
    if kind == "Z":
        return spider(of_q(1), d, s, pt(ph), z_legs(d, s, (*os, *is_)), rf)
    if kind == "X":
        poly = ip_mul(ip_v(s), ip_sub(sum_l(os), sum_l(is_)))
        return spider(rt_d(d, -(len(os) + len(is_))), d, s, pt(ph), (), ((d, poly),) + rf)
    if kind == "F" and len(os) == 1 and len(is_) == 1:
        return [(rt_d(d, -1), (), (), ((d, ip_mul(ip_v(os[0]), ip_v(is_[0]))),))]
    if kind == "T" and len(os) == 1 and len(is_) == 1:
        o, i = os[0], is_[0]
        return [(of_q(1), (), ((d, ip_v(o)),), ()),
                (of_q(1), (), ((d, ip_sub(ip_v(o), ip_v(i))),), ()),
                (of_q(-1), (), ((d, ip_v(o)), (d, ip_v(i))), ())]
    if kind == "Ti" and len(os) == 1 and len(is_) == 1:
        o, i = os[0], is_[0]
        return [(of_q(1), (), ((d, ip_sub(ip_v(o), ip_v(i))),), ()),
                (of_q(-1), (), ((d, ip_v(o)),), ()),
                (of_q(1), (), ((d, ip_v(o)), (d, ip_v(i))), ())]
    if kind == "W" and len(is_) == 1:
        i = is_[0]
        return w_s(d, i, (), os) + [(of_q(1 - len(os)), (), ((d, ip_v(i)),) + w_zeros(d, os),
                                     ())]
    if kind == "I" and len(os) == 1 and len(is_) == 1:
        return [(of_q(1), (), ((d, ip_sub(ip_v(os[0]), ip_v(is_[0]))),), ())]
    return None


def ex_mul(a: Sequence[STm], b: Sequence[STm]) -> list[STm]:
    return [tm_mul(s, t) for s in a for t in b]


def ex_mul_ok(a: Sequence[STm], b: Sequence[STm]) -> bool:
    return all(mul_ok(s, t) for s in a for t in b)


def tr_nodes(fl: Sequence[int], dt: Callable[[int], DMon],
             pt: Callable[[int], Sequence[tuple[int, CMon]]], labs: dict[int, int],
             internal: Sequence[int], pre: tuple[int, ...], i: int,
             nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]]
             ) -> list[STm] | None:
    """``trNodesB``."""
    if not nodes:
        return [(of_q(1), (), (), ())]
    _kind, os, is_, _dim, _ph = nodes[0]
    legs = (*os, *is_)
    a = tr_node(dt, pt, OFF + i, *nodes[0])
    b = tr_nodes(fl, dt, pt, labs, internal, pre + tuple(legs), i + 1, nodes[1:])
    if a is None or b is None or not ex_mul_ok(a, b):
        return None
    first = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal
                  if lab in legs and lab not in pre)
    return list(nf_ex(fl, [(t[0], first + t[1], t[2], t[3]) for t in ex_mul(a, b)]))


def tr_nodes_full(dt: Callable[[int], DMon], pt: Callable[[int], Sequence[tuple[int, CMon]]],
                  i: int, nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]]
                  ) -> list[STm] | None:
    """``trNodesF``."""
    if not nodes:
        return [(of_q(1), (), (), ())]
    a = tr_node(dt, pt, OFF + i, *nodes[0])
    b = tr_nodes_full(dt, pt, i + 1, nodes[1:])
    if a is None or b is None or not ex_mul_ok(a, b):
        return None
    return ex_mul(a, b)


def tr_net_full(fl: Sequence[int], dt: Callable[[int], DMon],
                pt: Callable[[int], Sequence[tuple[int, CMon]]], labs: dict[int, int],
                nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
                internal: Sequence[int]) -> list[STm] | None:
    """``trNetF``."""
    e = tr_nodes_full(dt, pt, 0, nodes)
    if e is None:
        return None
    sums = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal)
    return [(t[0], sums + t[1], t[2], t[3]) for t in e]


def lab_dm(dt: Callable[[int], DMon], labs: dict[int, int], lab: int) -> DMon:
    return dt(labs.get(lab, 0))


def tr_net(fl: Sequence[int], dt: Callable[[int], DMon],
           pt: Callable[[int], Sequence[tuple[int, CMon]]], labs: dict[int, int],
           nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
           internal: Sequence[int]) -> list[STm] | None:
    e = tr_nodes(fl, dt, pt, labs, internal, (), 0, nodes)
    if e is None:
        return None
    used = {lab for n in nodes for lab in (*n[1], *n[2])}
    sums = tuple((lab, lab_dm(dt, labs, lab)) for lab in internal if lab not in used)
    return [(t[0], sums + t[1], t[2], t[3]) for t in e]


def node_chk(fl: Sequence[int], dt: Callable[[int], DMon],
             pt: Callable[[int], Sequence[tuple[int, CMon]]], dim: int, ph: int) -> bool:
    d = dt(dim)
    return d[0] != 0 and all(j < dm_den(d, lambda i: flv(fl, i)) for j, _ in pt(ph))


def net_chk(fl: Sequence[int], dt: Callable[[int], DMon],
            pt: Callable[[int], Sequence[tuple[int, CMon]]], labs: dict[int, int],
            ix: Sequence[int], nodes: Sequence[tuple[str, Sequence[int], Sequence[int], int, int]],
            internal: Sequence[int], bnd: Sequence[int]) -> bool:
    for _kind, os, is_, dim, ph in nodes:
        if not node_chk(fl, dt, pt, dim, ph) or dim not in ix:
            return False
        for lab in (*os, *is_):
            if not (lab < OFF and (lab in bnd or lab in internal)
                    and lab_dm(dt, labs, lab) == dt(dim) and labs.get(lab, 0) in ix):
                return False
    return all(labs.get(lab, 0) in ix for lab in internal)


def ex_neg(e: Sequence[STm]) -> list[STm]:
    return [with_q(t, -t[0][0]) for t in e]


def ex_const(e: Sequence[STm]) -> bool:
    return all(not t[1] and not t[2] and not t[3] for t in e)


def sym_check(fl: Sequence[int], dt: Callable[[int], DMon],
              pt: Callable[[int], Sequence[tuple[int, CMon]]], labs: dict[int, int],
              ix: Sequence[int], n0: Any, n1: Any, s0: Sequence[STm], s1: Sequence[STm]) -> bool:
    """``symCheck``; a network is ``(nodes, internal, boundary)``."""
    if sorted(n0[2]) != sorted(n1[2]):
        return False
    if not (net_chk(fl, dt, pt, labs, ix, *n0) and net_chk(fl, dt, pt, labs, ix, *n1)):
        return False
    if not (ex_const(s0) and ex_const(s1)):
        return False
    for tr in (tr_net, tr_net_full):
        e0 = tr(fl, dt, pt, labs, n0[0], n0[1])
        e1 = tr(fl, dt, pt, labs, n1[0], n1[1])
        if e0 is None or e1 is None:
            continue
        neg = ex_neg(s1)
        if ex_mul_ok(s0, e0) and ex_mul_ok(neg, e1) and not nf_ex(
                fl, ex_mul(s0, e0) + ex_mul(neg, e1)):
            return True
    return False


# the export

def _dmon(e: sp.Expr, syms: Sequence[sp.Symbol]) -> DMon:
    """A dimension expression as ``c Π d_i^e_i``."""
    e = sp.expand(sp.sympify(e))
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
    def split(v: sp.Expr, drop: bool = False) -> tuple[sp.Rational, dict[sp.Expr, sp.Rational],
                                                         sp.Expr]:
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
    rest only at the boundary, bare symbols first; :class:`NetUnsupported` for any other box."""
    from archytaszx.diagram.bangbox import crossing_wires

    names = set()
    for d in (left, right):
        for box in d.bang_boxes.values():
            if box.is_node_scope and crossing_wires(d, box.node_scope):
                raise NetUnsupported("a box meeting the rest at a wire")
            names.add(box.multiplicity.to_sympy())
    return sorted(names, key=lambda m: (not m.is_Symbol, str(m)))


def _peeled(split: Any, net: Any) -> Any:
    """The nodes, summed and boundary labels of ``split`` that ``net`` lacks."""
    from archytaszx.semantics.lean_net import LNet

    rest = list(split.nodes)
    for n in net.nodes:
        if n not in rest:
            raise NetUnsupported("the split network does not contain the network")
        rest.remove(n)
    return LNet(tuple(rest), tuple(x for x in split.summed if x not in net.summed),
                tuple(x for x in split.bnd if x not in net.bnd))


def _split_ok(m: int, h: int, t: str) -> str:
    """A proof of ``Spec.splitOK`` at the family from the kernel's check at ``SpC m``."""
    return (f"(Spec.splitOK_struct _ {m} _ _ {h} {t} "
            f"(by decide +kernel : Spec.splitOK (SpC {m}) {h} {t} = true))")


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

    def __init__(self, left: Diagram, right: Diagram, describe: Callable[[Diagram], str] | None,
                 floors: Mapping[str, int]) -> None:
        out = _Export(describe or (lambda d: f"a diagram of {len(d.nodes)} nodes"))
        out.mixed_ok, out.mixed = True, set()
        self.out = out
        left, alias_l = _unnest(left)
        right, alias_r = _unnest(right)
        self.alias = {**alias_l, **alias_r}
        dim_syms = {x for d in (left, right) for n in d.nodes.values()
                    for p in (*n.inputs, *n.outputs) for x in p.dim.to_sympy().free_symbols}
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
        self.m0 = out.scalar(left.scalar)
        self.m1 = out.scalar(right.scalar)
        self.common: Any = None
        opaque = {v[0] for v in out.sc.opaques.values()}
        varying = set(out.mult_exprs)
        m0, m1 = sp.sympify(self.m0), sp.sympify(self.m1)
        if (m0.free_symbols | m1.free_symbols) & (opaque | varying) and m1 != 0:
            ratio = sp.simplify(sp.powsimp(m0 / m1, force=True))
            if not ratio.free_symbols & (opaque | varying):
                self.common = m1
                self.stated = (_paren(out.text(m0)), _paren(out.text(m1)))
                self.m0, self.m1 = ratio, sp.Integer(1)
        if out.sc.opaques and self.common is None:
            raise NetUnsupported("the diagrams' scalars differ by a factor outside the exact model")
        if out.family and not self.mults:
            raise NetUnsupported("bang boxes")
        out.set_params()
        params = [(raw, n, ty) for raw, (n, ty) in out.sym.params.items()
                  if raw != "k_lean_index" and raw not in out.mults]
        self.params = params
        def atom(e: sp.Expr) -> bool:
            e = sp.sympify(e)
            return bool(e.is_Pow and e.args[0].is_Symbol and not e.args[1].is_number)

        dsyms = sorted({x for e in out.dim_exprs for x in (
            {sp.sympify(e).args[0]} if atom(e) else sp.sympify(e).free_symbols)}, key=str)
        self.nsym = len(dsyms)
        dsyms += [sp.sympify(e) for e in out.dim_exprs if atom(e)]
        self.dsyms = dsyms
        self.dnames = [out.sym.name(x, "ℕ") if x.is_Symbol else _paren(out.sym.render(x, "ℕ", "ℕ"))
                       for x in dsyms]
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
            values += list(entries.values())
        self.exs: list[sp.Expr] = []
        for model in (self.m0, self.m1):
            for x in sp.preorder_traversal(sp.sympify(model)):
                if isinstance(x, Ex):
                    self.exs.append(sp.expand(x.args[0] / (2 * sp.pi * sp.I)))
        self.atoms = _Atoms(self.exs, values)
        self.pt: list[list[tuple[int, CMon, sp.Expr]]] = []
        for entries, _ in out.phase_specs:
            row = []
            for j, v in sorted(entries.items()):
                q, ks, _ = self.atoms.ks(v, True)
                row.append((int(j), ph_c(q, ks), v))
            self.pt.append(row)
        fl = {str(k): int(v) for k, v in floors.items()}
        for n in (*self.a.net.nodes, *self.b.net.nodes):
            if n.kind in ("Z", "X") and n.ph < len(self.pt) and self.pt[n.ph]:
                need = max(j for j, _, _ in self.pt[n.ph]) + 1
                c, e = self.dt[n.dim]
                if sum(e) == 1 and c == 1:
                    name = str(dsyms[e.index(1)])
                    fl[name] = max(fl.get(name, 1), need)
        self.fl = trim([fl.get(str(x), 0) if fl.get(str(x), 0) > 1 else 0 for x in dsyms])
        msyms = set(out.mult_exprs)
        varying = any(sp.sympify(m).free_symbols & msyms for m in (self.m0, self.m1))
        self.s0 = None if varying else self.scalar_terms(self.m0)
        self.s1 = None if varying else self.scalar_terms(self.m1)
        self.stabs: dict[str, tuple[str, str, list[STm]]] = {}
        self.stab_decls: list[str] = []
        self.ix = [i for i in range(len(self.dt)) if i not in out.bund_of]
        self.fam_decls: list[str] = []
        self.net_decls: list[str] = []
        self.planned: bool | None = None

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
        cs = [poly.coeff_monomial(INDEX ** i) for i in range(poly.degree() + 1)]
        if not all(c.is_Integer for c in cs) or cs[0] != 0:
            raise NetUnsupported(f"the phase rule {turns} has no integer form")
        return _dmon(den, self.dsyms), tuple(int(c) for c in cs)

    @staticmethod
    def net(lnet: Any) -> tuple[list[tuple[str, tuple[int, ...], tuple[int, ...], int, int]],
                                tuple[int, ...], tuple[int, ...]]:
        nodes = []
        for n in lnet.nodes:
            if not isinstance(n.kind, str):
                raise NetUnsupported(f"the kind {n.kind}")
            nodes.append((n.kind, n.outs, n.ins, n.dim, n.ph))
        return nodes, lnet.summed, lnet.bnd

    def check(self) -> bool:
        if not self.mults:
            return sym_check(self.fl, self.dtf, self.ptf, dict(self.out.labs), self.ix,
                             self.net(self.a.net), self.net(self.b.net), self.s0, self.s1)
        if self.planned is None:
            try:
                self.texts()
                self.fam(self.a.net, self.a.name, self.b.net, self.b.name,
                         (sp.sympify(self.m0), sp.sympify(self.m1), self.s0t, self.s1t))
                self.planned = True
            except NetUnsupported as exc:
                self.planned, self.why = False, str(exc)
        return self.planned

    # scalars

    def scalar_terms(self, model: sp.Expr) -> list[STm]:
        """The scalar as constant terms, each with its Lean constructor in ``self.texts_of``."""
        terms: list[STm] = []
        self.texts_of: dict[STm, str] = getattr(self, "texts_of", {})
        for term in sp.Add.make_args(sp.expand(sp.sympify(model))):
            cm, text = self.monomial(term)
            t: STm = (cm, (), (), ())
            terms.append(t)
            self.texts_of[t] = text
        return terms

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
            elif isinstance(base, Inv) and (base.args[0] in self.dsyms
                                            or isinstance(base.args[0], NC)):
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
        for i, nm in enumerate(self.dnames[:self.nsym]):
            low = self.fl[i] if i < len(self.fl) else 0
            hyps.append(f"(h_{nm} : {max(low, 1)} ≤ {nm})")
        hyps += [f"(hc{j} : {c})" for j, c in enumerate(self.constraints)]
        hyp_args = " ".join([*(f"h_{nm}" for nm in self.dnames[:self.nsym]),
                             *(f"hc{j}" for j in range(len(self.constraints)))])
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
        return ct, f"Equal.congr_s (Equal.smul {ct} ({proof})) (by ring) (by ring)"

    def stated_s(self, side: int) -> str:
        """The scalar the theorem states for ``side``."""
        if self.common is None:
            return self.s0t if side == 0 else self.s1t
        return self.stated[side]

    def call(self, lemma: str, *more: str) -> str:
        text = _join(lemma, self.ah, *more)
        return f"({text})" if " " in text else text

    def sym(self, envt: str, ks: str, n0: str, n1: str, a0: str, a1: str, chk: str,
            l0: str = "scal0", l1: str = "scal1") -> str:
        """``equal_sym`` at the environment ``envt``, multiplicities ``ks``."""
        return (f"equal_sym {envt} {self.rho} {self.hr} FL {self.call('flOK')} DT PT RL labs IX "
                f"{self.call('envD', ks)} {self.call('envP', ks)} rfl {n0} {n1} {a0} {a1} _ _ "
                f"{self.call(l0)} {self.call(l1)} {chk}")

    def source(self, name: str) -> str:
        out = self.out
        self.texts()
        bh, args, rho = self.bh, self.args, self.rho
        dims = "[" + ", ".join(self.dnames) + "]"
        gs = [out.sym.render(b / L, "ℝ", "ℝ") for b, L in zip(self.atoms.bases, self.atoms.scale,
                                                                strict=True)]
        gtext = "[" + ", ".join(gs) + "]"
        gsa = f"(GS {args})" if args else "GS"
        family = bool(self.mults)
        kb = "(ks : ℕ → ℕ)" if family else ""
        fam = f"(F {args})" if args else "F"
        env = f"({fam}.env ks)" if family else self.env_at()
        unfold = "Fam.env, F, bundL, Fam.dimv, List.lookup" if family else "E"
        decls = ["open ArchytasZX.Sym\n\nset_option linter.unusedSimpArgs false"]
        decls.append("/-- Each dimension index as a monomial in the dimension symbols. -/\n"
                     "def DT (i : ℕ) : DMon := ([" + ", ".join(_lean_dm(m) for m in self.dt)
                     + "] : List DMon).getD i ⟨1, []⟩")
        rows = []
        for row in self.pt:
            items = []
            for j, _, v in row:
                q, ks, _ = self.atoms.ks(v, True)
                items.append(f"({j}, phC {_lean_q(q)} {_lean_ints(ks)})")
            rows.append("[" + ", ".join(items) + "]")
        decls.append("/-- Each phase vector's stored entries. -/\n"
                     "def PT (i : ℕ) : List (ℕ × CMon) := ([" + ", ".join(rows)
                     + "] : List (List (ℕ × CMon))).getD i []")
        rl = ", ".join(f"({_lean_dm(r[0])}, {_lean_ints(list(r[1]))})" if r is not None
                       else "(⟨1, []⟩, [])" for r in self.rules)
        decls.append("/-- Each phase vector's rule as a dimension monomial and the coefficients of "
                     "its numerator. -/\ndef RL (i : ℕ) : DMon × List ℤ := ([" + rl
                     + "] : List (DMon × List ℤ)).getD i (⟨1, []⟩, [])")
        decls.append("/-- The dimension indices the networks use. -/\ndef IX : List ℕ := "
                     + _lean_ints(self.ix))
        decls.append(f"/-- Floors of the dimension symbols. -/\ndef FL : List ℕ := "
                     f"{_lean_ints(list(self.fl))}")
        decls.append(f"/-- The atoms' arguments. -/\nnoncomputable def GS {self.binders} : "
                     "List ℝ := "
                     f"{gtext}".replace("GS  :", "GS :"))
        decls.append("/-- Dimension symbol values and atoms. -/\nnoncomputable def ρ "
                     f"{self.binders} : "
                     f"Val := ⟨fun i => ({dims} : List ℕ).getD i 1, fun j => ex ({gsa}.getD j 0)⟩"
                     .replace("ρ  :", "ρ :"))

        def stext(terms: list[STm]) -> str:
            return "[" + ", ".join(f"⟨{self.texts_of[t]}, [], [], []⟩" for t in terms) + "]"

        if not family:
            decls.append(f"/-- The left scalar. -/\ndef S0 : List STm := {stext(self.s0)}")
            decls.append(f"/-- The right scalar. -/\ndef S1 : List STm := {stext(self.s1)}")
        nds = len(self.dnames)
        cases = " | ".join(["_"] * (nds + 1))
        decls.append(f"theorem rhoOK {bh} : {rho}.OK :=\n  ⟨fun i => by rcases i with {cases} <;> "
                     "simp [ρ] <;> first | omega | exact Nat.pos_of_ne_zero (pow_ne_zero _ "
                     "(by omega)), fun j => Complex.exp_ne_zero _⟩")
        decls.append(f"theorem flOK {bh} : FloorOK FL {rho} := fun i => by\n  rcases i with "
                     f"{cases} <;> simp [flv, FL, ρ] <;> omega")
        nix = len(self.ix)
        decls.append(
            f"theorem envD {_join(bh, kb)} : ∀ i ∈ IX, {env}.dimv i = (DT i).den {rho}.dv := by\n"
            "  intro i hi\n  simp only [IX, List.mem_cons, List.not_mem_nil, or_false] at hi\n"
            f"  obtain {' | '.join(['rfl'] * nix)} := hi <;> "
            f"simp [{unfold}, Δ, DT, ρ, DMon.den, npow"
            + "".join(f", hc{j}" for j in range(len(self.constraints))) + "]")
        nr = len(self.pt)
        ph_lines = [f"· {self.rule_pre(p, unfold, rho)}{self.ph_proof(p, row, unfold, rho, gsa)}"
                    for p, row in enumerate(self.pt)]
        ph_lines.append(f"· simp [{unfold}, Φ, PT, phF, ex_zero, rV, rEv, RL]")
        decls.append(
            f"theorem envP {_join(bh, kb)} : ∀ ph j, ex ({env}.phase ph j) = phF {rho} (PT ph) j "
            f"* rV {rho} (RL ph) j "
            f":= by\n  intro ph j\n  rcases ph with {' | '.join(['_'] * (nr + 1))}\n"
            + "\n".join("  " + line for line in ph_lines))
        hr = self.hr
        for side, sm, st in (("0", self.m0, self.s0t), ("1", self.m1, self.s1t)):
            if family:
                break
            decls.append(f"theorem scal{side} {bh} : {st} = exDen {rho} (fun _ => 0) "
                         f"S{side} := by\n  have hρ := {hr}\n  have hg : ∀ j, {rho}.atom j = ex "
                         f"({gsa}.getD j 0) := fun j => rfl\n  {self.scalar_proof(sm, rho)}")
        A, B = self.a.name, self.b.name
        lead = "  equality over symbolic dimensions by the verified normaliser of ArchytasZX.SymNet"
        if not family:
            decls.append("set_option maxHeartbeats 0 in\n"
                         "/-- The kernel's normalisation of the scaled difference. -/\n"
                         f"theorem chk : symCheck FL DT PT RL labs IX {A} {B} S0 S1 = true := by "
                         "decide +kernel")
            _, proof = self.common_wrap(self.sym(env, '', A, B, 'S0', 'S1', 'chk'))
            theorem = ("/-- The diagrams are equal for every value of their dimensions and "
                       f"phases. -/\ntheorem {name} {bh} :\n"
                       f"    Equal {env} {self.stated_s(0)} {A} {self.stated_s(1)} {B} :=\n"
                       f"  {proof}")
            out.decls += [*decls, theorem]
        else:
            if not self.check():
                raise NetUnsupported(self.why)
            keys = list(out.mult_ix)
            names = [n for k, n, bare in zip(keys, out.mult_names, out.mult_bare, strict=True)
                     if bare and k not in self.alias]
            def val(k: str) -> str:
                j = out.mult_ix[self.alias.get(k, k)]
                return _paren(out.mult_names[j])

            vals = [val(k) for k in keys]
            _, proof = self.common_wrap(self.call(self.top, *vals))
            theorem = ("/-- The families are equal for every multiplicity and every value of "
                       f"their dimensions and phases. -/\ntheorem {name} {bh} :\n"
                       f"    ∀ {' '.join(names)} : ℕ, Equal {self.env_at(vals)} {self.stated_s(0)} "
                       f"{A} {self.stated_s(1)} {B} :=\n  fun {' '.join(names)} => {proof}")
            out.decls += [*decls, *self.stab_decls, *self.net_decls, *reversed(self.fam_decls),
                          theorem]
            mm = ", ".join(names)
            lead += (f"\n  by induction on the multiplicit{'ies' if len(names) > 1 else 'y'} {mm}, "
                     "one copy peeled per step")
        floors = [f"{n} ≥ {self.fl[i]}" for i, n in enumerate(self.dnames)
                  if i < len(self.fl) and self.fl[i] > 1]
        if floors:
            lead += "\n  assumes the dimension floors " + ", ".join(floors)
        if self.constraints:
            lead += ("\n  assumes the dimension constraints " + ", ".join(self.constraints)
                     + " (hypotheses of the theorem)")
        return out.source(lead=lead)

    # families

    def var_names(self) -> list[str]:
        """Each multiplicity index's variable in the family lemmas."""
        out = self.out
        keys = list(out.mult_ix)
        return [out.mult_names[j] if out.mult_bare[j] and keys[j] not in self.alias else f"v{j}"
                for j in range(len(out.mult_names))]

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
        return (mx.subs(sym, value), my.subs(sym, value), self.retext(tx, name, text),
                self.retext(ty, name, text))

    def ratio(self, model: Any, j: int) -> Any:
        """The factor ``model`` gains when index ``j`` grows by one."""
        sym = self.mult_sym(j)
        if sym is None or sym not in model.free_symbols:
            return sp.Integer(1)
        if model == 0:
            raise NetUnsupported("a zero scalar depending on a multiplicity")
        r = sp.simplify(sp.powsimp(sp.expand_power_exp(model.subs(sym, sym + 1) / model),
                                   force=True))
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
        body = "[" + ", ".join(f"⟨{self.texts_of[t]}, [], [], []⟩" for t in terms) + "]"
        rho, gsa = self.rho, (f"(GS {self.args})" if self.args else "GS")
        self.stab_decls += [
            f"/-- A scalar's terms. -/\ndef {name} : List STm := {body}",
            (f"theorem {lem} {self.bh} : {_paren(text)} = exDen {rho} (fun _ => 0) {name} := by\n"
             f"  have hρ := {self.hr}\n  have hg : ∀ j, {rho}.atom j = ex ({gsa}.getD j 0) := "
             f"fun j => rfl\n  {self.scalar_proof(model, rho, name)}")]
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
            self.net_decls += [f"{note} {na} : Net := {nx.lean()}",
                               f"{note} {nb} : Net := {ny.lean()}"]
            lj = self.fam(nx, na, ny, nb, sc)
            env = self.env_at(vs)

            def iso(a_text: str, a: Any, b_text: str, b: Any) -> str:
                return out.iso_b(env, a_text, a, b_text, b)

            self.fam_decls[i] = (
                f"theorem P{i} {self.bh} : ∀ {vb} : ℕ, Equal {env} {sxt} {xt} {syt} {yt} :=\n"
                f"  fun {vb} => Equal.congr_s (Equal.mul_trans {iso(xt, x, na, nx)}\n"
                f"    (Equal.mul_trans {self.call(lj, *vs)} {iso(nb, ny, yt, y)})) (by ring) "
                "(by ring)")
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
            self.fam_decls[i] = (
                "set_option maxHeartbeats 0 in\n/-- The kernel's normalisation. -/\n"
                f"theorem C{i} : symCheck FL DT PT RL labs IX {xt} {yt} {sx} {sy} = true := by "
                f"decide +kernel\n\ntheorem P{i} {self.bh} : ∀ {vb} : ℕ, Equal {env} {sxt} {xt} "
                f"{syt} {yt} :=\n  fun {vb} => "
                + self.sym(env, f"(ksOf [{', '.join(vs)}])", xt, yt, sx, sy, f"C{i}", lx, ly))
            if i == 0:
                self.top = f"P{i}"
            return f"P{i}"
        cands = sorted(heads - inner - ported) or sorted(heads - inner)
        if not cands:
            raise NetUnsupported("boxes that do not nest")
        m = cands[0]
        top = max((lb for net in (x, y) for lb in (*net.summed, *net.bnd,
                                                     *(q for n in net.nodes for q in n.labels()))),
                  default=0)
        H = 1 << max(16, top.bit_length())
        for net in (x, y):
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                if out.label_chain(lab)[:1] == (m,):
                    out.peel_reg(lab, H)
        from archytaszx.semantics.lean_net import _split_net, _zero_net

        isb = out.isb(m)
        spc = f"(SpC {m})"
        if m in ported:
            return self.port(i, x, xt, y, yt, sc, m, H, vs)
        al, be = self.ratio(mx, m), self.ratio(my, m)
        zx, zy = _zero_net(x, m, isb), _zero_net(y, m, isb)
        lz = self.fam(zx, f"(Net.zero {spc} {xt})", zy, f"(Net.zero {spc} {yt})",
                      self.scal_at(sc, m, "0", 0))
        sp0, sp1 = _split_net(x, m, isb, H), _split_net(y, m, isb, H)
        t0, t1 = _peeled(sp0, x), _peeled(sp1, y)
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

        def iso(a_text: str, a: Any, b_text: str, b: Any) -> str:
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
        plug0 = (f"(plug {envk} {ta}.nodes {xt}.nodes {yt}.nodes {ta}.int {xt}.int {yt}.int {bl} "
                 f"{xt}.bnd {sk[2]} {sk[3]} (Equal.toLocal _ _ _ ih) (by chk) (by chk) (by chk))")
        plug1 = (f"(plug {envk} {yt}.nodes {ta}.nodes {tb}.nodes {yt}.int {ta}.int {tb}.int {bl} "
                 f"{ta}.bnd {alt} {bet} (Equal.toLocal _ _ _ {self.call(lt, *kvs)}) (by chk) "
                 "(by chk) (by chk))")
        chain = (f"Equal.congr_s (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans "
                 f"(Equal.mul_trans {iso(f'(Net.split {spc} {H} {xt})', sp0, tp0, p0)}\n"
                 f"    {plug0})\n    {iso(tq0, q0, tq0r, q0r)})\n    {plug1})\n"
                 f"    {iso(tq1, q1, f'(Net.split {spc} {H} {yt})', sp1)}) (by ring) (by ring)")
        ok0, ok1 = _split_ok(m, H, xt), _split_ok(m, H, yt)
        args = self.args
        fam = f"(F {args})" if args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(args, hyp0)} _)" if (args or hyp0) else "(Fpos _)"
        zvs = [("0" if j == m else v) for j, v in enumerate(vs)]
        kb = " ".join(f"({v} : ℕ)" for v in kvs)
        self.net_decls += [f"/-- A peeled copy. -/\ndef {ta} : Net := {t0.lean()}",
                           f"/-- A peeled copy. -/\ndef {tb} : Net := {t1.lean()}"]
        self.fam_decls[i] = "\n\n".join([
            (f"theorem Q{i} {self.bh} {kb} (ih : Equal {envk} {sk[2]} {xt} {sk[3]} {yt}) :\n"
             f"    Equal {envk} {sk1[2]} (Net.split {spc} {H} {xt}) {sk1[3]} "
             f"(Net.split {spc} {H} {yt}) :=\n  {chain}"),
            (f"theorem P{i} {self.bh} : ∀ {vb} : ℕ, Equal {self.env_at(vs)} {sxt} {xt} {syt} {yt} "
             f":= by\n  intro {vb}\n  induction {name} with\n"
             f"  | zero => exact Equal.of_zero_at {fam} {m} {H} [{', '.join(zvs)}] (by simp) {xt} "
             f"{yt} {ok0} {ok1} (by decide +kernel) (Equal.congr_s {self.call(lz, *zvs)} "
             "(by ring) (by ring))\n"
             f"  | succ k_ ih => exact Equal.of_split_at {fam} {m} {H} k_ [{', '.join(kvs)}] "
             "(by simp) "
             f"{xt} {yt} {ok0} {ok1} (fun _ _ _ => Fam.dimv_pos _ _ _ {fpos}) "
             f"{self.call(f'Q{i}', *kvs, 'ih')}"),
        ])
        if i == 0:
            self.top = f"P{i}"
        return f"P{i}"

    def port(self, i: int, x: Any, xt: str, y: Any, yt: str, sc: Any, m: int, H: int,
             vs: list[str]) -> str:
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
        lz = self.fam(zx, f"(Net.zero {spc} {xt})", zy, f"(Net.zero {spc} {yt})",
                      self.scal_at(sc, m, "0", 0))
        sx, sy = _split_net(x, m, isb, H), _split_net(y, m, isb, H)
        H2 = 2 * H
        for net in (sx, sy):
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                if out.label_chain(lab)[:1] == (m,):
                    out.peel_reg(lab, H2)
        sxt_, syt_ = f"(Net.split {spc} {H} {xt})", f"(Net.split {spc} {H} {yt})"
        l1 = self.fam(_zero_net(sx, m, isb), f"(Net.zero {spc} {sxt_})", _zero_net(sy, m, isb),
                      f"(Net.zero {spc} {syt_})", self.scal_at(sc, m, "1", 1))
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
            f"(by decide +kernel : Spec.portOK {spc} {xt} {yt} = true)) ih)) (by ring) (by ring)")
        if i == 0:
            self.top = f"P{i}"
        return f"P{i}"

    def rule_pre(self, p: int, unfold: str, rho: str) -> str:
        """The tactic lines splitting phase vector ``p``'s rule off its stored entries."""
        rule = self.rules[p] if p < len(self.rules) else None
        if rule is None:
            return f"rw [show rV {rho} (RL {p}) j = 1 by simp [rV, rEv, RL], mul_one]\n    "
        from archytaszx.algebra.phase import INDEX

        k = sp.Symbol("k_lean_index", integer=True, nonnegative=True)
        text = self.out.sym.render(self.out.phase_specs[p][1].xreplace({INDEX: k}), "ℝ", "ℝ")
        dm, cs = _lean_dm(rule[0]), _lean_ints(list(rule[1]))
        return (f"simp only [{unfold}, Φ]\n    have hr := rule_ex {rho} {dm} {cs} (fun k : ℕ => "
                f"{text}) (fun k => by simp [DMon.den, npow, ρ, rEv]; push_cast; ring) "
                f"(by decide) j\n    try simp only at hr\n    rw [ex_add, hr, show RL {p} = "
                f"({dm}, {cs}) from rfl]\n    congr 1\n    ")

    def ph_proof(self, p: int, row: list[tuple[int, CMon, sp.Expr]], unfold: str, rho: str,
                 gsa: str) -> str:
        if not row:
            return f"simp [{unfold}, Φ, PT, phF, ex_zero]"
        lines = [(f"simp only [{unfold}, Φ, PT, phF, List.getD_cons_succ, List.getD_cons_zero, "
                  "List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]")]
        indent = ""
        for idx, (j, _, v) in enumerate(row):
            q, ks, z = self.atoms.ks(v, True)
            theta = self.out.sym.render(v, "ℝ", "ℝ")
            lines.append(f"{indent}by_cases h{idx} : j = {j}")
            others = "".join(f"show ((({j} : ℕ) = {k}) ↔ False) from by decide, "
                             for k, _, _ in row if k != j)
            lines.append(f"{indent}· subst h{idx}; simp only [{others}eq_self_iff_true, if_true, "
                         "if_false, add_zero]")
            if z == 0:
                lines.append(f"{indent}  rw [phC_den {rho} {gsa} (fun _ => rfl) {_lean_q(q)} "
                             f"{_lean_ints(ks)} {_paren(theta)} (by first | (simp [lin, GS]; done) "
                             "| (simp [lin, GS]; push_cast; ring))]; ring")
            else:
                zt = self.out.sym.render(z, "ℤ", "ℤ")
                lines.append(f"{indent}  rw [phC_den_int {rho} {gsa} (fun _ => rfl) {_lean_q(q)} "
                             f"{_lean_ints(ks)} {_paren(theta)} ⟨{zt}, by first | "
                             "(simp [lin, GS]; done) | (simp [lin, GS]; push_cast; ring)⟩]; ring")
            lines.append(f"{indent}· ")
            indent += "  "
        hs = ", ".join(f"h{i}" for i in range(len(row)))
        lines[-1] += f"simp [{hs}, ex_zero]"
        return "\n    ".join(lines)

    def scalar_proof(self, model: sp.Expr, rho: str, tab: str = "S0, S1") -> str:
        return (f"simp (disch := decide) only [{tab}, exDen, List.map_cons, List.map_nil, "
                f"List.sum_cons, List.sum_nil, add_zero, den_zero, bodyOf, List.prod_nil, "
                f"mul_one, CMon.den_mul {rho} hρ, ofQ_den, exQ_den, rtD_den, dP_den, "
                f"phC_den {rho} _ hg] <;> "
                f"simp [DMon.den, npow, ρ, zpow_neg, inv_pow] <;> push_cast <;> ring_nf")


def export_symbolic(left: Diagram, right: Diagram, name: str = "main",
                    describe: Callable[[Diagram], str] | None = None,
                    floors: Mapping[str, int] | None = None) -> str:
    """Lean 4 source proving ``left = right`` for every value of the dimensions and phases by
    the verified normaliser; :class:`NetUnsupported` when the normaliser does not reach it."""
    from archytaszx.semantics.decide import dimension_floors

    ex = _SymExport(left, right, describe, floors if floors is not None
                    else dimension_floors(left, right))
    if not ex.check():
        raise NetUnsupported(getattr(ex, "why", "the normal forms of the two networks differ"))
    return ex.source(name)

