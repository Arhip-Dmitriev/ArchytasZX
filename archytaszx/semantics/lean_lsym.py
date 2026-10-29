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

"""The mirror of ``ArchytasZX.LSym``, ``LDigit`` and ``LCanon``: symbolic sums with integer
atoms and their normaliser, on tuples.

An atom is ``(id, f, ks, dp, args)``, each argument ``(IPoly, DMon)``; a term is
``(CMon, sums, atoms, dels, phs)``. Every function follows its Lean namesake, so that
:func:`nf_a` predicts ``nfA``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any

from archytaszx.semantics import lean_sym as Y

Atom = tuple[int, int, tuple[int, ...], tuple[Y.DMon, ...], tuple[tuple[Y.IPoly, Y.DMon], ...]]
ATm = tuple[
    Y.CMon, tuple[tuple[int, Y.DMon], ...], tuple[Atom, ...], tuple[Y.Fac, ...], tuple[Y.Fac, ...]
]

ABASE = 3000000
RBASE = 2000000
AROUNDS = 8


# variables


def ip_vs(p: Y.IPoly) -> list[int]:
    return [x for m, _ in p for x in m]


def at_vs(a: Atom) -> list[int]:
    return [x for p, _ in a[4] for x in ip_vs(p)]


def arg_vs(l: Sequence[Atom]) -> list[int]:
    return [x for a in l for x in at_vs(a)]


def ids(l: Sequence[Atom]) -> list[int]:
    return [a[0] for a in l]


def fvs(t: ATm) -> list[int]:
    return [x for f in (*t[3], *t[4]) for x in ip_vs(f[1])]


def rvs(t: ATm) -> list[int]:
    return fvs(t) + arg_vs(t[2]) + ids(t[2])


def abvs(t: ATm) -> list[int]:
    return [x for x, _ in t[1]]


def all_v(t: ATm) -> list[int]:
    return rvs(t) + abvs(t)


def ax_vs(e: Sequence[ATm]) -> list[int]:
    return [x for t in e for x in rvs(t)]


# renaming


def im_ren(m: Callable[[int], int], mon: Sequence[int]) -> tuple[int, ...]:
    acc: tuple[int, ...] = ()
    for x in reversed(mon):
        acc = Y.im_ins(m(x), acc)
    return acc


def ip_ren(m: Callable[[int], int], p: Y.IPoly) -> Y.IPoly:
    acc: Y.IPoly = ()
    for mon, c in reversed(p):
        acc = Y.ip_ins(im_ren(m, mon), c, acc)
    return acc


def fac_ren(m: Callable[[int], int], f: Y.Fac) -> Y.Fac:
    return (f[0], ip_ren(m, f[1]))


def at_ren(m: Callable[[int], int], a: Atom) -> Atom:
    return (m(a[0]), a[1], a[2], a[3], tuple((ip_ren(m, p), d) for p, d in a[4]))


def t_ren(m: Callable[[int], int], t: ATm) -> ATm:
    return (
        t[0],
        tuple((m(x), d) for x, d in t[1]),
        tuple(at_ren(m, a) for a in t[2]),
        tuple(fac_ren(m, f) for f in t[3]),
        tuple(fac_ren(m, f) for f in t[4]),
    )


def inj_on(m: Callable[[int], int], s: Sequence[int]) -> bool:
    return all(m(a) != m(b) or a == b for a in s for b in s)


# products


def t_mul(s: ATm, t: ATm) -> ATm:
    return (Y.cm_mul(s[0], t[0]), s[1] + t[1], s[2] + t[2], s[3] + t[3], s[4] + t[4])


def t_mul_ok(s: ATm, t: ATm) -> bool:
    rt_, rs = set(rvs(t)), set(rvs(s))
    return all(x not in rt_ for x in abvs(s) + ids(s[2])) and all(
        x not in rs for x in abvs(t) + ids(t[2])
    )


def ax_mul(a: Sequence[ATm], b: Sequence[ATm]) -> list[ATm]:
    return [t_mul(s, t) for s in a for t in b]


def ax_mul_ok(a: Sequence[ATm], b: Sequence[ATm]) -> bool:
    return all(t_mul_ok(s, t) for s in a for t in b)


# eliminations


def at_sub(x: int, v: Y.IPoly, a: Atom) -> Atom:
    return (a[0], a[1], a[2], a[3], tuple((Y.ip_subst(x, v, p), d) for p, d in a[4]))


def args_per(x: int, m: Y.DMon, l: Sequence[Atom]) -> bool:
    return all(not Y.ip_has(x, p) or Y.dm_dvd(d, m) for a in l for p, d in a[4])


def plain_p(l: Sequence[Atom], p: Y.IPoly) -> bool:
    i = set(ids(l))
    return all(y not in i for y in ip_vs(p))


AElim = tuple[Y.CMon, tuple[Atom, ...], tuple[Y.Fac, ...], tuple[Y.Fac, ...]]


def elim_plain(
    x: int, d: Y.DMon, ats: tuple[Atom, ...], ds: tuple[Y.Fac, ...], ps: tuple[Y.Fac, ...]
) -> AElim | None:
    if x in arg_vs(ats) + ids(ats):
        return None
    e = Y.elim_var(x, d, ds, ps)
    if e is None:
        return None
    return e[0], ats, e[1], e[2]


def elim_pin1(
    x: int,
    d: Y.DMon,
    ats: tuple[Atom, ...],
    pre: tuple[Y.Fac, ...],
    f: Y.Fac,
    post: tuple[Y.Fac, ...],
    ps: tuple[Y.Fac, ...],
) -> AElim | None:
    lin = Y.ip_lin(x, f[1])
    if lin is None:
        return None
    a, b = lin
    s = Y.ip_const(a)
    if s is None:
        return None
    if (
        s in (1, -1)
        and d[0] != 0
        and Y.dm_dvd(f[0], d)
        and x not in ids(ats)
        and plain_p(ats, b)
        and args_per(x, f[0], ats)
        and all(not Y.fac_has(x, g) or Y.dm_dvd(g[0], f[0]) for g in (*pre, *post, *ps))
    ):
        v = Y.pin_val(s, b)
        return (
            Y.rt_d(Y.dm_div(d, f[0]), 2),
            tuple(at_sub(x, v, t) for t in ats),
            tuple(Y.fac_sub(x, v, g) for g in (*pre, *post)),
            tuple(Y.fac_sub(x, v, g) for g in ps),
        )
    return None


def elim_pin(
    x: int, d: Y.DMon, ats: tuple[Atom, ...], ds: tuple[Y.Fac, ...], ps: tuple[Y.Fac, ...]
) -> AElim | None:
    for pre, f, post in Y._splits(ds):
        e = elim_pin1(x, d, ats, pre, f, post, ps)
        if e is not None:
            return e
    return None


def dig_at(x: int, d: Y.DMon, n: Y.DMon, a: Atom) -> Y.IPoly | None:
    if not (a[1] == 0 and a[2] == () and a[3] == (d, n)):
        return None
    if len(a[4]) != 2:
        return None
    (p, m1), (q, m2) = a[4]
    if m1 == n and m2 == n and p == Y.ip_v(x) and not Y.ip_has(x, q):
        return q
    if m1 == n and m2 == n and q == Y.ip_v(x) and not Y.ip_has(x, p):
        return p
    return None


def coefs(zs: Sequence[int], r: Y.IPoly) -> tuple[list[int], Y.IPoly] | None:
    if not zs:
        return [], r
    lin = Y.ip_lin(zs[0], r)
    if lin is None:
        return None
    k = Y.ip_const(lin[0])
    rest = coefs(zs[1:], lin[1])
    if k is None or rest is None:
        return None
    return [k, *rest[0]], rest[1]


def fac_any(zs: Sequence[int], f: Y.Fac) -> bool:
    return any(Y.fac_has(z, f) for z in zs)


def psum(ps: Sequence[Y.Fac]) -> Y.IPoly:
    acc: Y.IPoly = ()
    for f in reversed(ps):
        acc = Y.ip_add(f[1], acc)
    return acc


def elim_digit(
    pw: Sequence[tuple[Y.DMon, Y.DMon]],
    x: int,
    dd_: Y.DMon,
    ats: tuple[Atom, ...],
    ds: tuple[Y.Fac, ...],
    ps: tuple[Y.Fac, ...],
) -> AElim | None:
    others = tuple(a for a in ats if x not in at_vs(a))
    dd = tuple(a for a in ats if x in at_vs(a))
    if not dd or len(dd[0][3]) != 2:
        return None
    a0 = dd[0]
    d, n = a0[3]
    As = [dig_at(x, d, n, a) for a in dd]
    if any(a is None for a in As):
        return None
    zs = ids(dd)

    def sel(f: Y.Fac) -> bool:
        return f[0] == d and (Y.fac_has(x, f) or fac_any(zs, f))

    lin = Y.ip_lin(x, psum([f for f in ps if sel(f)]))
    if lin is None:
        return None
    b, r0 = lin
    co = coefs(zs, r0)
    if co is None:
        return None
    cs, r = co
    rest = tuple(f for f in ps if not sel(f))
    zset = set(zs)
    ok = (
        ats == others + dd
        and dd_ == n
        and (d, n) in pw
        and dd_[0] != 0
        and len(zs) == len(zset)
        and x not in ids(ats)
        and x not in arg_vs(others)
        and all(z not in arg_vs(ats) and z not in ip_vs(b) and z not in ip_vs(r) for z in zs)
        and all(plain_p(ats, a) for a in As)  # type: ignore[arg-type]
        and all(not Y.fac_has(x, f) and not fac_any(zs, f) for f in ds)
        and all(not Y.fac_has(x, f) and not fac_any(zs, f) for f in rest)
        and not Y.ip_has(x, r)
        and not Y.ip_has(x, b)
    )
    if not ok:
        return None
    dz: Atom = (a0[0], 3, tuple(cs), (d, n), tuple((a, n) for a in As) + ((b, d),))  # type: ignore[assignment]
    return Y.rt_d(n, 2), others + (dz,), ds + ((n, Y.ip_v(a0[0])),), rest + ((d, r),)


def a_elim_var(
    pw: Sequence[tuple[Y.DMon, Y.DMon]],
    x: int,
    d: Y.DMon,
    ats: tuple[Atom, ...],
    ds: tuple[Y.Fac, ...],
    ps: tuple[Y.Fac, ...],
) -> AElim | None:
    for fn in (elim_plain, elim_pin):
        e = fn(x, d, ats, ds, ps)
        if e is not None:
            return e
    return elim_digit(pw, x, d, ats, ds, ps)


def a_elim_step(pw: Sequence[tuple[Y.DMon, Y.DMon]], t: ATm) -> ATm | None:
    for pre, (x, d), post in Y._splits(t[1]):
        if x in [y for y, _ in (*pre, *post)]:
            continue
        e = a_elim_var(pw, x, d, t[2], t[3], t[4])
        if e is not None:
            return (Y.cm_mul(t[0], e[0]), pre + post, e[1], e[2], e[3])
    return None


def a_elim_all(pw: Sequence[tuple[Y.DMon, Y.DMon]], n: int, t: ATm) -> ATm:
    for _ in range(n):
        t2 = a_elim_step(pw, a_rules(len(t[2]) + 1, t))
        if t2 is None:
            return t
        t = t2
    return t


# canonicalisation


def canon_f(fl: Sequence[int], ds: tuple[Y.Fac, ...], ps: tuple[Y.Fac, ...]) -> Any:
    ds0, ps0 = Y.piv_pass(len(ds), (), ds, ps)
    dn = Y.norm_dels(fl, ds0)
    if dn is None:
        return None
    k, pn = Y.norm_phs(Y.ph_merge(ps0))
    dsort: tuple[Y.Fac, ...] = ()
    for f in reversed(dn):
        dsort = Y.del_ins(f, dsort)
    psort: tuple[Y.Fac, ...] = ()
    for f in reversed(pn):
        psort = Y.fac_ins(f, psort)
    return k, dsort, psort


def dz_poly(pw: Sequence[tuple[Y.DMon, Y.DMon]], ats: Sequence[Atom], a: Atom) -> Any:
    if a[1] != 3 or len(a[3]) != 2:
        return None
    d, n = a[3]
    ks, args = a[2], a[4]
    if len(ks) == 1 and len(args) == 2:
        (aa, m1), (bp, m2) = args
        b = Y.ip_const(bp)
        if b is None:
            return None
        c = ks[0]
        const = Y.dm_const(d)
        r = (-c * b) % d[0] if const and d[0] != 0 else -c * b
        if (
            (d, n) in pw
            and m1 == n
            and m2 == d
            and c in (1, -1)
            and 0 <= r
            and (r <= 1 or const)
            and plain_p(ats, aa)
        ):
            return n, Y.ip_sub(aa, Y.ip_c(r))
        if (d, n) in pw and m1 == n and m2 == d and aa == () and b in (1, -1):
            return n, Y.ip_c(1)
        return None
    if len(ks) == 2 and len(args) == 3:
        (a1, m1), (a2, m2), (bp, m3) = args
        c, c2 = ks
        if (
            (d, n) in pw
            and m1 == n
            and m2 == n
            and m3 == d
            and c in (1, -1)
            and c2 == -c
            and Y.ip_const(bp) == 0
            and plain_p(ats, a1)
            and plain_p(ats, a2)
        ):
            return n, Y.ip_sub(a1, a2)
    return None


def dz_find(pw: Sequence[tuple[Y.DMon, Y.DMon]], ats: Sequence[Atom], f: Y.Fac) -> Y.Fac | None:
    p = f[1]
    if not (len(p) == 1 and len(p[0][0]) == 1 and p[0][1] == 1):
        return None
    z = p[0][0][0]
    a = next((b for b in ats if b[0] == z), None)
    if a is None:
        return None
    r = dz_poly(pw, ats, a)
    if r is None:
        return None
    n, pp = r
    if f[0] == n and plain_p(ats, pp):
        return n, pp
    return None


def dz_step(pw: Sequence[tuple[Y.DMon, Y.DMon]], t: ATm) -> ATm:
    i = ids(t[2])
    if len(i) != len(set(i)):
        return t
    ds = tuple(dz_find(pw, t[2], f) or f for f in t[3])
    return (t[0], t[1], t[2], ds, t[4])


def is_zero_dig(a: Atom) -> bool:
    return a[1] == 0 and len(a[4]) == 2 and (a[4][0][0] == () or a[4][1][0] == ())


def zero_at(t: ATm, pre: tuple[Atom, ...], a: Atom, post: tuple[Atom, ...]) -> ATm | None:
    if is_zero_dig(a) and a[0] not in ids(post):
        z = a[0]
        return (
            t[0],
            t[1],
            pre + tuple(at_sub(z, (), b) for b in post),
            tuple(Y.fac_sub(z, (), f) for f in t[3]),
            tuple(Y.fac_sub(z, (), f) for f in t[4]),
        )
    return None


def drop_at(t: ATm, pre: tuple[Atom, ...], a: Atom, post: tuple[Atom, ...]) -> ATm | None:
    z = a[0]
    if z not in fvs(t) and z not in arg_vs(post) and z not in ids(post):
        return (t[0], t[1], pre + post, t[3], t[4])
    return None


def dedup_at(t: ATm, pre: tuple[Atom, ...], b: Atom, post: tuple[Atom, ...]) -> ATm | None:
    a = next((x for x in pre if x[0] != b[0] and x[1:] == b[1:]), None)
    if a is None:
        return None
    i = ids(t[2])
    if (
        len(i) == len(set(i))
        and all(plain_p(t[2], p) for p, _ in b[4])
        and b[0] not in arg_vs(t[2])
    ):
        v = Y.ip_v(a[0])
        return (
            t[0],
            t[1],
            t[2],
            tuple(Y.fac_sub(b[0], v, f) for f in t[3]),
            tuple(Y.fac_sub(b[0], v, f) for f in t[4]),
        )
    return None


def dz_drop_at(t: ATm, pre: tuple[Atom, ...], a: Atom, post: tuple[Atom, ...]) -> ATm | None:
    ks, args = a[2], a[4]
    if a[1] != 3 or len(args) != len(ks) + 1:
        return None
    pairs = list(zip(ks, args))
    if not any(p[1][0] == () for p in pairs):
        return None
    kept = [p for p in pairs if p[1][0] != ()]
    if not kept:
        return None
    a2: Atom = (
        a[0],
        3,
        tuple(k for k, _ in kept),
        a[3],
        tuple(x for _, x in kept) + tuple(args[len(ks) :]),
    )
    return (t[0], t[1], pre + (a2,) + post, t[3], t[4])


def a_rule1(t: ATm) -> ATm | None:
    for fn in (zero_at, drop_at, dedup_at, dz_drop_at):
        for pre, a, post in Y._splits(t[2]):
            r = fn(t, pre, a, post)
            if r is not None:
                return r
    return None


def a_rules(n: int, t: ATm) -> ATm:
    for _ in range(n):
        t2 = a_rule1(t)
        if t2 is None:
            return t
        t = t2
    return t


def wf_a(l: Sequence[Atom]) -> bool:
    for k, a in enumerate(l):
        later = set(ids(l[k:]))
        if any(v in later for v in at_vs(a)):
            return False
    return True


def uniq(l: Sequence[int]) -> list[int]:
    out: list[int] = []
    for x in l:
        if x not in out:
            out.append(x)
    return out


def canon_ren(t: ATm) -> ATm:
    occ = uniq(fvs(t) + arg_vs(t[2]) + ids(t[2]) + abvs(t))
    b = set(abvs(t))
    i = set(ids(t[2]))
    bl = [v for v in occ if v in b]
    al = [v for v in occ if v in i and v not in b]

    def m(v: int) -> int:
        if v in bl:
            return RBASE + bl.index(v)
        if v in al:
            return ABASE + al.index(v)
        return v

    if inj_on(m, all_v(t)) and wf_a(t[2]):
        return t_ren(m, t)
    return t


def sort_sums(t: ATm) -> ATm:
    b = abvs(t)
    if len(b) != len(set(b)):
        return t
    out: list[tuple[int, Y.DMon]] = []
    for p in reversed(t[1]):
        k = 0
        while k < len(out) and not p[0] <= out[k][0]:
            k += 1
        out.insert(k, p)
    return (t[0], tuple(out), t[2], t[3], t[4])


def dvd_p(fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], k: Y.DMon, m: Y.DMon) -> bool:
    return Y.dm_dvd(k, m) or ((k, m) in pw and Y.dm_den(m, lambda i: Y.flv(fl, i)) >= 2)


def mod_drop_at(
    fl: Sequence[int],
    pw: Sequence[tuple[Y.DMon, Y.DMon]],
    t: ATm,
    pre: tuple[Atom, ...],
    a: Atom,
    post: tuple[Atom, ...],
) -> ATm | None:
    if a[1] != 2 or len(a[4]) != 1:
        return None
    q, m = a[4][0]
    z = a[0]
    if (
        all(y != z and y not in ids(post) for y in ip_vs(q))
        and z not in arg_vs(t[2])
        and z not in ids(post)
        and all(
            not Y.fac_has(z, g) or (g[0][0] != 0 and dvd_p(fl, pw, g[0], m)) for g in (*t[3], *t[4])
        )
    ):
        return (
            t[0],
            t[1],
            t[2],
            tuple(Y.fac_sub(z, q, g) for g in t[3]),
            tuple(Y.fac_sub(z, q, g) for g in t[4]),
        )
    return None


def mod_rules(fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], n: int, t: ATm) -> ATm:
    for _ in range(n):
        t2 = None
        for pre, a, post in Y._splits(t[2]):
            t2 = mod_drop_at(fl, pw, t, pre, a, post)
            if t2 is not None:
                break
        if t2 is None:
            return t
        t = t2
    return t


def pre_a(fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], t: ATm) -> ATm:
    t2 = dz_step(pw, mod_rules(fl, pw, len(t[2]) + 1, a_rules(len(t[2]) + 1, t)))
    return a_rules(len(t2[2]) + 1, t2)


def canon_a(fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], t: ATm) -> ATm | None:
    t3 = pre_a(fl, pw, t)
    r = canon_f(fl, t3[3], t3[4])
    if r is None:
        return None
    k, ds, ps = r
    return sort_sums(canon_ren((Y.cm_mul(t3[0], k), t3[1], t3[2], ds, ps)))


def norm_a(fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], r: int, t: ATm) -> ATm | None:
    for _ in range(r):
        t2 = canon_a(fl, pw, a_elim_all(pw, len(t[1]) + 1, t))
        if t2 is None:
            return None
        t = t2
    return t


def a_shape(t: ATm) -> ATm:
    c = t[0]
    return ((Fraction(0), *c[1:]), *t[1:])


def a_with_q(t: ATm, q: Fraction) -> ATm:
    c = t[0]
    return ((q, *c[1:]), *t[1:])


def ta_add(t: ATm, lst: tuple[ATm, ...]) -> tuple[ATm, ...]:
    out = list(lst)
    st = a_shape(t)
    for i, u in enumerate(out):
        if a_shape(u) == st:
            q = u[0][0] + t[0][0]
            if q == 0:
                return tuple(out[:i] + out[i + 1 :])
            out[i] = a_with_q(u, q)
            return tuple(out)
    return tuple(out + ([t] if t[0][0] != 0 else []))


def nf_a(
    fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], e: Sequence[ATm]
) -> tuple[ATm, ...]:
    acc: tuple[ATm, ...] = ()
    for t in e:
        n = norm_a(fl, pw, AROUNDS, t)
        if n is not None:
            acc = ta_add(n, acc)
    return acc


def ax_neg(e: Sequence[ATm]) -> list[ATm]:
    return [a_with_q(t, -t[0][0]) for t in e]
