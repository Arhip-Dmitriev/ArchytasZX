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

"""Lean 4 certificates for results of the .azx language, by translation validation over the
verified normaliser of symbolic sums with integer atoms (``ArchytasZX.LCanon``).

The left side of a certificate is the program's structure (``ArchytasZX.Lang.Prog``), built
from the syntax trees the interpreter keeps: every gate instance, literal state and identity
is a leaf expression; applications, tensor products, scalar products, sums, inner products,
conjugates and loops are ``Prog`` nodes whose meaning ``Prog.sem`` is a sum or product over
ℂ. The right side is the printed value, as an expression. ``certify_lit`` proves the two equal
when the kernel normalises their difference to nothing (``decide +kernel``), for every value
of the dimension symbols, parameters, functions and output registers.

Variables: output register ``i`` is ``i``, input register ``i`` of a gate is ``IN + i``,
parameter ``k`` is ``PAR + k``, summation indices start at ``BV`` and atoms at ``AID``.

The functions under "the mirror" follow their Lean namesakes in ``ArchytasZX.Lang`` on the
tuples of :mod:`archytaszx.semantics.lean_lsym`, so that :func:`cert_check` predicts
``certCheck``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.semantics import lean_lsym as Z
from archytaszx.semantics import lean_sym as Y
from archytaszx.semantics.lean_lsym import ATm, Atom
from archytaszx.semantics.lean_net import NetUnsupported, _lake, lean_project
from archytaszx.semantics.pathsum import State

IN = 1000
PAR = 2000
BV = 10000


class CertifyError(Exception):
    """A result that cannot be certified; the message is the reason."""


# the mirror


def ax_ren(m: Callable[[int], int], e: Sequence[ATm]) -> list[ATm]:
    return [Z.t_ren(m, t) for t in e]


def ren_oka(m: Callable[[int], int], e: Sequence[ATm]) -> bool:
    return all(Z.inj_on(m, Z.all_v(t)) for t in e)


def ax_sums(s: Sequence[tuple[int, Y.DMon]], e: Sequence[ATm]) -> list[ATm]:
    return [(t[0], tuple(s) + t[1], t[2], t[3], t[4]) for t in e]


def max_var(e: Sequence[ATm]) -> int:
    return max([0, *(x for t in e for x in Z.all_v(t))])


def fresh(a: Sequence[ATm], b: Sequence[ATm]) -> int:
    return max(3 * IN, max(max_var(a), max_var(b)) + 1)


def t_apart(base: int, t: ATm) -> ATm:
    bl = Z.abvs(t)
    al = [v for v in Z.ids(t[2]) if v not in bl]

    def m(v: int) -> int:
        if v in bl:
            return base + bl.index(v)
        if v in al:
            return base + len(bl) + al.index(v)
        return v

    if Z.inj_on(m, Z.all_v(t)) and Z.wf_a(t[2]):
        return Z.t_ren(m, t)
    return t


def apart(base: int, e: Sequence[ATm]) -> list[ATm]:
    return [t_apart(base, t) for t in e]


def in_map(t0: int, k: int) -> Callable[[int], int]:
    return lambda v: t0 + (v - IN) if IN <= v < IN + k else v


def out_map(t0: int, k: int) -> Callable[[int], int]:
    return lambda v: t0 + v if v < k else v


def shift_n(k: int) -> Callable[[int], int]:
    return lambda v: v + k if v < 2 * IN else v


def to_in(k: int) -> Callable[[int], int]:
    return lambda v: IN + v if v < k else v


def avoids(t0: int, k: int, e: Sequence[ATm]) -> bool:
    return all(not (t0 <= v < t0 + k) for v in Z.ax_vs(e))


def blk(t0: int, levels: Sequence[Y.DMon]) -> list[tuple[int, Y.DMon]]:
    return [(t0 + i, d) for i, d in enumerate(levels)]


def app_sym(levels: Sequence[Y.DMon], a: Sequence[ATm], b: Sequence[ATm]) -> list[ATm] | None:
    t0, k = fresh(a, b), len(levels)
    a2, b2 = ax_ren(in_map(t0, k), a), ax_ren(out_map(t0, k), b)
    if (
        k < IN
        and 3 * IN <= t0
        and avoids(t0, k, a)
        and avoids(t0, k, b)
        and ren_oka(in_map(t0, k), a)
        and ren_oka(out_map(t0, k), b)
        and Z.ax_mul_ok(a2, b2)
    ):
        return ax_sums(blk(t0, levels), Z.ax_mul(a2, b2))
    return None


def ten_sym(k: int, a: Sequence[ATm], b: Sequence[ATm]) -> list[ATm] | None:
    b2 = ax_ren(shift_n(k), b)
    if ren_oka(shift_n(k), b) and Z.ax_mul_ok(a, b2):
        return Z.ax_mul(a, b2)
    return None


def c_conj(c: Y.CMon) -> Y.CMon:
    q, r = Y.rnorm(c[0], -c[1])
    return (q, r, c[2], c[3], c[4], c[5])


def t_conj(t: ATm) -> ATm:
    return (c_conj(t[0]), t[1], t[2], t[3], tuple((f[0], Y.ip_smul(-1, f[1])) for f in t[4]))


def inn_sym(levels: Sequence[Y.DMon], a: Sequence[ATm], b: Sequence[ATm]) -> list[ATm] | None:
    k = len(levels)
    ca = [t_conj(t) for t in a]
    if all(not t[0][4] for t in a) and ren_oka(to_in(k), ca) and avoids(IN, k, a):
        return app_sym(levels, ax_ren(to_in(k), ca), b)
    return None


def t_sub(x: int, v: Y.IPoly, t: ATm) -> ATm:
    return (
        t[0],
        t[1],
        tuple(Z.at_sub(x, v, a) for a in t[2]),
        tuple(Y.fac_sub(x, v, f) for f in t[3]),
        tuple(Y.fac_sub(x, v, f) for f in t[4]),
    )


def sub_ok(x: int, v: Y.IPoly, t: ATm) -> bool:
    b = Z.abvs(t)
    return (
        x not in b
        and x not in Z.ids(t[2])
        and Z.plain_p(t[2], v)
        and all(y not in b for y in Z.ip_vs(v))
    )


def loop_chk(
    fl: Sequence[int],
    pw: Sequence[tuple[Y.DMon, Y.DMon]],
    levels: Sequence[Y.DMon],
    j: int,
    lo: int,
    s: Sequence[ATm],
    i0: Sequence[ATm],
    inv: Sequence[ATm],
) -> bool:
    v0 = Y.ip_c(lo)
    v1 = Y.ip_add(Y.ip_v(j), Y.ip_c(1))
    if not (
        2 * IN <= j and all(sub_ok(j, v0, t) for t in inv) and all(sub_ok(j, v1, t) for t in inv)
    ):
        return False
    if Z.nf_a(fl, pw, list(i0) + Z.ax_neg([t_sub(j, v0, t) for t in inv])):
        return False
    e1 = app_sym(levels, apart(fresh(s, inv), s), inv)
    return e1 is not None and not Z.nf_a(fl, pw, e1 + Z.ax_neg([t_sub(j, v1, t) for t in inv]))


@dataclass(frozen=True, eq=False)
class Prog:
    """``Prog``: ``kind`` is leaf, app, ten, mul, add or inn; ``name`` is the Lean definition a
    named node is written as."""

    kind: str
    args: tuple[Any, ...]
    name: str | None = None


def counts(p: Prog) -> list[Y.IPoly]:
    """``Prog.counts``."""
    if p.kind == "leaf":
        return []
    if p.kind == "loop":
        return [p.args[3], *counts(p.args[4]), *counts(p.args[5])]
    return [c for q in p.args if isinstance(q, Prog) for c in counts(q)]


def prog_sym(
    p: Prog, fl: Sequence[int] = (), pw: Sequence[tuple[Y.DMon, Y.DMon]] = ()
) -> list[ATm] | None:
    """``Prog.sym``."""
    if p.kind == "leaf":
        return list(p.args[0])
    if p.kind == "loop":
        levels, j, lo, cnt, step, init, inv = p.args
        x0, y0 = prog_sym(step, fl, pw), prog_sym(init, fl, pw)
        if x0 is None or y0 is None:
            return None
        v = Y.ip_add(Y.ip_c(lo), cnt)
        if (
            not counts(step)
            and not counts(init)
            and len(levels) < IN
            and all(2 * IN <= y for y in Z.ip_vs(cnt))
            and all(sub_ok(j, v, t) for t in inv)
            and loop_chk(fl, pw, levels, j, lo, x0, y0, inv)
        ):
            return [t_sub(j, v, t) for t in inv]
        return None
    if p.kind == "cnj":
        x0 = prog_sym(p.args[0], fl, pw)
        if x0 is None or any(t[0][4] for t in x0):
            return None
        return [t_conj(t) for t in x0]
    xs = [prog_sym(q, fl, pw) for q in p.args if isinstance(q, Prog)]
    if any(x is None for x in xs):
        return None
    x, y = xs[0], xs[1]
    assert x is not None and y is not None
    if p.kind == "app":
        return app_sym(p.args[0], apart(fresh(x, y), x), y)
    if p.kind == "ten":
        return ten_sym(p.args[0], x, apart(fresh(x, y), y))
    if p.kind == "mul":
        y2 = apart(fresh(x, y), y)
        return Z.ax_mul(x, y2) if Z.ax_mul_ok(x, y2) else None
    if p.kind == "inn":
        return inn_sym(p.args[0], apart(fresh(x, y), x), y)
    return x + y


def cert_check(
    fl: Sequence[int], pw: Sequence[tuple[Y.DMon, Y.DMon]], p: Prog, r: Sequence[ATm]
) -> bool:
    """``certCheck``."""
    e = prog_sym(p, fl, pw)
    return e is not None and not Z.nf_a(fl, pw, e + Z.ax_neg(r))


# Lean text


def lean_q(q: Fraction) -> str:
    if q.denominator == 1:
        return f"({q.numerator} : ℚ)"
    return f"(({q.numerator} : ℚ) / {q.denominator})"


def lean_ints(xs: Sequence[int]) -> str:
    return "[" + ", ".join(f"({x} : ℤ)" if x < 0 else str(x) for x in xs) + "]"


def lean_nats(xs: Sequence[int]) -> str:
    return "[" + ", ".join(map(str, xs)) + "]"


@dataclass
class Text:
    """Lean text of expressions, dimension monomials by their names in ``dms``."""

    dms: dict[Y.DMon, str] = field(default_factory=dict)

    def dm(self, m: Y.DMon) -> str:
        return self.dms.get(m, f"⟨{m[0]}, {lean_nats(m[1])}⟩")

    def cm(self, c: Y.CMon) -> str:
        if c[1:] == (Fraction(0), (), (), (), ()):
            return f"CMon.ofQ {lean_q(c[0])}"
        gc = "[" + ", ".join(f"({a}, {self.dm(m)})" for a, m in c[5]) + "]"
        return (
            f"⟨{lean_q(c[0])}, {lean_q(c[1])}, {lean_ints(c[2])}, {lean_nats(c[3])}, "
            f"{lean_ints(c[4])}, {gc}⟩"
        )

    @staticmethod
    def ip(p: Y.IPoly) -> str:
        return "[" + ", ".join(f"({lean_nats(m)}, {c})" for m, c in p) + "]"

    def fac(self, f: Y.Fac) -> str:
        return f"({self.dm(f[0])}, {self.ip(f[1])})"

    def atom(self, a: Atom) -> str:
        dp = "[" + ", ".join(self.dm(m) for m in a[3]) + "]"
        args = "[" + ", ".join(f"({self.ip(p)}, {self.dm(m)})" for p, m in a[4]) + "]"
        return f"⟨{a[0]}, {a[1]}, {lean_ints(a[2])}, {dp}, {args}⟩"

    def tm(self, t: ATm) -> str:
        sums = "[" + ", ".join(f"({x}, {self.dm(m)})" for x, m in t[1]) + "]"
        ats = "[" + ", ".join(self.atom(a) for a in t[2]) + "]"
        dels = "[" + ", ".join(self.fac(f) for f in t[3]) + "]"
        phs = "[" + ", ".join(self.fac(f) for f in t[4]) + "]"
        return f"⟨{self.cm(t[0])}, {sums}, {ats}, {dels}, {phs}⟩"

    def ex(self, e: Sequence[ATm], indent: str = "  ") -> str:
        if not e:
            return "[]"
        return "[" + f",\n{indent} ".join(self.tm(t) for t in e) + "]"

    def prog(self, p: Prog, top: bool = True) -> str:
        if p.name is not None and not top:
            return p.name
        if p.kind == "leaf":
            text = f".leaf {self.ex(p.args[0], '    ')}"
        elif p.kind in ("app", "inn"):
            levels = "[" + ", ".join(self.dm(m) for m in p.args[0]) + "]"
            text = f".{p.kind} {levels} {self.prog(p.args[1], False)} {self.prog(p.args[2], False)}"
        elif p.kind == "cnj":
            text = f".cnj {self.prog(p.args[0], False)}"
        elif p.kind == "loop":
            levels, j, lo, cnt, step, init, inv = p.args
            ls = "[" + ", ".join(self.dm(m) for m in levels) + "]"
            text = (
                f".loop {ls} {j} ({lo}) {self.ip(cnt)} {self.prog(step, False)} "
                f"{self.prog(init, False)}\n    {self.ex(inv, '    ')}"
            )
        elif p.kind == "ten":
            text = f".ten {p.args[0]} {self.prog(p.args[1], False)} {self.prog(p.args[2], False)}"
        else:
            text = f".{p.kind} {self.prog(p.args[0], False)} {self.prog(p.args[1], False)}"
        return text if top else f"({text})"


# sympy expressions as expressions

AID = 5000000
"""The first atom variable of a leaf."""


@dataclass
class Vars:
    """The variable of every symbol: dimension symbols (and powers ``d^n`` as their own
    symbols) by index, parameters from ``PAR``, pattern variables from ``IN``, summation
    indices from ``BV``, atoms from ``AID``; the range of every index variable that has one."""

    dsyms: list[sympy.Symbol] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    ids: dict[sympy.Symbol, int] = field(default_factory=dict)
    ranges: dict[int, sympy.Expr] = field(default_factory=dict)
    pw: list[tuple[Y.DMon, Y.DMon]] = field(default_factory=list)
    powers: dict[sympy.Symbol, tuple[sympy.Expr, sympy.Symbol]] = field(default_factory=dict)
    widths: list[sympy.Symbol] = field(default_factory=list)
    funcs: list[str] = field(default_factory=list)
    needs_width: bool = False
    next_bound: int = BV
    next_atom: int = AID
    atoms: list[Atom] = field(default_factory=list)

    def dim(self, s: sympy.Symbol) -> None:
        if s not in self.dsyms:
            self.dsyms.append(s)

    def bound(self, s: sympy.Symbol, size: sympy.Expr | None = None) -> int:
        if s not in self.ids:
            self.ids[s] = self.next_bound
            self.next_bound += 1
        if size is not None:
            self.ranges[self.ids[s]] = size
        return self.ids[s]

    def var(self, s: sympy.Basic) -> int:
        if s in self.ids:
            return self.ids[s]
        if not isinstance(s, sympy.Symbol) or isinstance(s, sympy.Dummy):
            raise CertifyError(f"{s} is not an index variable")
        if s in self.dsyms or s in self.widths:
            raise CertifyError(f"the size {s} appears in an index expression")
        if s.name not in self.params:
            self.params.append(s.name)
        return PAR + self.params.index(s.name)

    def count(self, e: sympy.Expr) -> Y.IPoly:
        """A loop count as a polynomial in parameters; a width symbol is read as one."""
        saved = self.widths
        self.widths = []
        try:
            return self.ipoly(e)
        finally:
            self.widths = saved

    def power(self, base: sympy.Expr, n: sympy.Symbol) -> sympy.Symbol:
        """The dimension symbol standing for ``base^n``."""
        for a, (b, m) in self.powers.items():
            if b == base and m == n:
                return a
        a = sympy.Symbol(f"{base}^{n}", positive=True)
        self.powers[a] = (base, n)
        if n not in self.widths:
            self.widths.append(n)
        self.dim(a)
        self.pw.append((self.dmon(base), self.dmon(a)))
        return a

    def dmon(self, e: sympy.Expr) -> Y.DMon:
        e = sympy.expand(sympy.sympify(e))
        if e == 0:
            return (0, ())
        c, rest = e.as_coeff_Mul()
        out = sympy.Integer(1)
        for f in sympy.Mul.make_args(rest):
            if f == 1:
                continue
            base, k = f.as_base_exp()
            if k.free_symbols:
                widths = [s for s in k.free_symbols]
                if len(widths) != 1:
                    raise CertifyError(f"the size {e}")
                n = widths[0]
                p = sympy.Poly(k, n)
                if p.degree() != 1 or not all(x.is_Integer for x in p.all_coeffs()):
                    raise CertifyError(f"the size {e}")
                a, b = (int(x) for x in p.all_coeffs())
                if a < 0 or b < 0:
                    raise CertifyError(f"the size {e}")
                out *= self.power(base, n) ** a * base**b
            else:
                out *= f
        if isinstance(c, sympy.Integer) and int(c) > 1:
            pass
        e2 = sympy.expand(c * out)
        for s in e2.free_symbols:
            if s not in self.dsyms:
                raise CertifyError(f"the size {e} depends on {s}, which is not a dimension")
        try:
            return Y._dmon(e2, self.dsyms)
        except NetUnsupported as exc:
            raise CertifyError(str(exc)) from exc

    def range_of(self, e: sympy.Expr, ctx: sympy.Expr | None) -> sympy.Expr:
        """The modulus an atom's argument is reduced by."""
        if isinstance(e, sympy.Symbol) and e in self.ids and self.ids[e] in self.ranges:
            return self.ranges[self.ids[e]]
        if ctx is not None:
            return ctx
        raise CertifyError(f"no range is known for the atom argument {e}")

    def atom(
        self,
        f: int,
        ks: Sequence[int],
        dp: Sequence[sympy.Expr],
        args: Sequence[tuple[sympy.Expr, sympy.Expr]],
    ) -> int:
        a: Atom = (
            0,
            f,
            tuple(ks),
            tuple(self.dmon(x) for x in dp),
            tuple((self.ipoly(x), self.dmon(m)) for x, m in args),
        )
        for b in self.atoms:
            if b[1:] == a[1:]:
                return b[0]
        a = (self.next_atom, *a[1:])
        self.next_atom += 1
        self.atoms.append(a)
        return a[0]

    def repunits(self, e: sympy.Expr) -> sympy.Expr:
        """``e`` with each ``(d^k - 1)/(d - 1)`` written as a repunit atom's variable."""
        e = sympy.sympify(e)
        mods = {m: sympy.Dummy("mod") for m in e.atoms(sympy.Mod)}
        e = e.xreplace(mods)
        for p in sorted(e.atoms(sympy.Pow), key=str):
            b, k = p.base, p.exp
            if not (k.free_symbols and b in self.dsyms):
                continue
            r = sympy.Dummy("rep", integer=True)
            e2 = sympy.cancel(sympy.together(e.subs(p, r * (b - 1) + 1)))
            if sympy.fraction(e2)[1].free_symbols:
                continue
            self.ids[r] = self.atom(4, [], [b, p], [])
            e = e2
        return e.xreplace({v: k for k, v in mods.items()})

    def ipoly(self, e: sympy.Expr, ctx: sympy.Expr | None = None) -> Y.IPoly:
        """An integer polynomial in index variables and atoms; ``ctx`` the modulus it is read
        under (a register's levels or a root of unity's order)."""
        p: Y.IPoly = ()
        for term in sympy.Add.make_args(sympy.expand(self.repunits(e))):
            c, rest = term.as_coeff_Mul()
            if not c.is_Integer:
                raise CertifyError(f"the index expression {e} has a non-integer coefficient")
            mon: list[int] = []
            for f in sympy.Mul.make_args(rest):
                if f == 1:
                    continue
                base, k = f.as_base_exp()
                if k.free_symbols and (base.is_Integer or base in self.dsyms):
                    if k.free_symbols <= set(self.widths) or (
                        base.is_Integer and k.free_symbols & set(self.widths)
                    ):
                        mon.append(self.atom(8, [], [f], []))
                        continue
                    if base == 2 and isinstance(k, sympy.Symbol):
                        mon.append(self.atom(7, [], [], [(k, sympy.Integer(0))]))
                        continue
                if not (k.is_Integer and int(k) > 0):
                    raise CertifyError(f"the index expression {e} is not a polynomial")
                if isinstance(base, sympy.Symbol):
                    mon += [self.var(base)] * int(k)
                    continue
                inner = self.atom_poly(base, ctx)
                if len(inner) == 1 and inner[0][1] == 1 and len(inner[0][0]) <= 1:
                    mon += list(inner[0][0]) * int(k)
                else:
                    if int(k) != 1:
                        raise CertifyError(f"the index expression {e}")
                    rest_mon = tuple(sorted(mon))
                    for m2, c2 in Y.ip_mul(inner, ((rest_mon, 1),)):
                        p = Y.ip_ins(m2, int(c) * c2, p)
                    mon = []
                    c = sympy.Integer(0)
                    break
            if c != 0:
                p = Y.ip_ins(tuple(sorted(mon)), int(c), p)
        return p

    def atom_poly(self, f: sympy.Expr, ctx: sympy.Expr | None) -> Y.IPoly:
        """A function application as a polynomial: an atom, or a polynomial it equals under
        ``ctx``."""
        from archytaszx.semantics import pathsum as P

        two = ctx is not None and sympy.expand(ctx - 2) == 0
        if isinstance(f, sympy.Mod):
            v, m = f.args
            if ctx is not None and sympy.expand(m - ctx) == 0:
                return self.ipoly(v, ctx)
            return Y.ip_v(self.atom(2, [], [], [(v, m)]))
        if isinstance(f, P.xor):
            a, b = f.args
            if two:
                return self.ipoly(a + b, ctx)
            ma = self.range_of(a, ctx)
            return Y.ip_v(self.atom(1, [], [], [(a, ma), (b, self.range_of(b, ma))]))
        if isinstance(f, P.bit):
            x, j = f.args
            if j == 0:
                if two:
                    return self.ipoly(x, ctx)
                return Y.ip_v(self.atom(2, [], [], [(x, 2)]))
            mx = self.range_of(x, ctx)
            return Y.ip_v(self.atom(5, [], [], [(x, mx), (j, j + 1)]))
        if isinstance(f, (P.bitdot, P.ddot)):
            a, b = f.args[0], f.args[1]
            d = sympy.Integer(2) if isinstance(f, P.bitdot) else f.args[2]
            n = self.range_of(a, None) if isinstance(a, sympy.Symbol) else self.range_of(b, ctx)
            return Y.ip_v(self.atom(0, [], [d, n], [(a, n), (b, n)]))
        if isinstance(f, (P.pop, P.dsum)):
            v = f.args[0]
            d = sympy.Integer(2) if isinstance(f, P.pop) else f.args[1]
            n = self.range_of(v, ctx)
            return Y.ip_v(self.atom(6, [], [d, n], [(v, n)]))
        if isinstance(f, sympy.Function):
            name = type(f).__name__
            if name not in self.funcs:
                self.funcs.append(name)
            args = [(a, self.range_of(a, ctx)) for a in f.args]
            return Y.ip_v(self.atom(100 + self.funcs.index(name), [], [], args))
        raise CertifyError(f"the index expression {f}")

    def phase(self, r: sympy.Expr) -> tuple[Y.CMon, list[Y.Fac]]:
        """``e^(2 pi i r)`` as a root of unity of a dimension monomial."""
        r = sympy.expand(r)
        den: sympy.Expr = sympy.Integer(1)
        for term in sympy.Add.make_args(r):
            den = sympy.lcm(den, sympy.fraction(term)[1])
        num = sympy.expand(r * den)
        dm = self.dmon(den)
        if not num.free_symbols and not num.atoms(sympy.Function):
            if dm[1]:
                return Y.of_q(1), [(dm, self.ipoly(num))]
            q = sympy.Rational(num, dm[0])
            return Y.ex_q(Fraction(int(q.p), int(q.q))), []
        poly = self.ipoly(num, den)
        if dm == (1, ()):
            return Y.of_q(1), []
        return Y.of_q(1), [(dm, poly)]

    def amp(self, amp: sympy.Expr) -> list[tuple[Y.CMon, list[Y.Fac], list[Y.Fac]]]:
        """An amplitude as terms: a coefficient, deltas and roots of unity."""
        out = []
        expanded = sympy.expand(amp, log=False, power_exp=False, power_base=False)
        for term in sympy.Add.make_args(expanded):
            if term == 0:
                continue
            cm, dels, phs = Y.of_q(1), [], []
            for f in sympy.Mul.make_args(term):
                if isinstance(f, sympy.KroneckerDelta):
                    dels.append(self.kronecker(f))
                    continue
                c2, p2 = self.factor(f)
                cm = Y.cm_mul(cm, c2)
                phs += p2
            out.append((cm, dels, phs))
        return out

    def kronecker(self, f: sympy.KroneckerDelta) -> Y.Fac:
        """``[a = b]`` as a delta modulo the range of an index on one side, the other side a
        constant 0 or 1 or an index of the same range."""
        a, b = f.args[0], f.args[1]
        for x, y in ((a, b), (b, a)):
            if isinstance(x, sympy.Symbol) and x in self.ids and self.ids[x] in self.ranges:
                m = self.ranges[self.ids[x]]
                ok = (y.is_Integer and 0 <= int(y) <= 1) or (
                    isinstance(y, sympy.Symbol)
                    and y in self.ids
                    and self.ranges.get(self.ids[y]) == m
                )
                if ok:
                    if y.is_Integer and int(y) == 1:
                        self.needs_width = True
                    return self.dmon(m), Y.ip_sub(self.ipoly(x), self.ipoly(y))
        raise CertifyError(f"the delta {f}")

    def factor(self, f: sympy.Expr) -> tuple[Y.CMon, list[Y.Fac]]:
        if f.is_Rational:
            return Y.of_q(Fraction(int(f.p), int(f.q))), []
        if f == sympy.I:
            return Y.ex_q(Fraction(1, 4)), []
        if isinstance(f, sympy.exp):
            return self.phase(f.args[0] / (2 * sympy.pi * sympy.I))
        base, k = f.as_base_exp()
        if base == -1:
            return self.phase(k / 2)
        if base == sympy.I:
            return self.phase(k / 4)
        if base.is_Rational and base < 0:
            raise CertifyError(f"the amplitude factor {f}")
        if k.is_Rational and (2 * k).is_Integer:
            return Y.rt_d(self.dmon(base), int(2 * k)), []
        if k.free_symbols and (base.is_Integer or base in self.dsyms):
            e2 = sympy.expand(2 * k)
            if len(e2.free_symbols) == 1:
                n = next(iter(e2.free_symbols))
                poly = sympy.Poly(e2, n)
                cs = poly.all_coeffs()
                if poly.degree() == 1 and all(c.is_Integer for c in cs):
                    a, b = int(cs[0]), int(cs[1])
                    m = Y.rt_d(self.dmon(self.power(base, n)), a)
                    return Y.cm_mul(m, Y.rt_d(self.dmon(base), b)), []
        raise CertifyError(f"the amplitude factor {f}")


# path sums read from syntax trees


@dataclass(frozen=True)
class Ket:
    """A ket built from a syntax tree: registers and terms ``(amp, sums, kets)``, each sum a
    symbol and its number of values from 0."""

    widths: tuple[sympy.Expr, ...]
    dims: tuple[sympy.Expr | None, ...]
    terms: tuple[
        tuple[sympy.Expr, tuple[tuple[sympy.Symbol, sympy.Expr], ...], tuple[sympy.Expr, ...]], ...
    ]


def _tensor(a: Ket, b: Ket) -> Ket:
    terms = tuple((x[0] * y[0], x[1] + y[1], x[2] + y[2]) for x in a.terms for y in b.terms)
    return Ket(a.widths + b.widths, a.dims + b.dims, terms)


def _scale(c: sympy.Expr, k: Ket) -> Ket:
    return Ket(k.widths, k.dims, tuple((c * t[0], t[1], t[2]) for t in k.terms))


@dataclass(frozen=True)
class Slot:
    """A pattern variable in scope: its symbol, width and dimension."""

    symbol: sympy.Symbol
    width: sympy.Expr | None = None
    dim: sympy.Expr | None = None
    size: sympy.Expr | None = None


class Reader:
    """Evaluates the syntax trees of states and gate bodies into :class:`Ket` and sympy
    scalars, against a snapshot of the program's variables."""

    def __init__(
        self,
        parse: Callable[[str], object],
        values: Mapping[str, object],
        funcs: Mapping[str, Callable[..., sympy.Expr]] | None = None,
    ) -> None:
        self.parse = parse
        self.values = values
        self.funcs = funcs or {}

    def scalar(self, node: object, loc: Mapping[str, Slot]) -> sympy.Expr:
        v = self.eval(node, loc)
        if isinstance(v, Ket):
            raise CertifyError("a state where a scalar is expected")
        return v

    def eval(self, node: object, loc: Mapping[str, Slot]) -> sympy.Expr | Ket:
        from archytaszx.repl import lang as L

        if isinstance(node, L.Num):
            return node.value
        if isinstance(node, L.Name):
            if node.id in loc:
                return loc[node.id].symbol
            if node.id in self.values:
                v = self.values[node.id]
                if isinstance(v, sympy.Basic):
                    return v
                raise CertifyError(f"{node.id} is not a scalar")
            consts = {"pi": sympy.pi, "e": sympy.E, "i": sympy.I}
            if node.id in consts:
                return consts[node.id]
            raise CertifyError(f"{node.id} is not a scalar")
        if isinstance(node, L.Neg):
            v = self.eval(node.arg, loc)
            return _scale(sympy.Integer(-1), v) if isinstance(v, Ket) else -v
        if isinstance(node, L.Call):
            fns = {"sqrt": sympy.sqrt, "exp": sympy.exp, **self.funcs}
            if node.name not in fns or node.name in ("rand", "arg", "normalise"):
                raise CertifyError(f"the function {node.name}")
            return fns[node.name](*(self.scalar(a, loc) for a in node.args))
        if isinstance(node, L.SumNode):
            lo = self.scalar(node.lo, loc)
            hi = self.scalar(node.hi, loc) + (1 if node.inclusive else 0)
            if lo != 0:
                raise CertifyError("a sum that does not start at 0")
            s = sympy.Dummy(node.var, integer=True)
            body = self.eval(node.body, {**loc, node.var: Slot(s, size=hi)})
            if not isinstance(body, Ket):
                raise CertifyError("a scalar sum")
            return Ket(
                body.widths, body.dims, tuple((t[0], ((s, hi),) + t[1], t[2]) for t in body.terms)
            )
        if isinstance(node, L.Braket):
            if node.bra:
                raise CertifyError("a bra inside a state")
            return self.ket(node, loc)
        if isinstance(node, L.Juxt) or (isinstance(node, L.Bin) and node.op in ("@", "*")):
            a, b = self.eval(node.left, loc), self.eval(node.right, loc)
            if isinstance(a, Ket) and isinstance(b, Ket):
                if isinstance(node, L.Bin) and node.op == "*":
                    raise CertifyError("a product of states")
                return _tensor(a, b)
            if isinstance(a, Ket):
                return _scale(b, a)
            if isinstance(b, Ket):
                return _scale(a, b)
            return a * b
        if isinstance(node, L.Bin):
            a, b = self.eval(node.left, loc), self.eval(node.right, loc)
            if node.op in ("+", "-") and isinstance(a, Ket) and isinstance(b, Ket):
                if a.widths != b.widths or a.dims != b.dims:
                    raise CertifyError("a sum of states of different layouts")
                if node.op == "-":
                    b = _scale(sympy.Integer(-1), b)
                return Ket(a.widths, a.dims, a.terms + b.terms)
            if node.op == "^@" and isinstance(a, Ket):
                return self.power(a, b)
            if isinstance(a, Ket) or isinstance(b, Ket):
                if node.op == "/" and isinstance(a, Ket):
                    return _scale(1 / b, a)  # type: ignore[operator]
                raise CertifyError(f"the operator {node.op} on a state")
            ops: dict[str, Callable[[Any, Any], Any]] = {
                "xor": self.funcs.get("xor", lambda x, y: sympy.Function("xor")(x, y)),
                "+": lambda x, y: x + y,
                "-": lambda x, y: x - y,
                "/": lambda x, y: x / y,
                "^": lambda x, y: x**y,
                "%": sympy.Mod,
            }
            if node.op not in ops:
                raise CertifyError(f"the operator {node.op}")
            return ops[node.op](a, b)
        raise CertifyError(f"the expression {type(node).__name__}")

    def power(self, a: Ket, n: sympy.Expr) -> Ket:
        if len(a.widths) != 1 or len(a.terms) != 1 or a.terms[0][1] or a.widths[0] != 1:
            raise CertifyError("a tensor power of a state other than one basis qudit")
        amp, _, (v,) = a.terms[0]
        if v != 0 or amp != 1:
            raise CertifyError("a tensor power of a basis state other than |0>")
        return Ket((n,), a.dims, ((sympy.Integer(1), (), (sympy.Integer(0),)),))

    def ket(self, node: object, loc: Mapping[str, Slot]) -> Ket:
        from archytaszx.repl import lang as L

        assert isinstance(node, L.Braket)
        content = node.content.strip()
        dim = None if node.dim is None else self.scalar(self.parse(node.dim), loc)
        if len(content) > 1 and content.endswith("`") and content[:-1].isdigit():
            parts = [
                self.basis(sympy.Integer(int(c)), None, dim, explicit=dim is not None)
                for c in content[:-1]
            ]
            out = parts[0]
            for p in parts[1:]:
                out = _tensor(out, p)
            return out
        if content in _BUILTIN_KETS and dim is None:
            c = _BUILTIN_KETS[content]
            half = 1 / sympy.sqrt(2)
            return Ket(
                (sympy.Integer(1),),
                (sympy.Integer(2),),
                ((half, (), (sympy.Integer(0),)), (c * half, (), (sympy.Integer(1),))),
            )
        ast = self.parse(content)
        value = self.scalar(ast, loc)
        slot = loc.get(ast.id) if isinstance(ast, L.Name) else None
        return self.basis(value, slot, dim, explicit=dim is not None)

    def basis(
        self, value: sympy.Expr, slot: Slot | None, dim: sympy.Expr | None, explicit: bool
    ) -> Ket:
        if dim is None and slot is not None and slot.dim is not None:
            dim = slot.dim
        width: sympy.Expr | None = None
        if slot is not None and slot.width is not None:
            width = slot.width
        elif slot is not None and slot.size is not None:
            d = dim if dim is not None else sympy.Integer(2)
            if sympy.expand(slot.size - d) == 0:
                width = sympy.Integer(1)
        elif explicit:
            width = sympy.Integer(1)
        if width is None and dim is None and value.is_Integer:
            width = sympy.Integer(1)
        return Ket((width,), (dim,), ((sympy.Integer(1), (), (value,)),))


def fix_layout(k: Ket, widths: Sequence[sympy.Expr], dims: Sequence[sympy.Expr]) -> Ket:
    """``k`` laid out on ``widths`` and ``dims``; its unresolved registers take them."""
    if len(k.widths) != len(widths):
        raise CertifyError("the registers do not match the engine's layout")
    for w, d, w2, d2 in zip(k.widths, k.dims, widths, dims, strict=True):
        if w is not None and sympy.expand(w - w2) != 0:
            raise CertifyError("a register width does not match the engine's layout")
        if d is not None and sympy.expand(d - d2) != 0:
            raise CertifyError("a register dimension does not match the engine's layout")
    return Ket(tuple(widths), tuple(dims), k.terms)


def term_atoms(vs: Vars, facs: Sequence[Y.Fac]) -> tuple[Atom, ...]:
    """The atoms the factors read, with the atoms those read, in creation order."""
    need = {v for f in facs for v in Z.ip_vs(f[1]) if v >= AID}
    grew = True
    while grew:
        grew = False
        for a in vs.atoms:
            if a[0] in need:
                for v in Z.at_vs(a):
                    if v >= AID and v not in need:
                        need.add(v)
                        grew = True
    return tuple(a for a in vs.atoms if a[0] in need)


def ket_terms(vs: Vars, k: Ket) -> list[ATm]:
    """``Σ amp Π_i [L_i ∣ o_i - v_i]``, output register ``i`` at variable ``i``."""
    lexprs = [d**w for w, d in zip(k.widths, k.dims, strict=True)]
    levels = [vs.dmon(x) for x in lexprs]
    for i, x in enumerate(lexprs):
        vs.ranges[i] = x
    out: list[ATm] = []
    for amp, sums, kets in k.terms:
        bound = tuple((vs.bound(s, n), vs.dmon(n)) for s, n in sums)
        dels = []
        for i, (v, lv, lx) in enumerate(zip(kets, levels, lexprs, strict=True)):
            if isinstance(v, sympy.Mod) and sympy.expand(v.args[1] - lx) == 0:
                v = v.args[0]
            dels.append((lv, Y.ip_sub(Y.ip_v(i), vs.ipoly(v, lx))))
        for cm, adels, phs in vs.amp(amp):
            facs = dels + adels + phs
            out.append((cm, bound, term_atoms(vs, facs), tuple(dels + adels), tuple(phs)))
    return out


def scalar_terms(vs: Vars, v: sympy.Expr) -> list[ATm]:
    """A scalar as constant terms."""
    out: list[ATm] = []
    for cm, dels, phs in vs.amp(v):
        out.append((cm, (), term_atoms(vs, dels + phs), tuple(dels), tuple(phs)))
    return out


def state_terms(vs: Vars, s: State) -> list[ATm]:
    """A printed state as an expression."""
    terms = []
    for t in s.terms:
        sums = []
        for b in t.bound:
            if b.lo != 0:
                raise CertifyError("a printed sum that does not start at 0")
            sums.append((b.symbol, b.hi + 1))
            vs.bound(b.symbol, b.hi + 1)
        terms.append((t.amp, tuple(sums), t.kets))
    widths = tuple(w for w in s.widths)
    dims = tuple(d for d in s.dims)
    return ket_terms(vs, Ket(widths, dims, tuple(terms)))


# checking


def check_file(path: str) -> tuple[bool, str]:
    """``lake env lean path`` inside the bundled Lean project: ``(accepted, output)``."""
    lake = _lake()
    if lake is None:
        return False, "Lean (lake) is not installed"
    done = subprocess.run(
        [lake, "env", "lean", os.path.abspath(path)],
        cwd=lean_project(),
        capture_output=True,
        text=True,
        check=False,
    )
    text = (done.stdout + done.stderr).strip()
    bad = done.returncode != 0 or "error" in text or "sorry" in text
    lines = text.splitlines()
    errors = [i for i, x in enumerate(lines) if "error" in x or "sorry" in x]
    shown = [x for i in errors for x in lines[i : i + 4]][:40]
    return not bad, "\n".join(shown) if bad else text


def lean_nat(e: sympy.Expr, names: Mapping[sympy.Symbol, str]) -> str:
    """A natural-number expression with floors as Lean text."""
    e = sympy.sympify(e)
    if e.is_Integer:
        return str(int(e))
    if isinstance(e, sympy.Symbol):
        if e not in names:
            raise CertifyError(f"the outcome expression {e}")
        return names[e]
    if isinstance(e, sympy.Add):
        return "(" + " + ".join(lean_nat(a, names) for a in e.args) + ")"
    if isinstance(e, sympy.Mul):
        return "(" + " * ".join(lean_nat(a, names) for a in e.args) + ")"
    if isinstance(e, sympy.floor):
        num, den = sympy.fraction(sympy.together(e.args[0]))
        if den.is_Integer:
            return f"({lean_nat(sympy.expand(num), names)} / {int(den)})"
    if isinstance(e, sympy.Mod) and e.args[1].is_Integer:
        return f"({lean_nat(e.args[0], names)} % {int(e.args[1])})"
    raise CertifyError(f"the outcome expression {e}")


DECIDES = "(by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)"


MINUS = Prog("leaf", ([(Y.of_q(-1), (), (), (), ())],))


# the certifier


Layout = tuple[tuple[sympy.Expr, ...], tuple[sympy.Expr, ...]]


@dataclass(frozen=True)
class Event:
    """A recorded state update: ``assign`` (``nodes`` is the right side), ``apply``
    (``nodes`` are the operators applied in place, outermost first) or ``opaque`` (``source``
    is the reason); ``value`` is the state after it and ``values`` the variables then."""

    kind: str
    name: str
    nodes: tuple[object, ...]
    source: str
    value: object
    values: Mapping[str, object]


@dataclass(frozen=True)
class GateTree:
    """A gate's pattern slots, body syntax tree and source line."""

    slots: tuple[tuple[str, str | None], ...]
    body: object
    source: str


_BUILTIN_KETS = {"+": sympy.Integer(1), "-": sympy.Integer(-1), "i": sympy.I, "-i": -sympy.I}


def _clean(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class Certifier:
    """Builds the certificate of one result from an interpreter's recorded trees."""

    def __init__(
        self,
        trees: Mapping[str, GateTree],
        log: Sequence[Event],
        parse: Callable[[str], object],
        script: str,
        funcs: Mapping[str, Callable[..., sympy.Expr]] | None = None,
    ) -> None:
        self.trees = trees
        self.log = list(log)
        self.parse = parse
        self.script = _clean(script) if script[:1].isalpha() else f"s_{_clean(script)}"
        self.funcs = funcs or {}
        self.upto = 0
        self.assumptions: list[sympy.Basic] = []
        self.outcome_fn: Callable[[sympy.Expr], sympy.Expr] = lambda v: v
        self.vs = Vars()
        self.decls: list[tuple[str, Prog]] = []
        self.done: dict[object, Prog] = {}
        self.text = Text()

    # names and declarations

    def declare(self, key: object, base: str, comment: str, p: Prog) -> Prog:
        names = {q.name for q in self.done.values()}
        name, k = base, 1
        while name in names:
            k += 1
            name = f"{base}_{k}"
        named = Prog(p.kind, p.args, name)
        self.done[key] = named
        self.decls.append((comment, named))
        return named

    # the left side

    def kind(self, node: object) -> str:
        from archytaszx.repl import lang as L

        if isinstance(node, L.Name):
            return "op" if node.id in self.trees or node.id == "I" else "scalar"
        if isinstance(node, L.Braket):
            return "bra" if node.bra else "ket"
        if isinstance(node, L.SumNode):
            return self.kind(node.body)
        if isinstance(node, L.Neg):
            return self.kind(node.arg)
        if isinstance(node, L.Juxt) or (isinstance(node, L.Bin) and node.op in ("*", "/")):
            a, b = self.kind(node.left), self.kind(node.right)
            if a == "scalar":
                return b
            if b == "scalar":
                return a
            if (a, b) == ("bra", "ket"):
                return "scalar"
            if (a, b) in (("op", "op"), ("ket", "bra")):
                return "op"
            return "ket" if b == "ket" else a
        if isinstance(node, L.Bin) and node.op in ("@", "^@", "+", "-"):
            return self.kind(node.left)
        return "scalar"

    def named(self, node: object) -> bool:
        """Whether ``node`` reads a named state or applies an operator."""
        from archytaszx.repl import lang as L

        if isinstance(node, L.Braket):
            return self.is_state(node.content.strip())
        if isinstance(node, L.Name):
            return self.kind(node) == "op"
        children = [getattr(node, f) for f in ("left", "right", "arg", "body") if hasattr(node, f)]
        return any(self.named(c) for c in children)

    def is_state(self, name: str) -> bool:
        return any(e.name == name for e in self.log[: self.upto])

    def layout_of(self, value: object) -> Layout:
        if not isinstance(value, State):
            raise CertifyError("a density matrix")
        if any(w is None or d is None for w, d in zip(value.widths, value.dims, strict=True)):
            raise CertifyError("an unresolved register")
        return tuple(value.widths), tuple(value.dims)

    def levels(self, lay: Layout) -> list[Y.DMon]:
        return [self.vs.dmon(d**w) for w, d in zip(*lay, strict=True)]

    def state(self, index: int) -> tuple[Prog, Layout]:
        """The program of the state after event ``index``."""
        ev = self.log[index]
        if index in self.done:
            return self.done[index], self.layout_of(ev.value)
        if ev.kind == "opaque":
            raise CertifyError(ev.source)
        saved = self.upto
        saved_reader = getattr(self, "reader", None)
        self.upto = index
        self.reader = Reader(self.parse, ev.values, self.funcs)
        try:
            lay = self.layout_of(ev.value)
            if ev.kind == "assign":
                p, got = self.expr(ev.nodes[0], lay)
            elif ev.kind == "loop":
                p, got = self.loop(ev, index)
            else:
                prev = self.previous(ev.name, index)
                p, got = self.state(prev)
                for op in reversed(ev.nodes):
                    q, out = self.op(op, got)
                    p, got = Prog("app", (self.levels(got), q, p)), out
        finally:
            self.upto = saved
            if saved_reader is not None:
                self.reader = saved_reader
        self.match(got, lay)
        return self.declare(index, f"state_{_clean(ev.name)}", ev.source, p), lay

    def loop(self, ev: Event, index: int) -> tuple[Prog, Layout]:
        """A loop over a symbolic range whose body updates the state by operators applied to
        it: the loop node with the engine's invariant."""
        from archytaszx.repl import lang as L

        body, var_sym, lo, hi, invariant = ev.nodes
        try:
            inv = invariant()  # type: ignore[operator]
        except Exception as exc:
            raise CertifyError(f"the loop has no closed-form invariant: {exc}") from exc
        init, lay = self.state(self.previous(ev.name, index))
        self.reader = Reader(self.parse, ev.values, self.funcs)
        ops: list[object] = []
        node = body
        while isinstance(node, L.Juxt):
            ops.append(node.left)
            node = node.right
        if not (
            ops and isinstance(node, L.Braket) and not node.bra and node.content.strip() == ev.name
        ):
            raise CertifyError("a loop body other than operators applied to the updated state")
        step, out = self.op(ops[-1], lay)
        for o in reversed(ops[:-1]):
            q, mid = self.op(o, out)
            step, out = Prog("app", (self.levels(out), q, step)), mid
        self.match(out, lay)
        if not (sympy.sympify(lo).is_Integer):
            raise CertifyError("a loop whose range does not start at an integer")
        lo_e, hi_e = sympy.sympify(lo), sympy.sympify(hi)
        j = self.vs.var(cast(sympy.Basic, var_sym))
        cnt = self.vs.count(sympy.expand(hi_e + 1 - lo_e))
        inv_terms = state_terms(self.vs, cast(State, inv))
        return Prog("loop", (self.levels(lay), j, int(lo_e), cnt, step, init, inv_terms)), lay

    def previous(self, name: str, index: int) -> int:
        for k in range(index - 1, -1, -1):
            if self.log[k].name == name:
                return k
        raise CertifyError(f"|{name}> is not defined")

    @staticmethod
    def match(got: Layout, want: Layout) -> None:
        for a, b in zip(got[0] + got[1], want[0] + want[1], strict=True):
            if sympy.expand(a - b) != 0:
                raise CertifyError("the registers differ from the engine's layout")

    def leaf(self, node: object, want: Layout | None) -> tuple[Prog, Layout]:
        k = self.reader.eval(node, {})
        if not isinstance(k, Ket):
            raise CertifyError("a scalar where a state is expected")
        if want is not None and len(want[0]) == len(k.widths):
            k = fix_layout(k, *want)
        else:
            k = fix_layout(
                k,
                [w if w is not None else sympy.Integer(1) for w in k.widths],
                [d if d is not None else sympy.Integer(2) for d in k.dims],
            )
        return Prog("leaf", (ket_terms(self.vs, k),)), (k.widths, k.dims)

    def scalar_leaf(self, node: object) -> Prog:
        return Prog("leaf", (scalar_terms(self.vs, self.reader.scalar(node, {})),))

    def scalar_prog(self, node: object) -> Prog:
        """A scalar-valued tree."""
        from archytaszx.repl import lang as L

        if not self.named(node):
            return self.scalar_leaf(node)
        if isinstance(node, L.Juxt) and isinstance(node.left, L.Braket) and node.left.bra:
            ket = L.Braket(node.left.content, False, node.left.dim)
            pa, la = self.expr(ket)
            pb, lb = self.expr(node.right)
            self.match(la, lb)
            return Prog("inn", (self.levels(la), pa, pb))
        if isinstance(node, L.Juxt) or (isinstance(node, L.Bin) and node.op == "*"):
            return Prog("mul", (self.scalar_prog(node.left), self.scalar_prog(node.right)))
        if isinstance(node, L.Bin) and node.op in ("+", "-"):
            q = self.scalar_prog(node.right)
            if node.op == "-":
                q = Prog("mul", (MINUS, q))
            return Prog("add", (self.scalar_prog(node.left), q))
        if isinstance(node, L.Neg):
            return Prog("mul", (MINUS, self.scalar_prog(node.arg)))
        raise CertifyError(f"the scalar expression {type(node).__name__}")

    def expr(self, node: object, want: Layout | None = None) -> tuple[Prog, Layout]:
        """A state-valued tree."""
        from archytaszx.repl import lang as L

        if not self.named(node):
            return self.leaf(node, want)
        if isinstance(node, L.Braket) and not node.bra:
            return self.state(self.previous(node.content.strip(), self.upto))
        if isinstance(node, L.Juxt) or (isinstance(node, L.Bin) and node.op in ("*", "/")):
            ka, kb = self.kind(node.left), self.kind(node.right)
            if isinstance(node, L.Juxt) and ka == "op" and kb == "ket":
                p, lay = self.expr(node.right)
                q, out = self.op(node.left, lay)
                return Prog("app", (self.levels(lay), q, p)), out
            if (
                ka == "scalar" and kb == "ket" and node.op != "/"
                if isinstance(node, L.Bin)
                else ka == "scalar" and kb == "ket"
            ):
                p, lay = self.expr(node.right)
                return Prog("mul", (self.scalar_leaf(node.left), p)), lay
            if ka == "ket" and kb == "scalar":
                p, lay = self.expr(node.left)
                s = (
                    node.right
                    if not (isinstance(node, L.Bin) and node.op == "/")
                    else L.Bin("/", L.Num(sympy.Integer(1)), node.right)
                )
                return Prog("mul", (self.scalar_leaf(s), p)), lay
            if ka == "ket" and kb == "ket" and isinstance(node, L.Juxt):
                return self.tensor(node.left, node.right)
            raise CertifyError(f"juxtaposing a {ka} and a {kb}")
        if isinstance(node, L.Bin) and node.op == "@":
            return self.tensor(node.left, node.right)
        if isinstance(node, L.Bin) and node.op in ("+", "-"):
            p, lay = self.expr(node.left, want)
            q, lay2 = self.expr(node.right, lay)
            self.match(lay2, lay)
            if node.op == "-":
                q = Prog("mul", (MINUS, q))
            return Prog("add", (p, q)), lay
        if isinstance(node, L.Neg):
            p, lay = self.expr(node.arg, want)
            return Prog("mul", (MINUS, p)), lay
        raise CertifyError(f"the state expression {type(node).__name__}")

    def tensor(self, a: object, b: object) -> tuple[Prog, Layout]:
        p, la = self.expr(a)
        q, lb = self.expr(b)
        return Prog("ten", (len(la[0]), p, q)), (la[0] + lb[0], la[1] + lb[1])

    def op(self, node: object, lay: Layout) -> tuple[Prog, Layout]:
        """An operator-valued tree acting on registers of layout ``lay``."""
        from archytaszx.repl import lang as L

        if isinstance(node, L.Name) and node.id == "I":
            return self.ident(lay), lay
        if isinstance(node, L.Name):
            return self.gate(node.id, lay)
        if isinstance(node, L.Bin) and node.op == "@":
            return self.op_tensor(self.factors(node), lay)
        if isinstance(node, L.Juxt) or (isinstance(node, L.Bin) and node.op == "*"):
            ka, kb = self.kind(node.left), self.kind(node.right)
            if ka == "op" and kb == "op":
                q, mid = self.op(node.right, lay)
                p, out = self.op(node.left, mid)
                return Prog("app", (self.levels(mid), p, q)), out
            if ka == "scalar" and kb == "op":
                p, out = self.op(node.right, lay)
                return Prog("mul", (self.scalar_leaf(node.left), p)), out
        if isinstance(node, L.Bin) and node.op == "^@":
            return self.power(node, lay)
        raise CertifyError(f"the operator expression {type(node).__name__}")

    def power(self, node: object, lay: Layout) -> tuple[Prog, Layout]:
        """``U^@n`` on one register of width ``n``: the digit-wise kernel
        ``c^n ω_d^(α ddot(x, o))`` of a one-qudit kernel ``c ω_d^(α x o)``."""
        from archytaszx.repl import lang as L

        assert isinstance(node, L.Bin)
        g = node.left
        if not (isinstance(g, L.Name) and g.id in self.trees and len(self.trees[g.id].slots) == 1):
            raise CertifyError("a tensor power of an operator other than a one-register gate")
        if len(lay[0]) != 1:
            raise CertifyError("a tensor power applied to several registers")
        w, d = lay[0][0], lay[1][0]
        count = self.reader.scalar(node.right, {})
        if sympy.expand(count - w) != 0:
            raise CertifyError(f"{g.id}^@{count} on a register of width {w}")
        key = ("pow", g.id, lay)
        if key in self.done:
            return self.done[key], lay
        tree = self.trees[g.id]
        q, want = tree.slots[0]
        if want is not None and sympy.expand(self.reader.scalar(self.parse(want), {}) - d) != 0:
            raise CertifyError(f"{g.id}: |{q}>_{want} on a dimension-{d} register")
        x = sympy.Dummy(q, integer=True)
        o = sympy.Dummy("o", integer=True)
        self.vs.ids[x], self.vs.ids[o] = IN, 0
        k = self.reader.eval(tree.body, {q: Slot(x, sympy.Integer(1), d)})
        if not (isinstance(k, Ket) and len(k.terms) == 1 and len(k.terms[0][1]) == 1):
            raise CertifyError(f"{g.id} is not a phase kernel")
        amp, ((y, size),), (val,) = k.terms[0]
        if isinstance(val, sympy.Mod):
            val = val.args[0]
        if val != y or sympy.expand(size - d) != 0:
            raise CertifyError(f"{g.id} is not a phase kernel")
        yid = self.vs.bound(y, d)
        terms = self.vs.amp(amp)
        dm = self.vs.dmon(d)
        if len(terms) != 1 or terms[0][1] or len(terms[0][2]) != 1:
            raise CertifyError(f"{g.id} is not a phase kernel")
        cm, _, ((pd, poly),) = terms[0]
        mono = tuple(sorted((IN, yid)))
        if pd != dm or len(poly) != 1 or poly[0][0] != mono:
            raise CertifyError(f"{g.id} is not a phase kernel")
        alpha = poly[0][1]
        ks = [kk for kk in range(-4, 5) if Y.rt_d(dm, kk) == cm]
        if not ks:
            raise CertifyError(f"{g.id} has a coefficient other than a power of sqrt({d})")
        n_lev = d**w
        self.vs.ranges[IN] = n_lev
        self.vs.ranges[0] = n_lev
        aid = self.vs.atom(0, [], [d, n_lev], [(x, n_lev), (o, n_lev)])
        atom = next(a for a in self.vs.atoms if a[0] == aid)
        term: ATm = (
            Y.rt_d(self.vs.dmon(n_lev), ks[0]),
            (),
            (atom,),
            (),
            ((dm, (((aid,), alpha),)),),
        )
        p = Prog("leaf", ([term],))
        return self.declare(
            key, f"gate_{_clean(g.id)}_pow", f"gate {tree.source}, to the power @{count}", p
        ), lay

    def factors(self, node: object) -> list[object]:
        from archytaszx.repl import lang as L

        if isinstance(node, L.Bin) and node.op == "@":
            return self.factors(node.left) + self.factors(node.right)
        return [node]

    def span(self, node: object) -> int | None:
        from archytaszx.repl import lang as L

        if isinstance(node, L.Bin) and node.op == "^@":
            return 1
        if isinstance(node, L.Name) and node.id in self.trees:
            return len(self.trees[node.id].slots)
        if isinstance(node, L.Juxt):
            return self.span(node.right)
        if isinstance(node, L.Bin) and node.op == "@":
            parts = [self.span(f) for f in self.factors(node)]
            return None if any(x is None for x in parts) else sum(x for x in parts if x is not None)
        return None

    def op_tensor(self, fs: list[object], lay: Layout) -> tuple[Prog, Layout]:
        from archytaszx.repl import lang as L

        n = len(lay[0])
        if len(fs) == 1:
            return self.op(fs[0], lay)
        f = fs[0]
        k = self.span(f)
        if k is None:
            if isinstance(f, L.Name) and f.id == "I":
                k = 1
            else:
                raise CertifyError("an operator factor of unknown span")
        if k >= n:
            raise CertifyError("the operator does not cover the registers")
        head = (lay[0][:k], lay[1][:k])
        tail = (lay[0][k:], lay[1][k:])
        p, ph = self.op(f, head)
        if len(ph[0]) != k:
            raise CertifyError("a tensor factor that changes the number of registers")
        q, pt = self.op_tensor(fs[1:], tail)
        return Prog("ten", (k, p, q)), (ph[0] + pt[0], ph[1] + pt[1])

    def ident(self, lay: Layout) -> Prog:
        key = ("I", lay)
        if key in self.done:
            return self.done[key]
        dels = tuple(
            (lv, Y.ip_sub(Y.ip_v(i), Y.ip_v(IN + i))) for i, lv in enumerate(self.levels(lay))
        )
        p = Prog("leaf", ([(Y.of_q(1), (), (), dels, ())],))
        return self.declare(key, "ident", f"I on {len(lay[0])} register(s)", p)

    def gate(self, name: str, lay: Layout) -> tuple[Prog, Layout]:
        from archytaszx.repl import lang as L

        if name not in self.trees:
            raise CertifyError(f"{name} is not a gate")
        tree = self.trees[name]
        if isinstance(tree.body, (L.MatrixNode, L.MatrixFor)):
            raise CertifyError(f"the matrix gate {name}")
        if len(tree.slots) != len(lay[0]):
            raise CertifyError(f"{name} acts on {len(tree.slots)} register(s), got {len(lay[0])}")
        key = ("gate", name, lay)
        loc = {}
        for i, ((q, want), w, d) in enumerate(zip(tree.slots, *lay, strict=True)):
            if want is not None and sympy.expand(self.reader.scalar(self.parse(want), {}) - d) != 0:
                raise CertifyError(f"{name}: |{q}>_{want} on a dimension-{d} register")
            s = sympy.Dummy(q, integer=True)
            self.vs.ids[s] = IN + i
            self.vs.ranges[IN + i] = d**w
            loc[q] = Slot(s, w, d)
        k = self.reader.eval(tree.body, loc)
        if not isinstance(k, Ket):
            raise CertifyError(f"the body of {name} is not a ket")
        if len(k.widths) == len(lay[0]):
            widths = [w if w is not None else lw for w, lw in zip(k.widths, lay[0], strict=True)]
            dims = [d if d is not None else ld for d, ld in zip(k.dims, lay[1], strict=True)]
        else:
            widths = [w if w is not None else sympy.Integer(1) for w in k.widths]
            dims = [d if d is not None else sympy.Integer(2) for d in k.dims]
        k = fix_layout(k, widths, dims)
        out: Layout = (k.widths, k.dims)
        if key in self.done:
            return self.done[key], out
        p = Prog("leaf", (ket_terms(self.vs, k),))
        return self.declare(key, f"gate_{_clean(name)}", f"gate {tree.source}", p), out

    # the theorem

    def setup(self, upto: int) -> None:
        self.upto = upto
        for ev in self.log[:upto]:
            if isinstance(ev.value, State):
                for d in ev.value.dims:
                    for s in sympy.sympify(d).free_symbols if d is not None else ():
                        self.vs.dim(s)
                for w in ev.value.widths:
                    for s in sympy.sympify(w).free_symbols if w is not None else ():
                        if s not in self.vs.widths:
                            self.vs.widths.append(s)

    def last(self, name: str) -> int:
        for k in range(len(self.log) - 1, -1, -1):
            if self.log[k].name == name:
                return k
        raise CertifyError(f"|{name}> has no recorded definition")

    def floors(self) -> list[int]:
        wide = self.vs.needs_width
        return [(2 if wide else 0) if s in self.vs.powers else 2 for s in self.vs.dsyms]

    def check(self, p: Prog, rhs: Sequence[ATm]) -> list[str]:
        """The translation and normal-form declarations of a closing check."""
        ok = cert_check(self.floors(), self.vs.pw, p, rhs)
        if not ok and self.vs.powers and not self.vs.needs_width:
            self.vs.needs_width = True
            ok = cert_check(self.floors(), self.vs.pw, p, rhs)
        if not ok:
            raise CertifyError("the normaliser does not close the difference")
        e = prog_sym(p, self.floors(), self.vs.pw)
        assert e is not None
        ns = [Z.norm_a(self.floors(), self.vs.pw, Z.AROUNDS, t) for t in e + Z.ax_neg(rhs)]
        self.sym, self.nfs = e, ns
        return []

    def hint_decls(self, k: str = "") -> list[str]:
        nfs = ",\n   ".join("none" if t is None else f"some {self.text.tm(t)}" for t in self.nfs)
        return [
            (
                f"/-- The program as one expression. -/\ndef SYM{k} : List ATm :=\n  "
                f"{self.text.ex(self.sym)}\n"
            ),
            (
                f"/-- The normal form of each term of the difference. -/\n"
                f"def NF{k} : List (Option ATm) :=\n  [{nfs}]\n"
            ),
        ]

    def lit(self, p: Prog, rhs: Sequence[ATm], k: str, label: str) -> tuple[str, str]:
        """The declarations and proof of ``p.sem V = fun ν => axDen V ν R``."""
        self.check(p, rhs)
        decls = [
            f"/-- {label} -/\ndef R{k} : List ATm :=\n  {self.text.ex(rhs)}\n",
            *self.hint_decls(k),
        ]
        return "\n".join(decls), self.lit_proof(p, f"R{k}", k)

    def lit_proof(self, p: Prog, r: str, k: str) -> str:
        if counts(p):
            return f"certify_lit_c V hρ FL FLOK PW PWOK {p.name} {r} SYM{k} NF{k} " + DECIDES
        return f"certify_lit V hρ FL FLOK PW PWOK {p.name} {r} SYM{k} NF{k} " + DECIDES

    def lit_stmt(self, p: Prog, r: str) -> str:
        if counts(p):
            return (
                f"∀ ν : ℕ → ℤ, (∀ c ∈ {p.name}.counts, 0 ≤ ipDen ν c) → "
                f"{p.name}.sem V ν = axDen V ν {r}"
            )
        return f"{p.name}.sem V = fun ν => axDen V ν {r}"

    def certify_state(self, name: str, value: State) -> str:
        """The Lean source certifying the printed value of ``|name>``."""
        index = self.last(name)
        self.setup(len(self.log))
        p, _ = self.state(index)
        rhs = state_terms(self.vs, value)
        self.check(p, rhs)
        printed = f"printed_{_clean(name)}"
        extra = [
            (
                f"/-- The printed value of |{name}>. -/\ndef {printed} : List ATm :=\n  "
                f"{self.text.ex(rhs)}\n"
            ),
            *self.hint_decls(),
        ]
        return self.source(
            f"|{name}>",
            f"{self.script}_{_clean(name)}",
            self.lit_stmt(p, printed),
            self.lit_proof(p, printed, ""),
            extra,
            [],
        )

    def certify_scalar(
        self, name: str, node: object, upto: int, values: Mapping[str, object], value: sympy.Expr
    ) -> str:
        """The Lean source certifying the printed value of the scalar ``name``."""
        self.setup(upto)
        self.reader = Reader(self.parse, values, self.funcs)
        p = self.declare(
            ("scalar", name), f"value_{_clean(name)}", f"var {name} = ...", self.scalar_prog(node)
        )
        rhs = scalar_terms(self.vs, value)
        self.check(p, rhs)
        printed = f"printed_{_clean(name)}"
        extra = [
            (
                f"/-- The printed value of {name}. -/\ndef {printed} : List ATm :=\n  "
                f"{self.text.ex(rhs)}\n"
            ),
            *self.hint_decls(),
        ]
        return self.source(
            name,
            f"{self.script}_{_clean(name)}",
            self.lit_stmt(p, printed),
            self.lit_proof(p, printed, ""),
            extra,
            [],
        )

    def certify_assert(
        self,
        label: str,
        left: object,
        right: object,
        upto: int,
        values: Mapping[str, object],
        source: str,
    ) -> str:
        """The Lean source certifying the assertion ``left == right``."""
        self.setup(upto)
        self.reader = Reader(self.parse, values, self.funcs)
        pl = self.declare(("lhs", label), "lhs", f"assert {source}", self.scalar_prog(left))
        pr = self.declare(("rhs", label), "rhs", f"assert {source}", self.scalar_prog(right))
        er = prog_sym(pr, self.floors(), self.vs.pw)
        if er is None:
            raise CertifyError("the right side does not translate")
        self.check(pl, er)
        extra = [
            (
                f"/-- The right side as one expression. -/\ndef RSYM : List ATm :=\n  "
                f"{self.text.ex(er)}\n"
            ),
            *self.hint_decls(),
        ]
        if counts(pl) or counts(pr):
            stmt = (
                f"∀ ν : ℕ → ℤ, (∀ c ∈ {pl.name}.counts ++ {pr.name}.counts, 0 ≤ ipDen ν c) "
                f"→ {pl.name}.sem V ν = {pr.name}.sem V ν"
            )
            proof = (
                f"certify_eq_lit_c V hρ FL FLOK PW PWOK {pl.name} {pr.name} SYM RSYM NF "
                + DECIDES
                + " (by decide +kernel) (by decide +kernel)"
            )
        else:
            stmt = f"{pl.name}.sem V = {pr.name}.sem V"
            proof = (
                f"certify_eq_lit V hρ FL FLOK PW PWOK {pl.name} {pr.name} SYM RSYM NF "
                + DECIDES
                + " (by decide +kernel) (by decide +kernel)"
            )
        return self.source(
            f"assert {source}", f"{self.script}_{_clean(label)}", stmt, proof, extra, []
        )

    def certify_dist(
        self, name: str, label: str, upto: int, values: Mapping[str, object], raw: Any, mapped: Any
    ) -> str:
        """The Lean source certifying the measured distribution of ``|label>``: each branch's
        probability is ``|<value|psi>|^2``, the probabilities sum to 1 and ``<psi|psi> = 1``;
        with ``mapped``, every outcome maps to its printed value."""
        index = None
        for k in range(upto - 1, -1, -1):
            if self.log[k].name == label and self.log[k].kind != "opaque":
                index = k
                break
        if index is None:
            raise CertifyError(f"|{label}> has no recorded definition before the measurement")
        self.setup(upto)
        psi, lay = self.state(index)
        levels = self.levels(lay)
        decls: list[str] = []
        parts: list[tuple[str, str, str, str]] = []
        total: list[ATm] = []
        for b_i, br in enumerate(raw.branches, 1):
            for b in br.free:
                if b.lo != 0:
                    raise CertifyError("a branch index that does not start at 0")
                self.vs.ids.setdefault(b.symbol, PAR + 500 + len(self.vs.ids))
                self.vs.params.append(b.symbol.name)
            ket = Ket(lay[0], lay[1], ((sympy.Integer(1), (), tuple(br.registers)),))
            leaf = Prog("leaf", (ket_terms(self.vs, ket),))
            amp = Prog("inn", (levels, leaf, psi))
            prob = self.declare(
                ("prob", b_i),
                f"prob_{b_i}",
                f"branch {b_i}: |<value|{label}>|^2",
                Prog("mul", (Prog("cnj", (amp,)), amp)),
            )
            d, pr = self.lit(
                prob, scalar_terms(self.vs, br.prob), f"_{b_i}", f"The probability of branch {b_i}."
            )
            decls.append(d)
            parts.append(
                (
                    f"branch {b_i} of measure|{label}>: probability {br.prob}",
                    f"{self.script}_{_clean(name)}_branch_{b_i}",
                    self.lit_stmt(prob, f"R_{b_i}"),
                    pr,
                )
            )
            sums = []
            for b in br.free:
                sid = self.vs.bound(sympy.Dummy(b.symbol.name, integer=True), b.hi + 1)
                sums.append((sid, self.vs.dmon(b.hi + 1)))
            for t in scalar_terms(self.vs, br.prob.xreplace({b.symbol: b.symbol for b in br.free})):
                total.append((t[0], tuple(sums) + t[1], t[2], t[3], t[4]))
        tot = self.declare(
            ("total",), "total", f"the probabilities of measure|{label}>", Prog("leaf", (total,))
        )
        d, pr = self.lit(
            tot, scalar_terms(self.vs, sympy.Integer(1)), "_total", "The total probability."
        )
        decls.append(d)
        parts.append(
            (
                f"the probabilities of measure|{label}> sum to 1",
                f"{self.script}_{_clean(name)}_total",
                self.lit_stmt(tot, "R_total"),
                pr,
            )
        )
        norm = self.declare(
            ("norm",), "norm", f"<{label}|{label}>", Prog("inn", (levels, psi, psi))
        )
        d, pr = self.lit(norm, scalar_terms(self.vs, sympy.Integer(1)), "_norm", "One.")
        decls.append(d)
        parts.append(
            (
                f"<{label}|{label}> = 1",
                f"{self.script}_{_clean(name)}_norm",
                self.lit_stmt(norm, "R_norm"),
                pr,
            )
        )
        if mapped is not None:
            parts.append(self.pushforward(name, raw, mapped))
        first = parts[0]
        return self.source(first[0], first[1], first[2], first[3], decls, parts[1:])

    def certify_mixed(self, name: str, upto: int, value: Any) -> str:
        """The Lean source certifying the density matrix a full measurement of ``|name>``
        leaves: vectorised, ``[o = o'] psi(o) conj(psi(o'))``."""
        index = None
        for k in range(upto - 1, -1, -1):
            if self.log[k].name == name and self.log[k].kind != "opaque":
                index = k
                break
        if index is None:
            raise CertifyError(f"|{name}> has no recorded definition before the measurement")
        self.setup(upto)
        psi, lay = self.state(index)
        k = len(lay[0])
        levels = self.levels(lay)
        diag = tuple((lv, Y.ip_sub(Y.ip_v(i), Y.ip_v(k + i))) for i, lv in enumerate(levels))
        rho = self.declare(
            ("mixed", name),
            f"rho_{_clean(name)}",
            f"measure|{name}>: the mixture of the outcomes, vectorised",
            Prog(
                "mul",
                (
                    Prog("leaf", ([(Y.of_q(1), (), (), diag, ())],)),
                    Prog("ten", (k, psi, Prog("cnj", (psi,)))),
                ),
            ),
        )
        rhs = state_terms(self.vs, value.vec)
        d, pr = self.lit(rho, rhs, "", f"The printed density matrix of |{name}>, vectorised.")
        return self.source(
            f"the density matrix of |{name}> after measuring it",
            f"{self.script}_{_clean(name)}",
            self.lit_stmt(rho, "R"),
            pr,
            [d],
            [],
        )

    def pushforward(self, name: str, raw: Any, mapped: Any) -> tuple[str, str, str, str]:
        """Every outcome of ``raw`` maps to the value of the single branch of ``mapped``."""
        if len(mapped.branches) != 1 or mapped.branches[0].free:
            raise CertifyError("a function of outcomes with several values")
        target = mapped.branches[0].value
        claims = []
        binders = []
        for br in raw.branches:
            names = {b.symbol: f"k{j}_{b.symbol.name.strip('_')}" for j, b in enumerate(br.free)}
            fn = self.outcome_fn
            image = fn(br.value)
            for b in br.free:
                binders.append(f"∀ {names[b.symbol]} : ℕ, {names[b.symbol]} < {b.hi + 1} →")
            claims.append(
                " ".join(binders) + f" {lean_nat(image, names)} = {lean_nat(target, names)}"
            )
            binders = []
        stmt = " ∧ ".join(f"({c})" for c in claims)
        return (
            f"every outcome of the measurement maps to {target}",
            f"{self.script}_{_clean(name)}_values",
            stmt,
            "by omega" if len(claims) == 1 else "⟨" + ", ".join("by omega" for _ in claims) + "⟩",
        )

    def lean_int(self, e: sympy.Basic) -> str:
        """An integer expression over dimension and width symbols and parameters as Lean
        text."""
        vs = self.vs
        if isinstance(e, sympy.Integer):
            return f"({int(e)} : ℤ)"
        if isinstance(e, sympy.Symbol):
            if e in vs.dsyms:
                return f"(V.ρ.dv {vs.dsyms.index(e)} : ℤ)"
            if e in vs.widths:
                return f"({e.name} : ℤ)"
            return f"ν {vs.var(e)}"
        if isinstance(e, sympy.Add):
            return "(" + " + ".join(self.lean_int(a) for a in e.args) + ")"
        if isinstance(e, sympy.Mul):
            return "(" + " * ".join(self.lean_int(a) for a in e.args) + ")"
        if isinstance(e, sympy.Pow) and e.exp.is_Integer and int(e.exp) >= 0:
            return f"({self.lean_int(e.base)} ^ {int(e.exp)})"
        if isinstance(e, sympy.Pow) and e.exp in vs.widths:
            return f"({self.lean_int(e.base)} ^ {e.exp.name})"
        if isinstance(e, sympy.Mod):
            return f"({self.lean_int(e.args[0])} % {self.lean_int(e.args[1])})"
        raise CertifyError(f"the assumption term {e}")

    def lean_prop(self, a: sympy.Basic) -> str:
        """An assumption as a Lean proposition."""
        ops = {
            sympy.Eq: "=",
            sympy.Ne: "≠",
            sympy.Lt: "<",
            sympy.Le: "≤",
            sympy.Gt: ">",
            sympy.Ge: "≥",
        }
        for cls, op in ops.items():
            if isinstance(a, cls):
                return f"{self.lean_int(a.lhs)} {op} {self.lean_int(a.rhs)}"
        if isinstance(a, sympy.And):
            return " ∧ ".join(f"({self.lean_prop(x)})" for x in a.args)
        raise CertifyError(f"the assumption {a}")

    def assumed(self, stmt: str, proof: str) -> tuple[str, str]:
        """The statement and proof with every assumption a hypothesis on each valuation."""
        if not self.assumptions:
            return stmt, proof
        props = [self.lean_prop(a) for a in self.assumptions]
        if stmt.startswith("∀ ν : ℕ → ℤ, "):
            hyp = " → ".join(f"({p})" for p in props)
            under = " ".join("_" for _ in props)
            return (
                f"∀ ν : ℕ → ℤ, {hyp} → {stmt[len('∀ ν : ℕ → ℤ, ') :]}",
                f"fun ν {under} => ({proof}) ν",
            )
        if " = fun ν => " in stmt:
            lhs, rhs = stmt.split(" = fun ν => ", 1)
            body = f"{lhs} ν = {rhs}"
        else:
            lhs, rhs = stmt.split(" = ", 1)
            body = f"{lhs} ν = {rhs} ν"
        hyp = " → ".join(f"({p})" for p in props)
        under = " ".join("_" for _ in props)
        return (f"∀ ν : ℕ → ℤ, {hyp} → {body}", f"fun ν {under} => congrFun ({proof}) ν")

    def source(
        self,
        what: str,
        thm: str,
        stmt: str,
        proof: str,
        extra: Sequence[str],
        more: Sequence[tuple[str, str, str, str]],
    ) -> str:
        vs = self.vs
        stmt, proof = self.assumed(stmt, proof)
        more = [
            (w, t, *self.assumed(st, pr))
            if " = " in st and not st.startswith("(")
            else (w, t, st, pr)
            for w, t, st, pr in more
        ]
        fl = self.floors()
        for i, s in enumerate(vs.dsyms):
            self.text.dms[(1, tuple([0] * i + [1]))] = f"dim_{_clean(s.name)}"
        dims = []
        for i, s in enumerate(vs.dsyms):
            shown = f"{vs.powers[s][0]}^{vs.powers[s][1]}" if s in vs.powers else s.name
            dims.append(
                f"/-- The dimension {shown}, `V.ρ.dv {i}`. -/\n"
                f"def dim_{_clean(s.name)} : DMon := ⟨1, {lean_nats([0] * i + [1])}⟩"
            )
        hyps = []
        for i, s in enumerate(vs.dsyms):
            if s not in vs.powers:
                hyps.append((f"h_{_clean(s.name)}", f"2 ≤ V.ρ.dv {i}"))
        for n in vs.widths:
            hyps.append((n.name, "ℕ"))
            if vs.needs_width:
                hyps.append((f"h_{n.name}", f"1 ≤ {n.name}"))
        pw_cases = []
        for s, (base, n) in vs.powers.items():
            j = vs.dsyms.index(s)
            rhs = (
                f"V.ρ.dv {vs.dsyms.index(base)} ^ {n.name}"
                if base in vs.dsyms
                else f"{base} ^ {n.name}"
            )
            hname = f"h_{_clean(s.name)}"
            hyps.append((hname, f"V.ρ.dv {j} = {rhs}"))
            names = f"dim_{_clean(s.name)}" + (
                f", dim_{_clean(base.name)}" if base in vs.dsyms else ""
            )
            pw_cases.append(
                f"  · exact ⟨by simp [{names}, DMon.den, npow] <;> omega, {n.name}, "
                f"by simp [{names}, DMon.den, npow, {hname}]⟩"
            )
        hyp_text = " ".join(f"({a} : {b})" for a, b in hyps)
        args_text = " ".join(a for a, _ in hyps)
        fl_cases = []
        for i, s in enumerate(vs.dsyms):
            if s in vs.powers and fl[i] == 2:
                base, n = vs.powers[s]
                hb = f"h_{_clean(base.name)}" if base in vs.dsyms else "(by norm_num)"
                fl_cases.append(
                    f"  · simp only [flv, FL, List.getD_cons_succ, List.getD_cons_zero]\n"
                    f"    exact pow_floor _ _ _ h_{_clean(s.name)} {hb} h_{n.name}"
                )
            else:
                fl_cases.append("  · simp [flv, FL] <;> omega")
        fl_cases.append("  · simp [flv, FL]")
        fl_proof = (
            "fun i => by\n  rcases i with "
            + " | ".join(["_"] * len(vs.dsyms) + ["i"])
            + "\n"
            + "\n".join(fl_cases)
        )
        pw_text = "[" + ", ".join(f"({self.text.dm(a)}, {self.text.dm(b)})" for a, b in vs.pw) + "]"
        if vs.pw:
            pw_proof = (
                "by\n  intro p hp\n  simp only [PW, List.mem_cons, List.not_mem_nil, "
                "or_false] at hp\n  rcases hp with "
                + " | ".join(["rfl"] * len(vs.pw))
                + "\n"
                + "\n".join(pw_cases)
            )
        else:
            pw_proof = "by intro p hp; simp [PW] at hp"
        params = ", ".join(f"{n} is ν {PAR + k}" for k, n in enumerate(vs.params))
        funcs = ", ".join(f"{n} is V.fa {k}" for k, n in enumerate(vs.funcs))
        decls = [f"-- {c}\ndef {p.name} : Prog :=\n  {self.text.prog(p)}\n" for c, p in self.decls]
        sep = " " if hyp_text else ""
        flok = f"(flOK V{sep}{args_text})"
        pwok = f"(pwOK V{sep}{args_text})"
        lines = [
            "/-",
            f"  ArchytasZX certificate: {what} of {self.script}.azx",
            "  The program's meaning (Prog.sem) equals the stated value for every value of the",
            "  dimension and width symbols, parameters, functions and output registers.",
            "  Variables: output register i is ν i, gate input register i is ν (IN + i)"
            + (f"; {params}" if params else "")
            + (f"; {funcs}" if funcs else "")
            + ".",
            "-/",
            "import ArchytasZX.Lang",
            "",
            "open ArchytasZX.Sym ArchytasZX.LSym ArchytasZX.Lang",
            "",
            *(d + "\n" for d in dims),
            *decls,
            *extra,
            f"/-- The dimension floors. -/\ndef FL : List ℕ := {lean_nats(list(fl))}\n",
            f"/-- The dimensions that are powers. -/\ndef PW : List (DMon × DMon) := {pw_text}\n",
            f"theorem flOK (V : LV){sep}{hyp_text} : FloorOK FL V.ρ :=",
            f"  {fl_proof}\n",
            f"theorem pwOK (V : LV){sep}{hyp_text} : PowOK PW V.ρ := {pw_proof}\n",
        ]
        for w, t, st, pr in [(what, thm, stmt, proof), *more]:
            if not t:
                continue
            lines += [
                "set_option maxHeartbeats 0 in",
                "set_option maxRecDepth 100000 in",
                f"/-- {w} -/",
                f"theorem {t} (V : LV) (hρ : V.ρ.OK){sep}{hyp_text} :",
                f"    {st} :=",
                "  " + pr.replace("FLOK", flok).replace("PWOK", pwok),
                "",
                f"#print axioms {t}",
                "",
            ]
        return "\n".join(lines)
