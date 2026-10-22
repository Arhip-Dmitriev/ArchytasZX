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

"""Lean 4 refutations by ball arithmetic.

:func:`export_numeric_refutation` refutes an equality at a point whose values lie outside
``ℚ(ζ)``: irrational phases, square roots of any natural number, ``π``. Both networks are
evaluated by ``eval_cyc`` with each non-rational phase part an atom ``ex g``; the difference
at one boundary assignment is written as a sum of products of atoms, and every subterm gets
a ball (``Near``) from the lemmas of ``ArchytasZX.Ball``: square roots from squares, ``π`` from
its 20-digit bounds, ``ex g`` as ``exp (2π (g/8) i) ^ 8`` with a Taylor bound. A ball away from
zero contradicts the assumed equality.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import mpmath  # type: ignore[import-untyped]
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.phase import INDEX
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_net import NetUnsupported, _head, _join, _paren
from archytaszx.semantics.lean_refute import (
    _Z,
    MAX_ASSIGNMENTS,
    _ints,
    _ktrim,
    _list_pairs,
    _Refuter,
)
from archytaszx.semantics.lean_scalar import II, NC, Ex, Inv, Om, Rt, ScalarUnsupported

DIGITS = 40
TAYLOR = 10
SQRT_DIGITS = 45

_SCALE = Fraction(10) ** DIGITS


def export_numeric_refutation(
    left: Diagram,
    right: Diagram,
    point: Mapping[str, object],
    name: str = "main",
    describe: Callable[[Diagram], str] | None = None,
) -> str:
    """Lean 4 source proving that ``left`` and ``right`` are not equal for every value of
    their parameters, at ``point``, by ball arithmetic."""
    try:
        return _NumRefuter(left, right, point, describe).run(name)
    except ScalarUnsupported as exc:
        raise NetUnsupported(str(exc)) from exc


def _round(q: Fraction) -> Fraction:
    return Fraction(round(q * _SCALE), _SCALE.numerator)


def _up(q: Fraction) -> Fraction:
    return Fraction(math.ceil(q * _SCALE), _SCALE.numerator)


def _q(text: Fraction) -> str:
    """A rational as a Lean real literal."""
    if text.denominator == 1:
        return f"({text.numerator} : ℝ)"
    return f"(({text.numerator} : ℝ) / {text.denominator})"


class _NumRefuter(_Refuter):
    """Evaluates both networks at a point, non-rational phase parts as atoms, and writes a
    refutation by balls."""

    def __init__(self, left: Diagram, right: Diagram, point: Mapping[str, object],
                 describe: Callable[[Diagram], str] | None) -> None:
        super().__init__(left, right, point, describe, opaque=True)
        if self.arity:
            raise NetUnsupported("the boundaries differ in size")
        self.opq = {sym: expr for sym, expr in self.out.sc.opaques.values()}
        for sym in self.opq:
            self.out.sym.params.pop(str(sym), None)
        self.out.set_params()
        self.lines: list[str] = []
        self.count = 0
        mpmath.mp.dps = 60

    # values at the point

    def phase_sym(self, ph: int, k: int) -> sp.Expr:
        specs = self.out.phase_specs
        entries, rule = specs[ph] if ph < len(specs) else ({}, None)
        v = entries.get(k, sp.Integer(0))
        if rule is not None and k != 0:
            v = v + rule.xreplace({INDEX: k})
        return sp.expand(self.at_point(sp.expand(v)))

    @staticmethod
    def split(v: sp.Expr) -> tuple[sp.Rational, dict[sp.Expr, sp.Rational]]:
        """``v`` as a rational plus rational multiples of non-rational terms."""
        q = sp.Integer(0)
        parts: dict[sp.Expr, sp.Rational] = {}
        for term in sp.Add.make_args(sp.expand(v)):
            if term.is_Rational:
                q += term
                continue
            c, b = term.as_coeff_Mul()
            if not c.is_Rational or b.is_number and not b.free_symbols and b.is_rational:
                raise NetUnsupported(f"the phase term {term}")
            parts[b] = parts.get(b, sp.Integer(0)) + sp.Rational(c)
        return sp.Rational(q), parts

    def ex_turns(self, a: sp.Expr) -> sp.Expr:
        """An exponential's argument in turns."""
        return sp.expand(a.args[0] / (2 * sp.pi * sp.I))

    def ks_of(self, parts: dict[sp.Expr, sp.Rational]) -> tuple[int, ...]:
        return _ktrim([int(parts.get(b, 0) * L) for b, L in zip(self.bases, self.scale,
                                                               strict=True)])

    def mono(self, ks: Sequence[int]) -> sp.Expr:
        out = sp.Integer(1)
        for i, k in enumerate(ks):
            out *= self.xsym[i] ** k if k >= 0 else self.xinv[i] ** (-k)
        return out

    def gs_text(self) -> str:
        return "[" + ", ".join(self.gtext) + "]"

    def atom_phase_proof(self, ph: int, j: int, unfold: str) -> str:
        q, parts = self.values[(ph, j)]
        p = self.P(ph, j)[0]
        z = int((q * self.M - p) / self.M)
        ks = list(self.ks_of(parts))
        return (f"ex_atoms {self.M} {p} ({z}) {self.gs_text()} {_ints(ks)} (by norm_num) "
                f"(by norm_num [{unfold}, List.zipWith] <;> ring)")

    def model_at(self, model: sp.Expr) -> sp.Expr:
        e = sp.sympify(model).xreplace(self.opq)
        subs = {s: self.point[str(s)] for s in e.free_symbols if s != II and str(s) in self.point}
        missing = [s for s in e.free_symbols if s != II and str(s) not in self.point]
        if missing:
            raise NetUnsupported(f"the counterexample gives no value for {missing[0]}")
        return e.xreplace(subs)

    def plan_atoms(self) -> None:
        """The atoms ``g = b / L``, one per non-rational phase term ``b`` at the point."""
        self.s0, self.s1 = self.model_at(self.s0), self.model_at(self.s1)
        self.values: dict[tuple[int, int], tuple[sp.Rational, dict[sp.Expr, sp.Rational]]] = {}
        self.keys = []
        for net in (self.a.net, self.b.net):
            for n in net.nodes:
                if _head(n.kind) in ("B", "S"):
                    continue
                d = self.dimv(n.dim)
                if d <= 0:
                    raise NetUnsupported("a node of dimension 0")
                for j in range(d):
                    if (n.ph, j) not in self.values:
                        self.keys.append((n.ph, j))
                        self.values[(n.ph, j)] = self.split(self.phase_sym(n.ph, j))
        exs = [self.split(self.ex_turns(a)) for model in (self.s0, self.s1)
               for a in sp.preorder_traversal(sp.sympify(model)) if isinstance(a, Ex)]
        dens: dict[sp.Expr, int] = {}
        for _, parts in (*self.values.values(), *exs):
            for b, c in parts.items():
                dens[b] = math.lcm(dens.get(b, 1), int(c.q))
        self.bases = sorted(dens, key=str)
        self.scale = [dens[b] for b in self.bases]
        self.xsym = [sp.Symbol(f"x_lean_{i}") for i in range(len(self.bases))]
        self.xinv = [sp.Symbol(f"xi_lean_{i}") for i in range(len(self.bases))]
        self.gtext = [self.out.sym.render(b / L, "ℝ", "ℝ")
                      for b, L in zip(self.bases, self.scale, strict=True)]

    def atom_needs(self, e: sp.Expr) -> set[int]:
        return set()

    def choose_m(self) -> None:
        self.plan_atoms()
        needs = {1}
        for net in (self.a.net, self.b.net):
            for n in net.nodes:
                if n.kind not in ("B", "S"):
                    needs.add(self.dimv(n.dim))
        for q, _ in self.values.values():
            needs.add(int(q.q))
        self.M = math.lcm(*needs)
        self.ptab = {}
        for key in self.keys:
            q, parts = self.values[key]
            self.ptab[key] = (int((q * self.M) % self.M), self.ks_of(parts))

    def scalar_text(self, model: sp.Expr) -> str:
        return self.ctext(sp.sympify(model))

    # numeric values

    def num(self, e: sp.Expr) -> mpmath.mpc:
        e = sp.sympify(e)
        if e.is_Rational:
            return mpmath.mpc(mpmath.mpf(e.p) / e.q)
        if e == _Z:
            return mpmath.exp(2j * mpmath.pi / self.M)
        if e == II:
            return mpmath.mpc(0, 1)
        if e is sp.pi:
            return mpmath.mpc(mpmath.pi)
        if e in self.xsym:
            return mpmath.exp(2j * mpmath.pi * self.realnum(self.gval_expr(self.xsym.index(e))))
        if e in self.xinv:
            return mpmath.exp(-2j * mpmath.pi * self.realnum(self.gval_expr(self.xinv.index(e))))
        if e.is_Add:
            return mpmath.fsum(self.num(a) for a in e.args)
        if e.is_Mul:
            out = mpmath.mpc(1)
            for a in e.args:
                out *= self.num(a)
            return out
        if e.is_Pow and e.args[1].is_Integer:
            return self.num(e.args[0]) ** int(e.args[1])
        if e.is_Pow and e.args[1] == sp.Rational(1, 2):
            return mpmath.sqrt(self.num(e.args[0]))
        if isinstance(e, Rt):
            return mpmath.mpc(mpmath.sqrt(int(e.args[0])))
        if isinstance(e, Inv):
            return 1 / self.num(e.args[0])
        if isinstance(e, Om):
            return mpmath.exp(2j * mpmath.pi / int(e.args[0]))
        if isinstance(e, NC):
            return mpmath.mpc(int(e.args[0]))
        if isinstance(e, (Ex, sp.exp)):
            return mpmath.exp(self.num(e.args[0]))
        raise NetUnsupported(f"no numeric value for {e}")

    def realnum(self, e: sp.Expr) -> mpmath.mpf:
        return mpmath.re(self.num(e))

    def gval_expr(self, i: int) -> sp.Expr:
        return self.bases[i] / self.scale[i]

    # Lean texts of real and complex constants

    def rtext(self, e: sp.Expr) -> str:
        """A real constant as Lean in the shape `rtree` follows."""
        e = sp.sympify(e)
        if e.is_Rational:
            return _q(Fraction(int(e.p), int(e.q)))
        if e is sp.pi:
            return "Real.pi"
        if e.is_Pow and e.args[1] == sp.Rational(1, 2) and e.args[0].is_Integer:
            return f"Real.sqrt (({int(e.args[0])} : ℕ) : ℝ)"
        if e.is_Pow and e.args[1] == sp.Rational(-1, 2) and e.args[0].is_Integer:
            return f"(Real.sqrt (({int(e.args[0])} : ℕ) : ℝ))⁻¹"
        if e.is_Add:
            out = self.rtext(e.args[0])
            for a in e.args[1:]:
                out = f"({out} + {self.rtext(a)})"
            return out
        if e.is_Mul:
            out = self.rtext(e.args[0])
            for a in e.args[1:]:
                out = f"({out} * {self.rtext(a)})"
            return out
        if e.is_Pow and e.args[1].is_Integer and int(e.args[1]) > 0:
            out = self.rtext(e.args[0])
            for _ in range(int(e.args[1]) - 1):
                out = f"({out} * {self.rtext(e.args[0])})"
            return out
        raise NetUnsupported(f"the real constant {e}")

    def ctext(self, e: sp.Expr) -> str:
        """A scalar at the point as Lean, constants without opaque symbols."""
        e = sp.sympify(e)
        if e is sp.pi:
            return "(Real.pi : ℂ)"
        if e.is_Pow and e.args[1] == sp.Rational(1, 2) and e.args[0].is_Integer:
            return f"((Real.sqrt (({int(e.args[0])} : ℕ) : ℝ) : ℝ) : ℂ)"
        if isinstance(e, sp.exp):
            return f"(Complex.exp {self.out.sc.complex_text(e.args[0])})"
        if e.is_Add:
            return "(" + " + ".join(self.ctext(a) for a in e.args) + ")"
        if e.is_Mul:
            return "(" + " * ".join(self.ctext(a) for a in e.args) + ")"
        if e.is_Pow and e.args[1].is_Integer and int(e.args[1]) >= 0:
            return f"({self.ctext(e.args[0])} ^ {int(e.args[1])})"
        if e.is_Pow and e.args[1].is_Integer:
            return f"({self.ctext(e.args[0])} ^ {-int(e.args[1])})⁻¹"
        return self.out.sc.text(e)

    # balls

    def have(self, stmt: str, proof: str) -> str:
        self.count += 1
        name = f"{getattr(self, 'prefix', 'b')}{self.count}"
        self.lines.append(f"have {name} : {stmt} := {proof}")
        return name

    def rball(self, e: sp.Expr) -> tuple[str, Fraction, Fraction]:
        """A NearR fact for the real constant ``e`` as `rtext` writes it."""
        e = sp.sympify(e)
        t = self.rtext(e)
        if e.is_Rational:
            m = Fraction(int(e.p), int(e.q))
            return self.have(f"NearR {_paren(t)} {_q(m)} 0", f"nearR_const {t}"), m, Fraction(0)
        if e is sp.pi:
            m = Fraction(314159265358979323846, 10**20)
            d = Fraction(1, 10**20)
            return self.have(f"NearR {_paren(t)} {_q(m)} {_q(d)}",
                             f"nearR_round nearR_pi {_q(m)} {_q(d)} (by norm_num)"), m, d
        if e.is_Pow and e.args[1] == sp.Rational(1, 2) and e.args[0].is_Integer:
            p = int(e.args[0])
            m, d = self.sqrt_ball(p)
            proof = (f"nearR_sqrt {p} {_q(m)} {_q(d)} (by norm_num) (by norm_num) (by norm_num) "
                     f"(by norm_num)")
            return self.have(f"NearR {_paren(t)} {_q(m)} {_q(d)}", proof), m, d
        if e.is_Pow and e.args[1] == sp.Rational(-1, 2) and e.args[0].is_Integer:
            p = int(e.args[0])
            name, m, d = self.rball(sp.sqrt(p))
            lo, hi = 1 / (m + d), 1 / (m - d)
            m2 = _round((lo + hi) / 2)
            d2 = _up(max(hi - m2, m2 - lo))
            proof = f"nearR_inv {name} (by norm_num) (by norm_num) (by norm_num)"
            return self.have(f"NearR {_paren(t)} {_q(m2)} {_q(d2)}", proof), m2, d2
        if e.is_Add or e.is_Mul or (e.is_Pow and e.args[1].is_Integer and int(e.args[1]) > 0):
            if e.is_Pow:
                args = [e.args[0]] * int(e.args[1])
                op = "mul"
            else:
                args = list(e.args)
                op = "add" if e.is_Add else "mul"
            name, m, d = self.rball(args[0])
            text = self.rtext(args[0])
            for a in args[1:]:
                n2, m2, d2 = self.rball(a)
                text = f"({text} {'+' if op == 'add' else '*'} {self.rtext(a)})"
                if op == "add":
                    cm, cd = m + m2, d + d2
                else:
                    cm, cd = m * m2, abs(m) * d2 + abs(m2) * d + d * d2
                rm, rd = _round(cm), _up(abs(cm - _round(cm)) + cd)
                name = self.have(f"NearR {_paren(text)} {_q(rm)} {_q(rd)}",
                                 f"nearR_round (nearR_{op} {name} {n2}) {_q(rm)} {_q(rd)} "
                                 f"(by norm_num)")
                m, d = rm, rd
            return name, m, d
        raise NetUnsupported(f"the real constant {e}")

    @staticmethod
    def sqrt_ball(p: int) -> tuple[Fraction, Fraction]:
        s = math.isqrt(p * 10 ** (2 * SQRT_DIGITS))
        m = Fraction(s, 10**SQRT_DIGITS)
        d = Fraction(2, 10**SQRT_DIGITS)
        return m, d

    def cball(self, text: str, proof: str, x: Fraction, y: Fraction, r: Fraction
              ) -> tuple[str, Fraction, Fraction, Fraction]:
        rx, ry = _round(x), _round(y)
        rr = _up(abs(x - rx) + abs(y - ry) + r)
        name = self.have(f"Near {_paren(text)} {_q(rx)} {_q(ry)} {_q(rr)}",
                         f"near_round ({proof}) {_q(rx)} {_q(ry)} {_q(rr)} (by norm_num)")
        return name, rx, ry, rr

    def ex_ball(self, ctx: str, theta: sp.Expr) -> tuple[str, Fraction, Fraction, Fraction]:
        """A Near fact for ``ex ctx`` where ``ctx`` is the real ``theta``."""
        theta = sp.nsimplify(theta)
        z = int(mpmath.nint(self.realnum(theta)))
        reduced = sp.expand(theta - z)
        rt = self.rtext(reduced) if reduced != 0 else "(0 : ℝ)"
        arg = f"(2 * Real.pi * ({rt} / 8))"
        w = f"(Complex.exp (({arg} : ℝ) * Complex.I))"
        if reduced == 0:
            a_name = self.have(f"NearR {_paren(arg)} 0 0", "by simp [NearR]")
            A, eps = Fraction(0), Fraction(0)
        else:
            a_name, A, eps = self.angle_ball(arg, reduced)
        n = TAYLOR
        cos_c = sum(Fraction((-1) ** k) * A ** (2 * k) / math.factorial(2 * k) for k in range(n))
        sin_c = sum(Fraction((-1) ** k) * A ** (2 * k + 1) / math.factorial(2 * k + 1)
                    for k in range(n))
        tb = abs(A) ** (2 * n) * Fraction(2 * n + 1, math.factorial(2 * n) * 2 * n)
        r0 = 2 * (eps + tb)
        name, x, y, r = self.cball(
            w, f"near_exp {arg} {_q(A)} {_q(eps)} {n} (by norm_num) {a_name} (by norm_num)",
            cos_c, sin_c, r0)
        # the cosT/sinT centers need their sums unfolded
        self.lines[-1] = self.lines[-1].replace("(by norm_num)\n", "")
        self.lines[-1] = self.lines[-1][: self.lines[-1].rindex("(by norm_num)")] + \
            "(by norm_num [cosT, sinT, Finset.sum_range_succ, Nat.factorial])"
        cur = (w, name, x, y, r)
        for _ in range(3):
            t, nm, x, y, r = cur
            prod = f"({t} * {t})"
            cx, cy = x * x - y * y, x * y + y * x
            cr = (abs(x) + abs(y)) * r + (abs(x) + abs(y)) * r + r * r
            nm2, x2, y2, r2 = self.cball(prod, f"near_mul {nm} {nm}", cx, cy, cr)
            cur = (prod, nm2, x2, y2, r2)
        t8, nm8, x8, y8, r8 = cur
        stmt_ctx = f"Near (ex {_paren(ctx)}) {_q(x8)} {_q(y8)} {_q(r8)}"
        proof = (f"by rw [ex_congr_int (⟨{z}, by push_cast; ring⟩ : ∃ k : ℤ, {_paren(ctx)} = "
                 f"{rt} + k), ex_pow_eight, ex_eq_exp, show {w} ^ 8 = {t8} by ring]; "
                 f"exact {nm8}")
        name = self.have(stmt_ctx, proof)
        return name, x8, y8, r8

    def angle_ball(self, arg: str, reduced: sp.Expr) -> tuple[str, Fraction, Fraction]:
        """A NearR fact for ``2π (reduced/8)`` written as ``arg``."""
        g_name, m, d = self.rball(reduced)
        pi_name, pm, pd = self.rball(sp.pi)
        two_pi_m = 2 * pm
        two_pi_d = 2 * pd
        n1 = self.have(f"NearR (2 * Real.pi) {_q(two_pi_m)} {_q(two_pi_d)}",
                       f"nearR_round (nearR_mul (nearR_const 2) {pi_name}) {_q(two_pi_m)} "
                       f"{_q(two_pi_d)} (by norm_num)")
        gm8, gd8 = m / 8, d / 8
        rt = self.rtext(reduced)
        n2 = self.have(f"NearR ({rt} / 8) {_q(gm8)} {_q(gd8)}",
                       f"by simpa [div_eq_mul_inv] using nearR_round (nearR_mul {g_name} "
                       f"(nearR_const (1 / 8 : ℝ))) {_q(gm8)} {_q(gd8)} (by norm_num)")
        cm = two_pi_m * gm8
        cd = abs(two_pi_m) * gd8 + abs(gm8) * two_pi_d + two_pi_d * gd8
        rm, rd = _round(cm), _up(abs(cm - _round(cm)) + cd)
        n3 = self.have(f"NearR {_paren(arg)} {_q(rm)} {_q(rd)}",
                       f"nearR_round (nearR_mul {n1} {n2}) {_q(rm)} {_q(rd)} (by norm_num)")
        n4 = self.have(f"|{arg} - {_q(rm)}| ≤ {_q(rd)}", n3)
        return n4, rm, rd

    def atom_ball(self, e: sp.Expr) -> tuple[str, str, Fraction, Fraction, Fraction]:
        """A Near fact for an atom of the difference: its text, name and ball."""
        if e == II:
            return "Complex.I", self.have("Near Complex.I 0 1 0", "near_I"), Fraction(0), \
                Fraction(1), Fraction(0)
        if e.is_Rational:
            t = f"(({int(e.p)} : ℂ) / {int(e.q)})"
            m = Fraction(int(e.p), int(e.q))
            name = self.have(f"Near {_paren(t)} {_q(m)} 0 0",
                             f"by simpa using near_rat ({int(e.p)}) ({int(e.q)})")
            return t, name, m, Fraction(0), Fraction(0)
        if isinstance(e, Rt) or (e.is_Pow and e.args[1] == sp.Rational(1, 2)):
            p = int(e.args[0])
            n, m, d = self.rball(sp.sqrt(p))
            t = f"(rt {p})" if isinstance(e, Rt) else f"((Real.sqrt (({p} : ℕ) : ℝ) : ℝ) : ℂ)"
            return t, self.have(f"Near {_paren(t)} {_q(m)} 0 {_q(d)}", f"near_ofReal {n}"), m, \
                Fraction(0), d
        if isinstance(e, Inv) and isinstance(e.args[0], Rt):
            p = int(e.args[0].args[0])
            n, m, d = self.rball(sp.Pow(sp.Integer(p), sp.Rational(-1, 2), evaluate=False))
            t = f"(rt {p})⁻¹"
            return t, self.have(f"Near {_paren(t)} {_q(m)} 0 {_q(d)}",
                                f"by rw [rt_inv_ofReal]; exact near_ofReal {n}"), m, \
                Fraction(0), d
        if e is sp.pi:
            n, m, d = self.rball(sp.pi)
            return "(Real.pi : ℂ)", self.have(f"Near (Real.pi : ℂ) {_q(m)} 0 {_q(d)}",
                                              f"near_ofReal {n}"), m, Fraction(0), d
        if isinstance(e, Om):
            k = int(e.args[0])
            ctx = f"((1 : ℕ) : ℝ) / (({k} : ℕ) : ℝ)"
            n, x, y, r = self.ex_ball(ctx, sp.Rational(1, k))
            t = f"(om {k})"
            return t, self.have(f"Near {_paren(t)} {_q(x)} {_q(y)} {_q(r)}",
                                f"by rw [show om {k} = om {k} ^ 1 from (pow_one _).symm, "
                                f"om_eq_ex]; exact {n}"), x, y, r
        if e == _Z or (e.is_Pow and e.args[0] == _Z and e.args[1].is_Integer):
            k = 1 if e == _Z else int(e.args[1])
            ctx = f"(({k} : ℕ) : ℝ) / (({self.M} : ℕ) : ℝ)"
            n, x, y, r = self.ex_ball(ctx, sp.Rational(k, self.M))
            t = f"(om {self.M} ^ {k})"
            return t, self.have(f"Near {t} {_q(x)} {_q(y)} {_q(r)}",
                                f"by rw [om_eq_ex {self.M} {k}]; exact {n}"), x, y, r
        if e in self.xsym or e in self.xinv:
            i = (self.xsym if e in self.xsym else self.xinv).index(e)
            g = self.gval_expr(i)
            if e in self.xsym:
                n, x, y, r = self.ex_ball(self.gtext[i], g)
                return f"(ex {_paren(self.gtext[i])})", n, x, y, r
            n, x, y, r = self.ex_ball(f"-{_paren(self.gtext[i])}", -g)
            t = f"(ex {_paren(self.gtext[i])})⁻¹"
            return t, self.have(f"Near {_paren(t)} {_q(x)} {_q(y)} {_q(r)}",
                                f"by rw [ex_inv]; exact {n}"), x, y, r
        if isinstance(e, (sp.exp, Ex)):
            turns = sp.nsimplify(sp.expand(e.args[0] / (2 * sp.pi * sp.I)))
            theta = self.rtext(turns)
            arg = self.out.sc.complex_text(e.args[0])
            n, x, y, r = self.ex_ball(theta, turns)
            t = f"(Complex.exp {arg})"
            return t, self.have(f"Near {_paren(t)} {_q(x)} {_q(y)} {_q(r)}",
                                f"by rw [exp_eq_ex {arg} {theta} (by push_cast; ring)]; exact {n}"
                                ), x, y, r
        raise NetUnsupported(f"no ball for {e}")

    def tree(self, e: sp.Expr) -> tuple[str, str, Fraction, Fraction, Fraction]:
        """A Near fact for ``e``: its text, name and ball."""
        e = sp.sympify(e)
        if e.is_Pow and e.args[0] == _Z:
            return self.atom_ball(e)
        if e.is_Add or e.is_Mul or (e.is_Pow and e.args[1].is_Integer and int(e.args[1]) > 1):
            if e.is_Pow:
                args, op = [e.args[0]] * int(e.args[1]), "mul"
            else:
                args, op = list(e.args), ("add" if e.is_Add else "mul")
            t, n, x, y, r = self.tree(args[0])
            for a in args[1:]:
                t2, n2, x2, y2, r2 = self.tree(a)
                if op == "add":
                    cx, cy, cr = x + x2, y + y2, r + r2
                    text = f"({t} + {t2})"
                else:
                    cx, cy = x * x2 - y * y2, x * y2 + y * x2
                    cr = (abs(x) + abs(y)) * r2 + (abs(x2) + abs(y2)) * r + r * r2
                    text = f"({t} * {t2})"
                n, x, y, r = self.cball(text, f"near_{op} {n} {n2}", cx, cy, cr)
                t = text
            return t, n, x, y, r
        if e.is_Pow and e.args[1] == 1:
            return self.tree(e.args[0])
        if e.is_Pow and e.args[0] == _Z:
            return self.atom_ball(e)
        if isinstance(e, NC):
            return self.tree(sp.Integer(int(e.args[0])))
        return self.atom_ball(e)

    # the source

    def run(self, name: str) -> str:
        self.setup_env()
        self.choose_m()
        bnd = self.a.net.bnd
        if sorted(bnd) != sorted(self.b.net.bnd):
            raise NetUnsupported("the diagrams have different boundaries")
        sq0, sq1 = self.sq_list(self.a.net), self.sq_list(self.b.net)
        k0 = sp.Mul(*(Inv(Rt(d)) ** c for d, c in sq0))
        k1 = sp.Mul(*(Inv(Rt(d)) ** c for d, c in sq1))
        ranges = [range(self.labdim(x)) for x in bnd]
        best = None
        for count, values in enumerate(itertools.product(*ranges)):
            if count >= MAX_ASSIGNMENTS:
                break
            sigma = dict(zip(bnd, values, strict=True))
            g0, g1 = self.groups(self.a.net, sigma), self.groups(self.b.net, sigma)
            d = self.difference(k0, k1, g0, g1)
            v = self.num(d)
            size = max(abs(mpmath.re(v)), abs(mpmath.im(v)))
            if best is None or size > best[0]:
                best = (size, sigma, g0, g1, d)
            if size > mpmath.mpf(10) ** -6:
                break
        if best is None or best[0] < mpmath.mpf(10) ** -25:
            raise NetUnsupported("the networks agree numerically at every assignment checked")
        _, sigma, g0, g1, d = best
        text, n, x, y, r = self.tree(d)
        which = "Or.inl" if abs(x) > r else "Or.inr"
        if max(abs(x), abs(y)) <= r:
            raise NetUnsupported("the balls are too wide to separate the entry from zero")
        return self.source_num(name, sigma, sq0, sq1, g0, g1, text, n, which)

    def difference(self, k0: sp.Expr, k1: sp.Expr, g0: dict, g1: dict) -> sp.Expr:
        def gexpr(g: dict) -> sp.Expr:
            return sp.Add(*(self.mono(key) * sp.Add(*(c * _Z**i for i, c in enumerate(vec)
                                                       if c))
                            for key, vec in g.items()))
        return sp.Add(self.s0 * k0 * gexpr(g0), -(self.s1 * k1 * gexpr(g1)), evaluate=False)

    def source_num(self, name: str, sigma: Mapping[int, int], sq0: list[tuple[int, int]],
                   sq1: list[tuple[int, int]], g0: dict, g1: dict, dtext: str, dname: str,
                   which: str) -> str:
        out = self.out
        head = self.statement()
        params = [(raw, n, ty) for raw, (n, ty) in out.sym.params.items()
                  if raw != "k_lean_index" and raw not in out.mults]
        bare = [(str(e), n) for n, e, b in zip(out.mult_names, out.mult_exprs, out.mult_bare,
                                               strict=True) if b]
        args = [self.value_text(raw, ty) for raw, _, ty in params]
        args += [self.value_text(raw, "ℕ") for raw, _ in bare]
        args += ["(by norm_num)" for _, _, ty in params if ty == "ℕ"]
        env = self.instantiate(out.env)
        M = self.M
        gs = self.gs_text()
        pt = ", ".join(f"(({ph}, {j}), ({self.P(ph, j)[0]}, {_ints(list(self.P(ph, j)[1]))}))"
                       for ph, j in self.keys)
        keys = _list_pairs(self.keys)
        sig = ", ".join(f"({x}, {v})" for x, v in sorted(sigma.items()))
        unfold = "Fam.env, F, Φ" if out.family else "E, Φ"
        phok = ", ".join(self.atom_phase_proof(ph, j, unfold) for ph, j in self.keys)
        phok_args = phok + ", " if phok else ""
        nets = [(self.a.name, self.a.net, sq0, g0)]
        if self.b.name != self.a.name:
            nets.append((self.b.name, self.b.net, sq1, g1))
        evals, shows, pdecls = self.eval_rewrites(env, gs, keys, nets)
        decls = [*pdecls,
            (f"/-- Each phase entry as a power of `om {M}` times atoms. -/\n"
             f"def Pt (i k : ℕ) : ℕ × List ℤ := ((([{pt}] : List ((ℕ × ℕ) × (ℕ × List ℤ)))"
             f".lookup (i, k)).getD (0, []))"),
            (f"/-- The boundary assignment where the diagrams differ. -/\n"
             f"def σ0 (l : ℕ) : ℕ := ((([{sig}] : List (ℕ × ℕ)).lookup l).getD 0)"),
        ]
        body = [
            "  intro h",
            f"  have he := h {' '.join(args)}" if args else "  have he := h",
            "  refine refute σ0 (inRange_of _ _ _ (by decide +kernel)) ?_ he",
            "  intro h0",
            f"  have hP : PhOK {_paren(env)} {M} {gs} Pt {keys} := ⟨{phok_args}trivial⟩",
            f"  rw [{evals}] at h0",
            "  unfold sqK at h0",
            f"  rw [{shows}] at h0",
            ("  simp only [gval, mono, mono_nil, List.map_cons, List.map_nil, List.prod_cons,"
             " List.prod_nil, List.sum_cons, List.sum_nil, one_mul, mul_one, cvz, zpow_neg,"
             " zpow_one, zpow_ofNat, zpow_zero] at h0"),
            "  try push_cast at h0",
            f"  have hD : {dtext} = 0 := by linear_combination h0",
            *(f"  {line}" for line in self.lines),
            f"  exact ne_zero_of_near {dname} ({which} (by norm_num)) hD",
        ]
        theorem = ("set_option maxHeartbeats 0 in\n"
                   f"/-- The diagrams are not equal: they differ at {self.point_note()}. -/\n"
                   f"theorem {name} :\n    {head} := by\n" + "\n".join(body))
        out.decls += [*decls, theorem]
        return out.source(lead=f"  refutation by ball arithmetic at {self.point_note()}")


def ball_tree(M: int, out: Any, expr: sp.Expr, prefix: str) -> tuple[list[str], str, str,
                                                                       Fraction, Fraction,
                                                                       Fraction]:
    """Lean facts giving ``expr``, a polynomial in ζ = om M, a ball: the facts, its text,
    the name of its fact and the ball."""
    b = _NumRefuter.__new__(_NumRefuter)
    b.M, b.out, b.lines, b.count, b.prefix = M, out, [], 0, prefix
    b.xsym, b.xinv, b.opq = [], [], {}
    mpmath.mp.dps = 60
    text, name, x, y, r = b.tree(expr)
    return b.lines, text, name, x, y, r


__all__ = ["ball_tree", "export_numeric_refutation"]

_ = (Any, Sequence, _join)
