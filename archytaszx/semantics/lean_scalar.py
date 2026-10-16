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

"""Exact scalars written in Lean, and Lean proofs of identities between them.

A scalar is a sympy expression over the atoms ``Rt(x)`` (``√x`` for a natural ``x``),
``Om(m)`` (``e^{2πi/m}``), ``Inv(e)`` (``e⁻¹``), ``Ex(t)`` (``e^t``), the imaginary unit and
free symbols. An identity ``a = b`` is proved by ``linear_combination`` over the relations
``√x² = x``, ``e · e⁻¹ = 1``, ``ω_m^m = 1`` and ``i² = -1``, with coefficients from a
polynomial division; Lean's ``ring`` checks the result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker


class ScalarUnsupported(Exception):
    """A scalar or identity outside what this module writes or proves."""


Rt = sp.Function("Rt")
Om = sp.Function("Om")
Inv = sp.Function("Inv")
Ex = sp.Function("Ex")
NC = sp.Function("NC")
II = sp.Symbol("ii")


@dataclass
class Scalars:
    """Renders scalars over named symbols and proves identities between them.

    ``nat`` names a natural-number symbol's Lean binder, ``other`` a real or complex one with
    its Lean type; ``positive`` gives the Lean proof that a natural expression is positive.
    """

    nat: Callable[[sp.Symbol], str]
    other: Callable[[sp.Symbol], tuple[str, str]]
    positive: Callable[[str], str]
    nats: set[str] = field(default_factory=set)
    opaques: dict[str, tuple[sp.Symbol, sp.Expr]] = field(default_factory=dict)

    # conversion from engine scalars

    def model(self, expr: sp.Expr) -> sp.Expr:
        """``expr`` over this module's atoms, a subterm without one as an opaque symbol."""
        try:
            out = self._model(sp.sympify(expr))
            self.text(out)
            return out
        except ScalarUnsupported:
            return self.opaque(sp.sympify(expr))

    def opaque(self, expr: sp.Expr) -> sp.Symbol:
        """The opaque complex symbol standing for ``expr``."""
        key = sp.srepr(expr)
        if key not in self.opaques:
            self.opaques[key] = (sp.Symbol(f"opaque_{len(self.opaques) + 1}"), expr)
        return self.opaques[key][0]

    def _model(self, expr: sp.Expr) -> sp.Expr:
        if expr.is_Number or expr.is_Symbol:
            return expr
        if expr is sp.I:
            return II
        if expr is sp.pi:
            raise ScalarUnsupported("a bare π")
        if expr.is_Add:
            return sp.Add(*(self.model(a) for a in expr.args))
        if expr.is_Mul:
            return sp.Mul(*(self.model(a) for a in expr.args))
        if expr.is_Pow and expr.args[1].free_symbols:
            base, exp = expr.args
            const, rest = exp.as_coeff_Add()
            out = self.model(base**const) if const != 0 else sp.Integer(1)
            for term in sp.Add.make_args(rest):
                coeff, sym = term.as_coeff_Mul()
                if not sym.is_Symbol:
                    raise ScalarUnsupported(f"the exponent {exp}")
                out = out * sp.Pow(self.model(base**coeff), sym)
            return out
        if expr.is_Pow:
            base, exp = expr.args
            if exp.is_Integer:
                inner = self.model(base)
                return inner ** int(exp) if int(exp) >= 0 else Inv(inner) ** (-int(exp))
            if exp.is_Rational and exp.q == 2 and self._natural(base):
                root = Rt(base)
                return root ** exp.p if exp.p > 0 else Inv(root) ** (-exp.p)
            raise ScalarUnsupported(f"the power {expr}")
        if isinstance(expr, sp.exp):
            arg = sp.expand(expr.args[0] / (2 * sp.pi * sp.I))
            if arg.is_Rational:
                q = sp.Rational(arg) % 1
                return Om(q.q) ** q.p if q != 0 else sp.Integer(1)
            return Ex(expr.args[0])
        raise ScalarUnsupported(f"the scalar {expr} ({type(expr).__name__})")

    def _natural(self, base: sp.Expr) -> bool:
        if base.is_Integer:
            return int(base) > 0
        return bool(base.free_symbols) and all(str(s) in self.nats for s in base.free_symbols) and (
            base.is_Mul or base.is_Pow or base.is_Symbol or base.is_Add
        )

    # rendering

    def nat_text(self, expr: sp.Expr) -> str:
        """A natural-number expression as a Lean ℕ term."""
        expr = sp.sympify(expr)
        if expr.is_Integer:
            return str(int(expr))
        if expr.is_Symbol:
            return self.nat(expr)
        if expr.is_Add:
            return "(" + " + ".join(self.nat_text(a) for a in expr.args) + ")"
        if expr.is_Mul:
            return "(" + " * ".join(self.nat_text(a) for a in expr.args) + ")"
        if expr.is_Pow and expr.args[1].is_Integer and int(expr.args[1]) >= 0:
            return f"({self.nat_text(expr.args[0])} ^ {int(expr.args[1])})"
        if expr.is_Pow and not expr.args[1].is_number:
            return f"({self.nat_text(expr.args[0])} ^ {self.nat_text(expr.args[1])})"
        raise ScalarUnsupported(f"{expr} is not a natural number")

    def text(self, expr: sp.Expr) -> str:
        """``expr`` as a Lean ℂ term."""
        expr = sp.sympify(expr)
        if expr == 1:
            return "1"
        if expr.is_Integer:
            v = int(expr)
            return f"({v} : ℂ)" if v >= 0 else f"(-{-v} : ℂ)"
        if expr.is_Rational:
            return f"(({expr.p} : ℂ) / {expr.q})"
        if expr == II:
            return "Complex.I"
        if expr.is_Symbol:
            if str(expr) in self.nats:
                return f"({self.nat(expr)} : ℂ)"
            name, ty = self.other(expr)
            return name if ty == "ℂ" else f"({name} : ℂ)"
        if expr.is_Add:
            return "(" + " + ".join(self.text(a) for a in expr.args) + ")"
        if expr.is_Mul:
            return "(" + " * ".join(self.text(a) for a in expr.args) + ")"
        if expr.is_Pow and expr.args[1].is_Integer and int(expr.args[1]) >= 0:
            return f"({self.text(expr.args[0])} ^ {int(expr.args[1])})"
        if expr.is_Pow and expr.args[1].free_symbols:
            return f"({self.text(expr.args[0])} ^ {self.nat_text(expr.args[1])})"
        if isinstance(expr, Rt):
            return f"(rt {self.nat_text(expr.args[0])})"
        if isinstance(expr, Om):
            return f"(om {self.nat_text(expr.args[0])})"
        if isinstance(expr, NC):
            return f"(({self.nat_text(expr.args[0])} : ℕ) : ℂ)"
        if isinstance(expr, Inv):
            return f"({self.text(expr.args[0])})⁻¹"
        if isinstance(expr, Ex):
            return f"(Complex.exp {self.complex_text(expr.args[0])})"
        raise ScalarUnsupported(f"cannot write {expr} in Lean")

    def complex_text(self, expr: sp.Expr) -> str:
        """A plain sympy expression (π, I, rationals, symbols) as a Lean ℂ term."""
        expr = sp.sympify(expr)
        if expr is sp.pi:
            return "(Real.pi : ℂ)"
        if expr is sp.I:
            return "Complex.I"
        if expr.is_Number or expr.is_Symbol or expr.is_Add or expr.is_Mul or expr.is_Pow:
            if expr.is_Add:
                return "(" + " + ".join(self.complex_text(a) for a in expr.args) + ")"
            if expr.is_Mul:
                return "(" + " * ".join(self.complex_text(a) for a in expr.args) + ")"
            if expr.is_Pow and expr.args[1].is_Integer:
                p = int(expr.args[1])
                base = self.complex_text(expr.args[0])
                return f"({base} ^ {p})" if p >= 0 else f"({base} ^ {-p})⁻¹"
            return self.text(expr)
        raise ScalarUnsupported(f"cannot write {expr} in Lean")

    # proofs

    def nonzero(self, atom: sp.Expr) -> str:
        """A Lean proof that ``atom ≠ 0``."""
        if isinstance(atom, Rt):
            x = self.nat_text(atom.args[0])
            return f"(rt_ne {x} {self.positive(x)})"
        if isinstance(atom, Om):
            x = self.nat_text(atom.args[0])
            return f"(om_ne_zero {x} {self.positive(x)})"
        if isinstance(atom, Ex):
            return "(Complex.exp_ne_zero _)"
        if isinstance(atom, NC):
            x = self.nat_text(atom.args[0])
            return f"(Nat.cast_ne_zero.2 (Nat.pos_iff_ne_zero.1 {self.positive(x)}) : (({x} : ℕ) : ℂ) ≠ 0)"
        if atom.is_Symbol and str(atom) in self.nats:
            x = self.nat(atom)
            return f"(Nat.cast_ne_zero.2 (Nat.pos_iff_ne_zero.1 {self.positive(x)}) : ({x} : ℂ) ≠ 0)"
        if atom.is_Integer:
            return f"(by norm_num : ({int(atom)} : ℂ) ≠ 0)"
        return f"(by norm_num [Complex.ext_iff] : {self.text(atom)} ≠ 0)"

    def nonzero_expr(self, expr: sp.Expr) -> str:
        """A Lean proof that ``expr`` as written by :meth:`text` is nonzero."""
        expr = sp.sympify(expr)
        if expr == 1:
            return "one_ne_zero"
        if expr.is_Rational and not expr.is_Integer:
            return f"(by norm_num : {self.text(expr)} ≠ 0)"
        if expr.is_Mul:
            args = list(expr.args)
            proof = self.nonzero_expr(args[0])
            for a in args[1:]:
                proof = f"(mul_ne_zero {proof} {self.nonzero_expr(a)})"
            return proof
        if expr.is_Pow and expr.args[1].is_Integer and int(expr.args[1]) >= 0:
            return f"(pow_ne_zero {int(expr.args[1])} {self.nonzero_expr(expr.args[0])})"
        if isinstance(expr, Inv):
            return f"(inv_ne_zero {self.nonzero_expr(expr.args[0])})"
        if expr == II:
            return "Complex.I_ne_zero"
        return self.nonzero(expr)

    def prove(self, lhs: sp.Expr, rhs: sp.Expr) -> str:
        """A Lean tactic proving ``lhs = rhs`` written by :meth:`text`."""
        relations: list[tuple[sp.Expr, str]] = []
        atoms: dict[sp.Expr, sp.Symbol] = {}

        def sym(atom: sp.Expr) -> sp.Symbol:
            if atom not in atoms:
                atoms[atom] = sp.Symbol(f"a{len(atoms)}")
            return atoms[atom]

        def poly(expr: sp.Expr) -> sp.Expr:
            if expr.is_Number:
                return expr
            if expr == II:
                s = sym(II)
                return s
            if expr.is_Symbol:
                return sym(expr)
            if expr.is_Add:
                return sp.Add(*(poly(a) for a in expr.args))
            if expr.is_Mul:
                return sp.Mul(*(poly(a) for a in expr.args))
            if expr.is_Pow and expr.args[1].is_Integer and int(expr.args[1]) >= 0:
                return poly(expr.args[0]) ** int(expr.args[1])
            if expr.is_Pow and expr.args[1].free_symbols:
                base, exp = expr.args
                const, rest = sp.expand(exp).as_coeff_Add()
                if not const.is_Integer or int(const) < 0:
                    raise ScalarUnsupported(f"the exponent {exp}")
                return sym(sp.Pow(base, rest, evaluate=False)) * poly(base) ** int(const)
            if isinstance(expr, NC):
                return poly(expr.args[0])
            if isinstance(expr, (Rt, Om, Ex, Inv)):
                return sym(expr)
            raise ScalarUnsupported(f"no polynomial model for {expr}")

        diff = poly(lhs) - poly(rhs)
        seen: set[sp.Expr] = set()
        queue = list(atoms)
        while queue:
            atom = queue.pop()
            if atom in seen:
                continue
            seen.add(atom)
            s = atoms[atom]
            if atom == II:
                relations.append((s**2 + 1, "Complex.I_sq"))
            elif isinstance(atom, Rt):
                x = atom.args[0]
                before = set(atoms)
                relations.append((s**2 - poly(x), f"(rt_sq' {self.nat_text(x)})"))
                queue += [a for a in atoms if a not in before]
            elif isinstance(atom, Om):
                m = atom.args[0]
                if not m.is_Integer:
                    raise ScalarUnsupported("a root of unity of symbolic order")
                order = int(m)
                relations.append((s**order - 1, f"(om_pow_self {order} (by norm_num))"))
                for prime in sp.primefactors(order):
                    step = order // prime
                    rel = sp.Add(*(s ** (j * step) for j in range(prime)))
                    text = " + ".join(
                        "(1 : ℂ)" if j == 0 else f"om {order} ^ {j * step}" for j in range(prime)
                    )
                    proof = (
                        f"({text} = 0 := (by simpa [Finset.sum_range_succ] using "
                        f"om_geom {order} {prime} (by norm_num) (by norm_num) (by norm_num)))"
                    )
                    relations.append((rel, proof))
            elif isinstance(atom, Inv):
                before = set(atoms)
                inner = poly(atom.args[0])
                inner_atom = atom.args[0]
                proof = self.nonzero(inner_atom)
                relations.append((inner * s - 1, f"(mul_inv_cancel₀ {proof})"))
                queue += [a for a in atoms if a not in before]
        for atom in list(atoms):
            if atom.is_Pow and atom.args[1].free_symbols and isinstance(atom.args[0], Inv):
                inner = atom.args[0].args[0]
                partner = sp.Pow(inner, atom.args[1], evaluate=False)
                if partner in atoms:
                    e = self.nat_text(atom.args[1])
                    stmt = f"{self.text(inner)} ^ {e} * ({self.text(inner)})⁻¹ ^ {e} = 1"
                    proof = (f"({stmt} := (by rw [← mul_pow, mul_inv_cancel₀ "
                             f"{self.nonzero(inner)}, one_pow]))")
                    relations.append((atoms[partner] * atoms[atom] - 1, proof))
        gens = list(atoms.values())
        if not gens or not relations:
            if sp.expand(diff) == 0:
                return "by ring"
            raise ScalarUnsupported(f"{lhs} = {rhs} does not hold")
        quotients = _lift(sp.expand(diff), [r for r, _ in relations], gens)
        if quotients is None:
            raise ScalarUnsupported(f"{lhs} = {rhs} is not implied by the relations")
        back = {s: a for a, s in atoms.items()}
        haves = []
        terms = []
        for i, (q, (_, proof)) in enumerate(zip(quotients, relations, strict=True)):
            q = sp.expand(q)
            if q == 0:
                continue
            if proof.startswith("(") and ":= (by" in proof:
                haves.append(f"have r{i} : {proof[1:-1]}")
            else:
                haves.append(f"have r{i} := {proof}")
            terms.append(f"({self.text(q.xreplace(back))}) * r{i}")
        if not terms:
            return "by ring"
        return "by " + "; ".join([*haves, "linear_combination " + " + ".join(terms)])


def _lift(target: sp.Expr, rels: list[sp.Expr], gens: list[sp.Symbol]) -> list[sp.Expr] | None:
    """Cofactors ``c`` with ``target = Σ c_i rels_i``, or None if ``target`` is not in the
    ideal; Buchberger's algorithm with every basis element's cofactors tracked."""
    order = "grevlex"

    def P(e: sp.Expr) -> sp.Poly:
        return sp.Poly(e, *gens, domain="QQ")

    n = len(rels)
    zero = P(0)
    basis: list[tuple[sp.Poly, list[sp.Poly]]] = []
    for i, r in enumerate(rels):
        cof = [zero] * n
        cof[i] = P(1)
        basis.append((P(r), cof))

    def lead(p: sp.Poly) -> tuple[tuple[int, ...], sp.Expr]:
        return p.LM(order=order).exponents, p.LC(order=order)

    def divides(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        return all(x <= y for x, y in zip(a, b, strict=True))

    def mono(exps: tuple[int, ...], c: sp.Expr) -> sp.Poly:
        return sp.Poly.from_dict({exps: c}, *gens, domain="QQ")

    def reduce(f: sp.Poly, cof: list[sp.Poly]) -> tuple[sp.Poly, list[sp.Poly]]:
        rem = zero
        while not f.is_zero:
            m, c = lead(f)
            for g, gc in basis:
                if g.is_zero:
                    continue
                gm, gcoef = lead(g)
                if divides(gm, m):
                    t = mono(tuple(x - y for x, y in zip(m, gm, strict=True)), c / gcoef)
                    f = f - t * g
                    cof = [ci - t * gi for ci, gi in zip(cof, gc, strict=True)]
                    break
            else:
                term = mono(m, c)
                rem = rem + term
                f = f - term
        return rem, cof

    pairs = [(i, j) for i in range(len(basis)) for j in range(i)]
    steps = 0
    while pairs and steps < 400:
        steps += 1
        i, j = pairs.pop()
        (f, fc), (g, gc) = basis[i], basis[j]
        fm, fcoef = lead(f)
        gm, gcoef = lead(g)
        lcm = tuple(max(x, y) for x, y in zip(fm, gm, strict=True))
        if all(x + y == z for x, y, z in zip(fm, gm, lcm, strict=True)):
            continue
        tf = mono(tuple(z - x for z, x in zip(lcm, fm, strict=True)), 1 / fcoef)
        tg = mono(tuple(z - y for z, y in zip(lcm, gm, strict=True)), 1 / gcoef)
        spoly = tf * f - tg * g
        scof = [tf * a - tg * b for a, b in zip(fc, gc, strict=True)]
        r, rc = reduce(spoly, scof)
        if not r.is_zero:
            basis.append((r, rc))
            pairs += [(len(basis) - 1, k) for k in range(len(basis) - 1)]
    rem, cof = reduce(P(target), [zero] * n)
    if not rem.is_zero:
        return None
    return [(-c).as_expr() for c in cof]
