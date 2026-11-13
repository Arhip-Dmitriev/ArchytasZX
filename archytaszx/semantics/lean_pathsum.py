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

"""Lean output for a path-sum EQUAL. export_pair writes each side's reduction trace as one
rule-lemma instance per step (ArchytasZX/PathSum.lean) and the canonical forms' equality by rfl.
export_proof writes the theorem that both sides' path sums have one value at every boundary
assignment (ArchytasZX/PathSumRules.lean): each side's trace replayed as a step list checked by
``decide``, the canonical pivots as canon steps, and the constants' identity by
``linear_combination``. export_net proves ``Equal`` of the two diagrams' networks: each network's
value as its fragments' path sum (ArchytasZX/PathSumNet.lean), pin steps onto from_diagram's
variables, then export_proof's steps. None enumerates a sum.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.pathsum import (
    Canon,
    Cyclo,
    ModPoly,
    PathSum,
    PathSumUnsupported,
    PhasePoly,
    QuditCanon,
    QuditPathSum,
    Step,
    canonical,
    canonical_qudit,
    reduce,
    reduce_qudit,
)
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.pathsum_diagram import from_diagram

_LEMMA = {
    "elim": "ps_elim",
    "hh_pivot": "ps_hh_pin",
    "hh_gen": "ps_hh_gen",
    "rem": "ps_rem",
    "hh_constraint": "ps_hh",
    "hh_nl": "ps_hh_nl",
    "q_elim": "q_elim",
    "q_hh_pivot": "q_hh",
    "q_hh_constraint": "q_hh",
    "q_gauss": "q_gauss",
}
"""The PathSum.lean lemma each rule's step instantiates."""


def _rat(v: Fraction) -> str:
    """A rational as a Lean term."""
    return f"({v.numerator} / {v.denominator} : ℚ)" if v.denominator != 1 else f"({v} : ℚ)"


def _f2(poly: frozenset[frozenset[int]] | None) -> str:
    """An F_2 polynomial as a list of monomials."""
    if poly is None:
        return "[]"
    return "[" + ", ".join(str(sorted(m)) for m in sorted(poly, key=lambda m: sorted(m))) + "]"


def _step_line(tag: str, k: int, step: Step, p: int | None) -> list[str]:
    """One step: its data as a comment and the instance of its rule lemma."""
    data = f"var {step.var}"
    if step.pivot is not None:
        data += f", pivot {step.pivot}"
    if step.value is not None:
        data += f", value {_f2(step.value)}"
    if step.group:
        data += f", digits {list(step.group)}, fresh {list(step.fresh)}"
        data += ", bits [" + ", ".join(_f2(b) for b in step.values) + "]"
    if step.poly:
        data += f", poly {[(list(m), c) for m, c in step.poly]}"
    if step.rule in ("omega", "omega_conj"):
        pair = (
            ("ps_omega0", "ps_omega1")
            if step.rule == "omega"
            else (
                "ps_omega_conj0",
                "ps_omega_conj1",
            )
        )
        body = f"example := And.intro {pair[0]} {pair[1]}"
    elif step.rule == "dyadic":
        body = f"example (S : ℕ) := ps_dyadic {len(step.group)} S"
    elif step.rule.startswith("q_") and p is not None:
        args = f" {p}" + (f" {step.pivot}" if step.rule == "q_gauss" else "")
        body = f"example := @{_LEMMA[step.rule]}{args}"
    else:
        body = f"example := @{_LEMMA[step.rule]}"
    return [f"-- {tag} step {k}: [{step.rule}] {data}", body]


def _const(c: Cyclo, n: int) -> str:
    """A constant's reduced coefficients over zeta_n."""
    return "[" + ", ".join(_rat(v) for v in c.reduced(n)) + "]"


def _canon_term(canon: Canon, n: int) -> str:
    """A qubit canonical form as a PSCanon literal."""
    rows = ", ".join(f"({p}, {[list(m) for m in rest]})" for p, rest in canon.rows)
    phase = ", ".join(f"({list(m)}, {_rat(c)})" for m, c in canon.phase)
    return (
        f"{{ rows := [{rows}], phase := [{phase}], const := {_const(canon.const, n)}, n := {n} }}"
    )


def _qcanon_term(canon: QuditCanon, n: int) -> str:
    """A qudit canonical form as a QCanon literal."""

    def terms(ts: Sequence[tuple[tuple[tuple[int, int], ...], int]]) -> str:
        return "[" + ", ".join(f"({[tuple(t) for t in m]}, {c})" for m, c in ts) + "]"

    rows = ", ".join(f"({q}, {terms(rest)})" for q, rest in canon.rows)
    return (
        f"{{ rows := [{rows}], phase := {terms(canon.phase)}, "
        f"const := {_const(canon.const, n)}, n := {n}, p := {canon.p} }}"
    )


def export_pair(left: Diagram, right: Diagram, name: str = "pair") -> str:
    """The Lean file text for a pair whose path sums reduce to one canonical form; raises
    :class:`PathSumUnsupported` otherwise."""
    raw_l, raw_r = from_diagram(left), from_diagram(right)
    lines = [
        "import ArchytasZX.PathSum",
        "",
        "/-!",
        f"# Path-sum certificate `{name}`",
        "",
        "Left trace, canonical-form equality, right trace (reversed for R = canonical form).",
        "-/",
        "",
        f"namespace ArchytasZX.PathSumCert.{name}",
        "",
    ]
    if isinstance(raw_l, QuditPathSum) and isinstance(raw_r, QuditPathSum):
        ql, qr = reduce_qudit(raw_l), reduce_qudit(raw_r)
        qcl, qcr = canonical_qudit(ql), canonical_qudit(qr)
        if qcl is None or qcr is None or not qcl.same(qcr) or qcl.zero:
            raise PathSumUnsupported("the qudit path sums do not reduce to one canonical form")
        n = math.lcm(qcl.const.n, qcr.const.n)
        canon_l, canon_r, struct = _qcanon_term(qcl, n), _qcanon_term(qcr, n), "QCanon"
        traces, p = (ql.trace, qr.trace), raw_l.p
    elif isinstance(raw_l, PathSum) and isinstance(raw_r, PathSum):
        bl, br = reduce(raw_l), reduce(raw_r)
        cl, cr = canonical(bl), canonical(br)
        if cl is None or cr is None or not cl.same(cr) or cl.zero:
            raise PathSumUnsupported("the path sums do not reduce to one canonical form")
        n = math.lcm(cl.const.n, cr.const.n)
        canon_l, canon_r, struct = _canon_term(cl, n), _canon_term(cr, n), "PSCanon"
        traces, p = (bl.trace, br.trace), None
    else:
        raise PathSumUnsupported("the two path sums have different bases")
    for k, step in enumerate(traces[0]):
        lines += _step_line("L", k, step, p)
    lines += [
        "",
        f"def canonL : {struct} := {canon_l}",
        f"def canonR : {struct} := {canon_r}",
        "",
        "theorem canon_eq : canonL = canonR := rfl",
        "",
    ]
    for k, step in enumerate(traces[1]):
        lines += _step_line("R", k, step, p)
    lines += ["", f"end ArchytasZX.PathSumCert.{name}", ""]
    return "\n".join(lines)


def _elimination_order(
    constraints: Sequence[frozenset[frozenset[int]]], boundary: Sequence[int]
) -> list[int]:
    """The pivots of pathsum.eliminate in the order it takes them."""
    from archytaszx.algebra.pathsum import f2_substitute

    work = [c for c in constraints if c]
    order: list[int] = []
    progress = True
    while work and progress:
        progress = False
        for v in sorted(set(boundary) - set(order)):
            single = frozenset({v})
            hit = next(
                (c for c in work if single in c and not any(v in m for m in c if m != single)),
                None,
            )
            if hit is None:
                continue
            work.remove(hit)
            rest = frozenset(hit - {single})
            work = [f2_substitute(c, v, rest) for c in work]
            work = [c for c in work if c]
            order.append(v)
            progress = True
    return order


def _data_term(ps: PathSum, m: int) -> str:
    """A path sum as a PSem.Data literal, coefficients k of k / m turns."""
    phase = ", ".join(
        f"({sorted(mono)}, {int(c * m) % m})"
        for mono, c in sorted(ps.phase.terms.items(), key=lambda t: (len(t[0]), sorted(t[0])))
    )
    cons = ", ".join(
        "[" + ", ".join(str(sorted(mono)) for mono in sorted(q, key=sorted)) + "]"
        for q in ps.constraints
    )
    return f"{{ path := {sorted(ps.path)}, phase := [{phase}], cons := [{cons}], rows := [] }}"


_HEAD_STEP = {
    "elim": ".elim",
    "hh_constraint": ".hhCons",
    "omega": ".omega",
    "omega_conj": ".omegaConj",
}
"""The PathSumRules step summing out a head variable, per rule."""

_PAIR_STEP = {"hh_pivot": ".hhPivot", "hh_gen": ".hhGen", "hh_nl": ".hhNl"}
"""The PathSumRules step reading the head two variables, per rule."""


def _steps(raw: PathSum, ps: PathSum, order: Sequence[int], boolean: bool) -> list[str]:
    """The PSem.Step list replaying the trace of ``ps``, reduced from ``raw``, then the canon
    steps at ``order`` (gather and bcanon steps when ``boolean``)."""
    cur = sorted(raw.path)
    out: list[str] = []
    for step in ps.trace:
        y, z = step.var, step.pivot
        if step.rule in _HEAD_STEP:
            cur.remove(y)
            out += [f".perm {[y, *cur]}", _HEAD_STEP[step.rule]]
        elif step.rule == "rem":
            cur = [y, *(v for v in cur if v != y)]
            out += [f".perm {cur}", ".rem"]
        elif step.rule == "dyadic":
            rest = [v for v in cur if v not in step.group]
            poly = ", ".join(f"({[v for v, _e in m]}, {c})" for m, c in step.poly)
            out += [
                f".perm {[*step.group, *rest]}",
                f".dyadic {list(step.group)} [{poly}] {list(step.fresh)}",
            ]
            cur = [*step.fresh, *rest]
        elif step.rule in _PAIR_STEP and z is not None:
            rest = [v for v in cur if v not in (y, z)]
            out += [f".perm {[y, z, *rest]}", _PAIR_STEP[step.rule]]
            cur = rest if step.rule == "hh_pivot" else [y, *rest]
        else:
            raise PathSumUnsupported(f"the rule {step.rule} has no PathSumRules step")
    if boolean:
        return [*out, ".gather", *(f".bcanon {v}" for v in order)]
    return out + [f".canon {v}" for v in order]


def _const_model(c: Cyclo, k: int) -> sp.Expr:
    """``c * sqrt(2)^k`` over the lean_scalar atoms Om and Rt."""
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

    value = sum(
        (sp.Rational(v.numerator, v.denominator) * Om(c.n) ** e for e, v in sorted(c.c.items())),
        sp.Integer(0),
    )
    root = Rt(2) ** k if k >= 0 else Inv(Rt(2)) ** (-k)
    return sp.expand(value * root)


def _fac_model(steps: Sequence[str], m: int) -> sp.Expr:
    """The product of the steps' PathSumRules factors, zeta_8 written as ``Om(m)^(m / 8)``."""
    from archytaszx.semantics.lean_scalar import Om, Rt

    out = sp.Integer(1)
    for st in steps:
        if st in (".elim", ".hhCons", ".hhPivot", ".hhNl"):
            out *= 2
        elif st == ".omega":
            out *= Rt(2) * Om(m) ** (m // 8)
        elif st == ".omegaConj":
            out *= Rt(2) * Om(m) ** (7 * m // 8)
    return out


def _fac_lines(side: str, steps: Sequence[str], m: int, text: str) -> list[str]:
    """The theorem that side ``side``'s step factors multiply to ``text``."""
    lines = [f"theorem fac{side} : facs steps{side} = {text} := by"]
    lines.append(f"  simp only [steps{side}, facs, fac]")
    if m != 8 and any(st in (".omega", ".omegaConj") for st in steps):
        lines.append(f"  rw [om_sub {m} 8 {m // 8} (by norm_num) (by norm_num)]")
    return [*lines, "  ring", ""]


def _const_turn(ps: PathSum, m: int, boolean: bool) -> int:
    """The constant phase term of the canonical form of a reduced sum, in units of 1 / m."""
    from archytaszx.algebra.pathsum import boolean_eliminate, eliminate

    rows = (
        boolean_eliminate(ps.constraints, ps.boundary)[0]
        if boolean
        else eliminate(ps.constraints, ps.boundary)
    )
    phase = ps.phase.copy()
    for pivot, rest in cast(list[tuple[int, frozenset[frozenset[int]]]], rows or []):
        phase.substitute(pivot, rest)
    return int(phase.terms.get(frozenset(), Fraction(0)) * m) % m


def _plan(
    raw_l: PathSum, raw_r: PathSum
) -> tuple[PathSum, PathSum, list[int], bool, int, list[str], list[str]]:
    """Both reduced sums, the canonical pivots, whether Boolean elimination is needed, the
    phase modulus M and each side's PSem.Step list; raises :class:`PathSumUnsupported` when the
    sums do not reduce to one canonical form."""
    if raw_l.amps or raw_r.amps:
        raise PathSumUnsupported("export_proof has no amplitude atoms")
    red_l, red_r = reduce(raw_l), reduce(raw_r)
    can_l, can_r = canonical(red_l), canonical(red_r)
    if can_l is None or can_r is None or not can_l.same(can_r) or can_l.zero:
        raise PathSumUnsupported("the path sums do not reduce to one canonical form")
    m = 4
    for ps in (raw_l, raw_r):
        for c in ps.phase.terms.values():
            m = math.lcm(m, c.denominator)
    if any(st.rule in ("omega", "omega_conj") for st in (*red_l.trace, *red_r.trace)):
        m = math.lcm(m, 8)
    from archytaszx.algebra.pathsum import boolean_eliminate, eliminate

    boolean = any(eliminate(red.constraints, red.boundary) is None for red in (red_l, red_r))
    if boolean:
        rows_l, g_l = boolean_eliminate(red_l.constraints, red_l.boundary)
        rows_r, g_r = boolean_eliminate(red_r.constraints, red_r.boundary)
        order = [v for v, _r in rows_l]
        one: frozenset[frozenset[int]] = frozenset({frozenset()})
        if g_l != one or g_r != one or order != [v for v, _r in rows_r]:
            raise PathSumUnsupported("the Boolean eliminations differ or leave a support")
    else:
        order = _elimination_order(red_l.constraints, red_l.boundary)
    steps_l = _steps(raw_l, red_l, order, boolean)
    steps_r = _steps(raw_r, red_r, order, boolean)
    return red_l, red_r, order, boolean, m, steps_l, steps_r


def _scalars() -> Any:
    """The lean_scalar renderer of path-sum constants."""
    from archytaszx.semantics.lean_scalar import Scalars

    return Scalars(
        nat=lambda sym: str(sym),
        other=lambda sym: (str(sym), "ℂ"),
        positive=lambda x: "(by norm_num)",
    )


def export_proof(left: Diagram, right: Diagram, name: str = "pair") -> str:
    """The Lean file proving that both diagrams' path sums agree at every boundary assignment;
    raises :class:`PathSumUnsupported` when they do not reduce to one canonical form or a
    trace step has no PathSumRules step."""
    from archytaszx.semantics.lean_scalar import Om

    raw_l, raw_r = from_diagram(left), from_diagram(right)
    if not isinstance(raw_l, PathSum) or not isinstance(raw_r, PathSum):
        raise PathSumUnsupported("export_proof covers qubit path sums only")
    red_l, red_r, _order, boolean, m, steps_l, steps_r = _plan(raw_l, raw_r)
    f_l, f_r = _fac_model(steps_l, m), _fac_model(steps_r, m)
    k_l, k_r = _const_turn(red_l, m, boolean), _const_turn(red_r, m, boolean)
    sc = _scalars()
    c_l, c_r = _const_model(raw_l.const, raw_l.sqrt2), _const_model(raw_r.const, raw_r.sqrt2)
    lhs = f_l * c_l * Om(m) ** k_l
    rhs = f_r * c_r * Om(m) ** k_r
    proof = sc.prove(lhs, rhs)
    ns = f"ArchytasZX.PathSumProof.{name}"
    lines = [
        "import ArchytasZX.PathSumRules",
        "",
        "/-!",
        f"# Path-sum proof `{name}`",
        "",
        "Both path sums, each side's steps replayed and checked by `decide`, the constants'",
        "identity, and the value equality at every boundary assignment.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        f"def sL : Data := {_data_term(raw_l, m)}",
        f"def sR : Data := {_data_term(raw_r, m)}",
        f"def stepsL : List Step := [{', '.join(steps_l)}]",
        f"def stepsR : List Step := [{', '.join(steps_r)}]",
        "",
        *_fac_lines("L", steps_l, m, sc.text(f_l)),
        *_fac_lines("R", steps_r, m, sc.text(f_r)),
        f"theorem consts : ({sc.text(f_l)}) * {sc.text(c_l)} * om {m} ^ {k_l} =",
        f"    ({sc.text(f_r)}) * {sc.text(c_r)} * om {m} ^ {k_r} := by",
        f"  have key : {sc.text(lhs)} = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        "  linear_combination key",
        "",
        "theorem main (σ : ℕ → ZMod 2) :",
        f"    PSem.val {m} {sc.text(c_l)} sL σ = PSem.val {m} {sc.text(c_r)} sR σ := by",
        f"  rw [run_sound {m} (by norm_num) stepsL sL _ σ (by decide),",
        f"    run_sound {m} (by norm_num) stepsR sR _ σ (by decide)]",
        f"  apply final_sound {m} (by norm_num) _ _ _ _ σ {k_l} {k_r} (by decide) (by decide)",
        "    (by decide)",
        "  simpa only [facL, facR] using consts",
        "",
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


_Q_HEAD = {"q_elim": ".elim", "q_hh_constraint": ".hhCons"}
"""The PathSumQudit step summing out a head variable, per rule."""


def _qdata_term(ps: QuditPathSum) -> str:
    """A qudit path sum as a QSem.Data literal."""

    def poly(q: ModPoly) -> str:
        return "[" + ", ".join(f"({[tuple(t) for t in m]}, {c})" for m, c in q.key()) + "]"

    cons = ", ".join(poly(q) for q in ps.constraints)
    return (
        f"{{ path := {sorted(ps.path)}, phase := {poly(ps.phase)}, cons := [{cons}], rows := [] }}"
    )


def _qudit_order(constraints: Sequence[ModPoly], boundary: Sequence[int]) -> list[int]:
    """The pivots of pathsum.eliminate_qudit in the order it takes them."""
    from archytaszx.algebra.pathsum import _q_linear_pivot

    work = [c for c in constraints if c.terms]
    order: list[int] = []
    progress = True
    while work and progress:
        progress = False
        for v in sorted(set(boundary) - set(order)):
            hit = next(
                ((c, f[1]) for c in work if (f := _q_linear_pivot(c, {v}, -1)) is not None),
                None,
            )
            if hit is None:
                continue
            c, a = hit
            work.remove(c)
            value = (c + ModPoly(c.p, {((v, 1),): -a % c.p})).scale(-pow(a, -1, c.p))
            work = [w.substitute(v, value) for w in work]
            work = [w for w in work if w.terms]
            order.append(v)
            progress = True
    return order


def _qudit_steps(raw: QuditPathSum, red: QuditPathSum, order: Sequence[int]) -> list[str]:
    """The QSem.Step list replaying the trace of ``red``, reduced from ``raw``, then the canon
    steps at ``order`` and a final normalisation."""
    cur = sorted(raw.path)
    out: list[str] = []
    for step in red.trace:
        y, z = step.var, step.pivot
        if step.rule in _Q_HEAD or step.rule == "q_gauss":
            cur.remove(y)
            out.append(f".perm {[y, *cur]}")
            out.append(f".gauss {z}" if step.rule == "q_gauss" else _Q_HEAD[step.rule])
        elif step.rule == "q_hh_pivot" and z is not None:
            cur = [v for v in cur if v not in (y, z)]
            out += [f".perm {[y, z, *cur]}", ".hhPivot"]
        else:
            raise PathSumUnsupported(f"the rule {step.rule} has no PathSumQudit step")
    return [*out, *(f".canon {v}" for v in order), ".norm"]


def _qudit_const_turn(red: QuditPathSum, order: Sequence[int]) -> int:
    """The constant phase term of the canonical form of a reduced qudit sum."""
    from archytaszx.algebra.pathsum import eliminate_qudit

    rows = cast(list[tuple[int, ModPoly]], eliminate_qudit(red.constraints, red.boundary) or [])
    if [v for v, _r in rows] != sorted(order):
        raise PathSumUnsupported("the qudit elimination pivots differ")
    phase = red.phase.copy()
    for pivot, rest in rows:
        phase = phase.substitute(pivot, rest)
    return phase.terms.get((), 0) % red.p


def _cyclo_model(c: Cyclo, n: int) -> sp.Expr:
    """``c`` over ``Om(n)``, n a multiple of c's order."""
    from archytaszx.semantics.lean_scalar import Om

    return sum(
        (
            sp.Rational(v.numerator, v.denominator) * Om(n) ** (e * n // c.n)
            for e, v in sorted(c.c.items())
        ),
        sp.Integer(0),
    )


def _legendre_model(a: int, p: int, n: int) -> sp.Expr:
    """``G(a)`` in its Legendre form over ``Om(n)``."""
    from archytaszx.algebra.pathsum import legendre
    from archytaszx.semantics.lean_scalar import Om

    return legendre(a, p) * sum(
        (legendre(k, p) * Om(n) ** (k * n // p) for k in range(1, p)), sp.Integer(0)
    )


def export_qudit_proof(left: Diagram, right: Diagram, name: str = "pair") -> str:
    """The Lean file proving that two diagrams' qudit path sums over Z_p, p an odd prime, agree
    at every boundary assignment (ArchytasZX/PathSumQudit.lean): each side's trace replayed as
    a step list checked by ``decide``, [Gauss] factors in their Legendre form, the canonical
    pivots as canon steps and the constants' identity by ``linear_combination``; raises
    :class:`PathSumUnsupported` when they do not reduce to one canonical form or a trace step
    has no PathSumQudit step."""
    from archytaszx.algebra.pathsum import legendre
    from archytaszx.semantics.lean_scalar import Om

    raw_l, raw_r = from_diagram(left), from_diagram(right)
    if not isinstance(raw_l, QuditPathSum) or not isinstance(raw_r, QuditPathSum):
        raise PathSumUnsupported("export_qudit_proof covers odd prime path sums only")
    if raw_l.p != raw_r.p or raw_l.boundary != raw_r.boundary:
        raise PathSumUnsupported("the two path sums have different bases or boundaries")
    if any(ps.atoms or ps.katoms for ps in (raw_l, raw_r)):
        raise PathSumUnsupported("export_qudit_proof has no atoms")
    p = raw_l.p
    red_l, red_r = reduce_qudit(raw_l), reduce_qudit(raw_r)
    can_l, can_r = canonical_qudit(red_l), canonical_qudit(red_r)
    if can_l is None or can_r is None or not can_l.same(can_r) or can_l.zero:
        raise PathSumUnsupported("the qudit path sums do not reduce to one canonical form")
    order = _qudit_order(red_l.constraints, red_l.boundary)
    if order != _qudit_order(red_r.constraints, red_r.boundary):
        raise PathSumUnsupported("the qudit elimination orders differ")
    steps_l = _qudit_steps(raw_l, red_l, order)
    steps_r = _qudit_steps(raw_r, red_r, order)
    k_l, k_r = _qudit_const_turn(red_l, order), _qudit_const_turn(red_r, order)
    n = math.lcm(p, raw_l.const.n, raw_r.const.n)
    sc = _scalars()

    def fac(steps: Sequence[str]) -> tuple[str, sp.Expr]:
        texts, model = [], sp.Integer(1)
        for st in steps:
            if st in (".elim", ".hhCons", ".hhPivot"):
                texts.append(f"({p} : ℂ)")
                model *= p
            elif st.startswith(".gauss "):
                a = int(st.split()[1])
                texts.append(f"qg {p} {a}")
                model *= _legendre_model(a, p, n)
        return (" * ".join(texts) or "1"), model

    (ft_l, fm_l), (ft_r, fm_r) = fac(steps_l), fac(steps_r)
    c_l, c_r = _cyclo_model(raw_l.const, n), _cyclo_model(raw_r.const, n)
    lhs = fm_l * c_l * Om(n) ** (k_l * n // p)
    rhs = fm_r * c_r * Om(n) ** (k_r * n // p)
    proof = sc.prove(lhs, rhs)
    gauss = sorted({int(st.split()[1]) for st in (*steps_l, *steps_r) if st.startswith(".gauss")})
    lg_vals = {k: legendre(k, p) if k % p else 0 for k in range(p)}
    for a in gauss:
        lg_vals[a] = legendre(a, p)
    sub = f"rw [om_sub {n} {p} {n // p} (by norm_num) (by norm_num)]" if n != p else "skip"
    lines = [
        "import ArchytasZX.PathSumQudit",
        "",
        "/-!",
        f"# Qudit path-sum proof `{name}` over Z_{p}",
        "",
        "Both path sums, each side's steps replayed and checked by `decide`, the Gauss sums in",
        "their Legendre form, the constants' identity, and the value equality at every boundary",
        "assignment.",
        "-/",
        "",
        f"namespace ArchytasZX.PathSumQuditProof.{name}",
        "",
        "open ArchytasZX ArchytasZX.QSem",
        "",
        "set_option maxRecDepth 100000",
        "set_option maxHeartbeats 4000000",
        "set_option linter.unusedTactic false",
        "set_option linter.unusedSimpArgs false",
        "set_option linter.unreachableTactic false",
        "",
        f"instance : Fact (Nat.Prime {p}) := ⟨by norm_num⟩",
        "",
        f"def sL : QSem.Data := {_qdata_term(raw_l)}",
        f"def sR : QSem.Data := {_qdata_term(raw_r)}",
        f"def stepsL : List QSem.Step := [{', '.join(steps_l)}]",
        f"def stepsR : List QSem.Step := [{', '.join(steps_r)}]",
        "",
        *(f"theorem lg{k} : lg {p} {k} = {v} := by decide" for k, v in sorted(lg_vals.items())),
        "",
    ]
    for a in gauss:
        value = sc.text(_legendre_model(a, p, n))
        lines += [
            f"theorem qg{a} : qg {p} {a} = {value} := by",
            f"  rw [qg_legendre (by norm_num) {a} (by decide)]",
            "  simp only [Finset.sum_range_succ, Finset.sum_range_zero, "
            + ", ".join(f"lg{k}" for k in sorted(lg_vals))
            + "]",
            f"  {sub}",
            "  push_cast",
            "  ring",
            "",
        ]
    for side, steps, text in (("L", steps_l, ft_l), ("R", steps_r, ft_r)):
        lines += [
            f"theorem fac{side} : QSem.facs {p} steps{side} = {text} := by",
            f"  simp only [steps{side}, QSem.facs, QSem.fac]",
            "  push_cast",
            "  ring",
            "",
        ]
    lines += [
        f"theorem consts : ({ft_l}) * ({sc.text(c_l)}) * om {p} ^ {k_l} =",
        f"    ({ft_r}) * ({sc.text(c_r)}) * om {p} ^ {k_r} := by",
        *(f"  rw [qg{a}]" for a in gauss),
        f"  {sub}",
        f"  have key : ({sc.text(lhs)} : ℂ) = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        "  linear_combination key",
        "",
        "/-- The two path sums agree at every boundary assignment. -/",
        f"theorem main (σ : ℕ → ZMod {p}) :",
        f"    QSem.val {p} ({sc.text(c_l)}) sL σ = QSem.val {p} ({sc.text(c_r)}) sR σ := by",
        (
            "  rw [QSem.run_sound stepsL sL _ σ (by decide), "
            "QSem.run_sound stepsR sR _ σ (by decide)]"
        ),
        f"  apply QSem.final_sound _ _ _ _ σ {k_l} {k_r} (by decide) (by decide) (by decide)",
        "  first | (rw [facL, facR]; exact consts) | rw [facL, facR]",
        "",
        f"end ArchytasZX.PathSumQuditProof.{name}",
        "",
    ]
    return "\n".join(lines)


PIN_OFFSET = 1 << 20
"""The first digit variable of a summed wire that is pinned to another digit variable."""


def _poly_text(terms: Iterable[tuple[Iterable[int], int]]) -> str:
    """A PSem.Poly literal."""
    return "[" + ", ".join(f"({sorted(m)}, {c})" for m, c in terms) + "]"


def _table_poly(digits: list[int], table: list[Fraction], m: int) -> tuple[str, list[int]]:
    """The interpolation of ``table`` over ``digits`` as a Poly in units of 1 / m, and the table
    in units of 1 / m; raises when an entry is off the 1 / m grid."""
    from archytaszx.semantics.pathsum_diagram import _interpolate

    ts = []
    for t in table:
        v = (t % 1) * m
        if v.denominator != 1:
            raise PathSumUnsupported(f"a phase {t} off the 1/{m} grid")
        ts.append(int(v))
    poly = PhasePoly()
    _interpolate(poly, digits, table)
    terms = []
    for mono, c in sorted(poly.terms.items(), key=lambda t: (len(t[0]), sorted(t[0]))):
        v = (c % 1) * m
        if v.denominator != 1:
            raise PathSumUnsupported(f"a phase coefficient {c} off the 1/{m} grid")
        if int(v):
            terms.append((sorted(mono), int(v)))
    return _poly_text(terms), ts


@dataclass
class _NetSide:
    """One diagram's network over digit variables, its fragments and its steps."""

    raw: PathSum
    nodes: list[str]
    frags: list[str]
    proofs: list[str]
    summed: list[int]
    labels: list[int]
    dig: dict[int, list[int]]
    k_of: dict[int, int]
    boundary: int
    bnd_vars: int
    sqrt2_inv: int
    steps: list[str]
    scalar: sp.Expr


def _net_side(
    diagram: Diagram, m: int, phases: list[tuple[str, int]], label0: int
) -> tuple[_NetSide, PathSum]:
    """The network of ``diagram`` over digit variables numbered as from_diagram's: boundary port
    i is label i with its boundary digits, a summed wire is label ``label0 + w``, its digit j
    the class's variable when that is a path variable not yet taken, else a variable from
    PIN_OFFSET; each Z spider's digit j is anchored at the leg through which a search from the
    class's variable first reaches it; the pin steps substitute the other summed digits deepest
    first; ``phases`` collects the phase vectors' Lean bodies."""
    from archytaszx.algebra.pathsum import constant
    from archytaszx.diagram.generators import (
        DIM_BINDER,
        DIM_SPLITTER,
        FOURIER_BOX,
        X_SPIDER,
        Z_SPIDER,
    )
    from archytaszx.diagram.graph import Direction, NodeId, PortRef
    from archytaszx.semantics.pathsum_diagram import _turns_table

    lay: dict[str, Any] = {}
    raw = from_diagram(diagram, lay)
    if not isinstance(raw, PathSum):
        raise PathSumUnsupported("the network export covers qubit path sums only")
    d: Diagram = lay["diagram"]
    digit: dict[PortRef, list[int]] = lay["digit"]
    names = {
        Z_SPIDER.name: ".Z",
        X_SPIDER.name: ".X",
        FOURIER_BOX.name: ".F",
        DIM_BINDER.name: ".B",
        DIM_SPLITTER.name: ".S",
    }
    bnd = [*d.boundary_outputs, *d.boundary_inputs]
    label: dict[PortRef, int] = {}
    dig: dict[int, list[int]] = {}
    cls: dict[int, int] = {}
    v = 0
    for i, ref in enumerate(bnd):
        label[ref] = i
        dig[i] = list(range(v, v + len(digit[ref])))
        for j, x in enumerate(dig[i]):
            cls[x] = digit[ref][j]
        v += len(digit[ref])
    nb_vars = v
    taken: set[int] = set()
    summed: list[int] = []
    nxt = PIN_OFFSET
    for w, wire in enumerate(sorted(d.wires, key=lambda w: w.sort_key())):
        if wire.a in label or wire.b in label:
            raise PathSumUnsupported("a wire on a boundary port")
        lab = label0 + w
        ds = []
        for cv in digit[wire.a]:
            if cv >= nb_vars and cv not in taken:
                x = cv
                taken.add(cv)
            else:
                x, nxt = nxt, nxt + 1
            ds.append(x)
            cls[x] = cv
        label[wire.a] = label[wire.b] = lab
        dig[lab] = ds
        summed.append(lab)
    owner = {x: (lab, j) for lab, ds in dig.items() for j, x in enumerate(ds)}
    k_of = {lab: len(ds) for lab, ds in dig.items()}
    outs: dict[NodeId, list[int]] = {}
    ins: dict[NodeId, list[int]] = {}
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        if node.generator_type.name not in names:
            raise PathSumUnsupported(f"node {nid!r} is a {node.generator_type.name}")
        n_o, n_i = len(node.legs(Direction.OUTPUT)), len(node.legs(Direction.INPUT))
        outs[nid] = [label[PortRef(nid, Direction.OUTPUT, j)] for j in range(n_o)]
        ins[nid] = [label[PortRef(nid, Direction.INPUT, j)] for j in range(n_i)]
        if not outs[nid] and not ins[nid]:
            raise PathSumUnsupported(f"node {nid!r} has no legs")
    # hyperedges over digit variables: ("z", nid, j) sets and binder pairs
    edges: list[tuple[object, list[int]]] = []
    pair_of: dict[NodeId, tuple[int, int, int]] = {}
    for nid in sorted(d.nodes):
        kind = names[d.nodes[nid].generator_type.name]
        legs = outs[nid] + ins[nid]
        if kind == ".Z":
            ks = {k_of[lab] for lab in legs}
            if len(ks) != 1:
                raise PathSumUnsupported(f"node {nid!r} joins legs of different dimensions")
            for j in range(ks.pop()):
                edges.append((("z", nid, j), [dig[lab][j] for lab in legs]))
        elif kind in (".B", ".S"):
            whole, (a, b) = (outs[nid][0], ins[nid]) if kind == ".B" else (ins[nid][0], outs[nid])
            pair_of[nid] = (whole, a, b)
            for x, y in zip(dig[whole], dig[b] + dig[a], strict=True):
                edges.append((("b", nid), [x, y]))
    on: dict[int, list[int]] = {}
    for e, (_key, vs) in enumerate(edges):
        for x in vs:
            on.setdefault(x, []).append(e)
    entry: dict[int, int] = {}
    parent: dict[int, int] = {}
    reached: list[int] = []
    for root in sorted({x for x, c in cls.items() if x == c}):
        seen, queue = {root}, [root]
        while queue:
            x = queue.pop(0)
            for e in on.get(x, []):
                if e in entry:
                    continue
                entry[e] = x
                for y in edges[e][1]:
                    if y not in seen:
                        seen.add(y)
                        parent[y] = x
                        reached.append(y)
                        queue.append(y)
    if len(entry) != len(edges):
        raise PathSumUnsupported("a digit constraint not reached from its class's variable")
    anchor = {key: owner[entry[e]][0] for e, (key, _vs) in enumerate(edges)}
    nodes, frags, proofs = [], [], []
    k_inv = 0
    common = "E σ ownL labsL digL hown hσ _ rfl"
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        kind = names[node.generator_type.name]
        os_, is_ = outs[nid], ins[nid]
        legs = os_ + is_
        if kind in (".B", ".S"):
            whole, a, b = pair_of[nid]
            p, q = k_of[a], k_of[b]
            nodes.append(f"⟨{kind}, {os_}, {is_}, {p}, {q}⟩")
            frags.append(f"bFrag digL {whole} {a} {b}")
            lemma = "b_frag" if kind == ".B" else "s_frag"
            proofs.append(
                f"{lemma} {common} {p} {q} rfl rfl {whole} {a} {b} rfl rfl (by decide) "
                f"(by decide) (by decide) rfl rfl rfl {m}"
            )
            continue
        k = k_of[legs[0]]
        if kind == ".F":
            nodes.append(f"⟨.F, {os_}, {is_}, {k}, 0⟩")
            frags.append(f"fFragD {m} {1 << k} {k} digL {os_[0]} {is_[0]}")
            proofs.append(
                f"f_frag_dig {common} {k} rfl {os_[0]} {is_[0]} rfl rfl (by decide) (by decide) "
                f"rfl rfl {m} {1 << k} {m >> k} (by norm_num) (by norm_num) (by norm_num)"
            )
            k_inv += k
            continue
        if node.phase is not None and node.phase.is_weighted:
            raise PathSumUnsupported(f"node {nid!r} carries weights")
        table = _turns_table(nid, node, 1 << k)
        if kind == ".Z":
            as_ = [anchor[("z", nid, j)] for j in range(k)]
            anchor_vars = [dig[a][j] for j, a in enumerate(as_)]
            poly, ts = _table_poly(anchor_vars, table, m)
        else:
            ys = lay["fresh"][nid]
            poly, ts = _table_poly(ys, table, m)
        ph = 0
        if any(ts):
            phases.append((f"(({ts} : List ℕ).getD v 0 : ℕ)", m))
            ph = len(phases)
        ts_text = f"({ts if ph else []} : List ℕ)"
        nodes.append(f"⟨{kind}, {os_}, {is_}, {k}, {ph}⟩")
        if kind == ".Z":
            frags.append(f"zFragD digL {as_} {legs} {poly}")
            proofs.append(
                f"z_frag_dig {common} {k} rfl (by decide) (by decide) {as_} rfl (by decide) "
                f"{legs[0]} (by decide) {m} (by norm_num) {poly} {ts_text} (fun v => rfl) "
                "(by decide) (by decide) (by decide)"
            )
        else:
            frags.append(f"xFragD {m} {1 << k} {k} digL {poly} {ys} {os_} {is_}")
            proofs.append(
                f"x_frag_dig {common} {k} rfl (by decide) (by decide) {ys} rfl (by decide) "
                f"(by decide) {m} {1 << k} {m >> k} (by norm_num) (by norm_num) (by norm_num) "
                f"{poly} {ts_text} (fun v => rfl) (by decide) (by decide)"
            )
            k_inv += k * len(legs)
    cur = [x for lab in summed for x in dig[lab]]
    cur += [y for nid in sorted(d.nodes) for y in lay["fresh"].get(nid, [])]
    steps: list[str] = []
    for x in reversed(reached):
        if x < PIN_OFFSET:
            continue
        cur.remove(x)
        steps += [f".perm {[x, *cur]}", f".pin [[{parent[x]}]]"]
    steps.append(".norm")
    if sorted(cur) != sorted(raw.path):
        raise PathSumUnsupported("the pinned network keeps other path variables")
    if cur:
        steps.append(f".perm {sorted(cur)}")
    c, kc = constant(d.scalar.to_sympy())
    side = _NetSide(
        raw,
        nodes,
        frags,
        proofs,
        summed,
        sorted(dig),
        dig,
        k_of,
        len(bnd),
        nb_vars,
        k_inv,
        steps,
        _const_model(c, kc),
    )
    return side, raw


def _net_lines(tag: str, side: _NetSide, m: int) -> list[str]:
    """The digit tables, the network, its fragments and the NetPSD facts of one side."""
    pairs = ", ".join(f"({n}, {f})" for n, f in zip(side.nodes, side.frags, strict=True))
    cases = " | ".join("rfl" for _ in side.nodes)
    own = [(x, (lab, j)) for lab in side.labels for j, x in enumerate(side.dig[lab])]
    dig_tbl = [(lab, side.dig[lab]) for lab in side.labels]
    proofs = [
        p.replace("ownL", f"own{tag}").replace("labsL", f"labs{tag}").replace("digL", f"dig{tag}")
        for p in side.proofs
    ]
    frags = pairs.replace("digL", f"dig{tag}")
    return [
        f"def own{tag} : Own := {own}",
        f"def labs{tag} : List ℕ := {side.labels}",
        f"def dig{tag} (l : ℕ) : List ℕ := ((({dig_tbl} : List (ℕ × List ℕ))).lookup l).getD []",
        "",
        f"theorem own{tag}OK : ownOK own{tag} labs{tag} dig{tag} = true := by decide",
        "",
        (
            f"def N{tag} : Net := ⟨[{', '.join(side.nodes)}], {side.summed}, "
            f"{list(range(side.boundary))}⟩"
        ),
        "",
        f"noncomputable def ps{tag} : List (Node × Frag) := [{frags}]",
        "",
        f"theorem net{tag} : NetPSD E N{tag} {m} own{tag} labs{tag} dig{tag} ps{tag} where",
        f"  ownok := own{tag}OK",
        "  nodes := rfl",
        "  node := by",
        "    intro p hp σ hσ",
        f"    have hown := own{tag}OK",
        f"    simp only [ps{tag}, List.mem_cons, List.not_mem_nil, or_false] at hp",
        f"    rcases hp with {cases}",
        *(f"    · exact {pr}" for pr in proofs),
        "  disj := by decide",
        "  dims := by decide",
        "  cover := by decide",
        "  sub := by decide",
        "",
        f"theorem c{tag} : (joinAll (ps{tag}.map Prod.snd)).c = (rt 2)⁻¹ ^ {side.sqrt2_inv} := by",
        (
            f"  simp only [ps{tag}, List.map_cons, List.map_nil, joinAll, Frag.join, Frag.unit, "
            "zFragD, xFragD, fFragD, bFrag, List.length_cons, List.length_nil]"
        ),
        "  ring",
        "",
    ]


@dataclass
class NetBody:
    """One network proof's Lean lines (over the shared ``E``), its labels' digit counts and the
    two diagrams' scalars as Lean terms."""

    lines: list[str]
    labs: dict[int, int]
    s_l: str
    s_r: str
    m: int


def net_body(left: Diagram, right: Diagram, name: str, phases: list[tuple[str, int]]) -> NetBody:
    """The lines proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X, F, B
    and S nodes on dimensions 2^k, their phase vectors appended to ``phases`` (see
    :func:`export_net`); raises :class:`PathSumUnsupported` outside that class or when the sums
    do not reduce to one canonical form."""
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

    raw_l, raw_r = from_diagram(left), from_diagram(right)
    if not isinstance(raw_l, PathSum) or not isinstance(raw_r, PathSum):
        raise PathSumUnsupported("the network export covers qubit path sums only")
    if raw_l.boundary != raw_r.boundary:
        raise PathSumUnsupported("the boundaries differ")
    red_l, red_r, _order, boolean, m, steps_l, steps_r = _plan(raw_l, raw_r)
    side_l, _ = _net_side(left, m, phases, 1 << 16)
    side_r, _ = _net_side(right, m, phases, 1 << 17)
    if [side_l.k_of[i] for i in range(side_l.boundary)] != [
        side_r.k_of[i] for i in range(side_r.boundary)
    ]:
        raise PathSumUnsupported("the boundary ports carry different dimensions")
    side_l.steps += steps_l
    side_r.steps += steps_r
    f_l, f_r = _fac_model(side_l.steps, m), _fac_model(side_r.steps, m)
    k_l, k_r = _const_turn(red_l, m, boolean), _const_turn(red_r, m, boolean)
    sc = _scalars()
    s_l, s_r = sc.text(side_l.scalar), sc.text(side_r.scalar)
    lhs = f_l * side_l.scalar * Inv(Rt(2)) ** side_l.sqrt2_inv * Om(m) ** k_l
    rhs = f_r * side_r.scalar * Inv(Rt(2)) ** side_r.sqrt2_inv * Om(m) ** k_r
    proof = sc.prove(lhs, rhs)
    bnd = list(range(side_l.bnd_vars))
    run_l = f"(run {m} stepsL (netDataD digL NL (psL.map Prod.snd)))"
    lines = [
        *_net_lines("L", side_l, m),
        *_net_lines("R", side_r, m),
        f"def stepsL : List Step := [{', '.join(side_l.steps)}]",
        f"def stepsR : List Step := [{', '.join(side_r.steps)}]",
        "",
        *_fac_lines("L", side_l.steps, m, sc.text(f_l)),
        *_fac_lines("R", side_r.steps, m, sc.text(f_r)),
        (
            f"theorem consts : ({sc.text(f_l)}) * (({s_l}) * (rt 2)⁻¹ ^ {side_l.sqrt2_inv}) * "
            f"om {m} ^ {k_l} ="
        ),
        f"    ({sc.text(f_r)}) * (({s_r}) * (rt 2)⁻¹ ^ {side_r.sqrt2_inv}) * om {m} ^ {k_r} := by",
        f"  have key : ({sc.text(lhs)} : ℂ) = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        "  linear_combination key",
        "",
        "/-- The two networks are equal. -/",
        f"theorem {name} : Equal E ({s_l}) NL ({s_r}) NR := by",
        "  apply equal_psd2 _ _ netL netR rfl",
        "  intro σ",
        "  rw [cL, cR, ← val_smul, ← val_smul,",
        (
            f"    run_sound {m} (by norm_num) stepsL (netDataD digL NL (psL.map Prod.snd)) _ _ "
            "(by decide),"
        ),
        (
            f"    run_sound {m} (by norm_num) stepsR (netDataD digR NR (psR.map Prod.snd)) _ _ "
            "(by decide),"
        ),
        f"    val_agree {m} _ {run_l} {bnd} _ _ (by decide) (by decide)",
        f"      (dbits_agree ownL ownR _ {bnd} (by decide))]",
        f"  apply final_sound {m} (by norm_num) _ _ _ _ _ {k_l} {k_r} (by decide) (by decide)",
        "    (by decide)",
        "  first | (rw [facL, facR]; exact consts) | rw [facL, facR]",
        "",
    ]
    return NetBody(lines, {**side_l.k_of, **side_r.k_of}, s_l, s_r, m)


def net_env(
    phases: Sequence[tuple[str, int]], labs: dict[int, int], dims: str, m: int
) -> list[str]:
    """The shared environment ``E`` of network proofs: ``Δ k`` as ``dims``, the phase vectors,
    phase index 0 the zero vector over ``m``, and each label's digit count."""
    phi = [
        f"  | {i + 1} => fun v => {body} / (({m} : ℕ) : ℝ)" for i, (body, m) in enumerate(phases)
    ]
    return [
        "/-- Dimension values by digit count. -/",
        f"def Δ (k : ℕ) : ℕ := {dims}",
        "",
        "/-- Phase vectors in turns. -/",
        "noncomputable def Φ : ℕ → ℕ → ℝ",
        *phi,
        f"  | _ => fun v => ((([] : List ℕ).getD v 0 : ℕ) : ℝ) / (({m} : ℕ) : ℝ)",
        "",
        f"def labs : List (ℕ × ℕ) := {sorted(labs.items())}",
        "",
        "noncomputable def E : Env := { dimv := Δ, phase := Φ, labs := labs }",
        "",
    ]


NET_OPTIONS = [
    "set_option maxRecDepth 100000",
    "set_option maxHeartbeats 4000000",
    "set_option linter.unusedSimpArgs false",
    "set_option linter.unusedTactic false",
]
"""The options of a network proof file."""


def export_net(left: Diagram, right: Diagram, name: str = "main") -> str:
    """Lean 4 source proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X,
    F, B and S nodes on dimensions 2^k: each network's value is its fragments' path sum over
    digit variables (NetPSD.eval_eq), whose pin steps reach from_diagram's variables, then each
    side's reduction steps, the two final sums matched on the boundary digits (val_agree) and
    the final comparison; raises :class:`PathSumUnsupported` outside that class or when the sums
    do not reduce to one canonical form."""
    phases: list[tuple[str, int]] = []
    body = net_body(left, right, name, phases)
    ns = "ArchytasZX.PathSumNetProof"
    lines = [
        "import ArchytasZX.PathSumDigits",
        "",
        "/-!",
        f"# Path-sum proof `{name}` over the networks",
        "",
        "Each network's value is its fragments' path sum over digit variables; pin steps",
        "substitute the summed digits of one class into one variable, the reduction steps",
        "follow, and the final comparison gives `Equal`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        *NET_OPTIONS,
        "",
        *net_env(phases, body.labs, "2 ^ k", body.m),
        *body.lines,
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


def _qpoly(q: ModPoly) -> str:
    """A QSem.Poly literal."""
    return "[" + ", ".join(f"({[tuple(t) for t in m]}, {c})" for m, c in q.key()) + "]"


def _qvar(p: int, v: int) -> ModPoly:
    """The variable ``v`` over F_p."""
    return ModPoly(p, {((v, 1),): 1})


@dataclass
class _QNetSide:
    """One diagram's network over Z_p: nodes, fragments, fragment proofs, summed labels, the
    number of boundary labels, the power of ``1 / sqrt p`` in its fragments, its steps, the
    reduced path sum after the pins, and its scalar."""

    nodes: list[str]
    frags: list[str]
    proofs: list[str]
    summed: list[int]
    boundary: int
    inv_root: int
    steps: list[str]
    red: QuditPathSum
    order: list[int]
    scalar: sp.Expr


def _qnet_side(diagram: Diagram, phases: list[tuple[str, int]], sc: Any) -> _QNetSide:
    """The network of ``diagram`` over Z_p, every label of dimension p: boundary port i is label
    i, a summed wire is labelled by its class's variable when that is a path variable not yet
    taken, else by a label from PIN_OFFSET; each Z spider is anchored at the leg through which a
    search from the class's variable first reaches it; the pin steps set the other summed labels
    to their anchors deepest first, then the trace of the pinned sum's reduction follows."""
    from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
    from archytaszx.diagram.graph import Direction, NodeId, PortRef
    from archytaszx.semantics.pathsum_diagram import _qudit_table, _turns_table

    lay: dict[str, Any] = {}
    raw = from_diagram(diagram, lay)
    if not isinstance(raw, QuditPathSum):
        raise PathSumUnsupported("the qudit network export covers odd prime path sums only")
    if raw.atoms or raw.katoms:
        raise PathSumUnsupported("the qudit network export has no atoms")
    p = raw.p
    d: Diagram = lay["diagram"]
    digit: dict[PortRef, list[int]] = lay["digit"]
    names = {Z_SPIDER.name: ".Z", X_SPIDER.name: ".X", FOURIER_BOX.name: ".F"}
    bnd = [*d.boundary_outputs, *d.boundary_inputs]
    label: dict[PortRef, int] = {}
    cls: dict[int, int] = {}
    for ref, ds in digit.items():
        if len(ds) != 1:
            raise PathSumUnsupported(f"a leg of dimension {p}^{len(ds)}")
    for i, ref in enumerate(bnd):
        label[ref] = i
        cls[i] = digit[ref][0]
    nb = len(bnd)
    taken: set[int] = set()
    summed: list[int] = []
    nxt = PIN_OFFSET
    for wire in sorted(d.wires, key=lambda w: w.sort_key()):
        if wire.a in label or wire.b in label:
            raise PathSumUnsupported("a wire on a boundary port")
        cv = digit[wire.a][0]
        if cv >= nb and cv not in taken:
            x = cv
            taken.add(cv)
        else:
            x, nxt = nxt, nxt + 1
        cls[x] = cv
        label[wire.a] = label[wire.b] = x
        summed.append(x)
    outs: dict[NodeId, list[int]] = {}
    ins: dict[NodeId, list[int]] = {}
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        if node.generator_type.name not in names:
            raise PathSumUnsupported(f"node {nid!r} is a {node.generator_type.name}")
        if node.phase is not None and node.phase.is_weighted:
            raise PathSumUnsupported(f"node {nid!r} carries weights")
        n_o, n_i = len(node.legs(Direction.OUTPUT)), len(node.legs(Direction.INPUT))
        outs[nid] = [label[PortRef(nid, Direction.OUTPUT, j)] for j in range(n_o)]
        ins[nid] = [label[PortRef(nid, Direction.INPUT, j)] for j in range(n_i)]
        if not outs[nid] and not ins[nid]:
            raise PathSumUnsupported(f"node {nid!r} has no legs")
    zs = [nid for nid in sorted(d.nodes) if names[d.nodes[nid].generator_type.name] == ".Z"]
    on: dict[int, list[NodeId]] = {}
    for nid in zs:
        for x in outs[nid] + ins[nid]:
            on.setdefault(x, []).append(nid)
    entry: dict[NodeId, int] = {}
    parent: dict[int, int] = {}
    reached: list[int] = []
    for root in sorted(x for x, c in cls.items() if x == c):
        seen, queue = {root}, [root]
        while queue:
            x = queue.pop(0)
            for nid in on.get(x, []):
                if nid in entry:
                    continue
                entry[nid] = x
                for y in outs[nid] + ins[nid]:
                    if y not in seen:
                        seen.add(y)
                        parent[y] = x
                        reached.append(y)
                        queue.append(y)
    if len(entry) != len(zs):
        raise PathSumUnsupported("a Z spider not reached from its class's variable")
    xs = [nid for nid in sorted(d.nodes) if names[d.nodes[nid].generator_type.name] == ".X"]
    fresh = dict(zip(xs, sorted(raw.path)[len(raw.path) - len(xs) :], strict=True))
    nodes, frags, proofs = [], [], []
    phase = ModPoly(p)
    cons: list[ModPoly] = []
    path = list(summed)
    inv_root = 0
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        kind = names[node.generator_type.name]
        os_, is_ = outs[nid], ins[nid]
        legs = os_ + is_
        if kind == ".F":
            if os_[0] == is_[0]:
                raise PathSumUnsupported(f"node {nid!r} is an F on a self-loop")
            nodes.append(f"⟨.F, {os_}, {is_}, 0, 0⟩")
            frags.append(f"fFragQ {p} {os_[0]} {is_[0]}")
            proofs.append(f"f_frag_q (p := {p}) E σ _ rfl rfl {os_[0]} {is_[0]} rfl rfl")
            phase = phase + _qvar(p, os_[0]) * _qvar(p, is_[0])
            inv_root += 1
            continue
        var = entry[nid] if kind == ".Z" else fresh[nid]
        table = _qudit_table(nid, node, [var], p)
        ts = []
        for t in _turns_table(nid, node, p):
            v = (t % 1) * p
            if v.denominator != 1:
                raise PathSumUnsupported(f"node {nid!r} has a phase {t} off the 1/{p} grid")
            ts.append(int(v))
        ph = 0
        if any(ts):
            phases.append((f"(({ts} : List ℕ).getD v 0 : ℕ)", p))
            ph = len(phases)
        ts_text = f"({ts if ph else []} : List ℕ)"
        nodes.append(f"⟨{kind}, {os_}, {is_}, 0, {ph}⟩")
        poly = _qpoly(table)
        phase = phase + table
        if kind == ".Z":
            frags.append(f"zFragQ {p} {var} {legs} {poly}")
            proofs.append(
                f"z_frag_q (p := {p}) E σ hσ _ rfl rfl {var} (by decide) {poly} {ts_text} "
                "(fun v => rfl) (by decide)"
            )
            cons += [_qvar(p, var) + _qvar(p, leg).scale(-1) for leg in legs]
        else:
            frags.append(f"xFragQ {p} {var} {poly} {os_} {is_}")
            proofs.append(
                f"x_frag_q (p := {p}) E σ hσ _ rfl rfl {var} (by decide) {poly} {ts_text} "
                "(fun v => rfl) (by decide)"
            )
            for leg in os_:
                phase = phase + _qvar(p, var) * _qvar(p, leg)
            for leg in is_:
                phase = phase + (_qvar(p, var) * _qvar(p, leg)).scale(-1)
            path.append(var)
            inv_root += len(legs)
    steps: list[str] = []
    for x in reversed(reached):
        if x < PIN_OFFSET:
            continue
        a = parent[x]
        hit = next(
            (
                q
                for q in cons
                if (c := q.terms.get(((x, 1),), 0))
                and x not in (q + ModPoly(p, {((x, 1),): -c % p})).variables()
                and (q + ModPoly(p, {((x, 1),): -c % p})).scale(-pow(c, -1, p)).terms
                == {((a, 1),): 1}
            ),
            None,
        )
        if hit is None:
            raise PathSumUnsupported(f"no constraint sets the label {x} to {a}")
        cons.remove(hit)
        path.remove(x)
        steps += [f".perm {[x, *path]}", f".pin {a}"]
        value = _qvar(p, a)
        phase = phase.substitute(x, value)
        cons = [q.substitute(x, value) for q in cons]
    steps.append(".norm")
    if sorted(path) != sorted(raw.path):
        raise PathSumUnsupported("the pinned network keeps other path variables")
    sim = QuditPathSum(p, raw.boundary, set(path), phase, raw.const, [q for q in cons if q.terms])
    red = reduce_qudit(sim)
    order = _qudit_order(red.constraints, red.boundary)
    steps += _qudit_steps(sim, red, order)
    scalar = sc.model(d.scalar.to_sympy())
    if sc.opaques:
        raise PathSumUnsupported(f"the scalar {d.scalar} has no exact Lean form")
    return _QNetSide(nodes, frags, proofs, summed, nb, inv_root, steps, red, order, scalar)


def _qnet_lines(tag: str, side: _QNetSide, p: int) -> list[str]:
    """The network, its fragments and its QNetPS facts of one side."""
    pairs = ", ".join(f"({n}, {f})" for n, f in zip(side.nodes, side.frags, strict=True))
    cases = " | ".join("rfl" for _ in side.nodes)
    return [
        (
            f"def N{tag} : Net := ⟨[{', '.join(side.nodes)}], {side.summed}, "
            f"{list(range(side.boundary))}⟩"
        ),
        "",
        f"noncomputable def ps{tag} : List (Node × QFrag) := [{pairs}]",
        "",
        f"theorem net{tag} : QNetPS E N{tag} {p} ps{tag} where",
        "  nodes := rfl",
        "  node := by",
        "    intro q hq σ hσ",
        f"    simp only [ps{tag}, List.mem_cons, List.not_mem_nil, or_false] at hq",
        f"    rcases hq with {cases}",
        *(f"    · exact {pr}" for pr in side.proofs),
        "  disj := by decide",
        "  dims := by decide",
        "  cover := by decide",
        "",
        (
            f"theorem c{tag} : (qjoinAll (ps{tag}.map Prod.snd)).c = "
            f"(rt {p})⁻¹ ^ {side.inv_root} := by"
        ),
        (
            f"  simp only [ps{tag}, List.map_cons, List.map_nil, qjoinAll, QFrag.join, QFrag.unit, "
            "zFragQ, xFragQ, fFragQ, List.length_cons, List.length_nil]"
        ),
        "  ring",
        "",
    ]


def export_qudit_net(left: Diagram, right: Diagram, name: str = "main") -> str:
    """Lean 4 source proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X
    and F nodes on one odd prime dimension p (ArchytasZX/PathSumQuditNet.lean): each network's
    value is its fragments' path sum over Z_p (QNetPS.eval_eq), pin steps set the summed labels
    of one class to one variable, each side's reduction trace is replayed, the Gauss sums take
    their Legendre form and the final comparison gives ``Equal``; raises
    :class:`PathSumUnsupported` outside that class or when the sums do not reduce to one
    canonical form, :class:`ScalarUnsupported` when the constants' identity has no proof."""
    from archytaszx.algebra.pathsum import legendre
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

    phases: list[tuple[str, int]] = []
    sc = _scalars()
    side_l = _qnet_side(left, phases, sc)
    side_r = _qnet_side(right, phases, sc)
    p = side_l.red.p
    if side_r.red.p != p or side_l.red.boundary != side_r.red.boundary:
        raise PathSumUnsupported("the two path sums have different bases or boundaries")
    can_l, can_r = canonical_qudit(side_l.red), canonical_qudit(side_r.red)
    if can_l is None or can_r is None or not can_l.same(can_r) or can_l.zero:
        raise PathSumUnsupported("the qudit path sums do not reduce to one canonical form")
    if side_l.order != side_r.order:
        raise PathSumUnsupported("the qudit elimination orders differ")
    k_l = _qudit_const_turn(side_l.red, side_l.order)
    k_r = _qudit_const_turn(side_r.red, side_r.order)
    n = 12 if p == 3 else p
    sub = f"rw [om_sub {n} {p} {n // p} (by norm_num) (by norm_num)]" if n != p else "skip"

    def fac(steps: Sequence[str]) -> tuple[str, sp.Expr]:
        texts, model = [], sp.Integer(1)
        for st in steps:
            if st in (".elim", ".hhCons", ".hhPivot"):
                texts.append(f"({p} : ℂ)")
                model *= p
            elif st.startswith(".gauss "):
                a = int(st.split()[1])
                texts.append(f"qg {p} {a}")
                model *= _legendre_model(a, p, n)
        return (" * ".join(texts) or "1"), model

    (ft_l, fm_l), (ft_r, fm_r) = fac(side_l.steps), fac(side_r.steps)
    s_l, s_r = sc.text(side_l.scalar), sc.text(side_r.scalar)
    root = Inv(Rt(sp.Integer(p)))
    lhs = fm_l * side_l.scalar * root**side_l.inv_root * Om(n) ** (k_l * n // p)
    rhs = fm_r * side_r.scalar * root**side_r.inv_root * Om(n) ** (k_r * n // p)
    proof = sc.prove(lhs, rhs)
    steps_all = (*side_l.steps, *side_r.steps)
    gauss = sorted({int(st.split()[1]) for st in steps_all if st.startswith(".gauss")})
    lg_vals = {k: legendre(k, p) if k % p else 0 for k in range(p)}
    ns = "ArchytasZX.PathSumQuditNetProof"
    lines = [
        "import ArchytasZX.PathSumQuditNet",
        "",
        "/-!",
        f"# Qudit network proof `{name}` over Z_{p}",
        "",
        "Each network's value is its fragments' path sum over Z_p; pin steps set the summed labels",
        "of one class to one variable, the reduction steps follow, and the final comparison gives",
        "`Equal`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.QSem",
        "",
        *NET_OPTIONS,
        "set_option linter.unreachableTactic false",
        "set_option linter.unusedVariables false",
        "",
        f"instance : Fact (Nat.Prime {p}) := ⟨by norm_num⟩",
        "",
        *net_env(phases, {}, str(p), p),
        *_qnet_lines("L", side_l, p),
        *_qnet_lines("R", side_r, p),
        f"def stepsL : List QSem.Step := [{', '.join(side_l.steps)}]",
        f"def stepsR : List QSem.Step := [{', '.join(side_r.steps)}]",
        "",
        *(f"theorem lg{k} : lg {p} {k} = {v} := by decide" for k, v in sorted(lg_vals.items())),
        "",
    ]
    for a in gauss:
        lines += [
            f"theorem qg{a} : qg {p} {a} = {sc.text(_legendre_model(a, p, n))} := by",
            f"  rw [qg_legendre (by norm_num) {a} (by decide)]",
            "  simp only [Finset.sum_range_succ, Finset.sum_range_zero, "
            + ", ".join(f"lg{k}" for k in sorted(lg_vals))
            + "]",
            f"  {sub}",
            "  push_cast",
            "  ring",
            "",
        ]
    for side, text in (("L", ft_l), ("R", ft_r)):
        lines += [
            f"theorem fac{side} : QSem.facs {p} steps{side} = {text} := by",
            f"  simp only [steps{side}, QSem.facs, QSem.fac]",
            "  push_cast",
            "  ring",
            "",
        ]
    lines += [
        (
            f"theorem consts : ({ft_l}) * (({s_l}) * (rt {p})⁻¹ ^ {side_l.inv_root}) * "
            f"om {p} ^ {k_l} ="
        ),
        f"    ({ft_r}) * (({s_r}) * (rt {p})⁻¹ ^ {side_r.inv_root}) * om {p} ^ {k_r} := by",
        f"  have key : ({sc.text(lhs)} : ℂ) = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        *(f"  all_goals rw [qg{a}]" for a in gauss),
        f"  all_goals {sub}",
        "  all_goals linear_combination key",
        "",
        "/-- The two networks are equal. -/",
        f"theorem {name} : Equal E ({s_l}) NL ({s_r}) NR := by",
        "  apply equal_qps _ _ netL netR rfl",
        "  intro τ",
        "  rw [← QSem.val_smul, ← QSem.val_smul,",
        "    QSem.run_sound stepsL (qnetData NL (psL.map Prod.snd)) _ τ (by decide),",
        "    QSem.run_sound stepsR (qnetData NR (psR.map Prod.snd)) _ τ (by decide)]",
        f"  apply QSem.final_sound _ _ _ _ τ {k_l} {k_r} (by decide) (by decide) (by decide)",
        "  rw [facL, facR, cL, cR]",
        "  all_goals exact consts",
        "",
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "NetBody",
    "export_net",
    "export_pair",
    "export_proof",
    "export_qudit_net",
    "export_qudit_proof",
    "net_body",
]
