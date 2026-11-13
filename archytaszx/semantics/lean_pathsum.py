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
``linear_combination``. Neither enumerates a sum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import cast

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.pathsum import (
    Canon,
    Cyclo,
    PathSum,
    PathSumUnsupported,
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


def export_proof(left: Diagram, right: Diagram, name: str = "pair") -> str:
    """The Lean file proving that both diagrams' path sums agree at every boundary assignment;
    raises :class:`PathSumUnsupported` when they do not reduce to one canonical form or a
    trace step has no PathSumRules step."""
    from archytaszx.semantics.lean_scalar import Om, Scalars

    raw_l, raw_r = from_diagram(left), from_diagram(right)
    if not isinstance(raw_l, PathSum) or not isinstance(raw_r, PathSum):
        raise PathSumUnsupported("export_proof covers qubit path sums only")
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
    f_l, f_r = _fac_model(steps_l, m), _fac_model(steps_r, m)
    k_l, k_r = _const_turn(red_l, m, boolean), _const_turn(red_r, m, boolean)
    sc = Scalars(
        nat=lambda sym: str(sym),
        other=lambda sym: (str(sym), "ℂ"),
        positive=lambda x: "(by norm_num)",
    )
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


__all__ = ["export_pair", "export_proof"]
