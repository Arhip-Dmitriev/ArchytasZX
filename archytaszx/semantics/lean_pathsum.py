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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.pathsum import (
    ONE,
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


def trace_step(step: Step, cur: list[int]) -> tuple[list[str], list[int]]:
    """The PSem.Step texts replaying one trace step on the path ``cur``, and the path after."""
    y, z = step.var, step.pivot
    if step.rule in _HEAD_STEP:
        rest = [v for v in cur if v != y]
        return [f".perm {[y, *rest]}", _HEAD_STEP[step.rule]], rest
    if step.rule == "rem":
        cur = [y, *(v for v in cur if v != y)]
        return [f".perm {cur}", ".rem"], cur
    if step.rule == "dyadic":
        rest = [v for v in cur if v not in step.group]
        poly = ", ".join(f"({[v for v, _e in m]}, {c})" for m, c in step.poly)
        return [
            f".perm {[*step.group, *rest]}",
            f".dyadic {list(step.group)} [{poly}] {list(step.fresh)}",
        ], [*step.fresh, *rest]
    if step.rule in _PAIR_STEP and z is not None:
        rest = [v for v in cur if v not in (y, z)]
        return [f".perm {[y, z, *rest]}", _PAIR_STEP[step.rule]], (
            rest if step.rule == "hh_pivot" else [y, *rest]
        )
    raise PathSumUnsupported(f"the rule {step.rule} has no PathSumRules step")


def _steps(raw: PathSum, ps: PathSum, order: Sequence[int], boolean: bool) -> list[str]:
    """The PSem.Step list replaying the trace of ``ps``, reduced from ``raw``, then the canon
    steps at ``order`` (gather and bcanon steps when ``boolean``)."""
    cur = sorted(raw.path)
    out: list[str] = []
    for step in ps.trace:
        texts, cur = trace_step(step, cur)
        out += texts
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
    raw_l: PathSum, raw_r: PathSum, m_min: int = 1
) -> tuple[PathSum, PathSum, list[int], bool, int, list[str], list[str]]:
    """Both reduced sums, the canonical pivots, whether Boolean elimination is needed, the
    phase modulus M (a multiple of ``m_min``) and each side's PSem.Step list; raises
    :class:`PathSumUnsupported` when the sums do not reduce to one canonical form."""
    if raw_l.amps or raw_r.amps:
        raise PathSumUnsupported("export_proof has no amplitude atoms")
    red_l, red_r = reduce(raw_l), reduce(raw_r)
    can_l, can_r = canonical(red_l), canonical(red_r)
    if can_l is None or can_r is None or not can_l.same(can_r) or can_l.zero:
        raise PathSumUnsupported("the path sums do not reduce to one canonical form")
    m = math.lcm(4, m_min)
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
    if raw_l.amps or raw_r.amps:
        return _export_amp_proof(raw_l, raw_r, name)
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


def _data_text(ps: PathSum, m: int, path: Sequence[int]) -> str:
    """A path sum with the path ``path`` as a PSem.Data literal."""
    rest = _data_term(ps, m).split(", phase := ", 1)[1]
    return f"{{ path := {list(path)}, phase := {rest}"


@dataclass
class AmpPlan:
    """Both sides' runs with atoms, their initial inner paths, held variables and atoms, the
    phase modulus and the constant turns of the reduced sums."""

    coefs: Any
    runs: tuple[Any, Any]
    starts: tuple[tuple[list[int], list[int], list[Any]], tuple[list[int], list[int], list[Any]]]
    m: int
    turns: tuple[int, int]


def amp_plan(sides: tuple[Any, Any]) -> tuple[AmpPlan, PathSum, PathSum]:
    """The runs of two path sums with atoms (each an initial AmpRun on its raw sum, the steps
    already written kept), reduced, with canon steps at the elimination pivots and the atoms
    substituted, normalised and merged (gather and bcanon steps when a side needs Boolean
    elimination); raises :class:`PathSumUnsupported` when the sums do not reduce to one
    canonical form, their Boolean eliminations differ or leave a support, or they end on
    different atoms."""
    from archytaszx.algebra.pathsum import boolean_eliminate, eliminate
    from archytaszx.semantics.lean_amp import reduce_run

    (raw_l, run_l), (raw_r, run_r) = sides
    red_l, red_r = reduce(raw_l), reduce(raw_r)
    can_l, can_r = canonical(red_l), canonical(red_r)
    if can_l is None or can_r is None or not can_l.same(can_r) or can_l.zero:
        raise PathSumUnsupported("the path sums do not reduce to one canonical form")
    rows_l = eliminate(red_l.constraints, red_l.boundary)
    rows_r = eliminate(red_r.constraints, red_r.boundary)
    boolean = rows_l is None or rows_r is None
    if boolean:
        rows_l, g_l = boolean_eliminate(red_l.constraints, red_l.boundary)
        rows_r, g_r = boolean_eliminate(red_r.constraints, red_r.boundary)
        order = [v for v, _r in rows_l]
        if g_l != ONE or g_r != ONE or order != [v for v, _r in rows_r]:
            raise PathSumUnsupported("the Boolean eliminations differ or leave a support")
        canon_steps = [".gather", *(f".bcanon {v}" for v in order)]
    else:
        order = _elimination_order(red_l.constraints, red_l.boundary)
        canon_steps = [f".canon {v}" for v in order]
    starts = []
    for run, red, rows in ((run_l, red_l, rows_l), (run_r, red_r, rows_r)):
        starts.append((list(run.cur), list(run.held), list(run.atoms)))
        reduce_run(run, red.trace)
        run.inner(canon_steps)
        run.substitute(cast(list[tuple[int, frozenset[frozenset[int]]]], rows))
    if run_l.final_key() != run_r.final_key():
        raise PathSumUnsupported("the reduced sums carry different amplitude atoms")
    m = run_l.coefs.m
    turns = (_const_turn(red_l, m, boolean), _const_turn(red_r, m, boolean))
    plan = AmpPlan(run_l.coefs, (run_l, run_r), (starts[0], starts[1]), m, turns)
    return plan, red_l, red_r


def amp_modulus(raws: Iterable[PathSum], m_min: int = 1) -> int:
    """The phase modulus of path sums with atoms: a multiple of 8, of ``m_min`` and of every
    phase denominator."""
    m = math.lcm(8, m_min)
    for ps in raws:
        for c in ps.phase.terms.values():
            m = math.lcm(m, c.denominator)
    return m


def _export_amp_proof(raw_l: PathSum, raw_r: PathSum, name: str) -> str:
    """:func:`export_proof` for path sums with amplitude atoms (ArchytasZX/PathSumAmp.lean):
    each side's AStep list replayed and checked by ``decide``, the coefficient identities of its
    [Average], zero, unit and merge steps, and the constants' identity."""
    from archytaszx.semantics.lean_amp import (
        Coefs,
        afac_lines,
        atoms_text,
        initial,
        reqs_lines,
    )
    from archytaszx.semantics.lean_scalar import Om

    m = amp_modulus((raw_l, raw_r))
    coefs = Coefs(m)
    plan, _red_l, _red_r = amp_plan(
        (
            (raw_l, initial(coefs, raw_l.path, raw_l.amps)),
            (raw_r, initial(coefs, raw_r.path, raw_r.amps)),
        )
    )
    sc = _scalars()
    run_l, run_r = plan.runs
    k_l, k_r = plan.turns
    fl_lines, f_l = afac_lines("L", run_l, sc)
    fr_lines, f_r = afac_lines("R", run_r, sc)
    c_l, c_r = _const_model(raw_l.const, raw_l.sqrt2), _const_model(raw_r.const, raw_r.sqrt2)
    lhs = f_l * c_l * Om(m) ** k_l
    rhs = f_r * c_r * Om(m) ** k_r
    proof = sc.prove(lhs, rhs)
    ns = f"ArchytasZX.PathSumProof.{name}"
    data = []
    for tag, raw, (cur, held, atoms), run in (
        ("L", raw_l, plan.starts[0], run_l),
        ("R", raw_r, plan.starts[1], run_r),
    ):
        data += [
            f"def s{tag} : Data := {_data_text(raw, m, cur)}",
            f"def t{tag} : AData := ⟨s{tag}, {held}, {atoms_text(atoms)}⟩",
            f"def steps{tag} : List AStep := [{', '.join(run.steps)}]",
            "",
        ]
    lines = [
        "import ArchytasZX.PathSumAmp",
        "",
        "/-!",
        f"# Path-sum proof `{name}` with amplitude atoms",
        "",
        "Both path sums with their atoms, each side's steps replayed and checked by `decide`, the",
        "coefficient identities, the constants' identity, and the value equality at every",
        "boundary assignment.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        *NET_OPTIONS,
        "",
        *coefs.lean_def(sc),
        *data,
        *fl_lines,
        *fr_lines,
        *reqs_lines("L", run_l.reqs, coefs, sc, "stepsL"),
        *reqs_lines("R", run_r.reqs, coefs, sc, "stepsR"),
        f"theorem consts : ({sc.text(f_l)}) * {sc.text(c_l)} * om {m} ^ {k_l} =",
        f"    ({sc.text(f_r)}) * {sc.text(c_r)} * om {m} ^ {k_r} := by",
        f"  have key : ({sc.text(lhs)} : ℂ) = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        "  linear_combination key",
        "",
        "/-- Both path sums with atoms agree at every boundary assignment. -/",
        "theorem main (σ : ℕ → ZMod 2) :",
        f"    valA {m} {sc.text(c_l)} A tL σ = valA {m} {sc.text(c_r)} A tR σ := by",
        f"  rw [aruns_sound {m} (by norm_num) A stepsL tL _ σ reqsL (by decide),",
        f"    aruns_sound {m} (by norm_num) A stepsR tR _ σ reqsR (by decide)]",
        f"  apply afinal_perm {m} (by norm_num) _ _ A _ _ σ {k_l} {k_r} (by decide) (by decide)",
        "    (by decide) (by decide) (by decide) (by decide)",
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
        "import ArchytasZX.GaussSign",
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

QPIN_OFFSET = 1 << 18
"""The first label of a summed wire on Z_p whose class variable is taken."""


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
    weights: list[sp.Expr]
    atoms: list[str] = field(default_factory=list)
    node_atoms: list[Any] = field(default_factory=list)
    pins: list[tuple[int, int]] = field(default_factory=list)
    held0: list[int] = field(default_factory=list)
    cur0: list[int] = field(default_factory=list)


def _net_side(
    diagram: Diagram,
    m: int,
    phases: list[tuple[str, int]],
    label0: int,
    coefs: dict[int, str] | None = None,
    amp: Any = None,
    tabs: dict[int, str] | None = None,
) -> tuple[_NetSide, PathSum]:
    """The network of ``diagram`` over digit variables numbered as from_diagram's: boundary port
    i is label i with its boundary digits, a summed wire is label ``label0 + w``, its digit j
    the class's variable when that is a path variable not yet taken, else a variable from
    PIN_OFFSET; each Z spider's digit j is anchored at the leg through which a search from the
    class's variable first reaches it; the pin steps substitute the other summed digits deepest
    first; ``phases`` collects the phase vectors' Lean bodies and ``coefs`` each weighted
    spider's base coefficient by its phase index; a legless spider takes its own digits as X
    does, a weighted one its turns over a common magnitude. With ``amp`` (a lean_amp.Coefs), a
    weighted spider whose weights are 0 or sqrt(2) times the magnitude takes atoms on its anchor
    digits (Z with legs) or its own digits (X, legless Z) and its coefficient index table by
    phase index in ``tabs``; the side then holds every summed digit and fragment path variable
    and keeps its pins and no steps."""
    from archytaszx.algebra.pathsum import constant
    from archytaszx.diagram.generators import (
        DIM_BINDER,
        DIM_SPLITTER,
        FOURIER_BOX,
        X_SPIDER,
        Z_SPIDER,
    )
    from archytaszx.diagram.graph import Direction, NodeId, PortRef
    from archytaszx.semantics.lean_scalar import Rt
    from archytaszx.semantics.pathsum_diagram import _phase_bits, _turns_table, _weights

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
        if not outs[nid] and not ins[nid] and names[node.generator_type.name] not in (".Z", ".X"):
            raise PathSumUnsupported(f"node {nid!r} has no legs")
    # hyperedges over digit variables: ("z", nid, j) sets and binder pairs
    edges: list[tuple[object, list[int]]] = []
    pair_of: dict[NodeId, tuple[int, int, int]] = {}
    for nid in sorted(d.nodes):
        kind = names[d.nodes[nid].generator_type.name]
        legs = outs[nid] + ins[nid]
        if kind == ".Z" and legs:
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
    side_atoms: list[list[Any]] = []
    weights: list[sp.Expr] = []
    k_inv = 0
    common = "E σ ownL labsL digL hown hσ _ rfl"
    own_ys = {**lay["fresh"], **lay.get("zdigits", {})}
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
            side_atoms.append([])
            if amp is not None:
                nil = f"atoms_nil A {m} _ _ ({proofs[-1]})"
                proofs[-1] = f"fvalA_of {m} A _ _ _ (by decide) ({nil})"
            continue
        k = k_of[legs[0]] if legs else _phase_bits(nid, node)
        if kind == ".F":
            nodes.append(f"⟨.F, {os_}, {is_}, {k}, 0⟩")
            frags.append(f"fFragD {m} {1 << k} {k} digL {os_[0]} {is_[0]}")
            proofs.append(
                f"f_frag_dig {common} {k} rfl {os_[0]} {is_[0]} rfl rfl (by decide) (by decide) "
                f"rfl rfl {m} {1 << k} {m >> k} (by norm_num) (by norm_num) (by norm_num)"
            )
            side_atoms.append([])
            if amp is not None:
                nil = f"atoms_nil A {m} _ _ ({proofs[-1]})"
                proofs[-1] = f"fvalA_of {m} A _ _ _ (by decide) ({nil})"
            k_inv += k
            continue
        weighted = node.phase is not None and node.phase.is_weighted
        zero: list[int] = []
        root2: list[int] = []
        if weighted:
            table, zero, root2, mag, mag_k = _weights(nid, node, 1 << k)
            if (any(zero) or any(root2)) and amp is None:
                raise PathSumUnsupported(f"node {nid!r} has weights of different magnitudes")
        else:
            table = _turns_table(nid, node, 1 << k)
        legless = kind == ".Z" and not legs
        if kind == ".Z" and legs:
            as_ = [anchor[("z", nid, j)] for j in range(k)]
            anchor_vars = [dig[a][j] for j, a in enumerate(as_)]
            poly, ts = _table_poly(anchor_vars, table, m)
        else:
            ys = own_ys[nid]
            anchor_vars = ys
            poly, ts = _table_poly(ys, table, m)
        ph = 0
        if any(ts) or weighted:
            phases.append((f"(({ts} : List ℕ).getD v 0 : ℕ)", m))
            ph = len(phases)
        ts_text = f"({ts if ph else []} : List ℕ)"
        nodes.append(f"⟨{kind}{'W' if weighted else ''}, {os_}, {is_}, {k}, {ph}⟩")
        if kind == ".Z" and legs:
            frag = f"zFragD digL {as_} {legs} {poly}"
            proof = (
                f"z_frag_dig {common} {k} rfl (by decide) (by decide) {as_} rfl (by decide) "
                f"{legs[0]} (by decide) {m} (by norm_num) {poly} {ts_text} (fun v => rfl) "
                "(by decide) (by decide) (by decide)"
            )
        else:
            frag = f"xFragD {m} {1 << k} {k} digL {poly} {ys} {os_} {is_}"
            proof = (
                f"x_frag_dig {common} {k} rfl (by decide) (by decide) {ys} rfl (by decide) "
                f"(by decide) {m} {1 << k} {m >> k} (by norm_num) (by norm_num) (by norm_num) "
                f"{poly} {ts_text} (fun v => rfl) (by decide) (by decide)"
            )
            k_inv += k * len(legs)
        if legless:
            proof = f"(val_z_legless E σ _ rfl rfl rfl).trans ({proof})"
        node_atoms: list[Any] = []
        if weighted:
            w = sp.Rational(mag.numerator, mag.denominator) * Rt(2) ** mag_k
            sc = _scalars()
            c_text = sc.text(w)
            if coefs is None:
                raise PathSumUnsupported(f"node {nid!r} carries weights and no coefficient table")
            coefs[ph] = c_text
            weights.append(w)
            lemma = "val_zw" if kind == ".Z" else "val_xw"
            if any(zero) or any(root2):
                from archytaszx.algebra.pathsum import SQRT2, Cyclo
                from archytaszx.semantics.lean_amp import atoms_text
                from archytaszx.semantics.pathsum_diagram import _f2_indicator

                inds = [(Cyclo.rational(0), zero), (SQRT2, root2)]
                inds = [(c, t) for c, t in inds if any(t)]
                node_atoms = [(amp.index(c), _f2_indicator(anchor_vars, t)) for c, t in inds]
                tbl = [
                    [j for (j, _f), (_c, t) in zip(node_atoms, inds, strict=True) if t[v]]
                    for v in range(1 << k)
                ]
                if tabs is None:
                    raise PathSumUnsupported(f"node {nid!r} carries atoms and no index table")
                tabs[ph] = str(tbl)
                frag = f"({frag}).scale ({c_text})"
                if kind == ".Z" and legs:
                    proof = (
                        f"za_frag E σ ownL labsL digL hown hσ _ rfl {k} rfl (by decide) "
                        f"(by decide) {as_} rfl (by decide) {legs[0]} (by decide) (by decide) "
                        f"{m} _ ({c_text}) A {tbl} {atoms_text(node_atoms)} (fun v => rfl) "
                        f"(by decide) ({proof})"
                    )
                else:
                    proof = (
                        f"xa_frag {common} {k} rfl (by decide) (by decide) {ys} rfl (by decide) "
                        f"(by decide) {m} {1 << k} {m >> k} (by norm_num) (by norm_num) "
                        f"(by norm_num) {poly} {ts_text} (fun v => rfl) (by decide) (by decide) "
                        f"({c_text}) A {tbl} {atoms_text(node_atoms)} (fun v => rfl) (by decide)"
                    )
                    if legless:
                        proof = f"(val_zw_legless E σ _ rfl rfl rfl).trans ({proof})"
            else:
                frag = f"({frag}).scale ({c_text})"
                proof = (
                    f"w_frag E σ _ _ _ _ _ ({c_text}) ({lemma} E σ _ rfl ({c_text}) "
                    f"(fun v => rfl)) ({proof})"
                )
        if amp is not None and not node_atoms:
            proof = f"atoms_nil A {m} _ _ ({proof})"
        if amp is not None and (kind == ".Z" and legs or not node_atoms):
            proof = f"fvalA_of {m} A _ _ _ (by decide) ({proof})"
        frags.append(frag)
        proofs.append(proof)
        side_atoms.append(node_atoms)
    cur = [x for lab in summed for x in dig[lab]]
    cur += [y for nid in sorted(d.nodes) for y in own_ys.get(nid, [])]
    if amp is not None:
        from archytaszx.semantics.lean_amp import atoms_text

        held0 = list(cur)
        pins = [(x, parent[x]) for x in reversed(reached) if x >= PIN_OFFSET]
        pinned = {x for x, _p in pins}
        if sorted(x for x in cur if x not in pinned) != sorted(raw.path):
            raise PathSumUnsupported("the pinned network keeps other path variables")
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
            [],
            _const_model(c, kc),
            weights,
            [atoms_text(a) for a in side_atoms],
            side_atoms,
            pins,
            held0,
            [],
        )
        return side, raw
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
        weights,
    )
    return side, raw


def _table_modulus(diagram: Diagram) -> int:
    """The least common denominator of the phase tables (in turns) of the Z and X spiders of
    ``diagram`` on dimensions 2^k."""
    from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
    from archytaszx.diagram.graph import Direction, PortRef
    from archytaszx.semantics.pathsum_diagram import _bits, _phase_bits, _turns_table, _weights

    out = 1
    for nid in sorted(diagram.nodes):
        node = diagram.nodes[nid]
        if node.generator_type.name not in (Z_SPIDER.name, X_SPIDER.name) or node.phase is None:
            continue
        if node.outputs or node.inputs:
            ref = PortRef(nid, Direction.OUTPUT if node.outputs else Direction.INPUT, 0)
            k = _bits(diagram, ref)
        else:
            k = _phase_bits(nid, node)
        table = (
            _weights(nid, node, 1 << k)[0]
            if node.phase.is_weighted
            else _turns_table(nid, node, 1 << k)
        )
        for t in table:
            out = math.lcm(out, Fraction(t % 1).denominator)
    return out


def _weights_text(side: _NetSide) -> str:
    """The product of a side's weighted spiders' base coefficients, each followed by ``*``."""
    sc = _scalars()
    return "".join(f"{sc.text(w)} * " for w in side.weights)


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
        (
            f"theorem c{tag} : (joinAll (ps{tag}.map Prod.snd)).c = "
            f"{_weights_text(side)}(rt 2)⁻¹ ^ {side.sqrt2_inv} := by"
        ),
        (
            f"  simp only [ps{tag}, List.map_cons, List.map_nil, joinAll, Frag.join, Frag.unit, "
            "zFragD, xFragD, fFragD, bFrag, Frag.scale, List.length_cons, List.length_nil]"
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


@dataclass
class _ZeroPlan:
    """The phase modulus of a pair of zero diagrams and, per side, its trace steps to a
    constraint 1 = 0, or None for a side whose scalar is zero."""

    m: int
    steps: tuple[list[str] | None, list[str] | None]


def _zero_plan(left: Diagram, right: Diagram, m_min: int) -> _ZeroPlan | None:
    """The zero plan of two diagrams each with scalar zero or a network reducing to a
    constraint 1 = 0; None otherwise."""
    m = math.lcm(4, m_min)
    steps: list[list[str] | None] = []
    for d in (left, right):
        raw = from_diagram(d)
        if not isinstance(raw, PathSum) or raw.amps:
            return None
        for c in raw.phase.terms.values():
            m = math.lcm(m, c.denominator)
        if sp.simplify(d.scalar.to_sympy()) == 0:
            steps.append(None)
            continue
        red = reduce(raw)
        if ONE not in red.constraints:
            return None
        if any(st.rule in ("omega", "omega_conj") for st in red.trace):
            m = math.lcm(m, 8)
        steps.append([*_steps(raw, red, [], False), ".norm"])
    return _ZeroPlan(m, (steps[0], steps[1]))


def _zero_body(
    left: Diagram,
    right: Diagram,
    name: str,
    phases: list[tuple[str, int]],
    coefs: dict[int, str] | None,
    plan: _ZeroPlan,
) -> NetBody:
    """:func:`net_body` for a zero plan: each network side run to a constraint 1 = 0 and
    ``val_of_zeroOK``, each other side times its zero scalar."""
    m = plan.m
    side_l, _ = _net_side(left, m, phases, 1 << 16, coefs)
    side_r, _ = _net_side(right, m, phases, 1 << 17, coefs)
    if [side_l.k_of[i] for i in range(side_l.boundary)] != [
        side_r.k_of[i] for i in range(side_r.boundary)
    ]:
        raise PathSumUnsupported("the boundary ports carry different dimensions")
    sc = _scalars()
    s_l, s_r = sc.text(side_l.scalar), sc.text(side_r.scalar)
    lines = [*_net_lines("L", side_l, m), *_net_lines("R", side_r, m)]
    rws: list[str] = []
    for tag, side, steps in (("L", side_l, plan.steps[0]), ("R", side_r, plan.steps[1])):
        if steps is None:
            continue
        lines += [f"def steps{tag} : List Step := [{', '.join([*side.steps, *steps])}]", ""]
        data = f"(netDataD dig{tag} N{tag} (ps{tag}.map Prod.snd))"
        rws += [
            f"run_sound {m} (by norm_num) steps{tag} {data} _ _ (by decide)",
            f"PSem.val_of_zeroOK {m} _ (run {m} steps{tag} {data}) _ (by decide)",
        ]
    lines += [
        "/-- The two networks are equal (both zero). -/",
        f"theorem {name} : Equal E ({s_l}) NL ({s_r}) NR := by",
        "  apply equal_psd2 _ _ netL netR rfl",
        "  intro σ",
        *([f"  rw [{',\n    '.join(rws)}]"] if rws else []),
        "  simp",
        "",
    ]
    return NetBody(lines, {**side_l.k_of, **side_r.k_of}, s_l, s_r, m)


def net_body(
    left: Diagram,
    right: Diagram,
    name: str,
    phases: list[tuple[str, int]],
    m_min: int = 1,
    coefs: dict[int, str] | None = None,
) -> NetBody:
    """The lines proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X, F, B
    and S nodes on dimensions 2^k, their phase vectors appended to ``phases``, phases over a
    multiple of ``m_min`` (see :func:`export_net`), each side zero by its scalar or by a
    constraint 1 = 0 included; raises :class:`PathSumUnsupported` outside that class or when
    the sums do not reduce to one canonical form."""
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

    raw_l, raw_r = from_diagram(left), from_diagram(right)
    if not isinstance(raw_l, PathSum) or not isinstance(raw_r, PathSum):
        raise PathSumUnsupported("the network export covers qubit path sums only")
    if raw_l.boundary != raw_r.boundary:
        raise PathSumUnsupported("the boundaries differ")
    m_min = math.lcm(m_min, _table_modulus(left), _table_modulus(right))
    try:
        red_l, red_r, _order, boolean, m, steps_l, steps_r = _plan(raw_l, raw_r, m_min)
    except PathSumUnsupported:
        zero = _zero_plan(left, right, m_min)
        if zero is None:
            raise
        return _zero_body(left, right, name, phases, coefs, zero)
    side_l, _ = _net_side(left, m, phases, 1 << 16, coefs)
    side_r, _ = _net_side(right, m, phases, 1 << 17, coefs)
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
    w_l, w_r = sp.Mul(*side_l.weights), sp.Mul(*side_r.weights)
    lhs = f_l * side_l.scalar * w_l * Inv(Rt(2)) ** side_l.sqrt2_inv * Om(m) ** k_l
    rhs = f_r * side_r.scalar * w_r * Inv(Rt(2)) ** side_r.sqrt2_inv * Om(m) ** k_r
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
            f"theorem consts : ({sc.text(f_l)}) * (({s_l}) * ({_weights_text(side_l)}"
            f"(rt 2)⁻¹ ^ {side_l.sqrt2_inv})) * om {m} ^ {k_l} ="
        ),
        (
            f"    ({sc.text(f_r)}) * (({s_r}) * ({_weights_text(side_r)}"
            f"(rt 2)⁻¹ ^ {side_r.sqrt2_inv})) * om {m} ^ {k_r} := by"
        ),
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
    phases: Sequence[tuple[str, int]],
    labs: dict[int, int],
    dims: str,
    m: int,
    coefs: Mapping[int, str] | None = None,
    tabs: Mapping[int, str] | None = None,
) -> list[str]:
    """The shared environment ``E`` of network proofs: ``Δ k`` as ``dims``, the phase vectors,
    phase index 0 the zero vector over ``m``, each label's digit count, and at each index of
    ``coefs`` the coefficient vector ``C ex (Φ i v)``, times the product of ``A`` over the
    indices ``tabs[i][v]`` at an index of ``tabs``, 1 elsewhere."""
    tabs = tabs or {}
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
        *(
            [
                "/-- Coefficient vectors. -/",
                "noncomputable def Ψ : ℕ → ℕ → ℂ",
                *(
                    f"  | {i} => fun v => {c} * ex (Φ {i} v)"
                    + (
                        f" * ((({tabs[i]} : List (List ℕ)).getD v []).map A).prod"
                        if i in tabs
                        else ""
                    )
                    for i, c in sorted(coefs.items())
                ),
                "  | _ => fun _ => 1",
                "",
                "noncomputable def E : Env := { dimv := Δ, phase := Φ, labs := labs, coef := Ψ }",
            ]
            if coefs
            else ["noncomputable def E : Env := { dimv := Δ, phase := Φ, labs := labs }"]
        ),
        "",
    ]


def _anet_lines(tag: str, side: _NetSide, m: int) -> list[str]:
    """The digit tables, the network, its fragments with atoms and the NetPSDA facts of one
    side."""
    triples = ", ".join(
        f"({n}, {f}, {a})" for n, f, a in zip(side.nodes, side.frags, side.atoms, strict=True)
    )
    cases = " | ".join("rfl" for _ in side.nodes)
    own = [(x, (lab, j)) for lab in side.labels for j, x in enumerate(side.dig[lab])]
    dig_tbl = [(lab, side.dig[lab]) for lab in side.labels]
    proofs = [
        p.replace("ownL", f"own{tag}").replace("labsL", f"labs{tag}").replace("digL", f"dig{tag}")
        for p in side.proofs
    ]
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
        (
            f"noncomputable def ps{tag} : List (Node × Frag × Atoms) := "
            f"[{triples.replace('digL', f'dig{tag}')}]"
        ),
        "",
        f"theorem net{tag} : NetPSDA E N{tag} {m} A own{tag} labs{tag} dig{tag} ps{tag} where",
        f"  ownok := own{tag}OK",
        "  nodes := rfl",
        "  node := by",
        "    intro p hp σ hσ",
        f"    have hown := own{tag}OK",
        f"    simp only [ps{tag}, List.mem_cons, List.not_mem_nil, or_false] at hp",
        *([f"    rcases hp with {cases}"] if proofs else []),
        *(f"    · exact {pr}" for pr in proofs),
        "  disj := by decide",
        "  dims := by decide",
        "  cover := by decide",
        "  sub := by decide",
        "",
        (
            f"theorem c{tag} : (joinAll (ps{tag}.map (·.2.1))).c = "
            f"{_weights_text(side)}(rt 2)⁻¹ ^ {side.sqrt2_inv} := by"
        ),
        (
            f"  simp only [ps{tag}, List.map_cons, List.map_nil, joinAll, Frag.join, Frag.unit, "
            "zFragD, xFragD, fFragD, bFrag, Frag.scale, List.length_cons, List.length_nil]"
        ),
        "  ring",
        "",
    ]


def _net_run(side: _NetSide, coefs: Any) -> Any:
    """The AmpRun of a network side from its held digits, fragment paths and atoms through its
    pins: a pinned digit an atom reads is substituted in the atoms along its pin constraint,
    its parent held first when on the path."""
    from archytaszx.semantics.lean_amp import AmpRun

    atoms = [a for node in side.node_atoms for a in node]
    run = AmpRun(coefs, list(side.cur0), list(side.held0), atoms)
    run.release()
    for x, p in side.pins:
        if x in run.avars():
            if p in run.cur:
                rest = [v for v in run.cur if v != p]
                run.inner([f".perm {[p, *rest]}"])
                run.steps.append(".hold")
                run.cur, run.held = rest, [p, *run.held]
            run.steps.append(f".asubB {x} [[{p}]]")
            run.atoms = [
                (j, frozenset((m - {x}) | {p} if x in m else m for m in f)) for j, f in run.atoms
            ]
            run.release()
        rest = [v for v in run.cur if v != x]
        run.inner([f".perm {[x, *rest]}", f".pin [[{p}]]"])
        run.cur = rest
    run.inner([".norm"])
    return run


def net_amp_body(
    left: Diagram,
    right: Diagram,
    name: str,
    phases: list[tuple[str, int]],
    coefs: dict[int, str],
    tabs: dict[int, str],
    m_min: int = 1,
) -> tuple[NetBody, Any]:
    """The lines proving ``Equal E sL NL sR NR`` for two networks with amplitude atoms (see
    :func:`export_net`), and the atoms' coefficients (a lean_amp.Coefs)."""
    from archytaszx.semantics.lean_amp import Coefs, afac_lines, reqs_lines
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

    raw_l, raw_r = from_diagram(left), from_diagram(right)
    if not isinstance(raw_l, PathSum) or not isinstance(raw_r, PathSum):
        raise PathSumUnsupported("the network export covers qubit path sums only")
    if raw_l.boundary != raw_r.boundary:
        raise PathSumUnsupported("the boundaries differ")
    m_min = math.lcm(m_min, _table_modulus(left), _table_modulus(right))
    m = amp_modulus((raw_l, raw_r), m_min)
    amp = Coefs(m)
    side_l, _ = _net_side(left, m, phases, 1 << 16, coefs, amp, tabs)
    side_r, _ = _net_side(right, m, phases, 1 << 17, coefs, amp, tabs)
    if [side_l.k_of[i] for i in range(side_l.boundary)] != [
        side_r.k_of[i] for i in range(side_r.boundary)
    ]:
        raise PathSumUnsupported("the boundary ports carry different dimensions")
    run_l, run_r = _net_run(side_l, amp), _net_run(side_r, amp)
    plan, _red_l, _red_r = amp_plan(((raw_l, run_l), (raw_r, run_r)))
    k_l, k_r = plan.turns
    sc = _scalars()
    fl_lines, f_l = afac_lines("L", run_l, sc)
    fr_lines, f_r = afac_lines("R", run_r, sc)
    s_l, s_r = sc.text(side_l.scalar), sc.text(side_r.scalar)
    w_l, w_r = sp.Mul(*side_l.weights), sp.Mul(*side_r.weights)
    lhs = f_l * side_l.scalar * w_l * Inv(Rt(2)) ** side_l.sqrt2_inv * Om(m) ** k_l
    rhs = f_r * side_r.scalar * w_r * Inv(Rt(2)) ** side_r.sqrt2_inv * Om(m) ** k_r
    proof = sc.prove(lhs, rhs)
    bnd = list(range(side_l.bnd_vars))
    t_l = "(netDataA digL NL (psL.map (·.2.1)) (psL.map (·.2.2)))"
    t_r = "(netDataA digR NR (psR.map (·.2.1)) (psR.map (·.2.2)))"
    lines = [
        *_anet_lines("L", side_l, m),
        *_anet_lines("R", side_r, m),
        f"def stepsL : List AStep := [{', '.join(run_l.steps)}]",
        f"def stepsR : List AStep := [{', '.join(run_r.steps)}]",
        "",
        *fl_lines,
        *fr_lines,
        *reqs_lines("L", run_l.reqs, amp, sc, "stepsL"),
        *reqs_lines("R", run_r.reqs, amp, sc, "stepsR"),
        (
            f"theorem consts : ({sc.text(f_l)}) * (({s_l}) * ({_weights_text(side_l)}"
            f"(rt 2)⁻¹ ^ {side_l.sqrt2_inv})) * om {m} ^ {k_l} ="
        ),
        (
            f"    ({sc.text(f_r)}) * (({s_r}) * ({_weights_text(side_r)}"
            f"(rt 2)⁻¹ ^ {side_r.sqrt2_inv})) * om {m} ^ {k_r} := by"
        ),
        f"  have key : ({sc.text(lhs)} : ℂ) = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        "  linear_combination key",
        "",
        "/-- The two networks are equal. -/",
        f"theorem {name} : Equal E ({s_l}) NL ({s_r}) NR := by",
        "  apply equal_psda2 _ _ netL netR rfl",
        "  intro σ",
        "  rw [cL, cR, ← valA_smul, ← valA_smul,",
        f"    aruns_sound {m} (by norm_num) A stepsL {t_l} _ _ reqsL (by decide),",
        f"    aruns_sound {m} (by norm_num) A stepsR {t_r} _ _ reqsR (by decide),",
        f"    valA_agree {m} _ A (aruns {m} stepsL {t_l}) {bnd} _ _ (by decide) (by decide)",
        f"      (by decide) (by decide) (dbits_agree ownL ownR _ {bnd} (by decide))]",
        f"  apply afinal_perm {m} (by norm_num) _ _ A _ _ _ {k_l} {k_r} (by decide) (by decide)",
        "    (by decide) (by decide) (by decide) (by decide)",
        "  first | (rw [facL, facR]; exact consts) | rw [facL, facR]",
        "",
    ]
    return NetBody(lines, {**side_l.k_of, **side_r.k_of}, s_l, s_r, m), amp


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
    coefs: dict[int, str] = {}
    raws = (from_diagram(left), from_diagram(right))
    if any(isinstance(r, PathSum) and r.amps for r in raws):
        return _export_amp_net(left, right, name)
    body = net_body(left, right, name, phases, coefs=coefs)
    ns = "ArchytasZX.PathSumNetProof"
    lines = [
        "import ArchytasZX.PathSumNetW",
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
        *net_env(phases, body.labs, "2 ^ k", body.m, coefs),
        *body.lines,
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


def _export_amp_net(left: Diagram, right: Diagram, name: str) -> str:
    """:func:`export_net` for networks whose path sums carry amplitude atoms
    (ArchytasZX/PathSumNetAmp.lean)."""
    phases: list[tuple[str, int]] = []
    coefs: dict[int, str] = {}
    tabs: dict[int, str] = {}
    body, amp = net_amp_body(left, right, name, phases, coefs, tabs)
    ns = "ArchytasZX.PathSumNetProof"
    sc = _scalars()
    lines = [
        "import ArchytasZX.PathSumNetAmp",
        "",
        "/-!",
        f"# Path-sum proof `{name}` over the networks with amplitude atoms",
        "",
        "Each network's value is its fragments' path sum with the summed digits held and the",
        "weighted spiders' atoms; pin steps, atom substitutions, [Average] and the reduction",
        "steps follow, and the final comparison gives `Equal`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        *NET_OPTIONS,
        "",
        *amp.lean_def(sc),
        *net_env(phases, body.labs, "2 ^ k", body.m, coefs, tabs),
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
    taken, else by a label from QPIN_OFFSET; each Z spider is anchored at the leg through which a
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
    nxt = QPIN_OFFSET
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
        if x < QPIN_OFFSET:
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


def qnet_body(left: Diagram, right: Diagram, name: str, phases: list[tuple[str, int]]) -> NetBody:
    """The lines proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X and F
    nodes on one odd prime dimension p, their phase vectors appended to ``phases`` (see
    :func:`export_qudit_net`); ``labs`` is empty and ``m`` is p."""
    from archytaszx.algebra.pathsum import legendre
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

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
    lines = [
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
    ]
    labs = {x: 1 for side in (side_l, side_r) for x in (*range(side.boundary), *side.summed)}
    return NetBody(lines, labs, s_l, s_r, p)


QNET_OPTIONS = [
    *NET_OPTIONS,
    "set_option linter.unreachableTactic false",
    "set_option linter.unusedVariables false",
]
"""The options of a qudit network proof file."""


def export_qudit_net(left: Diagram, right: Diagram, name: str = "main") -> str:
    """Lean 4 source proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X
    and F nodes on one odd prime dimension p (ArchytasZX/PathSumQuditNet.lean): each network's
    value is its fragments' path sum over Z_p (QNetPS.eval_eq), pin steps set the summed labels
    of one class to one variable, each side's reduction trace is replayed, the Gauss sums take
    their Legendre form and the final comparison gives ``Equal``; raises
    :class:`PathSumUnsupported` outside that class or when the sums do not reduce to one
    canonical form, :class:`ScalarUnsupported` when the constants' identity has no proof."""
    phases: list[tuple[str, int]] = []
    body = qnet_body(left, right, name, phases)
    p = body.m
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
        *QNET_OPTIONS,
        "",
        f"instance : Fact (Nat.Prime {p}) := ⟨by norm_num⟩",
        "",
        *net_env(phases, {}, str(p), p),
        *body.lines,
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "NET_OPTIONS",
    "QNET_OPTIONS",
    "NetBody",
    "export_net",
    "export_pair",
    "export_proof",
    "export_qudit_net",
    "export_qudit_proof",
    "net_body",
    "net_env",
    "qnet_body",
]
