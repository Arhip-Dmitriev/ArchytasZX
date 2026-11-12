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

"""Lean output for a path-sum EQUAL: each side's reduction trace as one rule-lemma instance per
step (ArchytasZX/PathSum.lean), then both canonical forms as data and their equality by rfl.

The file states the left trace, the canonical-form equality, then the right trace; it never
enumerates a sum. It does not yet state L = R: the lemma that composes a step's local identity
with the rest of the sum (phase split, Boolean-lift substitution, path-variable order) is not in
the library.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

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


__all__ = ["export_pair"]
