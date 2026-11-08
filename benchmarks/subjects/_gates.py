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

"""Gate-level building blocks shared by the subject files: named qubit gates, textbook
decompositions and the inverse of a unitary gate list."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction

import sympy as sp

from benchmarks import ir
from benchmarks.ir import CX, CZ, MCX, MCZ, F, Oracle, Perm, PhaseFn, Swap, X, XPhase, Z, ZPhase


def H(q: str) -> ir.Gate:
    """The Hadamard gate (F at d = 2)."""
    return F(q)


def P(q: str, t: object) -> ir.Gate:
    """The phase gate diag(1, e^(2 pi i t)) on a qubit."""
    return ZPhase(q, {1: t})


def S(q: str) -> ir.Gate:
    """diag(1, i)."""
    return P(q, Fraction(1, 4))


def Sdg(q: str) -> ir.Gate:
    """diag(1, -i)."""
    return P(q, Fraction(-1, 4))


def T(q: str) -> ir.Gate:
    """diag(1, e^(i pi / 4))."""
    return P(q, Fraction(1, 8))


def Tdg(q: str) -> ir.Gate:
    """diag(1, e^(-i pi / 4))."""
    return P(q, Fraction(-1, 8))


def XP(q: str, t: object) -> ir.Gate:
    """H P(t) H: the X-basis phase gate."""
    return XPhase(q, {1: t})


def Y(q: str) -> list[ir.Gate]:
    """The Pauli Y = i X Z as gates (Z then X); the factor i is left to the caller."""
    return [Z(q), X(q)]


def ccz(a: str, b: str, c: str) -> list[ir.Gate]:
    """The 7-T CCZ phase polynomial (Nielsen and Chuang, figure 4.9)."""
    return [
        CX(b, c),
        Tdg(c),
        CX(a, c),
        T(c),
        CX(b, c),
        Tdg(c),
        CX(a, c),
        T(b),
        T(c),
        CX(a, b),
        T(a),
        Tdg(b),
        CX(a, b),
    ]


def toffoli(a: str, b: str, c: str) -> list[ir.Gate]:
    """The Toffoli gate: H on the target around the 7-T CCZ."""
    return [H(c), *ccz(a, b, c), H(c)]


def cswap(c: str, a: str, b: str) -> list[ir.Gate]:
    """The Fredkin gate: CX(b, a), Toffoli(c, a, b), CX(b, a)."""
    return [CX(b, a), *toffoli(c, a, b), CX(b, a)]


def cphase(a: str, b: str, t: object) -> list[ir.Gate]:
    """The controlled phase diag(1, 1, 1, e^(2 pi i t)) from P(t/2) on both and P(-t/2)
    between two CX."""
    h = sp.Rational(sp.nsimplify(t)) / 2
    return [P(a, h), P(b, h), CX(a, b), P(b, -h), CX(a, b)]


def inverse_gate(g: ir.Gate) -> ir.Gate:
    """The inverse of one unitary gate."""
    if isinstance(g, (X, Z)):
        return replace(g, a=-sp.sympify(g.a) if isinstance(g.a, str) else -g.a)
    if isinstance(g, F):
        return replace(g, p=(-g.p) % 4)
    if isinstance(g, (ZPhase, XPhase)):
        if g.rule is not None:
            return replace(g, rule=f"-({g.rule})")
        return replace(g, entries={k: -sp.sympify(v) for k, v in g.entries.items()})
    if isinstance(g, (CX, CZ)):
        return replace(g, s=-g.s)
    if isinstance(g, Swap):
        return g
    if isinstance(g, PhaseFn):
        f = g.f
        return replace(g, f=lambda x, f=f: -sp.sympify(f(x)))
    if isinstance(g, (MCX, MCZ)):
        return replace(g, a=-g.a)
    if isinstance(g, Perm):
        raise TypeError(f"no inverse recorded for a permutation on {g.qs}")
    if isinstance(g, Oracle):
        f = g.f
        return replace(g, f=lambda x, f=f: -f(x))
    raise ValueError(f"{type(g).__name__} is not a unitary gate")


def inverse(gates: Sequence[ir.Gate]) -> list[ir.Gate]:
    """The inverse of a unitary gate list."""
    return [inverse_gate(g) for g in reversed(gates)]


def qft(qs: Sequence[str], d: int) -> list[ir.Gate]:
    """The textbook qudit QFT on ``qs`` (first most significant), the swaps included."""
    n = len(qs)
    out: list[ir.Gate] = []
    for j in range(n):
        out.append(F(qs[j]))
        for k in range(j + 1, n):
            m = k - j + 1
            out.append(PhaseFn((qs[k], qs[j]), lambda x, m=m: Fraction(x[0] * x[1], d**m)))
    for j in range(n // 2):
        out.append(Swap(qs[j], qs[n - 1 - j]))
    return out


__all__ = [
    "AND_GATES",
    "XP",
    "H",
    "P",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "Y",
    "ccz",
    "ch",
    "cphase",
    "cry",
    "cswap",
    "inverse",
    "inverse_gate",
    "qft",
    "rename",
    "ry",
    "toffoli",
]


def ry(q: str, t: object) -> list[ir.Gate]:
    """e^(i pi t) Ry(2 pi t) as S^dagger, H, P(t), H, S."""
    return [Sdg(q), H(q), P(q, t), H(q), S(q)]


def cry(c: str, q: str, t: object) -> list[ir.Gate]:
    """The controlled Ry(2 pi t): CX, Ry(-pi t), CX, Ry(pi t)."""
    h = sp.Rational(sp.nsimplify(t)) / 2
    return [CX(c, q), *ry(q, -h), CX(c, q), *ry(q, h)]


def ch(c: str, q: str) -> list[ir.Gate]:
    """The controlled Hadamard: Ry(pi/4), CX, Ry(-pi/4) on the target."""
    return [*ry(q, Fraction(1, 8)), CX(c, q), *ry(q, Fraction(-1, 8))]


def dephase_lines(port: str, name: str, D: str) -> list[str]:
    """DSL lines dephasing ``port`` into ``name``.out0 (Z copy, one leg discarded)."""
    return [
        f"{name} = Z(1 -> 2, {D})",
        f"{name}g = G(1 -> 0, {D})",
        f"wire {port} -- {name}.in0, {name}.out1 -- {name}g.in0",
    ]


def AND_GATES(a: str, b: str, t: str) -> list[ir.Gate]:
    """Gidney's 4-T temporary logical AND into the fresh qubit t."""
    return [
        ir.Init(t),
        H(t),
        T(t),
        CX(a, t),
        CX(b, t),
        CX(t, a),
        CX(t, b),
        Tdg(a),
        Tdg(b),
        T(t),
        CX(t, a),
        CX(t, b),
        H(t),
        S(t),
    ]


def rename(gates: Sequence[ir.Gate], mapping: dict[str, str]) -> list[ir.Gate]:
    """The gates with their wire names replaced through ``mapping``."""
    out = []
    for g in gates:
        changes: dict[str, object] = {}
        for name in ("q", "c", "t", "y"):
            v = getattr(g, name, None)
            if isinstance(v, str) and v in mapping:
                changes[name] = mapping[v]
        if isinstance(g, Swap):
            changes = {"a": mapping.get(g.a, g.a), "b": mapping.get(g.b, g.b)}
        for name in ("qs", "cs", "xs"):
            v = getattr(g, name, None)
            if v is not None:
                changes[name] = tuple(mapping.get(x, x) for x in v)
        out.append(replace(g, **changes) if changes else g)
    return out
