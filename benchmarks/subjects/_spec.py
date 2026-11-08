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

"""Specification writers shared by the subject files: the diagram of one defining gate
(a permutation, a phase table, a multi-controlled gate) and a small DSL builder."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from benchmarks import ir, translate
from benchmarks.subject import SpecContext


def defining(
    c: SpecContext,
    wires: Sequence[str],
    gates: Sequence[ir.Gate],
    *,
    registers: Mapping[str, Any] | None = None,
    dims: Mapping[str, Any] | None = None,
    scalar: str = "1",
) -> str:
    """The diagram of the defining gates ``gates`` on ``wires``, symbols written as ``c``
    writes them."""
    circ = ir.circuit(wires, gates, registers=registers, dims=dims, scalar=scalar)
    values = {k: v for k, v in (("n", c.nv), ("d", c.dv)) if v is not None}
    lit = frozenset({"d"}) if c.D.startswith("literal") else frozenset()
    return translate.translate(circ, translate.Context(values=values, literal=lit)).body


def permutation(c: SpecContext, wires: Sequence[str], p: Callable[[int], int], **kw: Any) -> str:
    """The permutation |x> -> |p(x)> of the joint basis of ``wires`` (first most significant)."""
    return defining(c, wires, [ir.Perm(tuple(wires), p)], **kw)


def diagonal(
    c: SpecContext, wires: Sequence[str], f: Callable[[tuple[int, ...]], Any], **kw: Any
) -> str:
    """The diagonal |x> -> e^(2 pi i f(x)) |x> on ``wires``."""
    return defining(c, wires, [ir.PhaseFn(tuple(wires), f)], **kw)


def basis_state(c: SpecContext, values: Sequence[int]) -> str:
    """The product basis state |v_1, ..., v_m> as X states."""
    lines = [
        f"s{i} = X(0 -> 1, {c.D})" + (f" phase{{k: -{v}*k/{c.d}}}" if v else "")
        for i, v in enumerate(values)
    ]
    if values:
        lines.append(f"scalar {c.d}^(-{len(values)}/2)")
    return "\n".join(lines)


class Dsl:
    """Accumulates named nodes, explicit wires, the boundary and a scalar."""

    def __init__(self) -> None:
        self.decls: list[str] = []
        self.wires: list[str] = []
        self.ins: list[str] = []
        self.outs: list[str] = []
        self.scalar: list[str] = []
        self.k = 0

    def node(self, text: str) -> str:
        """Declares a node and returns its name."""
        name = f"v{self.k}"
        self.k += 1
        self.decls.append(f"{name} = {text}")
        return name

    def wire(self, a: str, b: str) -> None:
        """Joins an output port to an input port."""
        self.wires.append(f"{a} -- {b}")

    def text(self) -> str:
        """The DSL body."""
        lines = list(self.decls)
        if self.wires:
            lines.append("wire " + ", ".join(self.wires))
        if self.ins:
            lines.append("inputs " + ", ".join(self.ins))
        if self.outs:
            lines.append("outputs " + ", ".join(self.outs))
        if self.scalar:
            lines.append("scalar " + " * ".join(f"({s})" for s in self.scalar))
        return "\n".join(lines)


def bind(b: Dsl, n: int, d: int) -> tuple[list[str], str]:
    """Binders joining n legs of dimension d into one of dimension d^n (first leg most
    significant): the n input ports and the bound output port."""
    if n == 1:
        z = b.node(f"Z(1 -> 1, literal {d})")
        return [f"{z}.in0"], f"{z}.out0"
    first = b.node(f"B(literal {d}, literal {d})")
    ins = [f"{first}.in0", f"{first}.in1"]
    port = f"{first}.out0"
    acc = d * d
    for _ in range(2, n):
        nxt = b.node(f"B(literal {acc}, literal {d})")
        b.wire(port, f"{nxt}.in0")
        ins.append(f"{nxt}.in1")
        port = f"{nxt}.out0"
        acc *= d
    return ins, port


def split(b: Dsl, port: str, n: int, d: int) -> list[str]:
    """Splitters taking a leg of dimension d^n back to n legs (first most significant)."""
    if n == 1:
        return [port]
    outs: list[str] = []
    for i in range(n - 1, 0, -1):
        s = b.node(f"S(literal {d**i}, literal {d})")
        b.wire(port, f"{s}.in0")
        outs.append(f"{s}.out1")
        port = f"{s}.out0"
    return [port, *reversed(outs)]


def register_adder(c: SpecContext, n: int, d: int, b: Dsl | None = None) -> Dsl:
    """|a, b> -> |a, a + b mod d^n> on two n-qudit registers: both bound, a copied by a Z
    spider and added into b by an X spider, split back."""
    b = b or Dsl()
    D = d**n
    ia, pa = bind(b, n, d)
    ib, pb = bind(b, n, d)
    z = b.node(f"Z(1 -> 2, literal {D})")
    x = b.node(f"X(2 -> 1, literal {D})")
    b.wire(pa, f"{z}.in0")
    b.wire(pb, f"{x}.in0")
    b.wire(f"{z}.out1", f"{x}.in1")
    b.ins += ia + ib
    b.outs += split(b, f"{z}.out0", n, d) + split(b, f"{x}.out0", n, d)
    b.scalar.append(f"{D}^(1/2)")
    return b


def pauli_gadget(
    b: Dsl, ports: dict[str, str], letters: Mapping[str, str], turn: Any, D: str
) -> None:
    """Appends the phase gadget e^(-i 2 pi turn P) of the Pauli string ``letters`` ({wire: X, Y
    or Z}) to the open ``ports`` (updated in place): a Z leg per wire in its letter's basis, an
    X hub, the phase 2 turn on the odd parity, the scalar 2^((m - 1)/2) e^(-i 2 pi turn)."""
    legs = []
    for w, letter in letters.items():
        port = ports[w]
        if letter == "Y":
            s = b.node(f"Z(1 -> 1, {D}) phase{{1: 3/4}}")
            b.wire(port, f"{s}.in0")
            port = f"{s}.out0"
        if letter in "XY":
            h = b.node(f"F({D})")
            b.wire(port, f"{h}.in0")
            port = f"{h}.out0"
        leg = b.node(f"Z(1 -> 2, {D})")
        b.wire(port, f"{leg}.in0")
        legs.append(f"{leg}.out1")
        port = f"{leg}.out0"
        if letter in "XY":
            h = b.node(f"F({D})")
            b.wire(port, f"{h}.in0")
            port = f"{h}.out0"
        if letter == "Y":
            s = b.node(f"Z(1 -> 1, {D}) phase{{1: 1/4}}")
            b.wire(port, f"{s}.in0")
            port = f"{s}.out0"
        ports[w] = port
    hub = b.node(f"X({len(legs)} -> 1, {D})")
    for i, leg_port in enumerate(legs):
        b.wire(leg_port, f"{hub}.in{i}")
    ph = b.node(f"Z(1 -> 0, {D}) phase{{1: 2*({turn})}}")
    b.wire(f"{hub}.out0", f"{ph}.in0")
    b.scalar.append(f"2^(({len(legs)} - 1)/2) * exp(-2*pi*I*({turn}))")


__all__ = [
    "Dsl",
    "basis_state",
    "bind",
    "defining",
    "diagonal",
    "pauli_gadget",
    "permutation",
    "register_adder",
    "split",
]
