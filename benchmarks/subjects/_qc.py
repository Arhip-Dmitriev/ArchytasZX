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

"""Reader for the Amy-Maslov-Mosca benchmark circuits in the .qc format (amm/, copied from
Feynman under its BSD licence, amm/LICENSE.md), and the subject builder for them: the circuit
on its .i inputs with every other wire initialised to |0>, against the reversible function it
is documented to compute."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks import ir
from benchmarks.ir import CX, MCX, MCZ, Init, Oracle, Post, X, circuit
from benchmarks.subject import Domain, Equation, SpecContext, Subject
from benchmarks.subjects._gates import H, S, Sdg, T, Tdg
from benchmarks.subjects._spec import defining

AMM = Path(__file__).with_name("amm")


@dataclass(frozen=True)
class Qc:
    """A parsed .qc circuit: wire names in .v order, the .i inputs, the gates."""

    wires: tuple[str, ...]
    inputs: tuple[str, ...]
    gates: tuple[ir.Gate, ...]

    @property
    def ancillas(self) -> tuple[str, ...]:
        return tuple(w for w in self.wires if w not in self.inputs)


def _wire(name: str) -> str:
    return name if name[0].isalpha() else f"w{name}"


def read(name: str) -> Qc:
    """The circuit amm/NAME.qc; wire names starting with a digit get the prefix w."""
    wires: list[str] = []
    inputs: list[str] = []
    gates: list[ir.Gate] = []
    body = False
    one = {"H": H, "X": X, "T": T, "T*": Tdg, "P": S, "P*": Sdg}
    for raw in (AMM / f"{name}.qc").read_text().splitlines():
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == ".v":
            wires = [_wire(t) for t in tok[1:]]
        elif tok[0] == ".i":
            inputs = [_wire(t) for t in tok[1:]]
        elif tok[0] == "BEGIN":
            body = True
        elif tok[0] == "END":
            body = False
        elif body:
            g, qs = tok[0], [_wire(t) for t in tok[1:]]
            if g in one and len(qs) == 1:
                gates.append(one[g](qs[0]))
            elif g == "tof" and len(qs) == 1:
                gates.append(X(qs[0]))
            elif g == "tof" and len(qs) == 2:
                gates.append(CX(qs[0], qs[1]))
            elif g == "tof":
                gates.append(MCX(tuple(qs[:-1]), qs[-1]))
            elif g in ("Z", "Zd") and len(qs) >= 2:
                gates.append(MCZ(tuple(qs[:-1]), qs[-1]))
            else:
                raise ValueError(f"{name}.qc: gate {raw.strip()!r}")
    return Qc(tuple(wires), tuple(inputs), tuple(gates))


def bits(x: int, k: int) -> list[int]:
    """The k bits of x, most significant first."""
    return [(x >> (k - 1 - i)) & 1 for i in range(k)]


def number(bs: Sequence[int]) -> int:
    """The integer of a bit list, most significant first."""
    v = 0
    for b in bs:
        v = 2 * v + b
    return v


def _bitwise(
    qc: Qc, fn: Callable[[dict[str, int]], dict[str, int]]
) -> tuple[list[str], list[ir.Gate]]:
    """Wires and gates of |x> -> |fn(x)> by XOR oracles: an ancilla w gets fn_w(x); a changed
    input w gets fn_w(x) on a fresh wire w_out after it, then w is cleared by the oracle of its
    preimage value on the unchanged inputs and the fresh wires, and post-selected on 0."""
    ins = qc.inputs
    table = {}
    for v in range(2 ** len(ins)):
        x = dict(zip(ins, bits(v, len(ins))))
        out = fn(x)
        table[tuple(x[w] for w in ins)] = {
            w: out[w] if w in ins else out.get(w, 0) for w in qc.wires
        }
    moved = [w for w in ins if any(o[w] != k[ins.index(w)] for k, o in table.items())]
    stay = [w for w in ins if w not in moved]
    fresh = {w: f"{w}_out" for w in moved}

    def image(w: str) -> Callable[[tuple[int, ...]], int]:
        return lambda k: table[k][w]

    pre = {tuple(o[w] for w in stay) + tuple(o[w] for w in moved): k for k, o in table.items()}

    def preimage(w: str) -> Callable[[tuple[int, ...]], int]:
        i = ins.index(w)
        return lambda k: pre[k][i] if k in pre else 0

    gs: list[ir.Gate] = [Init(a) for a in qc.ancillas]
    gs += [Oracle(ins, a, image(a)) for a in qc.ancillas if any(o[a] for o in table.values())]
    gs += [g for w in moved for g in (Init(fresh[w]), Oracle(ins, fresh[w], image(w)))]
    back = (*stay, *(fresh[w] for w in moved))
    gs += [g for w in moved for g in (Oracle(back, w, preimage(w)), Post(w))]
    wires = [u for w in qc.wires for u in ((w, fresh[w]) if w in fresh else (w,))]
    return wires, gs


def amm_subject(
    name: str,
    title: str,
    fn: Callable[[dict[str, int]], dict[str, int]] | None,
    note: str,
    clean: Sequence[str] = (),
    gates: Sequence[ir.Gate] = (),
) -> Subject:
    """The subject of amm/NAME.qc: the circuit with its ancillas in |0> against the isometry
    |x> -> |fn(x)> (fn maps the input bits to every wire's output bit, by name), written as the
    XOR oracles of :func:`_bitwise` (0 where fn gives none), or, with fn None, the defining
    ``gates`` after |0> on the ancillas; ``clean`` names inputs also initialised to |0>."""
    qc = read(name)
    qc = Qc(qc.wires, tuple(w for w in qc.inputs if w not in clean), qc.gates)

    def build(n: int | None, d: int) -> ir.Circuit:
        return circuit(qc.wires, [*(Init(a) for a in qc.ancillas), *qc.gates])

    def spec(c: SpecContext) -> str:
        if fn is None:
            return defining(c, qc.wires, [*(Init(a) for a in qc.ancillas), *gates])
        return defining(c, *_bitwise(qc, fn))

    sid = "amm_" + name.replace("^", "").replace("-", "_")
    return Subject(
        id=sid,
        name=f"{title} (Amy-Maslov-Mosca benchmark {name})",
        category="circuit",
        source="amy2014polynomial",
        size_parameter=None,
        dimension=2,
        domain=Domain(d_min=2, d_max=2),
        equations=(Equation("main", build, spec, form="fixed"),),
        note=note,
    )


__all__ = ["AMM", "Qc", "amm_subject", "bits", "number", "read"]
